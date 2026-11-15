"""补装的反应式触发与多轮重试（run 20261115-2223 的静默失效修复）。

背景：局内「死亡→复活」可能整个被 200ms 的观测刷新跳过，环境看不到 ``is_out``，
``_restock_after_respawn`` 永不触发 → ``_restock_pending`` 从未登记 → grace/strip 全不生效，
运行时于是连续十几步回 ``背包里没有剑``（ITEM_MISSING 27 次）。

这里验证三条兜底：运行时失败码当触发源、每 10 步主动体检、窗口到期改多轮重试。
"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.env import DEFAULT_ARMOR_ITEMS, loadout_report
from gladiator_env.spaces import ACTION_NAMES
from gladiator_env.testing import FakeLink, make_observation, make_target

FULL_ARMOR = {
    "head": {"name": "diamond_helmet"},
    "torso": {"name": "diamond_chestplate"},
    "legs": {"name": "diamond_leggings"},
    "feet": {"name": "diamond_boots"},
}

FULL_INVENTORY = [
    {"slot": 36, "name": "diamond_sword", "count": 1},
    {"slot": 37, "name": "golden_apple", "count": 5},
    {"slot": 38, "name": "fishing_rod", "count": 1},
]

ATTACK = ACTION_NAMES.index("attack")


class FakeRcon:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, text: str) -> str:
        self.commands.append(text)
        return ""


def observation(username: str, *, inventory: list[dict] | None = None,
                held: str = "diamond_sword", health: float = 20.0) -> dict:
    other = "Glad_Bot_2" if username == "Glad_Bot_1" else "Glad_Bot_1"
    raw = make_observation(
        username, health=health,
        targets=[make_target(2 if username == "Glad_Bot_1" else 1,
                             other, x=1.0, z=0.0, distance=2.0)])
    raw["self"]["inventory"] = list(FULL_INVENTORY if inventory is None else inventory)
    raw["self"]["heldItem"] = {"name": held, "count": 1}
    raw["self"]["armor"] = dict(FULL_ARMOR)
    return raw


def build_env(*, give_kit: bool = True) -> tuple[GladiatorEnv, list[FakeLink], list[str], FakeRcon]:
    links = [FakeLink("Glad_Bot_1", observation("Glad_Bot_1")),
             FakeLink("Glad_Bot_2", observation("Glad_Bot_2"))]
    logs: list[str] = []
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        logger=logs.append,
    )
    env.arena = type(env.arena)(**{**env.arena.__dict__, "give_kit": give_kit})
    rcon = FakeRcon()
    env._rcon = rcon
    env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]
    return env, links, logs, rcon


def empty_bot1(env: GladiatorEnv) -> None:
    env._raw[0] = observation("Glad_Bot_1", inventory=[], held="air")


class ReactiveTriggerTest(unittest.TestCase):
    def test_item_missing_result_triggers_restock_and_give(self):
        env, _links, logs, rcon = build_env()
        empty_bot1(env)

        env._handle_action_results(0, [{
            "ok": False, "code": "ITEM_MISSING", "reason": "背包里没有剑",
        }])

        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)
        self.assertGreaterEqual(env._respawn_grace.get("Glad_Bot_1", 0), 3)
        joined = " | ".join(rcon.commands)
        self.assertIn("give Glad_Bot_1 diamond_sword", joined)
        self.assertIn("give Glad_Bot_1 golden_apple", joined)
        self.assertTrue(any("触发补装（原因：code=ITEM_MISSING）" in line for line in logs), logs)

    def test_repeat_failure_tops_up_without_duplicate_log(self):
        env, _links, logs, _rcon = build_env()
        empty_bot1(env)

        env._handle_action_results(0, [{"ok": False, "code": "ITEM_MISSING", "reason": "x"}])
        # 消耗掉两帧，再重复失败：应补足到 3，但不再打「触发补装」
        env._confirm_restock()
        env._confirm_restock()
        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 1)
        env._handle_action_results(0, [{"ok": False, "code": "ITEM_MISSING", "reason": "x"}])

        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)
        triggers = [line for line in logs if "触发补装" in line]
        self.assertEqual(len(triggers), 1, logs)

    def test_non_item_codes_do_not_trigger(self):
        env, _links, logs, _rcon = build_env()
        env._handle_action_results(0, [{
            "ok": False, "code": "OUT_OF_REACH_CHASED", "reason": "",
        }])
        env._handle_action_results(0, [{"ok": False, "code": "RULE_DENIED", "reason": ""}])
        self.assertFalse(env._restock_pending)
        self.assertFalse(any("触发补装" in line for line in logs))

    def test_drain_results_path_also_triggers(self):
        """非阻塞链路（poll_results）与同步链路走同一套处理。"""

        class PollingLink(FakeLink):
            def __init__(self, username: str, obs: dict, results: list[dict]) -> None:
                super().__init__(username, obs)
                self.pending_results = results

            def poll_results(self) -> list[dict]:
                out = list(self.pending_results)
                self.pending_results.clear()
                return out

        env, _links, logs, rcon = build_env()
        env.links[0] = PollingLink("Glad_Bot_1", observation("Glad_Bot_1"),
                                   [{"ok": False, "code": "ITEM_MISSING", "reason": "x"}])
        empty_bot1(env)

        env._drain_results()

        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)
        self.assertTrue(any("give Glad_Bot_1 diamond_sword" in cmd for cmd in rcon.commands))


class PeriodicCheckTest(unittest.TestCase):
    def test_periodic_triggers_when_sword_missing(self):
        env, _links, logs, rcon = build_env()
        empty_bot1(env)

        env._periodic_loadout_check()

        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)
        self.assertTrue(any("原因：periodic" in line for line in logs), logs)
        self.assertTrue(any("diamond_sword" in cmd for cmd in rcon.commands))

    def test_periodic_skips_complete_loadout(self):
        env, _links, logs, rcon = build_env()

        env._periodic_loadout_check()

        self.assertFalse(env._restock_pending)
        self.assertEqual(rcon.commands, [])
        self.assertFalse(any("触发补装" in line for line in logs))

    def test_periodic_switches_sword_without_stripping(self):
        """装备齐了只是手里拿着金苹果：切剑就够，不要白剥 3 tick 战斗动作。"""
        env, _links, logs, rcon = build_env()
        env._raw[0] = observation("Glad_Bot_1", held="golden_apple")

        env._periodic_loadout_check()

        self.assertFalse(env._restock_pending)
        self.assertEqual(rcon.commands, [])
        self.assertFalse(any("触发补装" in line for line in logs))

    def test_periodic_never_triggers_without_give_kit(self):
        env, _links, logs, rcon = build_env(give_kit=False)
        empty_bot1(env)

        env._periodic_loadout_check()

        self.assertFalse(env._restock_pending)
        self.assertEqual(rcon.commands, [])
        self.assertFalse(any("触发补装" in line for line in logs))

    def test_step_runs_check_every_ten_steps(self):
        env, links, logs, _rcon = build_env()
        empty_bot1(env)
        # step 会用 _refresh_raw 从通道重新取观测，所以通道那份也要同步成「空手」。
        links[0].set_observation(observation("Glad_Bot_1", inventory=[], held="air"))
        env._steps = 9  # 下一步就是第 10 步

        env.step({"bot_0": 0, "bot_1": 0})

        self.assertEqual(env._steps, 10)
        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)
        self.assertTrue(any("原因：periodic" in line for line in logs), logs)
        env.close()


class NowaitLink(FakeLink):
    """能观察 ``send_actions_nowait``（``_send_use`` 走的通道）的假连接。"""

    def __init__(self, username: str, obs: dict | None = None) -> None:
        super().__init__(username, obs)
        self.nowait_actions: list[dict] = []

    def send_actions_nowait(self, actions: list[dict]) -> None:
        self.nowait_actions.extend(actions)
        self.actions.extend(actions)


def armor_observation(username: str, *, worn: tuple[str, ...],
                      inventory: list[dict] | None = None, held: str = "diamond_sword") -> dict:
    """按「穿了哪几件」构造观测：未列出的槽位就是 ``None``（= 掉落消失）。"""
    raw = observation(username, inventory=inventory, held=held)
    raw["self"]["armor"] = {
        slot: (dict(FULL_ARMOR[slot]) if slot in worn else None) for slot in FULL_ARMOR
    }
    return raw


class ArmorRestockTest(unittest.TestCase):
    def _env_with_nowait(self):
        env, links, logs, rcon = build_env()
        link = NowaitLink("Glad_Bot_1", observation("Glad_Bot_1"))
        env.links[0] = link
        return env, link, logs, rcon

    def test_give_only_missing_armor_slots(self):
        """armor=2/4 时只发缺失的两件，然后请运行时穿甲。"""
        env, link, _logs, rcon = self._env_with_nowait()
        # 头/腿还在，胸甲和靴子掉了
        env._raw[0] = armor_observation("Glad_Bot_1", worn=("head", "legs"))

        env._give_missing(0, env.bots[0], loadout_report(env._raw[0]))

        armor_gives = [cmd for cmd in rcon.commands if cmd.startswith("give")]
        self.assertEqual(sorted(armor_gives), [
            "give Glad_Bot_1 diamond_boots 1",
            "give Glad_Bot_1 diamond_chestplate 1",
        ])
        self.assertIn({"type": "use", "params": {"item": "armor"}}, link.nowait_actions)

    def test_no_armor_give_when_full(self):
        env, link, _logs, rcon = self._env_with_nowait()
        env._raw[0] = observation("Glad_Bot_1")

        env._give_missing(0, env.bots[0], loadout_report(env._raw[0]))

        self.assertEqual([cmd for cmd in rcon.commands if cmd.startswith("give")], [])
        self.assertNotIn({"type": "use", "params": {"item": "armor"}}, link.nowait_actions)

    def test_four_pieces_missing_gives_all(self):
        env, link, _logs, rcon = self._env_with_nowait()
        env._raw[0] = armor_observation("Glad_Bot_1", worn=())

        env._give_missing(0, env.bots[0], loadout_report(env._raw[0]))

        gave = sorted(cmd.split()[2] for cmd in rcon.commands if cmd.startswith("give")
                      and "diamond_" in cmd)
        # 剑/苹果/鱼竿 + 四件护甲（这里只看护甲）
        self.assertEqual([item for item in gave if item in DEFAULT_ARMOR_ITEMS.values()],
                         sorted(DEFAULT_ARMOR_ITEMS.values()))
        self.assertIn({"type": "use", "params": {"item": "armor"}}, link.nowait_actions)

    def test_missing_armor_matches_slots(self):
        raw = armor_observation("Glad_Bot_1", worn=("torso",))
        report = loadout_report(raw)
        self.assertEqual(report["missingArmor"], ["head", "legs", "feet"])
        self.assertEqual(report["armor"], 1)
        # missing 只表示物品数量，不能被护甲污染
        self.assertTrue(all(name in ("diamond_sword", "golden_apple", "fishing_rod")
                            for name, _need in report["missing"]))

    def test_armor_repair_succeeds_after_give(self):
        env, _link, logs, _rcon = self._env_with_nowait()
        env._raw[0] = armor_observation("Glad_Bot_1", worn=("head", "legs"))
        env._request_restock(0, reason="periodic")

        # give 落地：护甲穿齐
        env._raw[0] = observation("Glad_Bot_1")
        env._confirm_restock()

        self.assertFalse(env._restock_pending)
        self.assertTrue(any("已补齐" in line for line in logs), logs)
        self.assertFalse(any("补装失败" in line for line in logs), logs)

    def test_failure_log_carries_armor_count(self):
        env, _link, logs, _rcon = self._env_with_nowait()
        # 物品齐但护甲一直只有 2/4：3 轮后必须报出护甲数，便于定位
        env._raw[0] = armor_observation("Glad_Bot_1", worn=("head", "legs"))
        env._request_restock(0, reason="periodic")
        for _ in range(9):
            env._confirm_restock()

        fails = [line for line in logs if "补装失败" in line]
        self.assertEqual(len(fails), 1, logs)
        self.assertIn("护甲 2/4", fails[0])


class EquipNotStrippedTest(unittest.TestCase):
    def test_give_then_complete_without_sword_in_hand(self):
        """缺件→补发→数量齐但没持剑：窗口当帧结束，不再白跑 3 轮（FAILED_KIT 的成因）。"""
        env, links, logs, rcon = build_env()
        empty_bot1(env)
        env._request_restock(0, reason="code=ITEM_MISSING")

        # give 已下发；下一帧数量齐了，但手里仍是空（策略还没切剑）
        env._raw[0] = observation("Glad_Bot_1", held="air")
        env._confirm_restock()

        self.assertFalse(env._restock_pending)
        self.assertFalse(env._restock_rounds)
        self.assertTrue(any("数量齐但不持剑" in line for line in logs), logs)
        self.assertTrue(any("已补齐" in line for line in logs), logs)
        self.assertFalse(any("补装失败" in line for line in logs), logs)
        # 补装期间切剑用的 use 必须能下发（不能被 strip 吃掉）
        self.assertEqual(GladiatorEnv._strip_combat_actions(
            [{"type": "use", "params": {"item": "sword", "select": True}}]),
            [{"type": "use", "params": {"item": "sword", "select": True}}])
        env.close()


class SamePathTest(unittest.TestCase):
    def test_respawn_and_direct_request_log_identically(self):
        env, _links, logs, _rcon = build_env()
        empty_bot1(env)

        # 路径 1：死→活
        dead = make_observation("Glad_Bot_1", health=0.0, state="Dead")
        env._restock_after_respawn([dead, observation("Glad_Bot_2")])
        by_respawn = [line for line in logs if "触发补装" in line]
        self.assertEqual(len(by_respawn), 1, logs)
        self.assertIn("原因：respawn", by_respawn[0])
        self.assertEqual(env._restock_pending.get("Glad_Bot_1"), 3)

        # 路径 2：直接调用（清掉状态后）
        env._restock_pending.clear()
        env._respawn_grace.clear()
        env._restock_rounds.clear()
        logs.clear()
        env._request_restock(0, reason="respawn")

        by_direct = [line for line in logs if "触发补装" in line]
        self.assertEqual(by_direct, by_respawn, "两条路径必须共用同一文案")


class RetryRoundsTest(unittest.TestCase):
    def test_three_failed_rounds_release_strip(self):
        env, links, logs, _rcon = build_env()
        empty_bot1(env)
        env._request_restock(0, reason="code=ITEM_MISSING")

        # 3 轮 × 3 帧全失败
        for _ in range(9):
            env._confirm_restock()

        self.assertFalse(env._restock_pending)
        self.assertFalse(env._respawn_grace)
        fails = [line for line in logs if "补装失败" in line]
        self.assertEqual(len(fails), 1, logs)
        self.assertIn("已重试 3 轮", fails[0])

        # strip 解除：下一 tick 的攻击动作能正常下发
        links[0].actions.clear()
        env.step({"bot_0": ATTACK, "bot_1": 0})
        self.assertIn("attack", [item["type"] for item in links[0].actions])
        env.close()

    def test_success_in_second_round_reports_round(self):
        env, _links, logs, _rcon = build_env()
        empty_bot1(env)
        env._request_restock(0, reason="code=ITEM_MISSING")

        for _ in range(3):  # 第 1 轮失败
            env._confirm_restock()
        self.assertEqual(env._restock_rounds.get("Glad_Bot_1"), 1)

        # 第 2 轮第一帧就补齐
        env._raw[0] = observation("Glad_Bot_1")
        env._confirm_restock()

        done = [line for line in logs if "已补齐" in line]
        self.assertEqual(len(done), 1, logs)
        self.assertIn("第 2 轮", done[0])
        self.assertFalse(env._restock_pending)


if __name__ == "__main__":
    unittest.main()
