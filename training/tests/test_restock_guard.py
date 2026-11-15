"""重生补装期间「不进攻」的护栏（A2 收尾）。

两类噪音里的 ITEM_MISSING 全部来自学习方重生后的几 tick：背包还没同步，策略就已经在
attack/use。这里验证：
- 补装确认期间（pending>0）进攻动作被剥掉，look/move 保留；
- 补齐（或窗口用完）后动作恢复；
- 观察档 ``_respawn_grace`` 与 pending 同步；
- 补装成功的日志只打一次。
"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
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


def observation(username: str, *, inventory: list[dict] | None = None,
                held: str = "diamond_sword", health: float = 20.0) -> dict:
    # 目标名必须与对手名单一致，否则 decode_action 找不到可攻击对象（退回 idle）。
    other = "Glad_Bot_2" if username == "Glad_Bot_1" else "Glad_Bot_1"
    raw = make_observation(
        username, health=health,
        targets=[make_target(2 if username == "Glad_Bot_1" else 1,
                             other, x=1.0, z=0.0, distance=2.0)])
    raw["self"]["inventory"] = list(FULL_INVENTORY if inventory is None else inventory)
    raw["self"]["heldItem"] = {"name": held, "count": 1}
    raw["self"]["armor"] = dict(FULL_ARMOR)
    return raw


def build_env() -> tuple[GladiatorEnv, list[FakeLink], list[str]]:
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
    env.arena = type(env.arena)(**{**env.arena.__dict__, "give_kit": False})
    env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]
    return env, links, logs


class StripCombatActionsTest(unittest.TestCase):
    def test_strip_removes_combat_and_keeps_look_move(self):
        actions = [
            {"type": "look", "params": {"yaw": 0.0, "pitch": 0.0}},
            {"type": "attack", "params": {"targetEntityId": 2}},
            {"type": "move", "params": {"direction": "forward", "durationMs": 180}},
            {"type": "use", "params": {"item": "rod"}},
            {"type": "dig", "params": {}},
            {"type": "place", "params": {}},
        ]
        kept = GladiatorEnv._strip_combat_actions(actions)
        self.assertEqual([item["type"] for item in kept], ["look", "move"])

    def test_strip_keeps_equip_use_and_drops_offensive_use(self):
        """切剑/穿甲是补装动作本身，绝不能被剥（否则窗口 3 轮必然失败）。"""
        actions = [
            {"type": "attack", "params": {"targetEntityId": 2}},
            {"type": "use", "params": {"item": "sword", "select": True}},
            {"type": "use", "params": {"item": "armor"}},
            {"type": "use", "params": {"item": "gapple"}},
            {"type": "use", "params": {"item": "rod"}},
            {"type": "use", "params": {"item": "blocks"}},
            {"type": "use", "params": {"item": "food"}},
            {"type": "dig", "params": {}},
            {"type": "place", "params": {}},
            {"type": "jump", "params": {}},
            {"type": "look", "params": {"yaw": 1.0, "pitch": 0.0}},
            {"type": "move", "params": {"direction": "forward", "durationMs": 180}},
        ]
        kept = GladiatorEnv._strip_combat_actions(actions)

        kept_use = [(item.get("params") or {}).get("item") for item in kept
                    if item["type"] == "use"]
        self.assertEqual(kept_use, ["sword", "armor"])
        kinds = [item["type"] for item in kept]
        self.assertNotIn("attack", kinds)
        self.assertNotIn("dig", kinds)
        self.assertNotIn("place", kinds)
        self.assertEqual(kinds[-3:], ["jump", "look", "move"])

    def test_strip_keeps_select_use_with_any_item(self):
        kept = GladiatorEnv._strip_combat_actions(
            [{"type": "use", "params": {"item": "anything", "select": True}}])
        self.assertEqual([item["type"] for item in kept], ["use"])

    def test_strip_all_combat_falls_back_to_idle(self):
        kept = GladiatorEnv._strip_combat_actions(
            [{"type": "attack", "params": {"targetEntityId": 2}}])
        self.assertEqual([item["type"] for item in kept], ["idle"])


class StepGuardTest(unittest.TestCase):
    def test_learner_attack_is_stripped_while_pending(self):
        env, links, _logs = build_env()
        env._restock_pending = {"Glad_Bot_1": 3}
        env._restock_logged = set()
        env._restock_tick = {"Glad_Bot_1": 0}
        links[0].actions.clear()

        env.step({"bot_0": ATTACK, "bot_1": 0})

        sent = [item["type"] for item in links[0].actions]
        self.assertNotIn("attack", sent, sent)
        self.assertTrue(any(kind in ("look", "move") for kind in sent), sent)
        env.close()

    def test_learner_attack_resumes_after_restock_cleared(self):
        env, links, _logs = build_env()
        # 补装已确认（pending 空）：动作应原样下发。
        env._restock_pending = {}
        links[0].actions.clear()

        env.step({"bot_0": ATTACK, "bot_1": 0})

        self.assertIn("attack", [item["type"] for item in links[0].actions])
        env.close()

    def test_grace_never_shorter_than_pending(self):
        env, _links, _logs = build_env()
        # 上一帧还活着，这一帧血量回到 20 → 触发重生补装。
        dead = make_observation("Glad_Bot_1", health=0.0, state="Dead")
        previous = [dead, observation("Glad_Bot_2")]
        env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]

        env._restock_after_respawn(previous)

        self.assertEqual(env._restock_pending["Glad_Bot_1"], 3)
        self.assertGreaterEqual(env._respawn_grace["Glad_Bot_1"], 3)
        env.close()


class ConfirmRestockTest(unittest.TestCase):
    def test_success_logs_once_and_clears_grace(self):
        env, _links, logs = build_env()
        env._restock_pending = {"Glad_Bot_1": 3}
        env._restock_logged = set()
        env._restock_tick = {"Glad_Bot_1": 0}
        env._respawn_grace = {"Glad_Bot_1": 3}

        env._confirm_restock()
        env._confirm_restock()

        restock_lines = [line for line in logs if line.startswith("[restock]")]
        self.assertEqual(len(restock_lines), 1, logs)
        self.assertIn("已补齐", restock_lines[0])
        self.assertNotIn("Glad_Bot_1", env._restock_pending)
        self.assertNotIn("Glad_Bot_1", env._respawn_grace)
        env.close()

    def test_complete_kit_without_sword_in_hand_ends_window(self):
        """数量齐 + 护甲满即算补齐；没持剑只提示一行，不再白跑 3 轮。"""
        env, _links, logs = build_env()
        # 背包齐、护甲 4/4，但手里拿着金苹果
        env._raw[0] = observation("Glad_Bot_1", held="golden_apple")
        env._restock_pending = {"Glad_Bot_1": 3}
        env._restock_logged = set()
        env._restock_tick = {"Glad_Bot_1": 0}
        env._restock_rounds = {"Glad_Bot_1": 0}
        env._respawn_grace = {"Glad_Bot_1": 3}

        env._confirm_restock()

        self.assertNotIn("Glad_Bot_1", env._restock_pending)
        self.assertNotIn("Glad_Bot_1", env._restock_rounds)
        self.assertNotIn("Glad_Bot_1", env._respawn_grace)
        self.assertTrue(any("数量齐但不持剑" in line for line in logs), logs)
        self.assertTrue(any("已补齐" in line for line in logs), logs)
        self.assertFalse(any("补装失败" in line for line in logs), logs)
        env.close()

    def test_three_rounds_then_give_up_clears_grace(self):
        env, _links, logs = build_env()
        # 一直没有剑：3 轮 × 3 帧都失败后必须放弃并恢复行动（否则 Bot 永远站着）。
        env._raw[0] = observation("Glad_Bot_1", inventory=[], held="air")
        env._restock_pending = {"Glad_Bot_1": 3}
        env._restock_logged = set()
        env._restock_tick = {"Glad_Bot_1": 0}
        env._restock_rounds = {"Glad_Bot_1": 0}
        env._respawn_grace = {"Glad_Bot_1": 3}

        for _ in range(9):
            env._confirm_restock()

        self.assertNotIn("Glad_Bot_1", env._restock_pending)
        self.assertNotIn("Glad_Bot_1", env._respawn_grace)
        self.assertEqual(sum(1 for line in logs if "补装失败" in line), 1, logs)
        env.close()


if __name__ == "__main__":
    unittest.main()
