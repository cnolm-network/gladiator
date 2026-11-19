"""A2：重生补装确认（give 无回执，重生后 1–2 tick 内可能仍然缺件）。"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target

FULL_ARMOR = {
    "head": {"name": "diamond_helmet"},
    "torso": {"name": "diamond_chestplate"},
    "legs": {"name": "diamond_leggings"},
    "feet": {"name": "diamond_boots"},
}


class NowaitLink(FakeLink):
    def send_actions_nowait(self, actions: list[dict]) -> None:
        self.actions.extend(actions)


def observation(username: str, *, inventory: list[dict], held: str,
                armor: dict | None = None) -> dict:
    raw = make_observation(
        username, health=20.0,
        targets=[make_target(2, "Foe", x=1.0, z=0.0, distance=2.0)])
    raw["self"]["inventory"] = inventory
    raw["self"]["heldItem"] = {"name": held, "count": 1}
    raw["self"]["armor"] = armor if armor is not None else dict(FULL_ARMOR)
    return raw


class FakeRcon:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, text: str) -> str:
        self.commands.append(text)
        return ""


def build_env() -> tuple[GladiatorEnv, list[NowaitLink], list[str], FakeRcon]:
    links = [NowaitLink("Glad_Bot_1"), NowaitLink("Glad_Bot_2")]
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
    env.arena = type(env.arena)(**{**env.arena.__dict__, "give_kit": True})
    rcon = FakeRcon()
    env._rcon = rcon
    return env, links, logs, rcon


def set_bot2(env: GladiatorEnv, raw: dict) -> None:
    env._raw[1] = raw


class RestockConfirmTest(unittest.TestCase):

    def test_first_check_issues_give_for_missing_items(self):
        env, _links, logs, rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        env._restock_pending = {"Glad_Bot_2": 3}
        env._restock_logged = set()

        env._confirm_restock()

        joined = " | ".join(rcon.commands)
        self.assertIn("give Glad_Bot_2 diamond_sword", joined)
        self.assertIn("give Glad_Bot_2 golden_apple", joined)
        self.assertEqual(env._restock_pending["Glad_Bot_2"], 2)
        self.assertEqual(sum(1 for line in logs if "补装中" in line), 1)

    def test_second_check_clears_when_complete(self):
        env, _links, logs, rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        env._restock_pending = {"Glad_Bot_2": 3}
        env._restock_logged = set()
        env._confirm_restock()
        rcon.commands.clear()
        logs.clear()

        # 第二帧：套件已到位
        set_bot2(env, observation("Glad_Bot_2", inventory=[
            {"slot": 36, "name": "diamond_sword", "count": 1},
            {"slot": 37, "name": "golden_apple", "count": 5},
            {"slot": 38, "name": "fishing_rod", "count": 1},
        ], held="diamond_sword"))
        env._confirm_restock()

        self.assertEqual(rcon.commands, [])
        self.assertNotIn("Glad_Bot_2", env._restock_pending)
        self.assertTrue(any("已补齐" in line for line in logs), logs)

    def test_missing_log_only_once_per_cycle(self):
        env, _links, logs, _rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        env._restock_pending = {"Glad_Bot_2": 3}
        env._restock_logged = set()

        env._confirm_restock()
        env._confirm_restock()

        self.assertEqual(sum(1 for line in logs if "补装中" in line), 1)
        self.assertEqual(env._restock_pending["Glad_Bot_2"], 1)

    def test_window_exhausted_retries_then_gives_up(self):
        """窗口用完不再「到期就放弃」：先重开一轮，3 轮都失败才清掉 pending。"""
        env, _links, logs, _rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        env._restock_pending = {"Glad_Bot_2": 3}
        env._restock_logged = set()
        env._restock_rounds = {}

        # 第 1 轮用完 → 自动重开（pending 仍是 3，轮数 1）
        for _ in range(3):
            env._confirm_restock()
        self.assertEqual(env._restock_pending.get("Glad_Bot_2"), 3)
        self.assertEqual(env._restock_rounds.get("Glad_Bot_2"), 1)

        # 3 轮（9 帧）都失败 → 才放弃
        for _ in range(6):
            env._confirm_restock()
        self.assertNotIn("Glad_Bot_2", env._restock_pending)
        self.assertEqual(sum(1 for line in logs if "补装失败" in line), 1)

    def test_no_pending_is_noop(self):
        env, links, _logs, rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]

        env._confirm_restock()

        self.assertEqual(rcon.commands, [])
        self.assertEqual(links[0].actions, [])
        self.assertEqual(links[1].actions, [])

    def test_respawn_sets_pending_and_clears_log_flag(self):
        env, _links, _logs, _rcon = build_env()
        dead = make_observation("Glad_Bot_2", health=0.0, state="Dead")
        previous = [observation("Glad_Bot_1", inventory=[], held="air"), dead]
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        env._restock_logged = {"Glad_Bot_2"}

        env._restock_after_respawn(previous)

        self.assertEqual(env._restock_pending.get("Glad_Bot_2"), 3)
        self.assertNotIn("Glad_Bot_2", env._restock_logged)


class GiveSettleTimingTest(unittest.TestCase):
    """金苹果补装时序（4 场地日志反复出现「补装失败（缺 golden_apple）」的修复）。

    旧实现同一 tick 里 give 完就在下一帧判「仍缺 → 重发」，于是 1.8 的背包还没同步
    就已经被记成失败。新规则：逐物品记账、give 后至少等 2 tick 复核、只重发仍缺的项、
    单个物品整轮最多 3 次。
    """

    def _env(self):
        env, _links, logs, rcon = build_env()
        env._raw = [observation("Glad_Bot_1", inventory=[], held="air"),
                    observation("Glad_Bot_2", inventory=[], held="air")]
        return env, logs, rcon

    @staticmethod
    def _gives(rcon, item="golden_apple"):
        return [cmd for cmd in rcon.commands if cmd.startswith("give") and f" {item} " in cmd]

    def test_give_is_not_repeated_within_settle_window(self):
        env, _logs, rcon = self._env()
        env._request_restock(0, reason="code=ITEM_MISSING")
        self.assertEqual(len(self._gives(rcon)), 1)

        # 第 1 帧：等同步，不重发
        env._confirm_restock()
        self.assertEqual(len(self._gives(rcon)), 1)
        # 第 2 帧：等待窗口结束，仍缺 → 重发一次（第 2 次）
        env._confirm_restock()
        self.assertEqual(len(self._gives(rcon)), 2)

    def test_only_missing_items_are_resent(self):
        env, _logs, rcon = self._env()
        env._request_restock(0, reason="code=ITEM_MISSING")
        rcon.commands.clear()

        # 剑和鱼竿已经到位，只有金苹果没同步
        env._raw[0] = observation("Glad_Bot_1", inventory=[
            {"slot": 36, "name": "diamond_sword", "count": 1},
            {"slot": 38, "name": "fishing_rod", "count": 1},
        ], held="diamond_sword")
        env._confirm_restock()
        env._confirm_restock()

        gives = [cmd for cmd in rcon.commands if cmd.startswith("give")]
        self.assertTrue(gives, "仍缺金苹果，应该重发")
        self.assertTrue(all("golden_apple" in cmd for cmd in gives), gives)

    def test_at_most_three_gives_per_item_per_cycle(self):
        env, _logs, rcon = self._env()
        env._request_restock(0, reason="code=ITEM_MISSING")
        for _ in range(9):
            env._confirm_restock()

        self.assertEqual(len(self._gives(rcon)), 3)
        self.assertNotIn("Glad_Bot_1", env._restock_pending)
        self.assertNotIn("Glad_Bot_1", env._restock_gives)

    def test_new_cycle_resets_item_budget(self):
        env, _logs, rcon = self._env()
        env._request_restock(0, reason="code=ITEM_MISSING")
        for _ in range(9):
            env._confirm_restock()
        self.assertEqual(len(self._gives(rcon)), 3)

        # 新一轮补装：逐物品的重发预算重新记
        env._request_restock(0, reason="periodic")
        self.assertEqual(len(self._gives(rcon)), 4)

    def test_give_settle_state_cleared_on_success(self):
        env, _logs, rcon = self._env()
        env._request_restock(0, reason="code=ITEM_MISSING")
        env._raw[0] = observation("Glad_Bot_1", inventory=[
            {"slot": 36, "name": "diamond_sword", "count": 1},
            {"slot": 37, "name": "golden_apple", "count": 5},
            {"slot": 38, "name": "fishing_rod", "count": 1},
        ], held="diamond_sword")
        env._confirm_restock()

        self.assertNotIn("Glad_Bot_1", env._restock_pending)
        self.assertNotIn("Glad_Bot_1", env._restock_gives)
        self.assertEqual(len(self._gives(rcon)), 1)


if __name__ == "__main__":
    unittest.main()
