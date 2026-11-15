"""A1：回合中死亡重生后补发套件。"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target


class NowaitLink(FakeLink):
    """补上 `_send_use` 依赖的「不等结果发送」接口。"""

    def send_actions_nowait(self, actions: list[dict]) -> None:
        self.actions.extend(actions)


def build_env() -> tuple[GladiatorEnv, list[NowaitLink], list[str]]:
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
    return env, links, logs


def observation(username: str, *, health: float, state: str = "Alive") -> dict:
    return make_observation(
        username,
        health=health,
        state=state,
        targets=[make_target(2, "Foe", x=1.5, z=0.0, distance=3.0)],
    )


class RespawnRestockTest(unittest.TestCase):
    def test_respawn_reselects_sword(self):
        env, links, logs = build_env()
        previous = [observation("Glad_Bot_1", health=0.0, state="Dead"),
                    observation("Glad_Bot_2", health=20.0)]
        env._raw = [observation("Glad_Bot_1", health=20.0),
                    observation("Glad_Bot_2", health=20.0)]

        env._restock_after_respawn(previous)

        sent = [a for a in links[0].actions if a.get("type") == "use"]
        self.assertEqual(sent, [{"type": "use", "params": {"item": "sword", "select": True}}])
        self.assertEqual(links[1].actions, [])
        self.assertTrue(any("重生" in line for line in logs), logs)

    def test_no_respawn_does_nothing(self):
        env, links, logs = build_env()
        previous = [observation("Glad_Bot_1", health=20.0),
                    observation("Glad_Bot_2", health=20.0)]
        env._raw = [observation("Glad_Bot_1", health=20.0),
                    observation("Glad_Bot_2", health=18.0)]

        env._restock_after_respawn(previous)

        self.assertEqual(links[0].actions, [])
        self.assertEqual(links[1].actions, [])
        self.assertFalse(any("重生" in line for line in logs))

    def test_first_step_without_previous_health_is_ignored(self):
        env, links, _ = build_env()
        previous: list[dict] = [{}, {}]
        env._raw = [observation("Glad_Bot_1", health=20.0),
                    observation("Glad_Bot_2", health=20.0)]

        env._restock_after_respawn(previous)

        self.assertEqual(links[0].actions, [])
        self.assertEqual(links[1].actions, [])


if __name__ == "__main__":
    unittest.main()
