"""A2锛氱敤 STATE 甯цˉ銆岄噰鏍疯烦杩囩殑姝讳骸銆嶃€?
1.8 鐨勬浜♀啋鑷姩閲嶇敓鍙湁 2 tick锛堚増100ms锛夛紝瑙傛祴鏄?5Hz 閲囨牱锛屽父甯告暣涓法杩囧幓锛?涓嶈ˉ鐨勮瘽鍥炲悎涓嶇粨鏉熴€佸鎵嬭８濂斿弽澶嶉噸鐢熴€佽ˉ瑁呯獥鍙ｆ瘡 tick 鍙?give 鎶婂悶鍚愭墦鍒?1.8銆?"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target


class StateLink(FakeLink):
    def __init__(self, username: str):
        super().__init__(username)
        self.states: list[dict] = []

    def state_events(self) -> list[dict]:
        events = list(self.states)
        self.states.clear()
        return events


def build_env() -> tuple[GladiatorEnv, list[StateLink]]:
    links = [StateLink("Glad_Bot_1"), StateLink("Glad_Bot_2")]
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
    )
    env._raw = [
        make_observation("Glad_Bot_1", health=20.0,
                         targets=[make_target(2, "Glad_Bot_2", x=1.0, z=0.0, distance=1.0)]),
        make_observation("Glad_Bot_2", health=20.0,
                         targets=[make_target(1, "Glad_Bot_1", x=1.0, z=0.0, distance=1.0)]),
    ]
    return env, links


class StateDeathTest(unittest.TestCase):
    def test_state_dead_rewrites_observation(self):
        env, links = build_env()
        links[1].states.append({"state": "Dead", "previous": "Alive", "reason": "player-death"})

        deaths = env._apply_state_deaths()

        self.assertEqual(deaths, {"Glad_Bot_2"})
        self.assertEqual(env._raw[1]["self"]["health"], 0.0)
        self.assertEqual(env._raw[1]["state"], "Dead")
        self.assertEqual(env._raw[0]["self"]["health"], 20.0)
        self.assertEqual(env._winner(), "Glad_Bot_1")
        self.assertGreater(env._restock_pending["Glad_Bot_2"], 0)

    def test_learner_death_also_detected(self):
        env, links = build_env()
        links[0].states.append({"state": "Dead", "previous": "Alive"})

        deaths = env._apply_state_deaths()

        self.assertEqual(deaths, {"Glad_Bot_1"})
        self.assertEqual(env._winner(), "Glad_Bot_2")

    def test_respawn_event_is_not_a_death(self):
        env, links = build_env()
        links[0].states.append({"state": "Respawning", "previous": "Dead"})

        self.assertEqual(env._apply_state_deaths(), set())
        self.assertEqual(env._raw[0]["self"]["health"], 20.0)

    def test_repeated_dead_without_revive_is_ignored(self):
        env, links = build_env()
        env._raw[1]["self"]["health"] = 0.0
        env._raw[1]["state"] = "Dead"
        links[1].states.append({"state": "Dead", "previous": "Alive"})

        self.assertEqual(env._apply_state_deaths(), set())

    def test_events_are_drained(self):
        env, links = build_env()
        links[0].states.append({"state": "Dead", "previous": "Alive"})

        env._apply_state_deaths()

        self.assertEqual(links[0].states, [])


if __name__ == "__main__":
    unittest.main()
