from __future__ import annotations

import time
import unittest

import numpy as np

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM
from gladiator_env.testing import FakeLink, make_observation, make_target


def build_env(**kwargs) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
    links[0].set_observation(make_observation(
        "Glad_Bot_1", x=-1.5, z=0.0, targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=3.0)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, z=0.0, targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=3.0)]))
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        **kwargs,
    )
    return env, links


class GymApiTest(unittest.TestCase):
    def test_spaces(self):
        env, _ = build_env()
        self.assertEqual(set(env.observation_space.spaces), {"bot_0", "bot_1"})
        self.assertEqual(env.observation_space["bot_0"].shape, (OBS_DIM,))
        self.assertEqual(env.action_space["bot_1"].n, NUM_ACTIONS)
        env.close()

    def test_reset_returns_observation_and_info(self):
        env, _ = build_env()
        observation, info = env.reset()
        self.assertIn("bot_0", observation)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["step"], 0)
        env.close()

    def test_step_returns_five_tuple(self):
        env, links = build_env(max_steps=10)
        env.reset()
        observation, rewards, terminated, truncated, info = env.step({"bot_0": 1, "bot_1": 1})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(set(rewards), {"bot_0", "bot_1"})
        self.assertIsInstance(terminated, bool)
        self.assertFalse(truncated)
        self.assertEqual(info["step"], 1)
        # 动作确实下发到通道 B（首帧是 look）
        self.assertTrue(any(a["type"] == "move" for a in links[0].actions))
        env.close()

    def test_truncates_at_max_steps(self):
        env, _ = build_env(max_steps=2)
        env.reset()
        env.step({"bot_0": 0, "bot_1": 0})
        _, _, _, truncated, _ = env.step({"bot_0": 0, "bot_1": 0})
        self.assertTrue(truncated)
        env.close()

    def test_terminates_when_bot_dies(self):
        env, links = build_env()
        env.reset()
        links[1].set_observation(make_observation(
            "Glad_Bot_2", x=1.5, z=0.0, health=0.0, state="Dead",
            targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=3.0)]))
        _, rewards, terminated, _, info = env.step({"bot_0": 2, "bot_1": 0})
        self.assertTrue(terminated)
        self.assertEqual(info["winner"], "Glad_Bot_1")
        self.assertGreater(rewards["bot_0"], rewards["bot_1"])
        env.close()

    def test_close_stops_links(self):
        env, links = build_env()
        env.reset()
        env.close()
        self.assertTrue(all(link.stopped for link in links))

    def test_idle_snapshot_within_threshold(self):
        # G-405：持续下发动作时不应被判为空闲。
        env, _ = build_env(max_steps=5, idle_threshold=5.0)
        env.reset()
        env.step({"bot_0": 0, "bot_1": 1})
        snapshot = env.idle_snapshot()
        self.assertEqual(set(snapshot["seconds"]), {"Glad_Bot_1", "Glad_Bot_2"})
        self.assertFalse(snapshot["idle"]["Glad_Bot_2"])
        self.assertFalse(snapshot["idle"]["Glad_Bot_1"])
        env.close()

    def test_idle_flag_after_idle_actions(self):
        env, _ = build_env(max_steps=5, idle_threshold=0.0)
        env.reset()
        env.step({"bot_0": 0, "bot_1": 0})
        snapshot = env.idle_snapshot()
        self.assertTrue(snapshot["idle"]["Glad_Bot_1"])
        self.assertTrue(snapshot["idle"]["Glad_Bot_2"])
        env.close()

    def test_single_agent_returns_vector_and_float(self):
        env, _ = build_env(max_steps=5, single_agent=True)
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (OBS_DIM,))
        self.assertTrue(env.observation_space.contains(observation))
        observation, reward, terminated, truncated, _ = env.step(1)
        self.assertEqual(observation.shape, (OBS_DIM,))
        self.assertIsInstance(reward, float)
        env.close()


class PrepPhaseTest(unittest.TestCase):
    """开局准备期：压住双方、留出穿甲时间（观战里“没穿衣服就开打”的修复）。"""

    def test_prep_phase_sends_idle_pulses(self):
        env, links = build_env(arena=ArenaConfig(
            center=(0.0, 4.0, 0.0), spacing=3.0, prep_seconds=0.4, give_kit=False))

        class _StubRcon:
            def command(self, _command: str) -> str:
                return ""

        env._rcon = _StubRcon()
        started = time.monotonic()
        env._prep_phase()
        self.assertGreaterEqual(time.monotonic() - started, 0.35)
        for link in links:
            self.assertTrue(any(action["type"] == "idle" for action in link.actions))
        env.close()

    def test_prep_phase_applies_and_clears_resistance(self):
        env, _ = build_env(arena=ArenaConfig(
            center=(0.0, 4.0, 0.0), spacing=3.0, prep_seconds=0.2, give_kit=False))

        class _StubRcon:
            def __init__(self):
                self.commands = []

            def command(self, command: str) -> str:
                self.commands.append(command)
                return ""

        stub = _StubRcon()
        env._rcon = stub
        env._prep_phase()
        applied = [c for c in stub.commands if c.startswith("effect ") and " 11 " in c and not c.endswith(" 0 0")]
        cleared = [c for c in stub.commands if c.endswith(" 11 0 0")]
        self.assertEqual(len(applied), 2)
        self.assertEqual(len(cleared), 2)
        env.close()

    def test_spacing_is_clamped_to_minimum(self):
        # 表单留空/填 0 时两个 Bot 会叠在同一个点，死亡后原地复活互相卡死。
        env, _ = build_env(arena=ArenaConfig(spacing=0.0))
        self.assertGreaterEqual(env.arena.spacing, 4.0)
        env.close()

    def test_prep_phase_disabled_is_noop(self):
        env, links = build_env(arena=ArenaConfig(prep_seconds=0.0))
        env._prep_phase()
        self.assertEqual(links[0].actions, [])
        env.close()


if __name__ == "__main__":
    unittest.main()
