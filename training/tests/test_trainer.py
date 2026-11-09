from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target
from gladiator_train.policy import Policy
from gladiator_train.trainer import Trainer


def build_env(max_steps: int = 3) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
    links[0].set_observation(make_observation(
        "Glad_Bot_1", x=-1.5, z=0.0,
        targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=3.0)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, z=0.0,
        targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=3.0)]))
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        max_steps=max_steps,
        single_agent=True,
    )
    return env, links


class TrainerTest(unittest.TestCase):
    def _trainer(self, max_steps: int = 3) -> tuple[Trainer, GladiatorEnv, list[FakeLink]]:
        env, links = build_env(max_steps=max_steps)
        policy = Policy(obs_dim=int(env.observation_space.shape[0]),
                        n_actions=int(env.action_space.n), seed=11, hidden=8)
        return Trainer(env, policy, seed=12, log=lambda _msg: None), env, links

    def test_run_episode_collects_transitions(self):
        trainer, env, _ = self._trainer(max_steps=3)
        result = trainer.run_episode()
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(len(result.observations), 3)
        self.assertEqual(len(result.actions), 3)
        self.assertEqual(len(result.rewards), 3)
        env.close()

    def test_train_runs_updates_and_records_history(self):
        trainer, env, _ = self._trainer(max_steps=2)
        history = trainer.train(2)
        self.assertEqual(len(history), 2)
        self.assertEqual(trainer.policy.updates, 2)
        self.assertEqual(history[0]["episode"], 1)
        self.assertEqual(set(history[0]), {
            "episode", "steps", "totalReward", "terminated", "truncated",
            "winner", "durationSeconds", "startedAt", "startedAtEpoch", "meanHealth",
            "policyUpdates", "elapsedSeconds",
            "review",
        })
        env.close()

    def test_evaluate_uses_greedy_policy(self):
        trainer, env, _ = self._trainer(max_steps=2)
        results = trainer.evaluate(1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["steps"], 2)
        env.close()


if __name__ == "__main__":
    unittest.main()
