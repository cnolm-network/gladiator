"""P1.6 评估口径：训练同口径采样 vs 贪心 argmax。

背景：训练用采样、评估用 argmax，量到的是「另一个策略」；评估 0 胜但训练 62.5% 胜，
很大一部分差异来自这里。所以报告必须同时给两组。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import gladiator_env  # noqa: E402
from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM  # noqa: E402
from gladiator_train import train as train_module  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train.trainer import Trainer  # noqa: E402
from tests.test_bc import build_fake_env  # noqa: E402


class RecordingEnv:
    """最小 env：记录每一步收到的动作编号，用于比较「采样 vs 贪心」序列。"""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ActionSpace:
        n = NUM_ACTIONS

    def __init__(self, length: int = 30):
        self.observation_space = self._Space((OBS_DIM,))
        self.action_space = self._ActionSpace()
        self.length = length
        self.actions: list[int] = []
        self.bots = [type("B", (), {"username": "Bot_A"})(),
                     type("B", (), {"username": "Bot_B"})()]
        self.raw_observations = []
        self.opponent_mode = "scripted"
        self.shaping = None
        self._t = 0

    def reset(self, **_kw):
        self._t = 0
        return np.zeros(OBS_DIM, dtype=np.float32), {"step": 0}

    def step(self, action):
        self.actions.append(int(action))
        self._t += 1
        done = self._t >= self.length
        return (np.zeros(OBS_DIM, dtype=np.float32), 0.0, done, False,
                {"winner": None, "health": {}})

    def last_events(self):
        return {}

    def last_dispatch(self, _username):
        return []

    def close(self):
        return None


def near_uniform_policy(seed: int = 0) -> Policy:
    """把输出压成接近均匀（argmax 明确但采样常选到别的），才能看出两种口径的差别。"""
    policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=seed)
    policy.params["W2"] *= 0.0
    policy.params["b2"] *= 0.0
    policy.params["b2"][0] = 0.02   # 极小的偏好 → 概率接近均匀
    return policy


class SamplingVsGreedyTest(unittest.TestCase):
    def test_sequences_differ_under_same_seed(self):
        env = RecordingEnv(length=40)
        policy = near_uniform_policy()
        trainer = Trainer(env, policy, seed=3, log=lambda _m: None)

        trainer.rng = np.random.default_rng(11)
        trainer.evaluate(1, greedy=False)
        sampled = list(env.actions)

        env.actions.clear()
        trainer.rng = np.random.default_rng(11)
        trainer.evaluate(1, greedy=True)
        greedy = list(env.actions)

        self.assertEqual(len(sampled), len(greedy))
        self.assertNotEqual(sampled, greedy, "采样与贪心必须是不同序列，否则口径没分开")
        self.assertEqual(len(set(greedy)), 1, "贪心应恒定输出 argmax")

    def test_greedy_is_deterministic_across_seeds(self):
        env = RecordingEnv(length=10)
        policy = near_uniform_policy()
        trainer = Trainer(env, policy, seed=5, log=lambda _m: None)
        trainer.evaluate(1, greedy=True)
        first = list(env.actions)
        env.actions.clear()
        trainer.rng = np.random.default_rng(99)
        trainer.evaluate(1, greedy=True)
        self.assertEqual(first, env.actions)


class ReportKeysTest(unittest.TestCase):
    """报告里两组评估都在，且 episode 数一致。"""

    def _run(self, tmp: Path, *extra: str) -> dict:
        run_dir = tmp / "run"
        argv = [
            "--users", "Bot_T1", "Bot_T2", "--ports", "8766", "8767",
            "--run-dir", str(run_dir),
            "--save", str(run_dir / "policy.npz"),
            "--report", str(run_dir / "report.json"),
            "--review-dir", str(run_dir / "reviews"),
            "--episodes", "1", "--max-steps", "3",
            "--eval-epochs", "1", "--bc-epochs", "0",
            *extra,
        ]
        with mock.patch.object(gladiator_env, "GladiatorEnv", build_fake_env):
            code = train_module.main(argv)
        self.assertEqual(code, 0)
        return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    def test_both_pass_present_with_same_episode_count(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self._run(Path(folder))
            self.assertEqual(report["config"]["evalSampling"], True)
            self.assertIn("evaluation", report)
            self.assertIn("evaluationGreedy", report)
            self.assertEqual(len(report["evaluation"]), len(report["evaluationGreedy"]))
            self.assertGreater(len(report["evaluation"]), 0)
            for key in ("evaluationByOpponent", "evaluationGreedyByOpponent",
                        "evaluationOverall", "evaluationGreedyOverall"):
                self.assertIn(key, report)
            # 结构一致：同一条 record 的键集相同
            self.assertEqual(set(report["evaluation"][0]), set(report["evaluationGreedy"][0]))
            # P1.7：对手池区间结构化落盘（dashboard 可直接展示，不必改代码）
            pool = report["config"]["opponentPool"]
            self.assertEqual(pool["modes"]["rusher"]["startDelayTicks"], [10, 40])
            self.assertEqual(pool["modes"]["scripted"]["attackCooldownTicks"], [2, 5])
            self.assertEqual(pool["modes"]["aimbot"]["attackCooldownTicks"], [1, 2])
            self.assertEqual(pool["reachOffset"], [0.5, 1.0])

    def test_arena_jitter_cli_zero_is_deterministic(self):
        """--arena-jitter 0,0 必须回到完全确定的旧行为，并在报告里记录。"""
        with tempfile.TemporaryDirectory() as folder:
            report = self._run(Path(folder), "--arena-jitter", "0,0")
            self.assertEqual(report["config"]["arenaJitter"], [0.0, 0.0])
            # 默认值是 2,1（起始分布随机化）
        with tempfile.TemporaryDirectory() as folder:
            default_report = self._run(Path(folder))
            self.assertEqual(default_report["config"]["arenaJitter"], [2.0, 1.0])

    def test_eval_greedy_makes_both_passes_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self._run(Path(folder), "--eval-greedy")
            self.assertEqual(report["config"]["evalSampling"], False)
            self.assertEqual(report["evaluation"], report["evaluationGreedy"])
            self.assertEqual(report["evaluationByOpponent"], report["evaluationGreedyByOpponent"])


if __name__ == "__main__":
    unittest.main()
