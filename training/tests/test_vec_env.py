"""VecEnv 的离线测试（假 env，不连服务器）。

覆盖：
- 批次数/步数累加、每批一次参数更新（测试用 epochs=1 + minibatch=0）；
- worker 异常时关闭全部 env 并把异常抛出来；
- ``algo=reinforce`` 明确拒绝（SystemExit）；
- GAE 逐轨迹计算（monkeypatch ``trainer.gae`` 记录每次的 rewards）；
- steps/s 统计在假时钟下有限且 > 0；
- 场地规划、spawn/close 幂等。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import gladiator_train.trainer as trainer_module  # noqa: E402
from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train.trainer import EpisodeResult, Trainer  # noqa: E402
from gladiator_train.vec_env import (  # noqa: E402
    VecEnv,
    VecTrainer,
    plan_arenas,
    trajectory_gae,
)

FAST_KWARGS = {"epochs": 1, "minibatch": 0, "normalize_returns": False}


class FakeClock:
    """每次读时间推进固定步长：steps/s 可以在毫秒内被精确计算。"""

    def __init__(self, step: float = 0.25) -> None:
        self.t = 0.0
        self.step = float(step)

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class FakeEnv:
    """固定步数/固定 reward 的最小 env；可选在某一步抛异常。"""

    def __init__(self, name: str, *, steps: int = 3, reward: float = 1.0,
                 winner: str = "Bot_A", fail_at: int | None = None) -> None:
        self.name = name
        self.opponent_mode = "scripted"
        self.bots = [SimpleNamespace(username="Bot_A"), SimpleNamespace(username="Bot_B")]
        self.max_steps = int(steps)
        self.reward = float(reward)
        self.winner = winner
        self.fail_at = fail_at
        self.reset_seeds: list = []
        self.closed = 0
        self.spawned = 0
        self._t = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._t = 0
        return np.zeros(2, dtype=np.float32), {"step": 0}

    def step(self, action):
        self._t += 1
        if self.fail_at is not None and self._t >= self.fail_at:
            raise RuntimeError(f"boom@{self.name}")
        done = self._t >= self.max_steps
        info = {
            "step": self._t,
            "winner": self.winner if done else None,
            "health": {"Bot_A": 20.0, "Bot_B": 10.0},
        }
        return np.zeros(2, dtype=np.float32), self.reward, done, False, info

    def close(self) -> None:
        self.closed += 1

    def _spawn_bots(self) -> None:
        self.spawned += 1


def make_policy() -> Policy:
    return Policy(obs_dim=2, n_actions=2, hidden=4, seed=0, lr=1e-3)


class VecEnvLifecycleTests(unittest.TestCase):
    def test_plan_arenas_names_ports_and_centers(self) -> None:
        specs = plan_arenas(3, users_prefix="Bot_V", ports_base=8770,
                            arena_gap=80.0, center=(0.5, 4.0, 0.5))
        self.assertEqual([spec.users for spec in specs],
                         [("Bot_V0A", "Bot_V0B"), ("Bot_V1A", "Bot_V1B"),
                          ("Bot_V2A", "Bot_V2B")])
        self.assertEqual([spec.ports for spec in specs],
                         [(8770, 8771), (8772, 8773), (8774, 8775)])
        self.assertEqual([spec.center for spec in specs],
                         [(0.5, 4.0, 0.5), (80.5, 4.0, 0.5), (160.5, 4.0, 0.5)])

    def test_spawn_and_close_are_idempotent(self) -> None:
        envs = [FakeEnv("a"), FakeEnv("b")]
        vec = VecEnv(envs)
        self.assertEqual(vec.count, 2)
        vec.spawn_all()
        vec.spawn_all()
        self.assertEqual([env.spawned for env in envs], [1, 1])
        vec.close_all()
        vec.close_all()
        self.assertEqual([env.closed for env in envs], [1, 1])

    def test_spawn_failure_closes_all_and_raises(self) -> None:
        class BadSpawn(FakeEnv):
            def _spawn_bots(self) -> None:
                raise RuntimeError("spawn failed")

        envs = [FakeEnv("ok"), BadSpawn("bad")]
        vec = VecEnv(envs)
        with self.assertRaises(RuntimeError):
            vec.spawn_all()
        self.assertTrue(all(env.closed == 1 for env in envs))


class VecTrainerTests(unittest.TestCase):
    def _make(self, count: int = 3, **env_kwargs):
        envs = [FakeEnv(f"e{i}", **env_kwargs) for i in range(count)]
        vec = VecEnv(envs)
        policy = make_policy()
        trainer = VecTrainer(vec, policy, gamma=0.99, gae_lambda=0.95, seed=7,
                             workers=count, trainer_kwargs=dict(FAST_KWARGS),
                             max_steps=env_kwargs.get("steps", 3),
                             clock=FakeClock())
        return vec, envs, policy, trainer

    def test_two_batches_six_episodes_two_updates(self) -> None:
        vec, envs, policy, trainer = self._make(3, steps=3, reward=1.0)
        history = trainer.train(2)
        self.assertEqual(len(history), 2)
        self.assertEqual(trainer.summary()["episodes"], 6)
        self.assertEqual(trainer.summary()["steps"], 18)
        self.assertEqual([item["steps"] for item in history], [9, 9])
        # epochs=1 + minibatch=0 → 每批恰好一次参数更新
        self.assertEqual(policy.updates, 2)
        # 同 seed 集：seed + batch*M + index
        self.assertEqual([env.reset_seeds for env in envs], [[7, 10], [8, 11], [9, 12]])
        self.assertTrue(all(env.closed == 0 for env in envs), "成功路径不主动关闭 env")
        vec.close_all()

    def test_worker_exception_closes_all_envs_and_propagates(self) -> None:
        envs = [FakeEnv("ok"), FakeEnv("bad", fail_at=1)]
        vec = VecEnv(envs)
        trainer = VecTrainer(vec, make_policy(), seed=7, workers=2,
                             trainer_kwargs=dict(FAST_KWARGS), max_steps=3)
        with self.assertRaises(RuntimeError) as ctx:
            trainer.train(1)
        self.assertIn("boom@bad", str(ctx.exception))
        self.assertEqual([env.closed for env in envs], [1, 1])

    def test_reinforce_is_rejected(self) -> None:
        vec = VecEnv([FakeEnv("a")])
        with self.assertRaises(SystemExit) as ctx:
            VecTrainer(vec, make_policy(), algo="reinforce")
        self.assertIn("只支持 ppo", str(ctx.exception))

    def test_steps_per_second_is_finite_and_positive(self) -> None:
        vec, _envs, _policy, trainer = self._make(3, steps=3)
        trainer.train(2)
        summary = trainer.summary()
        self.assertTrue(math.isfinite(summary["stepsPerSecond"]))
        self.assertGreater(summary["stepsPerSecond"], 0.0)
        self.assertEqual(summary["batches"], 2)
        self.assertEqual(summary["policyUpdates"], _policy.updates)
        self.assertEqual(summary["byOpponent"]["scripted"]["episodes"], 6)
        self.assertEqual(summary["byOpponent"]["scripted"]["wins"], 6)
        vec.close_all()


class StubPolicy:
    """只实现 ppo_update_many 需要的接口，用来捕获传入的 advantages。"""

    PARAM_NAMES = ("W1", "b1", "W2", "b2", "Wv", "bv")

    def __init__(self) -> None:
        self.lr = 0.0
        self.updates = 0
        self.temperature = 1.0
        self.n_actions = 2

    def evaluate_actions(self, observations, actions):
        n = len(actions)
        return np.zeros(n), np.zeros(n), np.zeros(n)

    def ppo_gradients(self, *args, **kwargs):
        return {name: np.zeros(1) for name in self.PARAM_NAMES}

    def apply_gradients(self, grads, max_grad_norm: float = 0.0) -> float:
        self.updates += 1
        return 0.0

    def mean_max_prob(self, observations) -> float:
        return 0.0

    def logit_std(self, observations) -> float:
        return 0.0


def _result(rewards) -> EpisodeResult:
    steps = len(rewards)
    return EpisodeResult(
        steps=steps,
        total_reward=float(sum(rewards)),
        terminated=True,
        truncated=False,
        winner="Bot_A",
        duration_s=0.1,
        observations=[np.zeros(2) for _ in range(steps)],
        actions=[0] * steps,
        rewards=list(rewards),
    )


class PerTrajectoryGAETests(unittest.TestCase):
    def test_gae_called_once_per_trajectory(self) -> None:
        stub = StubPolicy()
        trainer = Trainer(env=SimpleNamespace(), policy=stub, algo="ppo",
                          epochs=1, minibatch=0, normalize_returns=False,
                          gamma=0.99, gae_lambda=0.95)
        calls = []
        original = trainer_module.gae

        def recording(rewards, values, dones, gamma=0.99, lam=0.95):
            advantages, returns = original(rewards, values, dones, gamma=gamma, lam=lam)
            calls.append({"rewards": list(rewards),
                          "advantages": np.asarray(advantages, dtype=float)})
            return advantages, returns

        with mock.patch.object(trainer_module, "gae", recording):
            trainer.ppo_update_many([_result([1.0, 1.0, 1.0]), _result([1.0, 0.0, 0.0])])

        self.assertEqual([call["rewards"] for call in calls],
                         [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        first = calls[0]["advantages"]
        second = calls[1]["advantages"]
        self.assertGreater(abs(first).sum(), abs(second).sum())
        self.assertGreater(abs(first[0]), abs(second[0]))
        # 逐轨迹 bootstrap：两条轨迹的末步 advantage 只由自身 reward 决定
        self.assertAlmostEqual(first[2], 1.0, places=12)
        self.assertAlmostEqual(second[2], 0.0, places=12)

    def test_trajectory_gae_matches_manual_recursion(self) -> None:
        advantages, returns = trajectory_gae([1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                             gamma=0.99, lam=0.95)
        self.assertAlmostEqual(advantages[2], 1.0, places=12)
        self.assertAlmostEqual(advantages[1], 1.0 + 0.99 * 0.95 * 1.0, places=12)
        self.assertAlmostEqual(advantages[0], 1.0 + 0.99 * 0.95 * advantages[1], places=12)
        self.assertAlmostEqual(returns[0], advantages[0], places=12)


class VecTrainerRealPolicyTests(unittest.TestCase):
    """用真 Policy 跑一批，确认 ppo_update_many 不会因参数形状炸掉。"""

    def test_one_batch_updates_real_policy(self) -> None:
        envs = [FakeEnv("a", steps=4), FakeEnv("b", steps=4)]
        vec = VecEnv(envs)
        policy = make_policy()
        trainer = VecTrainer(vec, policy, seed=3, workers=2,
                             trainer_kwargs=dict(FAST_KWARGS), max_steps=4,
                             clock=FakeClock())
        history = trainer.train(1)
        self.assertEqual(policy.updates, 1)
        self.assertEqual(history[0]["episodes"], 2)
        self.assertGreater(history[0]["stepsPerSecond"], 0.0)
        vec.close_all()


if __name__ == "__main__":
    unittest.main()
