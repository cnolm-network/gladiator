"""P3：PPO 稳定性（lr 衰减 / 熵下限 / minibatch / 梯度裁剪 / 早停 / 诊断）。

背景：run 20261118-0003 里 entropy 从 0.44 掉到 0.011，ep10 之后
policyLoss≈approxKL≈clipFraction≈0 —— 策略被优化器压成近似确定性后再也不学。
根因之一是 REINFORCE 时代的 lr=3e-3 对 PPO 太大。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train.trainer import Trainer  # noqa: E402


class FakeEnv:
    """最小 env：每步固定奖励，用于驱动 PPO 更新路径。"""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ActionSpace:
        n = NUM_ACTIONS

    def __init__(self, length: int = 40, reward: float = 1.0):
        self.observation_space = self._Space((OBS_DIM,))
        self.action_space = self._ActionSpace()
        self.length = length
        self.reward = reward
        self.bots = [type("B", (), {"username": "Bot_A"})(),
                     type("B", (), {"username": "Bot_B"})()]
        self.raw_observations = []
        self.opponent_mode = "scripted"
        self.shaping = None
        self._t = 0

    def reset(self, **_kw):
        self._t = 0
        return np.linspace(-1.0, 1.0, OBS_DIM).astype(np.float32), {"step": 0}

    def step(self, _action):
        self._t += 1
        done = self._t >= self.length
        return (np.linspace(-1.0, 1.0, OBS_DIM).astype(np.float32), self.reward,
                done, False, {"winner": None, "health": {}})

    def last_events(self):
        return {}

    def last_dispatch(self, _username):
        return []

    def close(self):
        return None


def make_policy(seed: int = 1, **kwargs) -> Policy:
    return Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=16, seed=seed, **kwargs)


def make_trainer(policy: Policy, env: FakeEnv | None = None, **kwargs) -> Trainer:
    kwargs.setdefault("log", lambda _m: None)
    # Trainer 的默认算法仍是 reinforce（为了旧调用方兼容），这里显式走 PPO。
    kwargs.setdefault("algo", "ppo")
    kwargs.setdefault("epochs", 1)
    # 默认关掉早停：这些用例测的是「每个 epoch 都更新」，KL 早停会让计数变少。
    kwargs.setdefault("early_stop_kl", 0.0)
    return Trainer(env or FakeEnv(), policy, seed=1, **kwargs)


class DefaultLrTest(unittest.TestCase):
    def test_policy_default_lr_is_3e4(self):
        self.assertEqual(Policy(obs_dim=4, n_actions=3).lr, 3e-4)


class LrDecayTest(unittest.TestCase):
    def test_decays_to_ten_percent(self):
        policy = make_policy(lr=1e-3)
        trainer = make_trainer(policy, epochs=1, minibatch=0, entropy_floor=0.0)
        trainer.train(2)  # 计划 2 个 update
        self.assertAlmostEqual(trainer._lr_updates, 2)
        self.assertAlmostEqual(policy.lr, 1e-3 * 0.1, places=9)

    def test_decay_can_be_disabled(self):
        policy = make_policy(lr=1e-3)
        trainer = make_trainer(policy, epochs=1, minibatch=0, entropy_floor=0.0, lr_decay=False)
        trainer.train(2)
        self.assertAlmostEqual(policy.lr, 1e-3, places=12)


class EntropyFloorTest(unittest.TestCase):
    def test_coefficient_doubles_then_caps(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.02, entropy_floor=0.15,
                               entropy_coef_max=0.1)
        trainer._adapt_entropy_coef(0.01)
        self.assertAlmostEqual(trainer._entropy_coef, 0.04)
        trainer._adapt_entropy_coef(0.01)
        self.assertAlmostEqual(trainer._entropy_coef, 0.08)
        trainer._adapt_entropy_coef(0.01)
        self.assertAlmostEqual(trainer._entropy_coef, 0.1, msg="应封顶到 entropy_coef_max")

    def test_coefficient_decays_back(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.02, entropy_floor=0.15)
        trainer._entropy_coef = 0.08
        trainer._adapt_entropy_coef(0.9)  # 远高于 2*floor
        self.assertLess(trainer._entropy_coef, 0.08)
        self.assertGreaterEqual(trainer._entropy_coef, 0.02)

    def test_disabled_when_floor_zero(self):
        # P2.2 起默认还有目标熵控制器（target=0.30），「退回 P2.1 行为」需要两者都关。
        trainer = make_trainer(make_policy(), entropy_coef=0.02, entropy_floor=0.0,
                               target_entropy=0.0)
        trainer._adapt_entropy_coef(0.0)
        self.assertAlmostEqual(trainer._entropy_coef, 0.02)


class MinibatchTest(unittest.TestCase):
    def test_zero_minibatch_updates_once_per_epoch(self):
        policy = make_policy()
        trainer = make_trainer(policy, epochs=2, minibatch=0, entropy_floor=0.0)
        trainer.train(1)
        self.assertEqual(policy.updates, 2, "minibatch=0 时每个 epoch 一次全量更新")

    def test_small_minibatch_updates_more_often(self):
        policy = make_policy()
        trainer = make_trainer(policy, epochs=2, minibatch=8, entropy_floor=0.0)
        trainer.train(1)
        self.assertGreater(policy.updates, 2, "minibatch 应把一个 epoch 拆成多次更新")

    def test_diagnostics_recorded(self):
        policy = make_policy()
        trainer = make_trainer(policy, epochs=1, minibatch=16)
        record = trainer.train(1)[0]
        for key in ("entropy", "entropyCoef", "lr", "gradNorm", "maxProb", "earlyStop"):
            self.assertIn(key, record)
        self.assertTrue(0.0 < record["maxProb"] <= 1.0)
        self.assertGreater(record["gradNorm"], 0.0)


class GradClipTest(unittest.TestCase):
    def test_clip_reduces_step_size(self):
        base = make_policy(lr=1e-2)
        clipped = make_policy(lr=1e-2)
        grads = {key: np.full_like(value, 50.0) for key, value in base.params.items()}

        base.apply_gradients({k: v.copy() for k, v in grads.items()}, max_grad_norm=0.0)
        norm = clipped.apply_gradients({k: v.copy() for k, v in grads.items()}, max_grad_norm=0.5)

        self.assertGreater(norm, 0.5)
        self.assertAlmostEqual(clipped.last_grad_norm, norm)
        delta_base = np.abs(base.params["W2"] - make_policy(lr=1e-2).params["W2"]).sum()
        delta_clipped = np.abs(clipped.params["W2"] - make_policy(lr=1e-2).params["W2"]).sum()
        self.assertLess(delta_clipped, delta_base)

    def test_zero_disables_clipping(self):
        policy = make_policy(lr=1e-2)
        grads = {key: np.full_like(value, 10.0) for key, value in policy.params.items()}
        norm = policy.apply_gradients(grads, max_grad_norm=0.0)
        self.assertGreater(norm, 0.0)


class EarlyStopTest(unittest.TestCase):
    def test_stops_after_first_epoch_when_kl_exceeds(self):
        policy = make_policy(lr=5e-2)
        trainer = make_trainer(policy, epochs=4, minibatch=0, entropy_floor=0.0,
                               early_stop_kl=1e-12)
        record = trainer.train(1)[0]
        self.assertTrue(record["earlyStop"])
        self.assertEqual(policy.updates, 1, "早停后不应继续跑剩余 epoch")

    def test_no_early_stop_by_default(self):
        policy = make_policy()
        trainer = make_trainer(policy, epochs=2, minibatch=0, entropy_floor=0.0)
        record = trainer.train(1)[0]
        self.assertFalse(record["earlyStop"])


class VictoryOverrideTest(unittest.TestCase):
    """P2.1 块 B：--victory 覆盖 ShapingWeights.victory，且优先于预设。"""

    def test_cli_default_is_none(self):
        from gladiator_train.train import build_parser

        self.assertIsNone(build_parser().parse_args([]).victory)
        self.assertEqual(build_parser().parse_args(["--victory", "35"]).victory, 35.0)

    def test_override_beats_preset(self):
        from gladiator_env.reward_presets import preset_for_env
        from gladiator_env.shaping import ShapingWeights

        preset = {"format": "gladiator.reward-preset/v1", "name": "ab",
                  "base": {"victory": 30.0}, "terms": []}
        kwargs, _terms = preset_for_env(preset)
        weights = ShapingWeights(**kwargs)
        self.assertEqual(weights.victory, 30.0, "预设里的 base.victory 应生效")

        # train.py 的做法：env 建好后直接覆盖
        weights.victory = 50.0
        self.assertEqual(weights.victory, 50.0, "--victory 应优先于预设")


if __name__ == "__main__":
    unittest.main()
