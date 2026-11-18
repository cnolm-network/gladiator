"""P2.2：控制器（单调 lr / 目标熵 / 价值裁剪）+ BC 与 PPO 贡献分离 + 诊断。

背景（run 20261118-0123-train-Bot_V50A-Bot_V50B，victory=50，12 局）：
- lr 每局开头回弹（0.00003 → 0.00023），说明衰减在每次 ``train()`` 里重新起算；
- ``entropyCoef`` 从头到尾钉在 0.1（旧上限），自适应下限实际失效；
- 每局开场都跑 4 局 BC 采集，但报告里没有任何字段能把「BC 的底子」与「PPO 学到的」分开。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train.trainer import (  # noqa: E402
    ENTROPY_COEF_MIN,
    Trainer,
    explained_variance,
)


class FakeEnv:
    """最小 env：固定奖励、固定长度，用于驱动 PPO 更新与 lr/熵路径。"""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ActionSpace:
        n = NUM_ACTIONS

    def __init__(self, length: int = 12, reward: float = 1.0):
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
        obs = np.linspace(-1.0, 1.0, OBS_DIM).astype(np.float32)
        obs[0] = self._t * 0.1 - 0.5
        return obs, self.reward, done, False, {"winner": None, "health": {}}

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
    kwargs.setdefault("algo", "ppo")
    kwargs.setdefault("epochs", 1)
    kwargs.setdefault("early_stop_kl", 0.0)
    return Trainer(env or FakeEnv(), policy, seed=1, **kwargs)


# --------------------------------------------------------------------- lr
class MonotonicLrTest(unittest.TestCase):
    """块 A1：lr 单调衰减、跨局不回弹。"""

    def _lrs(self, trainer: Trainer, episodes: int) -> list[float]:
        return [record["lr"] for record in trainer.train(episodes)]

    def test_strictly_decreasing_updates(self):
        policy = make_policy(lr=1e-3)
        trainer = make_trainer(policy, epochs=1, minibatch=0, entropy_floor=0.0,
                               target_entropy=0.0)
        trainer.plan_lr_schedule(3, 12)
        lrs = self._lrs(trainer, 3)
        self.assertEqual(len(lrs), 3)
        for earlier, later in zip(lrs, lrs[1:]):
            self.assertLess(later, earlier, f"lr 必须严格递减：{lrs}")
        self.assertAlmostEqual(lrs[-1], 1e-3 * 0.1, places=9)

    def test_no_rebound_across_train_calls(self):
        """分阶段训练（多次 train()）不能让 lr 回到高位（旧 bug 的锯齿）。"""
        policy = make_policy(lr=1e-3)
        trainer = make_trainer(policy, epochs=1, minibatch=0, entropy_floor=0.0,
                               target_entropy=0.0)
        trainer.plan_lr_schedule(4, 12)
        lrs = self._lrs(trainer, 2) + self._lrs(trainer, 2)
        for earlier, later in zip(lrs, lrs[1:]):
            self.assertLessEqual(later, earlier, f"lr 不能回弹：{lrs}")
        self.assertEqual(trainer._lr_updates, 4, "计数器跨调用累加，不重置")

    def test_schedule_only_grows(self):
        trainer = make_trainer(make_policy(), epochs=2, minibatch=256)
        total = trainer.plan_lr_schedule(10, 150)
        self.assertEqual(total, 10 * 2 * 1, "150 步 / minibatch 256 → 每局 1 批 × 2 epoch")
        self.assertEqual(trainer.plan_lr_schedule(2, 10), total, "计划只放大，不回缩")
        self.assertEqual(trainer.plan_lr_schedule(12, 600), 12 * 2 * 3)

    def test_decay_disabled_keeps_lr_flat(self):
        policy = make_policy(lr=1e-3)
        trainer = make_trainer(policy, epochs=1, minibatch=0, lr_decay=False)
        trainer.train(3)
        self.assertAlmostEqual(policy.lr, 1e-3, places=12)


# --------------------------------------------------------------- 目标熵
class TargetEntropyTest(unittest.TestCase):
    """块 A2：``coef ← clip(coef·exp(0.5·(target − H)), 0.005, 0.5)``。"""

    def test_low_entropy_raises_coef(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.02, entropy_floor=0.0,
                               target_entropy=0.30)
        trainer._adapt_entropy_coef(0.05)
        self.assertGreater(trainer._entropy_coef, 0.02)
        self.assertAlmostEqual(trainer._entropy_coef, 0.02 * np.exp(0.5 * 0.25), places=9)

    def test_high_entropy_lowers_coef(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.2, entropy_floor=0.0,
                               target_entropy=0.30)
        trainer._adapt_entropy_coef(0.90)
        self.assertLess(trainer._entropy_coef, 0.2)
        self.assertAlmostEqual(trainer._entropy_coef, 0.2 * np.exp(-0.30), places=9)

    def test_clamped_between_bounds(self):
        high = make_trainer(make_policy(), entropy_coef=0.5, entropy_floor=0.0,
                            target_entropy=0.30, entropy_coef_max=0.5)
        high._adapt_entropy_coef(0.0)
        self.assertAlmostEqual(high._entropy_coef, 0.5, msg="应被 entropy_coef_max 夹住")

        low = make_trainer(make_policy(), entropy_coef=0.006, entropy_floor=0.0,
                           target_entropy=0.30)
        low._adapt_entropy_coef(5.0)
        self.assertAlmostEqual(low._entropy_coef, ENTROPY_COEF_MIN, places=12)

    def test_default_coef_max_is_two(self):
        """P2.2 抬到 0.5，P2.3 再抬到 2.0：实测 0.1/0.5 顶死都拉不动饱和策略。"""
        trainer = make_trainer(make_policy())
        self.assertAlmostEqual(trainer.entropy_coef_max, 2.0)

    def test_floor_doubling_still_wins_when_higher(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.04, entropy_floor=0.15,
                               entropy_coef_max=0.5, target_entropy=0.30)
        trainer._adapt_entropy_coef(0.01)
        self.assertAlmostEqual(trainer._entropy_coef, 0.08,
                               msg="翻倍结果更大时取翻倍（下限保护）")

    def test_both_off_is_p21_behavior(self):
        trainer = make_trainer(make_policy(), entropy_coef=0.02, entropy_floor=0.0,
                               target_entropy=0.0)
        trainer._adapt_entropy_coef(0.01)
        self.assertAlmostEqual(trainer._entropy_coef, 0.02)


# ------------------------------------------------------------- 价值裁剪
class ValueClipTest(unittest.TestCase):
    """块 A3：``v_clipped = old_v + clip(v − old_v, ±c)``，loss 取两支最大。"""

    def _grads(self, policy, obs, actions, old_logp, old_values, returns, value_clip):
        return policy.ppo_gradients(
            obs, actions, old_logp, [0.0] * len(obs), returns,
            clip=0.2, value_coef=0.5, entropy_coef=0.0,
            old_values=old_values, value_clip=value_clip,
        )

    def _setup(self, policy):
        rng = np.random.default_rng(0)
        obs = [rng.normal(size=OBS_DIM) for _ in range(6)]
        actions = [int(rng.integers(0, NUM_ACTIONS)) for _ in range(6)]
        old_logp, _entropy, values = policy.evaluate_actions(obs, actions)
        return obs, actions, old_logp, values

    def test_out_of_range_is_clamped(self):
        policy = make_policy(lr=1e-3)
        obs, actions, old_logp, values = self._setup(policy)
        # 锚点比当前 V 高 10：v − old_v = −10，远超 ±0.2 → 裁剪支更差 → 价值头梯度为 0。
        old_values = values + 10.0
        returns = values + 1.0
        grads = self._grads(policy, obs, actions, old_logp, old_values, returns, value_clip=0.2)
        self.assertAlmostEqual(float(np.abs(grads["Wv"]).sum()), 0.0, places=12)
        self.assertAlmostEqual(float(np.abs(grads["bv"]).sum()), 0.0, places=12)

    def test_zero_disables_clipping(self):
        policy = make_policy(lr=1e-3)
        obs, actions, old_logp, values = self._setup(policy)
        old_values = values + 10.0
        returns = values + 1.0
        grads = self._grads(policy, obs, actions, old_logp, old_values, returns, value_clip=0.0)
        self.assertGreater(float(np.abs(grads["Wv"]).sum()), 0.0)

    def test_in_range_matches_unclipped(self):
        policy = make_policy(lr=1e-3)
        obs, actions, old_logp, values = self._setup(policy)
        clipped = self._grads(policy, obs, actions, old_logp, values, values + 0.05,
                              value_clip=0.2)
        plain = self._grads(policy, obs, actions, old_logp, values, values + 0.05,
                            value_clip=0.0)
        self.assertAlmostEqual(float(np.abs(clipped["Wv"] - plain["Wv"]).sum()), 0.0, places=12)

    def test_trainer_passes_value_clip_through(self):
        policy = make_policy()
        trainer = make_trainer(policy, minibatch=0, value_clip=0.2, entropy_floor=0.0,
                               target_entropy=0.0)
        trainer.train(1)  # 不崩即可：内部按批传 old_values
        self.assertEqual(trainer.value_clip, 0.2)


# ------------------------------------------------------------- 解释方差
class ExplainedVarianceTest(unittest.TestCase):
    def test_perfect_prediction_is_one(self):
        returns = np.array([1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(explained_variance(returns, returns), 1.0, places=9)

    def test_constant_prediction_is_zero(self):
        returns = np.array([1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(explained_variance(np.full(4, returns.mean()), returns),
                               0.0, places=9)
        # 常数预测但偏了：仍是 0（方差解释率只看残差方差）
        self.assertAlmostEqual(explained_variance(np.zeros(4), np.ones(4)), 0.0, places=9)

    def test_worse_than_mean_is_negative(self):
        returns = np.array([0.0, 0.0, 10.0, 10.0])
        self.assertLess(explained_variance(np.array([0.0, 10.0, 0.0, 10.0]), returns), 0.0)

    def test_empty_and_flat_are_safe(self):
        self.assertEqual(explained_variance([], []), 0.0)
        self.assertEqual(explained_variance([1.0, 1.0], [1.0, 1.0]), 0.0)

    def test_recorded_in_history(self):
        trainer = make_trainer(make_policy(), epochs=1, minibatch=0)
        record = trainer.train(1)[0]
        self.assertIn("explainedVariance", record)
        self.assertTrue(np.isfinite(record["explainedVariance"]))


# --------------------------------------------------- CLI / 报告字段（块 B）
class CliFlagsTest(unittest.TestCase):
    def test_new_flags_defaults(self):
        from gladiator_train.train import build_parser

        args = build_parser().parse_args([])
        self.assertAlmostEqual(args.target_entropy, 0.30)
        self.assertAlmostEqual(args.value_clip, 0.2)
        self.assertAlmostEqual(args.entropy_coef_max, 2.0)
        self.assertFalse(args.eval_before_train)

    def test_new_flags_parse(self):
        from gladiator_train.train import build_parser

        args = build_parser().parse_args(
            ["--bc-episodes", "0", "--eval-before-train",
             "--target-entropy", "0.25", "--value-clip", "0"])
        self.assertEqual(args.bc_episodes, 0)
        self.assertTrue(args.eval_before_train)
        self.assertAlmostEqual(args.target_entropy, 0.25)
        self.assertEqual(args.value_clip, 0.0)


class TrainMainReportTest(unittest.TestCase):
    """跑一遍完整 main()（env 用假实现），检查 BC / 训练前评估的字段契约。"""

    def _run(self, extra: list[str]) -> dict:
        import gladiator_env
        from gladiator_train import train as train_module

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = str(Path(tmp) / "run")
            argv = [
                "--users", "Bot_A", "Bot_B", "--ports", "8770", "8771",
                "--run-dir", run_dir,
                "--episodes", "1", "--eval-epochs", "1",
                "--max-steps", "8", "--ppo-epochs", "1",
                "--bc-max-steps", "6",
                *extra,
            ]
            with unittest.mock.patch.object(gladiator_env, "GladiatorEnv",
                                            lambda *a, **kw: FakeEnv(length=6)):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = train_module.main(argv)
            self.assertEqual(code, 0)
            with open(Path(run_dir) / "report.json", encoding="utf-8") as fh:
                return json.load(fh)

    def test_bc_episodes_zero_skips_and_is_reported(self):
        report = self._run(["--bc-episodes", "0"])
        self.assertEqual(report["config"]["bcEpisodes"], 0)
        self.assertEqual(report["config"]["bc"]["episodes"], 0)
        self.assertTrue(report["bc"]["skipped"], "bc-episodes=0 应标记 skipped")
        self.assertIsNone(report["bc"]["agreementBefore"])

    def test_eval_before_train_fields(self):
        report = self._run(["--bc-episodes", "0", "--eval-before-train"])
        self.assertTrue(report["config"]["evalBeforeTrain"])
        self.assertIsNotNone(report["evaluationBefore"])
        self.assertIsNotNone(report["evaluationGreedyBefore"])
        self.assertGreater(len(report["evaluationBefore"]), 0)
        # 训练后的评估仍然在
        self.assertGreater(len(report["evaluation"]), 0)

    def test_p22_config_fields(self):
        report = self._run(["--bc-episodes", "0"])
        config = report["config"]
        self.assertAlmostEqual(config["targetEntropy"], 0.30)
        self.assertAlmostEqual(config["valueClip"], 0.2)
        self.assertAlmostEqual(config["entropyCoefMax"], 2.0)
        self.assertIsNone(report["evaluationBefore"], "未开 --eval-before-train 时为空")
        # 每局都写了解释方差
        self.assertTrue(all("explainedVariance" in item for item in report["history"]))


if __name__ == "__main__":
    unittest.main()
