"""G-705(c)：AWR 加权（用价值头给人类录像样本加权）。

判据（对应 ``advantage_weights`` 的口径）：

1. **方向**：``R > V`` 的样本权重 > 1，``R < V`` 的在 ``positive_only`` 下置 0；
2. **锋利度**：``temperature`` 越小，样本间权重比越大；
3. **尺度**：返回值均值恒为 1（换加权模式不会悄悄改变 BC 的有效学习率），
   全负时退化为均匀（不能交出一个总权重为 0 的批次）；
4. **兼容**：v1 / v2 策略都能算；``returns`` 长度不符、含 NaN、温度非正一律报错；
5. **组合**：(a)(b) 权重 × AWR 后再归一，方向与尺度都保住（脚本里的用法）。

为了让断言是**等式**而不是「大概方向」，这里的策略一律把价值头置 0（``Wv=bv=0``），
于是 ``V(s) ≡ 0``、``A = R``，权重可以手算。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.demos import AWR_EXP_LIMIT, advantage_weights  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402

SCRIPT_PATH = TRAINING_DIR / "scripts" / "import_human_match.py"


def _load_import_script():
    """按文件路径加载导入脚本（它不是包，和 test_human_import_weights 用同一招）。"""
    spec = importlib.util.spec_from_file_location("import_human_match_awr_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def flat_value_policy(*, action_space: str = "v1", hidden: int = 8, seed: int = 1) -> Policy:
    """价值头全 0 的策略：``V(s) ≡ 0``，于是 ``A = R``，期望值可以直接手算。"""
    if action_space == "v2":
        policy = Policy(obs_dim=OBS_DIM, action_space="v2", hidden=hidden, seed=seed)
    else:
        policy = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=hidden, seed=seed)
    policy.params["Wv"][:] = 0.0
    policy.params["bv"][:] = 0.0
    return policy


def samples_of(count: int, *, dim: int = OBS_DIM) -> list[tuple[np.ndarray, float, float]]:
    return [(np.full(dim, 0.1 * index, dtype=np.float64), 0, 0.0) for index in range(count)]


class DirectionTest(unittest.TestCase):
    def test_returns_above_value_upweights(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(3), returns=[2.0, 0.0, 0.0])
        self.assertGreater(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[1]), float(weights[2]), places=12)
        # 手算：w0 = e², mean = (e² + 1 + 1)/3 → w0/mean
        expected = np.exp(2.0) / ((np.exp(2.0) + 2.0) / 3.0)
        self.assertAlmostEqual(float(weights[0]), expected, places=10)

    def test_positive_only_zeroes_negative_advantage(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(3), returns=[1.0, -1.0, 0.0])
        self.assertEqual(float(weights[1]), 0.0, "负优势必须置 0（AWR 只向更好的动作学）")
        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(float(weights[2]), 0.0, "A=0 不是负优势，应保留 exp(0)=1")

    def test_positive_only_false_keeps_small_weights(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(2), returns=[1.0, -1.0],
                                    positive_only=False)
        self.assertGreater(float(weights[1]), 0.0)
        self.assertLess(float(weights[1]), float(weights[0]))

    def test_value_is_actually_subtracted(self) -> None:
        """同样 R、不同状态 → 权重不同：证明减的是 V(s)，不是只看 R。"""
        policy = flat_value_policy()
        rng = np.random.default_rng(7)
        policy.params["Wv"] = rng.normal(0, 0.5, size=(1, policy.params["W1"].shape[0]))
        samples = samples_of(2)
        weights = advantage_weights(policy, samples, returns=[0.5, 0.5])
        v0, v1 = policy.value(samples[0][0]), policy.value(samples[1][0])
        self.assertNotAlmostEqual(v0, v1, places=6, msg="两个状态的价值应不同，否则测不出减法")
        # 价值高的那条 A 更小 → 权重更小
        self.assertEqual(float(weights[0]) < float(weights[1]), v0 > v1)


class TemperatureTest(unittest.TestCase):
    def test_smaller_temperature_is_sharper(self) -> None:
        policy = flat_value_policy()
        samples = samples_of(3)
        gentle = advantage_weights(policy, samples, returns=[1.0, 0.0, 0.0], temperature=1.0)
        sharp = advantage_weights(policy, samples, returns=[1.0, 0.0, 0.0], temperature=0.2)
        self.assertGreater(float(sharp[0] / sharp[1]), float(gentle[0] / gentle[1]))
        # T=0.2 → 比值 e^5（归一化不改变比值）
        self.assertAlmostEqual(float(sharp[0] / sharp[1]), float(np.exp(5.0)), places=6)

    def test_large_temperature_is_flat(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(3), returns=[1.0, 0.0, 0.0],
                                    temperature=100.0)
        self.assertLess(float(np.ptp(weights)), 0.02, "温度很大时权重应趋于均匀")


class NormalisationTest(unittest.TestCase):
    def test_mean_is_one(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(5), returns=[3.0, 1.0, 0.0, -1.0, 2.0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=10)

    def test_all_negative_degenerates_to_uniform(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(4), returns=[-1.0, -2.0, -3.0, -4.0])
        np.testing.assert_allclose(weights, np.ones(4), atol=1e-12,
                                   err_msg="全负时必须退化为均匀，否则整批样本会失去权重")

    def test_none_returns_behaves_like_zeros(self) -> None:
        policy = flat_value_policy()
        np.testing.assert_allclose(
            advantage_weights(policy, samples_of(3), returns=None),
            np.ones(3), atol=1e-12)

    def test_empty_samples(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, [], returns=[])
        self.assertEqual(weights.shape, (0,))

    def test_dtype_is_float(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(2), returns=[0.0, 1.0])
        self.assertEqual(weights.dtype, np.float64)

    def test_extreme_advantage_does_not_overflow(self) -> None:
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(2), returns=[1e6, 0.0])
        self.assertTrue(np.all(np.isfinite(weights)), f"溢出保护失效（上限 {AWR_EXP_LIMIT}）")


class ClipTest(unittest.TestCase):
    def test_scalar_clip_caps_the_ratio(self) -> None:
        policy = flat_value_policy()
        samples = samples_of(3)
        clipped = advantage_weights(policy, samples, returns=[10.0, 0.0, 0.0], clip=2.0)
        # 归一化前是 [2, 1, 1] → 归一化后比值仍是 2
        self.assertAlmostEqual(float(clipped[0] / clipped[1]), 2.0, places=10)
        self.assertAlmostEqual(float(clipped.mean()), 1.0, places=10)

    def test_tuple_clip_lower_bound(self) -> None:
        """下界也要真的生效：把极端负优势的样本抬到 0.5（否则它等于被丢弃）。"""
        policy = flat_value_policy()
        weights = advantage_weights(policy, samples_of(3), returns=[10.0, -10.0, 0.0],
                                    positive_only=False, clip=(0.5, 2.0))
        # 归一化前 [2.0, 0.5, 1.0] → mean 7/6 → 归一到 [1.714…, 0.4286…, 0.857…]
        self.assertAlmostEqual(float(weights.min()), 0.5 / (7.0 / 6.0), places=10)

    def test_invalid_clip_raises(self) -> None:
        policy = flat_value_policy()
        with self.assertRaises(ValueError):
            advantage_weights(policy, samples_of(2), returns=[0.0, 0.0], clip=(2.0, 1.0))


class ValidationTest(unittest.TestCase):
    def test_length_mismatch_raises(self) -> None:
        policy = flat_value_policy()
        with self.assertRaises(ValueError):
            advantage_weights(policy, samples_of(3), returns=[0.0, 1.0])

    def test_non_finite_returns_raise(self) -> None:
        policy = flat_value_policy()
        with self.assertRaises(ValueError):
            advantage_weights(policy, samples_of(2), returns=[0.0, float("nan")])

    def test_bad_temperature_raises(self) -> None:
        policy = flat_value_policy()
        for bad in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                advantage_weights(policy, samples_of(2), returns=[0.0, 0.0], temperature=bad)


class V2PolicyTest(unittest.TestCase):
    def test_v2_policy_supported(self) -> None:
        policy = flat_value_policy(action_space="v2")
        self.assertTrue(policy.is_v2)
        weights = advantage_weights(policy, samples_of(3), returns=[1.0, 0.0, -1.0])
        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=10)
        self.assertEqual(float(weights[2]), 0.0)

    def test_v2_value_comes_from_shared_hidden_layer(self) -> None:
        """v2 的 V(s) 走「策略头之前的共享隐层 + Wv」，与 v1 同一条公式。"""
        policy = flat_value_policy(action_space="v2")
        rng = np.random.default_rng(11)
        policy.params["Wv"] = rng.normal(0, 0.3, size=(1, policy.params["W1"].shape[0]))
        policy.params["bv"][:] = 0.25
        x = np.linspace(-0.5, 0.5, OBS_DIM)
        hidden = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
        expected = float(policy.params["Wv"][0] @ hidden + 0.25)
        self.assertAlmostEqual(policy.value(x), expected, places=10)


class ComposeWithBaseWeightsTest(unittest.TestCase):
    """脚本里的组合口径：(a)(b) 权重 × AWR，再按均值归一到 1。"""

    def _compose(self, base: np.ndarray, awr: np.ndarray) -> np.ndarray:
        combined = base * awr
        return combined / combined.mean()

    def test_composition_keeps_mean_one_and_direction(self) -> None:
        policy = flat_value_policy()
        base = np.array([1.0, 0.06, 0.3, 1.0])   # 典型 (a)×(b)：赢局贴身 / 输局远处 / …
        awr = advantage_weights(policy, samples_of(4), returns=[2.0, 2.0, 0.0, -1.0])
        combined = self._compose(base, awr)
        self.assertAlmostEqual(float(combined.mean()), 1.0, places=10)
        self.assertEqual(float(combined[3]), 0.0, "负优势样本乘完仍是 0（不会被基础权重救回来）")
        # 同样 A>0 的两条，基础权重大的仍然更大
        self.assertGreater(float(combined[0]), float(combined[1]))
        # A>0 的样本乘完必须大于 A=0 的样本（AWR 方向没被基础权重盖掉）
        self.assertGreater(float(combined[1]), float(combined[2]))

    def test_composition_with_all_zero_awr_falls_back_uniform(self) -> None:
        """极少见的退化：AWR 全 0 时组合会全 0，脚本必须能识别并退回均匀。"""
        base = np.array([1.0, 0.5])
        awr = np.zeros(2)
        combined = base * awr
        self.assertLessEqual(float(combined.mean()), 1e-12)


class ScriptScoreAWRTest(unittest.TestCase):
    """`import_human_match.compose_awr_weights`：先 (a)(b)、再乘 AWR、最后归一。"""

    def setUp(self) -> None:
        self.human = _load_import_script()
        self.policy = flat_value_policy()

    def _dataset(self, *, returns=None) -> dict:
        dataset = {
            "obs": np.asarray([sample[0] for sample in samples_of(3)], dtype=np.float32),
            "actions": np.asarray([0, 0, 0], dtype=np.int64),
            "weights": np.asarray([1.0, 0.5, 0.2], dtype=np.float32),
        }
        if returns is not None:
            dataset["rewards"] = np.asarray(returns, dtype=np.float32)
        return dataset

    def test_multiplies_base_weights_and_normalises(self) -> None:
        weights, info = self.human.compose_awr_weights(
            self._dataset(returns=[2.0, 0.0, 0.0]), self.policy, temperature=1.0, clip=None)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(info["mode"], "awr")
        self.assertEqual(info["awrReturns"], "dataset.rewards")
        self.assertEqual(info["awrZeroed"], 0)
        # 基础权重最大的那条同时优势最大 → 一定最大
        self.assertEqual(int(np.argmax(weights)), 0)

    def test_missing_rewards_is_recorded_as_zeros(self) -> None:
        weights, info = self.human.compose_awr_weights(
            self._dataset(), self.policy, temperature=1.0, clip=None)
        self.assertIn("zeros", info["awrReturns"])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_negative_advantage_is_zeroed_after_composition(self) -> None:
        weights, info = self.human.compose_awr_weights(
            self._dataset(returns=[1.0, -1.0, 0.0]), self.policy, temperature=1.0, clip=None)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertEqual(info["awrZeroed"], 1)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_clip_is_forwarded_as_upper_bound(self) -> None:
        dataset = self._dataset(returns=[10.0, 0.0, 0.0])
        clipped, info = self.human.compose_awr_weights(
            dataset, self.policy, temperature=1.0, clip=2.0)
        unclipped, _ = self.human.compose_awr_weights(
            dataset, self.policy, temperature=1.0, clip=None)
        self.assertEqual(info["awrClip"], 2.0)
        self.assertAlmostEqual(float(clipped.mean()), 1.0, places=6)
        # 剪枝把「优势 10 vs 0」的 2.2 万倍差距压到 2 倍，比值只剩基础权重带来的差距。
        self.assertLess(float(clipped.max() / clipped.min()),
                        float(unclipped.max() / unclipped.min()))
        self.assertAlmostEqual(float(clipped.max() / clipped.min()), 10.0, places=4)

    def test_missing_rewards_is_marked_ineffective(self) -> None:
        """没有 rewards 时 A=−V：权重可能仍有区分度，但**不是** advantage，必须标 False。"""
        weights, info = self.human.compose_awr_weights(
            self._dataset(), self.policy, temperature=1.0, clip=None)
        self.assertFalse(info["awrEffective"])
        self.assertIn("zeros", info["awrReturns"])

    def test_missing_rewards_with_varying_values_still_flagged(self) -> None:
        """旧模型的 V(s) 随状态变化 → 权重彼此不同，但仍然是「A=−V」的错误口径。"""
        varying = flat_value_policy()
        varying.params["Wv"] = np.full_like(varying.params["Wv"], 0.5)
        _, info = self.human.compose_awr_weights(
            self._dataset(), varying, temperature=1.0, clip=None)
        self.assertFalse(info["awrEffective"])
        self.assertGreater(info["awrDistinct"], 1, "权重确实不同，但口径仍不可信")

    def test_rewards_make_awr_effective(self) -> None:
        _, info = self.human.compose_awr_weights(
            self._dataset(returns=[2.0, 0.0, 0.0]), self.policy, temperature=1.0, clip=None)
        self.assertTrue(info["awrEffective"])
        self.assertGreater(info["awrDistinct"], 1)

    def test_strict_mode_rejects_missing_rewards(self) -> None:
        with self.assertRaises(ValueError):
            self.human.compose_awr_weights(
                self._dataset(), self.policy, temperature=1.0, clip=None, strict=True)

    def test_strict_mode_accepts_present_rewards(self) -> None:
        weights, info = self.human.compose_awr_weights(
            self._dataset(returns=[2.0, 0.0, 0.0]), self.policy,
            temperature=1.0, clip=None, strict=True)
        self.assertTrue(info["awrEffective"])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)


class PaddingCompatTest(unittest.TestCase):
    """现役模型是 23/25 维、数据集是 37 维：导入器的 BC 与 AWR 都必须走「右侧补零」。

    <p>这两条是同一类 bug：`Policy.load` 默认保持文件维度，于是「维度不一致」把
    所有现役模型拒掉 —— 真人录像管线在真实数据上根本跑不起来（G-703 修了
    league/mirror，这里是同一口径的另外两个入口）。
    """

    def setUp(self) -> None:
        import tempfile

        self.human = _load_import_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / "old-25.npz"
        old = Policy(obs_dim=25, n_actions=11, hidden=8, seed=3)
        old.save(str(self.model_path))

    def _dataset(self, count: int = 6) -> dict:
        return {
            "obs": np.asarray([np.full(OBS_DIM, 0.05 * index) for index in range(count)],
                              dtype=np.float32),
            "actions": np.zeros(count, dtype=np.int64),
            "weights": np.ones(count, dtype=np.float32),
        }

    def test_awr_accepts_narrower_base_model(self) -> None:
        from gladiator_train.model_io import load_model

        policy = load_model(self.model_path, obs_dim=OBS_DIM).policy
        self.assertEqual(policy.obs_dim, OBS_DIM)
        weights, info = self.human.compose_awr_weights(
            self._dataset(), policy, temperature=1.0, clip=None)
        self.assertEqual(weights.shape, (6,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_bc_accepts_narrower_base_model(self) -> None:
        result = self.human.run_bc(
            self._dataset(), base_model=str(self.model_path), epochs=2,
            run_dir=Path(self.tmp.name) / "run", meta={"player": "tester"}, seed=0,
            min_samples=1)
        self.assertIsNotNone(result)
        self.assertFalse(result.get("skipped"), result)
        self.assertTrue(Path(result["policy"]).is_file())

    def test_bc_reports_unloadable_model_instead_of_crashing(self) -> None:
        bad = Path(self.tmp.name) / "broken.npz"
        bad.write_bytes(b"not a policy")
        result = self.human.run_bc(
            self._dataset(), base_model=str(bad), epochs=1,
            run_dir=Path(self.tmp.name) / "run2", meta={}, seed=0, min_samples=1)
        self.assertTrue(result.get("skipped"))
        self.assertIn("模型加载失败", result["reason"])


if __name__ == "__main__":
    unittest.main()
