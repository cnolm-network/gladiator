"""权重融合单测。

- 前半部分（G-625）：``multi_train._merge_policies`` 的 mean/weighted/none 与报告字段；
- 后半部分（merge v2）：REBASIN 对齐、TIES / Fisher 合并、门禁判定、蒸馏兜底。

全部用内存假权重 + 临时目录，不启动真实训练、不依赖真实 artifacts、不连服务器。
"""

from __future__ import annotations

import math
import os
import tempfile
import types
import unittest

import numpy as np

from gladiator_train.merge import (
    MERGE_METHODS,
    apply_permutation,
    distill,
    fisher_information,
    fisher_merge,
    gate_verdict,
    hidden_features,
    match_permutation,
    merge,
    params_l2,
    ties_merge,
)
from gladiator_train.multi_train import PairResult, _merge_policies
from gladiator_train.policy import Policy


def _write_pair(d: str, idx: int, *, w1_value: float,
                obs_dim: int = 2, n_actions: int = 2, hidden: int = 2) -> str:
    """造一个 ``policy-pair{idx}.npz``，所有参数用 ``w1_value`` 的标量派生值填充。"""
    policy = Policy(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    policy.params["W1"][:] = w1_value
    policy.params["b1"][:] = w1_value * 0.5
    policy.params["W2"][:] = w1_value * 0.5
    policy.params["b2"][:] = w1_value * 0.25
    path = os.path.join(d, f"policy-pair{idx}.npz")
    policy.save(path, extra={"pair": idx})
    return path


class MergePolicyTest(unittest.TestCase):
    def _args(self, merge: str, d: str) -> types.SimpleNamespace:
        a = types.SimpleNamespace()
        a.merge = merge
        a.save_dir = d
        return a

    def _result(self, pair: int, *, steps: int, error: str | None = None) -> PairResult:
        return PairResult(pair=pair, users=[f"Bot_{pair}A", f"Bot_{pair}B"],
                          steps=steps, error=error)

    # ---------------------------------------------------------------- 等权
    def test_mean_average(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)
            _write_pair(d, 2, w1_value=3.0)
            _write_pair(d, 3, w1_value=5.0)
            results = [self._result(0, steps=10), self._result(1, steps=30), self._result(2, steps=20)]
            info = _merge_policies(self._args("mean", d), results, d)
            self.assertTrue(info["available"])
            self.assertFalse(info["skipped"])
            merged = Policy.load(os.path.join(d, "policy.npz"))
            # (1 + 3 + 5) / 3 = 3.0
            self.assertAlmostEqual(float(merged.params["W1"][0, 0]), 3.0, places=9)

    # ---------------------------------------------------------------- 加权
    def test_weighted_by_steps(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)
            _write_pair(d, 2, w1_value=3.0)
            _write_pair(d, 3, w1_value=5.0)
            # 按步数加权：10*1 + 30*3 + 20*5 = 200，总步数 60 → 200/60
            results = [self._result(0, steps=10), self._result(1, steps=30), self._result(2, steps=20)]
            info = _merge_policies(self._args("weighted", d), results, d)
            merged = Policy.load(os.path.join(d, "policy.npz"))
            self.assertAlmostEqual(float(merged.params["W1"][0, 0]), 200.0 / 60.0, places=9)

    def test_weighted_falls_back_to_equal_when_steps_le_zero(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)
            _write_pair(d, 2, w1_value=5.0)
            # 步数为 0 / 负数 → 退化成等权：(1 + 5) / 2 = 3.0
            results = [self._result(0, steps=0), self._result(1, steps=-5)]
            info = _merge_policies(self._args("weighted", d), results, d)
            merged = Policy.load(os.path.join(d, "policy.npz"))
            self.assertAlmostEqual(float(merged.params["W1"][0, 0]), 3.0, places=9)

    # ---------------------------------------------------------------- none
    def test_merge_none_skips_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)
            results = [self._result(0, steps=10)]
            info = _merge_policies(self._args("none", d), results, d)
            self.assertFalse(info["available"])
            self.assertTrue(info["skipped"])
            self.assertIn("merge=none", info.get("error", ""))
            self.assertIsNone(info["mergedSha256"])
            self.assertEqual(info["pairs"], [])
            # 不应写出 policy.npz
            self.assertFalse(os.path.exists(os.path.join(d, "policy.npz")))

    # ---------------------------------------------------------------- 失败跳过
    def test_single_pair_failure_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)   # pair 0 成功
            # pair 1（value=9）失败：不写文件，error 置位 → 应被跳过
            _write_pair(d, 3, w1_value=5.0)   # pair 2 成功
            results = [
                self._result(0, steps=10),
                self._result(1, steps=30, error="boom"),
                self._result(2, steps=20),
            ]
            info = _merge_policies(self._args("mean", d), results, d)
            used = sorted(p["pair"] for p in info["pairs"])
            self.assertEqual(used, [1, 3], "应只融合成功的 pair 1 与 3，跳过失败的 pair 2")
            merged = Policy.load(os.path.join(d, "policy.npz"))
            # (1 + 5) / 2 = 3.0（被跳过的 9 不应影响结果）
            self.assertAlmostEqual(float(merged.params["W1"][0, 0]), 3.0, places=9)

    def test_all_pairs_failed_sets_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            results = [self._result(0, steps=10, error="a"), self._result(1, steps=20, error="b")]
            info = _merge_policies(self._args("mean", d), results, d)
            self.assertFalse(info["available"])
            self.assertTrue(info["skipped"])
            self.assertIsNone(info["mergedSha256"])

    # ---------------------------------------------------------------- 形状不一致
    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0, hidden=2)
            _write_pair(d, 2, w1_value=2.0, hidden=4)  # 不同 hidden → W1 形状不一致
            results = [self._result(0, steps=10), self._result(1, steps=20)]
            with self.assertRaises(ValueError):
                _merge_policies(self._args("mean", d), results, d)

    # ---------------------------------------------------------------- 报告字段
    def test_report_merge_fields(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=2.0)
            _write_pair(d, 2, w1_value=4.0)
            results = [self._result(0, steps=10), self._result(1, steps=20)]
            info = _merge_policies(self._args("mean", d), results, d)

            # 顶层字段齐全
            for key in ("method", "pairs", "mergedSha256", "skipped",
                        "available", "mergedPolicyPath"):
                self.assertIn(key, info, f"report.merge 缺少字段 {key}")

            # 取值类型 / 契约
            self.assertEqual(info["method"], "mean")
            self.assertTrue(info["available"])
            self.assertFalse(info["skipped"])
            self.assertIsInstance(info["mergedSha256"], str)
            self.assertEqual(len(info["mergedSha256"]), 64, "sha256 应为 64 位十六进制")
            self.assertEqual(info["mergedPolicyPath"], os.path.join(d, "policy.npz"))
            self.assertTrue(os.path.isfile(info["mergedPolicyPath"]))

            # pairs 内每条都记录 pair / steps / sha256
            self.assertEqual([p["pair"] for p in info["pairs"]], [1, 2])
            self.assertEqual([p["steps"] for p in info["pairs"]], [10, 20])
            for p in info["pairs"]:
                self.assertIn("sha256", p)
                self.assertEqual(len(p["sha256"]), 64)

    def test_report_merge_fields_for_none(self):
        with tempfile.TemporaryDirectory() as d:
            _write_pair(d, 1, w1_value=1.0)
            results = [self._result(0, steps=10)]
            info = _merge_policies(self._args("none", d), results, d)
            # none 也要有完整字段骨架（只是 skipped / 无产物）
            for key in ("method", "pairs", "mergedSha256", "skipped", "available"):
                self.assertIn(key, info)
            self.assertEqual(info["method"], "none")
            self.assertTrue(info["skipped"])
            self.assertFalse(info["available"])
            self.assertIsNone(info["mergedSha256"])


# ===========================================================================
# merge v2：对齐（REBASIN）→ 合并（TIES / Fisher）→ 门禁 → 蒸馏
# ===========================================================================

def _random_params(hidden: int = 8, obs_dim: int = 5, n_actions: int = 3,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.5, (hidden, obs_dim)),
        "b1": rng.normal(0.0, 0.2, hidden),
        "W2": rng.normal(0.0, 0.4, (n_actions, hidden)),
        "b2": rng.normal(0.0, 0.1, n_actions),
        "Wv": rng.normal(0.0, 0.4, (1, hidden)),
        "bv": rng.normal(0.0, 0.1, 1),
    }


def _scalar_params(value: float, n_actions: int = 2) -> dict:
    """单隐藏单元的极简参数，便于手算合并结果。"""
    return {
        "W1": np.array([[value]], dtype=np.float64),
        "b1": np.zeros(1),
        "W2": np.zeros((n_actions, 1)),
        "b2": np.zeros(n_actions),
        "Wv": np.zeros((1, 1)),
        "bv": np.zeros(1),
    }


class HiddenFeaturesTest(unittest.TestCase):
    def test_matches_manual_tanh(self) -> None:
        params = _random_params(hidden=6, obs_dim=4, seed=1)
        obs = np.random.default_rng(2).normal(size=(9, 4))
        got = hidden_features(params, obs)
        want = np.tanh(obs @ params["W1"].T + params["b1"])
        self.assertEqual(got.shape, (9, 6))
        self.assertLess(float(np.max(np.abs(got - want))), 1e-12)

    def test_single_vector_and_dimension_guard(self) -> None:
        params = _random_params(hidden=3, obs_dim=4, seed=3)
        self.assertEqual(hidden_features(params, np.zeros(4)).shape, (1, 3))
        with self.assertRaises(ValueError):
            hidden_features(params, np.zeros(5))


class PermutationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params_a = _random_params(hidden=16, obs_dim=6, n_actions=4, seed=7)
        self.rng = np.random.default_rng(11)
        self.obs = self.rng.normal(size=(256, 6))

    def test_recovers_permutation(self) -> None:
        """B = 置换后的 A：match_permutation 应把它对齐回去（误差 ~1e-16）。"""
        truth = self.rng.permutation(16)
        params_b = apply_permutation(self.params_a, truth)
        feats_a = hidden_features(self.params_a, self.obs)
        feats_b = hidden_features(params_b, self.obs)
        perm = match_permutation(feats_a, feats_b)
        aligned = apply_permutation(params_b, perm)
        error = float(np.linalg.norm(aligned["W1"] - self.params_a["W1"]))
        self.assertLess(error, 1e-8)
        # perm[k] = A 的第 k 个单元在 B 中的位置 → 等于 truth 的逆。
        self.assertEqual(list(perm), list(np.argsort(truth)))
        # 价值头必须跟着换，否则 h 与 Wv 错位。
        self.assertLess(float(np.max(np.abs(aligned["Wv"] - self.params_a["Wv"]))), 1e-12)

    def test_rebasin_ties_reproduces_first_branch(self) -> None:
        params_b = apply_permutation(self.params_a, self.rng.permutation(16))
        merged, info = merge([self.params_a, params_b], method="rebasin-ties",
                             calibration_obs=self.obs, return_info=True)
        worst = max(float(np.max(np.abs(merged[k] - self.params_a[k]))) for k in self.params_a)
        self.assertLess(worst, 1e-9)
        self.assertTrue(info["aligned"])
        self.assertEqual(len(info["perms"]), 2)

    def test_apply_permutation_does_not_mutate_input(self) -> None:
        snapshot = {k: v.copy() for k, v in self.params_a.items()}
        out = apply_permutation(self.params_a, self.rng.permutation(16))
        for key, value in snapshot.items():
            np.testing.assert_array_equal(self.params_a[key], value)
        out["W1"][0, 0] += 100.0
        self.assertNotEqual(self.params_a["W1"][0, 0], out["W1"][0, 0])

    def test_permutation_validation(self) -> None:
        with self.assertRaises(ValueError):
            apply_permutation(self.params_a, [0, 0])       # 长度不足
        with self.assertRaises(ValueError):
            apply_permutation(self.params_a, [0] * 16)     # 重复
        with self.assertRaises(ValueError):
            match_permutation(hidden_features(self.params_a, self.obs),
                              hidden_features(_random_params(hidden=8, obs_dim=6), self.obs))

    def test_match_identity_is_identity(self) -> None:
        feats = hidden_features(self.params_a, self.obs)
        self.assertEqual(list(match_permutation(feats, feats)), list(range(16)))

    def test_two_calibration_samples_rejected(self) -> None:
        """2 条样本时标准化后每个单元只剩 ±1，匹配等于随机 → 必须报错而不是给出错排列。"""
        feats = hidden_features(self.params_a, self.obs[:2])
        with self.assertRaises(ValueError):
            match_permutation(feats, feats)


class TiesTest(unittest.TestCase):
    def test_trim_drops_minority_sign(self) -> None:
        merged = ties_merge([_scalar_params(1.0), _scalar_params(1.0), _scalar_params(-0.2)],
                            trim=0.2)
        self.assertAlmostEqual(float(merged["W1"][0, 0]), 1.0, places=12)

    def test_all_negative_uses_negative_mean(self) -> None:
        merged = ties_merge([_scalar_params(-1.0), _scalar_params(-1.0), _scalar_params(-3.0)],
                            trim=0.0)
        self.assertAlmostEqual(float(merged["W1"][0, 0]), -5.0 / 3.0, places=12)

    def test_trim_zero_is_weighted_mean(self) -> None:
        merged = ties_merge([_scalar_params(1.0), _scalar_params(3.0)],
                            weights=[1, 3], trim=0.0)
        self.assertAlmostEqual(float(merged["W1"][0, 0]), 2.5, places=12)

    def test_weight_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            ties_merge([_scalar_params(1.0), _scalar_params(2.0)], weights=[1, 1, 1])


class FisherTest(unittest.TestCase):
    def test_fisher_weighted_average(self) -> None:
        left = {"a": np.array([1.0, 2.0])}
        right = {"a": np.array([3.0, 4.0])}
        merged = fisher_merge([left, right],
                              [{"a": np.array([1.0, 1.0])}, {"a": np.array([3.0, 1.0])}])
        np.testing.assert_allclose(merged["a"], np.array([2.5, 3.0]), atol=1e-12)

    def test_fisher_all_zero_falls_back_to_mean(self) -> None:
        merged = fisher_merge([{"a": np.array([2.0])}, {"a": np.array([4.0])}],
                              [{"a": np.array([0.0])}, {"a": np.array([0.0])}])
        np.testing.assert_allclose(merged["a"], np.array([3.0]), atol=1e-12)

    def test_fisher_information_matches_numeric_gradient(self) -> None:
        params = _random_params(hidden=4, obs_dim=3, n_actions=3, seed=21)
        policy = Policy(obs_dim=3, n_actions=3, hidden=4, seed=0)
        policy.params = {k: v.copy() for k, v in params.items()}
        obs = np.array([0.3, -0.7, 0.2])
        action = 1
        fisher = fisher_information(policy, [(obs, action)], obs_dim=3, n_actions=3)

        def log_prob(table: dict) -> float:
            h = np.tanh(table["W1"] @ obs + table["b1"])
            logits = table["W2"] @ h + table["b2"]
            shifted = logits - np.max(logits)
            return float(shifted[action] - np.log(np.sum(np.exp(shifted))))

        delta = 1e-6
        plus = {k: v.copy() for k, v in params.items()}
        minus = {k: v.copy() for k, v in params.items()}
        plus["W1"][0, 0] += delta
        minus["W1"][0, 0] -= delta
        numeric = (log_prob(plus) - log_prob(minus)) / (2 * delta)
        self.assertAlmostEqual(math.sqrt(float(fisher["W1"][0, 0])), abs(numeric), places=5)
        # 价值头不参与 softmax 似然 → Fisher 全 0（fisher_merge 会退回等权）。
        self.assertEqual(float(np.max(np.abs(fisher["Wv"]))), 0.0)

    def test_fisher_information_requires_samples(self) -> None:
        policy = Policy(obs_dim=3, n_actions=3, hidden=3, seed=0)
        with self.assertRaises(ValueError):
            fisher_information(policy, [], obs_dim=3, n_actions=3)


class MergeEntryTest(unittest.TestCase):
    def test_mean_matches_manual_average(self) -> None:
        a = _random_params(hidden=4, obs_dim=3, seed=1)
        b = _random_params(hidden=4, obs_dim=3, seed=2)
        merged = merge([a, b], method="mean")
        for key in a:
            np.testing.assert_allclose(merged[key], (a[key] + b[key]) / 2.0, atol=1e-12)

    def test_mean_with_weights(self) -> None:
        merged = merge([_scalar_params(1.0), _scalar_params(3.0)], method="mean",
                       weights=[3, 1])
        self.assertAlmostEqual(float(merged["W1"][0, 0]), 1.5, places=12)

    def test_unknown_method_lists_choices(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            merge([_scalar_params(1.0)], method="magic")
        for name in MERGE_METHODS:
            self.assertIn(name, str(ctx.exception))

    def test_rebasin_requires_calibration(self) -> None:
        with self.assertRaises(ValueError):
            merge([_scalar_params(1.0), _scalar_params(1.0)], method="rebasin-ties")

    def test_rebasin_fisher_requires_fisher(self) -> None:
        with self.assertRaises(ValueError):
            merge([_scalar_params(1.0), _scalar_params(1.0)], method="rebasin-fisher",
                  calibration_obs=np.zeros((4, 1)))

    def test_mean_does_not_mutate_inputs(self) -> None:
        a = _random_params(hidden=3, obs_dim=2, seed=5)
        snapshot = {k: v.copy() for k, v in a.items()}
        merge([a, a], method="mean")
        for key, value in snapshot.items():
            np.testing.assert_array_equal(a[key], value)

    def test_params_l2(self) -> None:
        self.assertAlmostEqual(params_l2(_scalar_params(1.0), _scalar_params(3.0)), 2.0,
                               places=12)
        self.assertAlmostEqual(params_l2(_scalar_params(1.0), _scalar_params(1.0)), 0.0,
                               places=12)


class GateVerdictTest(unittest.TestCase):
    @staticmethod
    def _summary(rate: float, episodes: int = 8) -> dict:
        return {"overall": {"episodes": episodes, "winRate": rate}}

    def test_significant_better_passes(self) -> None:
        result = gate_verdict(self._summary(0.25), self._summary(0.75), 0.01)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "significant-better")

    def test_non_inferior_passes(self) -> None:
        result = gate_verdict(self._summary(0.5), self._summary(0.5), 0.62)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "non-inferior")

    def test_significant_worse_fails(self) -> None:
        result = gate_verdict(self._summary(0.75), self._summary(0.25), 0.01)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "significant-worse")

    def test_regression_beyond_tolerance_fails(self) -> None:
        # 8 局里差 2 局（0.75 → 0.5）超过 1/8 容差，且 p 不显著 → 按掉点处理。
        result = gate_verdict(self._summary(0.75), self._summary(0.5), 0.45)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "regression")
        self.assertAlmostEqual(result["tolerance"], 0.125, places=12)

    def test_within_tolerance_is_non_inferior(self) -> None:
        # 差 1 局（0.75 → 0.625）恰好落在 1/8 容差内。
        result = gate_verdict(self._summary(0.75), self._summary(0.625), 0.5)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "non-inferior")

    def test_allow_regression_never_blocks(self) -> None:
        result = gate_verdict(self._summary(0.75), self._summary(0.25), 0.01,
                              allow_regression=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "significant-worse")

    def test_accepts_bare_group_summary(self) -> None:
        result = gate_verdict({"episodes": 8, "winRate": 0.5},
                              {"episodes": 8, "winRate": 0.5}, 0.9)
        self.assertTrue(result["passed"])


class DistillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = Policy(obs_dim=6, n_actions=4, hidden=8, seed=13)
        self.obs = np.random.default_rng(4).normal(size=(200, 6))

    def test_noop_when_student_equals_teacher(self) -> None:
        """学生已等于老师时不能「往坏里走」（Adam 会把 1e-17 的浮点噪声放大成 lr 步）。"""
        out, info = distill([self.policy.params], self.obs,
                            init_params=self.policy.params, epochs=2, seed=0)
        worst = max(float(np.max(np.abs(out[k] - self.policy.params[k]))) for k in out)
        self.assertLess(worst, 1e-12)
        self.assertEqual(info["steps"], 0)
        self.assertGreater(info["skippedSteps"], 0)

    def test_kl_decreases_from_random_student(self) -> None:
        rng = np.random.default_rng(5)
        noisy = {k: v + rng.normal(0, 0.05, v.shape) for k, v in self.policy.params.items()}
        student = Policy(obs_dim=6, n_actions=4, hidden=8, seed=99).params
        out, info = distill([self.policy.params, noisy], self.obs,
                            init_params=student, epochs=3, seed=0)
        self.assertTrue(info["improved"])
        self.assertLess(info["klAfter"], info["klBefore"])
        self.assertGreater(info["steps"], 0)
        # 价值头不参与蒸馏，保持学生初值。
        np.testing.assert_allclose(out["Wv"], student["Wv"], atol=0.0)

    def test_empty_samples_rejected(self) -> None:
        with self.assertRaises(ValueError):
            distill([self.policy.params], np.zeros((0, 6)), epochs=1)


if __name__ == "__main__":
    unittest.main()
