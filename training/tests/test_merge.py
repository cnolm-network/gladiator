"""多场地权重融合（G-625）单测：``multi_train._merge_policies``。

覆盖：等权平均、按步加权、merge=none、单 pair 失败跳过、形状不一致报错、
以及写入 ``report.merge`` 的字段（method / pairs / mergedSha256 / skipped）。

全部用内存假权重 + 临时目录，不启动真实训练、不依赖真实 artifacts。
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
