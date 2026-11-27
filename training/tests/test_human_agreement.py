"""`human_agreement.py` 的纯函数单测（v1 与 v2 两种动作空间）。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.human_agreement import agreement  # noqa: E402


class V1AgreementTests(unittest.TestCase):
    def test_perfect_match(self):
        truth = np.asarray([0, 1, 2, 1])
        report = agreement(truth, truth.copy())
        self.assertEqual(report["overall"], 1.0)
        self.assertEqual(report["samples"], 4)
        self.assertEqual(report["perHead"], [])
        self.assertEqual(report["confusions"], [])

    def test_half_match_and_confusions(self):
        truth = np.asarray([0, 0, 1, 1])
        guess = np.asarray([0, 2, 1, 3])
        report = agreement(truth, guess)
        self.assertEqual(report["overall"], 0.5)
        pairs = {row["pair"]: row["count"] for row in report["confusions"]}
        self.assertEqual(pairs, {"0->2": 1, "1->3": 1})

    def test_per_action_rates(self):
        truth = np.asarray([0, 0, 5, 5, 5])
        guess = np.asarray([0, 1, 5, 2, 2])
        report = agreement(truth, guess)
        rows = {row["action"]: row for row in report["perAction"]}
        self.assertEqual(rows[0]["samples"], 2)
        self.assertAlmostEqual(rows[0]["hit"], 0.5)
        self.assertEqual(rows[5]["samples"], 3)
        self.assertAlmostEqual(rows[5]["hit"], 1 / 3, places=4)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            agreement(np.zeros(3, dtype=int), np.zeros((3, 2), dtype=int))


class V2AgreementTests(unittest.TestCase):
    def test_exact_and_per_head(self):
        truth = np.zeros((4, 3), dtype=int)
        guess = truth.copy()
        guess[0, 1] = 1
        guess[1, 2] = 2
        report = agreement(truth, guess)
        self.assertEqual(report["samples"], 4)
        self.assertAlmostEqual(report["exact"], 0.5)
        self.assertEqual(len(report["perHead"]), 3)
        self.assertAlmostEqual(report["perHead"][0], 1.0)
        self.assertAlmostEqual(report["perHead"][1], 0.75)
        self.assertAlmostEqual(report["perHead"][2], 0.75)
        self.assertAlmostEqual(report["overall"], (1.0 + 0.75 + 0.75) / 3)

    def test_confusions_are_per_head(self):
        truth = np.zeros((2, 2), dtype=int)
        guess = np.zeros((2, 2), dtype=int)
        guess[0, 0] = 1
        report = agreement(truth, guess)
        self.assertEqual(report["confusions"][0]["pair"], "head0:0->1")
        self.assertEqual(report["confusions"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
