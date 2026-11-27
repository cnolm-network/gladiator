"""`pfsp.py` 的单测：权重方向、退化情形、EMA 与池行为。"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gladiator_train.pfsp import (  # noqa: E402
    OpponentPool,
    pfsp_weights,
    update_win_rate,
)


class WeightTests(unittest.TestCase):
    def test_lower_win_rate_gets_more_weight(self):
        weights = pfsp_weights([0.0, 0.5, 1.0])
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)

    def test_unknown_opponent_treated_as_unbeaten(self):
        weights = pfsp_weights([None, 1.0])
        self.assertGreater(weights[0], weights[1])

    def test_all_won_falls_back_to_equal(self):
        weights = pfsp_weights([1.0, 1.0, 1.0])
        np.testing.assert_allclose(weights, np.full(3, 1 / 3), rtol=1e-9)

    def test_empty_pool_returns_empty(self):
        self.assertEqual(pfsp_weights([]).size, 0)

    def test_exponent_strengthens_priority(self):
        flat = pfsp_weights([0.0, 0.8], exponent=1.0)
        sharp = pfsp_weights([0.0, 0.8], exponent=4.0)
        self.assertGreater(sharp[0] - sharp[1], flat[0] - flat[1])

    def test_out_of_range_values_are_clamped(self):
        weights = pfsp_weights([-3.0, 7.0])
        self.assertGreater(weights[0], weights[1])

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            pfsp_weights([0.5], exponent=-1)
        with self.assertRaises(ValueError):
            pfsp_weights([0.5], epsilon=2.0)


class WinRateTests(unittest.TestCase):
    def test_first_batch_is_plain_ratio(self):
        self.assertAlmostEqual(update_win_rate(None, 3, 1), 0.75)

    def test_ema_blends_with_previous(self):
        self.assertAlmostEqual(update_win_rate(0.5, 4, 0, ema=0.5), 0.75)

    def test_empty_batch_keeps_previous(self):
        self.assertEqual(update_win_rate(0.42, 0, 0), 0.42)
        self.assertEqual(update_win_rate(None, 0, 0), 0.0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            update_win_rate(None, -1, 0)
        with self.assertRaises(ValueError):
            update_win_rate(0.5, 1, 1, ema=0.0)


class PoolTests(unittest.TestCase):
    def test_sample_prefers_hard_opponent(self):
        pool = OpponentPool()
        easy = pool.add("easy.npz")
        hard = pool.add("hard.npz")
        pool.record(easy, 9, 1)      # 已经很好打
        rng = np.random.default_rng(0)
        picks = [pool.sample(rng) for _ in range(400)]
        self.assertGreater(picks.count(hard), picks.count(easy))

    def test_record_updates_ema_and_keeps_entry(self):
        pool = OpponentPool()
        path = pool.add("a.npz")
        self.assertIsNone(pool.win_rates()[path])
        pool.record(path, 1, 1)
        self.assertAlmostEqual(pool.win_rates()[path], 0.5)
        pool.record(path, 1, 0)
        self.assertGreater(pool.win_rates()[path], 0.5)

    def test_empty_pool_sample_is_none(self):
        self.assertIsNone(OpponentPool().sample(np.random.default_rng(0)))

    def test_discover_reads_npz_from_root(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("round0.npz", "round1.npz"):
                with open(os.path.join(folder, name), "wb") as handle:
                    handle.write(b"x")
            with open(os.path.join(folder, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("ignore me")
            pool = OpponentPool(folder)
            found = pool.discover()
            self.assertEqual([os.path.basename(path) for path in found], ["round0.npz", "round1.npz"])
            self.assertEqual(pool.summary()["size"], 2)

    def test_summary_shape(self):
        pool = OpponentPool()
        pool.add("a.npz")
        summary = pool.summary()
        self.assertEqual(summary["size"], 1)
        self.assertIsNone(summary["opponents"][0]["winRate"])
        self.assertAlmostEqual(summary["opponents"][0]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
