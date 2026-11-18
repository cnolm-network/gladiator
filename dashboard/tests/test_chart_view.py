"""P2.0 C：图表视图参数（分位裁剪 / 胜局标记 / 窗口胜率 / PPO 均值）。

要点：裁剪只是「视图」，不改数据 —— 断言的是区间与坐标映射，不是把数值改小。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from chart_view import (  # noqa: E402
    build_chart_view,
    clamp_range,
    entropy_series,
    is_frozen,
    last_metrics,
    mean_metrics,
    win_episodes,
    window_win_rate,
)


class ClampRangeTest(unittest.TestCase):
    def test_spike_does_not_stretch_the_axis(self):
        """带尖峰的序列：区间被 1%/99% 分位夹住，尖峰仍在数据里。"""
        values = [float(i) for i in range(98)] + [-400.0, 400.0]
        y_range = clamp_range(values)
        self.assertEqual(len(y_range), 2)
        self.assertGreater(y_range[0], -400.0)
        self.assertLess(y_range[1], 400.0)
        self.assertLessEqual(y_range[0], y_range[1])

    def test_flat_series_gives_nonzero_span(self):
        y_range = clamp_range([3.0] * 20)
        self.assertIsNotNone(y_range)
        self.assertGreater(y_range[1], y_range[0])

    def test_empty_and_missing_values(self):
        self.assertIsNone(clamp_range([]))
        self.assertIsNone(clamp_range([None, "x", float("nan")]))
        self.assertIsNotNone(clamp_range([None, 1.0, 2.0]))

    def test_custom_percentiles(self):
        values = list(range(1, 101))
        wide = clamp_range(values, 0, 100)
        narrow = clamp_range(values, 10, 90)
        self.assertLessEqual(wide[0], narrow[0])
        self.assertGreaterEqual(wide[1], narrow[1])


class WinRateTest(unittest.TestCase):
    def test_win_episodes_are_one_based(self):
        history = [{"winner": "A"}, {"winner": "B"}, {"winner": "A"}, {}]
        self.assertEqual(win_episodes(history, "A"), [1, 3])
        self.assertEqual(win_episodes(history, None), [])

    def test_win_flag_wins_over_winner(self):
        history = [{"win": True, "winner": None}, {"win": False, "winner": "A"}]
        self.assertEqual(win_episodes(history, "A"), [1])

    def test_window_win_rate(self):
        # 12 局，赢第 1–3 与第 11 局 → 窗口 1: 3/10，窗口 2: 1/2
        history = [{"winner": "A"} if i in (0, 1, 2, 10) else {"winner": "B"} for i in range(12)]
        windows = window_win_rate(history, "A", 10)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["startEpisode"], 1)
        self.assertEqual(windows[0]["episodes"], 10)
        self.assertEqual(windows[0]["wins"], 3)
        self.assertAlmostEqual(windows[0]["winRate"], 0.3)
        self.assertEqual(windows[1]["startEpisode"], 11)
        self.assertEqual(windows[1]["episodes"], 2)
        self.assertEqual(windows[1]["wins"], 1)
        self.assertAlmostEqual(windows[1]["winRate"], 0.5)

    def test_missing_fields_do_not_crash(self):
        history = [{}, {"totalReward": None}, {"winner": "A"}]
        view = build_chart_view(history, "A", 10)
        self.assertEqual(view["winEpisodes"], [3])
        self.assertEqual(sum(w["wins"] for w in view["winRate"]), 1)
        self.assertEqual(build_chart_view([], "A", 10)["winRate"], [])
        self.assertIsNone(build_chart_view([], "A", 10)["yRange"])


class EntropyCurveTest(unittest.TestCase):
    """P2.1：熵曲线与冻结告警；各曲线刻度互不影响。"""

    def test_entropy_series_keeps_alignment(self):
        history = [{"entropy": 0.4}, {"entropy": None}, {}, {"entropy": 0.1}]
        self.assertEqual(entropy_series(history), [0.4, None, None, 0.1])

    def test_frozen_after_three_low_episodes(self):
        history = [{"entropy": 0.5}, {"entropy": 0.04}, {"entropy": 0.03}, {"entropy": 0.02}]
        self.assertTrue(is_frozen(history))

    def test_not_frozen_when_a_high_episode_intervenes(self):
        history = [{"entropy": 0.04}, {"entropy": 0.03}, {"entropy": 0.2}, {"entropy": 0.01}]
        self.assertFalse(is_frozen(history))

    def test_missing_entropy_is_not_frozen(self):
        self.assertFalse(is_frozen([{}, {"entropy": None}, {"entropy": 0.01}]))

    def test_threshold_is_configurable(self):
        history = [{"entropy": 0.08}, {"entropy": 0.07}, {"entropy": 0.09}]
        self.assertFalse(is_frozen(history))
        self.assertTrue(is_frozen(history, threshold=0.1))

    def test_ranges_are_independent(self):
        """reward 的 1%/99% 裁剪不能影响熵曲线的刻度。"""
        history = [{"totalReward": 1e6 if i == 0 else float(i), "entropy": 0.2 + 0.01 * i}
                   for i in range(10)]
        view = build_chart_view(history, "A", 10)
        self.assertLess(view["yRange"][1], 1e6, "reward 轴应被裁剪")
        self.assertLessEqual(view["entropyRange"][1], 1.0, "熵轴应按熵自己定刻度")
        self.assertNotEqual(view["yRange"], view["entropyRange"])

    def test_last_metrics_takes_latest_non_null(self):
        history = [{"lr": 3e-4, "gradNorm": 1.0, "entropyCoef": 0.01},
                   {"lr": 2e-4, "gradNorm": None},
                   {"entropyCoef": 0.02}]
        out = last_metrics(history)
        self.assertAlmostEqual(out["lr"], 2e-4)
        self.assertAlmostEqual(out["gradNorm"], 1.0)
        self.assertAlmostEqual(out["entropyCoef"], 0.02)
        self.assertIsNone(last_metrics([])["lr"])


class ExplainedVarianceViewTest(unittest.TestCase):
    """P2.2：解释方差透出到视图层（运行详情「解释方差」一行）。"""

    def test_last_metrics_carries_explained_variance(self):
        history = [{"explainedVariance": 0.1}, {"explainedVariance": 0.55}]
        self.assertAlmostEqual(last_metrics(history)["explainedVariance"], 0.55)
        self.assertIsNone(last_metrics([{"lr": 1e-3}])["explainedVariance"])

    def test_build_chart_view_exposes_it(self):
        history = [{"totalReward": 1.0, "explainedVariance": 0.42}]
        view = build_chart_view(history, "A", 10)
        self.assertAlmostEqual(view["lastMetrics"]["explainedVariance"], 0.42)

    def test_mean_metrics_includes_it(self):
        records = [{"explainedVariance": 0.2}, {"explainedVariance": 0.4}]
        self.assertAlmostEqual(mean_metrics(records)["explainedVariance"], 0.3)
        self.assertIsNone(mean_metrics([{}])["explainedVariance"])


class MeanMetricsTest(unittest.TestCase):
    def test_means_and_missing(self):
        records = [
            {"valueLoss": 1.0, "approxKL": 0.1, "clipFraction": 0.0},
            {"valueLoss": 3.0, "approxKL": 0.3, "clipFraction": 0.2},
            {"valueLoss": None},
        ]
        out = mean_metrics(records)
        self.assertAlmostEqual(out["valueLoss"], 2.0)
        self.assertAlmostEqual(out["approxKL"], 0.2)
        self.assertAlmostEqual(out["clipFraction"], 0.1)
        self.assertIsNone(out["policyLoss"])
        self.assertIsNone(mean_metrics([])["valueLoss"])


class ChartViewTest(unittest.TestCase):
    def test_build_chart_view_shape(self):
        history = [{"totalReward": float(i), "winner": "A" if i % 5 == 0 else "B"}
                   for i in range(10)]
        view = build_chart_view(history, "A", 5)
        self.assertEqual(view["episodes"], 10)
        self.assertEqual(view["winEpisodes"], [1, 6])
        self.assertEqual(len(view["winRate"]), 2)
        self.assertEqual(view["winRate"][0]["episodes"], 5)
        self.assertEqual(view["yRange"][0] <= view["yRange"][1], True)


if __name__ == "__main__":
    unittest.main()
