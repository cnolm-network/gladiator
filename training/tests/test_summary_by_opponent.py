"""P1-1：训练报告的分对手胜率（构造 2 对手 × 各 2 局的 history）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_train.train import summarize_records  # noqa: E402
from gladiator_train.trainer import EpisodeResult  # noqa: E402

USER = "Bot_T1"

HISTORY = [
    {"opponentMode": "scripted", "winner": "Bot_T1", "win": True, "totalReward": 1.0},
    {"opponentMode": "scripted", "winner": "Bot_T2", "win": False, "totalReward": -2.0},
    {"opponentMode": "rusher", "winner": "Bot_T1", "win": True, "totalReward": 3.0},
    {"opponentMode": "rusher", "winner": None, "win": False, "totalReward": 0.5},
]


class SummarizeRecordsTest(unittest.TestCase):
    def test_counts_and_win_rate_per_opponent(self):
        summary = summarize_records(HISTORY, USER)

        by_opponent = summary["byOpponent"]
        self.assertEqual(set(by_opponent), {"scripted", "rusher"})
        self.assertEqual(by_opponent["scripted"]["episodes"], 2)
        self.assertEqual(by_opponent["scripted"]["wins"], 1)
        self.assertAlmostEqual(by_opponent["scripted"]["winRate"], 0.5)
        self.assertAlmostEqual(by_opponent["scripted"]["meanReward"], -0.5)
        self.assertEqual(by_opponent["rusher"]["wins"], 1)
        self.assertAlmostEqual(by_opponent["rusher"]["winRate"], 0.5)
        self.assertAlmostEqual(by_opponent["rusher"]["meanReward"], 1.75)

        overall = summary["overall"]
        self.assertEqual(overall["episodes"], 4)
        self.assertEqual(overall["wins"], 2)
        self.assertAlmostEqual(overall["winRate"], 0.5)
        self.assertAlmostEqual(overall["meanReward"], 0.625)

    def test_empty_history(self):
        summary = summarize_records([], USER)
        self.assertEqual(summary["byOpponent"], {})
        self.assertEqual(summary["overall"]["episodes"], 0)
        self.assertEqual(summary["overall"]["winRate"], 0.0)

    def test_missing_opponent_mode_bucketed_as_unknown(self):
        summary = summarize_records([{"winner": "Bot_T1", "totalReward": 1.0}], USER)
        self.assertIn("unknown", summary["byOpponent"])
        self.assertEqual(summary["byOpponent"]["unknown"]["wins"], 1)

    def test_falls_back_to_winner_when_win_flag_missing(self):
        """旧 history 没有 win 字段时，用 winner == 学习方 判定。"""
        summary = summarize_records(
            [{"opponentMode": "idle", "winner": "Bot_T1", "totalReward": 0.0},
             {"opponentMode": "idle", "winner": "Bot_T2", "totalReward": 0.0}],
            USER,
        )
        self.assertEqual(summary["byOpponent"]["idle"]["wins"], 1)
        self.assertAlmostEqual(summary["byOpponent"]["idle"]["winRate"], 0.5)


class EpisodeResultSummaryTest(unittest.TestCase):
    def test_summary_carries_opponent_and_win(self):
        result = EpisodeResult(
            steps=10, total_reward=1.5, terminated=True, truncated=False,
            winner=USER, duration_s=1.0,
        )
        result.opponent_mode = "rusher"
        result.win = True
        payload = result.summary()
        self.assertEqual(payload["opponentMode"], "rusher")
        self.assertTrue(payload["win"])

    def test_defaults_are_empty_and_false(self):
        payload = EpisodeResult(
            steps=0, total_reward=0.0, terminated=False, truncated=True,
            winner=None, duration_s=0.0,
        ).summary()
        self.assertEqual(payload["opponentMode"], "")
        self.assertFalse(payload["win"])


if __name__ == "__main__":
    unittest.main()
