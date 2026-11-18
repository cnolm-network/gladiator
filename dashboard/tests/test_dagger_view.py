"""P2.3：DAgger 轮次表在 dashboard 的透出（结构固定、旧 run 给空表）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from artifacts import ArtifactStore, _dagger_rounds  # noqa: E402

DAGGER = [
    {"round": 0, "samples": 120, "agreement": 0.55, "loss": 0.42,
     "evaluation": {"sampling": {"wins": 3, "episodes": 12, "winRate": 0.25, "meanReward": 1.0},
                    "greedy": {"wins": 4, "episodes": 12, "winRate": 0.3333, "meanReward": 2.0}}},
    {"round": 1, "samples": 240, "agreement": 0.78, "agreementBefore": 0.55, "loss": 0.11,
     "evaluation": {"sampling": {"wins": 6, "episodes": 12, "winRate": 0.5, "meanReward": 5.0},
                    "greedy": {"wins": 7, "episodes": 12, "winRate": 0.5833, "meanReward": 6.0}}},
]


class DaggerRoundsUnitTest(unittest.TestCase):
    def test_normalises_rounds(self):
        rows = _dagger_rounds(DAGGER)
        self.assertEqual([row["round"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["samples"], 240)
        self.assertAlmostEqual(rows[1]["agreement"], 0.78)
        self.assertEqual(rows[1]["sampling"]["wins"], 6)
        self.assertEqual(rows[1]["greedy"]["episodes"], 12)

    def test_missing_evaluation_becomes_none(self):
        rows = _dagger_rounds([{"round": 2, "samples": 10}])
        self.assertIsNone(rows[0]["sampling"])
        self.assertIsNone(rows[0]["greedy"])
        self.assertIsNone(rows[0]["agreement"])

    def test_non_list_and_junk_are_safe(self):
        self.assertEqual(_dagger_rounds(None), [])
        self.assertEqual(_dagger_rounds("oops"), [])
        self.assertEqual(_dagger_rounds([1, "x", None]), [])


class DaggerReportTest(unittest.TestCase):
    def test_train_report_carries_dagger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "20260103-0100-train-Bot_D"
            run_dir.mkdir(parents=True)
            (run_dir / "report.json").write_text(json.dumps({
                "kind": "train",
                "config": {"users": ["Bot_D", "Bot_E"]},
                "history": [],
                "evaluation": [],
                "dagger": DAGGER,
            }), encoding="utf-8")
            store = ArtifactStore(root / "runs")
            report = store.train_report(run_dir / "report.json")
            self.assertTrue(report["exists"])
            self.assertEqual(len(report["dagger"]), 2)
            self.assertEqual(report["dagger"][0]["greedy"]["wins"], 4)

    def test_old_report_has_no_dagger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "20251231-0100-train-Bot_O"
            run_dir.mkdir(parents=True)
            (run_dir / "report.json").write_text(json.dumps({
                "kind": "train", "config": {"users": ["Bot_O"]},
                "history": [], "evaluation": [],
            }), encoding="utf-8")
            store = ArtifactStore(root / "runs")
            self.assertEqual(store.train_report(run_dir / "report.json")["dagger"], [])


if __name__ == "__main__":
    unittest.main()
