"""`smoke_board_match.py` 的判据单测：不联网、不读真机产物。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smoke_board_match import (  # noqa: E402
    dataset_report,
    latest_match_dir,
    match_snapshot,
    record_files,
    verify,
)


class SnapshotTests(unittest.TestCase):
    def test_unwraps_match_key(self):
        self.assertEqual(match_snapshot({"ok": True, "match": {"status": "finished"}})["status"], "finished")

    def test_passes_through_plain_snapshot(self):
        self.assertEqual(match_snapshot({"status": "running"})["status"], "running")


class VerifyTests(unittest.TestCase):
    def _ok_match(self, rounds=2):
        return {
            "status": "finished",
            "score": {"player": 0, "bot": rounds, "draws": 0},
            "roundsDetail": [
                {"round": index + 1, "winner": "bot", "reason": "one-team-left", "seconds": 12.0}
                for index in range(rounds)
            ],
        }

    def _patch_dataset(self, report):
        import scripts.smoke_board_match as module

        original = module.dataset_report
        module.dataset_report = lambda _path: report  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(module, "dataset_report", original))

    def _patch_records(self, count):
        import scripts.smoke_board_match as module

        original = module.record_files
        module.record_files = lambda _path: ["r%d.jsonl" % index for index in range(count)]  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(module, "record_files", original))

    def test_happy_path(self):
        self._patch_records(2)
        self._patch_dataset(
            {
                "ok": True,
                "obsShape": [120, 39],
                "rounds": [1, 2],
                "rewardSource": "arena-record",
                "rewardSum": -44.8,
                "nonZeroFrames": 28,
                "frames": 120,
            }
        )
        result = verify(self._ok_match(2), "/tmp/fake", 2, 39)
        self.assertTrue(result["ok"], result["problems"])

    def test_flags_unfinished_match(self):
        self._patch_records(2)
        self._patch_dataset(
            {"ok": True, "obsShape": [10, 39], "rounds": [1], "rewardSource": "arena-record",
             "rewardSum": -1.0, "nonZeroFrames": 1, "frames": 10}
        )
        match = self._ok_match(2)
        match["status"] = "failed"
        match["error"] = "端口 8787 已被占用"
        problems = verify(match, "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("finished" in problem for problem in problems))
        self.assertTrue(any("端口" in problem for problem in problems))

    def test_flags_missing_records(self):
        self._patch_records(1)
        self._patch_dataset(
            {"ok": True, "obsShape": [10, 39], "rounds": [1], "rewardSource": "arena-record",
             "rewardSum": -1.0, "nonZeroFrames": 1, "frames": 10}
        )
        problems = verify(self._ok_match(2), "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("录制" in problem for problem in problems), problems)

    def test_flags_health_delta_rewards(self):
        self._patch_records(2)
        self._patch_dataset(
            {"ok": True, "obsShape": [10, 39], "rounds": [1, 2], "rewardSource": "health-delta",
             "rewardSum": -1.0, "nonZeroFrames": 1, "frames": 10}
        )
        problems = verify(self._ok_match(2), "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("arena-record" in problem for problem in problems), problems)

    def test_flags_all_zero_rewards(self):
        self._patch_records(2)
        self._patch_dataset(
            {"ok": True, "obsShape": [10, 39], "rounds": [1, 2], "rewardSource": "arena-record",
             "rewardSum": 0.0, "nonZeroFrames": 0, "frames": 10}
        )
        problems = verify(self._ok_match(2), "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("全是 0" in problem for problem in problems), problems)

    def test_flags_wrong_obs_dim(self):
        self._patch_records(2)
        self._patch_dataset(
            {"ok": True, "obsShape": [10, 37], "rounds": [1, 2], "rewardSource": "arena-record",
             "rewardSum": -2.0, "nonZeroFrames": 2, "frames": 10}
        )
        problems = verify(self._ok_match(2), "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("维度" in problem for problem in problems), problems)

    def test_flags_missing_dataset(self):
        self._patch_records(2)
        self._patch_dataset({"ok": False, "error": "dataset.npz 不存在"})
        problems = verify(self._ok_match(2), "/tmp/fake", 2, 39)["problems"]
        self.assertTrue(any("数据集不可读" in problem for problem in problems), problems)


class DirectoryTests(unittest.TestCase):
    def test_latest_match_dir_skips_old_and_returns_newest(self):
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as folder:
            old = os.path.join(folder, "20260101-000000-a")
            new = os.path.join(folder, "20261231-235959-b")
            os.makedirs(old)
            os.makedirs(new)
            stale = time.time() - 7200
            os.utime(old, (stale, stale))
            since = time.time() - 60
            self.assertEqual(latest_match_dir(folder, since), new)

    def test_latest_match_dir_empty_when_absent(self):
        self.assertIsNone(latest_match_dir(os.path.join(os.getcwd(), "no-such-dir"), 0.0))

    def test_record_files_empty_without_subdir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(record_files(folder), [])


if __name__ == "__main__":
    unittest.main()
