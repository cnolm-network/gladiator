"""`MatchRunner._ensure_arena_record` 的重试语义测试。

真机上 Arena 在 `onMatchEnd` 里才把录制落盘，而 `_write_summary` 紧跟着就跑，
第一次 harvest 经常扑空——这里钉住「会重试、拿到就停、拿不到才放弃」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from match_runner import MatchRunner  # noqa: E402


class EnsureArenaRecordTest(unittest.TestCase):
    def _runner(self):
        runner = object.__new__(MatchRunner)
        runner.logs = []
        runner._log = lambda match, text, ok=True: runner.logs.append(text)
        return runner

    def test_retries_until_first_hit(self):
        runner = self._runner()
        calls = []

        def fake_harvest(match):
            calls.append(1)
            if len(calls) == 3:
                match["arena_record"] = "x.jsonl"

        runner._harvest_arena_record = fake_harvest
        match = {}
        self.assertTrue(runner._ensure_arena_record(match, attempts=5, delay=0.0))
        self.assertEqual(len(calls), 3)
        self.assertTrue(any("重试" in text for text in runner.logs))

    def test_stops_immediately_when_already_harvested(self):
        runner = self._runner()
        calls = []
        runner._harvest_arena_record = lambda match: calls.append(1)
        self.assertTrue(runner._ensure_arena_record({"arena_record": "x"}, attempts=5,
                                                    delay=0.0))
        self.assertEqual(calls, [])

    def test_gives_up_with_warning(self):
        runner = self._runner()
        calls = []
        runner._harvest_arena_record = lambda match: calls.append(1)
        self.assertFalse(runner._ensure_arena_record({}, attempts=3, delay=0.0))
        self.assertEqual(len(calls), 3)
        self.assertTrue(any("血量差" in text for text in runner.logs))

    def test_zero_attempts_still_tries_once(self):
        runner = self._runner()
        calls = []
        runner._harvest_arena_record = lambda match: calls.append(1)
        self.assertFalse(runner._ensure_arena_record({}, attempts=0, delay=0.0))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
