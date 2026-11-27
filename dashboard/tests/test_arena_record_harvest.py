"""`MatchRunner._harvest_arena_record` 的行为测试。

只测「挑哪份文件、放去哪里、拿不到怎么办」——不启动服务器、不需要 RCON。
用 `object.__new__(MatchRunner)` 绕过构造函数，挂上最小可用的 `cfg` 与 `match` 字典。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from match_runner import MatchRunner  # noqa: E402


def _runner(server_dir: Path):
    runner = object.__new__(MatchRunner)
    runner.cfg = SimpleNamespace(server_dir=server_dir)
    runner._log = lambda match, text, ok=True: match.setdefault("_logs", []).append(text)
    return runner


class HarvestArenaRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.server = self.root / "server"
        self.records = self.server / "plugins" / "Arena" / "records"
        self.records.mkdir(parents=True)
        self.record_dir = self.root / "match"
        self.record_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name: str, body: str, age_seconds: float = 0.0) -> Path:
        path = self.records / name
        path.write_text(body, encoding="utf-8")
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
        return path

    def _match(self, started_ago: float = 30.0) -> dict:
        return {
            "started_at": time.time() - started_ago,
            "snapshot": {"recordDir": str(self.record_dir)},
        }

    def test_copies_newest_record_from_this_match(self):
        self._write("old.jsonl", '{"seq":1}\n', age_seconds=600)
        self._write("new.jsonl", '{"seq":2}\n', age_seconds=1)
        match = self._match()
        _runner(self.server)._harvest_arena_record(match)
        target = self.record_dir / "arena-record.jsonl"
        self.assertTrue(target.is_file())
        self.assertIn('"seq":2', target.read_text(encoding="utf-8"))
        self.assertEqual(match.get("arena_record"), str(target))

    def test_skips_records_older_than_match_start(self):
        self._write("old.jsonl", '{"seq":1}\n', age_seconds=600)
        match = self._match(started_ago=30.0)
        _runner(self.server)._harvest_arena_record(match)
        self.assertFalse((self.record_dir / "arena-record.jsonl").exists())
        self.assertNotIn("arena_record", match)

    def test_missing_records_dir_is_not_an_error(self):
        empty = self.root / "nope"
        empty.mkdir()
        match = self._match()
        _runner(empty)._harvest_arena_record(match)
        self.assertNotIn("arena_record", match)
        self.assertNotIn("_logs", match)

    def test_missing_record_dir_in_snapshot_is_noop(self):
        self._write("new.jsonl", '{"seq":2}\n', age_seconds=1)
        match = {"started_at": time.time() - 30, "snapshot": {}}
        _runner(self.server)._harvest_arena_record(match)
        self.assertNotIn("arena_record", match)

    def test_overwrites_previous_copy(self):
        target = self.record_dir / "arena-record.jsonl"
        target.write_text("stale\n", encoding="utf-8")
        self._write("new.jsonl", '{"seq":3}\n', age_seconds=1)
        match = self._match()
        _runner(self.server)._harvest_arena_record(match)
        self.assertIn('"seq":3', target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

class ImportCommandRecordsTest(unittest.TestCase):
    """录制目录里有 arena-record.jsonl 时，赛后导入要带上 --records。"""

    def test_adds_records_when_artifact_present(self):
        from match_runner import ARENA_RECORD_NAME, build_import_command

        with tempfile.TemporaryDirectory() as raw:
            record_dir = Path(raw)
            script = record_dir / "import_human_match.py"
            script.write_text("", encoding="utf-8")
            plain = build_import_command("py", script, record_dir, bc=False,
                                         base_model=None)
            self.assertNotIn("--records", plain)

            (record_dir / ARENA_RECORD_NAME).write_text("", encoding="utf-8")
            with_events = build_import_command("py", script, record_dir, bc=True,
                                               base_model="dagger-1118b")
            self.assertIn("--records", with_events)
            self.assertEqual(with_events[with_events.index("--records") + 1],
                             str(record_dir / ARENA_RECORD_NAME))
            # 顺序：录制目录 → --records → --bc → --base-model
            self.assertLess(with_events.index("--records"), with_events.index("--bc"))
