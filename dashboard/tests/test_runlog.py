"""P-6 日志管道：增量切片、切段、轮转、启动失败、历史 no-log。

覆盖：
- ``rotate_log`` / ``should_rotate_log``：超 5MB、跨天、保留份数；
- ``ProcessManager`` 启动时把 ``log_offset`` 记成 header 之前的位置（切段依据）；
- ``TrainingRunner._attach_run_log`` 只读本次增量（旧内容不串场）；
- ``TrainingRunner._fail_before_spawn`` 写一份含原因的 run.log；
- 历史 run 没有 run.log 时 ``artifacts`` 返回 ``logState=no-log``，``terminal`` 明确 noLog。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from processes import (  # noqa: E402
    LOG_KEEP,
    LOG_ROTATE_BYTES,
    ProcessManager,
    rotate_log,
    should_rotate_log,
)
from training_runner import TrainingJob, TrainingRunner  # noqa: E402


def _cfg(root: Path, *, port: int = 8080, mc_port: int = 25993) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1",
        port=port,
        artifacts_dir=root / "artifacts",
        var_dir=root / "var",
        server_dir=root / "server",
        bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training",
        python_exe=sys.executable,
        java_exe="java",
        node_exe="node",
        allow_process_control=False,
        scan_external=False,
        server_jar="missing.jar",
        mc_port=mc_port,
    )


class RotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_small_file_not_rotated(self) -> None:
        path = self.dir / "training.log"
        path.write_text("small\n", encoding="utf-8")
        now = time.time()
        self.assertFalse(should_rotate_log(path, now=now))
        self.assertIsNone(rotate_log(path, now=now))

    def test_oversize_file_rotated(self) -> None:
        path = self.dir / "training.log"
        path.write_text("x" * (LOG_ROTATE_BYTES + 10), encoding="utf-8")
        archived = rotate_log(path)
        self.assertIsNotNone(archived)
        self.assertFalse(path.exists(), "轮转后当前文件应消失，下一次启动重建")
        self.assertTrue(archived.name.startswith("training-"))
        self.assertTrue(archived.name.endswith(".log"))

    def test_cross_day_file_rotated(self) -> None:
        path = self.dir / "training.log"
        path.write_text("yesterday\n", encoding="utf-8")
        old = time.time() - 26 * 3600
        os.utime(path, (old, old))
        self.assertTrue(should_rotate_log(path))
        self.assertIsNotNone(rotate_log(path))

    def test_keeps_only_recent_archives(self) -> None:
        path = self.dir / "training.log"
        for index in range(LOG_KEEP + 3):
            stale = self.dir / f"training-2026010{index}-000000.log"
            stale.write_text(f"old {index}\n", encoding="utf-8")
            stamp = time.time() - (1000 - index * 10)
            os.utime(stale, (stamp, stamp))
        path.write_text("x" * (LOG_ROTATE_BYTES + 1), encoding="utf-8")
        self.assertIsNotNone(rotate_log(path))
        remaining = sorted(self.dir.glob("training-*.log"))
        self.assertEqual(len(remaining), LOG_KEEP, f"应只保留最近 {LOG_KEEP} 份: {remaining}")

    def test_rotation_does_not_touch_other_process_logs(self) -> None:
        target = self.dir / "training.log"
        other = self.dir / "paper.log"
        other.write_text("paper output\n", encoding="utf-8")
        target.write_text("x" * (LOG_ROTATE_BYTES + 1), encoding="utf-8")
        rotate_log(target)
        self.assertTrue(other.is_file())


class SpawnOffsetTests(unittest.TestCase):
    """ProcessManager 记录 log_offset：这是增量切片与切段的依据。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.procs = ProcessManager(self.dir / "logs", self.dir / "managed.json")

    def tearDown(self) -> None:
        self.procs.stop_all(timeout=3)
        self.tmp.cleanup()

    def test_offset_points_at_this_run(self) -> None:
        log_path = self.dir / "logs" / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("旧内容不应进 run.log\n", encoding="utf-8")
        previous = log_path.stat().st_size

        proc = self.procs.spawn(
            proc_id="job",
            kind="test",
            label="demo",
            argv=[sys.executable, "-c", "print('hello')"],
            cwd=self.dir,
        )
        self.assertEqual(proc.log_offset, previous)
        proc.popen.wait(timeout=10)
        # pump 线程把输出写盘是异步的：轮询等它落盘，避免固定 sleep 造成的偶发失败。
        deadline = time.monotonic() + 10
        text = ""
        while time.monotonic() < deadline:
            data = log_path.read_bytes()[proc.log_offset:]
            text = data.decode("utf-8", errors="replace")
            if "退出码" in text:
                break
            time.sleep(0.02)
        self.assertIn("启动 job", text)
        self.assertIn("hello", text)
        self.assertNotIn("旧内容", text)


class _RunnerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = _cfg(self.root)
        self.runner = TrainingRunner(self.cfg, ProcessManager(self.cfg.logs_dir, self.cfg.state_file),
                                     _NullArtifacts())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _job(self, run_dir: Path, argv: list[str] | None = None) -> TrainingJob:
        return TrainingJob(id="training", kind="train", params={}, total_episodes=1,
                           run_dir=str(run_dir), argv=argv or [])


class _NullArtifacts:
    """_attach_run_log 不碰 artifacts；给个占位避免构造真实 ArtifactStore。"""


class IncrementalRunLogTests(_RunnerFixture):
    def test_only_new_bytes_are_copied(self) -> None:
        log_path = self.root / "logs" / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("=== 2026-11-08T10:00:00 启动 training ===\n$ old run\nOLD-LINE\n",
                            encoding="utf-8")
        offset = log_path.stat().st_size
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("=== 2026-11-08T11:00:00 启动 training ===\n$ python -m gladiator_train.train\nNEW-LINE\n=== 退出码 0 ===\n")

        run_dir = self.root / "runs" / "run-a"
        job = self._job(run_dir)
        proc = SimpleNamespace(log_path=log_path, log_offset=offset)
        self.runner._attach_run_log(job, proc)

        text = (run_dir / "run.log").read_text(encoding="utf-8")
        self.assertIn("NEW-LINE", text)
        self.assertNotIn("OLD-LINE", text, "offset 之前的内容不能进 run.log")
        self.assertNotIn("启动 training", text, "Dashboard 的 header 不应进 run.log")
        self.assertNotIn("$ python -m", text)

    def test_existing_run_log_not_overwritten(self) -> None:
        log_path = self.root / "logs" / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("NEW\n", encoding="utf-8")
        run_dir = self.root / "runs" / "run-b"
        run_dir.mkdir(parents=True)
        (run_dir / "run.log").write_text("keep me\n", encoding="utf-8")
        self.runner._attach_run_log(self._job(run_dir), SimpleNamespace(log_path=log_path, log_offset=0))
        self.assertEqual((run_dir / "run.log").read_text(encoding="utf-8"), "keep me\n")

    def test_truncated_file_reads_from_start(self) -> None:
        log_path = self.root / "logs" / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("=== 2026-11-08T11:00:00 启动 training ===\nAFTER-ROTATE\n", encoding="utf-8")
        run_dir = self.root / "runs" / "run-c"
        # offset 大于文件长度：模拟运行中轮转/截断，应回退到 0 继续读
        self.runner._attach_run_log(self._job(run_dir), SimpleNamespace(log_path=log_path, log_offset=99999))
        self.assertIn("AFTER-ROTATE", (run_dir / "run.log").read_text(encoding="utf-8"))


class FailureRunLogTests(_RunnerFixture):
    def test_fail_before_spawn_writes_reason(self) -> None:
        run_dir = self.root / "runs" / "run-fail"
        run_dir.mkdir(parents=True)
        job = self._job(run_dir, argv=[sys.executable, "-m", "gladiator_train.train", "--epochs", "2"])
        job.notes.append("未找到 runtime")
        self.runner._fail_before_spawn(job, "端口 8770 没有 bot-runtime 在监听")
        text = (run_dir / "run.log").read_text(encoding="utf-8")
        self.assertIn("训练未启动", text)
        self.assertIn("端口 8770 没有 bot-runtime 在监听", text)
        self.assertIn("gladiator_train.train", text)
        self.assertIn("未找到 runtime", text)
        self.assertEqual(job.status, "failed")

    def test_finish_before_spawn_writes_reason(self) -> None:
        run_dir = self.root / "runs" / "run-stop"
        run_dir.mkdir(parents=True)
        job = self._job(run_dir)
        job.stop_requested = True
        job.status = "running"
        self.runner._finish_before_spawn(job)
        text = (run_dir / "run.log").read_text(encoding="utf-8")
        self.assertIn("启动前被停止", text)
        self.assertEqual(job.status, "stopped")

    def test_existing_run_log_kept(self) -> None:
        run_dir = self.root / "runs" / "run-keep"
        run_dir.mkdir(parents=True)
        (run_dir / "run.log").write_text("已有内容\n", encoding="utf-8")
        self.runner._fail_before_spawn(self._job(run_dir), "boom")
        self.assertEqual((run_dir / "run.log").read_text(encoding="utf-8"), "已有内容\n")


class NoLogStateTests(unittest.TestCase):
    """历史 run 没有 run.log：API 与终端都要给明确状态。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.root = root
        cfg = _cfg(root, mc_port=25994)
        run_dir = cfg.artifacts_dir / "runs" / "20261108-2100-train-Bot_Hist"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text('{"runId": "x", "kind": "train"}', encoding="utf-8")
        cls.run_id = run_dir.name
        cls.app = DashboardApp(cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      kwargs={"poll_interval": 0.1}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.app.close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def test_run_detail_marks_no_log(self) -> None:
        detail = self.app.artifacts.run_detail(self.run_id)
        self.assertEqual(detail["logState"], "no-log")
        self.assertIsNone(detail["logPath"])

    def test_run_with_log_marks_ok(self) -> None:
        run_dir = self.root / "artifacts" / "runs" / "20261108-2101-train-Bot_WithLog"
        run_dir.mkdir(parents=True)
        (run_dir / "run.log").write_text("hello\n", encoding="utf-8")
        detail = self.app.artifacts.run_detail(run_dir.name)
        self.assertEqual(detail["logState"], "ok")

    def test_terminal_returns_nolog_meta(self) -> None:
        spec = self.app.terminal.resolve("training", self.run_id)
        self.assertTrue(spec.meta.get("noLog"))
        self.assertEqual(spec.meta.get("logState"), "no-log")
        self.assertIn("没有日志", spec.note)
        self.assertIsNone(spec.path)


if __name__ == "__main__":
    unittest.main()
