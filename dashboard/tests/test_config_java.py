"""P-8（回修）：Java 探测**只接受 Java 8**，并把它暴露到 /api/health。

不依赖本机真实安装：假 exe 是临时空文件，版本由注入的 runner 冒充。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib import request as urlrequest

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import config  # noqa: E402
from config import (  # noqa: E402
    DashboardConfig,
    discover_java,
    java_major_version,
    load_config_file,
    parse_java_version,
)
from errors import Conflict  # noqa: E402
from processes import ProcessManager  # noqa: E402
from server_manager import ServerManager  # noqa: E402


def _fake_exe(root: Path, name: str = "java.exe") -> str:
    """在临时目录造一个假 java 可执行文件（内容无关，存在性才算数）。"""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return str(path)


def _runner(mapping: dict[str, int]):
    """冒充 ``version_probe``：按路径返回 major 版本（未登记视为探测不到）。"""
    def run(path: str) -> int | None:
        return mapping.get(str(Path(path)))
    return run


class VersionParseTests(unittest.TestCase):
    def test_legacy_1_8(self) -> None:
        self.assertEqual(parse_java_version('java version "1.8.0_492"'), 8)

    def test_modern(self) -> None:
        self.assertEqual(parse_java_version('openjdk version "17.0.19" 2026-01-20'), 17)
        self.assertEqual(parse_java_version('openjdk version "21" 2026-01-20'), 21)

    def test_garbage(self) -> None:
        self.assertIsNone(parse_java_version(""))
        self.assertIsNone(parse_java_version("command not found"))


class JavaMajorVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config.clear_java_version_cache()

    def tearDown(self) -> None:
        config.clear_java_version_cache()
        self.tmp.cleanup()

    def test_reads_injected_runner(self) -> None:
        exe = _fake_exe(self.root / "jdk8")
        # 每次给独立缓存，避免前一次结果命中缓存（同一文件 mtime 相同）
        self.assertEqual(
            java_major_version(exe, runner=lambda _p: 'java version "1.8.0_492"', cache={}), 8)
        self.assertEqual(
            java_major_version(exe, runner=lambda _p: 'openjdk version "17.0.19"', cache={}), 17)

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(java_major_version(str(self.root / "gone.exe"), runner=lambda _p: "x"))

    def test_runner_failure_returns_none(self) -> None:
        exe = _fake_exe(self.root / "broken")

        def boom(_path: str) -> str:
            raise OSError("cannot execute")

        self.assertIsNone(java_major_version(exe, runner=boom))

    def test_cache_avoids_repeat_probe(self) -> None:
        exe = _fake_exe(self.root / "jdk8")
        calls: list[str] = []

        def runner(path: str) -> str:
            calls.append(path)
            return 'java version "1.8.0_492"'

        cache: dict = {}
        self.assertEqual(java_major_version(exe, runner=runner, cache=cache), 8)
        self.assertEqual(java_major_version(exe, runner=runner, cache=cache), 8)
        self.assertEqual(len(calls), 1, "同一文件+mtime 应命中缓存")

        # 文件被替换（mtime/size 变化）后应重新探测
        Path(exe).write_text("changed!", encoding="utf-8")
        os.utime(exe, (1, 1))
        self.assertEqual(java_major_version(exe, runner=runner, cache=cache), 8)
        self.assertEqual(len(calls), 2)


class DiscoverJavaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _discover(self, mapping: dict[str, int], **kwargs):
        return discover_java(is_file=os.path.isfile, version_probe=_runner(mapping), roots=[], **kwargs)

    def test_accepts_java8(self) -> None:
        exe = _fake_exe(self.root / "jdk8")
        choice = self._discover({exe: 8}, explicit=exe)
        self.assertEqual(choice.path, exe)
        self.assertEqual(choice.version, 8)
        self.assertEqual(choice.hint, "")

    def test_skips_java17_and_keeps_looking(self) -> None:
        bad = _fake_exe(self.root / "jdk17")
        good = _fake_exe(self.root / "jdk8")
        # explicit 是 17，PATH 才是 8：应跳过 17 用 8
        choice = self._discover({bad: 17, good: 8}, explicit=bad, path_lookup=good)
        self.assertEqual(choice.path, good)
        self.assertEqual(choice.version, 8)
        self.assertIn((bad, 17), choice.rejected)

    def test_only_java17_returns_none_with_hint(self) -> None:
        bad = _fake_exe(self.root / "jdk17")
        choice = self._discover({bad: 17}, path_lookup=bad)
        self.assertIsNone(choice.path)
        self.assertIsNone(choice.version)
        self.assertIn("检测到 Java 17", choice.hint)
        self.assertIn(bad, choice.hint)
        self.assertIn("Paper 1.8.8", choice.hint)

    def test_explicit_17_does_not_silently_use_it(self) -> None:
        """显式配置 17 且没有别的 8：必须 None，绝不静默返回 17。"""
        bad = _fake_exe(self.root / "jdk17")
        choice = self._discover({bad: 17}, explicit=bad)
        self.assertIsNone(choice.path)
        self.assertIn(bad, choice.hint)
        self.assertIn("Java 17", choice.hint)

    def test_nothing_found_has_generic_hint(self) -> None:
        choice = discover_java(roots=[], is_file=os.path.isfile, version_probe=_runner({}))
        self.assertIsNone(choice.path)
        self.assertIn("未找到 Java 8", choice.hint)

    def test_scan_roots_only_picks_8(self) -> None:
        root = self.root / "Eclipse Adoptium"
        jdk17 = _fake_exe(root / "jdk-17.0.19.10-hotspot" / "bin", "java.exe")
        target = _fake_exe(root / "jdk8u402-b06" / "bin", "java.exe")
        choice = discover_java(roots=[str(root)], is_file=os.path.isfile,
                               version_probe=_runner({jdk17: 17, target: 8}))
        self.assertEqual(choice.path, target)
        # 目录名不含 8 的 jdk-17 被名字预筛掉，连探测都不做
        self.assertEqual(choice.rejected, ())

    def test_mislabeled_jdk8_dir_is_rejected_by_version(self) -> None:
        """目录名像 8 但实测是 17：必须拒掉并继续找真正的 8。"""
        root = self.root / "Java"
        fake = _fake_exe(root / "jdk8-old" / "bin", "java.exe")
        good = _fake_exe(root / "jdk8u402-b06" / "bin", "java.exe")
        choice = discover_java(roots=[str(root)], is_file=os.path.isfile,
                               version_probe=_runner({fake: 17, good: 8}))
        self.assertEqual(choice.path, good)
        self.assertIn((fake, 17), choice.rejected)

    def test_java_home_checked_for_version(self) -> None:
        home = self.root / "jdk17"
        exe = _fake_exe(home / "bin", "java.exe")
        choice = self._discover({exe: 17}, java_home=str(home))
        self.assertIsNone(choice.path)
        self.assertIn("检测到 Java 17", choice.hint)


class ConfigFileTests(unittest.TestCase):
    def test_missing_file_is_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_config_file(Path(tmp) / "none.json"), {})

    def test_broken_file_is_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_config_file(path), {})

    def test_reads_java_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"javaPath": "C:\\\\jdk8\\\\bin\\\\java.exe"}', encoding="utf-8")
            self.assertEqual(load_config_file(path)["javaPath"], "C:\\jdk8\\bin\\java.exe")


class ConfigExposureTests(unittest.TestCase):
    def test_no_java_exposes_null_and_hint(self) -> None:
        cfg = DashboardConfig(java_exe="", java_version=None, java_hint="")
        self.assertFalse(cfg.java_available)
        self.assertFalse(cfg.java_ready)
        public = cfg.public()
        self.assertIsNone(public["java"])
        self.assertIsNone(public["javaVersion"])
        self.assertIn("未找到 Java 8", public["javaHint"])

    def test_java17_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _fake_exe(Path(tmp) / "jdk17")
            cfg = DashboardConfig(java_exe=exe, java_version=17,
                                  java_hint="检测到 Java 17（x），Paper 1.8.8 需要 Java 8")
            self.assertTrue(cfg.java_available)
            self.assertFalse(cfg.java_ready)
            public = cfg.public()
            self.assertIsNone(public["java"], "非 8 不应被当作可用 java 暴露")
            self.assertEqual(public["javaVersion"], 17)
            self.assertIn("检测到 Java 17", public["javaHint"])

    def test_java8_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _fake_exe(Path(tmp) / "jdk8")
            cfg = DashboardConfig(java_exe=exe, java_version=8)
            self.assertTrue(cfg.java_ready)
            public = cfg.public()
            self.assertEqual(public["java"], exe)
            self.assertEqual(public["javaVersion"], 8)
            self.assertEqual(public["javaHint"], "")

    def test_health_payload_contains_java_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = _fake_exe(root / "jdk17")
            cfg = DashboardConfig(
                host="127.0.0.1", port=8080,
                artifacts_dir=root / "artifacts", var_dir=root / "var",
                server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
                training_dir=root / "training", python_exe=sys.executable,
                java_exe=exe, java_version=17, java_hint="检测到 Java 17（x）",
                node_exe="node", allow_process_control=False, scan_external=False,
                server_jar="missing.jar", mc_port=25990,
            )
            from app import DashboardApp, make_server
            app = DashboardApp(cfg)
            httpd = make_server(app, "127.0.0.1", 0)
            thread = threading.Thread(target=httpd.serve_forever,
                                      kwargs={"poll_interval": 0.1}, daemon=True)
            thread.start()
            try:
                with urlrequest.urlopen(
                    f"http://127.0.0.1:{httpd.server_address[1]}/api/health", timeout=10
                ) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                public = payload["config"]
                for key in ("java", "javaVersion", "javaHint"):
                    self.assertIn(key, public)
                self.assertIsNone(public["java"])
                self.assertEqual(public["javaVersion"], 17)
                self.assertIn("Java 17", public["javaHint"])
            finally:
                httpd.shutdown()
                httpd.server_close()
                app.close()
                thread.join(timeout=5)


class ServerStartGuardTests(unittest.TestCase):
    def _manager(self, root: Path, *, java_version: int | None) -> ServerManager:
        exe = _fake_exe(root / "bin")
        jar = root / "server" / "paper-1.8.8-445.jar"
        jar.parent.mkdir(parents=True, exist_ok=True)
        jar.write_bytes(b"fake jar")
        cfg = DashboardConfig(
            host="127.0.0.1", port=8080,
            artifacts_dir=root / "artifacts", var_dir=root / "var",
            server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training", python_exe=sys.executable,
            java_exe=exe, java_version=java_version, java_hint="检测到 Java 17（x）",
            node_exe="node", allow_process_control=True, scan_external=False,
            server_jar="paper-1.8.8-445.jar", mc_port=25989,
        )
        procs = ProcessManager(cfg.logs_dir, cfg.state_file)
        return ServerManager(cfg, procs)

    def test_rejects_java17_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp), java_version=17)
            with self.assertRaises(Conflict) as ctx:
                manager.start()
            message = str(ctx.exception)
            self.assertIn("未找到 Java 8", message)
            self.assertIn("Java 17", message)
            self.assertIsNone(manager.procs.get("paper"), "拒绝时不应 spawn 任何进程")

    def test_rejects_missing_java(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp), java_version=None)
            with self.assertRaises(Conflict) as ctx:
                manager.start()
            self.assertIn("未找到 Java 8", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
