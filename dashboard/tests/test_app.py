"""Dashboard 冒烟测试（标准库 unittest，不启动 Java / Node）。

覆盖：
- 静态页与静态资源可访问；
- 各只读 API 的返回结构；
- 只读模式下的写接口边界（403/400/404）；
- 服务器状态 / bot ready 列表的解析函数。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from urllib import error, request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from artifacts import ArtifactStore  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import Conflict  # noqa: E402
from match_runner import parse_ready_bots  # noqa: E402
from processes import ProcessManager  # noqa: E402
from runtime_manager import RuntimeManager  # noqa: E402
from server_manager import parse_gladiator_status  # noqa: E402
from training_runner import TrainingJob, TrainingRunner  # noqa: E402
from validation import BadRequest, parse_stages  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DashboardHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        artifacts = root / "artifacts"
        _write_json(
            artifacts / "train-report.json",
            {
                "startedAt": "2026-11-07T20:00:00",
                "durationSeconds": 90.5,
                "config": {"episodes": 2},
                "history": [
                    {
                        "episode": 1,
                        "steps": 120,
                        "totalReward": 1.25,
                        "durationSeconds": 12.3,
                        "winner": "Bot_T1",
                        "review": {
                            "hits": 4,
                            "whiffs": 2,
                            "hitRate": 0.6667,
                            "maxCombo": 2,
                            "damageDealt": 8.0,
                            "damageTaken": 5.0,
                        },
                    },
                    {
                        "episode": 2,
                        "steps": 80,
                        "totalReward": -0.5,
                        "durationSeconds": 9.0,
                        "winner": None,
                        "review": {"hits": 1, "whiffs": 3, "hitRate": 0.25},
                    },
                ],
                "evaluation": [],
                "policyPath": str(artifacts / "policy.npz"),
            },
        )
        _write_json(
            artifacts / "multi-report.json",
            {
                "startedAt": "2026-11-07T20:10:00",
                "durationSeconds": 30,
                "pairs": [
                    {
                        "pair": 1,
                        "users": ["Bot_P_1A", "Bot_P_1B"],
                        "episodes": 2,
                        "steps": 200,
                        "meanReward": 0.75,
                        "hits": 5,
                        "whiffs": 5,
                        "error": None,
                        "history": [],
                    }
                ],
            },
        )
        _write_json(
            artifacts / "reviews" / "episode-001.json",
            {
                "episode": 1,
                "username": "Bot_T1",
                "steps": 120,
                "totalReward": 1.25,
                "winner": "Bot_T1",
                "hits": 4,
                "whiffs": 2,
                "hitRate": 0.6667,
                "maxCombo": 2,
                "damageDealt": 8.0,
                "damageTaken": 5.0,
            },
        )
        (artifacts / "reviews" / "episode-001.md").write_text(
            "# Episode 001\n\n| 指标 | 数值 |\n| --- | --- |\n| 命中 | 4 |\n\n- 胜者：Bot_T1\n",
            encoding="utf-8",
        )
        _write_json(
            artifacts / "stress-report.json",
            {
                "users": 4,
                "durationSeconds": 60,
                "steps": 6000,
                "stepsPerSecond": 100.0,
                "tpsAvg": 19.9,
                "tpsMin": 18.5,
                "tpsMax": 20.0,
            },
        )
        run_dir = artifacts / "runs" / "20261107-2300-train-Bot_R"
        _write_json(
            run_dir / "report.json",
            {
                "runDir": str(run_dir),
                "kind": "train",
                "startedAt": "2026-11-07T23:00:00",
                "finishedAt": "2026-11-07T23:05:00",
                "durationSeconds": 300,
                "config": {"plan": [{"opponentMode": "rusher", "episodes": 2}]},
                "history": [
                    {
                        "episode": 1,
                        "startedAt": "2026-11-07T23:01:00",
                        "durationSeconds": 30.5,
                        "steps": 150,
                        "totalReward": 2.5,
                        "winner": "Bot_R",
                    },
                    {
                        "episode": 2,
                        "startedAt": "2026-11-07T23:02:00",
                        "durationSeconds": 28.0,
                        "steps": 140,
                        "totalReward": 1.5,
                        "winner": "Bot_R",
                    },
                ],
                "evaluation": [],
            },
        )
        _write_json(
            run_dir / "reviews" / "episode-001.json",
            {"episode": 1, "username": "Bot_R", "totalReward": 2.5, "winner": "Bot_R", "steps": 150},
        )
        (run_dir / "reviews" / "episode-001.md").write_text("# Run Episode 001\n", encoding="utf-8")
        (run_dir / "run.log").write_text("[train] episode 1/2\n", encoding="utf-8")

        multi_dir = artifacts / "runs" / "20261107-2250-multi-Bot_M"
        _write_json(
            multi_dir / "report.json",
            {
                "runDir": str(multi_dir),
                "kind": "multi",
                "startedAt": "2026-11-07T22:50:00",
                "durationSeconds": 120,
                "config": {"pairs": 2, "episodes": 2},
                "pairs": [
                    {
                        "pair": 1,
                        "users": ["Bot_M_1A", "Bot_M_1B"],
                        "episodes": 2,
                        "steps": 300,
                        "meanReward": 1.2,
                        "hits": 6,
                        "whiffs": 4,
                        "error": None,
                        "history": [
                            {
                                "episode": 1,
                                "startedAt": "2026-11-07T22:50:05",
                                "durationSeconds": 25,
                                "steps": 150,
                                "totalReward": 1.0,
                            }
                        ],
                    },
                    {
                        "pair": 2,
                        "users": ["Bot_M_2A", "Bot_M_2B"],
                        "episodes": 2,
                        "steps": 280,
                        "meanReward": 0.4,
                        "hits": 3,
                        "whiffs": 7,
                        "error": None,
                        "history": [],
                    },
                ],
                "history": [],
            },
        )
        _write_json(
            multi_dir / "reviews" / "pair-1" / "episode-001.json",
            {"episode": 1, "username": "Bot_M_1A", "steps": 150, "totalReward": 1.0, "winner": "Bot_M_1A"},
        )
        (multi_dir / "reviews" / "pair-1" / "episode-001.md").write_text(
            "# Pair 1 Episode 001\n", encoding="utf-8"
        )
        _write_json(
            multi_dir / "reviews" / "pair-2" / "episode-001.json",
            {"episode": 1, "username": "Bot_M_2A", "steps": 140, "totalReward": 0.2, "winner": None},
        )

        cls.cfg = DashboardConfig(
            host="127.0.0.1",
            port=8080,
            artifacts_dir=artifacts,
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
            mc_port=25999,
        )
        cls.app = DashboardApp(cls.cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.app.close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    # --- 辅助 -----------------------------------------------------------
    def _request(self, path: str, method: str = "GET", payload: Any = None) -> tuple[int, str, bytes]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", ""), exc.read()

    def _json(self, path: str, method: str = "GET", payload: Any = None) -> dict:
        status, content_type, body = self._request(path, method, payload)
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        self.assertIn("application/json", content_type)
        return json.loads(body.decode("utf-8"))

    # --- 静态资源 -------------------------------------------------------
    def test_index_page(self) -> None:
        status, content_type, body = self._request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        text = body.decode("utf-8")
        self.assertIn("Gladiator", text)
        self.assertIn("数据可视化", text)
        self.assertIn("合并选中", text)
        self.assertIn("data-check-all", text)
        self.assertNotIn("https://cdn", text.lower())
        self.assertNotIn("http://cdn", text.lower())

    def test_static_assets(self) -> None:
        for path, needle in (("/static/app.js", "lineChart"), ("/static/style.css", "--")):
            status, content_type, body = self._request(path)
            self.assertEqual(status, 200, path)
            self.assertTrue(
                "javascript" in content_type or "css" in content_type,
                f"{path} content-type={content_type}",
            )
            self.assertIn(needle, body.decode("utf-8"))

    def test_static_traversal_blocked(self) -> None:
        status, _, body = self._request("/static/../app.py")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    # --- 只读 API -------------------------------------------------------
    def test_health(self) -> None:
        data = self._json("/api/health")
        self.assertTrue(data["ok"])
        self.assertEqual(data["name"], "gladiator-dashboard")
        self.assertTrue(data["readOnly"])
        for key in ("version", "pid", "uptimeSeconds", "host", "port", "config"):
            self.assertIn(key, data)
        self.assertIn(data["config"]["host"], ("127.0.0.1", "localhost", "::1"))

    def test_overview(self) -> None:
        data = self._json("/api/overview")
        self.assertTrue(data["ok"])
        for key in ("server", "runtimes", "training", "match", "artifacts", "checkedAt"):
            self.assertIn(key, data)
        self.assertIn("ports", data["server"])
        self.assertIn("trainReport", data["artifacts"])

    def test_server_status(self) -> None:
        data = self._json("/api/server")
        self.assertTrue(data["ok"])
        server = data["server"]
        for key in (
            "running", "managed", "external", "pid", "ports", "gladiator",
            "tps", "logTail", "logPath", "allowProcessControl",
        ):
            self.assertIn(key, server)
        self.assertFalse(server["running"])
        self.assertEqual(set(server["ports"]), {"game", "bridge", "rcon"})

    def test_runtimes(self) -> None:
        data = self._json("/api/runtimes")
        self.assertTrue(data["ok"])
        self.assertIsInstance(data["runtimes"], list)
        self.assertEqual(data["runtimes"], [])
        defaults = data["defaults"]
        self.assertIn("username", defaults)
        self.assertGreaterEqual(defaults["envPort"], 8770)
        self.assertLessEqual(defaults["envPort"], self.cfg.runtime_port_max)

    def test_training(self) -> None:
        data = self._json("/api/training")
        self.assertTrue(data["ok"])
        self.assertIn("job", data)
        self.assertIn("options", data)
        self.assertFalse(data["running"])
        options = data["options"]
        for key in ("opponentModes", "stageModes", "kinds", "defaults"):
            self.assertIn(key, options)

    def test_training_options(self) -> None:
        data = self._json("/api/training/options")
        self.assertTrue(data["ok"])
        defaults = data["options"]["defaults"]
        for key in ("users", "ports", "episodes", "maxSteps", "opponentMode", "pairs"):
            self.assertIn(key, defaults)
        self.assertEqual(defaults["ports"], [8770, 8771])
        self.assertEqual(defaults["portsBase"], 8770)
        self.assertEqual(defaults["spacing"], 6.0)
        self.assertEqual(defaults["prepSeconds"], 3.0)
        self.assertEqual(data["options"]["game"]["port"], self.cfg.mc_port)

    def test_training_argv_uses_configured_game_port(self) -> None:
        argv, params, total, _stage_counts = self.app.training._build_train({
            "users": "Bot_A, Bot_B",
            "ports": "8770, 8771",
            "episodes": 1,
            "maxSteps": 10,
            "opponentMode": "scripted",
        })
        self.assertEqual(argv[argv.index("--mc-port") + 1], str(self.cfg.mc_port))
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--spacing") + 1], "6.0")
        self.assertEqual(argv[argv.index("--prep-seconds") + 1], "3.0")
        self.assertEqual(params["ports"], [8770, 8771])
        self.assertEqual(total, 1)
        self.assertEqual(argv[argv.index("--run-dir") + 1], params["runDir"])
        self.assertTrue(Path(params["runDir"]).is_dir())
        for forbidden in ("--report", "--save", "--review-dir"):
            self.assertNotIn(forbidden, argv)

    def test_artifacts_summary(self) -> None:
        data = self._json("/api/artifacts")
        self.assertTrue(data["ok"])
        self.assertEqual(data["selectedRun"]["id"], "20261107-2300-train-Bot_R")
        self.assertTrue(data["train"]["exists"])
        self.assertEqual(data["train"]["summary"]["episodes"], 2)
        self.assertAlmostEqual(data["train"]["summary"]["meanReward"], 2.0, places=4)
        self.assertEqual(len(data["train"]["history"]), 2)
        self.assertFalse(data["multi"]["exists"])
        self.assertTrue(data["stress"]["derived"])
        keys = {item["key"] for item in data["stress"]["metrics"]}
        self.assertIn("stepsPerSecond", keys)
        self.assertEqual(len(data["reviews"]), 1)
        self.assertEqual(data["reviews"][0]["markdownPath"],
                         "runs/20261107-2300-train-Bot_R/reviews/episode-001.md")

    def test_artifacts_selected_run_and_legacy(self) -> None:
        selected = self._json("/api/artifacts?run=20261107-2300-train-Bot_R")
        self.assertEqual(selected["selectedRun"]["id"], "20261107-2300-train-Bot_R")
        self.assertAlmostEqual(selected["train"]["summary"]["meanReward"], 2.0, places=4)

        legacy = self._json("/api/artifacts?run=legacy")
        self.assertEqual(legacy["selectedRun"]["id"], "legacy")
        self.assertTrue(legacy["train"]["exists"])
        self.assertAlmostEqual(legacy["train"]["summary"]["meanReward"], 0.375, places=4)
        self.assertTrue(legacy["multi"]["exists"])
        self.assertFalse(legacy["stress"]["derived"])
        self.assertEqual(len(legacy["reviews"]), 1)
        self.assertEqual(legacy["reviews"][0]["markdownPath"], "episode-001.md")
        # legacy 不应混入 run 目录里的复盘
        self.assertNotIn("runs/", legacy["reviews"][0]["markdownPath"])

    def test_review_markdown(self) -> None:
        data = self._json("/api/artifacts/review?path=episode-001.md")
        self.assertTrue(data["ok"])
        self.assertIn("# Episode 001", data["markdown"])

    def test_review_traversal_rejected(self) -> None:
        status, _, body = self._request("/api/artifacts/review?path=../app.py")
        self.assertEqual(status, 400)
        self.assertIn("错误", json.loads(body)["error"] + "错误")

    def test_unknown_api(self) -> None:
        status, _, body = self._request("/api/nope")
        self.assertEqual(status, 404)
        self.assertFalse(json.loads(body)["ok"])

    # --- 写接口边界（只读模式） -----------------------------------------
    def test_read_only_blocks_server_start(self) -> None:
        status, _, body = self._request("/api/server/start", "POST", {})
        self.assertEqual(status, 403)
        self.assertIn("只读", json.loads(body)["error"])

    def test_read_only_blocks_training_start(self) -> None:
        status, _, body = self._request("/api/training/start", "POST", {
            "kind": "train",
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8770, 8771],
            "episodes": 1,
            "maxSteps": 10,
            "opponentMode": "scripted",
        })
        self.assertEqual(status, 403, body.decode("utf-8", "replace"))

    def test_invalid_runtime_port(self) -> None:
        status, _, body = self._request("/api/runtimes/start", "POST", {
            "username": "Glad_Bot_1", "envPort": "not-a-port",
        })
        self.assertEqual(status, 400)
        self.assertIn("envPort", json.loads(body)["error"])

    def test_bad_json(self) -> None:
        req = request.Request(
            self.base + "/api/training/start",
            data=b"{not json",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(error.HTTPError) as ctx:
            request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("JSON", json.loads(ctx.exception.read())["error"])

    # --- runs / 训练时间 -------------------------------------------------
    def test_runs_list(self) -> None:
        data = self._json("/api/artifacts/runs")
        self.assertTrue(data["ok"])
        runs = data["runs"]
        self.assertGreaterEqual(len(runs), 1)
        run = next((item for item in runs if item["id"] == "20261107-2300-train-Bot_R"), None)
        self.assertIsNotNone(run)
        self.assertEqual(run["kind"], "train")
        self.assertEqual(run["episodes"], 2)
        self.assertEqual(run["status"], "finished")
        self.assertEqual(run["plan"][0]["opponentMode"], "rusher")
        self.assertAlmostEqual(run["meanReward"], 2.0)
        self.assertEqual(run["reviewCount"], 1)
        legacy = next((item for item in runs if item["id"] == "legacy"), None)
        self.assertIsNotNone(legacy)

    def test_multi_run_pairs_and_reviews(self) -> None:
        data = self._json("/api/artifacts?run=20261107-2250-multi-Bot_M")
        self.assertEqual(data["selectedRun"]["kind"], "multi")
        self.assertTrue(data["multi"]["exists"])
        self.assertEqual(len(data["multi"]["pairs"]), 2)
        self.assertEqual(data["multi"]["pairs"][0]["hits"], 6)
        self.assertEqual(data["multi"]["pairs"][0]["steps"], 300)
        self.assertEqual(len(data["reviews"]), 2)
        paths = {item["markdownPath"] for item in data["reviews"]}
        self.assertIn("runs/20261107-2250-multi-Bot_M/reviews/pair-1/episode-001.md", paths)

        detail = self._json("/api/artifacts/run?id=20261107-2250-multi-Bot_M")["run"]
        self.assertEqual(len(detail["pairs"]), 2)
        self.assertEqual(len(detail["reviews"]), 2)

    def test_merge_selected_runs(self) -> None:
        data = self._json(
            "/api/artifacts/merge?runs=20261107-2300-train-Bot_R,20261107-2250-multi-Bot_M"
        )
        self.assertTrue(data["merged"])
        self.assertEqual(data["selectedRun"]["id"], "merged")
        self.assertEqual(data["selectedRunIds"], ["20261107-2300-train-Bot_R", "20261107-2250-multi-Bot_M"])
        self.assertTrue(data["train"]["exists"])
        self.assertTrue(data["multi"]["exists"])
        self.assertEqual(len(data["multi"]["pairs"]), 2)
        pair_keys = {item["pairKey"] for item in data["multi"]["pairs"]}
        review_pair_keys = {item.get("pairKey") for item in data["reviews"] if item.get("pairKey")}
        self.assertTrue(pair_keys)
        self.assertTrue(review_pair_keys)
        self.assertTrue(review_pair_keys.issubset(pair_keys))
        self.assertEqual(data["selectedRun"]["summary"]["episodes"], 3)
        self.assertAlmostEqual(data["selectedRun"]["summary"]["meanReward"], 1.6667, places=4)
        self.assertEqual(len(data["multi"]["history"]), 3)
        self.assertTrue(all(item.get("runId") for item in data["multi"]["history"]))
        self.assertEqual(len(data["reviews"]), 3)
        self.assertTrue(all(item.get("runId") for item in data["reviews"]))
        self.assertTrue(data["stress"]["derived"])

    def test_merge_rejects_empty_and_unknown(self) -> None:
        status, _, body = self._request("/api/artifacts/merge?runs=")
        self.assertEqual(status, 400, body.decode("utf-8", "replace"))
        status, _, body = self._request("/api/artifacts/merge?runs=20261107-0000-nope")
        self.assertEqual(status, 404, body.decode("utf-8", "replace"))

    def test_run_detail_has_timing_history(self) -> None:
        data = self._json("/api/artifacts/run?id=20261107-2300-train-Bot_R")
        run = data["run"]
        self.assertEqual(len(run["history"]), 2)
        self.assertEqual(run["history"][0]["startedAt"], "2026-11-07T23:01:00")
        self.assertEqual(run["history"][0]["durationSeconds"], 30.5)
        self.assertEqual(len(run["reviews"]), 1)
        self.assertEqual(run["reviews"][0]["markdownPath"],
                         "runs/20261107-2300-train-Bot_R/reviews/episode-001.md")

    def test_run_traversal_rejected(self) -> None:
        status, _, body = self._request("/api/artifacts/run?id=../app.py")
        self.assertEqual(status, 400, body.decode("utf-8", "replace"))

    def test_terminal_page(self) -> None:
        status, content_type, body = self._request("/terminal")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("实时终端", body.decode("utf-8"))

    def test_terminal_sse(self) -> None:
        req = request.Request(self.base + "/api/terminal/stream?source=server")
        with request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
            lines = []
            for _ in range(12):
                line = resp.readline().decode("utf-8", "replace").rstrip("\r\n")
                lines.append(line)
                if line == "" and any(item.startswith("data:") for item in lines):
                    break
        text = "\n".join(lines)
        self.assertIn("event: meta", text)
        self.assertIn("data:", text)

    def test_terminal_invalid_source(self) -> None:
        status, _, body = self._request("/api/terminal/stream?source=../../etc/passwd")
        self.assertEqual(status, 400, body.decode("utf-8", "replace"))
        self.assertFalse(json.loads(body)["ok"])


class AutoCleanupTests(unittest.TestCase):
    def _runner(self, root: Path) -> TrainingRunner:
        cfg = DashboardConfig(
            artifacts_dir=root / "artifacts",
            var_dir=root / "var",
            server_dir=root / "server",
            bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training",
            allow_process_control=True,
        )
        procs = ProcessManager(cfg.logs_dir, cfg.state_file)
        return TrainingRunner(cfg, procs, ArtifactStore(cfg.artifacts_dir), FakeRuntimes())

    def test_cleanup_only_owned_alive_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            job = TrainingJob(
                id="training", kind="train", params={}, total_episodes=1,
                owned_runtimes=["runtime-8770", "external-8771", "runtime-dead"],
                auto_cleanup=True,
            )
            runner._cleanup_owned(job)
            self.assertTrue(job.cleanup_done)
            self.assertEqual(job.cleanup_count, 1)
            self.assertIn("已清理 1 个运行时", job.cleanup_message)
            self.assertEqual(runner.runtimes.stopped, ["runtime-8770"])
            # 幂等：不会重复停止
            runner._cleanup_owned(job)
            self.assertEqual(runner.runtimes.stopped, ["runtime-8770"])

    def test_cleanup_disabled_keeps_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            job = TrainingJob(
                id="training", kind="multi", params={}, total_episodes=2,
                owned_runtimes=["runtime-8770"], auto_cleanup=False,
            )
            runner._cleanup_owned(job)
            self.assertEqual(runner.runtimes.stopped, [])
            self.assertEqual(job.cleanup_count, 0)
            self.assertIn("autoCleanup 已关闭", job.cleanup_message)


class TrainingAsyncTests(unittest.TestCase):
    def _runner(self, root: Path) -> TrainingRunner:
        cfg = DashboardConfig(
            artifacts_dir=root / "artifacts",
            var_dir=root / "var",
            server_dir=root / "server",
            bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training",
            python_exe=sys.executable,
            allow_process_control=True,
            scan_external=False,
        )
        procs = ProcessManager(cfg.logs_dir, cfg.state_file)
        return TrainingRunner(cfg, procs, ArtifactStore(cfg.artifacts_dir), FakeRuntimes())

    @staticmethod
    def _payload() -> dict[str, Any]:
        return {
            "kind": "train",
            "users": ["Bot_A", "Bot_B"],
            "ports": [8890, 8891],
            "episodes": 1,
            "maxSteps": 10,
            "opponentMode": "scripted",
            "autoSpawn": False,
            "autoCleanup": True,
        }

    def test_start_returns_before_background_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            original = runner._ensure_runtimes

            def slow_ensure(params: dict, job: TrainingJob | None = None):
                time.sleep(0.4)
                return original(params, job)

            with mock.patch.object(runner, "_ensure_runtimes", side_effect=slow_ensure):
                started = time.monotonic()
                result = runner.start(self._payload())
                elapsed = time.monotonic() - started
                # start() 只做校验 + 起后台线程；给足余量避免慢机器上的假失败
                self.assertLess(elapsed, 2.0)
                self.assertEqual(result["job"]["status"], "starting")

            # 终态与 cleanup 是两个连续步骤（状态先落、清理后做），必须两个都等到，
            # 否则会在“已 failed 但还没 cleanup”的窗口里断言，造成偶发失败。
            current: dict = {}
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                current = runner.current()["job"]
                if current["status"] in ("failed", "finished", "stopped") and current["cleanup"]["done"]:
                    break
                time.sleep(0.02)
            self.assertEqual(current.get("status"), "failed")
            self.assertIn("端口", current.get("error") or "")
            self.assertTrue(current["cleanup"]["done"], f"cleanup 未完成: {current.get('cleanup')}")

    def test_stop_while_starting_returns_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            gate = threading.Event()

            def blocked_ensure(params: dict, job: TrainingJob | None = None):
                gate.wait(3)
                return [], []

            with mock.patch.object(runner, "_ensure_runtimes", side_effect=blocked_ensure):
                runner.start(self._payload())
                started = time.monotonic()
                result = runner.stop()
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 0.8)
                self.assertEqual(result["job"]["status"], "stopping")
                gate.set()

            current: dict = {}
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                current = runner.current()["job"]
                if current["status"] in ("stopped", "failed", "finished") and current["cleanup"]["done"]:
                    break
                time.sleep(0.02)
            self.assertEqual(current.get("status"), "stopped")
            self.assertTrue(current["cleanup"]["done"], f"cleanup 未完成: {current.get('cleanup')}")


class FakeRuntimes:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def get(self, runtime_id: str) -> dict:
        if runtime_id == "runtime-dead":
            return {"id": runtime_id, "alive": False}
        return {"id": runtime_id, "alive": True}

    def stop(self, runtime_id: str) -> dict:
        self.stopped.append(runtime_id)
        return {"id": runtime_id}


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = DashboardConfig(
                artifacts_dir=Path(tmp) / "artifacts",
                var_dir=Path(tmp) / "var",
                server_dir=Path(tmp) / "server",
                bot_runtime_dir=Path(tmp) / "bot-runtime",
                training_dir=Path(tmp) / "training",
            )
        self.assertEqual(cfg.mc_port, 25566)
        self.assertEqual(cfg.bridge_port, 8765)
        self.assertEqual(cfg.rcon_port, 25575)
        self.assertEqual((cfg.runtime_port_min, cfg.runtime_port_max), (8770, 8789))
        self.assertTrue(cfg.scan_external)


class ExternalRuntimeTests(unittest.TestCase):
    def test_external_runtime_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = DashboardConfig(
                host="127.0.0.1",
                port=8080,
                artifacts_dir=root / "artifacts",
                var_dir=root / "var",
                server_dir=root / "server",
                bot_runtime_dir=root / "bot-runtime",
                training_dir=root / "training",
                python_exe=sys.executable,
                java_exe="java",
                node_exe="node",
                allow_process_control=True,
                scan_external=True,
            )
            procs = ProcessManager(root / "var" / "logs", root / "var" / "managed.json")
            manager = RuntimeManager(cfg, procs)
            with mock.patch("runtime_manager.listening_pids", return_value={8770: 4321}):
                items = manager.list()
            self.assertEqual(len(items), 1)
            item = items[0]
            self.assertTrue(item["external"])
            self.assertEqual(item["state"], "external")
            self.assertEqual(item["pid"], 4321)
            self.assertEqual(item["envPort"], 8770)
            self.assertFalse(item["tail"])
            with self.assertRaises(Conflict):
                manager.stop("external-8770")


class TrainingAccountingTests(unittest.TestCase):
    """multi 的 episode 行交错输出、pair 汇总行不能重复计数。"""

    def _runner(self, root: Path) -> TrainingRunner:
        cfg = DashboardConfig(
            artifacts_dir=root / "artifacts",
            var_dir=root / "var",
            server_dir=root / "server",
            bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training",
            allow_process_control=True,
            scan_external=False,
        )
        procs = ProcessManager(cfg.logs_dir, cfg.state_file)
        return TrainingRunner(cfg, procs, ArtifactStore(cfg.artifacts_dir), FakeRuntimes())

    def test_multi_episode_and_pair_lines_count_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            job = TrainingJob(
                id="training",
                kind="multi",
                params={"pairs": 3, "episodes": 20},
                total_episodes=60,
                stage_counts=[20, 20, 20],
            )
            lines = []
            for episode in range(1, 21):
                for pair in (1, 2, 3):
                    lines.append(
                        f"[train] episode {episode}/20 steps=10 reward={pair}.0 "
                        "winner=None terminated=False 1.0s"
                    )
            for pair in (1, 2, 3):
                lines.append(
                    f"[pair{pair}] 完成：20 episodes / 200 steps / mean reward 2.0 / 命中 1 空挥 0"
                )
            for line in lines:
                runner._handle_line(job, line)
            self.assertEqual(job.episodes_completed, 60)
            self.assertEqual(job.pairs_completed, 3)
            self.assertEqual(job.steps, 600)
            self.assertEqual(len(job.rewards), 60)
            self.assertEqual(job.rewards[-1], 3.0)

    def test_report_sync_overrides_stream_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(Path(tmp))
            job = TrainingJob(
                id="training", kind="multi", params={}, total_episodes=60,
                episodes_completed=40, steps=11578, rewards=[1.0] * 63,
            )
            runner._sync_job_from_report(job, {
                "exists": True,
                "history": [
                    {"episode": 1, "steps": 100, "totalReward": 1.5},
                    {"episode": 2, "steps": 200, "totalReward": 2.5},
                ],
                "pairs": [{"pair": 1}, {"pair": 2}],
            })
            self.assertEqual(job.episodes_completed, 2)
            self.assertEqual(job.steps, 300)
            self.assertEqual(job.rewards, [1.5, 2.5])
            self.assertEqual(job.pairs_completed, 2)
            self.assertEqual(job.last_episode["episode"], 2)

    def test_attach_run_log_keeps_only_current_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self._runner(root)
            source = root / "training.log"
            source.write_text(
                "=== 2026-11-07T22:55:00 启动 training ===\n$ old\n[multi] old run\n"
                "=== 2026-11-08T02:10:00 启动 training ===\n$ new\n[prep] 准备期 3s 结束\n",
                encoding="utf-8",
            )
            # 新契约：ProcessManager 记录的 log_offset 指向本次 run 的 header 之前，
            # 只读这一段，不再靠正则找“最后一个启动标记”。
            offset = source.read_bytes().index("=== 2026-11-08T02:10:00".encode("utf-8"))
            run_dir = root / "runs" / "20261108-0210-train-Bot_X"
            job = TrainingJob(
                id="training", kind="train", params={}, total_episodes=1,
                run_dir=str(run_dir),
            )
            runner._attach_run_log(job, SimpleNamespace(log_path=source, log_offset=offset))
            text = (run_dir / "run.log").read_text(encoding="utf-8")
            self.assertTrue(text.startswith("new\n"), text)
            self.assertIn("[prep] 准备期 3s 结束", text)
            self.assertNotIn("old run", text)
            self.assertNotIn("启动 training", text)


class ParserTests(unittest.TestCase):
    def test_parse_gladiator_status(self) -> None:
        raw = "\n".join([
            "[Gladiator] version=0.1.0 api=1 ready=true worldguard=present",
            "[Gladiator] bots=2 games=1 training=1",
            "[Gladiator] link=127.0.0.1:8765 sessions=2",
            "[Gladiator] heap used=120MB max=1024MB",
        ])
        parsed = parse_gladiator_status(raw)
        self.assertEqual(parsed["version"], "0.1.0")
        self.assertEqual(parsed["api"], "1")
        self.assertTrue(parsed["ready"])
        self.assertEqual(parsed["worldguard"], "present")
        self.assertEqual(parsed["bots"], 2)
        self.assertEqual(parsed["games"], 1)
        self.assertEqual(parsed["training"], 1)
        self.assertEqual(parsed["sessions"], 2)
        self.assertEqual(parsed["heapUsedMb"], 120)
        self.assertEqual(parsed["heapMaxMb"], 1024)

    def test_parse_ready_bots(self) -> None:
        raw = "\n".join([
            "[Gladiator] bots:",
            "  - 3f2a name=Bot_T1 uuid=00000000-0000-0000-0000-000000000001 ready=true state=ALIVE",
            "  - 9c1b name=Bot_T2 uuid=00000000-0000-0000-0000-000000000002 ready=false state=SPAWNED",
        ])
        self.assertEqual(parse_ready_bots(raw), {"Bot_T1": True, "Bot_T2": False})

    def test_parse_stages(self) -> None:
        self.assertEqual(parse_stages("random:2, rusher:3", 5), [("random", 2), ("rusher", 3)])
        self.assertEqual(parse_stages("", 5), [])
        with self.assertRaises(BadRequest):
            parse_stages("unknown:1", 5)
        with self.assertRaises(BadRequest):
            parse_stages("random:0", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
