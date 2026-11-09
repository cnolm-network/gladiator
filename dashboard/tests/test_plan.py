"""训练计划与预设的测试（纯标准库 unittest）。

覆盖：
- ``parse_plan`` 的三种输入形态与非法值；
- ``PresetStore`` 的保存/覆盖/列表/加载/删除/字段白名单；
- HTTP 层的 ``/api/presets*`` 与 ``/api/training/options`` 新字段。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import BadRequest, NotFound  # noqa: E402
from presets import PresetStore  # noqa: E402
from validation import format_plan, parse_plan  # noqa: E402

from urllib import error, request  # noqa: E402


class ParsePlanTests(unittest.TestCase):
    def test_array_of_objects(self) -> None:
        plan = parse_plan([{"opponentMode": "rusher", "episodes": 3}])
        self.assertEqual(plan, [("rusher", 3)])

    def test_array_of_pairs(self) -> None:
        self.assertEqual(parse_plan([["idle", 2], ["scripted", 1]]), [("idle", 2), ("scripted", 1)])

    def test_legacy_string(self) -> None:
        self.assertEqual(parse_plan("idle:2, rusher:3"), [("idle", 2), ("rusher", 3)])

    def test_missing_uses_fallback(self) -> None:
        self.assertEqual(
            parse_plan(None, fallback_mode="wander", fallback_episodes=5),
            [("wander", 5)],
        )
        self.assertEqual(
            parse_plan("  ", fallback_mode="wander", fallback_episodes=5),
            [("wander", 5)],
        )

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            parse_plan([{"opponentMode": "human", "episodes": 1}])

    def test_non_positive_episodes_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            parse_plan([{"opponentMode": "idle", "episodes": 0}])
        with self.assertRaises(BadRequest):
            parse_plan([{"opponentMode": "idle", "episodes": "x"}])

    def test_empty_array_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            parse_plan([])

    def test_too_many_items_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            parse_plan([{"opponentMode": "idle", "episodes": 1}] * 51)

    def test_total_episodes_capped(self) -> None:
        with self.assertRaises(BadRequest):
            parse_plan([{"opponentMode": "idle", "episodes": 500}] * 11)

    def test_format_round_trip(self) -> None:
        plan = [("scripted", 10), ("rusher", 10)]
        self.assertEqual(format_plan(plan), "scripted:10,rusher:10")
        self.assertEqual(parse_plan(format_plan(plan)), plan)


class PresetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PresetStore(Path(self.tmp.name) / "presets")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_list(self) -> None:
        preset = self.store.save({
            "name": "Preset-1",
            "plan": [{"opponentMode": "scripted", "episodes": 10}],
            "fields": {"maxSteps": 200, "seed": 7, "unknown": "drop"},
        })
        self.assertEqual(preset["name"], "Preset-1")
        self.assertEqual(preset["totalEpisodes"], 10)
        self.assertEqual(preset["planText"], "scripted:10")
        self.assertEqual(preset["fields"], {"maxSteps": 200, "seed": 7})

        items = self.store.list()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], preset["id"])

    def test_same_name_overwrites_and_keeps_created_at(self) -> None:
        first = self.store.save({"name": "P", "plan": [{"opponentMode": "idle", "episodes": 1}]})
        second = self.store.save({"name": "P", "plan": [{"opponentMode": "idle", "episodes": 2}]})
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["totalEpisodes"], 2)
        self.assertEqual(first["createdAt"], second["createdAt"])
        self.assertEqual(len(self.store.list()), 1)

    def test_load_and_delete(self) -> None:
        preset = self.store.save({"name": "P2", "plan": [{"opponentMode": "rusher", "episodes": 4}]})
        loaded = self.store.load(preset["id"])
        self.assertEqual(loaded["plan"], [{"opponentMode": "rusher", "episodes": 4}])
        self.assertEqual(self.store.delete(preset["id"]), preset["id"])
        with self.assertRaises(NotFound):
            self.store.load(preset["id"])

    def test_delete_moves_to_trash(self) -> None:
        preset = self.store.save({"name": "P4", "plan": [{"opponentMode": "idle", "episodes": 1}]})
        self.store.delete(preset["id"])
        trash = Path(self.tmp.name) / "presets-trash"
        files = list(trash.glob(f"{preset['id']}-*.json"))
        self.assertEqual(len(files), 1, "删除应留下可恢复副本")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "P4")

    def test_invalid_id_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            self.store.load("../app")
        with self.assertRaises(BadRequest):
            self.store.delete("a/b")

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(BadRequest):
            self.store.save({"name": "  ", "plan": [{"opponentMode": "idle", "episodes": 1}]})

    def test_json_shape(self) -> None:
        preset = self.store.save({"name": "P3", "plan": [{"opponentMode": "idle", "episodes": 3}]})
        path = Path(self.tmp.name) / "presets" / f"{preset['id']}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["format"], "gladiator.training-preset/v1")
        self.assertEqual(data["plan"], [{"opponentMode": "idle", "episodes": 3}])


class PresetHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.cfg = DashboardConfig(
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
            allow_process_control=False,
            scan_external=False,
            server_jar="missing.jar",
            mc_port=25998,
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

    def _request(self, path: str, method: str = "GET", payload: Any = None) -> tuple[int, bytes]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()

    def _json(self, path: str, method: str = "GET", payload: Any = None) -> dict:
        status, body = self._request(path, method, payload)
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        return json.loads(body.decode("utf-8"))

    def test_options_exposes_scripts_and_plan(self) -> None:
        data = self._json("/api/training/options")
        options = data["options"]
        self.assertIn("plan", options["defaults"])
        self.assertEqual(options["defaults"]["plan"], [{"opponentMode": "scripted", "episodes": 4}])
        self.assertTrue(any(item["id"] == "rusher" for item in options["scripts"]))
        # G-622：模型注册表已接入，default 表示「全新初始化」。
        self.assertTrue(options["modelSupport"])
        self.assertIn("default", [item["id"] for item in options["models"]])

    def test_presets_crud(self) -> None:
        self.assertEqual(self._json("/api/presets")["presets"], [])
        saved = self._json("/api/presets/save", "POST", {
            "name": "Http-1",
            "plan": [{"opponentMode": "scripted", "episodes": 2},
                     {"opponentMode": "rusher", "episodes": 3}],
        })["preset"]
        self.assertEqual(saved["totalEpisodes"], 5)
        listed = self._json("/api/presets")["presets"]
        self.assertEqual(len(listed), 1)
        loaded = self._json("/api/presets/load", "POST", {"id": saved["id"]})["preset"]
        self.assertEqual(loaded["planText"], "scripted:2,rusher:3")
        deleted = self._json("/api/presets/delete", "POST", {"id": saved["id"]})
        self.assertEqual(deleted["deleted"], saved["id"])
        self.assertEqual(self._json("/api/presets")["presets"], [])

    def test_invalid_plan_returns_400(self) -> None:
        status, body = self._request("/api/presets/save", "POST", {
            "name": "bad",
            "plan": [{"opponentMode": "human", "episodes": 1}],
        })
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body.decode("utf-8")))

    def test_missing_preset_returns_404(self) -> None:
        status, body = self._request("/api/presets/load", "POST", {"id": "nope-000000"})
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body.decode("utf-8")))


class TrainingEpochTests(unittest.TestCase):
    """回合（epochs）→ 计划（plan）→ 轮次（episodes）三层的命令映射。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = DashboardConfig(
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
            allow_process_control=False,
            scan_external=False,
            server_jar="missing.jar",
            mc_port=25996,
        )
        self.app = DashboardApp(self.cfg)

    def tearDown(self) -> None:
        self.app.close()
        self.tmp.cleanup()

    def _arg(self, argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def test_train_argv_carries_epochs(self) -> None:
        argv, params, total, counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8770, 8771],
            "plan": [{"opponentMode": "rusher", "episodes": 5},
                     {"opponentMode": "scripted", "episodes": 5}],
            "epochs": 2,
            "evalEpochs": 1,
        })
        self.assertEqual(self._arg(argv, "--stages"), "rusher:5,scripted:5")
        self.assertEqual(self._arg(argv, "--epochs"), "2")
        self.assertEqual(self._arg(argv, "--eval-epochs"), "1")
        self.assertEqual(params["episodesPerEpoch"], 10)
        self.assertEqual(params["epochs"], 2)
        self.assertEqual(params["totalEpisodes"], 20)
        self.assertEqual(total, 20)
        self.assertEqual(counts, [5, 5])

    def test_train_defaults_keep_one_epoch(self) -> None:
        argv, params, total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8770, 8771],
            "plan": [{"opponentMode": "idle", "episodes": 3}],
        })
        self.assertEqual(self._arg(argv, "--epochs"), "1")
        self.assertEqual(self._arg(argv, "--eval-epochs"), "1")
        self.assertEqual(total, 3)

    def test_multi_epochs_multiplies_pairs(self) -> None:
        argv, params, total, _counts = self.app.training._build_multi({
            "pairs": 2,
            "usersPrefix": "Bot_P",
            "portsBase": 8790,
            "plan": [{"opponentMode": "idle", "episodes": 3}],
            "epochs": 2,
        })
        self.assertEqual(self._arg(argv, "--epochs"), "2")
        self.assertEqual(params["episodesPerEpoch"], 3)
        self.assertEqual(params["episodes"], 6)
        self.assertEqual(total, 12)

    def test_rejects_zero_epochs(self) -> None:
        with self.assertRaises(BadRequest):
            self.app.training._build_train({
                "users": ["Bot_T1", "Bot_T2"],
                "ports": [8770, 8771],
                "plan": [{"opponentMode": "idle", "episodes": 1}],
                "epochs": 0,
            })


if __name__ == "__main__":
    unittest.main()
