"""P-3 脚本注册表：校验、CRUD、内置保护、备份、API 契约。"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from urllib import error, request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import BadRequest, NotFound  # noqa: E402
from scripts import BUILTIN_IDS, ScriptStore, normalize_color  # noqa: E402


class ColorTests(unittest.TestCase):
    def test_amp_and_section_normalized(self) -> None:
        self.assertEqual(normalize_color("&c"), "§c")
        self.assertEqual(normalize_color("§C"), "§c")
        self.assertEqual(normalize_color(" &a "), "§a")

    def test_rejects_unknown_or_multi_codes(self) -> None:
        for bad in ("", "c", "&z", "&l", "&c&l", "§", "red", "#ff0000"):
            with self.subTest(bad=bad), self.assertRaises(BadRequest):
                normalize_color(bad)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = ScriptStore(self.root / "scripts.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ensure_writes_builtins(self) -> None:
        self.store.ensure()
        items = self.store.items()
        self.assertEqual(len(items), len(BUILTIN_IDS))
        self.assertTrue(all(item["builtin"] for item in items))
        self.assertTrue((self.root / "scripts.json").is_file())

    def test_builtin_color_override(self) -> None:
        self.store.ensure()
        self.store.save({"id": "rusher", "name": "Rusher", "color": "&b", "description": "改色"})
        item = next(entry for entry in self.store.items() if entry["id"] == "rusher")
        self.assertEqual(item["color"], "§b")
        self.assertTrue(item["builtin"])
        self.assertEqual(self.store.color_for("rusher"), "§b")

    def test_custom_add_and_delete(self) -> None:
        self.store.ensure()
        self.store.save({"id": "mine-hold", "name": "Hold Center", "color": "&5", "description": "自定义"})
        item = next(entry for entry in self.store.items() if entry["id"] == "mine-hold")
        self.assertFalse(item["builtin"])
        self.assertEqual(self.store.delete("mine-hold"), "mine-hold")
        self.assertNotIn("mine-hold", [entry["id"] for entry in self.store.items()])

    def test_builtin_cannot_be_deleted(self) -> None:
        self.store.ensure()
        with self.assertRaises(BadRequest) as ctx:
            self.store.delete("rusher")
        self.assertIn("内置脚本不能删除", str(ctx.exception))

    def test_missing_delete_returns_not_found(self) -> None:
        self.store.ensure()
        with self.assertRaises(NotFound):
            self.store.delete("not-there")

    def test_validation_matrix(self) -> None:
        cases = [
            {"id": "Bad-Upper", "name": "x", "color": "&c"},
            {"id": "9bad", "name": "x", "color": "&c"},  # 允许数字开头（规则如此）
            {"id": "", "name": "x", "color": "&c"},
            {"id": "ok-id", "name": "", "color": "&c"},
            {"id": "ok-id", "name": "x" * 41, "color": "&c"},
            {"id": "ok-id", "name": "x", "color": "&l"},
            {"id": "ok-id", "name": "x", "color": "&c", "description": "d" * 201},
            {"id": "a" * 25, "name": "x", "color": "&c"},
        ]
        for payload in cases:
            expect_error = payload["id"] != "9bad"
            with self.subTest(payload=payload):
                if expect_error:
                    with self.assertRaises(BadRequest):
                        self.store.save(payload)
                else:
                    saved = self.store.save(payload)
                    self.assertEqual(saved["id"], "9bad")

    def test_backup_before_write(self) -> None:
        self.store.ensure()
        self.store.save({"id": "rusher", "name": "Rusher", "color": "&b"})
        backups = list((self.root / "scripts-trash").glob("scripts-*.json"))
        self.assertEqual(len(backups), 1, "写入前应备份旧表")
        data = json.loads(backups[0].read_text(encoding="utf-8"))
        self.assertEqual(data["format"], "gladiator.scripts/v1")

    def test_broken_file_falls_back_to_builtins(self) -> None:
        (self.root / "scripts.json").write_text("{", encoding="utf-8")
        items = self.store.items()
        self.assertEqual(len(items), len(BUILTIN_IDS))


class ScriptHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.root = root
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
            mc_port=25995,
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

    def test_get_lists_builtins(self) -> None:
        data = self._json("/api/scripts")
        ids = [item["id"] for item in data["scripts"]]
        for builtin in BUILTIN_IDS:
            self.assertIn(builtin, ids)
        self.assertEqual(len(data["colors"]), 16)
        self.assertTrue(all(item["builtin"] for item in data["scripts"]))

    def test_post_upsert_custom(self) -> None:
        saved = self._json("/api/scripts", "POST", {
            "id": "http-mine", "name": "HTTP Mine", "color": "&9", "description": "契约测试",
        })
        self.assertEqual(saved["script"]["color"], "§9")
        self.assertFalse(saved["script"]["builtin"])
        listing = self._json("/api/scripts")["scripts"]
        self.assertIn("http-mine", [item["id"] for item in listing])

    def test_post_invalid_returns_400(self) -> None:
        status, body = self._request("/api/scripts", "POST", {"id": "Bad", "name": "x", "color": "&c"})
        self.assertEqual(status, 400)
        self.assertIn("id", json.loads(body.decode("utf-8"))["error"])

    def test_delete_builtin_400_and_missing_404(self) -> None:
        status, body = self._request("/api/scripts?id=rusher", "DELETE")
        self.assertEqual(status, 400)
        self.assertIn("内置脚本不能删除", json.loads(body.decode("utf-8"))["error"])

        status, _body = self._request("/api/scripts?id=nope-nope", "DELETE")
        self.assertEqual(status, 404)

    def test_delete_custom(self) -> None:
        self._json("/api/scripts", "POST", {"id": "tmp-del", "name": "Tmp", "color": "&f"})
        deleted = self._json("/api/scripts?id=tmp-del", "DELETE")
        self.assertEqual(deleted["deleted"], "tmp-del")
        self.assertNotIn("tmp-del", [item["id"] for item in deleted["scripts"]])

    def test_options_use_registry_colors(self) -> None:
        self._json("/api/scripts", "POST", {"id": "rusher", "name": "Rusher", "color": "&b"})
        options = self._json("/api/training/options")["options"]
        entry = next(item for item in options["scripts"] if item["id"] == "rusher")
        self.assertEqual(entry["color"], "§b")


if __name__ == "__main__":
    unittest.main()
