"""keep-alive 连接上的请求体消费回归。

历史 bug：同一连接上 ``POST /api/server/stop`` 之后紧跟 ``GET /api/server``，
服务端回 ``501 Unsupported method ('{}GET')`` —— body 没被读走，残留字节被当成下一行请求行。

根因有两个，这里各覆盖一条：
1. ``BaseHTTPRequestHandler`` 复用 handler 实例，旧的 ``_body_cache`` 命中导致本次 body 不读；
2. ``Transfer-Encoding: chunked`` 没有 Content-Length，旧实现直接当空 body。
"""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402


class KeepAliveTests(unittest.TestCase):
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
            # 只读模式：`/api/server/stop` 直接 403，正好是“handler 不读 body”的路径。
            allow_process_control=False,
            scan_external=False,
            server_jar="missing.jar",
            mc_port=25988,
        )
        cls.app = DashboardApp(cls.cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
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

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)

    def test_post_then_get_on_same_connection(self) -> None:
        """用户复现路径：POST /api/server/stop（body {}）→ GET /api/server。"""
        conn = self._conn()
        try:
            conn.request("POST", "/api/server/stop", body=b"{}",
                         headers={"Content-Type": "application/json"})
            first = conn.getresponse()
            first_body = first.read()
            self.assertEqual(first.status, 403, first_body[:200])

            conn.request("GET", "/api/server")
            second = conn.getresponse()
            second_body = second.read()
            self.assertEqual(second.status, 200, second_body[:200])
            self.assertTrue(json.loads(second_body.decode("utf-8"))["ok"])
        finally:
            conn.close()

    def test_post_start_then_get_on_same_connection(self) -> None:
        conn = self._conn()
        try:
            conn.request("POST", "/api/server/start", body=b'{"why": "test"}',
                         headers={"Content-Type": "application/json"})
            first = conn.getresponse()
            self.assertEqual(first.status, 403)
            first.read()  # 必须读完，否则同一连接上无法发下一个请求

            conn.request("GET", "/api/overview")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertTrue(payload["ok"])
        finally:
            conn.close()

    def test_second_post_uses_its_own_body(self) -> None:
        """同连接两个 POST：第二个必须读自己的 body（缓存未重置时会读到上一个）。"""
        conn = self._conn()
        try:
            for name in ("KA-First", "KA-Second"):
                body = json.dumps({"name": name,
                                   "plan": [{"opponentMode": "idle", "episodes": 1}]}).encode("utf-8")
                conn.request("POST", "/api/presets/save", body=body,
                             headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200, payload)
                self.assertEqual(payload["preset"]["name"], name)

            conn.request("GET", "/api/presets")
            listing = json.loads(conn.getresponse().read().decode("utf-8"))
            names = {item["name"] for item in listing["presets"]}
            self.assertIn("KA-First", names)
            self.assertIn("KA-Second", names)
        finally:
            conn.close()

    def test_chunked_post_then_get(self) -> None:
        """chunked body（无 Content-Length）：必须读完，否则残留分块会污染下一个请求。"""
        conn = self._conn()
        try:
            def chunks():
                yield b'{"name": "KA-Chunked", '
                yield b'"plan": [{"opponentMode": "idle", "episodes": 2}]}'

            conn.request("POST", "/api/presets/save", body=chunks(),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200, payload)
            self.assertEqual(payload["preset"]["name"], "KA-Chunked")
            self.assertEqual(payload["preset"]["totalEpisodes"], 2)

            conn.request("GET", "/api/presets")
            second = conn.getresponse()
            listing = json.loads(second.read().decode("utf-8"))
            self.assertEqual(second.status, 200)
            self.assertIn("KA-Chunked", {item["name"] for item in listing["presets"]})
        finally:
            conn.close()

    def test_many_requests_on_one_connection(self) -> None:
        """连续 10 个请求（POST/GET 混合）都在同一连接上成功。"""
        conn = self._conn()
        try:
            for index in range(5):
                body = json.dumps({"name": f"KA-{index}",
                                   "plan": [{"opponentMode": "idle", "episodes": 1}]}).encode("utf-8")
                conn.request("POST", "/api/presets/save", body=body,
                             headers={"Content-Type": "application/json"})
                post = conn.getresponse()
                post.read()
                self.assertEqual(post.status, 200, f"第 {index} 个 POST 失败")
                conn.request("GET", "/api/health")
                get = conn.getresponse()
                get.read()
                self.assertEqual(get.status, 200, f"第 {index} 个 GET 失败")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
