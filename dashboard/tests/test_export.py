"""G-626：把策略模型导出为 ``models/<id>.gmod`` 并回扫。

覆盖：导出成功（包内只有 meta.json + policy.npz、meta 字段齐全、sha256 与权重文件一致）、
未知 id 404、目标已存在 409 / overwrite 覆盖、无 policy.npz 400、query 形式 `?id=`、
导出后 `GET /api/models` 扫描为 valid 且带 training 累计。

全部使用临时目录 + 手写 npz；不依赖 numpy、不碰真实 ``dashboard/var`` / ``models/``。
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from urllib import error, request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from tests.test_gmod import npz_bytes  # noqa: E402


class ExportGmodHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.gmod_dir = root / "models"
        cls.gmod_dir.mkdir(parents=True, exist_ok=True)
        cls.cfg = DashboardConfig(
            host="127.0.0.1", port=8080,
            artifacts_dir=root / "artifacts", var_dir=root / "var",
            server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training", gmod_dir=cls.gmod_dir,
            python_exe=sys.executable, java_exe="java", node_exe="node",
            allow_process_control=False, scan_external=False,
            server_jar="missing.jar", mc_port=25962,
        )
        cls.app = DashboardApp(cls.cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                     kwargs={"poll_interval": 0.1}, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

        # default 模型：权重 + 训练累计（模拟一次持久化训练后的状态）
        models = cls.cfg.var_dir / "models"
        cls.default_dir = models / "default"
        cls.default_dir.mkdir(parents=True, exist_ok=True)
        cls.policy = cls.default_dir / "policy.npz"
        cls.policy.write_bytes(npz_bytes())
        (cls.default_dir / "training.json").write_text(json.dumps({
            "format": "gladiator.model-training/v1",
            "lastRunId": "20261109-0107-train-Bot_D1",
            "lastTrainedAt": "2026-11-09T01:07:30",
            "totalEpisodes": 12,
            "totalSteps": 2400,
            "lastMeanReward": 3.25,
            "writeCount": 3,
            "runs": [{"runId": "20261109-0107-train-Bot_D1", "at": "2026-11-09T01:07:30",
                      "episodes": 12, "steps": 2400, "meanReward": 3.25}],
        }, ensure_ascii=False), encoding="utf-8")

        # 一个没有任何权重的模型（导出应 400）
        empty = models / "nopolicy"
        empty.mkdir(parents=True, exist_ok=True)
        (empty / "model.json").write_text(json.dumps({
            "format": "gladiator.model/v1", "id": "nopolicy", "name": "No Policy",
            "description": "", "createdAt": "2026-11-09T00:00:00", "updatedAt": "2026-11-09T00:00:00",
            "source": {"runId": None, "runName": None, "pair": None},
            "obsDim": 19, "numActions": 11, "hidden": 64,
            "episodes": 0, "steps": 0, "meanReward": None, "sha256": None, "note": "",
            "builtin": False,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.app.close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    # --- 辅助 -----------------------------------------------------------
    def _post(self, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else b""
        req = request.Request(self.base + path, data=data, method="POST",
                              headers={"Content-Type": "application/json"} if data else {})
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _get(self, path: str) -> tuple[int, dict]:
        try:
            with request.urlopen(self.base + path, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _target(self, ident: str = "default") -> Path:
        return self.gmod_dir / f"{ident}.gmod"

    def _remove_target(self, ident: str = "default") -> None:
        target = self._target(ident)
        if target.exists():
            target.unlink()

    # --- 成功 / 元数据 ---------------------------------------------------
    def test_export_creates_package_with_full_meta(self) -> None:
        self._remove_target()
        status, data = self._post("/api/models/export", {"id": "default"})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["ok"])
        self.assertEqual(data["id"], "default")
        target = Path(data["path"])
        self.assertEqual(target, self._target())
        self.assertTrue(target.is_file())
        self.assertEqual(data["sizeBytes"], target.stat().st_size)
        self.assertEqual(len(data["sha256"]), 64)
        self.assertEqual(data["relativePath"], "models/default.gmod")

        with zipfile.ZipFile(target) as archive:
            self.assertEqual(sorted(archive.namelist()), ["meta.json", "policy.npz"],
                             "包内只允许 meta.json + policy.npz")
            blob = archive.read("policy.npz")
            meta = json.loads(archive.read("meta.json").decode("utf-8"))

        self.assertEqual(hashlib.sha256(blob).hexdigest(), meta["sha256"],
                         "meta.sha256 必须等于 policy.npz 内容哈希")
        self.assertEqual(blob, self.policy.read_bytes(), "包内权重应是注册表副本")

        for key in ("id", "name", "description", "color", "obsDim", "actDim", "hidden",
                    "version", "createdAt", "exportedAt", "sha256", "training"):
            self.assertIn(key, meta, f"meta 缺少 {key}")
        self.assertEqual(meta["id"], "default")
        self.assertEqual(meta["obsDim"], 19)
        self.assertEqual(meta["actDim"], 11)
        self.assertEqual(meta["hidden"], 64)
        self.assertEqual(meta["version"], 1)
        self.assertRegex(meta["exportedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        training = meta["training"]
        self.assertEqual(training["totalEpisodes"], 12)
        self.assertEqual(training["totalSteps"], 2400)
        self.assertEqual(training["lastMeanReward"], 3.25)
        self.assertEqual(training["writeCount"], 3)
        self.assertEqual(len(training["runs"]), 1)
        self.assertEqual(training["format"], "gladiator.model-training/v1")

    def test_export_keeps_reward_preset_when_recorded(self) -> None:
        self._remove_target()
        raw = json.loads((self.default_dir / "model.json").read_text(encoding="utf-8"))
        raw["rewardPreset"] = "rush-v1"
        (self.default_dir / "model.json").write_text(json.dumps(raw, ensure_ascii=False),
                                                     encoding="utf-8")
        try:
            status, data = self._post("/api/models/export", {"id": "default"})
            self.assertEqual(status, 200)
            with zipfile.ZipFile(self._target()) as archive:
                meta = json.loads(archive.read("meta.json").decode("utf-8"))
            self.assertEqual(meta["rewardPreset"], "rush-v1")
        finally:
            raw.pop("rewardPreset", None)
            (self.default_dir / "model.json").write_text(json.dumps(raw, ensure_ascii=False),
                                                         encoding="utf-8")

    # --- 错误路径 -------------------------------------------------------
    def test_unknown_model_returns_404(self) -> None:
        status, data = self._post("/api/models/export", {"id": "does-not-exist"})
        self.assertEqual(status, 404, data)

    def test_existing_target_returns_409_then_overwrite_succeeds(self) -> None:
        self._remove_target()
        self.assertEqual(self._post("/api/models/export", {"id": "default"})[0], 200)
        status, data = self._post("/api/models/export", {"id": "default"})
        self.assertEqual(status, 409, data)
        self.assertIn("已存在", data["error"])
        status, data = self._post("/api/models/export", {"id": "default", "overwrite": True})
        self.assertEqual(status, 200, data)
        self.assertTrue(Path(data["path"]).is_file())

    def test_model_without_policy_returns_400(self) -> None:
        status, data = self._post("/api/models/export", {"id": "nopolicy"})
        self.assertEqual(status, 400, data)
        self.assertIn("policy.npz", data["error"])

    def test_missing_id_returns_400(self) -> None:
        status, data = self._post("/api/models/export", {})
        self.assertEqual(status, 400, data)

    def test_query_id_variant(self) -> None:
        self._remove_target()
        status, data = self._post("/api/models/export?id=default&overwrite=1")
        self.assertEqual(status, 200, data)
        self.assertEqual(data["id"], "default")

    # --- 回扫 -----------------------------------------------------------
    def test_exported_gmod_is_scanned_with_training(self) -> None:
        self._remove_target()
        self.assertEqual(self._post("/api/models/export", {"id": "default"})[0], 200)

        status, data = self._get("/api/models")
        self.assertEqual(status, 200)
        listed = {entry["id"]: entry for entry in data["models"]}
        # 同名 .gmod 与可持久化模型**并存**，不是顶掉。
        self.assertIn("default", listed, "可持久化模型必须还在")
        self.assertIn("gmod:default", listed, "只读副本必须并列出现")
        policy_row = listed["default"]
        self.assertEqual(policy_row["kind"], "policy")
        self.assertTrue(policy_row["hasPolicy"], "导出后 default 仍必须是可持久化模型")
        self.assertFalse(policy_row.get("conflict"))

        item = listed["gmod:default"]
        self.assertEqual(item["kind"], "gmod")
        self.assertTrue(item["valid"], item.get("error"))
        self.assertTrue(item["conflict"])
        self.assertIn("同名", item["warning"])
        self.assertEqual(item["episodes"], 12)
        self.assertEqual(item["steps"], 2400)
        self.assertEqual(item["meanReward"], 3.25)
        self.assertEqual(item["training"]["totalEpisodes"], 12)
        self.assertEqual(item["source"]["file"], "default.gmod")
        self.assertEqual(item["fileId"], "default")

        # 裸 id 指向策略模型，前缀 id 指向只读副本。
        status, detail = self._get("/api/models?id=gmod:default")
        self.assertEqual(status, 200)
        model = detail["model"]
        self.assertEqual(model["kind"], "gmod")
        self.assertEqual(model["training"]["totalSteps"], 2400)
        self.assertEqual(model["training"]["writeCount"], 3)
        status, policy_detail = self._get("/api/models?id=default")
        self.assertEqual(status, 200)
        self.assertEqual(policy_detail["model"]["kind"], "policy")



    def test_exporting_gmod_id_returns_400(self) -> None:
        self._remove_target()
        self.assertEqual(self._post("/api/models/export", {"id": "default"})[0], 200)
        status, data = self._post("/api/models/export", {"id": "gmod:default"})
        self.assertEqual(status, 400, data)
        self.assertIn("无需导出", data["error"])

if __name__ == "__main__":
    unittest.main()
