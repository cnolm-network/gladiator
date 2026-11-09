"""模型可视化接口（G-625）单测：``GET /api/models/viz?id=<model>``。

覆盖：
- ``?id=default`` → 200，关键字段齐全；无 policy / 无备份 / 无训练记录时各段 ``available:false`` 而非异常；
- 未知 id → 404；
- 完整数据模型 → 各段 ``available:true``（直方图 24 桶、11 动作、对手分组、累计表等）；
- 只有权重、没有训练记录的模型 → ``runs.available:false`` 且不抛异常（缺数据段返回 null/note）；
- multi 融合成功/失败的持久化写回与降级路径（在 test_persist.py 中）。

全部使用临时目录 + 手写 npz（不依赖 numpy），不启动真实训练、不连 Paper。
"""

from __future__ import annotations

import json
import struct
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
from models import ModelStore  # noqa: E402


def _npy_bytes(shape: tuple[int, ...]) -> bytes:
    """造一个合法的 v1 ``.npy``（float64 全零），不依赖 numpy。"""
    body = "(" + ", ".join(str(item) for item in shape) + (",)" if len(shape) == 1 else ")")
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': %s, }" % body
    total = 10 + len(header) + 1
    pad = 16 - (total % 16)
    header = header + " " * (pad - 1) + "\n"
    count = 1
    for item in shape:
        count *= int(item)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + b"\x00" * (8 * count)


def write_policy(path: Path, *, obs: int = 19, actions: int = 11, hidden: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for key, shape in {
            "W1": (hidden, obs), "b1": (hidden,), "W2": (actions, hidden), "b2": (actions,),
        }.items():
            archive.writestr(f"{key}.npy", _npy_bytes(shape))
        archive.writestr("_meta.npy", _npy_bytes((1,)))
    return path


class ModelVizHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        artifacts = root / "artifacts"
        models_dir = root / "var" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        cls.store = ModelStore(models_dir, artifacts)
        cls.store.ensure()

        # ---- 完整数据模型：一次 train run + 从中创建模型 + 写回 training.json ----
        run_id = "20261109-0100-train-Bot_V"
        run_dir = artifacts / "runs" / run_id
        (run_dir / "reviews").mkdir(parents=True, exist_ok=True)
        history = [
            {
                "episode": 1, "steps": 120, "totalReward": 1.25, "winner": "Bot_V",
                "review": {
                    "hits": 4, "whiffs": 2, "hitRate": 0.6667, "maxCombo": 2,
                    "damageDealt": 8.0, "damageTaken": 5.0, "damageRatio": 1.6,
                    "durationSeconds": 12.3,
                    "actionCounts": {"idle": 10, "attack": 5, "approach": 3},
                },
            },
            {
                "episode": 2, "steps": 100, "totalReward": 0.5, "winner": "Bot_Opp",
                "review": {
                    "hits": 2, "whiffs": 6, "hitRate": 0.25, "maxCombo": 1,
                    "damageDealt": 3.0, "damageTaken": 7.0, "damageRatio": 0.43,
                    "durationSeconds": 9.0,
                    "actionCounts": {"idle": 5, "attack": 8, "block": 2},
                },
            },
        ]
        report = {
            "kind": "train",
            "summary": {"episodes": 2, "totalSteps": 220, "meanReward": 0.875},
            "history": history,
            "config": {"users": ["Bot_V", "Bot_Opp"]},
        }
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        (run_dir / "run.json").write_text(
            json.dumps({"runId": run_id, "kind": "train"}, ensure_ascii=False), encoding="utf-8")
        write_policy(run_dir / "policy.npz")

        created = cls.store.create({"name": "Viz Full", "sourceRunId": run_id})
        cls.full_id = created["id"]
        # 手动登记训练记录（模拟真实训练结束后的写回），让 viz 的 runs 段有数据。
        (models_dir / cls.full_id / "training.json").write_text(json.dumps({
            "runs": [{"runId": run_id, "at": "2026-11-09T01:00:00",
                      "episodes": 2, "steps": 220, "meanReward": 0.875}],
            "lastRunId": run_id, "totalEpisodes": 2, "totalSteps": 220,
            "lastMeanReward": 0.875, "writeCount": 1,
        }, ensure_ascii=False), encoding="utf-8")
        # 一份「最近备份」，让 backupDiff 段也能算出余弦 / L2 相对变化。
        (models_dir.parent / "models-trash").mkdir(parents=True, exist_ok=True)
        write_policy(models_dir.parent / "models-trash" / f"{cls.full_id}-20261109-005000.npz")

        # ---- 只有权重、没有训练记录的模型（缺数据段应优雅降级） ----
        partial = models_dir / "vizpartial"
        partial.mkdir(parents=True, exist_ok=True)
        write_policy(partial / "policy.npz")
        (partial / "model.json").write_text(json.dumps({
            "id": "vizpartial", "name": "Viz Partial", "kind": "train",
            "obsDim": 19, "numActions": 11, "hidden": 64,
            "builtin": False,
        }, ensure_ascii=False), encoding="utf-8")

        cls.cfg = DashboardConfig(
            host="127.0.0.1", port=8080,
            artifacts_dir=artifacts, var_dir=root / "var",
            server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
            training_dir=root / "training",
            python_exe=sys.executable, java_exe="java", node_exe="node",
            allow_process_control=False, scan_external=False,
            server_jar="missing.jar", mc_port=25999,
        )
        cls.app = DashboardApp(cls.cfg)
        cls.httpd = make_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                     kwargs={"poll_interval": 0.1}, daemon=True)
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
    def _request(self, path: str) -> tuple[int, bytes]:
        req = request.Request(self.base + path)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()

    def _json(self, path: str) -> tuple[int, dict]:
        status, body = self._request(path)
        return status, json.loads(body.decode("utf-8"))

    # --- 关键字段 / 缺数据降级 ------------------------------------------
    def test_default_model_returns_200_with_key_fields(self) -> None:
        status, data = self._json("/api/models/viz?id=default")
        self.assertEqual(status, 200)
        for key in ("ok", "id", "model", "weights", "backupDiff", "runs"):
            self.assertIn(key, data, f"响应缺少顶层字段 {key}")
        self.assertEqual(data["id"], "default")
        self.assertTrue(data["ok"])
        # default 没有 policy.npz / 备份 / 训练记录 → 各段 available:false，附 note，不抛异常。
        self.assertFalse(data["weights"]["available"])
        self.assertFalse(data["backupDiff"]["available"])
        self.assertFalse(data["runs"]["available"])
        self.assertIn("note", data["weights"])
        self.assertIn("note", data["backupDiff"])
        self.assertIn("note", data["runs"])

    def test_unknown_model_returns_404(self) -> None:
        status, _body = self._request("/api/models/viz?id=doesnotexist123")
        self.assertEqual(status, 404)

    def test_full_model_populates_all_sections(self) -> None:
        status, data = self._json(f"/api/models/viz?id={self.full_id}")
        self.assertEqual(status, 200)
        # 权重段
        self.assertTrue(data["weights"]["available"])
        self.assertEqual(len(data["weights"]["histogram"]["counts"]), 24, "直方图应为 24 桶")
        self.assertIn("norms", data["weights"])
        self.assertIn("totalNorm", data["weights"])
        # 备份差异段
        self.assertTrue(data["backupDiff"]["available"])
        self.assertIn("l2RelativeChange", data["backupDiff"])
        self.assertIn("cosineSimilarity", data["backupDiff"])
        # 训练 run 聚合段
        runs = data["runs"]
        self.assertTrue(runs["available"])
        self.assertEqual(runs["runCount"], 1)
        self.assertEqual(runs["episodeCount"], 2)
        self.assertIn("byRun", runs)
        self.assertIn("cumulative", runs)
        self.assertIn("metrics", runs)
        self.assertGreater(len(runs["opponents"]), 0, "应有对手分组")
        self.assertEqual(len(runs["actions"]), 11, "动作分布应为 11 个")
        self.assertGreater(len(runs["table"]["rows"]), 0, "累计统计表应有行")
        self.assertIn("totals", runs["table"], "累计统计表应带合计行数据")
        self.assertIsNotNone(runs["table"]["totals"], "合计行不应为 null")
        self.assertIn("episodes", runs["table"]["totals"])
        # 效果指标字段
        for field in ("winRate", "hitRate", "whiffRate", "damageRatio", "avgDuration"):
            self.assertIn(field, runs["metrics"])

    def test_required_sections_and_fields(self) -> None:
        """验收口径：viz 响应必须含 weights/backupDiff/runs 三大段及子字段。"""
        status, data = self._json(f"/api/models/viz?id={self.full_id}")
        self.assertEqual(status, 200)
        for key in ("id", "model", "weights", "backupDiff", "runs"):
            self.assertIn(key, data)
        weights = data["weights"]
        for key in ("available", "totalNorm", "norms", "histogram", "count"):
            self.assertIn(key, weights, f"weights 缺少 {key}")
        self.assertIn("binEdges", weights["histogram"])
        self.assertIn("counts", weights["histogram"])
        for key in ("l2RelativeChange", "cosineSimilarity", "available"):
            self.assertIn(key, data["backupDiff"], f"backupDiff 缺少 {key}")
        runs = data["runs"]
        for key in ("available", "byRun", "cumulative", "metrics", "opponents",
                    "actions", "table", "runCount", "episodeCount"):
            self.assertIn(key, runs, f"runs 缺少 {key}")
        self.assertEqual(len(runs["actions"]), 11)
        self.assertTrue(runs["table"]["rows"])
        self.assertTrue(runs["table"]["totals"])

    def test_partial_model_missing_training_is_null_not_exception(self) -> None:
        status, data = self._json("/api/models/viz?id=vizpartial")
        self.assertEqual(status, 200)  # 没有训练记录也返回 200，不抛 500
        self.assertTrue(data["weights"]["available"], "有 policy.npz → 权重段应可用")
        self.assertFalse(data["runs"]["available"], "无训练记录 → runs 段 available:false")
        self.assertIn("note", data["runs"])
        self.assertFalse(data["backupDiff"]["available"], "无备份 → backupDiff 段 available:false")
        self.assertIn("note", data["backupDiff"])


if __name__ == "__main__":
    unittest.main()
