"""模型注册表 / 单局详情 / 打开文件夹（G-622）。

约束：全部使用临时目录；``.npz`` 用手写的 ``.npy`` 成员构造，不依赖 numpy，
也不依赖本机真实安装或真实 artifacts。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from typing import Any
from urllib import error, request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import ApiError, BadRequest, Forbidden, NotFound  # noqa: E402
from models import ModelStore, inspect_policy, spaces_dimensions  # noqa: E402
from processes import ProcessManager  # noqa: E402
from artifacts import ArtifactStore  # noqa: E402
from training_runner import TrainingRunner  # noqa: E402


def _npy_bytes(shape: tuple[int, ...]) -> bytes:
    """造一个合法的 v1 ``.npy``（float64 全零）。"""
    body = "(" + ", ".join(str(item) for item in shape) + (",)" if len(shape) == 1 else ")")
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': %s, }" % body
    total = 10 + len(header) + 1
    pad = 16 - (total % 16)
    header = header + " " * (pad - 1) + "\n"
    count = 1
    for item in shape:
        count *= int(item)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + b"\x00" * (8 * count)


def write_policy(path: Path, *, obs: int = 19, actions: int = 11, hidden: int = 64,
                 skip: tuple[str, ...] = (), include_meta: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        shapes = {
            "W1": (hidden, obs),
            "b1": (hidden,),
            "W2": (actions, hidden),
            "b2": (actions,),
        }
        for key, shape in shapes.items():
            if key in skip:
                continue
            archive.writestr(f"{key}.npy", _npy_bytes(shape))
        # Policy.load 需要 _meta.npy；这里只验结构，内容不影响 Dashboard 侧的形状校验。
        if include_meta:
            archive.writestr("_meta.npy", _npy_bytes((1,)))
    return path


def write_run(root: Path, run_id: str, *, model: dict | None = None,
              episodes: list[dict] | None = None, kind: str = "train") -> Path:
    run_dir = root / "artifacts" / "runs" / run_id
    (run_dir / "reviews").mkdir(parents=True, exist_ok=True)
    write_policy(run_dir / "policy.npz")
    history = episodes if episodes is not None else [{"episode": 1, "steps": 20, "totalReward": 3.5,
                                                      "winner": "Bot_A", "terminated": True,
                                                      "truncated": False, "startedAt": "2026-11-08T22:00:00",
                                                      "durationSeconds": 12.5}]
    report = {
        "kind": kind,
        "summary": {"episodes": len(history),
                    "totalSteps": sum(int(item.get("steps") or 0) for item in history),
                    "meanReward": 3.5},
        "history": history,
        "config": {},
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    meta = {"runId": run_id, "kind": kind, "tag": "Bot_A"}
    if model:
        meta.update(model)
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return run_dir


class InspectPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.obs, self.actions = spaces_dimensions()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_policy(self) -> None:
        path = write_policy(self.root / "p.npz", obs=self.obs, actions=self.actions)
        info = inspect_policy(path)
        self.assertEqual(info["obsDim"], self.obs)
        self.assertEqual(info["numActions"], self.actions)
        self.assertEqual(info["hidden"], 64)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(info["sha256"], expected)

    def test_missing_array_rejected(self) -> None:
        path = write_policy(self.root / "bad.npz", skip=("W2",))
        with self.assertRaises(BadRequest) as ctx:
            inspect_policy(path)
        self.assertIn("W2", str(ctx.exception))

    def test_wrong_obs_dim_rejected(self) -> None:
        path = write_policy(self.root / "bad.npz", obs=self.obs + 3, actions=self.actions)
        with self.assertRaises(BadRequest) as ctx:
            inspect_policy(path)
        self.assertIn("obsDim", str(ctx.exception))

    def test_wrong_actions_rejected(self) -> None:
        path = write_policy(self.root / "bad.npz", obs=self.obs, actions=self.actions + 2)
        with self.assertRaises(BadRequest) as ctx:
            inspect_policy(path)
        self.assertIn("numActions", str(ctx.exception))

    def test_non_zip_rejected(self) -> None:
        path = self.root / "junk.npz"
        path.write_bytes(b"not a zip")
        with self.assertRaises(BadRequest):
            inspect_policy(path)

    def test_missing_meta_rejected(self) -> None:
        path = write_policy(self.root / "no-meta.npz", include_meta=False)
        with self.assertRaises(BadRequest) as ctx:
            inspect_policy(path)
        self.assertIn("_meta", str(ctx.exception))


class ModelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = ModelStore(self.root / "var" / "models", self.root / "artifacts")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_is_builtin_without_policy(self) -> None:
        self.store.ensure()
        items = self.store.list()
        self.assertEqual([item["id"] for item in items], ["default"])
        self.assertTrue(items[0]["builtin"])
        self.assertFalse(items[0]["hasPolicy"])
        # G-624：default 变成可持久化模型，policy_path 指向一个「首次写回才创建」的路径。
        path = self.store.policy_path("default")
        self.assertEqual(path.name, "policy.npz")
        self.assertEqual(path.parent.name, "default")
        self.assertFalse(path.is_file(), "尚未训练写回时不应有文件")
        self.assertEqual(items[0]["training"]["totalEpisodes"], 0)
        with self.assertRaises(BadRequest):
            self.store.delete("default")

    def test_import_from_run(self) -> None:
        write_run(self.root, "20261108-2200-train-Bot_A")
        model = self.store.create({"name": "Rush A", "sourceRunId": "20261108-2200-train-Bot_A",
                                   "description": "导入测试"})
        self.assertEqual(model["id"], "20261108-2200-train-bot_a")
        self.assertEqual(model["obsDim"], 19)
        self.assertEqual(model["numActions"], 11)
        self.assertEqual(model["episodes"], 1)
        self.assertEqual(model["source"]["runId"], "20261108-2200-train-Bot_A")
        self.assertTrue((self.store.root / model["id"] / "policy.npz").is_file())
        self.assertTrue(model["hasPolicy"])
        self.assertEqual(self.store.policy_path(model["id"]).name, "policy.npz")

    def test_duplicate_import_gets_suffix(self) -> None:
        write_run(self.root, "20261108-2201-train-Bot_B")
        first = self.store.create({"name": "B", "sourceRunId": "20261108-2201-train-Bot_B"})
        second = self.store.create({"name": "B2", "sourceRunId": "20261108-2201-train-Bot_B"})
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(second["id"].endswith("-2"))

    def test_update_and_delete_soft(self) -> None:
        write_run(self.root, "20261108-2202-train-Bot_C")
        model = self.store.create({"name": "C", "sourceRunId": "20261108-2202-train-Bot_C"})
        updated = self.store.update({"id": model["id"], "name": "C2", "description": "改了"})
        self.assertEqual(updated["name"], "C2")
        self.assertEqual(updated["description"], "改了")
        self.store.delete(model["id"])
        with self.assertRaises(NotFound):
            self.store.get(model["id"])
        backups = list((self.root / "var" / "models-trash").glob(f"{model['id']}-*"))
        # 需求是「删除/覆盖前都要备份」：update 覆盖一次、delete 一次，所以至少 2 份。
        self.assertGreaterEqual(len(backups), 2, "覆盖与删除都应留 models-trash 副本")
        self.assertTrue(any((item / "policy.npz").is_file() for item in backups),
                        "备份里要保留策略文件，便于恢复")

    def test_recent_runs(self) -> None:
        write_run(self.root, "20261108-2203-train-Bot_D",
                  model={"modelId": "m1", "modelName": "M1"})
        write_run(self.root, "20261108-2204-train-Bot_E")
        detail = self.store.detail("default")
        self.assertEqual(detail["recentRuns"], [])
        # 手工造一个 m1 模型目录，检查使用记录
        self.store.root.mkdir(parents=True, exist_ok=True)
        (self.store.root / "m1").mkdir()
        (self.store.root / "m1" / "model.json").write_text(json.dumps({
            "format": "gladiator.model/v1", "id": "m1", "name": "M1",
            "obsDim": 19, "numActions": 11, "hidden": 64, "builtin": False,
        }), encoding="utf-8")
        detail = self.store.detail("m1")
        self.assertEqual([item["runId"] for item in detail["recentRuns"]],
                         ["20261108-2203-train-Bot_D"])

    def test_policy_path_outside_artifacts_rejected(self) -> None:
        outside = write_policy(self.root / "elsewhere" / "p.npz")
        with self.assertRaises(BadRequest) as ctx:
            self.store.create({"name": "X", "policyPath": str(outside)})
        self.assertIn("training/artifacts", str(ctx.exception))

    def test_missing_run_returns_404(self) -> None:
        with self.assertRaises(NotFound):
            self.store.create({"name": "X", "sourceRunId": "20200101-0000-train-Nope"})

    def test_import_by_absolute_policy_path(self) -> None:
        run_dir = write_run(self.root, "20261108-2205-train-Bot_F")
        model = self.store.create({"name": "F", "policyPath": str(run_dir / "policy.npz")})
        self.assertEqual(model["source"]["runId"], None)
        self.assertTrue(model["hasPolicy"])


def _cfg(root: Path, *, port: int = 8080, read_only: bool = False, mc_port: int = 25986) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1", port=port,
        artifacts_dir=root / "artifacts", var_dir=root / "var",
        server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training", python_exe=sys.executable,
        java_exe="java", node_exe="node",
        allow_process_control=not read_only, scan_external=False,
        server_jar="missing.jar", mc_port=mc_port,
    )


class _HttpFixture(unittest.TestCase):
    read_only = False
    mc_port = 25985

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cfg = _cfg(cls.root, read_only=cls.read_only, mc_port=cls.mc_port)
        cls.app = DashboardApp(cls.cfg)
        cls.opened: list[str] = []
        cls.app.open_folder_supported = True
        cls.app.folder_opener = lambda path: cls.opened.append(path)
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

    def _request(self, path: str, method: str = "GET", payload: Any = None) -> tuple[int, bytes]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
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


class ModelHttpTests(_HttpFixture):
    mc_port = 25984

    def test_list_contains_default(self) -> None:
        data = self._json("/api/models")
        ids = [item["id"] for item in data["models"]]
        self.assertIn("default", ids)
        default = next(item for item in data["models"] if item["id"] == "default")
        self.assertTrue(default["builtin"])
        self.assertEqual(ids[0], "default", "内置 default 必须排在最前")

    def test_create_detail_update_delete(self) -> None:
        write_run(self.root, "20261108-2230-train-Bot_H")
        created = self._json("/api/models", "POST", {
            "name": "H", "sourceRunId": "20261108-2230-train-Bot_H",
        })["model"]
        self.assertEqual(created["obsDim"], 19)
        detail = self._json(f"/api/models?id={created['id']}")["model"]
        self.assertEqual(detail["name"], "H")
        self.assertEqual(detail["recentRuns"], [])
        updated = self._json("/api/models/update", "POST", {
            "id": created["id"], "name": "H2", "note": "n",
        })["model"]
        self.assertEqual(updated["name"], "H2")
        deleted = self._json(f"/api/models?id={created['id']}", "DELETE")
        self.assertEqual(deleted["deleted"], created["id"])
        status, _body = self._request(f"/api/models?id={created['id']}")
        self.assertEqual(status, 404)

    def test_delete_builtin_400(self) -> None:
        status, body = self._request("/api/models?id=default", "DELETE")
        self.assertEqual(status, 400)
        self.assertIn("default", json.loads(body.decode("utf-8"))["error"])

    def test_invalid_import_400(self) -> None:
        write_run(self.root, "20261108-2231-train-Bot_I")
        good = self.root / "artifacts" / "runs" / "20261108-2231-train-Bot_I" / "policy.npz"
        write_policy(good, obs=16)
        status, body = self._request("/api/models", "POST", {
            "name": "bad", "sourceRunId": "20261108-2231-train-Bot_I",
        })
        self.assertEqual(status, 400)
        self.assertIn("obsDim", json.loads(body.decode("utf-8"))["error"])

    def test_options_expose_models(self) -> None:
        options = self._json("/api/training/options")["options"]
        self.assertTrue(options["modelSupport"])
        self.assertIn("default", [item["id"] for item in options["models"]])


class OpenFolderTests(_HttpFixture):
    mc_port = 25983

    def test_open_scripts_returns_path(self) -> None:
        self.opened.clear()
        data = self._json("/api/open-folder", "POST", {"kind": "scripts"})
        self.assertEqual(Path(data["path"]), (self.root / "var").resolve())
        self.assertEqual(self.opened, [data["path"]])

    def test_open_run_and_reviews(self) -> None:
        run_dir = write_run(self.root, "20261108-2232-train-Bot_J")
        data = self._json("/api/open-folder", "POST",
                          {"kind": "run", "id": "20261108-2232-train-Bot_J"})
        self.assertEqual(Path(data["path"]), run_dir.resolve())
        data = self._json("/api/open-folder", "POST",
                          {"kind": "run-reviews", "id": "20261108-2232-train-Bot_J"})
        self.assertEqual(Path(data["path"]), (run_dir / "reviews").resolve())

    def test_traversal_rejected(self) -> None:
        status, body = self._request("/api/open-folder", "POST", {"kind": "run", "id": "../etc"})
        self.assertEqual(status, 400)
        self.assertIn("run", json.loads(body.decode("utf-8"))["error"])

    def test_unknown_kind_400(self) -> None:
        status, _body = self._request("/api/open-folder", "POST", {"kind": "etc"})
        self.assertEqual(status, 400)

    def test_missing_run_404(self) -> None:
        status, _body = self._request("/api/open-folder", "POST",
                                      {"kind": "run", "id": "20200101-0000-train-None"})
        self.assertEqual(status, 404)

    def test_non_windows_501(self) -> None:
        self.app.open_folder_supported = False
        try:
            status, body = self._request("/api/open-folder", "POST", {"kind": "scripts"})
            self.assertEqual(status, 501)
            self.assertIn("Windows", json.loads(body.decode("utf-8"))["error"])
        finally:
            self.app.open_folder_supported = True


class OpenFolderReadOnlyTests(_HttpFixture):
    read_only = True
    mc_port = 25982

    def test_read_only_403(self) -> None:
        status, body = self._request("/api/open-folder", "POST", {"kind": "scripts"})
        self.assertEqual(status, 403)
        self.assertIn("只读", json.loads(body.decode("utf-8"))["error"])


class EpisodeTests(_HttpFixture):
    mc_port = 25981

    def test_episode_detail_with_review(self) -> None:
        run_dir = write_run(self.root, "20261108-2233-train-Bot_K",
                            episodes=[
                                {"episode": 1, "steps": 20, "totalReward": 3.5, "winner": "Bot_K",
                                 "terminated": True, "truncated": False,
                                 "startedAt": "2026-11-08T22:00:00", "durationSeconds": 12.5},
                                {"episode": 2, "steps": 30, "totalReward": -1.0, "winner": None,
                                 "terminated": False, "truncated": True,
                                 "startedAt": "2026-11-08T22:01:00", "durationSeconds": 18.0},
                            ], model={"modelId": "m1", "modelName": "M1"})
        (run_dir / "reviews" / "episode-001.md").write_text("# 复盘\n内容", encoding="utf-8")
        (run_dir / "reviews" / "episode-001.json").write_text(json.dumps({
            "episode": 1, "damageDealt": 12.0, "damageTaken": 4.0,
            "actionCounts": {"attack": 8, "approach": 5},
            "rewardBreakdown": {"kill": 20.0, "damageDealt": 2.5},
            "events": [{"step": 12, "type": "kill"}],
        }), encoding="utf-8")

        data = self._json("/api/training/episodes?run=20261108-2233-train-Bot_K&episode=1")
        self.assertEqual(data["episode"], 1)
        self.assertEqual(data["episodeCount"], 2)
        self.assertEqual(data["winner"], "Bot_K")
        self.assertEqual(data["model"]["id"], "m1")
        self.assertIn("复盘", data["review"]["markdown"])
        self.assertEqual(data["review"]["data"]["actionCounts"]["attack"], 8)

        second = self._json("/api/training/episodes?run=20261108-2233-train-Bot_K&episode=2")
        self.assertIsNone(second["review"], "第 2 局没有复盘文件")

    def test_episode_out_of_range_400(self) -> None:
        write_run(self.root, "20261108-2234-train-Bot_L")
        for episode in ("0", "5", "abc"):
            with self.subTest(episode=episode):
                status, body = self._request(
                    f"/api/training/episodes?run=20261108-2234-train-Bot_L&episode={episode}")
                self.assertEqual(status, 400, body.decode("utf-8", "replace"))

    def test_run_not_found_404(self) -> None:
        status, _body = self._request("/api/training/episodes?run=20200101-0000-train-None&episode=1")
        self.assertEqual(status, 404)


class TrainingModelParamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = _cfg(self.root, read_only=True, mc_port=25980)
        self.store = ModelStore(self.cfg.var_dir / "models", self.cfg.artifacts_dir)
        self.store.ensure()
        self.runner = TrainingRunner(
            self.cfg,
            ProcessManager(self.cfg.logs_dir, self.cfg.state_file),
            ArtifactStore(self.cfg.artifacts_dir),
            models=self.store,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _payload(self, **extra: Any) -> dict:
        payload = {
            "users": ["Bot_A", "Bot_B"],
            "ports": [8892, 8893],
            "plan": [{"opponentMode": "idle", "episodes": 1}],
        }
        payload.update(extra)
        return payload

    def test_default_model_passes_no_init_policy(self) -> None:
        argv, params, _total, _counts = self.runner._build_train(self._payload(modelId="default"))
        self.assertNotIn("--init-policy", argv)
        self.assertEqual(params["modelId"], "default")
        self.assertIsNone(params["initPolicy"])

    def test_runs_are_written_for_registered_model(self) -> None:
        write_run(self.root, "20261108-2235-train-Bot_M")
        model = self.store.create({"name": "M", "sourceRunId": "20261108-2235-train-Bot_M"})
        argv, params, _total, _counts = self.runner._build_train(self._payload(modelId=model["id"]))
        self.assertEqual(argv[argv.index("--init-policy") + 1], str(self.store.policy_path(model["id"])))
        self.assertEqual(argv[argv.index("--model-id") + 1], model["id"])
        self.assertEqual(params["modelName"], "M")
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["modelId"], model["id"])
        self.assertEqual(meta["modelName"], "M")

    def test_unknown_model_400(self) -> None:
        with self.assertRaises(BadRequest):
            self.runner._build_train(self._payload(modelId="nope"))

    def test_dimension_mismatch_rejected_before_start(self) -> None:
        write_run(self.root, "20261108-2236-train-Bot_N")
        model = self.store.create({"name": "N", "sourceRunId": "20261108-2236-train-Bot_N"})
        path = self.store.root / model["id"] / "model.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["obsDim"] = 16
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self._payload(modelId=model["id"]))
        self.assertIn("obsDim", str(ctx.exception))

    def test_multi_also_passes_init_policy(self) -> None:
        write_run(self.root, "20261108-2237-train-Bot_O")
        model = self.store.create({"name": "O", "sourceRunId": "20261108-2237-train-Bot_O"})
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8894,
            "plan": [{"opponentMode": "idle", "episodes": 1}], "modelId": model["id"],
        })
        self.assertEqual(argv[argv.index("--init-policy") + 1], str(self.store.policy_path(model["id"])))
        self.assertEqual(params["modelId"], model["id"])


if __name__ == "__main__":
    unittest.main()
