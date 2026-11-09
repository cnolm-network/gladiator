"""G-623：.gmod 模型包的热扫描、校验、缓存、删除保护与前端契约。

全部使用临时目录；npz / gmod 都手写 zip，不依赖 numpy、不依赖本机 artifacts，
也不碰真实的 ``dashboard/var`` / ``models/``。
"""

from __future__ import annotations

import io
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
from artifacts import ArtifactStore  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import BadRequest, Conflict  # noqa: E402
from gmod import GmodScanner  # noqa: E402
from models import ModelStore  # noqa: E402
from processes import ProcessManager  # noqa: E402
from training_runner import TrainingJob, TrainingRunner  # noqa: E402

OBS, ACT, HIDDEN = 19, 11, 64


def _npy_bytes(shape: tuple[int, ...]) -> bytes:
    """造一个合法的 v1 .npy（float64 全零）。"""
    body = "(" + ", ".join(str(item) for item in shape) + (",)" if len(shape) == 1 else ")")
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': %s, }" % body
    total = 10 + len(header) + 1
    pad = 16 - (total % 16)
    header = header + " " * (pad - 1) + "\n"
    count = 1
    for item in shape:
        count *= int(item)
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header))
            + header.encode("latin1") + b"\x00" * (8 * count))


def _zip_info(name: str) -> zipfile.ZipInfo:
    """固定 zip 时间戳：否则跨秒生成的 npz 字节不同，全量套件会偶发失败。"""
    return zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))


def npz_bytes(*, obs: int = OBS, act: int = ACT, hidden: int = HIDDEN,
              skip: tuple[str, ...] = (), include_meta: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        shapes = {"W1": (hidden, obs), "b1": (hidden,), "W2": (act, hidden), "b2": (act,)}
        for key, shape in shapes.items():
            if key not in skip:
                archive.writestr(_zip_info(f"{key}.npy"), _npy_bytes(shape))
        if include_meta:
            archive.writestr(_zip_info("_meta.npy"), _npy_bytes((1,)))
    return buf.getvalue()


def default_meta(**extra: Any) -> dict:
    meta = {
        "format": "gladiator.model/v1",
        "id": "rush-v3",
        "name": "Rush v3",
        "obsDim": OBS,
        "actDim": ACT,
        "color": "§6",
        "version": "0.3.0",
        "createdAt": "2026-11-08T19:00:00",
        "description": "以 rusher 为对手训练 120 局",
    }
    meta.update(extra)
    return meta


def write_gmod(directory: Path, name: str, *, meta: dict | None = None,
               policy: bytes | None = None, with_meta: bool = True,
               with_policy: bool = True, raw: bytes | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if raw is not None:
        path.write_bytes(raw)
        return path
    with zipfile.ZipFile(path, "w") as archive:
        if with_meta:
            archive.writestr(_zip_info("meta.json"),
                             json.dumps(meta if meta is not None else default_meta(),
                                        ensure_ascii=False))
        if with_policy:
            archive.writestr(_zip_info("policy.npz"),
                             policy if policy is not None else npz_bytes())
    return path


def write_run(root: Path, run_id: str, *, with_report: bool = False) -> Path:
    run = root / "artifacts" / "runs" / run_id
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps({"runId": run_id, "kind": "train"}), encoding="utf-8")
    if with_report:
        history = [{"episode": 1, "steps": 20, "totalReward": 3.5, "winner": "Bot_A",
                    "terminated": True, "truncated": False, "startedAt": "2026-11-08T23:00:00",
                    "durationSeconds": 12.5}]
        (run / "report.json").write_text(json.dumps({
            "kind": "train", "summary": {"episodes": 1, "totalSteps": 20, "meanReward": 3.5},
            "history": history, "config": {},
        }, ensure_ascii=False), encoding="utf-8")
    return run


def _cfg(root: Path, *, read_only: bool = False, gmod_dir: Path | None = None,
         mc_port: int = 25970) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1", port=8080,
        artifacts_dir=root / "artifacts", var_dir=root / "var",
        server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training", gmod_dir=gmod_dir or root / "models",
        python_exe=sys.executable, java_exe="java", node_exe="node",
        allow_process_control=not read_only, scan_external=False,
        server_jar="missing.jar", mc_port=mc_port,
    )


class ScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = self.root / "models"
        self.dir.mkdir()
        self.scanner = GmodScanner(self.dir, self.root / "var" / "gmod-cache")
        self.scanner.ensure()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _by_id(self, ident: str) -> dict:
        return next(item for item in self.scanner.scan()["items"] if item["id"] == ident)

    def test_valid_package(self) -> None:
        path = write_gmod(self.dir, "rush-v3.gmod")
        item = self._by_id("rush-v3")
        self.assertTrue(item["valid"])
        self.assertIsNone(item["error"])
        self.assertEqual(item["kind"], "gmod")
        self.assertEqual(item["name"], "Rush v3")
        self.assertEqual(item["obsDim"], OBS)
        self.assertEqual(item["numActions"], ACT)
        self.assertEqual(item["hidden"], HIDDEN)
        self.assertEqual(item["color"], "§6")
        self.assertEqual(item["source"], {"type": "gmod", "file": "rush-v3.gmod", "dir": "models"})
        self.assertEqual(item["sizeBytes"], path.stat().st_size)
        self.assertEqual(len(item["sha256"]), 64)
        self.assertTrue(item["hasPolicy"])
        self.assertIsNone(item["warning"])

    def test_missing_meta(self) -> None:
        write_gmod(self.dir, "no-meta.gmod", with_meta=False)
        item = self._by_id("no-meta")
        self.assertFalse(item["valid"])
        self.assertIn("meta.json", item["error"])

    def test_missing_policy(self) -> None:
        write_gmod(self.dir, "no-npz.gmod", with_policy=False)
        item = self._by_id("no-npz")
        self.assertFalse(item["valid"])
        self.assertIn("policy.npz", item["error"])

    def test_meta_dimension_mismatch(self) -> None:
        write_gmod(self.dir, "bad-dim.gmod", meta=default_meta(obsDim=16, actDim=7))
        item = self._by_id("bad-dim")
        self.assertFalse(item["valid"])
        self.assertIn("19", item["error"])
        self.assertIn("11", item["error"])

    def test_npz_shape_mismatch_with_meta(self) -> None:
        write_gmod(self.dir, "shape.gmod", policy=npz_bytes(obs=16))
        item = self._by_id("shape")
        self.assertFalse(item["valid"])
        self.assertIn("不一致", item["error"])

    def test_npz_missing_array_or_meta(self) -> None:
        write_gmod(self.dir, "no-w2.gmod", policy=npz_bytes(skip=("W2",)))
        self.assertIn("W2", self._by_id("no-w2")["error"])
        write_gmod(self.dir, "no-_meta.gmod", policy=npz_bytes(include_meta=False))
        self.assertIn("_meta", self._by_id("no-_meta")["error"])

    def test_corrupt_zip(self) -> None:
        write_gmod(self.dir, "broken.gmod", raw=b"not a zip")
        item = self._by_id("broken")
        self.assertFalse(item["valid"])
        self.assertIn("zip", item["error"])

    def test_invalid_meta_fields(self) -> None:
        write_gmod(self.dir, "bad-id.gmod", meta=default_meta(id="Rush V3"))
        self.assertIn("id", self._by_id("bad-id")["error"])
        write_gmod(self.dir, "bad-color.gmod", meta=default_meta(color="§gg"))
        self.assertIn("color", self._by_id("bad-color")["error"])
        write_gmod(self.dir, "bad-name.gmod", meta=default_meta(name="x" * 41))
        self.assertIn("40", self._by_id("bad-name")["error"])

    def test_invalid_does_not_break_list(self) -> None:
        write_gmod(self.dir, "a-good.gmod")
        write_gmod(self.dir, "b-broken.gmod", raw=b"junk")
        result = self.scanner.scan()
        self.assertEqual(result["scanned"], 2)
        self.assertEqual([item["file"] for item in result["invalid"]], ["b-broken.gmod"])
        self.assertEqual(len(result["items"]), 2)

    def test_meta_id_vs_filename_warning(self) -> None:
        write_gmod(self.dir, "alpha.gmod", meta=default_meta(id="another-id"))
        item = self._by_id("alpha")
        self.assertEqual(item["id"], "alpha")
        self.assertTrue(item["valid"])
        self.assertIn("文件名", item["warning"])

    def test_duplicate_meta_id_both_appear(self) -> None:
        write_gmod(self.dir, "one.gmod", meta=default_meta(id="same"))
        write_gmod(self.dir, "two.gmod", meta=default_meta(id="same"))
        items = self.scanner.scan()["items"]
        self.assertEqual(sorted(item["id"] for item in items), ["one", "two"])
        self.assertTrue(all(item["warning"] for item in items))

    def test_deleted_file_disappears(self) -> None:
        path = write_gmod(self.dir, "gone.gmod")
        self.assertEqual(len(self.scanner.scan()["items"]), 1)
        path.unlink()
        self.assertEqual(self.scanner.scan()["items"], [])
        self.assertEqual(self.scanner.scan()["scanned"], 0)

    def test_mtime_size_cache_reuse(self) -> None:
        path = write_gmod(self.dir, "cached.gmod")
        calls: list[str] = []
        original = self.scanner._inspect

        def counted(target: Path, key: tuple[int, int]) -> dict:
            calls.append(target.name)
            return original(target, key)

        self.scanner._inspect = counted
        self.scanner.scan()
        self.scanner.scan()
        self.assertEqual(calls, ["cached.gmod"], "未变化的文件应命中 mtime+size 缓存")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.scanner.scan()
        self.assertEqual(len(calls), 2, "mtime 变化后应重新解析")

    def test_reload_reports_added_and_invalid(self) -> None:
        write_gmod(self.dir, "first.gmod")
        write_gmod(self.dir, "bad.gmod", raw=b"junk")
        result = self.scanner.reload()
        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["added"], 2)
        self.assertEqual([item["file"] for item in result["invalid"]], ["bad.gmod"])
        write_gmod(self.dir, "second.gmod")
        again = self.scanner.reload()
        self.assertEqual(again["added"], 1)

    def test_cache_policy_is_content_addressed_and_reused(self) -> None:
        policy = npz_bytes()
        write_gmod(self.dir, "a.gmod", policy=policy)
        write_gmod(self.dir, "b.gmod", meta=default_meta(id="b"), policy=policy)
        self.scanner.scan()
        first = self.scanner.cache_policy("a")
        second = self.scanner.cache_policy("b")
        self.assertEqual(first, second, "相同 policy 内容应复用同一缓存文件")
        self.assertEqual(first.read_bytes(), policy)
        self.assertTrue(first.name.endswith(".npz"))
        self.assertEqual(first.parent.name, "gmod-cache")

    def test_invalid_cannot_be_cached(self) -> None:
        write_gmod(self.dir, "bad.gmod", raw=b"junk")
        self.scanner.scan()
        with self.assertRaises(BadRequest) as ctx:
            self.scanner.cache_policy("bad")
        self.assertIn("不可用", str(ctx.exception))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = self.root / "models"
        self.dir.mkdir()
        self.cfg = _cfg(self.root)
        self.gmod = GmodScanner(self.dir, self.cfg.var_dir / "gmod-cache")
        self.store = ModelStore(self.cfg.var_dir / "models", self.cfg.artifacts_dir, self.gmod)
        self.store.ensure()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_list_merges_default_policy_and_gmod(self) -> None:
        write_gmod(self.dir, "rush-v3.gmod")
        write_run(self.root, "20261108-2301-train-Bot_R", with_report=True)
        run_dir = self.root / "artifacts" / "runs" / "20261108-2301-train-Bot_R"
        # 造一个策略产物副本，让 registry 里不只有 default
        (run_dir / "policy.npz").write_bytes(npz_bytes())
        self.store.create({"name": "R", "sourceRunId": run_dir.name})
        items = self.store.list()
        self.assertEqual(items[0]["id"], "default")
        kinds = {item["id"]: item["kind"] for item in items}
        self.assertEqual(kinds["gmod:rush-v3"], "gmod")
        self.assertTrue(all(value == "policy" for key, value in kinds.items()
                            if key != "gmod:rush-v3"))
        gmod = next(item for item in items if item["id"] == "gmod:rush-v3")
        self.assertEqual(gmod["fileId"], "rush-v3")
        self.assertFalse(gmod["conflict"])
        self.assertTrue(gmod["valid"])
        self.assertEqual(gmod["source"]["type"], "gmod")
        self.assertIsNone(gmod["error"])

    def test_same_id_conflict_keeps_both_rows(self) -> None:
        # 先注册一个 id=dupe 的策略产物，再放同名的 .gmod：两行必须并存，互不顶掉。
        write_run(self.root, "20261108-2302-train-Bot_S", with_report=True)
        run_dir = self.root / "artifacts" / "runs" / "20261108-2302-train-Bot_S"
        (run_dir / "policy.npz").write_bytes(npz_bytes())
        self.store.create({"name": "S", "id": "dupe", "sourceRunId": run_dir.name})
        write_gmod(self.dir, "dupe.gmod", meta=default_meta(id="dupe"))
        items = self.store.list()
        policy_rows = [item for item in items if item["id"] == "dupe"]
        gmod_rows = [item for item in items if item["id"] == "gmod:dupe"]
        self.assertEqual(len(policy_rows), 1, "可持久化模型必须保留")
        self.assertEqual(len(gmod_rows), 1, "只读副本必须并列出现")
        self.assertEqual(policy_rows[0]["kind"], "policy")
        self.assertTrue(policy_rows[0]["hasPolicy"])
        self.assertEqual(gmod_rows[0]["kind"], "gmod")
        self.assertTrue(gmod_rows[0]["conflict"])
        self.assertIn("同名", gmod_rows[0]["warning"])
        # 裸 id 指向策略模型（可训练 + 持久化），前缀 id 指向只读副本。
        self.assertEqual(self.store.get("dupe")["kind"], "policy")
        self.assertEqual(self.store.get("gmod:dupe")["kind"], "gmod")

    def test_delete_gmod_returns_409(self) -> None:
        write_gmod(self.dir, "keep-me.gmod")
        with self.assertRaises(Conflict) as ctx:
            self.store.delete("gmod:keep-me")
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn("请删除 models/keep-me.gmod", str(ctx.exception))
        self.assertTrue((self.dir / "keep-me.gmod").is_file(), "用户文件不能被删")

    def test_update_gmod_returns_409(self) -> None:
        write_gmod(self.dir, "read-only.gmod")
        with self.assertRaises(Conflict) as ctx:
            self.store.update({"id": "gmod:read-only", "name": "改名"})
        self.assertEqual(ctx.exception.status, 409)

    def test_detail_gmod_has_package_and_recent_runs(self) -> None:
        write_gmod(self.dir, "detail.gmod", meta=default_meta(id="detail"))
        run = write_run(self.root, "20261108-2303-train-Bot_D")
        (run / "run.json").write_text(json.dumps({
            "runId": run.name, "kind": "train", "modelId": "detail", "modelName": "Detail",
        }), encoding="utf-8")
        detail = self.store.detail("gmod:detail")
        self.assertEqual(detail["kind"], "gmod")
        self.assertIsNone(detail["policyPath"])
        self.assertIn("detail.gmod", detail["packagePath"])
        self.assertEqual([item["runId"] for item in detail["recentRuns"]], [run.name])

    def test_payload_and_reload_payload(self) -> None:
        write_gmod(self.dir, "x.gmod", meta=default_meta(id="x"))
        write_gmod(self.dir, "bad.gmod", raw=b"junk")
        payload = self.store.payload()
        self.assertEqual(payload["scanned"], 2)
        self.assertEqual(len(payload["invalid"]), 1)
        self.assertEqual(Path(payload["gmodDir"]).name, "models")
        reload_payload = self.store.reload_payload()
        self.assertEqual(reload_payload["scanned"], 2)
        self.assertEqual(reload_payload["added"], 2)
        self.assertIn("gmod:x", [item["id"] for item in reload_payload["models"]])


class TrainingWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = self.root / "models"
        self.dir.mkdir()
        self.cfg = _cfg(self.root)
        self.gmod = GmodScanner(self.dir, self.cfg.var_dir / "gmod-cache")
        self.store = ModelStore(self.cfg.var_dir / "models", self.cfg.artifacts_dir, self.gmod)
        self.store.ensure()
        self.runner = TrainingRunner(
            self.cfg, ProcessManager(self.cfg.logs_dir, self.cfg.state_file),
            ArtifactStore(self.cfg.artifacts_dir), models=self.store,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _payload(self, **extra: Any) -> dict:
        payload = {"users": ["Bot_A", "Bot_B"], "ports": [8892, 8893],
                   "plan": [{"opponentMode": "idle", "episodes": 1}]}
        payload.update(extra)
        return payload

    def test_options_include_gmod_with_valid_flag(self) -> None:
        write_gmod(self.dir, "good.gmod", meta=default_meta(id="good"))
        write_gmod(self.dir, "bad.gmod", raw=b"junk")
        options = self.runner.options()
        self.assertTrue(options["modelSupport"])
        models = {item["id"]: item for item in options["models"]}
        self.assertEqual(models["gmod:good"]["kind"], "gmod")
        self.assertTrue(models["gmod:good"]["valid"])
        self.assertFalse(models["gmod:bad"]["valid"], "invalid 必须能被前端识别并置灰")
        self.assertIsNotNone(models["gmod:bad"]["error"])

    def test_invalid_model_cannot_be_selected(self) -> None:
        write_gmod(self.dir, "bad.gmod", raw=b"junk")
        with self.assertRaises(BadRequest) as ctx:
            self.runner._build_train(self._payload(modelId="bad"))
        self.assertIn("不可用", str(ctx.exception))

    def test_training_uses_cached_policy_path(self) -> None:
        write_gmod(self.dir, "good.gmod", meta=default_meta(id="good", name="Good Model"))
        argv, params, _total, _counts = self.runner._build_train(self._payload(modelId="good"))
        self.assertIn("--init-policy", argv)
        path = Path(argv[argv.index("--init-policy") + 1])
        self.assertEqual(path.parent.name, "gmod-cache")
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), npz_bytes())
        self.assertEqual(params["modelId"], "good")
        self.assertEqual(params["modelName"], "Good Model")
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["modelId"], "good")
        self.assertEqual(meta["initPolicy"], str(path))

    def test_multi_also_uses_cached_policy(self) -> None:
        write_gmod(self.dir, "multi.gmod", meta=default_meta(id="multi"))
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8894,
            "plan": [{"opponentMode": "idle", "episodes": 1}], "modelId": "multi",
        })
        self.assertIn("--init-policy", argv)
        self.assertEqual(Path(argv[argv.index("--init-policy") + 1]).parent.name, "gmod-cache")
        self.assertEqual(params["modelId"], "multi")


class _HttpFixture(unittest.TestCase):
    mc_port = 25969

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dir = cls.root / "models"
        cls.dir.mkdir()
        cls.cfg = _cfg(cls.root, mc_port=cls.mc_port)
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
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = request.Request(self.base + path, data=data, method=method,
                              headers={"Content-Type": "application/json"} if data else {})
        try:
            with request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()

    def _json(self, path: str, method: str = "GET", payload: Any = None) -> dict:
        status, body = self._request(path, method, payload)
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        return json.loads(body.decode("utf-8"))


class GmodHttpTests(_HttpFixture):
    def test_list_and_reload_and_delete_409(self) -> None:
        write_gmod(self.dir, "web-good.gmod", meta=default_meta(id="web-good"))
        write_gmod(self.dir, "web-bad.gmod", raw=b"junk")
        data = self._json("/api/models")
        ids = [item["id"] for item in data["models"]]
        self.assertIn("default", ids)
        self.assertIn("gmod:web-good", ids)
        self.assertIn("gmod:web-bad", ids)
        self.assertEqual(data["scanned"], 2)
        self.assertEqual([item["file"] for item in data["invalid"]], ["web-bad.gmod"])
        good = next(item for item in data["models"] if item["id"] == "gmod:web-good")
        self.assertEqual(good["kind"], "gmod")
        self.assertTrue(good["valid"])
        self.assertEqual(good["source"]["file"], "web-good.gmod")

        reloaded = self._json("/api/models/reload", "POST", {})
        self.assertEqual(reloaded["scanned"], 2)
        self.assertEqual(reloaded["added"], 2)

        status, body = self._request("/api/models?id=gmod:web-good", "DELETE")
        self.assertEqual(status, 409)
        self.assertIn("请删除 models/web-good.gmod", json.loads(body)["error"])
        status, body = self._request("/api/models/update", "POST",
                                     {"id": "gmod:web-good", "name": "改名"})
        self.assertEqual(status, 409)

    def test_open_gmod_folder(self) -> None:
        self.opened.clear()
        data = self._json("/api/open-folder", "POST", {"kind": "gmod"})
        self.assertEqual(Path(data["path"]), self.dir.resolve())
        self.assertEqual(self.opened, [data["path"]])

    def test_options_contains_gmod(self) -> None:
        write_gmod(self.dir, "opt.gmod", meta=default_meta(id="opt"))
        options = self._json("/api/training/options")["options"]
        self.assertTrue(options["modelSupport"])
        self.assertIn("gmod:opt", [item["id"] for item in options["models"]])

    def test_partial_episode_from_running_job(self) -> None:
        run_id = "20261108-2310-train-Bot_P"
        write_run(self.root, run_id)  # 有 run.json 但没有 report.json
        job = TrainingJob(
            id="training", kind="train",
            params={"modelId": "default", "modelName": "default（全新初始化）"},
            total_episodes=2, status="running", run_id=run_id,
        )
        job.episodes = [
            {"episode": 1, "steps": 20, "totalReward": 3.5, "winner": "Bot_P",
             "terminated": True, "truncated": False, "startedAt": "2026-11-08T23:10:00",
             "durationSeconds": 12.5},
            {"episode": 2, "steps": 30, "totalReward": -1.0, "winner": None,
             "terminated": False, "truncated": True, "startedAt": "2026-11-08T23:11:00",
             "durationSeconds": 18.0},
        ]
        self.app.training._job = job
        try:
            data = self._json(f"/api/training/episodes?run={run_id}&episode=1")
            self.assertTrue(data["partial"])
            self.assertEqual(data["episodeCount"], 2)
            self.assertEqual(data["winner"], "Bot_P")
            self.assertIsNone(data["review"])
            self.assertEqual(data["model"]["id"], "default")
            status, _body = self._request(f"/api/training/episodes?run={run_id}&episode=3")
            self.assertEqual(status, 400, "内存里也没有第 3 局时仍应 400")
        finally:
            self.app.training._job = None

    def test_missing_run_still_404(self) -> None:
        status, _body = self._request("/api/training/episodes?run=20200101-0000-train-None&episode=1")
        self.assertEqual(status, 404)

    def test_frontend_404_hint_present(self) -> None:
        source = (DASHBOARD_DIR / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("后端可能是旧进程，请重启 Dashboard", source)
        self.assertIn("未知接口", source)


if __name__ == "__main__":
    unittest.main()
