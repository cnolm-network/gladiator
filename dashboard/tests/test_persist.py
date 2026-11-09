"""G-624：持久化训练（训练结束把最终权重写回所选模型）。

全部使用临时目录 + 假训练进程（SimpleNamespace）/ 伪造产物，**不启动真实训练**、不碰
真实 `dashboard/var`、不连 Paper。npz / gmod 用手写 zip，不依赖 numpy。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from artifacts import ArtifactStore  # noqa: E402
from config import DashboardConfig  # noqa: E402
from errors import Conflict  # noqa: E402
from gmod import GmodScanner  # noqa: E402
from model_locks import ModelLock  # noqa: E402
from models import ModelStore  # noqa: E402
from processes import ProcessManager  # noqa: E402
from tests.test_gmod import default_meta, npz_bytes, write_gmod  # noqa: E402
from training_runner import TrainingJob, TrainingRunner  # noqa: E402


def _cfg(root: Path, *, gmod_dir: Path | None = None) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1", port=8080,
        artifacts_dir=root / "artifacts", var_dir=root / "var",
        server_dir=root / "server", bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training", gmod_dir=gmod_dir or root / "models",
        python_exe=sys.executable, java_exe="java", node_exe="node",
        allow_process_control=True, scan_external=False,
        server_jar="missing.jar", mc_port=25960,
    )


class _Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gdir = self.root / "models"
        self.gdir.mkdir()
        self.cfg = _cfg(self.root, gmod_dir=self.gdir)
        self.scanner = GmodScanner(self.gdir, self.cfg.var_dir / "gmod-cache")
        self.store = ModelStore(self.cfg.var_dir / "models", self.cfg.artifacts_dir,
                                self.scanner)
        self.store.ensure()
        self.runner = TrainingRunner(
            self.cfg, ProcessManager(self.cfg.logs_dir, self.cfg.state_file),
            ArtifactStore(self.cfg.artifacts_dir), models=self.store,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def policy(self, name: str = "run-policy.npz") -> Path:
        path = self.root / name
        path.write_bytes(npz_bytes())
        return path

    def payload(self, **extra):
        base = {"users": ["Bot_A", "Bot_B"], "ports": [8892, 8893],
                "plan": [{"opponentMode": "idle", "episodes": 1}]}
        base.update(extra)
        return base

    def job(self, *, status: str = "running", code: int = 0,
            stop_requested: bool = False, persist: bool = True,
            model_id: str = "default", with_policy: bool = True,
            kind: str = "train") -> TrainingJob:
        run_dir = self.cfg.artifacts_dir / "runs" / "20261109-0040-train-Bot_A"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({"runId": run_dir.name}), encoding="utf-8")
        if with_policy:
            (run_dir / "policy.npz").write_bytes(npz_bytes())
        params = {"persist": persist, "persistModelId": model_id if persist else None,
                  "runDir": str(run_dir), "runId": run_dir.name}
        job = TrainingJob(
            id="training", kind=kind, params=params, total_episodes=2,
            status=status, run_id=run_dir.name, run_dir=str(run_dir),
            report_path=str(run_dir / "report.json"),
            auto_cleanup=False, stop_requested=stop_requested,
            persist={"enabled": persist, "modelId": model_id if persist else None,
                     "ok": None, "error": None},
        )
        job.episodes = [{"episode": 1, "steps": 20, "totalReward": 3.0},
                        {"episode": 2, "steps": 30, "totalReward": 5.0}]
        job.rewards = [3.0, 5.0]
        job.steps = 50
        job.episodes_completed = 2
        if persist:
            self.runner.model_locks.acquire(model_id, run_dir.name)
        return job

    @staticmethod
    def proc(code: int = 0, tail: list[str] | None = None):
        return SimpleNamespace(exit_code=code, log_path=None, log_offset=0,
                               snapshot=lambda _n: {"tail": tail or []})


class WriteBackTests(_Fixture):
    def test_legacy_default_name_is_migrated(self) -> None:
        path = self.cfg.var_dir / "models" / "default" / "model.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["name"] = "default（全新初始化）"
        data["description"] = "不加载任何权重，按 seed 随机初始化后再训练。"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.store.ensure()
        model = self.store.get("default")
        self.assertIn("持久化训练", model["name"])
        self.assertNotIn("全新初始化", model["description"])

    def test_first_write_back_creates_default_policy(self) -> None:
        target = self.cfg.var_dir / "models" / "default" / "policy.npz"
        self.assertFalse(target.is_file())
        result = self.store.write_back("default", self.policy(), {
            "runId": "run-1", "episodes": 2, "steps": 50, "meanReward": 4.0})
        self.assertTrue(result["ok"], result.get("error"))
        self.assertTrue(target.is_file())
        self.assertFalse((target.parent / "policy.npz.tmp").exists(), "临时文件必须清理")
        self.assertEqual(result["training"]["totalEpisodes"], 2)
        self.assertEqual(result["training"]["totalSteps"], 50)
        self.assertEqual(result["training"]["lastMeanReward"], 4.0)
        self.assertEqual(result["training"]["lastRunId"], "run-1")
        self.assertEqual(self.store.get("default")["episodes"], 2)

    def test_second_write_back_accumulates_and_backs_up(self) -> None:
        first = self.policy("p1.npz")
        second = self.policy("p2.npz")
        r1 = self.store.write_back("default", first, {"runId": "r1", "episodes": 2, "steps": 40})
        r2 = self.store.write_back("default", second, {"runId": "r2", "episodes": 3, "steps": 60})
        self.assertTrue(r1["ok"] and r2["ok"])
        self.assertIsNone(r1["backup"], "首次写回没有旧文件，不应有备份")
        self.assertIsNotNone(r2["backup"], "覆盖前必须备份旧权重")
        self.assertTrue(Path(r2["backup"]).is_file())
        self.assertEqual(Path(r2["backup"]).read_bytes(), first.read_bytes())
        training = r2["training"]
        self.assertEqual(training["totalEpisodes"], 5, "局数必须累加")
        self.assertEqual(training["totalSteps"], 100, "步数必须累加")
        self.assertEqual(training["writeCount"], 2)
        self.assertEqual([item["runId"] for item in training["runs"]], ["r1", "r2"])
        self.assertEqual(self.store.get("default")["episodes"], 5)

    def test_training_json_keeps_only_last_20_runs(self) -> None:
        policy = self.policy()
        for index in range(22):
            result = self.store.write_back("default", policy, {
                "runId": f"r{index}", "episodes": 1, "steps": 10})
            self.assertTrue(result["ok"])
        training = self.store.training_stats("default")
        self.assertEqual(len(training["runs"]), 20)
        self.assertEqual(training["runs"][0]["runId"], "r2")
        self.assertEqual(training["runs"][-1]["runId"], "r21")
        self.assertEqual(training["totalEpisodes"], 22)

    def test_atomic_replace_leaves_no_tmp(self) -> None:
        directory = self.cfg.var_dir / "models" / "default"
        self.store.write_back("default", self.policy(), {"runId": "r", "episodes": 1, "steps": 1})
        self.assertTrue((directory / "policy.npz").is_file())
        self.assertFalse((directory / "policy.npz.tmp").exists())
        self.assertEqual((directory / "policy.npz").read_bytes(), npz_bytes())

    def test_missing_policy_returns_error_without_raising(self) -> None:
        result = self.store.write_back("default", self.root / "nope.npz",
                                       {"runId": "r", "episodes": 1, "steps": 1})
        self.assertFalse(result["ok"])
        self.assertIn("训练产物不存在", result["error"])

    def test_invalid_npz_returns_error_without_raising(self) -> None:
        bad = self.root / "bad.npz"
        bad.write_bytes(b"not a zip")
        result = self.store.write_back("default", bad,
                                       {"runId": "r", "episodes": 1, "steps": 1})
        self.assertFalse(result["ok"])
        self.assertIn("写回失败", result["error"])

    def test_gmod_write_back_rejected(self) -> None:
        write_gmod(self.gdir, "file.gmod", meta=default_meta(id="file"))
        result = self.store.write_back("file", self.policy(),
                                       {"runId": "r", "episodes": 1, "steps": 1})
        self.assertFalse(result["ok"])
        self.assertIn("只读", result["error"])
        self.assertFalse((self.cfg.var_dir / "models" / "file" / "policy.npz").exists())

    def test_disk_error_is_captured_not_raised(self) -> None:
        with mock.patch("models.shutil.copy2", side_effect=OSError("磁盘已满")):
            result = self.store.write_back("default", self.policy(),
                                           {"runId": "r", "episodes": 1, "steps": 1})
        self.assertFalse(result["ok"])
        self.assertIn("磁盘已满", result["error"])

    def test_default_becomes_init_policy_after_first_write_back(self) -> None:
        self.assertIsNone(self.runner._model_from_payload({"modelId": "default"})["path"])
        self.store.write_back("default", self.policy(), {"runId": "r", "episodes": 1, "steps": 1})
        path = self.runner._model_from_payload({"modelId": "default"})["path"]
        self.assertEqual(Path(path), self.cfg.var_dir / "models" / "default" / "policy.npz")
        argv, params, _total, _counts = self.runner._build_train(self.payload())
        self.assertEqual(argv[argv.index("--init-policy") + 1], path)
        self.assertEqual(params["persistModelId"], "default")


class LockTests(_Fixture):
    def test_lock_conflict_returns_409(self) -> None:
        self.runner.model_locks.acquire("default", "other-run")
        with self.assertRaises(Conflict) as ctx:
            with mock.patch.object(self.runner, "_launch", lambda *args, **kwargs: None):
                self.runner.start(self.payload())
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn("占用", str(ctx.exception))

    def test_stale_lock_is_recovered_with_warning(self) -> None:
        self.runner.model_locks = ModelLock(self.cfg.var_dir / "models",
                                            pid_check=lambda _pid: False)
        self.runner.model_locks.acquire("default", "dead-run")
        with mock.patch.object(self.runner, "_launch", lambda *args, **kwargs: None):
            state = self.runner.start(self.payload())
        notes = state["job"]["notes"]
        self.assertTrue(any("回收" in note and "过期锁" in note for note in notes))
        self.assertEqual(state["job"]["persist"]["modelId"], "default")

    def test_release_only_removes_own_lock(self) -> None:
        locks = ModelLock(self.cfg.var_dir / "models")
        locks.acquire("default", "run-A")
        self.assertFalse(locks.release("default", "run-B"))
        self.assertIsNotNone(locks.peek("default"))
        self.assertTrue(locks.release("default", "run-A"))
        self.assertIsNone(locks.peek("default"))


class ExitFlowTests(_Fixture):
    def test_finished_writes_back(self) -> None:
        job = self.job()
        self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "finished")
        self.assertTrue(job.persist["ok"], job.persist.get("error"))
        self.assertTrue((self.cfg.var_dir / "models" / "default" / "policy.npz").is_file())
        meta = json.loads((Path(job.run_dir) / "run.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["persist"]["ok"])
        self.assertIsNone(self.runner.model_locks.peek("default"), "终态必须释放锁")

    def test_failed_does_not_write_back(self) -> None:
        job = self.job()
        self.runner._handle_exit(job, self.proc(1, ["Traceback: boom"]))
        self.assertEqual(job.status, "failed")
        self.assertFalse(job.persist["ok"])
        self.assertTrue(job.persist["skipped"])
        self.assertFalse((self.cfg.var_dir / "models" / "default" / "policy.npz").exists())
        self.assertIsNone(self.runner.model_locks.peek("default"))

    def test_stopped_does_not_write_back(self) -> None:
        job = self.job(stop_requested=True)
        self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "stopped")
        self.assertFalse(job.persist["ok"])
        self.assertFalse((self.cfg.var_dir / "models" / "default" / "policy.npz").exists())
        self.assertIsNone(self.runner.model_locks.peek("default"))

    def test_write_back_failure_keeps_job_finished(self) -> None:
        job = self.job()
        with mock.patch.object(self.store, "write_back",
                               return_value={"ok": False, "modelId": "default",
                                             "error": "模型写回失败: 权限不足"}):
            self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "finished", "写回失败不能让 job 变 failed")
        self.assertFalse(job.persist["ok"])
        self.assertIn("权限不足", job.persist["error"])
        self.assertTrue(any("写回失败" in note for note in job.notes))
        meta = json.loads((Path(job.run_dir) / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["persist"]["ok"])

    def test_fail_before_spawn_releases_lock_and_marks_skipped(self) -> None:
        job = self.job()
        self.runner._fail_before_spawn(job, "端口 8892 没有 runtime")
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.persist["skipped"])
        self.assertIsNone(self.runner.model_locks.peek("default"))

    def test_finish_before_spawn_releases_lock(self) -> None:
        job = self.job(stop_requested=True)
        self.runner._finish_before_spawn(job)
        self.assertEqual(job.status, "stopped")
        self.assertIsNone(self.runner.model_locks.peek("default"))


class MultiMergePersistTests(_Fixture):
    """G-625：多场地融合后的写回，以及融合失败时的自动降级。"""

    def _multi_job(self, *, merge: dict, with_policy: bool = True) -> TrainingJob:
        run_dir = self.cfg.artifacts_dir / "runs" / "20261109-0100-multi-Bot_M"
        (run_dir / "reviews").mkdir(parents=True, exist_ok=True)
        history = [
            {"episode": 1, "steps": 20, "totalReward": 3.0, "opponentMode": "idle"},
            {"episode": 2, "steps": 30, "totalReward": 5.0, "opponentMode": "rusher"},
        ]
        (run_dir / "report.json").write_text(json.dumps({
            "kind": "multi", "history": history,
            "summary": {"episodes": 2, "totalSteps": 50, "meanReward": 4.0},
            "pairs": [{"pair": 1, "steps": 20}, {"pair": 2, "steps": 30}],
            "merge": merge,
        }, ensure_ascii=False), encoding="utf-8")
        (run_dir / "run.json").write_text(json.dumps({"runId": run_dir.name, "kind": "multi"}),
                                          encoding="utf-8")
        if with_policy:
            (run_dir / "policy.npz").write_bytes(npz_bytes())
        params = {"persist": True, "persistModelId": "default",
                  "runDir": str(run_dir), "runId": run_dir.name}
        job = TrainingJob(
            id="training", kind="multi", params=params, total_episodes=2,
            status="running", run_id=run_dir.name, run_dir=str(run_dir),
            report_path=str(run_dir / "report.json"),
            auto_cleanup=False,
            persist={"enabled": True, "modelId": "default", "ok": None, "error": None},
        )
        job.episodes = history
        job.rewards = [3.0, 5.0]
        job.steps = 50
        job.episodes_completed = 2
        self.runner.model_locks.acquire("default", run_dir.name)
        return job

    def test_merged_policy_is_written_back(self) -> None:
        merge = {
            "method": "mean", "available": True, "skipped": False,
            "pairs": [{"pair": 1, "steps": 20, "sha256": "a" * 64},
                      {"pair": 2, "steps": 30, "sha256": "b" * 64}],
            "mergedSha256": "c" * 64,
        }
        job = self._multi_job(merge=merge)
        self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "finished")
        self.assertTrue(job.persist["ok"], job.persist.get("error"))
        self.assertTrue(job.persist["merged"], "多场地写回必须标记 merged")
        self.assertEqual(job.persist["mergePairCount"], 2)
        self.assertEqual(job.persist["mergeMethod"], "mean")
        # 写回的是融合产物 run/policy.npz；累计口径取所有 pair 之和。
        target = self.cfg.var_dir / "models" / "default" / "policy.npz"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), (Path(job.run_dir) / "policy.npz").read_bytes())
        self.assertEqual(job.persist["training"]["totalEpisodes"], 2)
        self.assertEqual(job.persist["training"]["totalSteps"], 50)
        self.assertIsNone(self.runner.model_locks.peek("default"), "终态必须释放锁")
        meta = json.loads((Path(job.run_dir) / "run.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["persist"]["merged"])
        self.assertEqual(meta["persist"]["mergePairCount"], 2)

    def test_merge_failure_degrades_without_writing_back(self) -> None:
        merge = {"method": "mean", "available": False, "skipped": True,
                 "pairs": [], "mergedSha256": None,
                 "error": "所有场地都失败，未产出融合权重"}
        job = self._multi_job(merge=merge, with_policy=False)
        self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "finished", "融合失败不能让 job 变 failed")
        self.assertFalse(job.persist["ok"])
        self.assertTrue(job.persist["degraded"], "融合失败应标记 degraded")
        self.assertTrue(job.persist["skipped"])
        self.assertIn("融合失败", job.persist["error"])
        self.assertIn("融合失败", job.persist_warning)
        self.assertFalse((self.cfg.var_dir / "models" / "default" / "policy.npz").exists())
        self.assertIsNone(self.runner.model_locks.peek("default"))

    def test_missing_merge_section_also_degrades(self) -> None:
        # 报告里完全没有 merge 段（旧版本 multi / 报告损坏）→ 同样降级，不写回不存在的 policy。
        job = self._multi_job(merge={}, with_policy=False)
        report_path = Path(job.run_dir) / "report.json"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        data.pop("merge")
        report_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.runner._handle_exit(job, self.proc(0))
        self.assertEqual(job.status, "finished")
        self.assertFalse(job.persist["ok"])
        self.assertTrue(job.persist["degraded"])


class ParameterTests(_Fixture):
    def test_train_defaults_persist_true(self) -> None:
        argv, params, _total, _counts = self.runner._build_train(self.payload())
        self.assertTrue(params["persist"])
        self.assertEqual(params["persistModelId"], "default")
        self.assertEqual(argv[argv.index("--persist-to") + 1], "default")
        self.assertIn("--model-id", argv)

    def test_persist_false_is_readonly_start(self) -> None:
        argv, params, _total, _counts = self.runner._build_train(
            self.payload(persist=False))
        self.assertFalse(params["persist"])
        self.assertIsNone(params["persistModelId"])
        self.assertNotIn("--persist-to", argv)

    def test_gmod_disables_persist(self) -> None:
        write_gmod(self.gdir, "file.gmod", meta=default_meta(id="file"))
        argv, params, _total, _counts = self.runner._build_train(
            self.payload(modelId="file", persist=True))
        self.assertFalse(params["persist"])
        self.assertIn("只读", params["persistWarning"])
        self.assertNotIn("--persist-to", argv)
        self.assertEqual(Path(argv[argv.index("--init-policy") + 1]).parent.name, "gmod-cache")

    def test_same_id_gmod_does_not_break_persistable_model(self) -> None:
        """G-626 回归：同名 .gmod 存在时，选 default + 持久化仍写回策略模型。"""
        # 让 default 有 policy.npz，再放一个同名 .gmod（导出/手工放入）。
        (self.cfg.var_dir / "models" / "default" / "policy.npz").write_bytes(npz_bytes())
        write_gmod(self.gdir, "default.gmod", meta=default_meta(id="default"))
        argv, params, _total, _counts = self.runner._build_train(self.payload())
        self.assertTrue(params["persist"], "同名 gmod 不能把可持久化模型降级为只读")
        self.assertEqual(params["persistModelId"], "default")
        self.assertIn("--persist-to", argv)
        init = Path(argv[argv.index("--init-policy") + 1])
        self.assertEqual(init, self.cfg.var_dir / "models" / "default" / "policy.npz",
                         "训练起点必须是策略模型的权重，不是 gmod-cache")

    def test_gmod_prefixed_id_degrades_persist(self) -> None:
        """选 ``gmod:<stem>`` 只读副本：可作起点，但 persist 自动降级、不报错。"""
        write_gmod(self.gdir, "file.gmod", meta=default_meta(id="file"))
        argv, params, _total, _counts = self.runner._build_train(
            self.payload(modelId="gmod:file"))
        self.assertFalse(params["persist"])
        self.assertIsNone(params["persistModelId"])
        self.assertIn("只读", params["persistWarning"])
        self.assertNotIn("--persist-to", argv)
        init = Path(argv[argv.index("--init-policy") + 1])
        self.assertEqual(init.parent.name, "gmod-cache")
        self.assertEqual(params["modelId"], "gmod:file")

    def test_multi_merge_none_disables_persist_with_reason(self) -> None:
        # G-625：merge=none 时多场地不允许持久化，给出「需要开启融合」的说明。
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8894,
            "plan": [{"opponentMode": "idle", "episodes": 1}], "persist": True,
            "merge": "none",
        })
        self.assertFalse(params["persist"])
        self.assertFalse(params["persistSupported"])
        self.assertIn("融合", params["persistWarning"])
        self.assertNotIn("--persist-to", argv)
        self.assertIn("--merge", argv)
        self.assertEqual(argv[argv.index("--merge") + 1], "none")

    def test_multi_mean_merge_enables_persist(self) -> None:
        # G-625：merge=mean（默认）时多场地允许持久化，写回融合后的权重。
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8894,
            "plan": [{"opponentMode": "idle", "episodes": 1}], "persist": True,
            "merge": "mean",
        })
        self.assertTrue(params["persist"])
        self.assertTrue(params["persistSupported"])
        self.assertIsNone(params["persistWarning"])
        self.assertEqual(params["persistModelId"], "default")
        self.assertIn("--persist-to", argv)
        self.assertEqual(argv[argv.index("--persist-to") + 1], "default")
        self.assertIn("--merge", argv)
        self.assertEqual(argv[argv.index("--merge") + 1], "mean")

    def test_multi_default_merge_enables_persist(self) -> None:
        # 不显式传 merge 时默认 mean，行为与上式一致（向后兼容旧的 multi 调用）。
        argv, params, _total, _counts = self.runner._build_multi({
            "pairs": 1, "usersPrefix": "Bot_P", "portsBase": 8894,
            "plan": [{"opponentMode": "idle", "episodes": 1}], "persist": True,
        })
        self.assertTrue(params["persist"])
        self.assertTrue(params["persistSupported"])
        self.assertIn("--merge", argv)
        self.assertEqual(argv[argv.index("--merge") + 1], "mean")

    def test_import_gmod_creates_persistable_policy_model(self) -> None:
        write_gmod(self.gdir, "file.gmod", meta=default_meta(id="file"))
        created = self.store.create({"name": "File Copy", "gmodId": "file"})
        self.assertNotEqual(created["kind"], "gmod")
        self.assertTrue((self.cfg.var_dir / "models" / created["id"] / "policy.npz").is_file())
        argv, params, _total, _counts = self.runner._build_train(
            self.payload(modelId=created["id"]))
        self.assertTrue(params["persist"])
        self.assertEqual(params["persistModelId"], created["id"])
        self.assertEqual(argv[argv.index("--init-policy") + 1],
                         str(self.cfg.var_dir / "models" / created["id"] / "policy.npz"))
        self.assertEqual(argv[argv.index("--persist-to") + 1], created["id"])

    def test_snapshot_exposes_persist_and_warning(self) -> None:
        job = self.job()
        job.persist_warning = "示例"
        snapshot = job.snapshot([])
        self.assertIn("persist", snapshot)
        self.assertEqual(snapshot["persistWarning"], "示例")


class HttpContractTests(unittest.TestCase):
    """只读/导入类 HTTP 契约；不启动训练。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls._root = Path(cls._tmp.name)
        cls._gdir = cls._root / "models"
        cls._gdir.mkdir()
        cls._app = DashboardApp(_cfg(cls._root, gmod_dir=cls._gdir))
        cls._httpd = make_server(cls._app, "127.0.0.1", 0)
        cls._port = cls._httpd.server_address[1]
        cls._thread = __import__("threading").Thread(
            target=cls._httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._httpd.shutdown()
        cls._httpd.server_close()
        cls._app.close()
        cls._thread.join(timeout=5)
        cls._tmp.cleanup()

    def _get(self, path: str) -> dict:
        from urllib import request

        with request.urlopen(f"http://127.0.0.1:{self._port}{path}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        from urllib import request

        req = request.Request(f"http://127.0.0.1:{self._port}{path}",
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              method="POST", headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_models_list_has_training_field(self) -> None:
        data = self._get("/api/models")
        default = next(item for item in data["models"] if item["id"] == "default")
        self.assertEqual(default["training"]["totalEpisodes"], 0)
        self.assertFalse(default["hasPolicy"])

    def test_import_gmod_via_http(self) -> None:
        write_gmod(self._gdir, "http-file.gmod", meta=default_meta(id="http-file"))
        created = self._post("/api/models", {"name": "HTTP Copy", "gmodId": "http-file"})["model"]
        self.assertNotEqual(created["kind"], "gmod")
        self.assertTrue(created["hasPolicy"])
        listed = {item["id"]: item for item in self._get("/api/models")["models"]}
        self.assertIn(created["id"], listed)
        self.assertEqual(listed[created["id"]]["training"]["totalEpisodes"], 0)

    def test_run_detail_exposes_persist(self) -> None:
        run_dir = self._cfg_run()
        data = self._get(f"/api/artifacts/run?id={run_dir.name}")
        self.assertEqual(data["run"]["persist"]["ok"], True)
        self.assertEqual(data["run"]["persist"]["modelId"], "default")

    def _cfg_run(self) -> Path:
        run_dir = self._root / "artifacts" / "runs" / "20261109-0050-train-Bot_H"
        (run_dir / "reviews").mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({
            "runId": run_dir.name, "kind": "train", "modelId": "default",
            "persist": {"enabled": True, "modelId": "default", "ok": True},
        }, ensure_ascii=False), encoding="utf-8")
        (run_dir / "report.json").write_text(json.dumps({
            "kind": "train", "history": [], "summary": {}, "config": {},
        }), encoding="utf-8")
        return run_dir


if __name__ == "__main__":
    unittest.main()
