"""B1：评估回合口径（evalRounds 每阶段 / evalEpisodes 派生总数）。

覆盖两层：
1. 构造级 —— ``normalize_eval`` / ``eval_rounds_from_payload`` / ``_build_train`` 的响应与 run.json；
2. HTTP 级 —— ``/api/training`` 与 ``/api/artifacts/runs`` 的返回，含旧 run 的回退。
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib import request

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import DashboardApp, make_server  # noqa: E402
from config import DashboardConfig  # noqa: E402
from eval_semantics import eval_rounds_from_payload, normalize_eval  # noqa: E402
from training_runner import TrainingJob  # noqa: E402

PLAN_3 = [
    {"opponentMode": "scripted", "episodes": 6},
    {"opponentMode": "rusher", "episodes": 6},
    {"opponentMode": "aimbot", "episodes": 4},
]
PLAN_1 = [{"opponentMode": "idle", "episodes": 3}]


class NormalizeEvalTest(unittest.TestCase):
    def test_new_field_wins(self):
        counts = normalize_eval({"evalRounds": 2, "evalEpisodes": 99}, PLAN_3)
        self.assertEqual(counts["evalRounds"], 2)
        self.assertEqual(counts["evalEpisodes"], 32)  # 2 回合 × Σ(6+6+4)
        self.assertEqual(counts["stages"], 3)

    def test_legacy_eval_epochs_is_equivalent(self):
        counts = normalize_eval({"evalEpochs": 4}, PLAN_3)
        self.assertEqual(counts["evalRounds"], 4)
        self.assertEqual(counts["evalEpisodes"], 64)  # 4 回合 × 16

    def test_legacy_total_episodes_falls_back_by_stage_count(self):
        # 旧 run：evalEpisodes=16 是「总局数」，3 阶段 → 每阶段 6（向上取整）
        counts = normalize_eval({"evalEpisodes": 16}, PLAN_3)
        self.assertEqual(counts["evalRounds"], 6)
        self.assertEqual(counts["evalEpisodes"], 16)

    def test_empty_meta_is_zero(self):
        counts = normalize_eval({}, PLAN_3)
        self.assertEqual(counts["evalRounds"], 0)
        self.assertEqual(counts["evalEpisodes"], 0)

    def test_zero_rounds_is_preserved(self):
        counts = normalize_eval({"evalRounds": 0, "evalEpisodes": 16}, PLAN_3)
        self.assertEqual(counts["evalRounds"], 0)
        self.assertEqual(counts["evalEpisodes"], 0)


class PayloadParsingTest(unittest.TestCase):
    def test_priority(self):
        self.assertEqual(eval_rounds_from_payload({"evalRounds": 2, "evalEpochs": 5}, PLAN_3), 2)
        self.assertEqual(eval_rounds_from_payload({"evalEpochs": 5}, PLAN_3), 5)
        self.assertEqual(eval_rounds_from_payload({}, PLAN_3, default=1), 1)

    def test_legacy_total_divided_by_stages(self):
        self.assertEqual(eval_rounds_from_payload({"evalEpisodes": 16}, PLAN_3), 6)
        self.assertEqual(eval_rounds_from_payload({"evalEpisodes": 4}, PLAN_1), 4)


class _Fixture(unittest.TestCase):
    mc_port = 25993

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
            mc_port=cls.mc_port,
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

    def _json(self, path: str) -> dict:
        with request.urlopen(self.base + path, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


class BuildTrainTest(_Fixture):
    def test_three_stage_plan_derives_total(self):
        argv, params, _total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8794, 8795],
            "plan": PLAN_3,
            "epochs": 1,
            "evalRounds": 2,
        })
        self.assertEqual(params["evalRounds"], 2)
        self.assertEqual(params["evalEpisodes"], 32)
        self.assertNotIn("evalEpochs", params)
        self.assertEqual(argv[argv.index("--eval-epochs") + 1], "2")
        meta = json.loads((Path(params["runDir"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["evalRounds"], 2)
        self.assertEqual(meta["evalEpisodes"], 32)

    def test_legacy_payload_eval_epochs_still_accepted(self):
        _argv, params, _total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8796, 8797],
            "plan": PLAN_3,
            "epochs": 1,
            "evalEpochs": 1,
        })
        self.assertEqual(params["evalRounds"], 1)
        self.assertEqual(params["evalEpisodes"], 16)

    def test_zero_disables_eval(self):
        argv, params, _total, _counts = self.app.training._build_train({
            "users": ["Bot_T1", "Bot_T2"],
            "ports": [8798, 8799],
            "plan": PLAN_1,
            "epochs": 1,
            "evalRounds": 0,
        })
        self.assertEqual(params["evalRounds"], 0)
        self.assertEqual(params["evalEpisodes"], 0)
        self.assertEqual(argv[argv.index("--eval-epochs") + 1], "0")


class HttpEvalCountsTest(_Fixture):
    def test_training_response_carries_both_fields(self):
        self.app.training._job = TrainingJob(
            id="job-eval-1",
            kind="train",
            # 旧结构：只有 evalEpochs / evalEpisodes
            params={"plan": PLAN_3, "evalEpochs": 1, "evalEpisodes": 16},
            total_episodes=16,
        )
        job = self._json("/api/training")["job"]
        self.assertEqual(job["params"]["evalRounds"], 1)
        self.assertEqual(job["params"]["evalEpisodes"], 16)

    def test_new_job_params_untouched(self):
        self.app.training._job = TrainingJob(
            id="job-eval-2",
            kind="train",
            params={"plan": PLAN_3, "evalRounds": 2, "evalEpisodes": 6},
            total_episodes=16,
        )
        job = self._json("/api/training")["job"]
        self.assertEqual(job["params"]["evalRounds"], 2)
        self.assertEqual(job["params"]["evalEpisodes"], 6)

    def test_legacy_run_falls_back_in_artifact_list(self):
        run_dir = self.root / "artifacts" / "runs" / "20260101-0100-train-Bot_E"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({
            "plan": PLAN_3,
            "evalEpisodes": 16,
        }), encoding="utf-8")
        (run_dir / "report.json").write_text(json.dumps({
            "kind": "train",
            "plan": PLAN_3,
            "summary": {"episodes": 16, "meanReward": -1.0},
            "history": [],
        }), encoding="utf-8")

        runs = self._json("/api/artifacts/runs")["runs"]
        target = next(item for item in runs if item["id"] == "20260101-0100-train-Bot_E")
        self.assertEqual(target["evalRounds"], 6)     # ceil(16 / 3)
        self.assertEqual(target["evalEpisodes"], 16)  # 旧值原样保留
        self.assertEqual(target["evalStages"], 3)


if __name__ == "__main__":
    unittest.main()
