"""P1-1：dashboard 侧的分对手胜率（含旧 run 回退）。"""

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
from artifacts import _by_opponent  # noqa: E402
from config import DashboardConfig  # noqa: E402

USER = "Bot_T1"
HISTORY = [
    {"opponentMode": "scripted", "winner": "Bot_T1", "win": True, "totalReward": 1.0, "steps": 10},
    {"opponentMode": "scripted", "winner": "Bot_T2", "win": False, "totalReward": -1.0, "steps": 10},
    {"opponentMode": "rusher", "winner": None, "win": False, "totalReward": 0.0, "steps": 10},
]


class ByOpponentUnitTest(unittest.TestCase):
    def test_prefers_summary_from_report(self):
        stored = {"scripted": {"episodes": 2, "wins": 2, "winRate": 1.0, "meanReward": 5.0}}
        self.assertEqual(_by_opponent({"byOpponent": stored}, HISTORY, USER), stored)

    def test_computes_from_history_when_summary_missing(self):
        result = _by_opponent(None, HISTORY, USER)
        self.assertEqual(result["scripted"]["episodes"], 2)
        self.assertEqual(result["scripted"]["wins"], 1)
        self.assertAlmostEqual(result["scripted"]["winRate"], 0.5)
        self.assertEqual(result["rusher"]["episodes"], 1)
        self.assertEqual(result["rusher"]["wins"], 0)

    def test_legacy_history_without_opponent_mode_is_empty(self):
        legacy = [{"winner": "Bot_T1", "totalReward": 1.0}]
        self.assertEqual(_by_opponent(None, legacy, USER), {})

    def test_no_train_user_is_empty(self):
        self.assertEqual(_by_opponent(None, HISTORY, None), {})

    def test_non_dict_summary_is_ignored(self):
        result = _by_opponent(["oops"], HISTORY, USER)  # type: ignore[arg-type]
        self.assertEqual(result["scripted"]["wins"], 1)


class ByOpponentHttpTest(unittest.TestCase):
    mc_port = 25994

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        runs = root / "artifacts" / "runs"
        # 新 run：有 summary.byOpponent
        new_run = runs / "20260102-0100-train-Bot_N"
        new_run.mkdir(parents=True)
        (new_run / "run.json").write_text(json.dumps({
            "plan": [{"opponentMode": "scripted", "episodes": 2},
                     {"opponentMode": "rusher", "episodes": 1}],
        }), encoding="utf-8")
        (new_run / "report.json").write_text(json.dumps({
            "kind": "train",
            "config": {"users": [USER, "Bot_T2"]},
            "plan": [{"opponentMode": "scripted", "episodes": 2},
                     {"opponentMode": "rusher", "episodes": 1}],
            "history": HISTORY,
            "summary": {"overall": {"episodes": 3, "wins": 1, "winRate": 0.3333, "meanReward": 0.0},
                        "byOpponent": {"scripted": {"episodes": 2, "wins": 1, "winRate": 0.5,
                                                    "meanReward": 0.0},
                                       "rusher": {"episodes": 1, "wins": 0, "winRate": 0.0,
                                                  "meanReward": 0.0}}},
            "evaluation": [],
        }), encoding="utf-8")
        # 旧 run：没有 summary 也没有 opponentMode
        old_run = runs / "20251231-0100-train-Bot_O"
        old_run.mkdir(parents=True)
        (old_run / "report.json").write_text(json.dumps({
            "kind": "train",
            "config": {"users": [USER, "Bot_T2"]},
            "history": [{"winner": "Bot_T1", "totalReward": 1.0, "steps": 5}],
            "evaluation": [],
        }), encoding="utf-8")

        cls.root = root
        cfg = DashboardConfig(
            host="127.0.0.1", port=8080, artifacts_dir=root / "artifacts",
            var_dir=root / "var", server_dir=root / "server",
            bot_runtime_dir=root / "bot-runtime", training_dir=root / "training",
            python_exe=sys.executable, java_exe="java", node_exe="node",
            allow_process_control=False, scan_external=False,
            server_jar="missing.jar", mc_port=cls.mc_port,
        )
        cls.app = DashboardApp(cfg)
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

    def test_run_list_carries_by_opponent(self):
        runs = {item["id"]: item for item in self._json("/api/artifacts/runs")["runs"]}
        newest = runs["20260102-0100-train-Bot_N"]
        self.assertEqual(newest["byOpponent"]["scripted"]["wins"], 1)
        self.assertAlmostEqual(newest["byOpponent"]["rusher"]["winRate"], 0.0)

        legacy = runs["20251231-0100-train-Bot_O"]
        self.assertEqual(legacy["byOpponent"], {})  # 旧 run 不报错，返回空表

    def test_run_detail_carries_by_opponent(self):
        payload = self._json("/api/artifacts/run?id=20260102-0100-train-Bot_N")
        detail = payload.get("run") or {}
        self.assertEqual(detail["byOpponent"]["scripted"]["episodes"], 2)
        self.assertEqual(detail["byOpponent"]["rusher"]["wins"], 0)

    def test_legacy_run_detail_returns_empty(self):
        payload = self._json("/api/artifacts/run?id=20251231-0100-train-Bot_O")
        detail = payload.get("run") or {}
        self.assertEqual(detail.get("byOpponent"), {})


if __name__ == "__main__":
    unittest.main()
