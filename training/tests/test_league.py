"""联赛评估器（``scripts/league.py``）的统计与编排。

纯离线：``run_league`` 的对局函数是注入的（``play_episode`` 只回答「哪一边赢」），
所以不需要服务端 / RCON 就能验证矩阵、交替左右位、排名、Wilson 区间与显著性判定。

覆盖：
- 赛程：所有两两组合、每对局数、奇偶局交换左右位；
- 矩阵双向对称（A 视角的胜 = B 视角的负）；
- 排名按总胜率；
- 每对 Wilson 95% CI 与精确二项检验；
- 全平局不产生「显著更强」；
- v2 模型被拒绝（与 ``mirror_eval`` 一致）；
- markdown 表格与 JSON 文件名。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.eval_suite import wilson_interval  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from scripts.league import (  # noqa: E402
    build_schedule,
    format_markdown,
    load_policies,
    run_league,
    summarize_rows,
    unique_labels,
    write_summary,
)


def strength_player(strength: dict[str, int]):
    """按「实力」决定胜负的对局函数：强的那一边赢（与座位无关）。"""

    def play(left_path: str, right_path: str, _seed: int, _max_steps: int,
             _greedy: bool = False) -> str:
        left = Path(left_path).stem
        right = Path(right_path).stem
        if strength[left] == strength[right]:
            return "draw"
        return "left" if strength[left] > strength[right] else "right"

    return play


class ScheduleTest(unittest.TestCase):
    def test_alternates_seats_within_pair(self):
        schedule = build_schedule(["a", "b"], 4)
        self.assertEqual(len(schedule), 4)
        self.assertEqual([item["left"] for item in schedule], ["a", "b", "a", "b"])
        self.assertEqual({item["a"] for item in schedule}, {"a"})
        self.assertEqual({item["b"] for item in schedule}, {"b"})

    def test_covers_every_pair(self):
        schedule = build_schedule(["a", "b", "c"], 2)
        pairs = {(item["a"], item["b"]) for item in schedule}
        self.assertEqual(pairs, {("a", "b"), ("a", "c"), ("b", "c")})
        self.assertEqual(len(schedule), 6)


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.paths = {label: f"/tmp/{label}.npz" for label in ("a", "b", "c")}
        self.play = strength_player({"a": 3, "b": 2, "c": 1})
        self.summary = run_league(self.paths, 4, self.play, seed=7, verbose=False)

    def test_matrix_is_symmetric(self):
        matrix = self.summary["matrix"]
        for left in ("a", "b", "c"):
            for right in ("a", "b", "c"):
                if left == right:
                    continue
                cell, mirror = matrix[left][right], matrix[right][left]
                self.assertEqual(cell["wins"], mirror["losses"])
                self.assertEqual(cell["losses"], mirror["wins"])
                self.assertEqual(cell["draws"], mirror["draws"])
                self.assertEqual(cell["episodes"], 4)

    def test_total_games_match_schedule(self):
        self.assertEqual(len(self.summary["rows"]), 3 * 4)

    def test_ranking_is_by_win_rate(self):
        labels = [item["label"] for item in self.summary["ranking"]]
        self.assertEqual(labels, ["a", "b", "c"])
        self.assertEqual([item["rank"] for item in self.summary["ranking"]], [1, 2, 3])
        best = self.summary["ranking"][0]
        self.assertEqual(best["wins"], 8)      # 4 局 vs b + 4 局 vs c，全胜
        self.assertEqual(best["losses"], 0)
        self.assertAlmostEqual(best["winRate"], 1.0)

    def test_win_rate_uses_all_episodes_as_denominator(self):
        """胜率口径与 mirror_eval 一致：wins / 总局数（平局算在分母里）。"""
        for pair in self.summary["pairs"]:
            self.assertAlmostEqual(pair["aWinRate"],
                                   pair["aWins"] / pair["episodes"], places=6)


class StatisticsTest(unittest.TestCase):
    def test_wilson_interval_per_pair(self):
        paths = {"a": "/tmp/a.npz", "b": "/tmp/b.npz"}
        summary = run_league(paths, 8, strength_player({"a": 2, "b": 1}), seed=1, verbose=False)
        pair = summary["pairs"][0]
        self.assertEqual(pair["aWins"], 8)
        self.assertEqual(pair["aWinRateCI"], list(wilson_interval(8, 8)))

    def test_binomial_test_detects_dominant_policy(self):
        paths = {"a": "/tmp/a.npz", "b": "/tmp/b.npz"}
        summary = run_league(paths, 8, strength_player({"a": 2, "b": 1}), seed=1, verbose=False)
        pair = summary["pairs"][0]
        self.assertLess(pair["p"], 0.05)
        self.assertTrue(pair["significant"])
        self.assertEqual(pair["stronger"], "a")

    def test_small_sample_is_not_significant(self):
        paths = {"a": "/tmp/a.npz", "b": "/tmp/b.npz"}
        summary = run_league(paths, 4, strength_player({"a": 2, "b": 1}), seed=1, verbose=False)
        pair = summary["pairs"][0]
        # 4:0 → 双侧 p = 0.125，不能宣称显著（小样本不做正态近似）。
        self.assertGreater(pair["p"], 0.05)
        self.assertFalse(pair["significant"])
        self.assertIsNone(pair["stronger"])

    def test_all_draws_are_not_significant(self):
        paths = {"a": "/tmp/a.npz", "b": "/tmp/b.npz"}
        summary = run_league(paths, 4, strength_player({"a": 1, "b": 1}), seed=1, verbose=False)
        pair = summary["pairs"][0]
        self.assertEqual(pair["draws"], 4)
        self.assertEqual(pair["p"], 1.0)
        self.assertFalse(pair["significant"])
        self.assertEqual(summary["ranking"][0]["winRate"], 0.0)


class V2RejectionTest(unittest.TestCase):
    def test_rejects_v2_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy-v2.npz"
            Policy(obs_dim=OBS_DIM, n_actions=37, hidden=8, seed=0,
                   action_space="v2").save(str(path))
            with self.assertRaises(SystemExit) as caught:
                load_policies([str(path)])
            self.assertIn("v2", str(caught.exception))

    def test_accepts_v1_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.npz"
            Policy(obs_dim=OBS_DIM, n_actions=11, hidden=8, seed=0).save(str(path))
            loaded = load_policies([str(path)])
            self.assertEqual(len(loaded), 1)
            record = next(iter(loaded.values()))
            self.assertEqual(record["path"], str(path))

    def test_unique_labels_for_generic_stems(self):
        labels = unique_labels([
            "/x/branches/selfplay-final/policy.npz",
            "/x/branch-b/policy.npz",
            "/x/stage1.npz",
        ])
        self.assertEqual(labels, ["selfplay-final", "branch-b", "stage1"])
        self.assertEqual(len(set(labels)), len(labels))


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.paths = {"a": "/tmp/a.npz", "b": "/tmp/b.npz", "c": "/tmp/c.npz"}
        self.summary = run_league(self.paths, 2, strength_player({"a": 3, "b": 2, "c": 1}),
                                  seed=3, verbose=False)

    def test_markdown_has_three_sections(self):
        markdown = format_markdown(self.summary)
        for heading in ("### 排名", "### 对阵", "### 胜负矩阵"):
            self.assertIn(heading, markdown)
        self.assertIn("| A \\ B |", markdown)
        # 每对 2 局、强者全胜 → 矩阵格子里应出现 2-0-0。
        self.assertIn("2-0-0", markdown)

    def test_json_written_with_expected_name(self):
        with tempfile.TemporaryDirectory() as folder:
            target = write_summary(self.summary, folder)
            self.assertRegex(target.name, r"^\d{8}-\d{4}-league\.json$")
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "league")
            self.assertEqual(len(payload["ranking"]), 3)
            self.assertEqual(payload["episodesPerPair"], 2)
            self.assertEqual(sorted(payload["matrix"]), ["a", "b", "c"])


class SummarizeRowsTest(unittest.TestCase):
    def test_handles_empty_rows(self):
        summary = summarize_rows(["a", "b"], [])
        self.assertEqual(summary["pairs"][0]["episodes"], 0)
        self.assertEqual(summary["pairs"][0]["p"], 1.0)
        self.assertEqual(summary["ranking"][0]["winRate"], 0.0)


if __name__ == "__main__":
    unittest.main()
