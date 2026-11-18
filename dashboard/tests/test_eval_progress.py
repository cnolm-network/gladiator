"""P1.7：训练跑完后的「两遍评估」进度（采样 + 贪心）。

之前面板只跟 `episodesCompleted`，训练 20/20 后评估还要跑 40 分钟，
但 `/api/training` 里没有任何评估进度 → 用户看到「进度条满格、状态仍运行中」以为卡死。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from artifacts import ArtifactStore  # noqa: E402
from processes import ProcessManager  # noqa: E402
from tests.test_models import _cfg  # noqa: E402
from training_runner import TrainingJob, TrainingRunner  # noqa: E402

TRAIN_LINES = [
    "[train] === 回合 1/1 ===",
    "[train] --- 阶段 scripted：2 轮次 ---",
    "[train] episode 1/2 steps=200 reward=-1.5 winner=None terminated=False 20.1s",
    "[train] episode 2/2 steps=180 reward=-2.5 winner=Bot_B terminated=True 18.2s",
    "[train] 完成：2 episodes / 380 steps / mean reward -2.000 / 命中 3 空挥 1 / 用时 40.0s",
]


def eval_line(index: int, total: int, *, greedy: bool, winner: str = "None") -> str:
    tag = "[eval/greedy]" if greedy else "[eval]"
    return f"{tag} episode {index}/{total} steps=351 reward=190.952 winner={winner}"


class EvalProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cfg = _cfg(self.root, read_only=True, mc_port=25981)
        self.runner = TrainingRunner(
            cfg,
            ProcessManager(cfg.logs_dir, cfg.state_file),
            ArtifactStore(cfg.artifacts_dir),
        )
        self.job = TrainingJob(
            id="job-1", kind="train",
            params={"evalEpisodes": 20, "evalRounds": 1, "plan": [{"opponentMode": "scripted",
                                                                   "episodes": 20}]},
            total_episodes=20,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def feed(self, lines: list[str]) -> TrainingJob:
        for line in lines:
            self.runner._handle_line(self.job, line)
        return self.job

    def test_train_lines_keep_phase_train(self):
        job = self.feed(TRAIN_LINES)
        self.assertEqual(job.phase, "train")
        self.assertEqual(job.eval_completed, 0)
        self.assertEqual(job.training_completed, 2)
        self.assertEqual(job.episodes_completed, 2)

    def test_no_eval_lines_stays_train(self):
        """旧 run（没有评估阶段）phase 必须停在 train。"""
        job = self.feed(TRAIN_LINES)
        self.assertEqual(job.phase, "train")
        self.assertEqual(job.eval_total, 0)

    def test_sampled_pass_progress(self):
        self.feed(TRAIN_LINES)
        job = self.feed([eval_line(1, 20, greedy=False)])
        self.assertEqual(job.phase, "eval")
        self.assertEqual(job.eval_pass, 1)
        self.assertEqual(job.eval_completed, 1)
        self.assertEqual(job.eval_total, 20)
        # 训练进度不被评估行污染
        self.assertEqual(job.episodes_completed, 2)

        # 真实 stdout 每局一行；逐行累加才多阶段评估也不会乱（阶段内序号会从 1 重来）。
        job = self.feed([eval_line(i, 20, greedy=False) for i in range(2, 21)])
        snapshot = job.snapshot([])
        self.assertEqual(snapshot["phase"], "eval")
        self.assertEqual(snapshot["evalCompleted"], 20)
        self.assertEqual(snapshot["evalTotal"], 20)
        self.assertEqual(snapshot["evalPass"], 1)

    def test_greedy_pass_resets_counter(self):
        self.feed(TRAIN_LINES)
        self.feed([eval_line(20, 20, greedy=False)])
        job = self.feed([eval_line(1, 20, greedy=True)])
        self.assertEqual(job.phase, "eval-greedy")
        self.assertEqual(job.eval_pass, 2)
        self.assertEqual(job.eval_completed, 1, "换遍必须从 1 重新计数")
        self.assertEqual(job.eval_total, 20)

    def test_done_after_exit(self):
        self.feed(TRAIN_LINES)
        self.feed([eval_line(20, 20, greedy=True)])
        job = self.job
        # 直接走退出路径（不真的起进程）
        with job.lock:
            job.phase = "done"
        self.assertEqual(job.snapshot([])["phase"], "done")


if __name__ == "__main__":
    unittest.main()
