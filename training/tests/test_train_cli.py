"""训练 CLI 的参数契约：回合（epochs）→ 计划（stages）→ 轮次（episodes）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_train.multi_train import build_parser as build_multi_parser  # noqa: E402
from gladiator_train.train import build_parser as build_train_parser  # noqa: E402


class TrainCliTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_train_parser().parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertIsNone(args.eval_epochs)
        self.assertEqual(args.init_model, "")

    def test_epochs_eval_and_init_model(self) -> None:
        args = build_train_parser().parse_args(
            ["--epochs", "3", "--eval-epochs", "2", "--init-model", "policy.npz"]
        )
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.eval_epochs, 2)
        self.assertEqual(args.init_model, "policy.npz")

    def test_legacy_eval_episodes_alias(self) -> None:
        args = build_train_parser().parse_args(["--eval-episodes", "4"])
        self.assertIsNone(args.eval_epochs)
        self.assertEqual(args.eval_episodes, 4)

    def test_multi_parser_has_epochs(self) -> None:
        self.assertEqual(build_multi_parser().parse_args([]).epochs, 1)
        self.assertEqual(build_multi_parser().parse_args(["--epochs", "2"]).epochs, 2)

    def test_multi_parser_registers_reward_preset(self) -> None:
        """multi_train 读取 args.reward_preset，解析器必须注册同名参数（G-620 修复）。"""
        self.assertIsNone(build_multi_parser().parse_args([]).reward_preset)
        args = build_multi_parser().parse_args(["--reward-preset", "presets/a.json"])
        self.assertEqual(args.reward_preset, "presets/a.json")

    def test_train_parser_registers_reward_preset(self) -> None:
        self.assertIsNone(build_train_parser().parse_args([]).reward_preset)
        args = build_train_parser().parse_args(["--reward-preset", "presets/b.json"])
        self.assertEqual(args.reward_preset, "presets/b.json")

    def test_multi_has_no_eval_phase(self) -> None:
        """多场地没有评估阶段：不应存在 --eval-epochs（Dashboard 侧据此拒绝 evalEpochs>0）。"""
        with self.assertRaises(SystemExit):
            build_multi_parser().parse_args(["--eval-epochs", "1"])


if __name__ == "__main__":
    unittest.main()
