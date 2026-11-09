"""G-622：`--init-policy` 的参数解析、npz 维度校验与启动前置拒绝。

所有 npz 都用 ``Policy.save()`` 写到临时目录，不依赖本机 artifacts 或真实服务器；
``multi_train.main()`` 的用例只跑到「前置校验」就返回，不会创建目录/线程/环境。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM
from gladiator_train.multi_train import build_parser as build_multi_parser
from gladiator_train.multi_train import main as multi_main
from gladiator_train.policy import (
    Policy,
    adopt_weights,
    check_compat,
    validate_init_policy,
)
from gladiator_train.train import build_parser as build_train_parser
from gladiator_train.train import main as train_main

HIDDEN = 64


def save_policy(path: Path, *, obs: int = OBS_DIM, actions: int = NUM_ACTIONS,
                hidden: int = HIDDEN, updates: int = 0) -> Path:
    policy = Policy(obs_dim=obs, n_actions=actions, hidden=hidden, seed=7)
    policy.updates = updates
    policy.save(str(path), extra={"episodes": 3})
    return path


class InitPolicyParserTests(unittest.TestCase):
    def test_train_parser_defaults_and_values(self) -> None:
        args = build_train_parser().parse_args([])
        self.assertEqual(args.init_policy, "")
        self.assertEqual(args.model_id, "")
        self.assertEqual(args.model_name, "")
        args = build_train_parser().parse_args([
            "--init-policy", "models/x/policy.npz", "--model-id", "x", "--model-name", "模型 X",
        ])
        self.assertEqual(args.init_policy, "models/x/policy.npz")
        self.assertEqual(args.model_id, "x")
        self.assertEqual(args.model_name, "模型 X")

    def test_multi_parser_defaults_and_values(self) -> None:
        args = build_multi_parser().parse_args([])
        self.assertEqual(args.init_policy, "")
        self.assertEqual(args.model_id, "")
        self.assertEqual(args.model_name, "")
        args = build_multi_parser().parse_args([
            "--init-policy", "models/y/policy.npz", "--model-id", "y", "--model-name", "模型 Y",
        ])
        self.assertEqual(args.init_policy, "models/y/policy.npz")
        self.assertEqual(args.model_id, "y")
        self.assertEqual(args.model_name, "模型 Y")

    def test_legacy_init_model_still_accepted(self) -> None:
        self.assertEqual(
            build_train_parser().parse_args(["--init-model", "old.npz"]).init_model,
            "old.npz",
        )

    def test_parsers_reject_unknown_model_flags_on_multi(self) -> None:
        # 与 G-620 契约保持一致：multi 依旧没有评估阶段。
        with self.assertRaises(SystemExit):
            build_multi_parser().parse_args(["--eval-epochs", "1", "--init-policy", "a.npz"])


class ValidateInitPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def kwargs(self, **extra):
        params = {"obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS, "hidden": HIDDEN}
        params.update(extra)
        return params

    def test_empty_path_is_noop(self) -> None:
        self.assertEqual(validate_init_policy("", **self.kwargs()), (None, None))

    def test_missing_file_rejected(self) -> None:
        loaded, error = validate_init_policy(str(self.root / "nope.npz"), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("不存在", error or "")

    def test_valid_npz_accepted(self) -> None:
        path = save_policy(self.root / "policy.npz", updates=12)
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(error)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.obs_dim, OBS_DIM)
        self.assertEqual(loaded.n_actions, NUM_ACTIONS)
        self.assertEqual(loaded.hidden, HIDDEN)
        self.assertEqual(loaded.updates, 12)
        self.assertEqual(check_compat(loaded, **self.kwargs()), [])

    def test_obs_dim_mismatch(self) -> None:
        path = save_policy(self.root / "bad-obs.npz", obs=OBS_DIM + 3)
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("obs_dim", error or "")
        self.assertIn("不匹配", error or "")

    def test_num_actions_mismatch(self) -> None:
        path = save_policy(self.root / "bad-act.npz", actions=NUM_ACTIONS + 2)
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("n_actions", error or "")

    def test_hidden_mismatch(self) -> None:
        path = save_policy(self.root / "bad-hidden.npz", hidden=32)
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("hidden", error or "")

    def test_corrupt_npz_rejected(self) -> None:
        path = self.root / "junk.npz"
        path.write_bytes(b"not an npz")
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("无法加载", error or "")

    def test_npz_without_meta_rejected(self) -> None:
        """Policy.load 需要 _meta：缺它的 npz 不能当成可用模型。"""
        path = self.root / "no-meta.npz"
        np.savez(path, W1=np.zeros((HIDDEN, OBS_DIM)), b1=np.zeros(HIDDEN),
                 W2=np.zeros((NUM_ACTIONS, HIDDEN)), b2=np.zeros(NUM_ACTIONS))
        loaded, error = validate_init_policy(str(path), **self.kwargs())
        self.assertIsNone(loaded)
        self.assertIn("无法加载", error or "")


class AdoptWeightsTests(unittest.TestCase):
    def test_only_weights_are_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            loaded = Policy.load(str(save_policy(Path(folder) / "p.npz", updates=99)))
        policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=HIDDEN,
                        seed=42, lr=0.123, beta=0.456)
        policy.updates = 3
        before_lr = policy.lr
        adopt_weights(policy, loaded)
        for key in Policy.PARAM_NAMES:
            np.testing.assert_array_equal(policy.params[key], loaded.params[key])
        self.assertEqual(policy.updates, 3, "不能接管旧模型的 optimizer/updates 状态")
        self.assertEqual(policy._steps, 0)
        self.assertEqual(policy.lr, before_lr)
        self.assertEqual(policy.beta, 0.456)


class MainPreflightTests(unittest.TestCase):
    """两个 main 都在创建环境之前拒绝无效模型，且不留目录。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_train_main_missing_policy_returns_2(self) -> None:
        code = train_main([
            "--init-policy", str(self.root / "missing.npz"),
            "--run-dir", str(self.root / "train-run"),
            "--save", str(self.root / "train-run" / "policy.npz"),
            "--report", str(self.root / "train-run" / "report.json"),
            "--review-dir", str(self.root / "train-run" / "reviews"),
        ])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "train-run").exists())

    def test_train_main_mismatched_policy_returns_2(self) -> None:
        bad = save_policy(self.root / "bad.npz", obs=OBS_DIM + 3)
        code = train_main([
            "--init-policy", str(bad), "--run-dir", str(self.root / "train-run2"),
            "--save", str(self.root / "train-run2" / "policy.npz"),
            "--report", str(self.root / "train-run2" / "report.json"),
            "--review-dir", str(self.root / "train-run2" / "reviews"),
        ])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "train-run2").exists())

    def test_multi_main_missing_policy_returns_2(self) -> None:
        code = multi_main([
            "--init-policy", str(self.root / "missing.npz"),
            "--pairs", "1", "--run-dir", str(self.root / "multi-run"),
            "--save-dir", str(self.root / "multi-run"),
            "--review-dir", str(self.root / "multi-run" / "reviews"),
            "--report", str(self.root / "multi-run" / "report.json"),
        ])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "multi-run").exists())

    def test_multi_main_mismatched_policy_returns_2(self) -> None:
        bad = save_policy(self.root / "bad-multi.npz", actions=NUM_ACTIONS + 2)
        code = multi_main([
            "--init-policy", str(bad), "--pairs", "1",
            "--run-dir", str(self.root / "multi-run2"),
            "--save-dir", str(self.root / "multi-run2"),
            "--review-dir", str(self.root / "multi-run2" / "reviews"),
            "--report", str(self.root / "multi-run2" / "report.json"),
        ])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "multi-run2").exists())


if __name__ == "__main__":
    unittest.main()
