"""BC 软目标（温度 T / 标签平滑 eps）：不再把策略压死。

背景：硬标签 + 30 轮 Adam 之后策略 entropy 只有 0.01~0.03，softmax 饱和 → 后续 PPO
推不动（见 HANDOVER「P2.3 饱和治理」）。块 B 给 ``behavior_clone`` 加两个旋钮：
``temperature``（``q = softmax(log p_teacher / T)``）与 ``label_smoothing``（与均匀分布混合）。

覆盖：
- ``T=1`` 且 ``eps=0`` 与旧实现**逐位等价**（用旧算法的本地副本做对照）；
- ``T=2`` 让策略熵与老师熵都变大；
- smoothing 与均匀分布混合；
- ``train.py`` 把参数与 report 字段透传出去；
- 非法参数直接报错。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import gladiator_env  # noqa: E402
from gladiator_env.spaces import ACTION_NAMES, OBS_DIM  # noqa: E402
from gladiator_train import train as train_module  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    behavior_clone,
    heuristic_action,
    teacher_target,
)
from gladiator_train.policy import Policy  # noqa: E402
from tests.test_bc import build_fake_env, obs_vector  # noqa: E402


def _samples(count: int = 64) -> list[tuple[np.ndarray, int, float]]:
    """两类状态（射程内 → attack / 远距 → approach），带一点噪声。"""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(count):
        reach = 1.0 if rng.random() < 0.5 else 0.0
        vector = obs_vector(in_reach=reach, distance=2.0 if reach else 6.0)
        vector += rng.normal(0, 0.01, size=OBS_DIM).astype(np.float32)
        out.append((vector, heuristic_action(vector), 0.5))
    return out


def _make_policy(seed: int = 1, hidden: int = 32) -> Policy:
    return Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=hidden, seed=seed)


def legacy_behavior_clone(policy: Policy, samples, epochs: int = 5, lr: float = 5e-3,
                          batch: int = 16, seed: int = 0) -> None:
    """旧实现（硬标签 + one-hot 的交叉熵）的**逐行副本**，只用于等价性对照。

    不改动它：这是「T=1 与旧行为等价」这个断言的参照物。
    """
    data = [(np.asarray(x, dtype=np.float64).reshape(-1), int(a)) for x, a, _r in samples]
    rng = np.random.default_rng(seed)
    names = ("W1", "b1", "W2", "b2")
    m = {k: np.zeros_like(policy.params[k]) for k in names}
    v = {k: np.zeros_like(policy.params[k]) for k in names}
    step = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for _epoch in range(epochs):
        order = rng.permutation(len(data))
        for start in range(0, len(data), batch):
            chunk = [data[i] for i in order[start:start + batch]]
            grads = {k: np.zeros_like(policy.params[k]) for k in names}
            for x, action in chunk:
                h = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
                z = policy.params["W2"] @ h + policy.params["b2"]
                e = np.exp(z - np.max(z))
                p = e / np.sum(e)
                onehot = np.zeros(policy.n_actions)
                onehot[action] = 1.0
                dlogits = p - onehot
                grads["W2"] += np.outer(dlogits, h)
                grads["b2"] += dlogits
                dz = (policy.params["W2"].T @ dlogits) * (1.0 - h * h)
                grads["W1"] += np.outer(dz, x)
                grads["b1"] += dz
            n = float(len(chunk))
            step += 1
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            for key in names:
                g = grads[key] / n
                m[key] = beta1 * m[key] + (1.0 - beta1) * g
                v[key] = beta2 * v[key] + (1.0 - beta2) * (g * g)
                policy.params[key] -= lr * (m[key] / bc1) / (np.sqrt(v[key] / bc2) + eps)


class EquivalenceTest(unittest.TestCase):
    def test_temperature_one_matches_legacy(self):
        samples = _samples()
        new_policy, old_policy = _make_policy(seed=1), _make_policy(seed=1)
        behavior_clone(new_policy, samples, epochs=5, lr=5e-3, batch=16,
                       temperature=1.0, label_smoothing=0.0)
        legacy_behavior_clone(old_policy, samples, epochs=5, lr=5e-3, batch=16)
        for key in ("W1", "b1", "W2", "b2"):
            np.testing.assert_allclose(new_policy.params[key], old_policy.params[key],
                                       rtol=0, atol=1e-12, err_msg=f"{key} 与旧实现不一致")


class TemperatureTest(unittest.TestCase):
    def test_higher_temperature_keeps_policy_entropy(self):
        samples = _samples()
        probes = [np.asarray(x, dtype=np.float64) for x, _a, _r in samples]
        sharp = _make_policy(seed=3)
        flat = _make_policy(seed=3)
        behavior_clone(sharp, samples, epochs=20, lr=5e-3, batch=16, temperature=1.0)
        behavior_clone(flat, samples, epochs=20, lr=5e-3, batch=16, temperature=2.0)
        sharp_entropy = sharp.mean_prob_entropy(probes)
        flat_entropy = flat.mean_prob_entropy(probes)
        self.assertGreater(flat_entropy, sharp_entropy * 1.5,
                           f"T=2 的熵 {flat_entropy:.4f} 没有明显高于 T=1 的 {sharp_entropy:.4f}")

    def test_teacher_entropy_reported(self):
        samples = _samples(32)
        sharp = _make_policy(seed=4)
        flat = _make_policy(seed=4)
        sharp_result = behavior_clone(sharp, samples, epochs=1, temperature=1.0)
        flat_result = behavior_clone(flat, samples, epochs=1, temperature=2.0)
        self.assertEqual(sharp_result["teacherEntropy"], 0.0)   # one-hot
        self.assertGreater(flat_result["teacherEntropy"], 0.5)  # 摊平后熵明显 > 0
        self.assertEqual(flat_result["temperature"], 2.0)

    def test_lower_temperature_sharpens(self):
        q_sharp = teacher_target(3, 11, temperature=0.5)
        q_flat = teacher_target(3, 11, temperature=2.0)
        self.assertGreater(float(q_sharp[3]), float(q_flat[3]))


class LabelSmoothingTest(unittest.TestCase):
    def test_mixes_uniform_distribution(self):
        q = teacher_target(2, 11, temperature=1.0, label_smoothing=0.2)
        self.assertAlmostEqual(float(q[2]), 0.8 + 0.2 / 11, places=9)
        self.assertAlmostEqual(float(q[0]), 0.2 / 11, places=9)
        self.assertAlmostEqual(float(np.sum(q)), 1.0, places=12)

    def test_smoothing_keeps_entropy_even_at_temperature_one(self):
        samples = _samples(32)
        result = behavior_clone(_make_policy(seed=5), samples, epochs=2,
                                label_smoothing=0.1)
        self.assertGreater(result["teacherEntropy"], 0.0)
        self.assertEqual(result["labelSmoothing"], 0.1)

    def test_invalid_parameters_rejected(self):
        policy = _make_policy(seed=6)
        with self.assertRaises(ValueError):
            behavior_clone(policy, _samples(4), epochs=1, temperature=0.0)
        with self.assertRaises(ValueError):
            behavior_clone(policy, _samples(4), epochs=1, label_smoothing=1.0)


class ReportSurfaceTest(unittest.TestCase):
    """``train.py --bc-temperature/--bc-label-smoothing`` 透传到 report。"""

    def _run(self, tmp: Path, *extra: str) -> Path:
        run_dir = tmp / "run"
        argv = [
            "--users", "Bot_T1", "Bot_T2", "--ports", "8766", "8767",
            "--run-dir", str(run_dir),
            "--save", str(run_dir / "policy.npz"),
            "--report", str(run_dir / "report.json"),
            "--review-dir", str(run_dir / "reviews"),
            "--episodes", "1", "--max-steps", "4", "--eval-epochs", "0",
            "--bc-episodes", "1", "--bc-epochs", "2", "--bc-max-steps", "4",
            *extra,
        ]
        with mock.patch.object(gladiator_env, "GladiatorEnv", build_fake_env), \
                mock.patch.object(train_module, "Trainer") as trainer_cls:
            trainer = trainer_cls.return_value
            trainer.train.return_value = []
            trainer.evaluate.return_value = []
            trainer.history = []
            self.assertEqual(train_module.main(argv), 0)
        return run_dir

    def test_report_contains_temperature_and_smoothing(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = self._run(Path(folder), "--bc-temperature", "2.0",
                                "--bc-label-smoothing", "0.05",
                                "--bc-teacher", "v2")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            for block in (report["config"]["bc"], report["bc"]):
                self.assertEqual(block["temperature"], 2.0)
                self.assertEqual(block["labelSmoothing"], 0.05)
                self.assertEqual(block["teacher"], "v2")
                self.assertIsNotNone(block["teacherEntropy"])
                self.assertGreater(block["teacherEntropy"], 0.0)

    def test_defaults_are_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = self._run(Path(folder))
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["bc"]["temperature"], 1.0)
            self.assertEqual(report["config"]["bc"]["labelSmoothing"], 0.0)
            self.assertEqual(report["config"]["bc"]["teacher"], "v1")
            self.assertEqual(report["bc"]["teacherEntropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
