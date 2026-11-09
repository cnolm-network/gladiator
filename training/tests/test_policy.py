from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from gladiator_train.policy import (
    Policy,
    discounted_returns,
    normalize_advantages,
)


def _objective(policy: Policy, obs, actions, advantages) -> float:
    """J = Σ A·logπ(a|x) + β·H(π)。"""
    total = 0.0
    for x, a, adv in zip(obs, actions, advantages):
        p = policy.probs(x)
        total += float(adv) * float(np.log(p[a] + 1e-12))
        total += policy.beta * float(-np.sum(p * np.log(p + 1e-12)))
    return total


class PolicyTest(unittest.TestCase):
    def test_probs_is_distribution(self):
        policy = Policy(obs_dim=16, n_actions=7, seed=1)
        p = policy.probs(np.zeros(16))
        self.assertEqual(p.shape, (7,))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertTrue(np.all(p > 0.0))

    def test_gradient_matches_finite_difference(self):
        policy = Policy(obs_dim=4, n_actions=3, hidden=5, seed=2, beta=0.05)
        rng = np.random.default_rng(3)
        obs = [rng.normal(size=4) for _ in range(6)]
        actions = [0, 1, 2, 1, 0, 2]
        advantages = [1.0, -0.5, 0.25, 2.0, -1.0, 0.75]

        grads = policy.gradients(obs, actions, advantages)

        eps = 1e-6
        n = float(len(obs))
        for name in ("W2", "b2", "W1", "b1"):
            numeric = np.zeros_like(policy.params[name])
            flat = policy.params[name]
            it = np.nditer(flat, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                original = flat[idx]
                flat[idx] = original + eps
                plus = _objective(policy, obs, actions, advantages)
                flat[idx] = original - eps
                minus = _objective(policy, obs, actions, advantages)
                flat[idx] = original
                numeric[idx] = (plus - minus) / (2 * eps) / n
                it.iternext()
            np.testing.assert_allclose(
                grads[name], numeric, rtol=1e-4, atol=1e-6,
                err_msg=f"{name} 的解析梯度与数值梯度不一致",
            )

    def test_update_changes_params_and_stays_finite(self):
        policy = Policy(obs_dim=16, n_actions=7, seed=4, lr=0.05)
        before = policy.params["W1"].copy()
        rng = np.random.default_rng(5)
        obs = [rng.normal(size=16) for _ in range(8)]
        actions = [int(i % 7) for i in range(8)]
        advantages = list(np.linspace(-1.0, 1.0, 8))
        policy.update(obs, actions, advantages)
        self.assertFalse(np.allclose(before, policy.params["W1"]))
        for name, param in policy.params.items():
            self.assertTrue(np.all(np.isfinite(param)), f"{name} 出现非有限值")
        self.assertEqual(policy.updates, 1)

    def test_save_and_load_roundtrip(self):
        policy = Policy(obs_dim=16, n_actions=7, seed=6, hidden=8)
        obs = np.linspace(-1.0, 1.0, 16)
        expected = policy.probs(obs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.npz")
            policy.save(path, extra={"episodes": 3})
            loaded = Policy.load(path)
            np.testing.assert_allclose(loaded.probs(obs), expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(loaded.obs_dim, 16)
            self.assertEqual(loaded.n_actions, 7)
            self.assertEqual(loaded.hidden, 8)
            with np.load(path, allow_pickle=True) as data:
                meta = json.loads(str(data["_meta"]))
            self.assertEqual(meta["episodes"], 3)

    def test_greedy_action_is_argmax(self):
        policy = Policy(obs_dim=16, n_actions=7, seed=7)
        obs = np.ones(16)
        self.assertEqual(policy.act(obs, np.random.default_rng(0), greedy=True), int(np.argmax(policy.probs(obs))))

    def test_discounted_returns(self):
        returns = discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0])

    def test_normalize_advantages_zero_std(self):
        out = normalize_advantages(np.asarray([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
