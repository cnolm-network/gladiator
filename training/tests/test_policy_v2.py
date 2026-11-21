"""v2（多离散头 / factorized）策略的单测。

覆盖任务书的五项：log_prob 逐头求和、mask 后概率为 0、entropy 可加性、
正 advantage 让被选动作概率上升、v1 行为不变，另加 save/load 与 v1→v2 转换。

全部离线（numpy），不连服务器。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from gladiator_env.spaces import ACTION_HEADS_V2, OBS_DIM
from gladiator_train.policy import (
    Policy,
    act_with_mask,
    adopt_weights,
    check_compat,
    validate_init_policy,
)

HEADS = list(ACTION_HEADS_V2)
SIZES = [ACTION_HEADS_V2[h] for h in HEADS]


def make_v2(seed: int = 7, hidden: int = 8, obs_dim: int = 6) -> Policy:
    return Policy(obs_dim=obs_dim, n_actions=11, hidden=hidden, seed=seed,
                  action_space="v2", head_sizes=SIZES, head_names=HEADS)


def full_mask() -> list:
    return [[True] * size for size in SIZES]


class HeadStructureTest(unittest.TestCase):
    def test_head_sizes_and_names(self) -> None:
        policy = make_v2()
        self.assertTrue(policy.is_v2)
        self.assertEqual(policy.head_names(), HEADS)
        self.assertEqual(policy.head_sizes(), SIZES)
        self.assertEqual(policy.head_sizes(), [5, 7, 5, 2, 3, 2, 2, 2, 9, 3])
        self.assertEqual(policy.n_actions, sum(SIZES))

    def test_param_names_follow_head_names(self) -> None:
        policy = make_v2()
        names = policy.param_names()
        self.assertIn("W1", names)
        self.assertIn("b1", names)
        for head, size in zip(HEADS, SIZES):
            self.assertIn(f"W2_{head}", names)
            self.assertIn(f"b2_{head}", names)
            self.assertEqual(policy.params[f"W2_{head}"].shape, (size, policy.hidden))
            self.assertEqual(policy.params[f"b2_{head}"].shape, (size,))
        self.assertIn("Wv", names)
        self.assertIn("bv", names)
        # PARAM_NAMES：类上是 v1 名字（兼容旧代码），实例上可调用
        self.assertEqual(list(Policy.PARAM_NAMES), ["W1", "b1", "W2", "b2", "Wv", "bv"])
        self.assertEqual(policy.PARAM_NAMES(), names)

    def test_logits_all_shapes(self) -> None:
        policy = make_v2()
        logits = policy.logits_all(np.zeros(policy.obs_dim))
        self.assertEqual(len(logits), len(SIZES))
        for got, size in zip(logits, SIZES):
            self.assertEqual(got.shape, (size,))

    def test_v1_guards(self) -> None:
        v1 = Policy(obs_dim=6, n_actions=4, hidden=5, seed=1)
        with self.assertRaises(ValueError):
            v1.sample(np.zeros(6), np.random.default_rng(0))
        v2 = make_v2()
        with self.assertRaises(ValueError):
            v2.act(np.zeros(6), np.random.default_rng(0))
        with self.assertRaises(ValueError):
            v2.forward(np.zeros(6))


class LogProbTest(unittest.TestCase):
    def test_log_prob_is_sum_of_head_log_softmax(self) -> None:
        policy = make_v2()
        rng = np.random.default_rng(3)
        obs = [rng.normal(size=policy.obs_dim) for _ in range(3)]
        actions = [policy.sample(x, rng) for x in obs]

        got = policy.log_prob(obs, actions)
        for index, (x, action) in enumerate(zip(obs, actions)):
            h = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
            manual = 0.0
            for head, size in zip(HEADS, SIZES):
                logits = policy.params[f"W2_{head}"] @ h + policy.params[f"b2_{head}"]
                z = logits / policy.temperature
                z = z - np.max(z)
                log_probs = z - np.log(np.sum(np.exp(z)))
                manual += float(log_probs[int(action[HEADS.index(head)])])
            self.assertAlmostEqual(float(got[index]), manual, places=10)

    def test_log_prob_rejects_wrong_action_shape(self) -> None:
        policy = make_v2()
        with self.assertRaises(ValueError):
            policy.log_prob([np.zeros(policy.obs_dim)], [np.array([1, 2])])

    def test_log_prob_rejects_out_of_range(self) -> None:
        policy = make_v2()
        action = np.zeros(len(SIZES), dtype=np.int64)
        action[0] = SIZES[0]                      # 越界（move 只有 5 个取值）
        with self.assertRaises(ValueError):
            policy.log_prob([np.zeros(policy.obs_dim)], [action])


class MaskTest(unittest.TestCase):
    def test_masked_action_probability_is_zero(self) -> None:
        policy = make_v2()
        x = np.zeros(policy.obs_dim)
        mask = full_mask()
        mask[HEADS.index("slot")] = [True] + [False] * 8      # 只允许第 0 格
        action = np.zeros(len(SIZES), dtype=np.int64)
        action[HEADS.index("slot")] = 5

        logp = float(policy.log_prob([x], [action], masks=[mask])[0])
        self.assertLess(float(np.exp(logp)), 1e-12, "被屏蔽的 slot 概率必须为 0")

        allowed = action.copy()
        allowed[HEADS.index("slot")] = 0
        logp_allowed = float(policy.log_prob([x], [allowed], masks=[mask])[0])
        self.assertGreater(logp_allowed, logp)

    def test_sample_never_picks_masked_slot(self) -> None:
        policy = make_v2()
        x = np.zeros(policy.obs_dim)
        mask = full_mask()
        mask[HEADS.index("slot")] = [True] * 3 + [False] * 6
        slot = HEADS.index("slot")
        rng = np.random.default_rng(11)
        for _ in range(200):
            action = policy.sample(x, rng, mask=mask)
            self.assertIn(int(action[slot]), (0, 1, 2))
        # 同一头在无 mask 时应当能采到 3 以外的位置（否则这条断言没有意义）
        unseen = {int(policy.sample(x, rng)[slot]) for _ in range(200)}
        self.assertTrue(unseen - {0, 1, 2})

    def test_all_false_head_falls_back(self) -> None:
        """空热键栏时 slot 头 9 个 False：退回全开而不是崩（并计数）。"""
        policy = make_v2()
        mask = full_mask()
        mask[HEADS.index("slot")] = [False] * 9
        before = policy.mask_fallbacks
        action = policy.sample(np.zeros(policy.obs_dim), np.random.default_rng(0), mask=mask)
        self.assertEqual(len(action), len(SIZES))
        self.assertEqual(policy.mask_fallbacks, before + 1)

    def test_mask_shape_validation(self) -> None:
        policy = make_v2()
        with self.assertRaises(ValueError):
            policy.sample(np.zeros(policy.obs_dim), np.random.default_rng(0),
                          mask=[[True] * 5])
        bad = full_mask()
        bad[0] = [True, False]
        with self.assertRaises(ValueError):
            policy.sample(np.zeros(policy.obs_dim), np.random.default_rng(0), mask=bad)

    def test_normalize_mask_from_env_dict(self) -> None:
        policy = make_v2()
        env_mask = {head: [True] * size for head, size in ACTION_HEADS_V2.items()}
        env_mask["attack"] = [True, False]
        rows = policy.normalize_mask(env_mask)
        self.assertEqual(len(rows), len(SIZES))
        self.assertEqual(rows[HEADS.index("attack")], [True, False])
        # 多智能体形态 {bot_0: {...}}
        wrapped = policy.normalize_mask({"bot_0": env_mask})
        self.assertEqual(wrapped[HEADS.index("attack")], [True, False])


class EntropyTest(unittest.TestCase):
    def test_entropy_is_sum_of_head_entropies(self) -> None:
        policy = make_v2()
        rng = np.random.default_rng(5)
        obs = [rng.normal(size=policy.obs_dim) for _ in range(3)]
        got = policy.entropy(obs)
        for index, x in enumerate(obs):
            manual = 0.0
            for head in HEADS:
                h = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
                logits = policy.params[f"W2_{head}"] @ h + policy.params[f"b2_{head}"]
                z = logits / policy.temperature
                z = z - np.max(z)
                p = np.exp(z) / np.sum(np.exp(z))
                manual += float(-np.sum(p * np.log(p + 1e-12)))
            self.assertAlmostEqual(float(got[index]), manual, places=10)

    def test_entropy_bounded_by_log_of_head_sum(self) -> None:
        policy = make_v2()
        x = np.zeros(policy.obs_dim)
        entropy = float(policy.entropy([x])[0])
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, float(np.sum(np.log(SIZES))) + 1e-9)

    def test_masked_entropy_not_larger(self) -> None:
        policy = make_v2()
        x = np.zeros(policy.obs_dim)
        mask = full_mask()
        mask[HEADS.index("slot")] = [True] + [False] * 8
        self.assertLessEqual(float(policy.entropy([x], masks=[mask])[0]),
                             float(policy.entropy([x])[0]) + 1e-9)


class GradientDirectionTest(unittest.TestCase):
    def _probs(self, policy: Policy, x, action) -> list:
        return [float(policy._head_probs(l)[action[i]])          # noqa: SLF001 - 断言用
                for i, l in enumerate(policy.logits_all(x))]

    def test_positive_advantage_raises_selected_head_actions(self) -> None:
        policy = make_v2(seed=11, hidden=6)
        policy.lr = 0.5
        x = np.zeros(policy.obs_dim) + 0.3
        action = policy.sample(x, np.random.default_rng(7))
        before = self._probs(policy, x, action)

        logp = policy.log_prob([x], [action])
        grads = policy.ppo_gradients([x], [action], logp, [1.0], [0.0],
                                     clip=0.2, value_coef=0.0, entropy_coef=0.0)
        policy.apply_gradients(grads)
        after = self._probs(policy, x, action)

        raised = sum(1 for b, a in zip(before, after) if a > b)
        self.assertGreaterEqual(raised, 1, "正 advantage 至少要让一个头的被选动作概率上升")
        self.assertEqual(raised, len(before), "factorized 下应当每个头都上升")

    def test_zero_advantage_only_entropy(self) -> None:
        policy = make_v2(seed=13, hidden=6)
        x = np.zeros(policy.obs_dim)
        action = policy.sample(x, np.random.default_rng(1))
        grads = policy.ppo_gradients([x], [action], policy.log_prob([x], [action]),
                                     [0.0], [0.0], clip=0.2, value_coef=0.0,
                                     entropy_coef=0.0)
        total = sum(float(np.abs(g).sum()) for g in grads.values())
        self.assertAlmostEqual(total, 0.0, places=12)

    def test_clipped_sample_has_zero_policy_gradient(self) -> None:
        policy = make_v2(seed=17, hidden=6)
        x = np.zeros(policy.obs_dim)
        action = policy.sample(x, np.random.default_rng(2))
        logp = float(policy.log_prob([x], [action])[0])
        grads = policy.ppo_gradients([x], [action], [logp - 5.0], [1.0], [0.0],
                                     clip=0.2, value_coef=0.0, entropy_coef=0.0)
        for head in HEADS:
            np.testing.assert_allclose(grads[f"W2_{head}"], 0.0, atol=1e-12)


class V1UnchangedTest(unittest.TestCase):
    def test_v1_forward_and_gradients_still_work(self) -> None:
        policy = Policy(obs_dim=4, n_actions=3, hidden=6, seed=11)
        self.assertFalse(policy.is_v2)
        self.assertEqual(policy.head_sizes(), [3])
        probs = policy.probs(np.ones(4) * 0.3)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=10)
        action = policy.act(np.ones(4) * 0.3, np.random.default_rng(0), greedy=True)
        self.assertIsInstance(action, int)
        logp, entropy, values = policy.evaluate_actions([np.ones(4) * 0.3], [action])
        self.assertEqual(logp.shape, (1,))
        self.assertTrue(np.isfinite(entropy).all())
        self.assertTrue(np.isfinite(values).all())

    def test_v1_gradient_direction(self) -> None:
        """符号回归：正 advantage 必须让被选中动作概率**上升**（旧实现是反的）。"""
        policy = Policy(obs_dim=4, n_actions=3, hidden=6, seed=11, lr=0.5)
        obs = [np.ones(4) * 0.3]
        before = float(policy.probs(obs[0])[1])
        logp, _e, _v = policy.evaluate_actions(obs, [1])
        grads = policy.ppo_gradients(obs, [1], [float(logp[0])], [1.0], [0.0],
                                     clip=0.2, value_coef=0.0, entropy_coef=0.0)
        policy.apply_gradients(grads)
        self.assertGreater(float(policy.probs(obs[0])[1]), before)


class PersistenceTest(unittest.TestCase):
    def test_roundtrip_v2(self) -> None:
        policy = make_v2(seed=21)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2.npz")
            policy.save(path)
            with np.load(path, allow_pickle=True) as data:
                meta = json.loads(str(data["_meta"]))
            self.assertEqual(meta["actionSpace"], "v2")
            self.assertEqual(meta["headSizes"], SIZES)
            self.assertEqual(meta["n_actions"], sum(SIZES))

            loaded = Policy.load(path)
            self.assertTrue(loaded.is_v2)
            self.assertEqual(loaded.head_sizes(), SIZES)
            for key in policy.param_names():
                np.testing.assert_allclose(loaded.params[key], policy.params[key], atol=1e-12)
            # 同一观测、同一动作：log_prob 一致
            x = np.zeros(policy.obs_dim)
            action = policy.sample(x, np.random.default_rng(0))
            self.assertAlmostEqual(float(loaded.log_prob([x], [action])[0]),
                                   float(policy.log_prob([x], [action])[0]), places=12)

    def test_v1_meta_has_no_action_space_key(self) -> None:
        """v1 的 _meta 保持旧形态（不写 actionSpace），老文件读新文件也不会被误判。"""
        policy = Policy(obs_dim=4, n_actions=3, hidden=5, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.npz")
            policy.save(path)
            with np.load(path, allow_pickle=True) as data:
                meta = json.loads(str(data["_meta"]))
            self.assertNotIn("actionSpace", meta)
            self.assertNotIn("headSizes", meta)
            self.assertFalse(Policy.load(path).is_v2)

    def test_missing_action_space_means_v1(self) -> None:
        policy = Policy(obs_dim=4, n_actions=3, hidden=5, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.npz")
            # 手工写一份「老格式」npz：meta 里没有 actionSpace
            payload = {k: v for k, v in policy.params.items()}
            payload["_meta"] = np.array(json.dumps({
                "obs_dim": 4, "n_actions": 3, "hidden": 5, "updates": 0}))
            np.savez(path, **payload)
            loaded = Policy.load(path)
            self.assertFalse(loaded.is_v2)
            self.assertEqual(loaded.head_sizes(), [3])


class AdoptWeightsTest(unittest.TestCase):
    def test_v1_to_v2_keeps_representation(self) -> None:
        v1 = Policy(obs_dim=6, n_actions=11, hidden=8, seed=5)
        v2 = make_v2(seed=6)
        rng_before = [v2.params[f"W2_{h}"].copy() for h in HEADS]
        adopt_weights(v2, v1)
        np.testing.assert_allclose(v2.params["W1"], v1.params["W1"], atol=1e-12)
        np.testing.assert_allclose(v2.params["b1"], v1.params["b1"], atol=1e-12)
        np.testing.assert_allclose(v2.params["Wv"], v1.params["Wv"], atol=1e-12)
        # 新头保留各自的随机初始化（没有被 v1 的 W2 污染）
        for head, before in zip(HEADS, rng_before):
            np.testing.assert_allclose(v2.params[f"W2_{head}"], before, atol=1e-12)
        self.assertEqual(float(np.abs(v2.params["W2_move"]).sum()) > 0, True)

    def test_v2_to_v1_rejected(self) -> None:
        v1 = Policy(obs_dim=6, n_actions=11, hidden=8, seed=5)
        v2 = make_v2(seed=6)
        with self.assertRaises(ValueError) as ctx:
            adopt_weights(v1, v2)
        self.assertIn("不能把 v2", str(ctx.exception))

    def test_v2_to_v2_different_heads_rejected(self) -> None:
        a = make_v2(seed=1)
        b = Policy(obs_dim=6, n_actions=11, hidden=8, seed=2, action_space="v2",
                   head_sizes=[2] * 9, head_names=[f"h{i}" for i in range(9)])
        with self.assertRaises(ValueError):
            adopt_weights(a, b)


class CompatTest(unittest.TestCase):
    def test_action_space_mismatch_reported(self) -> None:
        v2 = make_v2()
        problems = check_compat(v2, obs_dim=6, n_actions=11, hidden=8, action_space="v1")
        self.assertTrue(any("动作空间" in p for p in problems))
        self.assertTrue(any("--action-space v2" in p for p in problems))

    def test_v2_compatible_when_space_matches(self) -> None:
        v2 = make_v2()
        self.assertEqual(check_compat(v2, obs_dim=6, n_actions=11, hidden=8,
                                      action_space="v2"), [])

    def test_validate_init_policy_rejects_v2_for_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2.npz")
            make_v2().save(path)
            loaded, error = validate_init_policy(path, obs_dim=6, n_actions=11, hidden=8,
                                                 action_space="v1")
            self.assertIsNone(loaded)
            self.assertIn("动作空间", error)

    def test_validate_init_policy_accepts_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2.npz")
            make_v2().save(path)
            loaded, error = validate_init_policy(path, obs_dim=6, n_actions=11, hidden=8,
                                                 action_space="v2")
            self.assertIsNone(error)
            self.assertTrue(loaded.is_v2)


class ActWithMaskTest(unittest.TestCase):
    def test_dispatch_by_action_space(self) -> None:
        v1 = Policy(obs_dim=6, n_actions=4, hidden=5, seed=1)
        self.assertIsInstance(act_with_mask(v1, np.zeros(6), np.random.default_rng(0)), int)
        v2 = make_v2()
        action = act_with_mask(v2, np.zeros(6), np.random.default_rng(0), mask=full_mask())
        self.assertEqual(np.asarray(action).shape, (len(SIZES),))


if __name__ == "__main__":
    unittest.main()
