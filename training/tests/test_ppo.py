"""PPO + GAE 的正确性测试（P1）。

覆盖：GAE 手算对照 / 优势零方差 / 裁剪后梯度为 0 / 价值损失方向 / 熵项 / softmax 稳定 /
旧模型 obsDim 补零加载 / REINFORCE 路径回归。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from gladiator_env.testing import FakeLink, make_observation, make_target
from gladiator_train.policy import (
    Policy,
    adopt_weights,
    check_compat,
    normalize_advantages,
    validate_init_policy,
)
from gladiator_train.trainer import Trainer, gae


class FakeEnv:
    """最小 env 替身：observation_space / action_space / step / reset。"""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ActionSpace:
        n = 3

    def __init__(self, length: int = 3, reward: float = 1.0, opponent_mode: str = "scripted"):
        self.observation_space = self._Space((4,))
        self.action_space = self._ActionSpace()
        self.length = length
        self.reward = reward
        self.opponent_mode = opponent_mode
        self.bots = [type("B", (), {"username": "Bot_A"})(),
                     type("B", (), {"username": "Bot_B"})()]
        self.raw_observations = []
        self._t = 0
        self.shaping = None

    def reset(self, **_kw):
        self._t = 0
        return np.ones(4) * 0.1, {"step": 0}

    def step(self, _action):
        self._t += 1
        done = self._t >= self.length
        return (np.ones(4) * 0.1, self.reward, done, False,
                {"winner": "Bot_A" if done else None, "health": {}})

    def last_events(self):
        return {}

    def close(self):
        return None


class GaeTest(unittest.TestCase):
    def test_gae_matches_hand_computation(self):
        """3 步、gamma=0.5、lam=0.5：A0 = δ0 + 0.25·δ1 + 0.0625·δ2。"""
        rewards = [1.0, 0.0, 2.0]
        values = [0.5, 0.0, 1.0]
        dones = [0.0, 0.0, 1.0]
        gamma, lam = 0.5, 0.5

        adv, ret = gae(rewards, values, dones, gamma=gamma, lam=lam)

        d0 = rewards[0] + gamma * values[1] - values[0]      # 1.0 + 0 - 0.5 = 0.5
        d1 = rewards[1] + gamma * values[2] - values[1]      # 0 + 0.5 - 0 = 0.5
        d2 = rewards[2] + 0.0 - values[2]                    # 2 - 1 = 1.0
        a2 = d2
        a1 = d1 + gamma * lam * a2                           # 0.5 + 0.25*1 = 0.75
        a0 = d0 + gamma * lam * a1                           # 0.5 + 0.25*0.75 = 0.6875

        np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-12)
        np.testing.assert_allclose(ret, adv + np.asarray(values), rtol=1e-12)

    def test_gae_terminates_bootstrap_at_done(self):
        """done=1 的那一步之后不再 bootstrap（末步价值不参与）。"""
        adv, _ret = gae([1.0], [5.0], [1.0], gamma=0.9, lam=0.9)
        self.assertAlmostEqual(float(adv[0]), 1.0 - 5.0, places=12)

    def test_gae_empty(self):
        adv, ret = gae([], [], [])
        self.assertEqual(adv.shape, (0,))
        self.assertEqual(ret.shape, (0,))


class AdvantageTest(unittest.TestCase):
    def test_normalize_zero_variance(self):
        out = normalize_advantages(np.asarray([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


class PpoGradientTest(unittest.TestCase):
    def _policy(self, **kw):
        return Policy(obs_dim=4, n_actions=3, hidden=6, seed=11, **kw)

    def test_clipped_sample_has_zero_policy_gradient(self):
        """ratio 冲出 1+ε：min 选裁剪支，策略梯度必须为 0（只剩余熵项，这里 β=0）。"""
        policy = self._policy()
        obs = [np.ones(4) * 0.3]
        actions = [1]
        # old_logp 远小于当前 logp → ratio 很大
        logp, _ent, _val = policy.evaluate_actions(obs, actions)
        old_logp = [float(logp[0]) - 5.0]
        adv = [1.0]  # A>0，ratio 大 → surr_unclipped 更大 → 选中 clip 支

        grads = policy.ppo_gradients(obs, actions, old_logp, adv, [0.0],
                                     clip=0.2, value_coef=0.0, entropy_coef=0.0)

        np.testing.assert_allclose(grads["W2"], 0.0, atol=1e-12)
        np.testing.assert_allclose(grads["W1"], 0.0, atol=1e-12)

    def test_inside_clip_range_policy_gradient_nonzero(self):
        policy = self._policy()
        obs = [np.ones(4) * 0.3]
        actions = [1]
        logp, _ent, _val = policy.evaluate_actions(obs, actions)

        grads = policy.ppo_gradients(obs, actions, [float(logp[0])], [1.0], [0.0],
                                     clip=0.2, value_coef=0.0, entropy_coef=0.0)

        self.assertGreater(float(np.abs(grads["W2"]).sum()), 0.0)

    def test_value_gradient_pulls_up_when_return_above_value(self):
        policy = self._policy()
        obs = [np.ones(4) * 0.3]
        actions = [0]
        value = policy.value(obs[0])
        h = policy.hidden_of(obs[0])

        grads = policy.ppo_gradients(obs, actions, [0.0], [0.0], [value + 10.0],
                                     value_coef=1.0, entropy_coef=0.0)

        # 上升方向：grads["Wv"]·h > 0 → 更新后 V 会变大
        self.assertGreater(float(grads["Wv"][0] @ h), 0.0)
        before = value
        # 只施加价值头梯度：Adam 逐坐标归一化会把 W1 的步长放得很大，掩盖 V 的线性变化。
        only_value = {k: np.zeros_like(v) for k, v in policy.params.items()}
        only_value["Wv"] = grads["Wv"]
        only_value["bv"] = grads["bv"]
        policy.apply_gradients(only_value)
        self.assertGreater(policy.value(obs[0]), before)

    def test_value_gradient_pulls_down_when_return_below_value(self):
        policy = self._policy()
        obs = [np.ones(4) * 0.3]
        grads = policy.ppo_gradients(obs, [0], [0.0], [0.0],
                                     [policy.value(obs[0]) - 10.0],
                                     value_coef=1.0, entropy_coef=0.0)
        self.assertLess(float(grads["Wv"][0] @ policy.hidden_of(obs[0])), 0.0)

    def test_entropy_term_alone_produces_gradient(self):
        policy = self._policy()
        # 注意：零观测会让 h=0，W1/W2 的梯度结构性为 0，测不出熵项。
        obs = [np.ones(4) * 0.4]
        # 优势为 0、价值目标 = 当前值 → 只有熵项贡献梯度
        value = policy.value(obs[0])
        grads = policy.ppo_gradients(obs, [0], [0.0], [0.0], [value],
                                     value_coef=0.0, entropy_coef=0.05)
        self.assertGreater(float(np.abs(grads["W2"]).sum()), 0.0)

        zero = policy.ppo_gradients(obs, [0], [0.0], [0.0], [value],
                                    value_coef=0.0, entropy_coef=0.0)
        np.testing.assert_allclose(zero["W2"], 0.0, atol=1e-12)

    def test_gradient_ascent_increases_action_probability(self):
        """回归：apply_gradients 曾把梯度取负（等价于反向学习），这里锁死方向。"""
        policy = self._policy(lr=0.05)
        x = np.ones(4) * 0.3
        before = float(policy.probs(x)[1])
        policy.update([x], [1], [1.0])
        self.assertGreater(float(policy.probs(x)[1]), before)

    def test_softmax_is_numerically_stable(self):
        policy = self._policy()
        policy.params["W2"] *= 0.0
        policy.params["W2"][0, 0] = 1e4  # 极端 logits
        p = policy.probs(np.full(4, 1e3))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)


class ReturnNormalizationTest(unittest.TestCase):
    """P2：return/优势按 running mean-std 归一化（Welford）。"""

    def test_running_stats_matches_numpy(self):
        from gladiator_train.trainer import RunningStats

        stats = RunningStats()
        values = np.array([1.0, 5.0, -3.0, 8.0, 2.0])
        stats.update(values)
        self.assertEqual(stats.count, len(values))
        self.assertAlmostEqual(stats.mean, float(values.mean()), places=9)
        self.assertAlmostEqual(stats.std, float(values.std()), places=9)

    def test_running_stats_updates_in_batches(self):
        from gladiator_train.trainer import RunningStats

        stats = RunningStats()
        stats.update([1.0, 2.0])
        stats.update([3.0, 4.0])
        allv = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats.mean, float(allv.mean()), places=9)
        self.assertAlmostEqual(stats.std, float(allv.std()), places=9)

    def test_normalize_centers(self):
        from gladiator_train.trainer import RunningStats

        stats = RunningStats()
        stats.update(np.linspace(-10.0, 10.0, 41))
        out = stats.normalize(np.array([-10.0, 0.0, 10.0]))
        self.assertAlmostEqual(float(out.mean()), 0.0, places=6)
        self.assertLess(float(out[0]), 0.0)
        self.assertGreater(float(out[-1]), 0.0)

    def test_trainer_flag_controls_usage(self):
        policy = Policy(obs_dim=4, n_actions=3, hidden=6, seed=1)
        env = FakeEnv(length=3, reward=1.0)

        trainer = Trainer(env, policy, seed=1, log=lambda _m: None, algo="ppo", epochs=1)
        trainer.train(1)
        self.assertGreater(trainer._ret_stats.count, 0, "默认应启用归一化")

        policy2 = Policy(obs_dim=4, n_actions=3, hidden=6, seed=1)
        trainer2 = Trainer(env, policy2, seed=1, log=lambda _m: None, algo="ppo", epochs=1,
                           normalize_returns=False)
        trainer2.train(1)
        self.assertEqual(trainer2._ret_stats.count, 0, "关闭后不应更新统计量")
        self.assertIn("valueLoss", trainer2.history[0])


class LegacyModelTest(unittest.TestCase):
    """旧 .npz（无价值头、obsDim=19）必须能直接接着练。"""

    def _save_legacy(self, path: str, obs_dim: int = 4, hidden: int = 5) -> None:
        rng = np.random.default_rng(0)
        np.savez(
            path,
            W1=rng.normal(size=(hidden, obs_dim)),
            b1=np.zeros(hidden),
            W2=rng.normal(size=(3, hidden)),
            b2=np.zeros(3),
            _meta=np.array('{"obs_dim": %d, "n_actions": 3, "hidden": %d, "updates": 7}'
                           % (obs_dim, hidden)),
        )

    def test_load_pads_old_obs_dim_keeping_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.npz")
            self._save_legacy(path)
            with np.load(path, allow_pickle=True) as data:
                old_w1 = np.array(data["W1"])

            loaded = Policy.load(path, obs_dim=8)

            self.assertEqual(loaded.obs_dim, 8)
            self.assertEqual(loaded.params["W1"].shape, (5, 8))
            # 旧列保持原下标（新增观测追加在末尾）
            np.testing.assert_allclose(loaded.params["W1"][:, :4], old_w1)
            np.testing.assert_allclose(loaded.params["W1"][:, 4:], 0.0)
            # 旧文件没有价值头 → 随机初始化，形状正确
            self.assertEqual(loaded.params["Wv"].shape, (1, 5))
            self.assertEqual(loaded.params["bv"].shape, (1,))
            self.assertEqual(loaded.updates, 7)

    def test_validate_accepts_smaller_obs_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.npz")
            self._save_legacy(path)
            loaded, error = validate_init_policy(path, obs_dim=8, n_actions=3, hidden=5)
            self.assertIsNone(error)
            self.assertIsNotNone(loaded)
            self.assertEqual(check_compat(loaded, obs_dim=8, n_actions=3, hidden=5), [])

    def test_adopt_weights_upgrades_into_current_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.npz")
            self._save_legacy(path)
            loaded = Policy.load(path)
        policy = Policy(obs_dim=8, n_actions=3, hidden=5, seed=3)
        adopt_weights(policy, loaded)
        self.assertEqual(policy.params["W1"].shape, (5, 8))
        np.testing.assert_allclose(policy.params["W1"][:, 4:], 0.0)
        self.assertEqual(policy.params["W2"].shape, (3, 5))

    def test_validate_still_rejects_wider_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.npz")
            self._save_legacy(path, obs_dim=9)
            loaded, error = validate_init_policy(path, obs_dim=8, n_actions=3, hidden=5)
            self.assertIsNone(loaded)
            self.assertIn("obs_dim", error or "")


class TrainerPathTest(unittest.TestCase):
    def _trainer(self, algo: str, length: int = 3):
        policy = Policy(obs_dim=4, n_actions=3, hidden=6, seed=2, lr=0.01)
        return Trainer(FakeEnv(length=length), policy, seed=1,
                       log=lambda _m: None, algo=algo, epochs=2), policy

    def test_ppo_records_metrics(self):
        trainer, policy = self._trainer("ppo")
        history = trainer.train(1)
        record = history[0]
        for key in ("policyLoss", "valueLoss", "entropy", "approxKL", "clipFraction"):
            self.assertIn(key, record)
            self.assertTrue(np.isfinite(record[key]), f"{key} 非有限值")
        self.assertEqual(record["algo"], "ppo")
        self.assertEqual(policy.updates, 2)  # epochs=2

    def test_reinforce_path_unchanged(self):
        trainer, policy = self._trainer("reinforce")
        history = trainer.train(1)
        record = history[0]
        # 旧键集不能多出新字段，否则旧报告/旧前端会受影响
        for key in ("policyLoss", "valueLoss", "approxKL", "algo"):
            self.assertNotIn(key, record)
        self.assertEqual(policy.updates, 1)

    def test_ppo_updates_value_head(self):
        trainer, policy = self._trainer("ppo")
        before = policy.params["Wv"].copy()
        trainer.train(1)
        self.assertFalse(np.allclose(before, policy.params["Wv"]))


class ReportSummaryTest(unittest.TestCase):
    """report.summary.windows（每 10 局一个窗口）。"""

    def test_sliding_windows(self):
        from gladiator_train.train import sliding_windows

        records = [{"win": i < 3, "winner": "Bot_A" if i < 3 else "Bot_B",
                    "totalReward": -float(i)} for i in range(12)]
        windows = sliding_windows(records, "Bot_A", 10)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["startEpisode"], 1)
        self.assertEqual(windows[0]["episodes"], 10)
        self.assertEqual(windows[0]["wins"], 3)
        self.assertAlmostEqual(windows[0]["winRate"], 0.3)
        self.assertEqual(windows[1]["startEpisode"], 11)
        self.assertEqual(windows[1]["episodes"], 2)
        self.assertEqual(windows[1]["wins"], 0)

    def test_empty_records(self):
        from gladiator_train.train import sliding_windows

        self.assertEqual(sliding_windows([], "Bot_A", 10), [])


class ObservationContractTest(unittest.TestCase):
    """观测维度与策略输入必须一致（Block B 的联动断言也放这里做一次总检）。"""

    def test_obs_dim_matches_policy_input(self):
        from gladiator_env.spaces import OBS_DIM, encode_observation

        raw = make_observation("Bot_A", targets=[make_target(2, "Bot_B", x=2.0, z=0.0)])
        vector = encode_observation(raw, "Bot_A")
        self.assertEqual(vector.shape, (OBS_DIM,))
        self.assertEqual(vector.shape[0],
                         Policy(obs_dim=OBS_DIM, n_actions=3, hidden=8).obs_dim)


if __name__ == "__main__":
    unittest.main()
