"""P2.3：策略饱和治理 + 熵控制加强 + DAgger。

背景（run 20261118-0502，BC 关闭、纯 PPO、12 局）：
- 训练前 1/12 胜 → 训练后 1/12 胜，PPO 12 局没有任何贡献；
- ``entropy`` 从第 1 局就是 0.008–0.023，``entropyCoef`` 顶在 0.5 也拉不动，
  ``approxKL≈1e-4``、``clipFraction=0``；
- 权重不大（|W1|=14、|W2|=6）但 ``maxProb≈0.97``——所以饱和来自「观测一致 + tanh 饱和 +
  logits 被拉得很开」，softmax 梯度接近 0，靠熵系数推不动。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import ACTION_NAMES, NUM_ACTIONS, OBS_DIM  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train.trainer import Trainer  # noqa: E402


class FakeEnv:
    """最小 env：记录 reset 次数，用于校验评估被调用了多少次。"""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ActionSpace:
        n = NUM_ACTIONS

    resets = 0

    def __init__(self, length: int = 6, reward: float = 1.0):
        self.observation_space = self._Space((OBS_DIM,))
        self.action_space = self._ActionSpace()
        self.length = length
        self.reward = reward
        self.bots = [type("B", (), {"username": "Bot_A"})(),
                     type("B", (), {"username": "Bot_B"})()]
        self.raw_observations = []
        self.opponent_mode = "scripted"
        self.shaping = None
        self._t = 0

    def reset(self, **_kw):
        type(self).resets += 1
        self._t = 0
        return np.linspace(-1.0, 1.0, OBS_DIM).astype(np.float32), {"step": 0}

    def step(self, _action):
        self._t += 1
        done = self._t >= self.length
        obs = np.linspace(-1.0, 1.0, OBS_DIM).astype(np.float32)
        obs[0] = self._t * 0.1 - 0.5
        return obs, self.reward, done, False, {"winner": None, "health": {}}

    def last_events(self):
        return {}

    def last_dispatch(self, _username):
        return []

    def close(self):
        return None


def make_policy(seed: int = 1, **kwargs) -> Policy:
    return Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=16, seed=seed, **kwargs)


def make_trainer(policy: Policy, env: FakeEnv | None = None, **kwargs) -> Trainer:
    kwargs.setdefault("log", lambda _m: None)
    kwargs.setdefault("algo", "ppo")
    kwargs.setdefault("epochs", 1)
    kwargs.setdefault("early_stop_kl", 0.0)
    return Trainer(env or FakeEnv(), policy, seed=1, **kwargs)


def random_obs(n: int = 24, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, OBS_DIM) for _ in range(n)]


def saturate(policy: Policy, scale: float = 25.0) -> Policy:
    """把策略头放大，制造 maxProb≈1 / logitStd 很大的「饱和策略」。"""
    policy.params["W2"] = policy.params["W2"] * scale
    return policy


# ------------------------------------------------------------------ 诊断量
class SaturationDiagnosticsTest(unittest.TestCase):
    """块 A1：logit_std / mean_prob_entropy，并在每次 update 记录 logitStd。"""

    def test_logit_std_grows_when_saturated(self):
        policy = make_policy()
        obs = random_obs()
        flat = policy.logit_std(obs)
        saturate(policy)
        self.assertGreater(policy.logit_std(obs), flat * 5)

    def test_mean_prob_entropy_matches_entropy_of_distribution(self):
        policy = make_policy()
        obs = random_obs()
        entropy = policy.mean_prob_entropy(obs)
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, np.log(NUM_ACTIONS) + 1e-9)
        saturate(policy)
        self.assertLess(policy.mean_prob_entropy(obs), entropy)

    def test_empty_batches_are_safe(self):
        policy = make_policy()
        self.assertEqual(policy.logit_std([]), 0.0)
        self.assertEqual(policy.mean_prob_entropy([]), 0.0)
        self.assertEqual(policy.mean_max_prob([]), 0.0)

    def test_logit_std_recorded_in_history(self):
        trainer = make_trainer(make_policy(), minibatch=0)
        record = trainer.train(1)[0]
        self.assertIn("logitStd", record)
        self.assertGreaterEqual(record["logitStd"], 0.0)
        self.assertIn("maxProb", record)


# ------------------------------------------------------------------- 温度
class TemperatureTest(unittest.TestCase):
    """块 A2：``p = softmax(logits / T)``，T 写进 _meta。"""

    def test_temperature_flattens_distribution(self):
        policy = make_policy()
        obs = random_obs()
        hot = policy.logit_std(obs)
        policy.temperature = 2.0
        self.assertAlmostEqual(policy.logit_std(obs), hot / 2.0, places=9)
        self.assertLess(policy.mean_max_prob(obs), Policy(
            obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=16, seed=1).mean_max_prob(obs))

    def test_temperature_one_is_legacy_behavior(self):
        a = make_policy()
        b = make_policy(temperature=1.0)
        obs = random_obs()
        np.testing.assert_allclose(a.probs(obs[0]), b.probs(obs[0]))
        self.assertEqual(a.temperature, 1.0)

    def test_invalid_temperature_rejected(self):
        with self.assertRaises(ValueError):
            Policy(obs_dim=4, n_actions=3, temperature=0.0)

    def test_temperature_round_trips_through_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "p.npz")
            policy = make_policy(temperature=1.75)
            policy.save(path)
            self.assertAlmostEqual(Policy.load(path).temperature, 1.75)
            # 续训时 adopt_weights 也要带上温度
            from gladiator_train.policy import adopt_weights

            target = make_policy(temperature=1.0)
            adopt_weights(target, Policy.load(path))
            self.assertAlmostEqual(target.temperature, 1.75)

    def test_old_npz_without_temperature_loads_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.npz")
            policy = make_policy(temperature=2.0)
            payload = {k: v for k, v in policy.params.items()}
            payload["_meta"] = np.array(json.dumps({
                "obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS, "hidden": 16, "updates": 3,
            }))
            np.savez(path, **payload)
            loaded = Policy.load(path)
            self.assertEqual(loaded.temperature, 1.0, "旧文件缺 temperature → 1.0（旧行为）")


# ------------------------------------------------------------- 重设策略头
class ResetPolicyHeadTest(unittest.TestCase):
    """块 A3：Xavier 重设 W2/b2，保留 W1/b1 与价值头。"""

    def test_head_is_reinitialised_and_rest_kept(self):
        policy = make_policy()
        saturate(policy)
        obs = random_obs()
        before_max_prob = policy.mean_max_prob(obs)
        w1, b1 = policy.params["W1"].copy(), policy.params["b1"].copy()
        wv, bv = policy.params["Wv"].copy(), policy.params["bv"].copy()
        old_w2 = policy.params["W2"].copy()

        policy.reset_policy_head(seed=7)

        self.assertLess(policy.mean_max_prob(obs), before_max_prob * 0.6,
                        "重设后分布应明显摊平")
        np.testing.assert_array_equal(policy.params["W1"], w1, "W1 必须逐位相等")
        np.testing.assert_array_equal(policy.params["b1"], b1)
        np.testing.assert_array_equal(policy.params["Wv"], wv)
        np.testing.assert_array_equal(policy.params["bv"], bv)
        self.assertEqual(policy.params["W2"].shape, old_w2.shape)
        self.assertEqual(policy.params["b2"].shape, (NUM_ACTIONS,))
        self.assertFalse(np.allclose(policy.params["W2"], old_w2), "W2 应被换掉")

    def test_optimizer_state_reset_for_head(self):
        policy = make_policy()
        obs = random_obs()
        actions = [0] * len(obs)
        policy.update(obs, actions, [1.0] * len(obs))  # 让 Adam 有一二阶矩
        self.assertGreater(float(np.abs(policy._m["W2"]).sum()), 0.0)
        policy.reset_policy_head(seed=7)
        self.assertEqual(float(np.abs(policy._m["W2"]).sum()), 0.0)
        self.assertEqual(float(np.abs(policy._v["b2"]).sum()), 0.0)

    def test_bias_starts_at_zero(self):
        policy = make_policy()
        policy.reset_policy_head(seed=3)
        np.testing.assert_array_equal(policy.params["b2"], np.zeros(NUM_ACTIONS))


# --------------------------------------------------------------- 权重衰减
class WeightDecayTest(unittest.TestCase):
    """块 A4：只衰减 W1/W2，bias 不动；0 = 关闭。"""

    def _norm(self, policy, key):
        return float(np.linalg.norm(policy.params[key]))

    def test_norms_shrink_with_decay(self):
        base = make_policy(lr=1e-2)
        decay = make_policy(lr=1e-2, weight_decay=1e-3)
        zeros = {k: np.zeros_like(v) for k, v in base.params.items()}
        base.apply_gradients({k: v.copy() for k, v in zeros.items()})
        decay.apply_gradients({k: v.copy() for k, v in zeros.items()})
        self.assertLess(self._norm(decay, "W1"), self._norm(base, "W1"))
        self.assertLess(self._norm(decay, "W2"), self._norm(base, "W2"))

    def test_bias_not_decayed(self):
        policy = make_policy(lr=1e-2, weight_decay=1e-2)
        before = policy.params["b2"].copy()
        zeros = {k: np.zeros_like(v) for k, v in policy.params.items()}
        policy.apply_gradients(zeros)
        np.testing.assert_array_equal(policy.params["b2"], before)

    def test_zero_means_no_decay(self):
        policy = make_policy(lr=1e-2, weight_decay=0.0)
        before = self._norm(policy, "W1")
        zeros = {k: np.zeros_like(v) for k, v in policy.params.items()}
        policy.apply_gradients(zeros)
        self.assertAlmostEqual(self._norm(policy, "W1"), before, places=12)


# ------------------------------------------------------------ 熵控制加强
class EntropyBoostTest(unittest.TestCase):
    """块 B：默认上限 2.0 + 饱和时额外 ×1.5。"""

    def test_default_max_is_two(self):
        from gladiator_train.train import build_parser

        self.assertAlmostEqual(build_parser().parse_args([]).entropy_coef_max, 2.0)
        self.assertAlmostEqual(make_trainer(make_policy()).entropy_coef_max, 2.0)

    def test_saturation_boost_applied_and_recorded(self):
        policy = saturate(make_policy(), scale=40.0)
        trainer = make_trainer(policy, minibatch=0, entropy_coef=0.02,
                               entropy_floor=0.0, target_entropy=0.0)
        record = trainer.train(1)[0]
        self.assertTrue(record["saturationBoost"], "饱和策略应触发额外加码")
        self.assertAlmostEqual(trainer._entropy_coef, 0.02 * 1.5, places=6)
        self.assertGreater(record["logitStd"], 4.0)

    def test_no_boost_when_not_saturated(self):
        trainer = make_trainer(make_policy(), minibatch=0, entropy_coef=0.02,
                               entropy_floor=0.0, target_entropy=0.0)
        record = trainer.train(1)[0]
        self.assertFalse(record["saturationBoost"])
        self.assertAlmostEqual(trainer._entropy_coef, 0.02, places=9)

    def test_boost_respects_max(self):
        policy = saturate(make_policy(), scale=40.0)
        trainer = make_trainer(policy, minibatch=0, entropy_coef=1.5,
                               entropy_coef_max=2.0, entropy_floor=0.0, target_entropy=0.0)
        trainer.train(1)
        self.assertLessEqual(trainer._entropy_coef, 2.0 + 1e-9)


# ------------------------------------------------------------------ DAgger
class DaggerCollectTest(unittest.TestCase):
    """块 C：采集器用**当前策略**走、老师打标签。"""

    def test_labels_come_from_teacher_actions_from_policy(self):
        from gladiator_train.demos import collect_dagger_samples

        policy = make_policy()
        env = FakeEnv(length=4)
        seen = []

        def teacher(obs):
            seen.append(obs[0])
            return 3

        samples = collect_dagger_samples(env, policy, episodes=2, max_steps=4,
                                         teacher=teacher, rng=np.random.default_rng(0))
        self.assertEqual(len(samples), 8)
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(action == 3 for _obs, action, _ret in samples))
        self.assertTrue(all(ret == 0.0 for _obs, _action, ret in samples))

    def test_rejects_dict_observations(self):
        from gladiator_train.demos import collect_dagger_samples

        class DictEnv(FakeEnv):
            def reset(self, **_kw):
                return {"Bot_A": np.zeros(OBS_DIM)}, {}

        with self.assertRaises(ValueError):
            collect_dagger_samples(DictEnv(), make_policy(), episodes=1, max_steps=2)


class DaggerMainTest(unittest.TestCase):
    """块 C：整跑 main()，检查报告结构 / agreement 单调 / 评估调用次数。"""

    def _run(self, extra: list[str]) -> tuple[dict, int]:
        import gladiator_env
        from gladiator_train import train as train_module

        FakeEnv.resets = 0
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = str(Path(tmp) / "run")
            argv = [
                "--users", "Bot_A", "Bot_B", "--ports", "8770", "8771",
                "--run-dir", run_dir,
                "--episodes", "1", "--eval-epochs", "1",
                "--max-steps", "4", "--ppo-epochs", "1",
                "--bc-episodes", "0",
                *extra,
            ]
            # 老师在这里恒返回 idle（观测 has_target=0），所以 BC 很快就能 100% 一致。
            with unittest.mock.patch.object(gladiator_env, "GladiatorEnv",
                                            lambda *a, **kw: FakeEnv(length=4)):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = train_module.main(argv)
            self.assertEqual(code, 0)
            with open(Path(run_dir) / "report.json", encoding="utf-8") as fh:
                return json.load(fh), FakeEnv.resets

    def test_two_rounds_report_and_monotone_agreement(self):
        report, _resets = self._run([
            "--dagger-rounds", "2", "--dagger-episodes", "1",
            "--dagger-epochs", "40", "--dagger-target-agreement", "1.1",
        ])
        dagger = report["dagger"]
        self.assertEqual(len(dagger), 3, "round 0（BC 基线）+ 2 轮")
        self.assertEqual([item["round"] for item in dagger], [0, 1, 2])
        for item in dagger:
            for key in ("round", "samples", "agreement", "loss", "evaluation"):
                self.assertIn(key, item)
            for key in ("sampling", "greedy"):
                self.assertIn(key, item["evaluation"])
        agreements = [item["agreement"] for item in dagger if item["agreement"] is not None]
        self.assertTrue(agreements, "至少有一轮能量到一致率")
        for earlier, later in zip(agreements, agreements[1:]):
            self.assertGreaterEqual(later, earlier, f"agreement 不应下降：{agreements}")
        self.assertAlmostEqual(agreements[-1], 1.0, places=6)
        self.assertEqual(report["config"]["daggerRounds"], 2)
        self.assertEqual(report["config"]["daggerEpisodes"], 1)

    def test_evaluation_call_count(self):
        """评估次数 = 每轮 2 组（采样+贪心）+ 收尾 2 组；DAgger 用局各 1 次。"""
        _report, resets = self._run([
            "--dagger-rounds", "2", "--dagger-episodes", "1", "--dagger-epochs", "1",
            "--dagger-target-agreement", "1.1",
        ])
        # eval_epochs=1 × 计划 1 局 → 每组评估 1 局
        # round0: 2 组；round1: 2 组(评估) + 1 局(DAgger 采集)；round2 同上；收尾评估 2 组
        # 训练 1 局 → 合计 2 + 3 + 3 + 2 + 1 = 11
        self.assertEqual(resets, 11)

    def test_dagger_disabled_by_default(self):
        report, _resets = self._run([])
        self.assertEqual(report["dagger"], [])
        self.assertEqual(report["config"]["daggerRounds"], 0)

    def test_early_stop_on_target_agreement(self):
        report, _resets = self._run([
            "--dagger-rounds", "5", "--dagger-episodes", "1",
            "--dagger-epochs", "40", "--dagger-target-agreement", "0.5",
        ])
        self.assertLess(len(report["dagger"]), 6, "达到目标一致率后应提前结束")
        self.assertGreaterEqual(report["dagger"][-1]["agreement"], 0.5)


class ResetHeadCliTest(unittest.TestCase):
    def test_flags_default_off(self):
        from gladiator_train.train import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(args.reset_policy_head)
        self.assertIsNone(args.logit_temp)
        self.assertEqual(args.weight_decay, 0.0)

    def test_flags_recorded_in_config(self):
        import gladiator_env
        from gladiator_train import train as train_module

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = str(Path(tmp) / "run")
            argv = [
                "--users", "Bot_A", "Bot_B", "--ports", "8770", "8771",
                "--run-dir", run_dir, "--episodes", "1", "--eval-epochs", "1",
                "--max-steps", "4", "--ppo-epochs", "1", "--bc-episodes", "4",
                "--bc-max-steps", "3", "--bc-epochs", "1",
                "--reset-policy-head", "--logit-temp", "1.5", "--weight-decay", "1e-4",
            ]
            with unittest.mock.patch.object(gladiator_env, "GladiatorEnv",
                                            lambda *a, **kw: FakeEnv(length=4)):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = train_module.main(argv)
            self.assertEqual(code, 0)
            with open(Path(run_dir) / "report.json", encoding="utf-8") as fh:
                report = json.load(fh)
        config = report["config"]
        self.assertTrue(config["resetPolicyHead"])
        self.assertAlmostEqual(config["logitTemp"], 1.5)
        self.assertAlmostEqual(config["weightDecay"], 1e-4)
        self.assertIn("logitStd", report["history"][0])


if __name__ == "__main__":
    unittest.main()
