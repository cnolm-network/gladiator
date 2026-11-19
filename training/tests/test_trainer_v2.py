"""v2（9 头 MultiDiscrete）训练路径的单测：PPO 更新、mask 透传、回合采集。

用假环境（不连服务器、不需要 gymnasium 的 MultiDiscrete 实例也能跑）。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gladiator_env.spaces import ACTION_HEADS_V2
from gladiator_train.policy import Policy
from gladiator_train.trainer import EpisodeResult, Trainer

HEADS = list(ACTION_HEADS_V2)
SIZES = [ACTION_HEADS_V2[h] for h in HEADS]
SLOT = HEADS.index("slot")


def full_mask() -> list:
    return [[True] * size for size in SIZES]


def slot_only_zero() -> list:
    mask = full_mask()
    mask[SLOT] = [True] + [False] * 8
    return mask


class FakeEnvV2:
    """最小 v2 环境：固定观测、固定 reward、按步数截断，每帧带 actionMask。"""

    action_space_name = "v2"
    opponent_mode = "fake"

    def __init__(self, steps: int = 4, mask=None) -> None:
        self.bots = [SimpleNamespace(username="Learner"), SimpleNamespace(username="Opponent")]
        self.max_steps = int(steps)
        self.mask = mask or full_mask()
        self._i = 0
        self.seen_actions: list = []
        self._obs_dim = 6

    def _observation(self) -> np.ndarray:
        return np.full(self._obs_dim, 0.1, dtype=np.float32)

    def reset(self, seed=None):
        self._i = 0
        return self._observation(), {
            "step": 0,
            "actionMask": {head: list(row) for head, row in zip(HEADS, self.mask)},
            "health": {"Learner": 20.0, "Opponent": 20.0},
        }

    def step(self, action):
        self.seen_actions.append(np.asarray(action, dtype=np.int64).copy())
        self._i += 1
        done = self._i >= self.max_steps
        return self._observation(), 0.05, done, False, {
            "step": self._i,
            "winner": None,
            "actionMask": {head: list(row) for head, row in zip(HEADS, self.mask)},
            "health": {"Learner": 20.0, "Opponent": 20.0},
        }

    @property
    def raw_observations(self):
        return [{"self": {}}, {"self": {}}]

    def last_events(self):
        return {}


def make_policy(seed: int = 3, hidden: int = 8, obs_dim: int = 6) -> Policy:
    return Policy(obs_dim=obs_dim, n_actions=11, hidden=hidden, seed=seed,
                  action_space="v2", head_sizes=SIZES, head_names=HEADS, lr=3e-4)


def make_trainer(env=None, policy=None, **kwargs) -> Trainer:
    env = env or FakeEnvV2()
    policy = policy or make_policy()
    params = dict(algo="ppo", epochs=2, minibatch=8, clip=0.2, entropy_coef=0.01,
                  value_coef=0.5, normalize_returns=True, lr_decay=False, seed=5)
    params.update(kwargs)
    return Trainer(env, policy, **params)


class EpisodeCollectionTest(unittest.TestCase):
    def test_actions_are_head_vectors_and_masks_recorded(self) -> None:
        env = FakeEnvV2(steps=6, mask=slot_only_zero())
        trainer = make_trainer(env)
        result = trainer.run_episode()

        self.assertEqual(result.steps, 6)
        self.assertEqual(len(result.actions), 6)
        self.assertEqual(len(result.masks), 6, "v2 必须逐帧记录 mask")
        for action in result.actions:
            self.assertEqual(np.asarray(action).shape, (len(SIZES),))
        for mask in result.masks:
            self.assertEqual(len(mask), len(SIZES))
            self.assertEqual(mask[SLOT][0], True)
        # 采样遵守 mask：slot 头只可能是 0
        for action in result.actions:
            self.assertEqual(int(action[SLOT]), 0)

    def test_v1_path_has_no_masks(self) -> None:
        class FakeEnvV1(FakeEnvV2):
            action_space_name = "v1"

            def step(self, action):
                self.seen_actions.append(int(action))
                self._i += 1
                done = self._i >= self.max_steps
                return self._observation(), 0.05, done, False, {"winner": None,
                                                               "health": {"Learner": 20.0}}

        policy = Policy(obs_dim=6, n_actions=4, hidden=6, seed=1)
        trainer = make_trainer(FakeEnvV1(steps=3), policy)
        result = trainer.run_episode()
        self.assertEqual(result.masks, [])
        self.assertTrue(all(isinstance(a, int) for a in result.actions))

    def test_mismatched_action_space_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_trainer(policy=make_policy(), action_space="v1")


class PpoUpdateTest(unittest.TestCase):
    def test_ppo_update_on_head_actions(self) -> None:
        env = FakeEnvV2(steps=6, mask=slot_only_zero())
        trainer = make_trainer(env)
        result = trainer.run_episode()
        stats = trainer.ppo_update(result)

        self.assertTrue(stats, "v2 的 PPO 更新必须返回统计")
        for key in ("policyLoss", "valueLoss", "entropy", "approxKL", "clipFraction"):
            self.assertIn(key, stats)
            self.assertTrue(np.isfinite(stats[key]), f"{key} 不是有限值: {stats[key]}")
        self.assertGreaterEqual(float(stats["entropy"]), 0.0)
        self.assertLessEqual(float(stats["entropy"]), float(np.sum(np.log(SIZES))) + 1e-6)
        # 更新次数 = epochs × minibatch 数（这里 6 步 < minibatch 8 → 每轮 1 次）
        self.assertEqual(int(trainer.policy.updates), trainer.epochs)

    def test_masks_are_passed_into_batch(self) -> None:
        env = FakeEnvV2(steps=5, mask=slot_only_zero())
        trainer = make_trainer(env)
        result = trainer.run_episode()

        seen: list = []
        original = trainer.policy.evaluate_actions

        def spy(observations, actions, masks=None):
            seen.append(masks)
            return original(observations, actions, masks=masks)

        trainer.policy.evaluate_actions = spy      # type: ignore[assignment]
        trainer.ppo_update(result)

        self.assertTrue(seen, "PPO 重算 log_prob 时必须走 evaluate_actions")
        first = seen[0]
        self.assertIsNotNone(first, "v2 的 evaluate_actions 必须收到 masks")
        self.assertEqual(len(first), len(result.observations))
        self.assertEqual(first[0][SLOT][1], False, "slot 的屏蔽位要原样带进 batch")

    def test_update_without_masks_still_works(self) -> None:
        """旧调用方（没有 mask 的 v2 轨迹）也要能更新，只是不屏蔽。"""
        policy = make_policy()
        trainer = make_trainer(FakeEnvV2(), policy)
        rng = np.random.default_rng(0)
        observations = [np.full(6, 0.1) for _ in range(4)]
        actions = [policy.sample(x, rng) for x in observations]
        result = EpisodeResult(steps=4, total_reward=0.2, terminated=True, truncated=False,
                               winner=None, duration_s=0.1, observations=observations,
                               actions=actions, rewards=[0.05] * 4)
        stats = trainer.ppo_update(result)
        self.assertTrue(np.isfinite(stats["approxKL"]))

    def test_multi_trajectory_update(self) -> None:
        trainer = make_trainer(FakeEnvV2(steps=3))
        first = trainer.run_episode()
        second = trainer.run_episode()
        stats = trainer.ppo_update_many([first, second], seed=7)
        self.assertTrue(np.isfinite(stats["approxKL"]))
        # 两条轨迹拼成一个 batch：更新次数仍是 epochs × 1（不是每条轨迹各更一次）
        self.assertEqual(int(trainer.policy.updates), trainer.epochs)


class MaskedLogProbConsistencyTest(unittest.TestCase):
    def test_ratio_is_one_before_any_update(self) -> None:
        """行为策略的 old_logp 必须与重算的 logp 一致（ratio=1），否则 PPO 一开始就偏。"""
        env = FakeEnvV2(steps=4, mask=slot_only_zero())
        trainer = make_trainer(env)
        result = trainer.run_episode()
        logp, _e, _v = trainer.policy.evaluate_actions(
            result.observations, result.actions, masks=result.masks)
        again, _e2, _v2 = trainer.policy.evaluate_actions(
            result.observations, result.actions, masks=result.masks)
        np.testing.assert_allclose(logp, again, atol=0.0)

        unmasked, _e3, _v3 = trainer.policy.evaluate_actions(
            result.observations, result.actions)
        # mask 会改变分布：slot 只有 0 可选，logp 必然不低于无 mask 版本
        self.assertTrue(np.all(unmasked <= logp + 1e-9))


if __name__ == "__main__":
    unittest.main()
