"""自博弈相关单测：策略驱动的对手（``opponent_policy``）+ 自博弈轮次循环。

全部离线：``FakeLink`` 假通道 B + 假 trainer/假 suite，不连服务器、不需要 RCON。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM
from gladiator_env.testing import FakeLink, make_observation, make_target
from gladiator_train.policy import Policy
from scripts.selfplay_train import run_selfplay, _train_win_rate


class CountingPolicy:
    """假策略：记录每次调用，返回固定动作。"""

    def __init__(self, action: int = 2) -> None:
        self.action = int(action)
        self.calls: list[tuple] = []

    def act(self, obs, rng=None, greedy: bool = False) -> int:
        self.calls.append((np.array(obs, dtype=np.float64, copy=True), bool(greedy)))
        return self.action


class FailingLink(FakeLink):
    """对手运行时回一个失败码（模拟 ITEM_MISSING / OUT_OF_REACH）。"""

    def __init__(self, username: str, code: str = "OUT_OF_REACH") -> None:
        super().__init__(username)
        self.code = code
        self.failures = 0

    def send_actions(self, actions: list[dict]) -> list[dict]:
        self.actions.extend(actions)
        # 只让「战斗动作」失败，避免把补装/切剑也打坏。
        if any(action.get("type") == "attack" for action in actions):
            self.failures += 1
            return [{"ok": False, "code": self.code, "reason": "测试注入"}]
        return [{"ok": True} for _ in actions]


def build_env(*, opponent_policy=None, opponent_epsilon: float = 0.0,
              opponent_mode: str = "scripted", health_bot2: float = 20.0,
              links=None, max_steps: int = 20) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = links or [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
    links[0].set_observation(make_observation(
        "Glad_Bot_1", x=-1.5, z=0.0,
        targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=2.5)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, z=0.0, health=health_bot2,
        targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=2.5)]))
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        single_agent=True,
        max_steps=max_steps,
        opponent_mode=opponent_mode,
        opponent_policy=opponent_policy,
        opponent_epsilon=opponent_epsilon,
    )
    return env, links


class OpponentPolicyTest(unittest.TestCase):
    def test_policy_called_every_step_and_action_sent(self) -> None:
        policy = CountingPolicy(action=2)          # 2 = attack
        env, links = build_env(opponent_policy=policy)
        try:
            env.reset()
            for _ in range(3):
                env.step(0)
            # 每一步都问过对手策略，且拿到的是 23 维观测。
            self.assertEqual(len(policy.calls), 3)
            self.assertEqual(policy.calls[0][0].shape, (OBS_DIM,))
            self.assertTrue(policy.calls[0][1], "默认 greedy=True")
            self.assertEqual(env.opponent_policy_stats["decisions"], 3)
            # 动作真的发到了对手那条通道（bot_1），且是 attack。
            attacks = [a for a in links[1].actions if a.get("type") == "attack"]
            self.assertEqual(len(attacks), 3)
            self.assertEqual(env.opponent_policy_stats["actionFailures"], {})
        finally:
            env.close()

    def test_policy_sees_opponent_own_observation(self) -> None:
        """编码用的是**对手自己**的观测（血量 12 → 第 0 维 0.6），不是学习方的满血。"""
        policy = CountingPolicy(action=0)
        env, _ = build_env(opponent_policy=policy, health_bot2=12.0)
        try:
            env.reset()
            env.step(0)
            vector = policy.calls[0][0]
            self.assertAlmostEqual(float(vector[0]), 12.0 / 20.0, places=6)
        finally:
            env.close()

    def test_opponent_mode_still_works_without_policy(self) -> None:
        env, links = build_env(opponent_mode="idle")
        try:
            env.reset()
            env.step(0)
            self.assertIsNone(env.opponent_policy)
            self.assertEqual(env.opponent_policy_source, "hand-written")
            self.assertEqual(env.opponent_mode_label, "idle")
            # idle 模式：对手只发 idle，不会攻击。
            self.assertFalse(any(a.get("type") == "attack" for a in links[1].actions))
            self.assertTrue(any(a.get("type") == "idle" for a in links[1].actions))
        finally:
            env.close()

    def test_opponent_label_marks_policy_opponent(self) -> None:
        """策略驱动的对手不能被报告误记成手写模式（否则分对手统计全挤在 "scripted"）。"""
        env, _ = build_env(opponent_policy=CountingPolicy(), opponent_mode="scripted")
        try:
            self.assertEqual(env.opponent_mode, "scripted")   # 原字段不动
            self.assertEqual(env.opponent_mode_label, "policy:memory")
        finally:
            env.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "round2.npz")
            Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=9).save(path)
            env_path, _ = build_env(opponent_policy=path)
            try:
                self.assertEqual(env_path.opponent_mode_label, "policy:round2.npz")
            finally:
                env_path.close()

    def test_epsilon_makes_actions_random(self) -> None:
        policy = CountingPolicy(action=0)          # 策略永远想 idle
        env, _ = build_env(opponent_policy=policy, opponent_epsilon=1.0)
        try:
            env.reset()
            seen = set()
            for _ in range(40):
                env.step(0)
                seen.add(env.opponent_policy_stats["lastAction"])
            self.assertEqual(env.opponent_policy_stats["epsilonPicks"], 40)
            self.assertGreaterEqual(len(seen), 2, f"epsilon=1 时动作应随机，实际只出现 {seen}")
            self.assertTrue(all(not greedy for _obs, greedy in policy.calls))
        finally:
            env.close()

    def test_policy_loading_from_path_matches_object(self) -> None:
        source = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.npz")
            source.save(path)
            env_path, _ = build_env(opponent_policy=path)
            env_obj, _ = build_env(opponent_policy=source)
            try:
                self.assertEqual(env_path.opponent_policy_source, path)
                self.assertEqual(env_obj.opponent_policy_source, "memory")
                # 参数逐元素一致 → 同一份策略。
                for key in Policy.PARAM_NAMES:
                    np.testing.assert_allclose(
                        env_path.opponent_policy.params[key], source.params[key], atol=1e-12)
                # 同一观测下动作一致（贪心、无 epsilon）。
                obs = np.zeros(OBS_DIM)
                rng = np.random.default_rng(0)
                self.assertEqual(int(env_path.opponent_policy.act(obs, rng, greedy=True)),
                                 int(env_obj.opponent_policy.act(obs, rng, greedy=True)))
            finally:
                env_path.close()
                env_obj.close()

    def test_action_failures_are_counted_not_raised(self) -> None:
        policy = CountingPolicy(action=2)          # 每步都攻击 → 每步都失败
        links = [FakeLink("Glad_Bot_1"), FailingLink("Glad_Bot_2")]
        env, _ = build_env(opponent_policy=policy, links=links)
        try:
            env.reset()
            _obs, reward, _terminated, _truncated, _info = env.step(0)
            self.assertGreaterEqual(env.opponent_policy_stats["actionFailures"].get("OUT_OF_REACH", 0), 1)
            # 学习方的 reward 照常结算（不因对手失误变成异常/NaN）。
            self.assertTrue(np.isfinite(float(reward)))
        finally:
            env.close()

    def test_invalid_arguments_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_env(opponent_policy=123)                     # 既不是路径也没有 act
        with self.assertRaises(ValueError):
            build_env(opponent_policy=CountingPolicy(), opponent_epsilon=1.5)

    def test_multi_agent_mode_rejects_opponent_policy(self) -> None:
        """多智能体模式下动作由调用方给，配了 opponent_policy 会静默失效 → 直接报错。"""
        links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
        with self.assertRaises(ValueError):
            GladiatorEnv(
                [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
                 BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
                links=links, connect=True, single_agent=False,
                opponent_policy=CountingPolicy(),
            )


class FakeVec:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close_all(self) -> None:
        self.closed = True
        self.close_calls += 1


class SelfPlayLoopTest(unittest.TestCase):
    def test_rounds_snapshot_opponent_and_report(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=3)
        built: list = []
        vecs: list[FakeVec] = []
        trained: list = []
        evaluated: list = []

        def build_vec(opponent):
            built.append(opponent)
            vec = FakeVec()
            vecs.append(vec)
            return vec

        def train(vec, episodes):
            trained.append((vec, episodes))
            return [{"episodes": 2, "wins": 1, "steps": 30}]

        def evaluate(current):
            evaluated.append(current)
            return {"summary": {"overall": {"episodes": 4, "wins": 2, "winRate": 0.5,
                                            "winRateCI": (0.2, 0.8)}}}

        with tempfile.TemporaryDirectory() as tmp:
            result = run_selfplay(
                policy, rounds=3, episodes_per_round=2, snapshot_dir=tmp,
                build_vec=build_vec, train=train, evaluate=evaluate,
                init_policy="init.npz", log=lambda message: None,
            )
            records = result["rounds"]
            self.assertEqual(len(records), 3)
            snapshots = [os.path.join(tmp, f"round{index}.npz") for index in range(3)]
            for path in snapshots:
                self.assertTrue(os.path.isfile(path), f"缺少快照 {path}")
            # 第 0 轮用 init-policy，之后每轮用上一轮快照。
            self.assertEqual(built, ["init.npz", snapshots[0], snapshots[1]])
            self.assertEqual([item["opponent"] for item in records],
                             ["init.npz", snapshots[0], snapshots[1]])
            self.assertEqual(records[1]["opponentSource"], "previous-snapshot")
            # 每轮都训练过、评估过，且 eval 拿到的是同一个 learner 对象。
            self.assertEqual(len(trained), 3)
            self.assertTrue(all(current is policy for current in evaluated))
            self.assertEqual(trained[0][1], 2)
            # reward 统计与快照 sha256 都有
            self.assertEqual(records[0]["trainWinRate"], 0.5)
            self.assertEqual(len(records[0]["snapshotSha256"]), 64)
            # 每轮的 VecEnv 都关过（异常路径也一样）。
            self.assertTrue(all(vec.closed for vec in vecs))

    def test_round_zero_falls_back_to_its_own_snapshot(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=4)
        built: list = []
        with tempfile.TemporaryDirectory() as tmp:
            run_selfplay(
                policy, rounds=1, episodes_per_round=1, snapshot_dir=tmp,
                build_vec=lambda opponent: (built.append(opponent), FakeVec())[1],
                train=lambda vec, episodes: [],
                evaluate=lambda current: {},
                init_policy=None, log=lambda message: None,
            )
            self.assertEqual(built, [os.path.join(tmp, "round0.npz")])

    def test_vec_closed_when_train_raises(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=6)
        vec = FakeVec()

        def boom(_vec, _episodes):
            raise RuntimeError("采样炸了")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_selfplay(
                    policy, rounds=1, episodes_per_round=1, snapshot_dir=tmp,
                    build_vec=lambda opponent: vec, train=boom,
                    evaluate=lambda current: {}, log=lambda message: None,
                )
        self.assertTrue(vec.closed, "训练异常时也要 close_all")

    def test_argument_validation(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=8, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_selfplay(policy, rounds=0, episodes_per_round=1, snapshot_dir=tmp,
                             build_vec=lambda o: FakeVec(), train=lambda v, e: [],
                             evaluate=lambda c: {}, log=lambda m: None)
            with self.assertRaises(ValueError):
                run_selfplay(policy, rounds=1, episodes_per_round=0, snapshot_dir=tmp,
                             build_vec=lambda o: FakeVec(), train=lambda v, e: [],
                             evaluate=lambda c: {}, log=lambda m: None)

    def test_train_win_rate_tolerates_missing_fields(self) -> None:
        self.assertIsNone(_train_win_rate([]))
        self.assertIsNone(_train_win_rate([{"steps": 10}]))
        self.assertAlmostEqual(_train_win_rate([{"wins": 1, "episodes": 4}]), 0.25)
        self.assertAlmostEqual(
            _train_win_rate([{"wins": 1, "episodes": 2}, {"wins": 2, "episodes": 2}]), 0.75)


if __name__ == "__main__":
    unittest.main()
