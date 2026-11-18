"""P1.6 随机化：起始分布（arena jitter）与对手池参数。

目的：训练分布里若只有「同一站位 + 同一节奏的对手」，策略会过拟合，
训练胜率 62.5% 而评估 0 胜。这里锁住随机化的三条契约：
同一 seed 可复现、jitter=0 回到完全确定、对手参数始终落在规定区间且 reach 有下限。
"""

from __future__ import annotations

import unittest

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.env import (
    MIN_OPPONENT_REACH,
    MIN_SPACING,
    OPPONENT_PROFILES,
    OPPONENT_REACH_OFFSET,
)
from gladiator_env.testing import FakeLink, make_observation, make_target


def build_env(*, mode: str = "rusher", **arena_kw) -> GladiatorEnv:
    links = [FakeLink("Bot_T1"), FakeLink("Bot_T2")]
    for index, link in enumerate(links):
        other = "Bot_T2" if index == 0 else "Bot_T1"
        link.set_observation(make_observation(
            link.username, x=-1.5 if index == 0 else 1.5, z=0.0,
            targets=[make_target(2 if index == 0 else 1, other,
                                 x=1.5 if index == 0 else -1.5, z=0.0, distance=3.0)]))
    kwargs = dict(center=(0.0, 4.0, 0.0), spacing=6.0, prep_seconds=0.0, give_kit=False)
    kwargs.update(arena_kw)
    return GladiatorEnv(
        [BotConfig("Bot_T1", "ws://127.0.0.1:8766"),
         BotConfig("Bot_T2", "ws://127.0.0.1:8767")],
        ArenaConfig(**kwargs),
        links=links, connect=True, control_hz=200.0, action_ticks=1,
        single_agent=True, opponent_mode=mode,
    )


def arena_of(env: GladiatorEnv, seed: int) -> dict:
    _obs, info = env.reset(seed=seed)
    return info["arena"]


class ArenaJitterTest(unittest.TestCase):
    def test_same_seed_reproduces_arena(self):
        env = build_env(jitter=(2.0, 1.0))
        first = arena_of(env, 1234)
        second = arena_of(env, 1234)
        self.assertEqual(first, second)
        env.close()

    def test_different_seed_changes_arena(self):
        env = build_env(jitter=(2.0, 1.0))
        seen = {tuple(arena_of(env, seed)["center"]) + (arena_of(env, seed)["spacing"],)
                for seed in (1, 2, 3, 4, 5)}
        self.assertGreater(len(seen), 1, "不同 seed 应该给出不同的起始分布")
        env.close()

    def test_zero_jitter_is_exactly_config(self):
        env = build_env(jitter=(0.0, 0.0))
        for seed in (1, 2, 3):
            arena = arena_of(env, seed)
            self.assertEqual(arena["center"], [0.0, 4.0, 0.0])
            self.assertEqual(arena["spacing"], 6.0)
        env.close()

    def test_spacing_never_below_min(self):
        # 抖动幅度大于间距本身：必须夹到 MIN_SPACING，不能抖出「贴脸开局」
        env = build_env(jitter=(2.0, 10.0), spacing=6.0)
        for seed in range(20):
            self.assertGreaterEqual(arena_of(env, seed)["spacing"], MIN_SPACING)
        env.close()

    def test_both_bots_share_the_same_arena(self):
        env = build_env(jitter=(2.0, 1.0))
        env.reset(seed=7)
        a = env._spawn_point(0)
        b = env._spawn_point(1)
        center = env._arena_center
        # 对称展开：两只 Bot 的 x 相对中心互为镜像，y/z 相同
        self.assertAlmostEqual((a[0] + b[0]) / 2.0, center[0], places=6)
        self.assertAlmostEqual(a[2], b[2], places=6)
        self.assertAlmostEqual(a[1], center[1], places=6)
        env.close()

    def test_center_offsets_within_range(self):
        env = build_env(jitter=(2.0, 0.0), center=(0.0, 4.0, 0.0))
        for seed in range(10):
            arena = arena_of(env, seed)
            self.assertLessEqual(abs(arena["center"][0]), 2.0)
            self.assertLessEqual(abs(arena["center"][2]), 2.0)
        env.close()


class OpponentProfileTest(unittest.TestCase):
    def test_same_seed_reproduces_profile(self):
        env = build_env(mode="rusher")
        env.reset(seed=42)
        first = dict(env._opp_profile)
        env.reset(seed=42)
        self.assertEqual(first, env._opp_profile)
        env.close()

    def test_different_seed_changes_profile(self):
        env = build_env(mode="scripted")
        seen = set()
        for seed in range(6):
            env.reset(seed=seed)
            seen.add(round(env._opp_profile["attackProb"], 6))
        self.assertGreater(len(seen), 1)
        env.close()

    def test_profile_within_declared_ranges(self):
        for mode in ("rusher", "scripted", "aimbot"):
            env = build_env(mode=mode)
            spec = OPPONENT_PROFILES[mode]
            for seed in range(12):
                _obs, info = env.reset(seed=seed)
                profile = info["opponentProfile"]
                for key, (low, high) in spec.items():
                    self.assertGreaterEqual(profile[key], low, f"{mode}.{key}")
                    self.assertLessEqual(profile[key], high, f"{mode}.{key}")
                self.assertGreaterEqual(profile["reachOffset"], OPPONENT_REACH_OFFSET[0])
                self.assertLessEqual(profile["reachOffset"], OPPONENT_REACH_OFFSET[1])
            env.close()

    def test_opponent_reach_has_floor(self):
        env = build_env(mode="rusher")
        env.reset(seed=3)
        # 极端小的 reach 也必须夹到下限
        self.assertGreaterEqual(env._opponent_reach(1, 1.0), MIN_OPPONENT_REACH)
        # 正常 reach：等于 reach - reachOffset，且仍 ≥ 下限
        value = env._opponent_reach(1, 3.99)
        self.assertAlmostEqual(value, 3.99 - env._opp_profile["reachOffset"], places=6)
        self.assertGreaterEqual(value, MIN_OPPONENT_REACH)
        env.close()

    def test_new_profile_fields_reproducible(self):
        """P1.7：起手延迟 / 出手间隔同样「同 seed 稳定、不同 seed 会变」。"""
        env = build_env(mode="rusher")
        env.reset(seed=5)
        first = (env._opp_profile["startDelayTicks"], env._opp_profile["attackCooldownTicks"])
        env.reset(seed=5)
        self.assertEqual(
            first,
            (env._opp_profile["startDelayTicks"], env._opp_profile["attackCooldownTicks"]))
        seen = set()
        for seed in range(8):
            env.reset(seed=seed)
            seen.add((env._opp_profile["startDelayTicks"],
                      env._opp_profile["attackCooldownTicks"]))
        self.assertGreater(len(seen), 1, "不同 seed 应该给出不同的起手延迟/间隔")
        env.close()

    def test_new_fields_are_ints_in_range(self):
        for mode in ("rusher", "scripted", "aimbot"):
            env = build_env(mode=mode)
            low_delay, high_delay = OPPONENT_PROFILES[mode]["startDelayTicks"]
            low_cd, high_cd = OPPONENT_PROFILES[mode]["attackCooldownTicks"]
            for seed in range(12):
                _obs, info = env.reset(seed=seed)
                profile = info["opponentProfile"]
                self.assertIsInstance(profile["startDelayTicks"], int)
                self.assertIsInstance(profile["attackCooldownTicks"], int)
                self.assertGreaterEqual(profile["startDelayTicks"], low_delay)
                self.assertLessEqual(profile["startDelayTicks"], high_delay)
                self.assertGreaterEqual(profile["attackCooldownTicks"], low_cd)
                self.assertLessEqual(profile["attackCooldownTicks"], high_cd)
            env.close()

    def test_start_delay_holds_opponent_idle(self):
        """startDelayTicks=10：前 10 个动作必须是 idle，第 11 个才恢复正常逻辑。"""
        env = build_env(mode="rusher")
        env.reset(seed=1)
        env._opp_profile = {"mode": "rusher", "attackProb": 1.0, "approachThreshold": 3.9,
                            "reachOffset": 0.5, "startDelayTicks": 10,
                            "attackCooldownTicks": 0}
        env._opp_hold_ticks = {"Bot_T1": 0, "Bot_T2": 10}
        env._opp_next_attack_tick = {"Bot_T1": 0, "Bot_T2": 0}

        for step in range(10):
            env._steps = step + 1
            self.assertFalse(env._opponent_ready(1))
            self.assertEqual(env._opponent_action(1), 0, f"第 {step + 1} tick 应 idle")

        self.assertTrue(env._opponent_ready(1))
        env._steps = 11
        self.assertNotEqual(env._opponent_action(1), 0, "延迟结束后应恢复正常行为")
        env.close()

    def test_attack_cooldown_spaces_out_attacks(self):
        """attackCooldownTicks=3：相邻两次出手至少隔 3 个 step。"""
        env = build_env(mode="rusher")
        env.reset(seed=1)
        env._opp_profile = {"mode": "rusher", "attackProb": 1.0, "approachThreshold": 3.9,
                            "reachOffset": 0.5, "startDelayTicks": 0,
                            "attackCooldownTicks": 3}
        env._opp_hold_ticks = {"Bot_T1": 0, "Bot_T2": 0}
        env._opp_next_attack_tick = {"Bot_T1": 0, "Bot_T2": 0}

        attack_steps = []
        for step in range(1, 13):
            env._steps = step
            if env._opponent_action(1) == 2:
                attack_steps.append(step)

        self.assertGreaterEqual(len(attack_steps), 2, "冷却期内应仍有多次出手")
        gaps = [b - a for a, b in zip(attack_steps, attack_steps[1:])]
        self.assertTrue(all(gap >= 3 for gap in gaps), f"间隔不足 3: {attack_steps}")
        env.close()

    def test_non_pool_mode_keeps_legacy_offset(self):
        """random/wander/idle 不采样，reach 折减保持旧常量 0.75。"""
        env = build_env(mode="random")
        _obs, info = env.reset(seed=1)
        self.assertNotIn("attackProb", info["opponentProfile"])
        self.assertEqual(info["opponentProfile"]["startDelayTicks"], 0)
        self.assertEqual(info["opponentProfile"]["attackCooldownTicks"], 0)
        self.assertAlmostEqual(env._opponent_reach(1, 3.99), 3.99 - 0.75, places=6)
        env.close()


if __name__ == "__main__":
    unittest.main()
