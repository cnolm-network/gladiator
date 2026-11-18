"""G-508：全方位塑形奖励的权重与约束。"""

from __future__ import annotations

import unittest

from gladiator_env.shaping import (
    COMBO_BONUS_CAP,
    ShapingReward,
    ShapingWeights,
)


def obs(health: float, state: str = "Alive") -> dict:
    return {"self": {"health": health, "pos": {"x": 0, "y": 4, "z": 0}}, "state": state}


class ShapingRewardTest(unittest.TestCase):
    """旧口径的数值断言：显式关掉 P2 的 PBRS 与 dense 封顶。

    P2 默认开启 PBRS（每步加势能差）与 ±5 封顶，会让「纯伤害 = 2.5×HP」这类
    单点断言随之改变；这里测的是**关掉新机制后的原语义**，新机制的断言见
    ``P2ShapingTest``。
    """

    def setUp(self) -> None:
        self.w = ShapingWeights(shaping_cap=0.0)
        self.reward = ShapingReward(self.w, use_pbrs=False)

    def compute(self, **kwargs):
        base = dict(
            previous=obs(20), current=obs(20), opponent_previous=obs(20),
            opponent_current=obs(20), terminated=False, attacked=False,
            distance=None, now=0.0, dt=0.2,
        )
        base.update(kwargs)
        return self.reward.compute("Bot_A", **base)

    def test_damage_dealt_is_2_5_per_hp(self) -> None:
        value, events = self.compute(opponent_current=obs(14), attacked=True)
        self.assertAlmostEqual(value, 2.5 * 6, places=6)
        self.assertEqual(events[0]["type"], "hit")

    def test_attack_in_reach_pays_once_per_attack(self) -> None:
        """P1.5：射程内真的挥刀给一次奖励（不是每 step 都给）。"""
        self.reward.reset()  # 独立状态，避免上一次的 last_taken 等串味
        value, events = self.compute(attacked=True, in_reach=True, distance=3.0)
        self.assertIn("attackInReach", [e["type"] for e in events])
        w = ShapingWeights()
        # 没命中 → 还要吃空挥惩罚；射程内 3.0 格 → 压迫位 + pressure·dt
        self.assertAlmostEqual(
            value,
            w.whiff + w.attack_attempt + w.attack_in_reach + w.pressure * 0.2,
            places=6)

        # 不在射程内挥刀：没有这一项
        self.reward.reset()
        _no_reach, far_events = self.compute(attacked=True, in_reach=False, distance=6.0)
        self.assertNotIn("attackInReach", [e["type"] for e in far_events])
        # 没有攻击动作：也没有这一项（哪怕站在原地贴着）
        self.reward.reset()
        _no_attack, idle_events = self.compute(attacked=False, in_reach=True, distance=3.0)
        self.assertNotIn("attackInReach", [e["type"] for e in idle_events])

    def test_attack_in_reach_default_weight(self) -> None:
        self.assertEqual(ShapingWeights().attack_in_reach, 0.05)
        self.assertEqual(ShapingWeights().as_dict()["attackInReachPerAttack"], 0.05)

    def test_damage_taken_is_minus_1_2_per_hp(self) -> None:
        value, _ = self.compute(current=obs(16))
        self.assertAlmostEqual(value, -1.2 * 4, places=6)

    def test_kill_and_death_are_symmetric(self) -> None:
        kill, events = self.compute(
            opponent_current=obs(0, "Dead"), terminated=True, now=5.0,
        )
        self.assertIn("kill", [e["type"] for e in events])
        self.assertGreaterEqual(kill, 20.0)

        self.reward.reset()
        death, events = self.compute(current=obs(0, "Dead"), terminated=True, now=5.0)
        self.assertIn("death", [e["type"] for e in events])
        self.assertLessEqual(death, -20.0)

    def test_whiff_penalty(self) -> None:
        # 空挥惩罚已从 -0.3 降到 -0.1（原来太重，模型宁可格挡也不敢出剑）。
        value, events = self.compute(attacked=True)
        self.assertAlmostEqual(value, -0.1, places=6)
        self.assertEqual(events[0]["type"], "whiff")

    def test_attack_attempt_bonus_in_range(self) -> None:
        value, _ = self.compute(attacked=True, distance=3.6, dt=0.0)
        self.assertAlmostEqual(value, -0.1 + 0.05, places=6)

    def test_waste_penalty_for_blocking_far_away(self) -> None:
        value, events = self.compute(action="block", distance=6.0, dt=0.0)
        self.assertAlmostEqual(value, -0.08, places=6)
        self.assertEqual(events[0]["type"], "waste")

    def test_closing_distance_reward(self) -> None:
        value, _ = self.compute(distance=5.0, previous_distance=6.0, dt=0.0)
        self.assertAlmostEqual(value, 0.02, places=6)

    def test_blocking_in_range_has_no_waste_penalty(self) -> None:
        value, events = self.compute(action="block", distance=3.2, dt=0.0)
        self.assertAlmostEqual(value, 0.0, places=6)
        self.assertFalse(any(event["type"] == "waste" for event in events))

    def test_combo_bonus_is_capped(self) -> None:
        total = 0.0
        for step in range(6):
            value, events = self.compute(
                opponent_current=obs(20 - (step + 1)), attacked=True, now=step * 0.5,
            )
            total += value
            for event in events:
                if event["type"] == "combo":
                    self.assertLessEqual(event["bonus"], COMBO_BONUS_CAP)
            # 下一次的“上一帧”要跟上当前血量
        # 6 连击的连击奖励合计不超过 3.0
        combo_total = self.reward.state("Bot_A").totals.get("combo", 0.0)
        self.assertLessEqual(combo_total, COMBO_BONUS_CAP)
        self.assertGreater(combo_total, 0.0)

    def test_threat_spacing_only_inside_window_and_range(self) -> None:
        # 命中后窗口内、距离 3.5 → 给间距奖励
        self.compute(opponent_current=obs(14), attacked=True, now=1.0, distance=3.5)
        value, _ = self.compute(now=1.2, distance=3.5, dt=0.2)
        self.assertGreater(value, 0.0)
        # 威胁间距 0.3/s + 压迫位 0.05/s，都按 dt 结算
        self.assertAlmostEqual(value, (0.3 + 0.05) * 0.2, places=6)

    def test_threat_spacing_ignored_outside_range(self) -> None:
        self.compute(opponent_current=obs(14), attacked=True, now=1.0, distance=8.0)
        value, _ = self.compute(now=1.2, distance=8.0, dt=0.2)
        self.assertAlmostEqual(value, 0.0, places=6)

    def test_standing_still_penalty_after_three_seconds(self) -> None:
        # 站在同一个点连续 3 秒以上（不移动）开始扣分；每步 0.1/s。
        total = 0.0
        for step in range(20):
            value, _ = self.compute(now=step * 0.2, dt=0.2)
            total += value
        self.assertAlmostEqual(total, -0.1 * 0.2 * 5, places=6)

    def test_walking_is_not_penalized(self) -> None:
        # 每步向前走 0.2 格：不会触发站桩惩罚，也没有任何隐藏的走位惩罚。
        for step in range(20):
            current = obs(20)
            current["self"]["pos"]["z"] = step * 0.2
            value, _ = self.compute(current=current, now=step * 0.2, dt=0.2)
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_reset_clears_state(self) -> None:
        self.compute(opponent_current=obs(14), attacked=True)
        self.assertGreater(self.reward.state("Bot_A").max_combo, 0)
        self.reward.reset("Bot_A")
        self.assertEqual(self.reward.state("Bot_A").max_combo, 0)

    def test_summary_reports_totals(self) -> None:
        self.compute(opponent_current=obs(14), attacked=True)
        summary = self.reward.summary("Bot_A")
        self.assertGreater(summary["totals"]["damageDealt"], 0)


class P2ShapingTest(unittest.TestCase):
    """P2：PBRS 势能、victory ±50、dense 每步封顶。"""

    def setUp(self) -> None:
        self.reward = ShapingReward()
        self.w = ShapingWeights()

    def compute(self, **kwargs):
        base = dict(
            previous=obs(20), current=obs(20), opponent_previous=obs(20),
            opponent_current=obs(20), terminated=False, attacked=False,
            distance=None, now=0.0, dt=0.2,
        )
        base.update(kwargs)
        return self.reward.compute("Bot_A", **base)

    def test_victory_is_fifty_and_kill_is_small(self):
        self.assertEqual(self.w.victory, 50.0)
        self.assertEqual(self.w.kill, 2.0)
        self.assertEqual(self.w.death, -2.0)
        won, events = self.compute(opponent_current=obs(0, "Dead"), terminated=True)
        self.assertIn("victory", [e["type"] for e in events])
        self.assertGreaterEqual(won, 50.0)
        lost, events = self.compute(current=obs(0, "Dead"), terminated=True)
        self.assertIn("victory", [e["type"] for e in events])
        self.assertLessEqual(lost, -50.0)

    def test_dense_is_capped_per_step(self):
        # 6 点伤害 ×2.5 = 15 → 封顶到 5；裁剪量记进 capped
        value, events = self.compute(opponent_current=obs(14), attacked=True)
        self.assertLessEqual(value, self.w.shaping_cap + 1e-9)
        self.assertIn("capped", [e["type"] for e in events])
        totals = self.reward.summary("Bot_A")["totals"]
        self.assertLess(totals.get("capped", 0.0), 0.0)

    def test_victory_is_not_capped(self):
        value, _ = self.compute(opponent_current=obs(0, "Dead"), terminated=True)
        self.assertGreater(value, self.w.shaping_cap)

    def test_pbrs_rewards_health_advantage(self):
        """对手掉血 → Φ 上升 → 首步 F = γΦ(s') − Φ(s) > 0。"""
        value, events = self.compute(opponent_current=obs(14), attacked=False, distance=None)
        pbrs_events = [e for e in events if e["type"] == "pbrs"]
        self.assertTrue(pbrs_events, events)
        self.assertGreater(pbrs_events[0]["value"], 0.0)

    def test_pbrs_uses_reset_state_for_first_step(self):
        """首步的 Φ(s) 取 reset 状态：同状态 → F = 0。"""
        _value, events = self.compute()
        pbrs = [e for e in events if e["type"] == "pbrs"]
        self.assertEqual(pbrs, [], "状态没变时势能差应为 0")

    def test_pbrs_can_be_disabled(self):
        reward = ShapingReward(use_pbrs=False)
        _value, events = reward.compute(
            "Bot_A", previous=obs(20), current=obs(20),
            opponent_previous=obs(20), opponent_current=obs(14),
            terminated=False, attacked=False, distance=None, now=0.0, dt=0.2,
        )
        self.assertNotIn("pbrs", [e["type"] for e in events])

    def test_potential_distance_term(self):
        # 相距 20 格 → phi_dist = 0；相距 0 格 → 1.0
        far_prev = {"self": {"health": 20, "pos": {"x": 0, "y": 4, "z": 0}}, "state": "Alive"}
        far_cur = {"self": {"health": 20, "pos": {"x": 20, "y": 4, "z": 0}}, "state": "Alive"}
        near_cur = {"self": {"health": 20, "pos": {"x": 0, "y": 4, "z": 0}}, "state": "Alive"}
        self.assertAlmostEqual(self.reward.potential(far_prev, far_cur), 0.0, places=6)
        self.assertAlmostEqual(self.reward.potential(far_prev, near_cur),
                               self.w.pbrs_beta, places=6)

    def test_as_dict_has_pbrs_fields(self):
        payload = self.w.as_dict()
        for key in ("pbrsAlpha", "pbrsBeta", "pbrsGamma", "victory", "shapingCap"):
            self.assertIn(key, payload)
        self.assertEqual(payload["pbrsGamma"], 1.0)


if __name__ == "__main__":
    unittest.main()
