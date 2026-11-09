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

    def test_damage_dealt_is_2_5_per_hp(self) -> None:
        value, events = self.compute(opponent_current=obs(14), attacked=True)
        self.assertAlmostEqual(value, 2.5 * 6, places=6)
        self.assertEqual(events[0]["type"], "hit")

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


if __name__ == "__main__":
    unittest.main()
