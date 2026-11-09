from __future__ import annotations

import unittest

from gladiator_env.reward import RewardContext, default_reward
from gladiator_env.testing import make_observation


def ctx(previous_health=20.0, current_health=20.0, opponent_previous=20.0, opponent_current=20.0,
        terminated=False, truncated=False, step=1) -> RewardContext:
    return RewardContext(
        username="Glad_Bot_1",
        index=0,
        previous=make_observation("Glad_Bot_1", health=previous_health),
        current=make_observation("Glad_Bot_1", health=current_health,
                                 state="Dead" if current_health <= 0 else "Alive"),
        step=step,
        terminated=terminated,
        truncated=truncated,
        opponent_previous=make_observation("Glad_Bot_2", health=opponent_previous),
        opponent_current=make_observation("Glad_Bot_2", health=opponent_current,
                                          state="Dead" if opponent_current <= 0 else "Alive"),
    )


class DefaultRewardTest(unittest.TestCase):
    def test_damage_dealt_is_positive(self):
        reward = default_reward(ctx(opponent_previous=20.0, opponent_current=16.0))
        self.assertGreater(reward, 0.0)

    def test_damage_taken_is_negative(self):
        reward = default_reward(ctx(previous_health=20.0, current_health=14.0))
        self.assertLess(reward, 0.0)

    def test_kill_bonus(self):
        reward = default_reward(ctx(opponent_previous=4.0, opponent_current=0.0, terminated=True))
        self.assertGreater(reward, 0.5)

    def test_death_penalty(self):
        reward = default_reward(ctx(previous_health=4.0, current_health=0.0, terminated=True))
        self.assertLess(reward, -0.5)

    def test_step_penalty_keeps_idle_slightly_negative(self):
        reward = default_reward(ctx(step=10))
        self.assertLess(reward, 0.0)

    def test_no_rule_judgement_without_termination(self):
        # 奖励只看物理量；没有结束信号时不应出现击杀/死亡项
        reward = default_reward(ctx())
        self.assertAlmostEqual(reward, -0.001, places=6)


if __name__ == "__main__":
    unittest.main()
