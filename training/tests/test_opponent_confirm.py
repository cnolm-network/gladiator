"""A1：代码驱动的对手「出手前先确认」（不引入随机性，只影响对手）。

场景来自真实训练日志：对手在 4.3~4.8 格被判 attack，但观测到出手之间隔着一个 tick
（≈1.1 格位移）+ 击退，运行时于是回 OUT_OF_REACH。
"""

from __future__ import annotations

import unittest

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target


def build_env(**kwargs) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
    links[0].set_observation(make_observation(
        "Glad_Bot_1", x=-1.5, z=0.0,
        targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=3.0)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, z=0.0,
        targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=3.0)]))
    options = dict(single_agent=True, opponent_mode="aimbot", control_hz=200.0, action_ticks=1)
    options.update(kwargs)
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        **options,
    )
    return env, links


def push_observation(env: GladiatorEnv, links: list[FakeLink], observation: dict,
                     index: int = 1) -> None:
    """把新观测同时塞给通道与 env._raw。

    env.step 用的是「上一帧刷新到的观测」，与真实链路一致（tick N 的观测在 tick N-1 末尾到达）；
    测试要模拟「这一 tick 已拿到新观测」，所以两处都要写。
    """
    links[index].set_observation(observation)
    env._raw[index] = observation


def set_opponent_distance(env: GladiatorEnv, links: list[FakeLink], distance: float) -> None:
    push_observation(env, links, make_observation(
        "Glad_Bot_2", x=1.5, z=0.0,
        targets=[make_target(1, "Glad_Bot_1", x=1.5 - distance, z=0.0, distance=distance)]))


class OpponentConfirmTest(unittest.TestCase):
    def test_first_tick_only_approaches(self):
        env, links = build_env()
        env.reset()
        links[1].actions.clear()

        env.step(0)

        kinds = [action["type"] for action in links[1].actions]
        self.assertNotIn("attack", kinds, kinds)
        self.assertIn("move", kinds)
        self.assertEqual(env._opponent_hold.get("Glad_Bot_2"), 1)
        env.close()

    def test_second_tick_actually_attacks(self):
        env, links = build_env()
        env.reset()
        links[1].actions.clear()
        env.step(0)  # 第一次判到出手 → 只贴近

        links[1].actions.clear()
        env.step(0)  # 第二 tick 复核仍在射程 → 真出手

        kinds = [action["type"] for action in links[1].actions]
        self.assertIn("attack", kinds, kinds)
        self.assertNotIn("Glad_Bot_2", env._opponent_hold)
        env.close()

    def test_distance_grows_on_second_tick_falls_back_to_move(self):
        env, links = build_env()
        env.reset()
        links[1].actions.clear()
        env.step(0)  # hold=1
        self.assertEqual(env._opponent_hold.get("Glad_Bot_2"), 1)

        set_opponent_distance(env, links, 4.6)  # 第二 tick 被击退/拉开
        links[1].actions.clear()
        env.step(0)

        kinds = [action["type"] for action in links[1].actions]
        self.assertNotIn("attack", kinds, kinds)
        self.assertIn("move", kinds)
        self.assertNotIn("Glad_Bot_2", env._opponent_hold)
        env.close()

    def test_switching_target_requires_reconfirm(self):
        env, links = build_env()
        env.reset()
        links[1].actions.clear()
        env.step(0)
        self.assertEqual(env._opponent_target.get("Glad_Bot_2"), 1)

        # 换一个目标：必须重新确认，不能拿上一次的确认直接出手。
        push_observation(env, links, make_observation(
            "Glad_Bot_2", x=1.5, z=0.0,
            targets=[make_target(9, "Glad_Bot_1", x=-1.0, z=0.0, distance=2.5)]))
        links[1].actions.clear()
        env.step(0)

        kinds = [action["type"] for action in links[1].actions]
        self.assertNotIn("attack", kinds, kinds)
        self.assertEqual(env._opponent_target.get("Glad_Bot_2"), 9)
        env.close()

    def test_learner_actions_untouched(self):
        """学习方（index 0）不受影响：同一 tick 的 attack 照发。"""
        env, links = build_env()
        env.reset()
        links[0].actions.clear()

        env.step(2)  # 学习方选 attack

        kinds = [action["type"] for action in links[0].actions]
        self.assertIn("attack", kinds, kinds)
        env.close()


if __name__ == "__main__":
    unittest.main()
