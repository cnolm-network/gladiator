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
        targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=2.0)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, z=0.0,
        targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=2.0)]))
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


def pin_profile(env: GladiatorEnv) -> None:
    """把对手参数固定成确定值：本文件测的是「出手前确认」，不是对手池随机性。

    P1.6 起 aimbot 的 attackProb/approachThreshold 每局采样，若不固定，
    「射程内必出手」的前提会以 ~10% 的概率不成立，测试变 flaky。
    """
    env._opp_profile = {"mode": "aimbot", "attackProb": 1.0,
                        "approachThreshold": 3.9, "reachOffset": 0.5,
                        "startDelayTicks": 0, "attackCooldownTicks": 0}
    # P1.7：reset 已按采样值装载了起手延迟/冷静期，这里一并清零，保持"射程内每次都能出手"。
    env._opp_hold_ticks = {bot.username: 0 for bot in env.bots}
    env._opp_next_attack_tick = {bot.username: 0 for bot in env.bots}


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
        pin_profile(env)
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
        pin_profile(env)
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
        pin_profile(env)
        links[1].actions.clear()
        env.step(0)  # hold=1
        self.assertEqual(env._opponent_hold.get("Glad_Bot_2"), 1)

        set_opponent_distance(env, links, 3.6)  # 第二 tick 被击退/拉开（> 2.25 的对手射程）
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
        pin_profile(env)
        links[1].actions.clear()
        env.step(0)
        self.assertEqual(env._opponent_target.get("Glad_Bot_2"), 1)

        # 换一个目标：必须重新确认，不能拿上一次的确认直接出手。
        push_observation(env, links, make_observation(
            "Glad_Bot_2", x=1.5, z=0.0,
            targets=[make_target(9, "Glad_Bot_1", x=-1.0, z=0.0, distance=2.0)]))
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
        pin_profile(env)
        links[0].actions.clear()

        env.step(2)  # 学习方选 attack

        kinds = [action["type"] for action in links[0].actions]
        self.assertIn("attack", kinds, kinds)
        env.close()


if __name__ == "__main__":
    unittest.main()
