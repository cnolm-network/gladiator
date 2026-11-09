from __future__ import annotations

import math
import unittest

import numpy as np

from gladiator_env.env import ArenaConfig, BotConfig
from gladiator_env.spaces import (
    ACTION_NAMES,
    NUM_ACTIONS,
    OBS_DIM,
    decode_action,
    encode_observation,
    nearest_player,
    pitch_to_target,
    yaw_to_target,
)
from gladiator_env.testing import make_observation, make_target


class EncodeObservationTest(unittest.TestCase):
    def test_shape_and_dtype(self):
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertEqual(vector.shape, (OBS_DIM,))
        self.assertEqual(vector.dtype, np.float32)

    def test_missing_fields_are_defaulted(self):
        vector = encode_observation({}, "Glad_Bot_1", step_fraction=0.5)
        self.assertAlmostEqual(float(vector[0]), 1.0)   # health 默认 20/20
        self.assertAlmostEqual(float(vector[1]), 1.0)   # food 默认 20/20
        self.assertEqual(float(vector[13]), 0.0)        # 没有对手
        self.assertAlmostEqual(float(vector[14]), 0.5)
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_nan_is_replaced(self):
        observation = make_observation("Glad_Bot_1")
        observation["self"]["health"] = float("nan")
        vector = encode_observation(observation, "Glad_Bot_1")
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_target_relative_position(self):
        observation = make_observation(
            "Glad_Bot_1",
            x=10.0,
            z=10.0,
            targets=[make_target(91, "Glad_Bot_2", x=13.0, z=14.0, distance=5.0)],
        )
        vector = encode_observation(observation, "Glad_Bot_1", opponent_health=12.0)
        self.assertAlmostEqual(float(vector[7]), 3.0 / 16.0)
        self.assertAlmostEqual(float(vector[9]), 4.0 / 16.0)
        self.assertAlmostEqual(float(vector[10]), 5.0 / 32.0)
        self.assertAlmostEqual(float(vector[11]), 12.0 / 20.0)
        self.assertEqual(float(vector[13]), 1.0)

    def test_opponent_health_override_beats_entity_default(self):
        observation = make_observation(
            "Glad_Bot_1", targets=[make_target(91, "Glad_Bot_2", distance=2.0)])
        vector = encode_observation(observation, "Glad_Bot_1", opponent_health=6.0)
        self.assertAlmostEqual(float(vector[11]), 0.3)

    def test_nearest_player_skips_self(self):
        observation = make_observation("Glad_Bot_1", targets=[
            make_target(1, "Glad_Bot_1", distance=0.1),
            make_target(2, "Glad_Bot_2", distance=4.0),
        ])
        target = nearest_player(observation, "Glad_Bot_1")
        self.assertIsNotNone(target)
        self.assertEqual(target["name"], "Glad_Bot_2")


class DecodeActionTest(unittest.TestCase):
    def setUp(self):
        self.observation = make_observation(
            "Glad_Bot_1", x=0.0, z=0.0,
            targets=[make_target(91, "Glad_Bot_2", x=0.0, z=2.0, distance=2.0)])

    def test_opponents_filter_excludes_spectators(self):
        # 观战者（Watcher）距离更近，但不在参赛名单里，不应成为目标。
        observation = make_observation(
            "Glad_Bot_1",
            x=0.0,
            y=4.0,
            z=0.0,
            targets=[
                make_target(11, "Watcher", x=1.0, y=4.0, z=0.0, distance=1.0),
                make_target(12, "Glad_Bot_2", x=4.0, y=4.0, z=0.0, distance=4.0),
            ],
        )
        self.assertEqual(nearest_player(observation, "Glad_Bot_1")["name"], "Watcher")
        picked = nearest_player(observation, "Glad_Bot_1", {"Glad_Bot_1", "Glad_Bot_2"})
        self.assertEqual(picked["name"], "Glad_Bot_2")

        vector = encode_observation(
            observation, "Glad_Bot_1", opponents={"Glad_Bot_1", "Glad_Bot_2"}
        )
        self.assertAlmostEqual(float(vector[7]), 4.0 / 16.0, places=4)

    def test_action_table_matches_protocol_types(self):
        self.assertEqual(len(ACTION_NAMES), NUM_ACTIONS)
        self.assertIn("attack", ACTION_NAMES)
        self.assertIn("idle", ACTION_NAMES)

    def test_idle_and_jump(self):
        self.assertEqual(decode_action(0, self.observation, "Glad_Bot_1")[0]["type"], "idle")
        self.assertEqual(decode_action(6, self.observation, "Glad_Bot_1")[0]["type"], "jump")

    def test_approach_looks_then_moves(self):
        actions = decode_action(1, self.observation, "Glad_Bot_1")
        self.assertEqual([a["type"] for a in actions], ["look", "move"])
        self.assertEqual(actions[1]["params"]["direction"], "forward")

    def test_attack_in_range_uses_entity_id(self):
        actions = decode_action(2, self.observation, "Glad_Bot_1")
        attack = [a for a in actions if a["type"] == "attack"]
        self.assertEqual(len(attack), 1)
        self.assertEqual(attack[0]["params"]["targetEntityId"], 91)
        # 挥击后补一次朝向，避免被击退后背对目标空挥
        self.assertEqual(actions[-1]["type"], "look")

    def test_approach_sprints_when_far(self):
        far = make_observation(
            "Glad_Bot_1", targets=[make_target(91, "Glad_Bot_2", x=0.0, z=8.0, distance=8.0)])
        actions = decode_action(1, far, "Glad_Bot_1")
        move = [a for a in actions if a["type"] == "move"][0]
        self.assertTrue(move["params"]["sprint"])
        near = make_observation(
            "Glad_Bot_1", targets=[make_target(91, "Glad_Bot_2", x=2.0, z=0.0, distance=2.0)])
        actions = decode_action(1, near, "Glad_Bot_1")
        move = [a for a in actions if a["type"] == "move"][0]
        self.assertFalse(move["params"]["sprint"])

    def test_attack_respects_reach(self):
        # 距离 3.5 在默认 reach(3.99) 内，在旧阈值 3.0 外
        obs = make_observation(
            "Glad_Bot_1", targets=[make_target(91, "Glad_Bot_2", x=3.5, z=0.0, distance=3.5)])
        actions = decode_action(2, obs, "Glad_Bot_1")
        self.assertTrue(any(a["type"] == "attack" for a in actions))
        actions = decode_action(2, obs, "Glad_Bot_1", reach=2.0)
        self.assertEqual(actions[-1]["type"], "move")

    def test_attack_out_of_range_approaches_instead(self):
        far = make_observation(
            "Glad_Bot_1", targets=[make_target(91, "Glad_Bot_2", x=0.0, z=8.0, distance=8.0)])
        actions = decode_action(2, far, "Glad_Bot_1")
        self.assertEqual(actions[-1]["type"], "move")

    def test_spectator_only_is_not_attacked(self):
        # 附近只有观战者时，attack 退化为 idle，而不是对着观战者挥空。
        observation = make_observation(
            "Glad_Bot_1",
            x=0.0,
            y=4.0,
            z=0.0,
            targets=[make_target(11, "Watcher", x=1.0, y=4.0, z=0.0, distance=1.0)],
        )
        actions = decode_action(2, observation, "Glad_Bot_1", {"Glad_Bot_1", "Glad_Bot_2"})
        self.assertEqual([a["type"] for a in actions], ["idle"])

    def test_no_target_falls_back_to_idle(self):
        alone = make_observation("Glad_Bot_1")
        self.assertEqual(decode_action(1, alone, "Glad_Bot_1")[0]["type"], "idle")

    def test_out_of_range_action_raises(self):
        with self.assertRaises(ValueError):
            decode_action(NUM_ACTIONS, self.observation, "Glad_Bot_1")


class GeometryTest(unittest.TestCase):
    def test_look_towards_positive_z_is_pi(self):
        # mineflayer 约定：yaw=0 面向 -z，因此正 +z 方向约为 ±pi
        yaw = yaw_to_target(0.0, 1.0)
        self.assertAlmostEqual(abs(yaw), math.pi, places=6)

    def test_look_towards_positive_x(self):
        yaw = yaw_to_target(1.0, 0.0)
        self.assertAlmostEqual(yaw, -math.pi / 2, places=6)

    def test_pitch_points_down_for_lower_target(self):
        self.assertGreater(pitch_to_target(-2.0, 2.0), 0.0)


if __name__ == "__main__":
    unittest.main()

class KitAndItemActionTests(unittest.TestCase):
    """动作细化：切物品/用物品/左右键交互。"""

    def test_action_space_contains_item_actions(self):
        self.assertEqual(NUM_ACTIONS, len(ACTION_NAMES))
        for name in ("use_rod", "eat_gapple", "block", "switch_sword"):
            self.assertIn(name, ACTION_NAMES)

    def test_spawn_args_carry_assist_flags(self):
        bot = BotConfig("Bot_A", wtap=True, auto_rod=True, auto_block=False, aim_lock=True)
        args = bot.spawn_args()
        self.assertIn("wtap=on", args)
        self.assertIn("autorod=on", args)
        self.assertIn("autoblock=off", args)
        self.assertIn("aimlock=on", args)
        self.assertIn("reach=3.99", args)

    def test_kit_has_diamond_armor_sword_gapple_rod(self):
        kit = dict(ArenaConfig().kit)
        self.assertIn("diamond_helmet", kit)
        self.assertIn("diamond_sword", kit)
        self.assertEqual(kit["golden_apple"], 5)
        self.assertIn("fishing_rod", kit)

    def test_decode_rod_and_attack(self):
        observation = make_observation("Bot_A", targets=[
            make_target(9, "Bot_B", x=4.0, y=4.0, z=0.0, distance=4.0),
        ])
        rod_actions = decode_action(ACTION_NAMES.index("use_rod"), observation, "Bot_A")
        self.assertEqual([a["type"] for a in rod_actions], ["look", "use"])
        self.assertEqual(rod_actions[1]["params"]["item"], "rod")

    def test_decode_eat_gapple_only_when_hurt(self):
        hurt = {"self": {"health": 8.0, "pos": {"x": 0.0, "y": 4.0, "z": 0.0}}, "entities": []}
        actions = decode_action(ACTION_NAMES.index("eat_gapple"), hurt, "Bot_A")
        self.assertEqual(actions[0]["params"]["item"], "gapple")

    def test_decode_block_and_switch(self):
        observation = {"self": {"health": 20.0, "pos": {"x": 0.0, "y": 4.0, "z": 0.0}}, "entities": []}
        block = decode_action(ACTION_NAMES.index("block"), observation, "Bot_A")
        self.assertEqual(block[0]["params"]["item"], "sword")
        switch = decode_action(ACTION_NAMES.index("switch_sword"), observation, "Bot_A")
        self.assertEqual(switch[0]["params"]["item"], "sword")
