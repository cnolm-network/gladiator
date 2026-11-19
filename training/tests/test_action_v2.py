"""动作空间 v2：头定义、意图解码、action mask，以及环境下发 intent 帧。

对应规格 §2.2–2.3。这里覆盖三件容易悄悄错的事：

1. **各头取值个数之和必须是 37**（模型维度/`.gmod` 校验都按它）；
2. ``decode_intent`` 的每一档都要映射到文档里的那个值（角度、布尔、槽位）；
3. mask 必须真的按背包/手持物算——空槽的 slot 被屏蔽、空手时 use 只剩「不按」。
"""

from __future__ import annotations

import unittest

import numpy as np

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.items import ItemTable
from gladiator_env.spaces import (
    ACTION_DIM_V2,
    ACTION_HEADS_V2,
    HOTBAR_SLOTS,
    LOOK_PITCH_V2,
    LOOK_YAW_V2,
    MOVE_VALUES_V2,
    MULTI_DISCRETE_V2,
    NUM_ACTIONS,
    action_mask,
    decode_intent,
    head_offsets,
)
from gladiator_env.testing import FakeLink, make_observation, make_target

FULL_ARMOR = {
    "head": {"name": "diamond_helmet"},
    "torso": {"name": "diamond_chestplate"},
    "legs": {"name": "diamond_leggings"},
    "feet": {"name": "diamond_boots"},
}
FULL_INVENTORY = [
    {"slot": 36, "name": "diamond_sword", "count": 1},
    {"slot": 37, "name": "golden_apple", "count": 5},
    {"slot": 38, "name": "fishing_rod", "count": 1},
]


def observation(username: str = "Glad_Bot_1", *, inventory=None, held="diamond_sword") -> dict:
    other = "Glad_Bot_2" if username == "Glad_Bot_1" else "Glad_Bot_1"
    raw = make_observation(
        username, health=20.0,
        targets=[make_target(2 if username == "Glad_Bot_1" else 1, other,
                             x=1.0, z=0.0, distance=2.0)])
    raw["self"]["inventory"] = list(FULL_INVENTORY if inventory is None else inventory)
    raw["self"]["heldItem"] = {"name": held, "count": 1}
    raw["self"]["armor"] = dict(FULL_ARMOR)
    return raw


class HeadsTest(unittest.TestCase):
    def test_head_sizes_match_spec(self):
        self.assertEqual(ACTION_HEADS_V2, {
            "move": 5, "look_yaw": 7, "look_pitch": 5,
            "attack": 2, "use": 3,
            "jump": 2, "sneak": 2, "sprint": 2, "slot": 9,
        })

    def test_total_dim_is_37(self):
        self.assertEqual(ACTION_DIM_V2, 37)
        self.assertEqual(sum(ACTION_HEADS_V2.values()), 37)
        self.assertEqual(sum(MULTI_DISCRETE_V2), 37)
        self.assertEqual(len(MULTI_DISCRETE_V2), len(ACTION_HEADS_V2))

    def test_head_order_and_offsets(self):
        offsets = head_offsets()
        self.assertEqual(list(offsets), list(ACTION_HEADS_V2))
        self.assertEqual(offsets["move"], 0)
        self.assertEqual(offsets["look_yaw"], 5)
        self.assertEqual(offsets["look_pitch"], 12)
        self.assertEqual(offsets["attack"], 17)
        self.assertEqual(offsets["use"], 19)
        self.assertEqual(offsets["jump"], 22)
        self.assertEqual(offsets["sneak"], 24)
        self.assertEqual(offsets["sprint"], 26)
        self.assertEqual(offsets["slot"], 28)
        self.assertEqual(offsets["slot"] + ACTION_HEADS_V2["slot"], ACTION_DIM_V2)

    def test_look_tables(self):
        self.assertEqual(LOOK_YAW_V2, (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0))
        self.assertEqual(LOOK_PITCH_V2, (-15.0, -5.0, 0.0, 5.0, 15.0))
        self.assertEqual(MOVE_VALUES_V2, ("idle", "forward", "back", "left", "right"))


class DecodeIntentTest(unittest.TestCase):
    def test_full_mapping(self):
        intent = decode_intent([1, 6, 4, 1, 2, 0, 1, 1, 8])
        self.assertEqual(intent["move"], "forward")
        self.assertEqual(intent["look"], {"dyaw": 30.0, "dpitch": 15.0})
        self.assertTrue(intent["attack"])
        self.assertEqual(intent["use"], "hold")
        self.assertEqual(intent["slot"], 8)
        self.assertFalse(intent["jump"])
        self.assertTrue(intent["sneak"])
        self.assertTrue(intent["sprint"])

    def test_intent_keys_are_fixed(self):
        self.assertEqual(
            list(decode_intent([0] * 9)),
            ["move", "look", "attack", "use", "slot", "jump", "sneak", "sprint"],
        )

    def test_zero_intent_is_idle(self):
        intent = decode_intent(np.zeros(9, dtype=np.int64))
        self.assertEqual(intent, {
            "move": "idle", "look": {"dyaw": -30.0, "dpitch": -15.0},
            "attack": False, "use": None, "slot": 0,
            "jump": False, "sneak": False, "sprint": False,
        })

    def test_every_head_value_is_reachable(self):
        base = [0] * 9
        heads = list(ACTION_HEADS_V2)
        for index, head in enumerate(heads):
            for value in range(ACTION_HEADS_V2[head]):
                raw = list(base)
                raw[index] = value
                intent = decode_intent(raw)
                if head == "move":
                    self.assertEqual(intent["move"], MOVE_VALUES_V2[value])
                elif head == "look_yaw":
                    self.assertEqual(intent["look"]["dyaw"], LOOK_YAW_V2[value])
                elif head == "look_pitch":
                    self.assertEqual(intent["look"]["dpitch"], LOOK_PITCH_V2[value])
                elif head == "attack":
                    self.assertEqual(intent["attack"], bool(value))
                elif head == "use":
                    self.assertEqual(intent["use"], (None, "click", "hold")[value])
                elif head == "slot":
                    self.assertEqual(intent["slot"], value)
                else:
                    self.assertEqual(intent[head], bool(value))

    def test_one_hot_expansion_matches_head_list(self):
        # 37 维展开 = 每个头一段 one-hot，必须与 9 个头的写法指向同一个意图
        raw = [2, 5, 3, 1, 1, 0, 1, 1, 4]
        offsets = head_offsets()
        flat = [0] * ACTION_DIM_V2
        for index, head in enumerate(ACTION_HEADS_V2):
            flat[offsets[head] + raw[index]] = 1
        self.assertEqual(decode_intent(flat), decode_intent(raw))

    def test_flat_int_is_a_single_hot_bit(self):
        offsets = head_offsets()
        intent = decode_intent(offsets["slot"] + 3)
        self.assertEqual(intent["slot"], 3)
        self.assertEqual(intent["move"], "idle")
        self.assertEqual(intent["look"]["dyaw"], LOOK_YAW_V2[0])

    def test_head_out_of_range_raises(self):
        bad = [0] * 9
        bad[0] = ACTION_HEADS_V2["move"]
        with self.assertRaises(ValueError) as ctx:
            decode_intent(bad)
        self.assertIn("move", str(ctx.exception))

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            decode_intent([0] * 8)

    def test_flat_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            decode_intent(ACTION_DIM_V2)
        with self.assertRaises(ValueError):
            decode_intent(-1)


class ActionMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = ItemTable.load()

    def test_mask_shape(self):
        mask = action_mask(observation(), self.table)
        self.assertEqual(set(mask), set(ACTION_HEADS_V2))
        for head, size in ACTION_HEADS_V2.items():
            self.assertEqual(len(mask[head]), size, head)

    def test_empty_inventory_blocks_every_slot(self):
        mask = action_mask(observation(inventory=[], held="air"), self.table)
        self.assertEqual(mask["slot"], [False] * HOTBAR_SLOTS)

    def test_occupied_hotbar_slots(self):
        mask = action_mask(observation(), self.table)
        self.assertEqual(mask["slot"], [True, True, True] + [False] * 6)

    def test_backpack_items_do_not_unlock_slots(self):
        # 背包（9..35）里的东西要先挪进快捷栏，不能直接把槽位点亮
        mask = action_mask(observation(inventory=[{"slot": 20, "name": "diamond_sword", "count": 1}],
                                      held="air"), self.table)
        self.assertEqual(mask["slot"], [False] * HOTBAR_SLOTS)

    def test_use_follows_held_item(self):
        sword = action_mask(observation(held="diamond_sword"), self.table)
        self.assertEqual(sword["use"], [True, True, True])

        apple = action_mask(observation(held="golden_apple"), self.table)
        self.assertEqual(apple["use"], [True, True, True])

    def test_use_blocked_when_holding_unusable_item(self):
        mask = action_mask(observation(held="mystery_widget"), self.table)
        self.assertEqual(mask["use"], [True, False, False])

    def test_use_blocked_when_empty_hand(self):
        mask = action_mask(observation(held="air"), self.table)
        self.assertEqual(mask["use"], [True, False, False])
        self.assertEqual(len(mask["use"]), 3)

    def test_other_heads_always_available(self):
        mask = action_mask({}, self.table)
        for head in ("move", "look_yaw", "look_pitch", "attack", "jump", "sneak", "sprint"):
            self.assertTrue(all(mask[head]), head)

    def test_unknown_observation_does_not_crash(self):
        mask = action_mask({"self": {"inventory": "not-a-list"}}, self.table)
        self.assertEqual(mask["slot"], [False] * HOTBAR_SLOTS)


def build_env(**kwargs) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1", observation("Glad_Bot_1")),
             FakeLink("Glad_Bot_2", observation("Glad_Bot_2"))]
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        arena=ArenaConfig(give_kit=False),
        **kwargs,
    )
    env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]
    return env, links


class EnvV2Test(unittest.TestCase):
    def test_default_is_v1(self):
        env, _links = build_env()
        self.assertEqual(env.action_space_name, "v1")
        self.assertEqual(env.action_space["bot_0"].n, NUM_ACTIONS)
        env.close()

    def test_v2_action_space_is_multidiscrete(self):
        env, _links = build_env(single_agent=True, action_space="v2")
        self.assertEqual(tuple(env.action_space.nvec), MULTI_DISCRETE_V2)
        sample = env.action_space.sample()
        self.assertEqual(sample.shape, (len(ACTION_HEADS_V2),))
        env.close()

    def test_unknown_action_space_raises(self):
        with self.assertRaises(ValueError):
            build_env(action_space="v3")

    def test_step_sends_intent_frame(self):
        env, links = build_env(action_space="v2")
        intent_raw = [1, 6, 3, 1, 0, 0, 0, 1, 2]
        links[0].actions.clear()
        links[1].actions.clear()

        env.step({"bot_0": np.asarray(intent_raw, dtype=np.int64),
                  "bot_1": np.zeros(9, dtype=np.int64)})

        frames = [action for action in links[0].actions if action.get("type") == "intent"]
        self.assertEqual(len(frames), 1, links[0].actions)
        intent = frames[0]["intent"]
        self.assertEqual(intent["move"], "forward")
        self.assertEqual(intent["look"], {"dyaw": 30.0, "dpitch": 5.0})
        self.assertTrue(intent["attack"])
        self.assertIsNone(intent["use"])
        self.assertEqual(intent["slot"], 2)
        self.assertTrue(intent["sprint"])
        # v1 的宏动作不该再出现在 v2 路径上
        self.assertNotIn("move", [action.get("type") for action in links[0].actions])
        env.close()

    def test_info_carries_action_mask(self):
        env, _links = build_env(single_agent=True, action_space="v2")
        _observation, info = env.reset()
        self.assertIn("actionMask", info)
        mask = info["actionMask"]
        self.assertEqual(set(mask), set(ACTION_HEADS_V2))
        self.assertEqual(mask["slot"], [True, True, True] + [False] * 6)

        _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(9, dtype=np.int64))
        self.assertIn("actionMask", info)
        self.assertEqual(info["actionMask"]["use"], [True, True, True])
        env.close()

    def test_multi_agent_mask_is_per_bot(self):
        env, _links = build_env(action_space="v2")
        _obs, _rewards, _terminated, _truncated, info = env.step({
            "bot_0": np.zeros(9, dtype=np.int64),
            "bot_1": np.zeros(9, dtype=np.int64),
        })
        self.assertEqual(set(info["actionMask"]), {"bot_0", "bot_1"})
        self.assertIn("slot", info["actionMask"]["bot_0"])
        env.close()

    def test_restock_strips_attack_and_use_but_keeps_move(self):
        env, links = build_env(action_space="v2")
        env._restock_pending = {"Glad_Bot_1": 3}
        env._restock_tick = {"Glad_Bot_1": 0}
        env._restock_logged = set()
        links[0].actions.clear()

        intent_raw = [1, 6, 0, 1, 2, 0, 0, 0, 0]  # 前进 + 攻击 + 按住右键
        env.step({"bot_0": np.asarray(intent_raw, dtype=np.int64),
                  "bot_1": np.zeros(9, dtype=np.int64)})

        frames = [action for action in links[0].actions if action.get("type") == "intent"]
        self.assertTrue(frames, links[0].actions)
        intent = frames[-1]["intent"]
        self.assertFalse(intent["attack"])
        self.assertIsNone(intent["use"])
        self.assertEqual(intent["move"], "forward")
        self.assertFalse(env._has_attack([frames[-1]]))
        env.close()

    def test_v1_path_unchanged(self):
        env, links = build_env()
        links[0].actions.clear()
        env.step({"bot_0": 1, "bot_1": 0})   # 1 = approach
        kinds = [action.get("type") for action in links[0].actions]
        self.assertIn("move", kinds)
        self.assertNotIn("intent", kinds)
        env.close()


if __name__ == "__main__":
    unittest.main()
