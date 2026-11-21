"""动作空间 v2：头定义、意图解码、action mask，以及环境下发 intent 帧。

对应规格 §2.2–2.3。这里覆盖三件容易悄悄错的事：

1. **各头取值个数之和必须是 40**（模型维度/`.gmod` 校验都按它；G-707 追加
   `sprint_reset` 后从 9 头/37 维变成 10 头/40 维）；
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
            "sprint_reset": 3,
        })

    def test_total_dim_is_40(self):
        self.assertEqual(ACTION_DIM_V2, 40)
        self.assertEqual(sum(ACTION_HEADS_V2.values()), 40)
        self.assertEqual(sum(MULTI_DISCRETE_V2), 40)
        self.assertEqual(len(MULTI_DISCRETE_V2), len(ACTION_HEADS_V2))
        self.assertEqual(list(MULTI_DISCRETE_V2),
                         [5, 7, 5, 2, 3, 2, 2, 2, 9, 3])

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
        # G-707：新头追加在末尾 → slot 的下标仍是 28（旧日志/探针不会错位）
        self.assertEqual(offsets["sprint_reset"], 37)
        self.assertEqual(offsets["sprint_reset"] + ACTION_HEADS_V2["sprint_reset"],
                         ACTION_DIM_V2)

    def test_look_tables(self):
        self.assertEqual(LOOK_YAW_V2, (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0))
        self.assertEqual(LOOK_PITCH_V2, (-15.0, -5.0, 0.0, 5.0, 15.0))
        self.assertEqual(MOVE_VALUES_V2, ("idle", "forward", "back", "left", "right"))


class DecodeIntentTest(unittest.TestCase):
    def test_full_mapping(self):
        intent = decode_intent([1, 6, 4, 1, 2, 0, 1, 1, 8, 1])
        self.assertEqual(intent["move"], "forward")
        self.assertEqual(intent["look"], {"dyaw": 30.0, "dpitch": 15.0})
        self.assertTrue(intent["attack"])
        self.assertEqual(intent["use"], "hold")
        self.assertEqual(intent["slot"], 8)
        self.assertFalse(intent["jump"])
        self.assertTrue(intent["sneak"])
        self.assertTrue(intent["sprint"])
        self.assertEqual(intent["sprintReset"], "release_w")

    def test_sprint_reset_maps_to_three_values(self):
        for raw, want in ((0, None), (1, "release_w"), (2, "block_hit")):
            intent = decode_intent([0] * 9 + [raw])
            if want is None:
                # none 时**不写**这个键：老 intent 形状必须逐字节一致
                self.assertNotIn("sprintReset", intent)
            else:
                self.assertEqual(intent["sprintReset"], want)

    def test_intent_keys_are_fixed(self):
        self.assertEqual(
            list(decode_intent([0] * 10)),
            ["move", "look", "attack", "use", "slot", "jump", "sneak", "sprint"],
        )
        self.assertEqual(
            list(decode_intent([0] * 9 + [2])),
            ["move", "look", "attack", "use", "slot", "jump", "sneak", "sprint",
             "sprintReset"],
        )

    def test_zero_intent_is_idle(self):
        intent = decode_intent(np.zeros(10, dtype=np.int64))
        self.assertEqual(intent, {
            "move": "idle", "look": {"dyaw": -30.0, "dpitch": -15.0},
            "attack": False, "use": None, "slot": 0,
            "jump": False, "sneak": False, "sprint": False,
        })

    def test_every_head_value_is_reachable(self):
        base = [0] * 10
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
                elif head == "sprint_reset":
                    want = ("none", "release_w", "block_hit")[value]
                    if want == "none":
                        self.assertNotIn("sprintReset", intent)
                    else:
                        self.assertEqual(intent["sprintReset"], want)
                else:
                    self.assertEqual(intent[head], bool(value))

    def test_one_hot_expansion_matches_head_list(self):
        # 40 维展开 = 每个头一段 one-hot，必须与 10 个头的写法指向同一个意图
        raw = [2, 5, 3, 1, 1, 0, 1, 1, 4, 2]
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
        bad = [0] * 10
        bad[0] = ACTION_HEADS_V2["move"]
        with self.assertRaises(ValueError) as ctx:
            decode_intent(bad)
        self.assertIn("move", str(ctx.exception))

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            decode_intent([0] * 8)
        with self.assertRaises(ValueError):
            decode_intent([0] * 9)

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
        for head in ("move", "look_yaw", "look_pitch", "attack", "jump", "sneak", "sprint",
                     "sprint_reset"):
            self.assertTrue(all(mask[head]), head)

    def test_sprint_reset_has_no_item_condition(self):
        """G-707：空手/空背包时 sprint_reset 仍然全开（release_w 不需要物品）。"""
        empty = action_mask(observation(inventory=[], held="air"), self.table)
        self.assertEqual(empty["sprint_reset"], [True, True, True])
        self.assertEqual(empty["use"], [True, False, False])

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
        intent_raw = [1, 6, 3, 1, 0, 0, 0, 1, 2, 0]
        links[0].actions.clear()
        links[1].actions.clear()

        env.step({"bot_0": np.asarray(intent_raw, dtype=np.int64),
                  "bot_1": np.zeros(10, dtype=np.int64)})

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

        _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(10, dtype=np.int64))
        self.assertIn("actionMask", info)
        self.assertEqual(info["actionMask"]["use"], [True, True, True])
        env.close()

    def test_multi_agent_mask_is_per_bot(self):
        env, _links = build_env(action_space="v2")
        _obs, _rewards, _terminated, _truncated, info = env.step({
            "bot_0": np.zeros(10, dtype=np.int64),
            "bot_1": np.zeros(10, dtype=np.int64),
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

        intent_raw = [1, 6, 0, 1, 2, 0, 0, 0, 0, 0]  # 前进 + 攻击 + 按住右键
        env.step({"bot_0": np.asarray(intent_raw, dtype=np.int64),
                  "bot_1": np.zeros(10, dtype=np.int64)})

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


class V2OpponentTest(unittest.TestCase):
    """G-707：单智能体 + **v2 对手策略**（teacher v4）也要走意图路径。

    学习方仍是 v1（``action_space="v1"``，例如 dagger-1118b），对手却是 10 头意图策略。
    env 通过 ``opponent_policy.is_v2`` 识别，把对手的动作交给 ``decode_intent``，
    否则 ``int(10 头数组)`` 会直接抛错——这条路径以前不存在，加头之后必须钉住。
    """

    class _ArrayPolicy:
        """最小可用的 v2 对手：永远返回同一个 10 头动作。"""

        is_v2 = True
        action_space = "v2"

        def __init__(self, action):
            self.action = list(action)
            self.calls = 0

        def act(self, vector, rng=None, greedy=True):
            self.calls += 1
            return list(self.action)

    def build(self, action):
        opponent = self._ArrayPolicy(action)
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
            single_agent=True,
            action_space="v1",
            opponent_policy=opponent,
        )
        env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]
        return env, links, opponent

    def test_v2_opponent_dispatches_intent_frames(self) -> None:
        # 前进 + 攻击 + release_w：三个头都要在对手的载荷里出现
        enemy_action = [1, 3, 2, 1, 0, 0, 0, 1, 0, 1]
        env, links, opponent = self.build(enemy_action)
        links[1].actions.clear()

        env.step(1)                          # 学习方 v1：1 = approach

        intents = [action for action in links[1].actions if action.get("type") == "intent"]
        self.assertEqual(len(intents), 1, links[1].actions)
        intent = intents[0]["intent"]
        self.assertEqual(intent["move"], "forward")
        self.assertTrue(intent["attack"])
        self.assertEqual(intent["sprintReset"], "release_w")
        self.assertGreaterEqual(opponent.calls, 1)
        # 学习方仍是 v1 宏动作（身份不能串）
        kinds = [action.get("type") for action in links[0].actions]
        self.assertNotIn("intent", kinds)
        env.close()

    def test_v2_opponent_block_hit(self) -> None:
        env, links, _opponent = self.build([1, 3, 2, 1, 0, 0, 0, 0, 0, 2])
        links[1].actions.clear()
        env.step(1)
        intent = [a for a in links[1].actions if a.get("type") == "intent"][0]["intent"]
        self.assertEqual(intent["sprintReset"], "block_hit")
        env.close()


if __name__ == "__main__":
    unittest.main()
