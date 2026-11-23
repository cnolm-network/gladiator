"""G-711：观测 25 → 37 维（补跨帧信息：上一帧十头动作 + 两个帧计数）。

为什么要这 12 维（背景，写在这里以免后人以为是「凑维度」）：BC 之后 v2 的 ``move`` /
``sprint`` 两个头一致率只有 0.85 / 0.87，而 ``attack`` / ``use`` / ``slot`` 是 0.99+。
差别不在「学得够不够」——单帧观测里根本没有「上一帧往哪走」的信息，而这两个头的
正确答案本质上是**节拍**（横移换向、命中后重置疾跑）。所以补的不是状态量，是时间。

判据（与 G-702 的 23→25 同一套）：

1. **只在末尾追加**：索引 0–24 的字段名与顺序一个都不许动；
2. **归一化正确**：``prev_action`` 每维是 ``idx / head_size``（0..1），帧计数是
   ``frames / 50``（clip 0..1）；
3. **旧权重兼容**：25 维 npz 经 ``adopt_weights`` 装进 37 维策略后，``W1[:, :25]``
   与原来**逐位相等**，右侧 12 列为 0；
4. **两侧逐项一致**：``bot-runtime/src/features.js::encodeObservation`` 与
   ``spaces.encode_observation`` 对同一帧（含跨帧三项）的编码差 < 1e-6。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gladiator_env.spaces import (
    ACTION_HEADS_V2,
    FRAMES_NORMALIZER,
    FRAMES_SINCE_ATTACK_INDEX,
    FRAMES_SINCE_HIT_INDEX,
    OBS_DIM,
    OBS_FIELDS,
    PREV_ACTION_OFFSET,
    encode_observation,
)
from gladiator_env.testing import make_observation, make_target
from gladiator_train.policy import Policy, adopt_weights

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
FEATURES_JS = REPO_ROOT / "bot-runtime" / "src" / "features.js"

#: G-711 之前的 25 个字段（冻结快照）：改任何一个都说明「插到中间」了。
FROZEN_FIELDS_V25 = (
    "self_health",
    "self_food",
    "self_saturation",
    "self_on_ground",
    "self_yaw_sin",
    "self_yaw_cos",
    "self_pitch",
    "target_dx",
    "target_dy",
    "target_dz",
    "target_distance",
    "target_health",
    "target_on_ground",
    "has_target",
    "step_fraction",
    "state_alive",
    "self_held_durability",
    "self_armor_remaining",
    "self_gapples",
    "opponent_speed",
    "opponent_vertical_speed",
    "self_since_attack",
    "in_reach",
    "self_vertical_speed",
    "self_attack_cooldown",
)

#: 追加的 12 个字段（顺序 = ACTION_HEADS_V2 + 两个帧计数）。
APPENDED_FIELDS_V37 = tuple(f"prev_{head}" for head in ACTION_HEADS_V2) + (
    "frames_since_attack",
    "frames_since_hit",
)

#: Node 侧对拍脚本：从 stdin 读 ``{observation, options}``，把 37 维向量打到 stdout。
NODE_CROSSCHECK = """
import { readFileSync } from 'node:fs';
import { encodeObservation } from __FEATURES_URL__;

const payload = JSON.parse(readFileSync(0, 'utf8'));
const vector = encodeObservation(payload.observation, payload.options);
console.log(JSON.stringify(Array.from(vector)));
"""


def rich_observation() -> dict:
    """一帧「字段全都有值」的观测（与 v25 的构造同源，便于逐项对照）。"""
    observation = make_observation(
        "Glad_Bot_1",
        x=0.5,
        y=4.0,
        z=-1.5,
        health=13.0,
        state="Alive",
        targets=[make_target(42, "Glad_Bot_2", x=3.0, y=4.6, z=0.0, distance=3.4)],
    )
    observation["self"].update(
        {
            "yaw": 3.141592653589793 / 3,
            "pitch": -0.35,
            "onGround": False,
            "food": 17,
            "saturation": 3.5,
            "velocity": {"x": 0.12, "y": -0.42, "z": -0.07},
            "heldItem": {
                "name": "diamond_sword",
                "count": 1,
                "slot": 36,
                "durability": {"used": 120, "max": 1561, "remaining": 0.9231},
            },
            "armorRemaining": 0.62,
            "inventory": [
                {"slot": 36, "name": "diamond_sword", "count": 1},
                {"slot": 37, "name": "golden_apple", "count": 3},
                {"slot": 38, "name": "fishing_rod", "count": 1},
            ],
        }
    )
    target = observation["entities"][0]
    target.update(
        {
            "onGround": False,
            "velocity": {"x": -0.9, "y": 1.4, "z": 0.25},
            "heldItem": {"name": "iron_sword", "count": 1},
        }
    )
    return observation


class LayoutV37Test(unittest.TestCase):
    """维度、字段顺序与常量。"""

    def test_dim_is_37(self) -> None:
        self.assertEqual(OBS_DIM, 37)
        self.assertEqual(len(OBS_FIELDS), 37)
        self.assertEqual(OBS_DIM, len(OBS_FIELDS))

    def test_first_25_fields_are_frozen(self) -> None:
        self.assertEqual(tuple(OBS_FIELDS[:25]), FROZEN_FIELDS_V25)

    def test_appended_fields_are_cross_frame(self) -> None:
        self.assertEqual(tuple(OBS_FIELDS[25:]), APPENDED_FIELDS_V37)
        # prev_* 的顺序必须与 ACTION_HEADS_V2 逐位一致（模型按这个顺序挂头）
        self.assertEqual(
            tuple(OBS_FIELDS[25:35]),
            tuple(f"prev_{head}" for head in ACTION_HEADS_V2),
        )

    def test_offsets_match_constants(self) -> None:
        self.assertEqual(PREV_ACTION_OFFSET, 25)
        self.assertEqual(FRAMES_SINCE_ATTACK_INDEX, 35)
        self.assertEqual(FRAMES_SINCE_HIT_INDEX, 36)
        self.assertEqual(FRAMES_NORMALIZER, 50.0)

    def test_vector_length_and_dtype(self) -> None:
        vector = encode_observation(rich_observation(), "Glad_Bot_1")
        self.assertEqual(vector.shape, (37,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vector)))


class CrossFrameEncodingTest(unittest.TestCase):
    """prev_action 归一化 + 帧计数裁剪。"""

    def test_prev_action_normalisation(self) -> None:
        heads = [1, 5, 2, 1, 0, 0, 0, 0, 0, 2]
        vector = encode_observation(
            make_observation("Glad_Bot_1"), "Glad_Bot_1", prev_action=heads)
        for offset, (head, index) in enumerate(zip(ACTION_HEADS_V2, heads)):
            expected = index / float(ACTION_HEADS_V2[head])
            self.assertAlmostEqual(
                float(vector[PREV_ACTION_OFFSET + offset]), expected, places=6,
                msg=f"{head} 头（idx={index}）应归一化为 {expected}")

    def test_prev_action_none_is_all_zero(self) -> None:
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertTrue(np.all(vector[25:35] == 0.0), "没有上一帧动作时 10 维必须全 0")

    def test_prev_action_is_clipped(self) -> None:
        vector = encode_observation(
            make_observation("Glad_Bot_1"), "Glad_Bot_1",
            prev_action=[-1, 99, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(float(vector[25]), 0.0)   # 负下标 → 0
        self.assertEqual(float(vector[26]), 1.0)   # 越界正下标 → 1（clip 上界）

    def test_short_prev_action_leaves_tail_zero(self) -> None:
        vector = encode_observation(
            make_observation("Glad_Bot_1"), "Glad_Bot_1", prev_action=[4])
        self.assertAlmostEqual(float(vector[25]), 4.0 / 5.0, places=6)
        self.assertTrue(np.all(vector[26:35] == 0.0), "缺的头按 0（未知）处理")

    def test_frames_are_normalised_and_clipped(self) -> None:
        cases = {0.0: 0.0, 25.0: 0.5, 50.0: 1.0, 500.0: 1.0, -5.0: 0.0}
        for frames, expected in cases.items():
            vector = encode_observation(
                make_observation("Glad_Bot_1"), "Glad_Bot_1",
                frames_since_attack=frames, frames_since_hit=frames)
            self.assertAlmostEqual(float(vector[FRAMES_SINCE_ATTACK_INDEX]), expected, places=6,
                                   msg=f"{frames} 帧 → {expected}")
            self.assertAlmostEqual(float(vector[FRAMES_SINCE_HIT_INDEX]), expected, places=6)

    def test_frames_default_is_saturated(self) -> None:
        """不传帧计数 = 「很久以前」（1.0），与 ``since_attack`` 缺省 5s 同口径。"""
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertEqual(float(vector[FRAMES_SINCE_ATTACK_INDEX]), 1.0)
        self.assertEqual(float(vector[FRAMES_SINCE_HIT_INDEX]), 1.0)


class OldPolicyV25AdoptionTest(unittest.TestCase):
    """25 维旧模型 → 37 维环境：右侧补零，左侧 25 列逐位不变。"""

    def test_pads_w1_and_keeps_left_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old25.npz"
            old = Policy(obs_dim=25, n_actions=1, hidden=16, seed=11, action_space="v2")
            rng = np.random.default_rng(3)
            old.params["W1"] = rng.normal(0, 0.1, size=(16, 25))
            old.save(str(path))

            loaded = Policy.load(str(path))
            self.assertEqual(loaded.obs_dim, 25)

            fresh = Policy(obs_dim=OBS_DIM, n_actions=1, hidden=16, seed=12, action_space="v2")
            adopt_weights(fresh, loaded)

            self.assertEqual(fresh.obs_dim, 37)
            self.assertEqual(fresh.params["W1"].shape, (16, 37))
            self.assertTrue(np.array_equal(
                fresh.params["W1"][:, :25], np.asarray(old.params["W1"], dtype=np.float64)))
            np.testing.assert_array_equal(fresh.params["W1"][:, 25:], np.zeros((16, 12)))

    def test_new_columns_are_exactly_zero_not_random(self) -> None:
        """补零必须真的是 0：若被随机初始化，「旧模型续训」会在新维上凭空引入噪声。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old25.npz"
            old = Policy(obs_dim=25, n_actions=1, hidden=8, seed=5, action_space="v1")
            old.save(str(path))
            fresh = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=8, seed=6)
            adopt_weights(fresh, Policy.load(str(path)))
            self.assertFalse(np.any(fresh.params["W1"][:, 25:] != 0.0))

    def test_shrinking_is_rejected(self) -> None:
        """37 维模型装进 25 维策略必须报错，而不是悄悄截断（截断会丢新特征）。"""
        big = Policy(obs_dim=37, n_actions=1, hidden=8, seed=1, action_space="v2")
        small = Policy(obs_dim=25, n_actions=1, hidden=8, seed=2, action_space="v2")
        with self.assertRaises(ValueError):
            adopt_weights(small, big)


class EnvCrossFrameStateTest(unittest.TestCase):
    """``GladiatorEnv._update_cross_frame`` 的计数规则（不连服务器，只测这段纯逻辑）。"""

    def _env(self):
        from gladiator_env.env import CROSS_FRAME_CAP, GladiatorEnv

        env = object.__new__(GladiatorEnv)
        env.bots = [SimpleNamespace(username="Bot_A")]
        env._prev_action = {"Bot_A": None}
        env._frames_since_attack = {"Bot_A": 3}
        env._frames_since_hit = {"Bot_A": 7}
        return env, CROSS_FRAME_CAP

    def test_attack_and_damage_reset_counters(self) -> None:
        env, _cap = self._env()
        env._raw = [{"self": {"health": 15.0}}]
        previous = [{"self": {"health": 18.0}}]
        env._update_cross_frame(previous, [[{"type": "attack"}]], {"Bot_A": (1,) * 10})
        self.assertEqual(env._frames_since_attack["Bot_A"], 0, "出手当帧必须清零")
        self.assertEqual(env._frames_since_hit["Bot_A"], 0, "掉血当帧必须清零")
        self.assertEqual(env._prev_action["Bot_A"], (1,) * 10)

    def test_quiet_frames_increment_and_saturate(self) -> None:
        env, cap = self._env()
        env._raw = [{"self": {"health": 20.0}}]
        previous = [{"self": {"health": 20.0}}]
        env._update_cross_frame(previous, [[{"type": "idle"}]], {"Bot_A": None})
        self.assertEqual(env._frames_since_attack["Bot_A"], 4)
        self.assertEqual(env._frames_since_hit["Bot_A"], 8)
        self.assertIsNone(env._prev_action["Bot_A"], "v1 宏动作没有十头标签")

        env._frames_since_attack["Bot_A"] = cap
        env._frames_since_hit["Bot_A"] = cap
        env._update_cross_frame(previous, [[{"type": "idle"}]], {"Bot_A": None})
        self.assertEqual(env._frames_since_attack["Bot_A"], cap, "计数器必须在 cap 饱和")
        self.assertEqual(env._frames_since_hit["Bot_A"], cap)

    def test_v2_intent_attack_counts_as_attack(self) -> None:
        """v2 的 attack 藏在 intent 里，``_has_attack`` 必须认出来（否则帧计数永远不清零）。"""
        env, _cap = self._env()
        env._raw = [{"self": {"health": 20.0}}]
        previous = [{"self": {"health": 20.0}}]
        dispatch = [[{"type": "intent", "intent": {"move": "forward", "attack": True}}]]
        env._update_cross_frame(previous, dispatch, {"Bot_A": (1, 3, 2, 1, 0, 0, 0, 0, 0, 0)})
        self.assertEqual(env._frames_since_attack["Bot_A"], 0)


class NodePythonCrossCheckTest(unittest.TestCase):
    """`features.js::encodeObservation` 与 `spaces.encode_observation` 对拍（37 维）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("本机没有 node，跳过 JS/Python 对拍")
        if not FEATURES_JS.is_file():
            raise unittest.SkipTest(f"找不到 {FEATURES_JS}")

    def _encode_in_node(self, observation: dict, options: dict) -> np.ndarray:
        script = NODE_CROSSCHECK.replace("__FEATURES_URL__", json.dumps(FEATURES_JS.as_uri()))
        completed = subprocess.run(  # noqa: S603 - 固定 node + 本仓库源码
            [self.node, "--input-type=module", "-e", script],
            input=json.dumps({"observation": observation, "options": options}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        if completed.returncode != 0:
            self.fail(f"node 编码失败: {completed.stderr.strip() or completed.stdout.strip()}")
        return np.asarray(json.loads(completed.stdout.strip()), dtype=np.float64)

    def _assert_match(self, observation: dict, **kwargs) -> float:
        username = kwargs.pop("username", "Glad_Bot_1")
        options = {
            "username": username,
            "stepFraction": kwargs.get("step_fraction", 0.0),
            "opponentHealth": kwargs.get("opponent_health"),
            "allowed": sorted(kwargs["opponents"]) if kwargs.get("opponents") else None,
            "reach": kwargs.get("reach", 3.99),
            "sinceAttack": kwargs.get("since_attack", 5.0),
            "selfVelocityY": kwargs.get("self_velocity_y"),
            "attackCooldownRemaining": kwargs.get("attack_cooldown_remaining"),
            "prevAction": kwargs.get("prev_action"),
            "framesSinceAttack": kwargs.get("frames_since_attack"),
            "framesSinceHit": kwargs.get("frames_since_hit"),
        }
        js = self._encode_in_node(observation, options)
        py = encode_observation(observation, username, **kwargs).astype(np.float64)
        self.assertEqual(js.shape, (OBS_DIM,))
        max_diff = float(np.max(np.abs(js - py)))
        self.assertLess(max_diff, 1e-6, f"最大偏差 {max_diff}")
        return max_diff

    def test_rich_frame_with_cross_frame_matches(self) -> None:
        self._assert_match(
            rich_observation(),
            step_fraction=0.37,
            opponent_health=9.0,
            opponents={"Glad_Bot_2"},
            reach=3.99,
            since_attack=0.3,
            attack_cooldown_remaining=0.2,
            prev_action=[1, 5, 2, 1, 0, 0, 0, 0, 0, 2],
            frames_since_attack=0,
            frames_since_hit=17,
        )

    def test_default_cross_frame_matches(self) -> None:
        """两侧都不传跨帧参数时，必须都落成「10 维 0 + 两个 1.0」。"""
        self._assert_match({}, step_fraction=1.0)

    def test_extreme_cross_frame_matches(self) -> None:
        self._assert_match(
            rich_observation(),
            prev_action=[-3, 99, 0, 0, 0, 0, 0, 0, 0, 0],
            frames_since_attack=999.0,
            frames_since_hit=-4.0,
        )

    def test_cross_frame_inside_observation_matches(self) -> None:
        """跨帧三项也可以挂在帧上（``observation.crossFrame``），JS 侧缺省读取。"""
        observation = rich_observation()
        observation["crossFrame"] = {
            "prevAction": [4, 0, 0, 0, 0, 0, 0, 0, 8, 0],
            "framesSinceAttack": 10,
            "framesSinceHit": 20,
        }
        # Python 侧没有「帧里读」的约定：显式把同样三个值传进去，验证两边算出同一向量。
        self._assert_match(
            observation,
            prev_action=[4, 0, 0, 0, 0, 0, 0, 0, 8, 0],
            frames_since_attack=10,
            frames_since_hit=20,
        )


if __name__ == "__main__":
    unittest.main()
