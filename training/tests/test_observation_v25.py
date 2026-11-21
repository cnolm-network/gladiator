"""G-702：观测 23 → 25 维（`self_vertical_speed` / `self_attack_cooldown`）。

三件事必须同时成立，否则「旧模型右侧补零续训」会在无人察觉的情况下失效：

1. **只在末尾追加**：索引 0–22 的名字、顺序、语义一个都不许动；
2. **旧权重兼容**：23 维 npz 经 ``adopt_weights`` 装进 25 维策略后，
   ``W1[:, :23]`` 与原来**逐位相等**，右侧两列为 0；
3. **两侧逐项一致**：``bot-runtime/src/features.js::encodeObservation``
   与 ``spaces.encode_observation`` 对同一帧的编码差 < 1e-6。

对拍用真实的 Node 进程（不重写一遍 JS 逻辑），这样任何一侧单独改动都会被抓住。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gladiator_env.spaces import (
    ATTACK_COOLDOWN_S,
    OBS_CLIP,
    OBS_DIM,
    OBS_FIELDS,
    encode_observation,
)
from gladiator_env.testing import make_observation, make_target
from gladiator_train.policy import Policy, adopt_weights

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
FEATURES_JS = REPO_ROOT / "bot-runtime" / "src" / "features.js"

#: G-702 之前的 23 个字段（冻结快照）：改任何一个都说明「插入中间」了。
FROZEN_FIELDS_V23 = (
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
)

#: Node 侧对拍脚本：从 stdin 读 ``{observation, options}``，把 25 维向量打到 stdout。
#: 用 ``__FEATURES_URL__`` 占位（不用 ``str.format``，避免和 JS 的 ``{}`` 解构冲突）。
NODE_CROSSCHECK = """
import { readFileSync } from 'node:fs';
import { encodeObservation } from __FEATURES_URL__;

const payload = JSON.parse(readFileSync(0, 'utf8'));
const vector = encodeObservation(payload.observation, payload.options);
console.log(JSON.stringify(Array.from(vector)));
"""


def rich_observation() -> dict:
    """一帧「字段全都有值」的观测：目标、速度、耐久、背包、护甲都非默认。"""
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
            "yaw": math_pi_over(3),
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


def math_pi_over(divisor: int) -> float:
    return 3.141592653589793 / divisor


class ObservationV25LayoutTest(unittest.TestCase):
    """维度、字段顺序与裁剪边界。"""

    def test_dim_is_25(self) -> None:
        self.assertEqual(OBS_DIM, 25)
        self.assertEqual(len(OBS_FIELDS), 25)

    def test_first_23_fields_are_untouched(self) -> None:
        self.assertEqual(tuple(OBS_FIELDS[:23]), FROZEN_FIELDS_V23)

    def test_appended_fields(self) -> None:
        self.assertEqual(OBS_FIELDS[23], "self_vertical_speed")
        self.assertEqual(OBS_FIELDS[24], "self_attack_cooldown")

    def test_attack_cooldown_constant(self) -> None:
        self.assertEqual(ATTACK_COOLDOWN_S, 0.5)

    def test_vector_length_and_dtype(self) -> None:
        vector = encode_observation(rich_observation(), "Glad_Bot_1")
        self.assertEqual(vector.shape, (25,))
        self.assertEqual(vector.dtype, np.float32)

    def test_vertical_speed_comes_from_self_payload(self) -> None:
        vector = encode_observation(rich_observation(), "Glad_Bot_1")
        self.assertAlmostEqual(float(vector[23]), -0.42 / 5.0, places=6)

    def test_vertical_speed_explicit_argument_wins(self) -> None:
        vector = encode_observation(rich_observation(), "Glad_Bot_1", self_velocity_y=2.5)
        self.assertAlmostEqual(float(vector[23]), 0.5, places=6)

    def test_vertical_speed_clip(self) -> None:
        high = encode_observation({}, "Glad_Bot_1", self_velocity_y=40.0)
        low = encode_observation({}, "Glad_Bot_1", self_velocity_y=-40.0)
        self.assertAlmostEqual(float(high[23]), 1.0, places=6)
        self.assertAlmostEqual(float(low[23]), -1.0, places=6)
        # 末尾统一 np.clip(-5, 5) 之后仍在 [-1, 1]
        self.assertLessEqual(abs(float(low[23])), OBS_CLIP)

    def test_missing_vertical_speed_is_zero(self) -> None:
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertEqual(float(vector[23]), 0.0)
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_attack_cooldown_normalisation(self) -> None:
        cases = {0.0: 0.0, 0.25: 0.5, 0.5: 1.0, 0.9: 1.0, -0.2: 0.0}
        for seconds, expected in cases.items():
            vector = encode_observation(
                make_observation("Glad_Bot_1"), "Glad_Bot_1", attack_cooldown_remaining=seconds)
            self.assertAlmostEqual(float(vector[24]), expected, places=6,
                                   msg=f"冷却 {seconds}s 应归一化为 {expected}")

    def test_attack_cooldown_default_is_ready(self) -> None:
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertEqual(float(vector[24]), 0.0)

    def test_nan_vertical_speed_is_replaced(self) -> None:
        observation = rich_observation()
        observation["self"]["velocity"] = {"x": 0.0, "y": float("nan"), "z": 0.0}
        vector = encode_observation(observation, "Glad_Bot_1")
        self.assertEqual(float(vector[23]), 0.0)
        self.assertTrue(np.all(np.isfinite(vector)))


class OldPolicyAdoptionTest(unittest.TestCase):
    """旧 23 维模型 → 25 维环境：右侧补零，左侧逐位不变。"""

    def _old_policy(self, path: Path) -> Policy:
        policy = Policy(obs_dim=23, n_actions=1, hidden=16, seed=11, action_space="v2")
        # 让 W1 有可识别的值，便于逐位比对
        rng = np.random.default_rng(3)
        policy.params["W1"] = rng.normal(0, 0.1, size=(16, 23))
        policy.save(str(path))
        return policy

    def test_pads_w1_columns_and_keeps_left_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old23.npz"
            old = self._old_policy(path)
            loaded = Policy.load(str(path))
            self.assertEqual(loaded.obs_dim, 23)

            fresh = Policy(obs_dim=OBS_DIM, n_actions=1, hidden=16, seed=12, action_space="v2")
            adopt_weights(fresh, loaded)

            self.assertEqual(fresh.obs_dim, 25)
            self.assertEqual(fresh.params["W1"].shape, (16, 25))
            # 关键断言：左侧 23 列逐位相等
            self.assertTrue(np.array_equal(fresh.params["W1"][:, :23],
                                           np.asarray(old.params["W1"], dtype=np.float64)))
            np.testing.assert_array_equal(fresh.params["W1"][:, 23:], np.zeros((16, 2)))

    def test_saving_and_reloading_keeps_obs_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new25.npz"
            policy = Policy(obs_dim=OBS_DIM, n_actions=1, hidden=8, seed=5, action_space="v2")
            policy.save(str(path))
            loaded = Policy.load(str(path))
            self.assertEqual(loaded.obs_dim, 25)
            self.assertEqual(loaded.params["W1"].shape, (8, 25))


class NodePythonCrossCheckTest(unittest.TestCase):
    """`features.js::encodeObservation` 与 `spaces.encode_observation` 对拍。"""

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
        }
        js = self._encode_in_node(observation, options)
        py = encode_observation(observation, username, **kwargs).astype(np.float64)
        self.assertEqual(js.shape, (25,))
        max_diff = float(np.max(np.abs(js - py)))
        self.assertLess(max_diff, 1e-6, f"最大偏差 {max_diff}")
        return max_diff

    def test_rich_frame_matches(self) -> None:
        self._assert_match(
            rich_observation(),
            step_fraction=0.37,
            opponent_health=9.0,
            opponents={"Glad_Bot_2"},
            reach=3.99,
            since_attack=0.3,
            attack_cooldown_remaining=0.2,
        )

    def test_empty_frame_matches(self) -> None:
        self._assert_match({}, step_fraction=1.0)

    def test_extreme_values_match(self) -> None:
        observation = rich_observation()
        observation["self"]["velocity"] = {"x": 99.0, "y": -99.0, "z": 99.0}
        observation["entities"][0]["velocity"] = {"x": -50.0, "y": 50.0, "z": 50.0}
        self._assert_match(
            observation,
            self_velocity_y=-40.0,
            attack_cooldown_remaining=5.0,
            since_attack=99.0,
        )

    def test_self_velocity_fallback_matches(self) -> None:
        """不传 selfVelocityY 时两侧都回落读 ``self.velocity.y``。"""
        self._assert_match(rich_observation(), since_attack=1.1)


if __name__ == "__main__":
    unittest.main()
