"""G-731：观测 37 → 39，末尾追加「朝向误差」两维（`target_bearing_sin/cos`）。

背景（真机 2026-11-28）：心跳里 ``dist=2.8 yaw=-108.1 bearing=-71.9 yawErr=36.3``——
被击退/横移之后朝向被甩开，而 37 维观测里**只有世界坐标差**（7–9 维）没有自身朝向，
模型无法感知 ``yawErr``，于是「两刀之后不主动攻击、朝反方向跑」。

本文件把这条新契约逐条钉住：

- 位置与顺序：37/38 在末尾，0–36 一位不动（旧模型右侧补零即可继续用）；
- 语义：``bearing = atan2(-dx, dz)``（1.8 口径），``yawErr = wrap180(bearing - yaw)``，
  取 ``sin/cos``；
- 无目标 / 无 ``self.yaw`` 时写 ``(0, 0)``（真实值恒满足 sin²+cos²=1，``(0,0)`` 无歧义）；
- Python ↔ JS 逐位对拍（``bot-runtime/src/features.js`` 必须同序同裁剪）。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from gladiator_env.spaces import (
    BEARING_COS_INDEX,
    BEARING_SIN_INDEX,
    OBS_DIM,
    OBS_FIELDS,
    bearing_sin_cos,
    encode_observation,
    self_yaw_of,
    wrap180,
    wrap_pi,
    yaw_error,
)
from gladiator_env.testing import make_observation, make_target

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
FEATURES_JS = REPO_ROOT / "bot-runtime" / "src" / "features.js"

#: 字段表的稳定指纹：顺序一旦被改动（哪怕只是挪一位），这里立刻炸。
OBS_FIELDS_SHA1_16 = "c77cc01255c90495"

NODE_CROSSCHECK = """
import { readFileSync } from 'node:fs';
import { OBS_DIM, OBS_FIELDS_JS, encodeObservation } from __FEATURES_URL__;

const payload = JSON.parse(readFileSync(0, 'utf8'));
const out = payload.observations.map((observation) => {
  const vector = encodeObservation(observation, payload.options);
  return Array.from(vector);
});
console.log(JSON.stringify({ obsDim: OBS_DIM, fields: OBS_FIELDS_JS, vectors: out }));
"""


def obs_with_target(*, yaw: float = 0.0, target_x: float = 0.0, target_z: float = 4.0,
                    has_target: bool = True, drop_yaw: bool = False) -> dict:
    """造一帧观测：自己站在原点，对手在 ``(target_x, 0, target_z)``。"""
    targets = ([make_target(2, "Foe", x=target_x, z=target_z)]
               if has_target else [])
    raw = make_observation("Glad_Bot_1", targets=targets)
    raw["self"]["yaw"] = yaw
    if drop_yaw:
        raw["self"].pop("yaw", None)
    return raw


def bearing_of(raw: dict, index: int) -> float:
    vector = encode_observation(raw, "Glad_Bot_1")
    return float(vector[index])


class LayoutV39Test(unittest.TestCase):
    """维度、字段名、顺序与常量。"""

    def test_dim_is_39(self) -> None:
        self.assertEqual(OBS_DIM, 39)
        self.assertEqual(len(OBS_FIELDS), OBS_DIM)

    def test_bearing_indices_and_names(self) -> None:
        self.assertEqual(BEARING_SIN_INDEX, 37)
        self.assertEqual(BEARING_COS_INDEX, 38)
        self.assertEqual(OBS_FIELDS[BEARING_SIN_INDEX], "target_bearing_sin")
        self.assertEqual(OBS_FIELDS[BEARING_COS_INDEX], "target_bearing_cos")

    def test_field_order_hash_is_stable(self) -> None:
        digest = hashlib.sha1("\n".join(OBS_FIELDS).encode("utf-8")).hexdigest()[:16]
        self.assertEqual(digest, OBS_FIELDS_SHA1_16,
                         "字段顺序变了：新增只能追加在末尾，不许插队/改名")

    def test_prefix_37_fields_are_frozen(self) -> None:
        """G-731 只动 37/38：前 37 位必须与 G-711 的结论一字不差。"""
        self.assertEqual(tuple(OBS_FIELDS[:35])[13], "has_target")
        self.assertEqual(OBS_FIELDS[35], "frames_since_attack")
        self.assertEqual(OBS_FIELDS[36], "frames_since_hit")
        self.assertEqual(len(set(OBS_FIELDS)), OBS_DIM, "字段名不许重复")

    def test_missing_fields_are_zero_not_nan(self) -> None:
        vector = encode_observation(make_observation("Glad_Bot_1"), "Glad_Bot_1")
        self.assertEqual(vector.shape, (OBS_DIM,))
        self.assertTrue(np.all(np.isfinite(vector)))


class BearingEncodingTest(unittest.TestCase):
    """bearing / yawErr 的取值语义。"""

    def test_no_target_is_zero_pair(self) -> None:
        raw = obs_with_target(has_target=False)
        self.assertEqual(bearing_of(raw, BEARING_SIN_INDEX), 0.0)
        self.assertEqual(bearing_of(raw, BEARING_COS_INDEX), 0.0)

    def test_front_target_is_zero_one(self) -> None:
        """yaw=0、对手在 +Z（正前方）→ yawErr=0 → (sin, cos) = (0, 1)。"""
        raw = obs_with_target(yaw=0.0, target_x=0.0, target_z=4.0)
        self.assertAlmostEqual(bearing_of(raw, BEARING_SIN_INDEX), 0.0, places=6)
        self.assertAlmostEqual(bearing_of(raw, BEARING_COS_INDEX), 1.0, places=6)

    def test_right_target_is_minus_one_zero(self) -> None:
        """yaw=0、对手在 +X（正右）→ bearing=−90° → yawErr=−90° → (−1, 0)。"""
        raw = obs_with_target(yaw=0.0, target_x=4.0, target_z=0.0)
        self.assertAlmostEqual(bearing_of(raw, BEARING_SIN_INDEX), -1.0, places=6)
        self.assertAlmostEqual(bearing_of(raw, BEARING_COS_INDEX), 0.0, places=6)

    def test_behind_target_is_minus_one_cos(self) -> None:
        """对手在 −Z（背后）→ yawErr=±180° → cos=−1、sin≈0（sin(−π)=−0）。"""
        raw = obs_with_target(yaw=0.0, target_x=0.0, target_z=-4.0)
        self.assertAlmostEqual(bearing_of(raw, BEARING_SIN_INDEX), 0.0, places=6)
        self.assertAlmostEqual(bearing_of(raw, BEARING_COS_INDEX), -1.0, places=6)

    def test_yaw_offset_rotates_the_pair(self) -> None:
        """自身已朝 +X（yaw=−90°）、对手也在 +X → 正对 → (0, 1)。"""
        raw = obs_with_target(yaw=-math.pi / 2, target_x=4.0, target_z=0.0)
        self.assertAlmostEqual(bearing_of(raw, BEARING_SIN_INDEX), 0.0, places=6)
        self.assertAlmostEqual(bearing_of(raw, BEARING_COS_INDEX), 1.0, places=6)

    def test_pair_is_unit_length(self) -> None:
        """真实值恒满足 sin²+cos²=1：(0,0) 因此可以安全地当「不知道」。"""
        for yaw in (-math.pi, -1.0, 0.0, 0.7, math.pi / 2):
            for (tx, tz) in ((3.0, 1.0), (-2.0, 2.5), (0.5, -3.0)):
                sin_v, cos_v = bearing_sin_cos(tx, tz, yaw)
                self.assertAlmostEqual(sin_v * sin_v + cos_v * cos_v, 1.0, places=6)

    def test_wrap180_crosses_the_seam(self) -> None:
        """跨 ±180 折返：190° → −170°、−190° → 170°、180° → −180。"""
        self.assertAlmostEqual(wrap180(190.0), -170.0, places=6)
        self.assertAlmostEqual(wrap180(-190.0), 170.0, places=6)
        self.assertAlmostEqual(wrap180(180.0), -180.0, places=6)
        self.assertAlmostEqual(wrap180(0.0), 0.0, places=6)
        self.assertAlmostEqual(wrap180(360.0 + 45.0), 45.0, places=6)

    def test_wrap_pi_is_the_radian_form(self) -> None:
        for degrees in (-370.0, -180.0, -20.0, 0.0, 20.0, 180.0, 370.0):
            self.assertAlmostEqual(wrap_pi(math.radians(degrees)), math.radians(wrap180(degrees)),
                                   places=9)

    def test_seam_is_continuous_for_the_pair(self) -> None:
        """跨缝时 sin/cos 不能跳变：179° 与 −179° 的误差只差 2°。"""
        left = bearing_sin_cos(math.sin(math.radians(179.0)) * 4.0,
                               math.cos(math.radians(179.0)) * 4.0, 0.0)
        right = bearing_sin_cos(math.sin(math.radians(-179.0)) * 4.0,
                                math.cos(math.radians(-179.0)) * 4.0, 0.0)
        delta = math.degrees(math.acos(max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1]))))
        self.assertLess(delta, 2.5, f"跨 ±180 时朝向编码跳了 {delta:.1f}°")

    def test_yaw_error_matches_wrap180(self) -> None:
        """``yaw_error`` 与「bearing − yaw 再 wrap」等价（同一口径，别各写一份）。"""
        dx, dz, yaw = 3.0, -1.0, 2.8
        expected = wrap_pi(math.atan2(-dx, dz) - yaw)
        self.assertAlmostEqual(yaw_error(dx, dz, yaw), expected, places=12)


class MissingYawTest(unittest.TestCase):
    """没有 ``self.yaw`` 时不许伪造别的值，只能写 (0,0)。"""

    def test_missing_yaw_leaves_zero_pair(self) -> None:
        raw = obs_with_target(yaw=0.0, target_x=4.0, target_z=0.0, drop_yaw=True)
        self.assertIsNone(self_yaw_of(raw))
        vector = encode_observation(raw, "Glad_Bot_1")
        self.assertEqual(float(vector[BEARING_SIN_INDEX]), 0.0)
        self.assertEqual(float(vector[BEARING_COS_INDEX]), 0.0)
        # 但其它维照常编码：缺 yaw 不能把整帧变成 0。
        self.assertEqual(float(vector[13]), 1.0, "has_target 仍应为 1")

    def test_explicit_self_yaw_overrides(self) -> None:
        """显式传 ``self_yaw`` 时以参数为准（环境就是这么调用的）。"""
        raw = obs_with_target(yaw=0.0, target_x=4.0, target_z=0.0)
        vector = encode_observation(raw, "Glad_Bot_1", self_yaw=-math.pi / 2)
        self.assertAlmostEqual(float(vector[BEARING_SIN_INDEX]), 0.0, places=6)
        self.assertAlmostEqual(float(vector[BEARING_COS_INDEX]), 1.0, places=6)

    def test_self_yaw_of_reads_radians(self) -> None:
        raw = obs_with_target(yaw=1.25)
        self.assertAlmostEqual(self_yaw_of(raw), 1.25, places=12)
        raw["self"]["yaw"] = "not-a-number"
        self.assertIsNone(self_yaw_of(raw))


class EnvYawAvailableTest(unittest.TestCase):
    """环境侧：``_encode_all`` 要把 yaw 传下去，并留 ``yawAvailable`` 诊断。"""

    def test_env_encodes_bearing_and_reports_availability(self) -> None:
        from gladiator_env import BotConfig, GladiatorEnv
        from gladiator_env.testing import FakeLink

        links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
        links[0].set_observation(make_observation(
            "Glad_Bot_1", x=-1.5, z=0.0,
            targets=[make_target(2, "Glad_Bot_2", x=1.5, z=0.0, distance=3.0)]))
        links[1].set_observation(make_observation(
            "Glad_Bot_2", x=1.5, z=0.0,
            targets=[make_target(1, "Glad_Bot_1", x=-1.5, z=0.0, distance=3.0)]))
        env = GladiatorEnv(
            [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
             BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
            links=links, connect=True, control_hz=200.0, action_ticks=1,
        )
        try:
            observation, _info = env.reset()
            self.assertEqual(observation["bot_0"].shape, (OBS_DIM,))
            # yaw=0、对手在 +X → yawErr=−90° → (−1, 0)
            self.assertAlmostEqual(float(observation["bot_0"][BEARING_SIN_INDEX]), -1.0, places=6)
            self.assertAlmostEqual(float(observation["bot_0"][BEARING_COS_INDEX]), 0.0, places=6)
            self.assertEqual(env.yaw_available, {"Glad_Bot_1": True, "Glad_Bot_2": True})
        finally:
            env.close()


class JsParityV39Test(unittest.TestCase):
    """Python ↔ JS 对拍：同序、同公式、同裁剪（maxDiff < 1e-6）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("本机没有 node，跳过 JS/Python 对拍")
        if not FEATURES_JS.is_file():
            raise unittest.SkipTest(f"找不到 {FEATURES_JS}")

    def test_js_fields_and_vectors_match_python(self) -> None:
        cases = [
            obs_with_target(yaw=0.0, target_x=0.0, target_z=4.0),
            obs_with_target(yaw=0.0, target_x=4.0, target_z=0.0),
            obs_with_target(yaw=2.4, target_x=-3.0, target_z=-2.0),
            obs_with_target(yaw=-3.0, target_x=1.5, target_z=2.5),
            obs_with_target(has_target=False),
            obs_with_target(yaw=0.0, target_x=4.0, target_z=0.0, drop_yaw=True),
        ]
        script = NODE_CROSSCHECK.replace("__FEATURES_URL__", json.dumps(FEATURES_JS.as_uri()))
        completed = subprocess.run(  # noqa: S603 - 固定 node + 本仓库源码
            [self.node, "--input-type=module", "-e", script],
            input=json.dumps({
                "observations": cases,
                "options": {"username": "Glad_Bot_1", "reach": 3.0},
            }),
            capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=str(REPO_ROOT),
        )
        if completed.returncode != 0:
            self.fail(f"node 编码失败: {completed.stderr.strip() or completed.stdout.strip()}")
        payload = json.loads(completed.stdout.strip())

        self.assertEqual(payload["obsDim"], OBS_DIM)
        self.assertEqual(tuple(payload["fields"]), OBS_FIELDS)

        for index, raw in enumerate(cases):
            py = encode_observation(raw, "Glad_Bot_1", reach=3.0).astype(np.float64)
            js = np.asarray(payload["vectors"][index], dtype=np.float64)
            self.assertEqual(js.shape, py.shape)
            max_diff = float(np.max(np.abs(js - py)))
            self.assertLess(max_diff, 1e-6, f"第 {index} 帧 Python/JS 不一致（maxDiff={max_diff}）")


if __name__ == "__main__":
    unittest.main()
