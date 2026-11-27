"""G-709 收口：`opponent_vertical_speed` 已在索引 20，不再扩到 38 维。

任务书写于「观测还是 25 维、没有对手 vy」的旧语境；实际该字段在 P1 就随
``in_reach`` 一起进了观测（见 ``git log -S opponent_vertical_speed``），
G-702/G-711 都只是**在末尾**加字段，没有挪动它。2026-11-22 与用户确认后按
「已满足、不扩维」关闭 G-709：本测试把这条结论钉住，防止后人重复追加一个
重名字段（会让 ``OBS_FIELDS.index()`` 与蒸馏标签的取位产生歧义）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from gladiator_env.spaces import OBS_DIM, OBS_FIELDS, encode_observation
from gladiator_env.testing import make_observation, make_target

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
FEATURES_JS = REPO_ROOT / "bot-runtime" / "src" / "features.js"

OPP_VY_FIELD = "opponent_vertical_speed"
OPP_VY_INDEX = 20

NODE_CROSSCHECK = """
import { readFileSync } from 'node:fs';
import { OBS_FIELDS_JS, encodeObservation } from __FEATURES_URL__;

const payload = JSON.parse(readFileSync(0, 'utf8'));
const vector = encodeObservation(payload.observation, payload.options);
console.log(JSON.stringify({
  fields: OBS_FIELDS_JS,
  vector: Array.from(vector),
}));
"""


def frame_with_opponent_vy(vy: float) -> dict:
    target = make_target(91, "Glad_Bot_2", x=2.6, z=0.0, distance=2.6)
    target["velocity"] = {"x": 0.0, "y": vy, "z": 0.0}
    return make_observation("Glad_Bot_1", targets=[target])


class G709LayoutTest(unittest.TestCase):
    """维度与字段位置：G-709 是「已满足」，不是「加一维」。"""

    def test_obs_dim_matches_fields(self) -> None:
        # 具体维数由后续里程碑决定（G-731 起是 39）；本用例只钉「维数与字段表同长」，
        # 以及「opponent_vertical_speed 没有被追加第二次」。
        self.assertEqual(len(OBS_FIELDS), OBS_DIM)
        self.assertGreaterEqual(OBS_DIM, 37)

    def test_opponent_vy_is_at_frozen_index_20(self) -> None:
        self.assertEqual(OBS_FIELDS[OPP_VY_INDEX], OPP_VY_FIELD)
        self.assertEqual(OBS_FIELDS.index(OPP_VY_FIELD), OPP_VY_INDEX)

    def test_no_duplicate_field_name(self) -> None:
        """若真按字面再追加一个同名字段，这里会立刻失败。"""
        self.assertEqual(len(OBS_FIELDS), len(set(OBS_FIELDS)))

    def test_index_37_does_not_exist(self) -> None:
        self.assertLess(OPP_VY_INDEX + 1, OBS_DIM)
        self.assertTrue(all(name != OPP_VY_FIELD for name in OBS_FIELDS[37:]))

    def test_python_encoding_uses_index_20(self) -> None:
        for vy, expected in ((2.5, 0.5), (-5.0, -1.0), (99.0, 1.0)):
            vector = encode_observation(frame_with_opponent_vy(vy), "Glad_Bot_1")
            self.assertEqual(vector.shape, (OBS_DIM,))
            self.assertAlmostEqual(float(vector[OPP_VY_INDEX]), expected, places=6,
                                   msg=f"vy={vy} 应编码为 {expected}")


class G709JsParityTest(unittest.TestCase):
    """`features.js` 也必须在索引 20，不能只改 Python 侧。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("本机没有 node，跳过 JS/Python 对拍")
        if not FEATURES_JS.is_file():
            raise unittest.SkipTest(f"找不到 {FEATURES_JS}")

    def _encode_in_node(self, observation: dict) -> dict:
        script = NODE_CROSSCHECK.replace("__FEATURES_URL__", json.dumps(FEATURES_JS.as_uri()))
        completed = subprocess.run(  # noqa: S603 - 固定 node + 本仓库源码
            [self.node, "--input-type=module", "-e", script],
            input=json.dumps({
                "observation": observation,
                "options": {"username": "Glad_Bot_1", "reach": 3.0},
            }),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        if completed.returncode != 0:
            self.fail(f"node 编码失败: {completed.stderr.strip() or completed.stdout.strip()}")
        return json.loads(completed.stdout.strip())

    def test_js_field_and_vector_match_python(self) -> None:
        observation = frame_with_opponent_vy(-3.5)
        js = self._encode_in_node(observation)
        self.assertEqual(len(js["fields"]), OBS_DIM)
        self.assertEqual(js["fields"][OPP_VY_INDEX], OPP_VY_FIELD)
        self.assertEqual(len(js["fields"]), len(set(js["fields"])))
        self.assertEqual(len(js["vector"]), OBS_DIM)

        py = encode_observation(observation, "Glad_Bot_1", reach=3.0).astype(np.float64)
        js_vector = np.asarray(js["vector"], dtype=np.float64)
        self.assertLess(float(np.max(np.abs(js_vector - py))), 1e-6)
        self.assertAlmostEqual(float(js_vector[OPP_VY_INDEX]), -0.7, places=6)


if __name__ == "__main__":
    unittest.main()
