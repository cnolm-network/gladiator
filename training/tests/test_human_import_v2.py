"""G-720：真人录像导入的 v2（十头）动作空间。

判据（为什么要有这组测试）：

1. ``macro_to_intent`` 对每个 v1 宏动作都给出**合法**的十头下标（越界会让 BC 直接炸）；
2. 语义别翻反：``attack`` 必须抬手、``eat_gapple``/``block`` 必须是 ``use=hold``、
   ``jump`` 必须跳；
3. ``build_dataset(action_space="v2")`` 产出 ``(N, 10)``，且 ``macroActions`` 保留原始
   宏动作（报告里的分布仍按宏动作统计，不受翻译影响）；
4. ``run_bc`` 能把 v2 数据集喂给 v2 策略（此前会抛「第 0 帧 v2 动作应有 10 个头」）；
5. 起点是 v2 但数据集是 v1 时必须**明确跳过**而不是瞎训。

运行：
    cd D:\\cbot-gladiator\\training
    .venv\\Scripts\\python.exe -m unittest tests.test_human_import_v2 -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
SCRIPTS_DIR = TRAINING_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gladiator_env.spaces import (  # noqa: E402
    ACTION_HEADS_V2,
    ACTION_NAMES,
    MULTI_DISCRETE_V2,
    decode_intent,
    macro_to_intent,
    macro_to_intent_array,
)
import import_human_match as ihm  # noqa: E402


def _frame(t: float, *, player_x: float, bot_x: float, player_health: float = 20.0,
           bot_health: float = 20.0, round_no: int = 1) -> dict:
    return {
        "t": t,
        "round": round_no,
        "player": {"name": "P1", "health": player_health, "x": player_x, "z": 0.0,
                   "y": 4.0, "yaw": 0.0, "pitch": 0.0, "armor": 20.0},
        "bot": {"name": "B1", "health": bot_health, "x": bot_x, "z": 0.0,
                "y": 4.0, "yaw": 0.0, "pitch": 0.0, "armor": 20.0},
    }


class MacroToIntentTests(unittest.TestCase):
    def test_every_head_value_in_range(self) -> None:
        for name in ACTION_NAMES:
            intent = macro_to_intent(name)
            self.assertEqual(len(intent), len(ACTION_HEADS_V2), name)
            for value, (head, size) in zip(intent, ACTION_HEADS_V2.items()):
                self.assertTrue(0 <= value < size, f"{name} 的 {head}={value} 越界")

    def test_index_and_name_agree(self) -> None:
        for index, name in enumerate(ACTION_NAMES):
            self.assertEqual(macro_to_intent(index), macro_to_intent(name))

    def test_array_version_matches(self) -> None:
        array = macro_to_intent_array("attack")
        self.assertEqual(array.dtype, np.int64)
        self.assertEqual(tuple(int(v) for v in array), macro_to_intent("attack"))

    def test_semantics_not_reversed(self) -> None:
        attack = decode_intent(macro_to_intent("attack"))
        self.assertTrue(attack["attack"])
        self.assertEqual(attack["move"], "forward")
        self.assertTrue(attack["sprint"])
        self.assertNotIn("sprintReset", attack)

        eat = decode_intent(macro_to_intent("eat_gapple"))
        self.assertEqual(eat["use"], "hold")
        self.assertFalse(eat["attack"])

        block = decode_intent(macro_to_intent("block"))
        self.assertEqual(block["use"], "hold")

        rod = decode_intent(macro_to_intent("use_rod"))
        self.assertEqual(rod["use"], "click")

        jump = decode_intent(macro_to_intent("jump"))
        self.assertTrue(jump["jump"])
        self.assertFalse(jump["attack"])

        idle = decode_intent(macro_to_intent("idle"))
        self.assertEqual(idle["move"], "idle")
        self.assertEqual((idle["look"]["dyaw"], idle["look"]["dpitch"]), (0.0, 0.0))

    def test_unknown_index_rejects(self) -> None:
        with self.assertRaises(ValueError):
            macro_to_intent(len(ACTION_NAMES))
        with self.assertRaises(ValueError):
            macro_to_intent(-1)


class BuildDatasetV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        # 两帧一组的简单轨迹：先近距离（会推 attack），再退开（会推 retreat/idle）。
        self.frames = [
            _frame(0.0, player_x=0.0, bot_x=2.0),
            _frame(0.5, player_x=0.0, bot_x=2.0, bot_health=18.4),
            _frame(1.0, player_x=0.0, bot_x=2.5, bot_health=16.8),
            _frame(1.5, player_x=0.0, bot_x=6.0, bot_health=16.8),
        ]

    def test_v1_shape_unchanged(self) -> None:
        dataset = ihm.build_dataset(self.frames, player_name="P1", bot_name="B1",
                                    round_seconds=20.0)
        self.assertEqual(dataset["actionSpace"], "v1")
        self.assertEqual(dataset["actions"].shape, (4,))
        self.assertEqual(dataset["headSizes"], [])

    def test_v2_shape_and_macro_side_channel(self) -> None:
        dataset = ihm.build_dataset(self.frames, player_name="P1", bot_name="B1",
                                    round_seconds=20.0, action_space="v2")
        self.assertEqual(dataset["actionSpace"], "v2")
        self.assertEqual(dataset["actions"].shape, (4, len(MULTI_DISCRETE_V2)))
        self.assertEqual(dataset["headSizes"], list(MULTI_DISCRETE_V2))
        # 原始宏动作仍可统计（报告用）
        self.assertEqual(dataset["macroActions"].shape, (4,))
        np.testing.assert_array_equal(dataset["macroActions"],
                                      ihm.build_dataset(self.frames, player_name="P1",
                                                        bot_name="B1",
                                                        round_seconds=20.0)["actions"])
        for row in dataset["actions"]:
            for value, size in zip(row, MULTI_DISCRETE_V2):
                self.assertTrue(0 <= int(value) < size)

    def test_bad_action_space_rejects(self) -> None:
        with self.assertRaises(ValueError):
            ihm.build_dataset(self.frames, player_name="P1", bot_name="B1",
                              action_space="v9")

    def test_rows_match_macro_translation(self) -> None:
        dataset = ihm.build_dataset(self.frames, player_name="P1", bot_name="B1",
                                    round_seconds=20.0, action_space="v2")
        for index, macro in enumerate(dataset["macroActions"]):
            self.assertEqual(tuple(int(v) for v in dataset["actions"][index]),
                             macro_to_intent(int(macro)))


class RunBcV2Tests(unittest.TestCase):
    def _dataset(self, action_space: str) -> dict:
        dataset = ihm.build_dataset(self.frames(), player_name="P1", bot_name="B1",
                                    round_seconds=20.0, action_space=action_space)
        dataset["weights"] = np.ones(len(dataset["actions"]), dtype=np.float32)
        return dataset

    def frames(self) -> list[dict]:
        return [
            _frame(0.0, player_x=0.0, bot_x=2.0),
            _frame(0.5, player_x=0.0, bot_x=2.0, bot_health=18.4),
            _frame(1.0, player_x=0.0, bot_x=2.2, bot_health=16.8),
            _frame(1.5, player_x=0.0, bot_x=2.4, bot_health=15.2),
            _frame(2.0, player_x=0.0, bot_x=2.6, bot_health=13.6),
        ]

    def _v2_policy_file(self, tmp: Path) -> Path:
        from gladiator_train.policy import Policy

        policy = Policy(obs_dim=ihm.OBS_DIM, n_actions=11, hidden=64, seed=3,
                        action_space="v2")
        path = tmp / "v2-start.npz"
        policy.save(str(path))
        return path

    def test_bc_trains_on_v2_dataset(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            model = self._v2_policy_file(tmp)
            report = ihm.run_bc(self._dataset("v2"), base_model=str(model), epochs=2,
                                run_dir=tmp / "out", meta={"player": "P1"},
                                seed=0, min_samples=1)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertNotIn("skipped", report)
            self.assertEqual(report["samples"], 5)
            self.assertIn("accuracy", report)
            # Windows 上 np.load 会握住文件句柄，必须显式 close，否则临时目录删不掉。
            # 动作空间记在 `_meta` 的 JSON 里（不是顶层键），头名 W2_<head> 也能自证。
            with np.load(tmp / "out" / "policy.npz", allow_pickle=False) as saved:
                self.assertIn("W2_move", saved.files)
                self.assertEqual(json.loads(str(saved["_meta"]))["actionSpace"], "v2")
            self.assertTrue((tmp / "out" / "report.json").is_file())

    def test_v2_policy_with_v1_dataset_skips(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            model = self._v2_policy_file(tmp)
            report = ihm.run_bc(self._dataset("v1"), base_model=str(model), epochs=2,
                                run_dir=tmp / "out", meta={"player": "P1"},
                                seed=0, min_samples=1)
            assert report is not None
            self.assertTrue(report["skipped"])
            self.assertIn("v2", report["reason"])

    def test_label_helper(self) -> None:
        self.assertEqual(ihm._action_label(np.int64(3)), 3)
        self.assertEqual(ihm._action_label(np.asarray([0, 3, 2, 1, 0, 0, 0, 1, 0, 0])),
                         (0, 3, 2, 1, 0, 0, 0, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
