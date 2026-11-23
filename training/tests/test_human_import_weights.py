"""人机录像导入的择优权重（a 结果过滤 + b 接触窗口）。

覆盖：
- ``round_weight`` / ``contact_weight`` 的边界；
- ``build_dataset`` 把「局结果 × 接触窗口」乘进 ``weights``，且样本数不变；
- 全 1 权重与不给权重的 BC **逐位一致**（回归保护，防止加权改造改坏旧行为）；
- 权重确实改变 BC 结果（否则 (a)(b) 就是装饰）。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import numpy as np  # noqa: E402

from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.demos import behavior_clone  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402

SCRIPT_PATH = TRAINING_DIR / "scripts" / "import_human_match.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("import_human_match_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


human = _load_module()


def _frame(t: float, *, px: float = 0.0, pz: float = 0.0, bx: float = 0.0,
           bz: float = 0.0, round_number: int = 1, bot_health: float = 20.0) -> dict:
    return {
        "t": t,
        "round": round_number,
        "player": {"health": 20.0, "food": 20, "saturation": 5, "armor": 20.0,
                   "x": px, "y": 4.0, "z": pz, "yaw": 0.0, "pitch": 0.0,
                   "sprinting": False, "blocking": False, "onGround": True, "hurt": 0},
        "bot": {"health": bot_health, "food": 20, "saturation": 5, "armor": 20.0,
                "x": bx, "y": 4.0, "z": bz, "yaw": 0.0, "pitch": 0.0,
                "sprinting": False, "blocking": False, "onGround": True, "hurt": 0},
    }


class RoundWeightTests(unittest.TestCase):
    def test_three_outcomes(self) -> None:
        self.assertEqual(human.round_weight("player", win_weight=1.0, loss_weight=0.3,
                                            draw_weight=0.5), 1.0)
        self.assertEqual(human.round_weight("bot", win_weight=1.0, loss_weight=0.3,
                                            draw_weight=0.5), 0.3)
        self.assertEqual(human.round_weight("draw", win_weight=1.0, loss_weight=0.3,
                                            draw_weight=0.5), 0.5)
        self.assertEqual(human.round_weight("unknown", win_weight=1.0, loss_weight=0.3,
                                            draw_weight=0.5), 0.5)


class ContactWeightTests(unittest.TestCase):
    def test_near_is_full_weight(self) -> None:
        frame = _frame(0.0, px=0.0, pz=0.0, bx=3.0, bz=0.0)
        self.assertEqual(human.contact_window_weight(frame, player_name="p", bot_name="b",
                                              contact_range=5.0, contact_weight=1.0,
                                              far_weight=0.2), 1.0)

    def test_far_is_reduced(self) -> None:
        frame = _frame(0.0, px=0.0, pz=0.0, bx=30.0, bz=0.0)
        self.assertEqual(human.contact_window_weight(frame, player_name="p", bot_name="b",
                                              contact_range=5.0, contact_weight=1.0,
                                              far_weight=0.2), 0.2)

    def test_boundary_uses_distance_not_axis(self) -> None:
        frame = _frame(0.0, px=3.0, pz=4.0, bx=0.0, bz=0.0)  # 距离正好 5
        self.assertEqual(human.contact_window_weight(frame, player_name="p", bot_name="b",
                                              contact_range=5.0, contact_weight=1.0,
                                              far_weight=0.2), 1.0)
        frame_far = _frame(0.0, px=3.1, pz=4.0, bx=0.0, bz=0.0)
        self.assertEqual(human.contact_window_weight(frame_far, player_name="p", bot_name="b",
                                              contact_range=5.0, contact_weight=1.0,
                                              far_weight=0.2), 0.2)

    def test_missing_side_falls_back_to_far(self) -> None:
        frame = {"t": 0.0, "round": 1, "player": {"health": 20.0}}
        self.assertEqual(human.contact_window_weight(frame, player_name="p", bot_name="b",
                                              contact_range=5.0, contact_weight=1.0,
                                              far_weight=0.2), 0.2)


class BuildDatasetWeightTests(unittest.TestCase):
    def test_weights_combine_round_and_contact(self) -> None:
        frames = [
            _frame(0.0, px=0.0, bx=2.0, round_number=1),
            _frame(0.1, px=0.0, bx=2.0, round_number=1),
            _frame(0.2, px=0.0, bx=40.0, round_number=1),
            _frame(0.3, px=0.0, bx=2.0, round_number=2),
        ]
        winners = {1: "player", 2: "bot"}
        dataset = human.build_dataset(frames, player_name="p", bot_name="b",
                                      winners=winners)
        self.assertEqual(len(dataset["actions"]), 4)
        self.assertEqual(len(dataset["weights"]), 4)
        weights = dataset["weights"].tolist()
        # 第 1 局赢：接触帧满权重 1.0；远处帧 1.0 × 0.2
        self.assertAlmostEqual(weights[0], 1.0, places=6)
        self.assertAlmostEqual(weights[2], 0.2, places=6)
        # 第 2 局输：接触帧 0.3
        self.assertAlmostEqual(weights[3], 0.3, places=6)

    def test_defaults_keep_every_sample(self) -> None:
        frames = [_frame(i * 0.1, bx=big) for i, big in enumerate([0.0, 50.0, 100.0])]
        dataset = human.build_dataset(frames, player_name="p", bot_name="b",
                                      winners={1: "unknown"}, far_weight=0.2)
        self.assertTrue((dataset["weights"] > 0).all())

    def test_observation_shape_unchanged(self) -> None:
        frames = [_frame(i * 0.1, bx=2.0) for i in range(5)]
        dataset = human.build_dataset(frames, player_name="p", bot_name="b",
                                      winners={1: "player"})
        self.assertEqual(dataset["obs"].shape, (5, OBS_DIM))


class BehaviorCloneWeightTests(unittest.TestCase):
    def test_all_ones_matches_no_weights(self) -> None:
        rng = np.random.default_rng(7)
        samples = [(rng.normal(size=OBS_DIM), int(rng.integers(0, 11)), 0.0)
                   for _ in range(64)]
        policy_a = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=16, seed=3)
        policy_b = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=16, seed=3)
        result_a = behavior_clone(policy_a, samples, epochs=3, seed=1)
        result_b = behavior_clone(policy_b, samples, epochs=3, seed=1,
                                  weights=[1.0] * len(samples))
        for key in ("W1", "b1", "W2", "b2"):
            np.testing.assert_allclose(policy_a.params[key], policy_b.params[key])
        self.assertEqual(result_a["loss"], result_b["loss"])
        self.assertAlmostEqual(result_b["totalWeight"], float(len(samples)), places=6)

    def test_zero_weight_sample_has_no_effect(self) -> None:
        rng = np.random.default_rng(11)
        base = [(rng.normal(size=OBS_DIM), int(rng.integers(0, 11)), 0.0)
                for _ in range(32)]
        extra = [(rng.normal(size=OBS_DIM) * 5.0, 10, 0.0) for _ in range(8)]
        policy_a = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=16, seed=5)
        policy_b = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=16, seed=5)
        behavior_clone(policy_a, base, epochs=2, seed=2)
        behavior_clone(policy_b, base + extra, epochs=2, seed=2,
                       weights=[1.0] * len(base) + [0.0] * len(extra))
        for key in ("W1", "b1", "W2", "b2"):
            np.testing.assert_allclose(policy_a.params[key], policy_b.params[key],
                                       rtol=1e-9, atol=1e-12)

    def test_weight_changes_direction(self) -> None:
        """同一批样本，把权重全压在 A 类动作上，策略应更偏向 A。

        G-711：这条用例的数据刻意用**固定 25 维**（不与 ``OBS_DIM`` 绑定）。
        它测的是「权重方向」，若跟着观测维度走，每次加维都会换一批随机数
        （``rng.normal(size=D)`` 的流不同），断言就退化成对种子敏感了。
        """

        dim = 25
        rng = np.random.default_rng(23)
        obs_a = [rng.normal(size=dim) for _ in range(16)]
        obs_b = [obs + 4.0 for obs in obs_a]
        samples = [(obs, 2, 0.0) for obs in obs_a] + [(obs, 9, 0.0) for obs in obs_b]
        policy = Policy(obs_dim=dim, n_actions=11, hidden=16, seed=9)
        before = policy.probs(obs_a[0])
        behavior_clone(policy, samples, epochs=20, seed=4,
                       weights=[1.0] * len(obs_a) + [0.0] * len(obs_b))
        after = policy.probs(obs_a[0])
        # 初始分布里 9 号动作概率更高，所以只能比较「变化方向」而不是绝对值。
        self.assertGreater(float(after[2]), float(before[2]))
        self.assertLess(float(after[9]), float(before[9]))

    def test_weight_length_mismatch_raises(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=8, seed=1)
        samples = [(np.zeros(OBS_DIM), 0, 0.0)]
        with self.assertRaises(ValueError):
            behavior_clone(policy, samples, epochs=1, weights=[1.0, 2.0])

    def test_negative_weight_raises(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=11, hidden=8, seed=1)
        samples = [(np.zeros(OBS_DIM), 0, 0.0)]
        with self.assertRaises(ValueError):
            behavior_clone(policy, samples, epochs=1, weights=[-1.0])


class RoundWinnerLoadingTests(unittest.TestCase):
    def test_missing_summary_returns_empty(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(human.load_round_winners(Path(tmp)), {})

    def test_reads_round_winners(self) -> None:
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"rounds": [
                {"round": 1, "winner": "player"},
                {"round": 2, "winner": "bot"},
                {"round": "3", "winner": "draw"},
            ]}), encoding="utf-8")
            winners = human.load_round_winners(Path(tmp))
            self.assertEqual(winners, {1: "player", 2: "bot", 3: "draw"})


class RewardColumnTests(unittest.TestCase):
    """``build_dataset`` 现在会带 ``rewards``（只由血量差推出，给 AWR 用）。"""

    @staticmethod
    def _frame(t: float, player_hp: float, bot_hp: float, round_no: int = 1) -> dict:
        return {
            "t": t,
            "round": round_no,
            "player": {"health": player_hp, "x": 0.0, "y": 4.0, "z": 0.0},
            "bot": {"health": bot_hp, "x": 3.0, "y": 4.0, "z": 0.0},
        }

    def test_rewards_track_health_delta(self) -> None:
        frames = [
            self._frame(0.0, 20.0, 20.0),
            self._frame(1.0, 20.0, 16.0),  # 打掉对手 4
            self._frame(2.0, 17.0, 16.0),  # 自己吃 3
        ]
        dataset = human.build_dataset(frames, player_name="p", bot_name="b")
        self.assertEqual(list(dataset["rewards"]), [0.0, 4.0, -3.0])

    def test_rewards_length_matches_samples(self) -> None:
        frames = [self._frame(float(i), 20.0, 20.0) for i in range(5)]
        dataset = human.build_dataset(frames, player_name="p", bot_name="b")
        self.assertEqual(len(dataset["rewards"]), len(dataset["actions"]))
        self.assertEqual(len(dataset["rewards"]), 5)

    def test_win_bonus_only_on_round_boundary(self) -> None:
        frames = [
            self._frame(0.0, 20.0, 20.0, 1),
            self._frame(1.0, 20.0, 10.0, 1),
            self._frame(2.0, 20.0, 20.0, 2),  # 第二局开局的跨界帧
        ]
        dataset = human.build_dataset(
            frames, player_name="p", bot_name="b", winners={1: "player"}, win_bonus=7.0)
        self.assertEqual(list(dataset["rewards"]), [0.0, 10.0, 7.0])


if __name__ == "__main__":
    unittest.main()
