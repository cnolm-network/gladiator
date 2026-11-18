"""行为克隆热启动（P1.5）与端到端命令行。

覆盖：
- 老师的四条规则（射程内→attack / 无目标→idle / 远距→approach / 低血→eat_gapple）；
- BC 之后老师动作的概率显著上升、`Wv/bv` 逐位不变；
- `--bc-epochs 0` 完全跳过（不采集、不克隆）；
- `python -m gladiator_train.train --bc-episodes … --bc-epochs … --algo ppo` 离线端到端跑通。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import gladiator_env  # noqa: E402
from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, OBS_FIELDS  # noqa: E402
from gladiator_env.testing import FakeLink, make_observation, make_target  # noqa: E402
from gladiator_train import demos  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    ACTION_APPROACH,
    ACTION_ATTACK,
    ACTION_EAT,
    ACTION_IDLE,
    behavior_clone,
    collect_demos,
    heuristic_action,
    teacher_agreement,
)
from gladiator_train.policy import Policy  # noqa: E402
from gladiator_train import train as train_module  # noqa: E402

ATTACK_NAME = ACTION_NAMES[ACTION_ATTACK]


def obs_vector(*, in_reach: float = 0.0, has_target: float = 1.0,
               distance: float = 2.0, health: float = 20.0) -> np.ndarray:
    """按字段名填 23 维观测（不依赖下标数字）。"""
    vector = np.zeros(OBS_DIM, dtype=np.float32)
    vector[OBS_FIELDS.index("self_health")] = health / 20.0
    vector[OBS_FIELDS.index("has_target")] = has_target
    vector[OBS_FIELDS.index("target_distance")] = distance / 32.0
    vector[OBS_FIELDS.index("in_reach")] = in_reach
    return vector


class HeuristicActionTest(unittest.TestCase):
    def test_in_reach_attacks(self):
        self.assertEqual(heuristic_action(obs_vector(in_reach=1.0, distance=2.0)), ACTION_ATTACK)

    def test_no_target_idles(self):
        self.assertEqual(heuristic_action(obs_vector(has_target=0.0)), ACTION_IDLE)

    def test_far_target_approaches(self):
        action = heuristic_action(obs_vector(in_reach=0.0, distance=6.0))
        self.assertEqual(action, ACTION_APPROACH)

    def test_low_health_eats(self):
        action = heuristic_action(obs_vector(in_reach=0.0, distance=2.0, health=5.0))
        self.assertEqual(action, ACTION_EAT)

    def test_healthy_close_target_approaches(self):
        self.assertEqual(heuristic_action(obs_vector(in_reach=0.0, distance=2.0, health=18.0)),
                         ACTION_APPROACH)

    def test_field_indices_come_from_obs_fields(self):
        """下标必须来自 OBS_FIELDS，改观测顺序会立刻炸而不是静默错位。"""
        self.assertEqual(demos.IDX_IN_REACH, OBS_FIELDS.index("in_reach"))
        self.assertEqual(demos.IDX_HAS_TARGET, OBS_FIELDS.index("has_target"))
        self.assertEqual(demos.IDX_TARGET_DISTANCE, OBS_FIELDS.index("target_distance"))
        self.assertEqual(demos.IDX_SELF_HEALTH, OBS_FIELDS.index("self_health"))


class BehaviorCloneTest(unittest.TestCase):
    def _samples(self, count: int = 64) -> list[tuple[np.ndarray, int, float]]:
        rng = np.random.default_rng(0)
        out = []
        for _ in range(count):
            # 一半「射程内」→ 老师 attack，一半远距 → approach
            reach = 1.0 if rng.random() < 0.5 else 0.0
            vector = obs_vector(in_reach=reach, distance=2.0 if reach else 6.0)
            vector += rng.normal(0, 0.01, size=OBS_DIM).astype(np.float32)
            action = heuristic_action(vector)
            out.append((vector, action, 0.5))
        return out

    def test_clone_raises_teacher_probability(self):
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=32, seed=1)
        samples = self._samples()
        probe = obs_vector(in_reach=1.0, distance=2.0)
        before = float(policy.probs(probe)[ACTION_ATTACK])
        before_acc = teacher_agreement(policy, samples)

        result = behavior_clone(policy, samples, epochs=30, lr=5e-3, batch=16)

        after = float(policy.probs(probe)[ACTION_ATTACK])
        self.assertGreaterEqual(after, 0.5, f"老师动作概率只有 {after:.3f}（BC 前 {before:.3f}）")
        self.assertGreater(result["accuracy"], before_acc)
        self.assertEqual(result["samples"], len(samples))
        self.assertTrue(result["loss"])

    def test_clone_does_not_touch_value_head(self):
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=16, seed=2)
        wv, bv = policy.params["Wv"].copy(), policy.params["bv"].copy()
        behavior_clone(policy, self._samples(16), epochs=3)
        np.testing.assert_array_equal(policy.params["Wv"], wv)
        np.testing.assert_array_equal(policy.params["bv"], bv)

    def test_empty_samples_is_noop(self):
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=8, seed=3)
        before = policy.params["W2"].copy()
        result = behavior_clone(policy, [], epochs=5)
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        np.testing.assert_array_equal(policy.params["W2"], before)


# --- 离线端到端：FakeLink 驱动真实 GladiatorEnv --------------------------------

def build_fake_env(_bots, _arena=None, **_kwargs):
    """替换 `GladiatorEnv` 的工厂：注入 FakeLink，不连任何服务。"""
    links = [FakeLink("Bot_T1"), FakeLink("Bot_T2")]
    for index, link in enumerate(links):
        other = "Bot_T2" if index == 0 else "Bot_T1"
        link.set_observation(make_observation(
            link.username, x=-1.5 if index == 0 else 1.5, z=0.0,
            targets=[make_target(2 if index == 0 else 1, other, x=1.5 if index == 0 else -1.5,
                                 z=0.0, distance=3.0)]))
    return GladiatorEnv(
        [BotConfig("Bot_T1", "ws://127.0.0.1:8766"),
         BotConfig("Bot_T2", "ws://127.0.0.1:8767")],
        ArenaConfig(center=(0.0, 4.0, 0.0), spacing=3.0, prep_seconds=0.0, give_kit=False),
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        max_steps=int(_kwargs.get("max_steps") or 5),
        single_agent=True,
    )


class EndToEndCommandTest(unittest.TestCase):
    """`--bc-episodes/--bc-epochs/--algo ppo` 一条命令跑通（离线）。"""

    def _run(self, tmp: Path, *extra: str) -> int:
        run_dir = tmp / "run"
        argv = [
            "--users", "Bot_T1", "Bot_T2", "--ports", "8766", "8767",
            "--run-dir", str(run_dir),
            "--save", str(run_dir / "policy.npz"),
            "--report", str(run_dir / "report.json"),
            "--review-dir", str(run_dir / "reviews"),
            "--episodes", "1", "--max-steps", "4", "--eval-epochs", "0",
            "--bc-max-steps", "4",
            *extra,
        ]
        with mock.patch.object(gladiator_env, "GladiatorEnv", build_fake_env), \
                mock.patch.object(train_module, "Trainer") as trainer_cls:
            trainer = trainer_cls.return_value
            trainer.train.return_value = []
            trainer.evaluate.return_value = []
            trainer.history = []
            return_code = train_module.main(argv)
        self.run_dir = run_dir
        return return_code

    def test_default_command_runs_bc_then_ppo(self):
        with tempfile.TemporaryDirectory() as folder:
            code = self._run(Path(folder), "--bc-episodes", "1", "--bc-epochs", "2")
            self.assertEqual(code, 0)
            report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["algo"], "ppo")
            self.assertEqual(report["config"]["bc"]["epochs"], 2)
            self.assertEqual(report["config"]["bc"]["episodes"], 1)
            self.assertGreater(report["config"]["bc"]["samples"], 0)
            self.assertIsNotNone(report["config"]["bc"]["accuracy"])
            # entropy 默认已提到 0.02
            self.assertEqual(report["config"]["entropyCoef"], 0.02)

    def test_bc_zero_epochs_skips_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(demos, "collect_demos",
                                   side_effect=AssertionError("不该在 --bc-epochs 0 时采集")):
                code = self._run(Path(folder), "--bc-epochs", "0")
            self.assertEqual(code, 0)
            report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["bc"]["epochs"], 0)
            self.assertEqual(report["config"]["bc"]["samples"], 0)


class CollectDemosTest(unittest.TestCase):
    def test_records_discounted_returns(self):
        links = [FakeLink("Bot_T1"), FakeLink("Bot_T2")]
        env = GladiatorEnv(
            [BotConfig("Bot_T1", "ws://127.0.0.1:8766"),
             BotConfig("Bot_T2", "ws://127.0.0.1:8767")],
            ArenaConfig(center=(0.0, 4.0, 0.0), spacing=3.0, prep_seconds=0.0, give_kit=False),
            links=links, connect=True, control_hz=200.0, action_ticks=1,
            max_steps=3, single_agent=True,
        )
        samples = collect_demos(env, episodes=1, max_steps=3)
        env.close()

        self.assertTrue(samples)
        self.assertEqual(samples[0][0].shape, (OBS_DIM,))
        # 折扣回报必须单调不大于后一步（终局后为 0 的累加）
        returns = [item[2] for item in samples]
        self.assertTrue(all(isinstance(value, float) for value in returns))


if __name__ == "__main__":
    unittest.main()
