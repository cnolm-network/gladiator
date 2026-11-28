"""v2 动作空间的 BC / 蒸馏通路（G-710）。

背景：teacher v4 输出的是 **v2 十头动作**（含 ``sprint_reset``），而 ``train.py`` 的
BC 通路原来只认 v1 的「一个动作编号」。这里覆盖三件必须成立的事：

1. **标签翻译**：``intent`` → 10 头下标元组，顺序必须与 ``ACTION_HEADS_V2`` 一致；
2. **损失口径**：逐头交叉熵按头平均——只在某一个头犯错时，损失只涨那一头的
   ``1/H``，而不是把其它头也一起惩罚（用「损失差恰好等于 2/H」这条恒等式验证）；
3. **权重语义**：v2 样本第三字段是权重；全 1 == 不传，统一缩放不改变结果，
   0 权等样本等于把它删掉（v1 的 15 条权重测试仍然守着旧路径）。

另有 CLI 组合校验（v2 下的 BC 只接受 teacher v4）与 v1 回归。
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
from gladiator_env.spaces import (  # noqa: E402
    ACTION_HEADS_V2,
    ACTION_NAMES,
    MULTI_DISCRETE_V2,
    OBS_DIM,
    decode_intent,
    encode_observation,
)
from gladiator_env.testing import FakeLink, make_observation, make_target  # noqa: E402
from gladiator_train import train as train_module  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    V2_HEADS,
    TeacherV4,
    behavior_clone,
    collect_dagger_samples,
    collect_demos,
    intent_to_tuple,
    make_teacher,
    teacher_agreement,
    v4_intent_of,
)
from gladiator_train.policy import Policy  # noqa: E402

N_HEADS = len(ACTION_HEADS_V2)
ATTACK_HEAD = list(ACTION_HEADS_V2).index("attack")

FULL_INVENTORY = [
    {"slot": 36, "name": "diamond_sword", "count": 1},
    {"slot": 37, "name": "golden_apple", "count": 5},
]


def raw_observation(username: str = "Glad_Bot_1", *, distance: float = 2.5,
                    target_health: float = 20.0) -> dict:
    other = "Glad_Bot_2" if username == "Glad_Bot_1" else "Glad_Bot_1"
    raw = make_observation(
        username, health=20.0,
        targets=[make_target(2, other, x=distance, z=0.0, distance=distance)])
    raw["self"]["inventory"] = list(FULL_INVENTORY)
    raw["self"]["heldItem"] = {"name": "diamond_sword", "count": 1}
    return raw


def obs_vector(distance: float = 2.5) -> np.ndarray:
    """真机同口径的 25 维观测（老师只要观测向量，不需要 env）。"""
    return encode_observation(
        raw_observation(distance=distance), "Glad_Bot_1",
        reach=3.99, opponent_health=20.0)


def flat_policy(*, hidden: int = 8, seed: int = 0) -> Policy:
    """所有头都「偏爱 0 号取值」的策略：softmax([2,0,...])，便于手算交叉熵。

    隐层权重与偏置全 0，于是 p 与输入无关，损失差可以直接解析算出来。
    """
    policy = Policy(obs_dim=OBS_DIM, action_space="v2", hidden=hidden, seed=seed)
    policy.params["W1"][:] = 0.0
    policy.params["b1"][:] = 0.0
    for head, size in zip(policy.head_names(), policy.head_sizes()):
        policy.params[f"W2_{head}"][:] = 0.0
        bias = np.zeros(size)
        bias[0] = 2.0
        policy.params[f"b2_{head}"][:] = bias
    return policy


def zero_labels() -> tuple[int, ...]:
    return tuple(0 for _ in range(N_HEADS))


def samples_of(vectors, label, weight: float = 1.0):
    return [(np.asarray(vector, dtype=np.float64), label, weight) for vector in vectors]


class HeadOrderTest(unittest.TestCase):
    def test_head_order_matches_spaces(self) -> None:
        self.assertEqual(V2_HEADS, tuple(ACTION_HEADS_V2))
        self.assertEqual(len(V2_HEADS), 10)
        self.assertEqual(tuple(MULTI_DISCRETE_V2), (5, 7, 5, 2, 3, 2, 2, 2, 9, 3))


class IntentToTupleTest(unittest.TestCase):
    def test_roundtrip_with_teacher_v4(self) -> None:
        """v4 的动作 → decode_intent → intent_to_tuple 必须回到同一个下标元组。"""
        teacher = TeacherV4()
        for distance in (1.5, 2.5, 3.2, 4.5, 6.0):
            meta: dict = {}
            action = teacher(obs_vector(distance), meta)
            self.assertEqual(len(action), N_HEADS)
            self.assertEqual(intent_to_tuple(meta), tuple(int(value) for value in action))
            self.assertEqual(intent_to_tuple(v4_intent_of(action)),
                             tuple(int(value) for value in action))

    def test_accepts_wrapped_meta(self) -> None:
        """``{"intent": {...}, "reason": ...}`` 与裸 intent 等价。"""
        intent = {"move": "forward", "look": {"dyaw": 10.0, "dpitch": -5.0},
                  "attack": True, "use": "click", "slot": 3,
                  "jump": True, "sneak": False, "sprint": True,
                  "sprintReset": "release_w"}
        direct = intent_to_tuple(intent)
        wrapped = intent_to_tuple({"intent": intent, "reason": "block_hit"})
        self.assertEqual(direct, wrapped)
        self.assertEqual(direct, (1, 5, 1, 1, 1, 1, 0, 1, 3, 1))

    def test_defaults_are_neutral(self) -> None:
        """缺省 = 不转视角 / 不按右键 / 不重置疾跑，与 decode_intent 的对称规则一致。"""
        self.assertEqual(intent_to_tuple({"move": "idle"}),
                         (0, 3, 2, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(intent_to_tuple({}), (0, 3, 2, 0, 0, 0, 0, 0, 0, 0))

    def test_sprint_reset_values(self) -> None:
        base = {"move": "idle"}
        self.assertEqual(intent_to_tuple({**base, "sprintReset": "none"})[-1], 0)
        self.assertEqual(intent_to_tuple({**base, "sprintReset": "release_w"})[-1], 1)
        self.assertEqual(intent_to_tuple({**base, "sprintReset": "block_hit"})[-1], 2)

    def test_unknown_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            intent_to_tuple({"move": "teleport"})
        with self.assertRaises(ValueError):
            intent_to_tuple({"move": "idle", "slot": 9})
        with self.assertRaises(ValueError):
            intent_to_tuple({"move": "idle", "attack": "yes"})
        with self.assertRaises(ValueError):
            intent_to_tuple({"move": "idle", "sprintReset": "dash"})
        with self.assertRaises(ValueError):
            intent_to_tuple(None)  # type: ignore[arg-type]


def v2_env(max_steps: int = 3):
    """离线 v2 单智能体环境（两个 FakeLink，什么都不连）。"""
    links = [FakeLink("Glad_Bot_1", raw_observation("Glad_Bot_1")),
             FakeLink("Glad_Bot_2", raw_observation("Glad_Bot_2"))]
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        arena=ArenaConfig(give_kit=False),
        max_steps=max_steps,
        single_agent=True,
        action_space="v2",
    )
    return env, links


class CollectDemosV2Test(unittest.TestCase):
    def _env(self, max_steps: int = 3):
        return v2_env(max_steps)

    def test_teacher_v4_yields_ten_head_labels(self) -> None:
        env, _links = self._env()
        try:
            samples = collect_demos(env, episodes=1, max_steps=3, teacher_version="v4")
        finally:
            env.close()
        self.assertTrue(samples)
        for vector, action, weight in samples:
            self.assertEqual(np.asarray(vector).shape, (OBS_DIM,))
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), N_HEADS)
            for index, (value, size) in enumerate(zip(action, MULTI_DISCRETE_V2)):
                self.assertTrue(0 <= value < size, f"第 {index} 个头越界: {value}")
            # v2 的第三字段是权重（不是折扣回报）
            self.assertEqual(weight, 1.0)

    def test_rejects_nine_head_teacher(self) -> None:
        """旧 9 头老师喂进 v2 通路必须报错，而不是把 sprint_reset 悄悄补 0。"""
        env, _links = self._env()
        nine = lambda _obs, _meta=None: [0] * 9  # noqa: E731
        try:
            with self.assertRaises(ValueError):
                collect_demos(env, episodes=1, max_steps=1, teacher=nine)
        finally:
            env.close()

    def test_v1_collect_still_returns_returns(self) -> None:
        """v1 回归：标签是 int，第三字段是 float 折扣回报（不是权重）。"""
        links = [FakeLink("Glad_Bot_1", raw_observation("Glad_Bot_1")),
                 FakeLink("Glad_Bot_2", raw_observation("Glad_Bot_2"))]
        env = GladiatorEnv(
            [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
             BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
            links=links, connect=True, control_hz=200.0, action_ticks=1,
            arena=ArenaConfig(give_kit=False), max_steps=3, single_agent=True,
        )
        try:
            samples = collect_demos(env, episodes=1, max_steps=3, teacher_version="v1")
        finally:
            env.close()
        self.assertTrue(samples)
        for _vector, action, value in samples:
            self.assertIsInstance(action, int)
            self.assertIsInstance(value, float)


class PerHeadLossTest(unittest.TestCase):
    """损失恒等式：只在一个头犯错时，损失只涨 ``2/H``（p=softmax([2,0,…]) → CE 差=2）。"""

    def test_only_the_wrong_head_is_penalized(self) -> None:
        vectors = [obs_vector(2.5)] * 4
        policy = flat_policy()
        clean = behavior_clone(policy, samples_of(vectors, zero_labels()),
                               epochs=1, lr=0.0, batch=64)
        wrong = list(zero_labels())
        wrong[ATTACK_HEAD] = 1
        dirty = behavior_clone(policy, samples_of(vectors, tuple(wrong)),
                              epochs=1, lr=0.0, batch=64)
        delta = float(dirty["loss"][0]) - float(clean["loss"][0])
        self.assertAlmostEqual(delta, 2.0 / N_HEADS, places=6)

    def test_wrong_head_is_averaged_not_summed(self) -> None:
        """同时错两个头时损失涨 ``2×2/H``：证明是按头**平均**而不是求和。"""
        vectors = [obs_vector(2.5)] * 3
        policy = flat_policy()
        clean = behavior_clone(policy, samples_of(vectors, zero_labels()),
                               epochs=1, lr=0.0)
        two = list(zero_labels())
        two[ATTACK_HEAD] = 1
        two[0] = 1  # move 头
        both = behavior_clone(policy, samples_of(vectors, tuple(two)),
                              epochs=1, lr=0.0)
        self.assertAlmostEqual(float(both["loss"][0]) - float(clean["loss"][0]),
                               4.0 / N_HEADS, places=6)


class BehaviorCloneV2Test(unittest.TestCase):
    def test_training_raises_per_head_accuracy(self) -> None:
        teacher = make_teacher("v4")
        vectors = [obs_vector(distance) for distance in
                   (1.6, 2.2, 2.8, 3.4, 4.6, 5.4, 1.9, 2.4)]
        labels = [intent_to_tuple({"intent": v4_intent_of(teacher(vector))})
                  for vector in vectors]
        samples = [(np.asarray(vector, dtype=np.float64), label, 1.0)
                   for vector, label in zip(vectors, labels)]
        policy = Policy(obs_dim=OBS_DIM, action_space="v2", hidden=32, seed=3)
        before = teacher_agreement(policy, samples)["meanHeadAgreement"]
        result = behavior_clone(policy, samples, epochs=40, lr=5e-3, batch=8)
        after = teacher_agreement(policy, samples)["meanHeadAgreement"]
        self.assertEqual(len(result["perHeadAccuracy"]), N_HEADS)
        self.assertGreater(after, before)
        self.assertGreater(result["accuracy"], before)
        self.assertEqual(result["samples"], len(samples))
        self.assertEqual(result["totalWeight"], float(len(samples)))
        # accuracy 与 agreement 的「逐头平均」同口径
        self.assertAlmostEqual(result["accuracy"],
                               sum(result["perHeadAccuracy"]) / N_HEADS, places=4)

    def test_value_head_untouched(self) -> None:
        samples = samples_of([obs_vector(2.5)] * 4, zero_labels())
        policy = flat_policy()
        wv, bv = policy.params["Wv"].copy(), policy.params["bv"].copy()
        behavior_clone(policy, samples, epochs=3, lr=1e-2)
        np.testing.assert_array_equal(policy.params["Wv"], wv)
        np.testing.assert_array_equal(policy.params["bv"], bv)

    def test_empty_samples_is_noop(self) -> None:
        policy = flat_policy()
        before = {key: value.copy() for key, value in policy.params.items()}
        result = behavior_clone(policy, [], epochs=5)
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["epochs"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["perHeadAccuracy"], [0.0] * N_HEADS)
        self.assertEqual(result["loss"], [])
        for key, value in policy.params.items():
            np.testing.assert_array_equal(value, before[key])

    def test_all_ones_weights_equal_no_weights(self) -> None:
        """第三字段全 1 且 ``weights`` 不传 == 显式传全 1（逐位一致）。"""
        vectors = [obs_vector(2.5), obs_vector(3.1), obs_vector(1.8)]
        label = zero_labels()
        a = flat_policy(seed=7)
        b = flat_policy(seed=7)
        behavior_clone(a, samples_of(vectors, label), epochs=4, lr=1e-2)
        behavior_clone(b, samples_of(vectors, label), epochs=4, lr=1e-2,
                       weights=[1.0] * len(vectors))
        for key in a.params:
            np.testing.assert_array_equal(a.params[key], b.params[key])

    def test_weight_scale_invariance(self) -> None:
        """全体权重同乘一个常数不改变结果（梯度按权重和归一化）。

        只保证到浮点误差（``(2.5·g)/(2.5·n)`` 与 ``g/n`` 的最后一位可以不同），
        这不是「差不多」而是归一化的定义：**相对**权重才影响梯度。
        """
        vectors = [obs_vector(2.5), obs_vector(3.1)]
        label = zero_labels()
        a = flat_policy(seed=9)
        b = flat_policy(seed=9)
        behavior_clone(a, samples_of(vectors, label), epochs=3, lr=1e-2,
                       weights=[1.0, 1.0])
        behavior_clone(b, samples_of(vectors, label), epochs=3, lr=1e-2,
                       weights=[2.5, 2.5])
        for key in a.params:
            np.testing.assert_allclose(a.params[key], b.params[key],
                                       rtol=0.0, atol=1e-12)

    def test_zero_weight_equals_removed_sample(self) -> None:
        """0 权样本等于把它删掉（v2 的权重确实作用到梯度与损失）。"""
        label = zero_labels()
        wrong = list(label)
        wrong[ATTACK_HEAD] = 1
        kept = [obs_vector(2.5), obs_vector(2.5)]
        a = flat_policy(seed=11)
        b = flat_policy(seed=11)
        only = behavior_clone(a, samples_of(kept, label), epochs=3, lr=1e-2)
        samples = samples_of(kept, label) + samples_of([obs_vector(6.0)], tuple(wrong))
        masked = behavior_clone(b, samples, epochs=3, lr=1e-2,
                                weights=[1.0, 1.0, 0.0])
        self.assertAlmostEqual(only["loss"][-1], masked["loss"][-1], places=6)
        for key in a.params:
            np.testing.assert_array_equal(a.params[key], b.params[key])

    def test_sample_weight_scales_like_weights_argument(self) -> None:
        """第三字段的权重与 ``weights`` 参数相乘：2.0 的样本权 == 1.0 + weights 2.0。"""
        vectors = [obs_vector(2.5), obs_vector(3.4)]
        label = zero_labels()
        a = flat_policy(seed=13)
        b = flat_policy(seed=13)
        behavior_clone(a, samples_of(vectors, label, weight=2.0), epochs=2, lr=1e-2)
        behavior_clone(b, samples_of(vectors, label, weight=1.0), epochs=2, lr=1e-2,
                       weights=[2.0, 2.0])
        self.assertEqual(a.params["b2_attack"].shape, (2,))
        for key in a.params:
            np.testing.assert_array_equal(a.params[key], b.params[key])

    def test_bad_label_length_raises(self) -> None:
        policy = flat_policy()
        with self.assertRaises(ValueError):
            behavior_clone(policy, samples_of([obs_vector()], (0,) * 9), epochs=1)

    def test_out_of_range_label_raises(self) -> None:
        policy = flat_policy()
        bad = list(zero_labels())
        bad[0] = 5  # move 头只有 5 档（0..4）
        with self.assertRaises(ValueError):
            behavior_clone(policy, samples_of([obs_vector()], tuple(bad)), epochs=1)

    def test_negative_weight_raises(self) -> None:
        policy = flat_policy()
        with self.assertRaises(ValueError):
            behavior_clone(policy, samples_of([obs_vector()], zero_labels(), weight=-1.0),
                           epochs=1)

    def test_temperature_softens_each_head(self) -> None:
        """T>1 时目标不再 one-hot：老师熵 > 0，且熵随 T 增大而增大。"""
        samples = samples_of([obs_vector(2.5)], zero_labels())
        hot = behavior_clone(flat_policy(), samples, epochs=1, lr=0.0, temperature=3.0)
        cold = behavior_clone(flat_policy(), samples, epochs=1, lr=0.0, temperature=1.0)
        self.assertEqual(cold["teacherEntropy"], 0.0)
        self.assertGreater(hot["teacherEntropy"], 0.0)


class TeacherAgreementV2Test(unittest.TestCase):
    def test_reports_mean_and_exact(self) -> None:
        policy = flat_policy()
        good = zero_labels()
        mixed = list(zero_labels())
        mixed[ATTACK_HEAD] = 1  # 只有 attack 头与策略不同
        samples = samples_of([obs_vector()] * 3, good) + samples_of(
            [obs_vector()] * 1, tuple(mixed))
        report = teacher_agreement(policy, samples)
        self.assertEqual(set(report), {"meanHeadAgreement", "exactMatch", "perHead",
                                       "heads", "samples"})
        self.assertEqual(report["samples"], 4)
        self.assertEqual(len(report["perHead"]), N_HEADS)
        self.assertEqual(report["heads"], list(ACTION_HEADS_V2))
        self.assertAlmostEqual(report["meanHeadAgreement"],
                               sum(report["perHead"]) / N_HEADS, places=4)
        # 3/4 帧十头全对；4/4 帧里除 attack 外都对
        self.assertAlmostEqual(report["exactMatch"], 0.75, places=4)
        self.assertAlmostEqual(report["perHead"][ATTACK_HEAD], 0.75, places=4)
        self.assertAlmostEqual(report["perHead"][0], 1.0, places=4)

    def test_empty_samples(self) -> None:
        report = teacher_agreement(flat_policy(), [])
        self.assertEqual(report["samples"], 0)
        self.assertEqual(report["meanHeadAgreement"], 0.0)
        self.assertEqual(report["perHead"], [0.0] * N_HEADS)

    def test_v1_agreement_is_still_a_float(self) -> None:
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=8, seed=1)
        value = teacher_agreement(policy, [(obs_vector(), 3, 0.0)])
        self.assertIsInstance(value, float)
        self.assertTrue(0.0 <= value <= 1.0)


class DaggerV2Test(unittest.TestCase):
    """``--dagger-rounds`` 在 v2 下可用：策略自己跑、同帧取 v4 标签。"""

    def test_collect_dagger_samples_uses_policy_and_v4_labels(self) -> None:
        env, _links = v2_env(max_steps=3)
        policy = Policy(obs_dim=OBS_DIM, action_space="v2", hidden=8, seed=1)
        try:
            samples = collect_dagger_samples(
                env, policy, episodes=1, max_steps=3,
                rng=np.random.default_rng(0), teacher_version="v4")
        finally:
            env.close()
        self.assertTrue(samples)
        for vector, label, weight in samples:
            self.assertEqual(np.asarray(vector).shape, (OBS_DIM,))
            self.assertIsInstance(label, tuple)
            self.assertEqual(len(label), N_HEADS)
            self.assertEqual(weight, 1.0)  # v2 的第三字段必须是权重，0.0 会让梯度全灭
        # 采出来的样本能直接喂 BC（说明「策略跑」与「老师标签」两条链路对齐）
        result = behavior_clone(policy, samples, epochs=2, lr=1e-3)
        self.assertEqual(result["samples"], len(samples))
        self.assertEqual(len(result["perHeadAccuracy"]), N_HEADS)


def build_v2_env(_bots, _arena=None, **_kwargs):
    """CLI 端到端用的 v2 单智能体环境（FakeLink，不连任何服务）。"""
    links = [FakeLink("Glad_Bot_1", raw_observation("Glad_Bot_1")),
             FakeLink("Glad_Bot_2", raw_observation("Glad_Bot_2"))]
    return GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        arena=ArenaConfig(center=(0.0, 4.0, 0.0), spacing=3.0, prep_seconds=0.0,
                          give_kit=False),
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        max_steps=int(_kwargs.get("max_steps") or 4),
        single_agent=True,
        action_space="v2",
    )


class TrainCliV2Test(unittest.TestCase):
    def _run(self, tmp: Path, *extra: str) -> int:
        run_dir = tmp / "run-v2"
        argv = [
            "--users", "Glad_Bot_1", "Glad_Bot_2", "--ports", "8766", "8767",
            "--run-dir", str(run_dir),
            "--save", str(run_dir / "policy.npz"),
            "--report", str(run_dir / "report.json"),
            "--review-dir", str(run_dir / "reviews"),
            "--max-steps", "4", "--eval-epochs", "0",
            "--bc-max-steps", "3",
            *extra,
        ]
        with mock.patch.object(gladiator_env, "GladiatorEnv", build_v2_env), \
                mock.patch.object(train_module, "Trainer") as trainer_cls:
            trainer = trainer_cls.return_value
            trainer.train.return_value = []
            trainer.evaluate.return_value = []
            trainer.history = []
            code = train_module.main(argv)
        self.run_dir = run_dir
        self.trainer = trainer
        return code

    def test_v2_bc_rejects_non_v4_teacher(self) -> None:
        """v2 + 非 v4 老师：exit 2 且不创建 run 目录。"""
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            for teacher in ("v1", "v2", "v3"):
                code = self._run(tmp, "--action-space", "v2", "--bc-episodes", "1",
                                 "--bc-epochs", "1", "--bc-teacher", teacher,
                                 "--episodes", "0")
                self.assertEqual(code, 2, f"teacher={teacher} 应被拒绝")
            self.assertFalse(self.run_dir.exists(), "参数被拒时不该留下 run 目录")

    def test_v2_dagger_rejects_non_v4_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code = self._run(Path(folder), "--action-space", "v2", "--episodes", "2",
                             "--bc-episodes", "0", "--dagger-rounds", "1",
                             "--bc-teacher", "v2")
        self.assertEqual(code, 2)

    def test_v2_dagger_with_v4_runs_end_to_end(self) -> None:
        """G-745：v2 + DAgger + teacher v4 必须能跑（不是 exit 2）。

        v2 的十头标签只有 v4 能给，所以门禁对「非 v4」拒绝、对 v4 放行；
        DAgger 与 BC 共用同一条门禁，这里钉住「放行」那一半——否则研究轨根本起不来。
        """
        with tempfile.TemporaryDirectory() as folder:
            code = self._run(Path(folder), "--action-space", "v2", "--bc-teacher", "v4",
                             "--bc-episodes", "1", "--bc-epochs", "1",
                             "--dagger-rounds", "1", "--dagger-episodes", "1",
                             "--dagger-epochs", "1", "--dagger-max-steps", "3",
                             "--episodes", "0", "--eval-epochs", "0")
            self.assertEqual(code, 0, "v2 + DAgger + v4 应被放行")
            report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["actionSpace"], "v2")
            self.assertEqual(report["config"]["daggerRounds"], 1)
            self.assertEqual(report["config"]["daggerEpisodes"], 1)
            rounds = report["dagger"]
            # 第 0 轮是 BC 基线，第 1 轮才是真正的 DAgger 采样。
            self.assertEqual(len(rounds), 2)
            self.assertEqual(rounds[0]["round"], 0)
            self.assertGreater(rounds[1]["samples"], 0, "DAgger 轮必须采到样本")
            self.assertEqual(len(rounds[1]["perHeadAccuracy"]), N_HEADS,
                             "DAgger 轮也要记十头明细")

    def test_v2_bc_with_v4_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code = self._run(Path(folder), "--action-space", "v2", "--bc-teacher", "v4",
                             "--bc-episodes", "1", "--bc-epochs", "2",
                             "--episodes", "0", "--eval-epochs", "1",
                             "--eval-before-train")
            self.assertEqual(code, 0)
            report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["actionSpace"], "v2")
            self.assertEqual(report["config"]["bc"]["teacher"], "v4")
            self.assertEqual(report["config"]["bc"]["episodes"], 1)
            self.assertGreater(report["config"]["bc"]["samples"], 0)
            self.assertIsNotNone(report["config"]["bc"]["accuracy"])
            self.assertIsNotNone(report["config"]["bc"]["agreementBefore"])
            self.assertIsNotNone(report["config"]["bc"]["agreementAfter"])
            # 十头明细
            per_head = report["bc"]["perHeadAccuracy"]
            self.assertEqual(len(per_head), N_HEADS)
            self.assertEqual(report["bc"]["heads"], list(ACTION_HEADS_V2))
            self.assertIsNotNone(report["bc"]["totalWeight"])
            self.assertIsNotNone(report["bc"]["agreementBeforeExact"])
            self.assertEqual(len(report["bc"]["headAgreementAfter"]), N_HEADS)
            # --eval-before-train 在 v2 下不炸（Trainer 被 mock，记录为空数组）
            self.assertEqual(report["evaluationBefore"], [])

    def test_eval_only_counts_rounds_not_squared(self) -> None:
        """``--episodes 0 --eval-epochs 3`` 每遍跑 3 局（不是 3² = 9 局）。

        训练计划为 0 局时，评估的「每回合局数」必须是 1，让 `eval_epochs` 遍
        正好等于 `eval_epochs` 局；塞成 eval_epochs 会变成平方（第一版踩过）。
        """
        with tempfile.TemporaryDirectory() as folder:
            code = self._run(Path(folder), "--action-space", "v2", "--bc-episodes", "0",
                             "--episodes", "0", "--eval-epochs", "3")
            self.assertEqual(code, 0)
            report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["evalEpisodes"], 3)
            self.assertEqual(report["config"]["evalRounds"], 3)
            counts = [call.args[0] for call in self.trainer.evaluate.call_args_list]
            # 两种口径（采样 + 贪心）各 3 回合，每回合 1 局：3×2 次调用，不是 9 局/遍
            self.assertEqual(counts, [1] * 6)
            self.assertEqual(len(set(counts)), 1, "每回合只该跑 1 局，靠回合数凑局数")


class V1RegressionTest(unittest.TestCase):
    def test_v1_behavior_clone_result_shape(self) -> None:
        """v1 的返回键不因 v2 改造而变（旧读者仍读 accuracy/loss/totalWeight）。"""
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=8, seed=2)
        samples = [(obs_vector(), 3, 0.5), (obs_vector(), 4, 0.2)]
        result = behavior_clone(policy, samples, epochs=2)
        for key in ("loss", "accuracy", "samples", "epochs", "temperature",
                    "labelSmoothing", "teacherEntropy", "totalWeight"):
            self.assertIn(key, result)
        self.assertNotIn("perHeadAccuracy", result, "v1 不应出现 v2 的逐头字段")


if __name__ == "__main__":
    unittest.main()
