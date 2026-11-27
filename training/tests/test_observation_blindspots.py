"""观测盲区修复的回归测试（对应 ``docs/observation-blindspots.md``）。

覆盖三处「死特征 / 单侧错位」：

1. ``target_on_ground``（index 12）：原为恒 0 的死特征——协议里根本没有可信的
   ``onGround``（mineflayer 的 ``prismarine-entity`` 把远端实体的 ``onGround`` 置成构造
   默认值 ``true`` 且从不更新，直接转发会变成恒 1，同样是死特征）。现在由
   ``vy≈0 && y 未变`` 逐帧推断（``GroundEstimator``），本文件锁住推断规则与编码优先级。
2. ``step_fraction``（index 14）：训练侧口径统一到 ``episode_fraction``
   （= 本局已过时间 / 时长上限），供运行时按 ``elapsed/roundSeconds`` 复现。
3. ``self_attack_cooldown``（index 24）：统一到 ``attack_cooldown_remaining``
   （= ``clamp(0.5 - since_attack)``），供运行时按 ``lastAttackAt`` 复现。

另附一条「盲帧基线」：没有目标时，目标相关维度必须**全部**为 0（``has_target`` 是唯一
判别位），这是文档里「28/37 维同塌」结论的可执行版本。
"""

from __future__ import annotations

import unittest

import numpy as np

from gladiator_env import BotConfig, GladiatorEnv
from gladiator_env.spaces import (
    ATTACK_COOLDOWN_S,
    GROUND_CONFIRM_FRAMES,
    OBS_DIM,
    OBS_FIELDS,
    GroundEstimator,
    attack_cooldown_remaining,
    encode_observation,
    episode_fraction,
    estimate_on_ground,
)
from gladiator_env.testing import FakeLink, make_observation, make_target

TARGET_ON_GROUND = 12
STEP_FRACTION = 14
SELF_ATTACK_COOLDOWN = 24
HAS_TARGET = 13


def entity(name: str = "Glad_Bot_2", *, y: float = 65.0, vy: float = 0.0,
           x: float = 2.0, z: float = 0.0) -> dict:
    """造一个协议形态的实体（与 ``buildObservation`` 输出同构：没有 health/onGround）。"""
    return {
        "id": 2, "kind": "player", "name": name,
        "pos": {"x": x, "y": y, "z": z},
        "velocity": {"x": 0.0, "y": vy, "z": 0.0},
        "distance": (x ** 2 + z ** 2) ** 0.5,
        "heldItem": None,
    }


def frame(*entities: dict) -> dict:
    return make_observation("Glad_Bot_1", targets=list(entities))


class EstimateOnGroundTest(unittest.TestCase):
    """单帧推断的三态语义：True / False / None（不知道）。"""

    def test_falling_is_definitely_airborne(self):
        # 1.8 自由落体每 tick 约 −0.08，远超阈值；不依赖上一帧 y。
        self.assertIs(estimate_on_ground(entity(vy=-0.5)), False)
        self.assertIs(estimate_on_ground(entity(vy=-0.5), previous_y=90.0), False)

    def test_rising_is_airborne(self):
        self.assertIs(estimate_on_ground(entity(vy=0.42)), False)
        self.assertIs(estimate_on_ground(entity(vy=0.42), previous_y=65.0), False)

    def test_still_and_same_y_is_grounded(self):
        self.assertIs(estimate_on_ground(entity(vy=0.0, y=65.0), previous_y=65.0), True)
        # 浮点抖动不该翻转结论
        self.assertIs(estimate_on_ground(entity(vy=0.0, y=65.0000004), previous_y=65.0), True)

    def test_still_but_y_moved_is_not_grounded(self):
        # vy 采样为 0 但 y 变了（掉帧/低采样率下的下落）→ 不能判着地
        self.assertIs(estimate_on_ground(entity(vy=0.0, y=64.5), previous_y=65.0), False)

    def test_without_previous_y_is_unknown_not_grounded(self):
        # 单帧「vy≈0」既可能是静立也可能是跳跃顶点，必须返回「不知道」
        self.assertIsNone(estimate_on_ground(entity(vy=0.0, y=65.0)))
        self.assertIsNone(estimate_on_ground(entity(vy=0.0, y=65.0), previous_y=None))

    def test_missing_fields_are_unknown(self):
        self.assertIsNone(estimate_on_ground({}))
        self.assertIsNone(estimate_on_ground(None))
        self.assertIsNone(estimate_on_ground({"pos": {"x": 1.0}, "velocity": {"y": 0.0}}))
        self.assertIsNone(estimate_on_ground({"pos": {"y": 65.0}}))
        self.assertIsNone(estimate_on_ground({"pos": {"y": "x"}, "velocity": {"y": 0.0}}))
        self.assertIsNone(estimate_on_ground({"pos": {"y": float("nan")}, "velocity": {"y": 0.0}}))


class GroundEstimatorTest(unittest.TestCase):
    """跨帧状态机：首帧不敢断言、连续帧才认、顶点误判要滤掉。"""

    def test_first_frame_is_not_grounded(self):
        estimator = GroundEstimator()
        result = estimator.update([entity(y=65.0, vy=0.0)])
        self.assertEqual(result, {"glad_bot_2": False})

    def test_needs_consecutive_frames(self):
        """首帧只建立 y 基线（不可判定），之后连续 ``GROUND_CONFIRM_FRAMES`` 帧 True 才着地。"""
        estimator = GroundEstimator()
        self.assertEqual(GROUND_CONFIRM_FRAMES, 2)
        estimator.update([entity(y=65.0, vy=0.0)])                     # 首帧：无上一帧 y
        self.assertFalse(estimator.as_map()["glad_bot_2"])
        estimator.update([entity(y=65.0, vy=0.0)])                     # 第 1 帧可判定（计数 1）
        self.assertFalse(estimator.as_map()["glad_bot_2"])
        estimator.update([entity(y=65.0, vy=0.0)])                     # 计数 2 → 着地
        self.assertTrue(estimator.as_map()["glad_bot_2"])

    def test_jump_apex_does_not_flip_to_grounded(self):
        """顶点那一帧 vy≈0、y 几乎没动——但它只能维持 1 帧，靠 confirm 帧数挡掉。"""
        estimator = GroundEstimator(confirm_frames=2)
        for _ in range(3):
            estimator.update([entity(y=65.0, vy=0.0)])
        self.assertTrue(estimator.as_map()["glad_bot_2"], "地面站定后应判着地")
        # 起跳：上升 → 顶点（vy=0，且这一帧 y 只挪了 0.0005）→ 下落
        estimator.update([entity(y=66.4, vy=0.42)])
        self.assertFalse(estimator.as_map()["glad_bot_2"])
        estimator.update([entity(y=66.4005, vy=0.0)])
        self.assertFalse(estimator.as_map()["glad_bot_2"], "跳跃顶点不许判成着地")
        estimator.update([entity(y=66.3, vy=-0.08)])
        self.assertFalse(estimator.as_map()["glad_bot_2"])

    def test_landing_recovers_after_confirm_frames(self):
        estimator = GroundEstimator()
        estimator.update([entity(y=65.0, vy=0.0)])
        estimator.update([entity(y=65.0, vy=0.0)])
        estimator.update([entity(y=66.0, vy=0.42)])
        self.assertFalse(estimator.as_map()["glad_bot_2"])
        estimator.update([entity(y=66.0, vy=0.0)])     # 落地首帧（计数 1）
        self.assertFalse(estimator.as_map()["glad_bot_2"])
        estimator.update([entity(y=66.0, vy=0.0)])     # 计数 2 → 着地
        self.assertTrue(estimator.as_map()["glad_bot_2"])

    def test_same_frame_is_idempotent(self):
        """同一 step 内学习方与策略对手各编码一次：只能推进一次，否则计数被推快。"""
        estimator = GroundEstimator()
        estimator.update([entity(y=65.0, vy=0.0)], frame=7)   # 基线
        estimator.update([entity(y=65.0, vy=0.0)], frame=7)   # 同一帧重复调用 → 不推进
        estimator.update([entity(y=65.0, vy=0.0)], frame=7)
        self.assertFalse(estimator.as_map()["glad_bot_2"], "重复调用不该凑满 confirm 帧")
        estimator.update([entity(y=65.0, vy=0.0)], frame=8)   # 计数 1
        estimator.update([entity(y=65.0, vy=0.0)], frame=9)   # 计数 2 → 着地
        self.assertTrue(estimator.as_map()["glad_bot_2"])

    def test_reset_clears_history(self):
        estimator = GroundEstimator()
        for _ in range(3):
            estimator.update([entity(y=65.0, vy=0.0)])
        self.assertTrue(estimator.as_map()["glad_bot_2"])
        estimator.reset()
        self.assertEqual(estimator.as_map(), {})
        estimator.update([entity(y=65.0, vy=0.0)])
        self.assertFalse(estimator.as_map()["glad_bot_2"], "reset 后第一帧不能凭空断言着地")

    def test_ignores_non_player_entities_and_names(self):
        estimator = GroundEstimator()
        result = estimator.update([
            {"kind": "mob", "name": "", "pos": {"y": 65.0}, "velocity": {"y": 0.0}},
            {"kind": "player", "pos": {"y": 65.0}, "velocity": {"y": 0.0}},
        ])
        self.assertEqual(result, {}, "没有名字的实体不参与推断")


class TargetOnGroundEncodingTest(unittest.TestCase):
    """编码优先级：环境推断 > 协议真 bool > 单帧推断 > 0。"""

    def test_env_estimate_wins(self):
        ground = frame(entity(y=65.0, vy=0.0), entity(name="Glad_Bot_3", x=6.0, vy=-0.4))
        vector = encode_observation(ground, "Glad_Bot_1",
                                    opponent_ground={"glad_bot_2": True})
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 1.0)
        # 名字大小写不敏感
        vector = encode_observation(ground, "Glad_Bot_1",
                                    opponent_ground={"Glad_Bot_2": False})
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0)

    def test_explicit_protocol_bool_is_used_when_no_estimate(self):
        target = entity(y=65.0, vy=0.0)
        target["onGround"] = True
        vector = encode_observation(frame(target), "Glad_Bot_1")
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 1.0)
        target["onGround"] = False
        vector = encode_observation(frame(target), "Glad_Bot_1")
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0)

    def test_non_bool_onGround_is_not_trusted(self):
        """``onGround: 1`` / ``"true"`` 一律不采信：prismarine 的默认值就是「看起来对」的假值，
        采信非 bool 字段等于给死特征开后门。"""
        for bogus in (1, "true", "yes", [1]):
            target = entity(y=65.0, vy=0.0)
            target["onGround"] = bogus
            vector = encode_observation(frame(target), "Glad_Bot_1")
            self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0, f"不该采信 {bogus!r}")

    def test_falls_back_to_single_frame_inference(self):
        vector = encode_observation(frame(entity(vy=-0.5)), "Glad_Bot_1")
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0)
        # 单帧 vy≈0 无历史 → 不知道 → 0（宁可判未着地，也不凭空断言）
        vector = encode_observation(frame(entity(vy=0.0)), "Glad_Bot_1")
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0)

    def test_no_target_keeps_zero(self):
        vector = encode_observation(frame(), "Glad_Bot_1")
        self.assertEqual(float(vector[HAS_TARGET]), 0.0)
        self.assertEqual(float(vector[TARGET_ON_GROUND]), 0.0)


class StepFractionAndCooldownTest(unittest.TestCase):
    """两处「运行时单侧错位」的规范口径（运行时按同一公式复现）。"""

    def test_episode_fraction_range(self):
        self.assertEqual(episode_fraction(0, 600), 0.0)
        self.assertAlmostEqual(episode_fraction(300, 600), 0.5, places=6)
        self.assertEqual(episode_fraction(600, 600), 1.0)
        self.assertEqual(episode_fraction(900, 600), 1.0, "超出上限要 clip")
        self.assertEqual(episode_fraction(10, 0), 0.0, "无上限时保持 0（不除零）")
        self.assertEqual(episode_fraction(10, -5), 0.0)
        self.assertEqual(episode_fraction(None, 600), 0.0)
        self.assertEqual(episode_fraction(10, None), 0.0)

    def test_episode_fraction_matches_elapsed_over_limit(self):
        """口径自证：steps/max_steps 与 elapsed/limit 是同一个量（步长固定时等价）。"""
        control_hz, max_steps = 20.0, 600
        steps = 123
        self.assertAlmostEqual(
            episode_fraction(steps, max_steps),
            min(1.0, (steps / control_hz) / (max_steps / control_hz)),
            places=9)

    def test_attack_cooldown_remaining(self):
        self.assertAlmostEqual(attack_cooldown_remaining(0.0), ATTACK_COOLDOWN_S, places=6)
        self.assertAlmostEqual(attack_cooldown_remaining(0.25), 0.25, places=6)
        self.assertEqual(attack_cooldown_remaining(0.5), 0.0)
        self.assertEqual(attack_cooldown_remaining(5.0), 0.0)
        self.assertEqual(attack_cooldown_remaining(-1.0), ATTACK_COOLDOWN_S, "clock 回拨要 clip")
        self.assertEqual(attack_cooldown_remaining(None), 0.0, "不知道 → 视为可打")
        self.assertEqual(attack_cooldown_remaining(float("nan")), 0.0)

    def test_encoding_uses_the_same_normalizers(self):
        obs = frame(entity(y=65.0, vy=0.0))
        vector = encode_observation(obs, "Glad_Bot_1", step_fraction=0.5,
                                    attack_cooldown_remaining=0.25)
        self.assertAlmostEqual(float(vector[STEP_FRACTION]), 0.5, places=6)
        self.assertAlmostEqual(float(vector[SELF_ATTACK_COOLDOWN]), 0.5, places=6)

    def test_fields_are_where_the_doc_says(self):
        self.assertEqual(OBS_FIELDS[STEP_FRACTION], "step_fraction")
        self.assertEqual(OBS_FIELDS[SELF_ATTACK_COOLDOWN], "self_attack_cooldown")
        self.assertEqual(OBS_FIELDS[TARGET_ON_GROUND], "target_on_ground")
        self.assertEqual(len(OBS_FIELDS), OBS_DIM)


class BlindFrameBaselineTest(unittest.TestCase):
    """盲帧基线（``docs/observation-blindspots.md`` §0）：目标维必须一起塌成 0。"""

    def test_no_target_zeros_every_target_dimension(self):
        vector = encode_observation(frame(), "Glad_Bot_1")
        # G-731：37/38（朝向误差 sin/cos）也是「目标维」——没有目标时必须一起塌成 0，
        # 否则网络会把「不知道对手在哪」读成「正对着某个方向」。
        for index in (7, 8, 9, 10, 11, 12, 13, 19, 20, 22, 37, 38):
            self.assertEqual(float(vector[index]), 0.0, f"index {index} 应为 0")
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_blind_frame_count_is_stable(self):
        """锁住「盲帧有多少维是 0」：改观测定义时这条会红，提醒同步文档。

        G-711 起是 27/37（文档写 28，make_observation 的自身帧 onGround=True 少 1）；
        G-731 追加 37/38 两维目标维，盲帧时同样为 0 → 29/39。
        """
        vector = encode_observation(frame(), "Glad_Bot_1")
        zeros = int(np.sum(vector == 0.0))
        self.assertEqual(zeros, 29,
                         "make_observation 的自身帧 onGround=True，故比文档里的 30 少 1")

    def test_target_health_defaults_to_full_when_no_opponent_health(self):
        vector = encode_observation(frame(entity()), "Glad_Bot_1")
        self.assertEqual(float(vector[11]), 1.0, "读不到对手血量时回落满血（不是 0）")


def build_env(observation_0: dict, observation_1: dict) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1", observation_0), FakeLink("Glad_Bot_2", observation_1)]
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        single_agent=True,
        opponent_mode="scripted",
        control_hz=200.0,
        action_ticks=1,
        max_steps=50,
    )
    return env, links


class EnvWiringTest(unittest.TestCase):
    """env 侧接线：``_encode_all`` 必须把推断结果喂给 ``target_on_ground``。"""

    def _obs(self, y: float, vy: float) -> dict:
        return make_observation(
            "Glad_Bot_1", y=y,
            targets=[{"id": 2, "kind": "player", "name": "Glad_Bot_2",
                      "pos": {"x": 2.0, "y": y, "z": 0.0},
                      "velocity": {"x": 0.0, "y": vy, "z": 0.0},
                      "distance": 2.0}])

    def test_grounded_opponent_reaches_index_12(self):
        env, _links = build_env(self._obs(65.0, 0.0), self._obs(65.0, 0.0))
        try:
            env.reset()                       # 帧 0：只建立 y 基线
            self.assertEqual(float(env._encode_all()["bot_0"][TARGET_ON_GROUND]), 0.0)
            env.step(0)                       # 帧 1：可判定（计数 1）
            self.assertEqual(float(env._encode_all()["bot_0"][TARGET_ON_GROUND]), 0.0)
            env.step(0)                       # 帧 2：计数 2 → 着地
            self.assertEqual(float(env._encode_all()["bot_0"][TARGET_ON_GROUND]), 1.0)
        finally:
            env.close()

    def test_airborne_opponent_is_zero(self):
        env, _links = build_env(self._obs(65.0, -0.4), self._obs(65.0, -0.4))
        try:
            env.reset()
            env._encode_all()
            self.assertEqual(float(env._encode_all()["bot_0"][TARGET_ON_GROUND]), 0.0)
        finally:
            env.close()

    def test_step_fraction_grows_with_steps(self):
        env, _links = build_env(self._obs(65.0, 0.0), self._obs(65.0, 0.0))
        try:
            env.reset()
            first = float(env._encode_all()["bot_0"][STEP_FRACTION])
            env.step(0)
            second = float(env._encode_all()["bot_0"][STEP_FRACTION])
            self.assertEqual(first, 0.0)
            self.assertGreater(second, first)
            self.assertAlmostEqual(second, 1.0 / 50.0, places=6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
