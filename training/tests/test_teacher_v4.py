"""老师 v4（1.8 招式表 / v2 十头动作）的单测。

对应 G-707 的 B 段：v4 必须严格按 ``docs/knowledge/pvp-1.8-mechanics.md`` §3 出手，
并且**互斥**地使用两条对刀路线：

- 有剑 → ``block_hit``（防砍：按住 W + 举剑），从不走 wtap；
- 无剑 → ``wtap``（``release_w``），**永不按右键**（没剑时右键只会拿 ITEM_MISSING）。

这里全部离线（只喂观测向量），所以「老师在想什么」可以被逐条断言，而不是靠真机录像反推。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from gladiator_env.spaces import (
    ACTION_HEADS_V2,
    LOOK_YAW_V2,
    MULTI_DISCRETE_V2,
    SPRINT_RESET_VALUES_V2,
    decode_intent,
)
from gladiator_env.testing import make_observation, make_target
from gladiator_env.spaces import encode_observation
from gladiator_train.demos import (
    TeacherV4,
    TeacherV4State,
    heuristic_action_v4,
    make_teacher,
    reset_teacher_v4,
    v4_intent_of,
)

REACH = 3.99


def obs(
    distance: float = 2.5,
    *,
    health: float = 20.0,
    target_health: float = 20.0,
    vy: float = 0.0,
    cooldown: float = 0.0,
    since_attack: float = 5.0,
    has_target: bool = True,
) -> np.ndarray:
    """构造一帧 25 维观测（真机同口径：距离走 entities、血量走 opponent_health）。"""
    targets = ([make_target(2, "Foe", x=distance, z=0.0, distance=distance)]
               if has_target else [])
    raw = make_observation("Glad_Bot_1", health=health, targets=targets)
    return encode_observation(
        raw,
        "Glad_Bot_1",
        reach=REACH,
        opponent_health=target_health,
        since_attack=since_attack,
        self_velocity_y=vy,
        attack_cooldown_remaining=cooldown,
    )


def decide(vector, teacher: TeacherV4, meta: dict | None = None):
    """跑一帧，返回 ``(intent, action, reason)``——三种视图方便分别断言。"""
    merged = dict(meta or {})
    action = teacher(vector, merged)
    return v4_intent_of(action), action, merged.get("reason")


class ActionShapeTest(unittest.TestCase):
    def test_action_is_ten_heads_and_decodable(self) -> None:
        teacher = TeacherV4()
        action = teacher(obs(2.5))
        self.assertIsInstance(action, list)
        self.assertEqual(len(action), len(ACTION_HEADS_V2))
        self.assertEqual(len(action), 10)
        # 每一档都必须落在 MultiDiscrete 的合法范围内（越界会被 env 直接拒收）
        for value, size in zip(action, MULTI_DISCRETE_V2):
            self.assertGreaterEqual(int(value), 0)
            self.assertLess(int(value), int(size))
        intent = decode_intent(action)
        self.assertIn(intent["move"], ("idle", "forward", "back", "left", "right"))

    def test_make_teacher_returns_v4(self) -> None:
        teacher = make_teacher("v4")
        self.assertIsInstance(teacher, TeacherV4)
        self.assertTrue(teacher.is_v2)
        self.assertEqual(teacher.action_space, "v2")
        # env 的 opponent_policy 约定：act() 返回动作向量
        self.assertEqual(len(teacher.act(obs(2.5))), 10)


class NoTargetTest(unittest.TestCase):
    def test_no_target_is_idle(self) -> None:
        teacher = TeacherV4()
        intent, _action, reason = decide(obs(2.5, has_target=False), teacher)
        self.assertEqual(reason, "no_target")
        self.assertEqual(intent["move"], "idle")
        self.assertFalse(intent["attack"])


class SwordModeTest(unittest.TestCase):
    """有剑：只用 block_hit（防砍），且必须与出手同帧。"""

    def test_block_hit_inside_three_blocks(self) -> None:
        teacher = TeacherV4(has_sword=True)
        intent, action, reason = decide(obs(2.0), teacher)
        self.assertEqual(reason, "block_hit")
        self.assertEqual(intent["sprintReset"], "block_hit")
        self.assertEqual(intent["move"], "forward")
        self.assertTrue(intent["attack"])
        self.assertIsNone(intent["use"], "举剑走 sprint_reset，不走 use 头")
        self.assertEqual(action[5], 0, "block_hit 不该同时带 jump")

    def test_block_hit_never_when_far(self) -> None:
        teacher = TeacherV4(has_sword=True)
        _intent, _action, reason = decide(obs(3.5), teacher)
        self.assertNotEqual(reason, "block_hit")

    def test_sword_mode_never_emits_release_w(self) -> None:
        teacher = TeacherV4(has_sword=True)
        seen = set()
        for distance in (1.5, 2.2, 2.9, 3.2, 3.8, 4.5, 5.5, 7.0):
            for cooldown in (0.0, 0.4):
                intent, _action, _reason = decide(obs(distance, cooldown=cooldown,
                                                      since_attack=0.1 if cooldown else 5.0),
                                                  teacher)
                seen.add(intent.get("sprintReset"))
        self.assertNotIn("release_w", seen, "有剑时不该用 wtap（视频明确说不要混用）")


class NoSwordTest(unittest.TestCase):
    """无剑：只用 wtap 的两种用法，永不按右键。"""

    def test_wtap_keep_combo_after_hit(self) -> None:
        teacher = TeacherV4(has_sword=False)
        decide(obs(2.5, target_health=20.0), teacher)          # 第一帧只建立基线
        intent, _action, reason = decide(obs(2.5, target_health=18.4, cooldown=0.4,
                                             since_attack=0.1), teacher)
        self.assertEqual(reason, "wtap_keep_combo")
        self.assertEqual(intent["sprintReset"], "release_w")
        self.assertEqual(intent["move"], "forward")
        self.assertTrue(intent["attack"], "命中后的 wtap 要立刻接一刀")

    def test_wtap_reduce_kb_after_being_hit(self) -> None:
        teacher = TeacherV4(has_sword=False)
        decide(obs(2.5, health=20.0), teacher)
        intent, _action, reason = decide(obs(2.5, health=17.2), teacher)
        self.assertEqual(reason, "wtap_reduce_kb")
        self.assertEqual(intent["sprintReset"], "release_w")
        self.assertFalse(intent["attack"], "受击后的 wtap 只是缩小自己的 KB，不追刀")

    def test_no_sword_never_presses_right_click(self) -> None:
        teacher = TeacherV4(has_sword=False)
        for distance in (1.5, 2.4, 3.0, 4.5, 6.0):
            intent, _action, _reason = decide(obs(distance), teacher)
            self.assertIsNone(intent["use"])
            self.assertNotEqual(intent.get("sprintReset"), "block_hit")

    def test_sword_flag_switch_on_same_state(self) -> None:
        """同一个 state 连续两帧：有剑→block_hit，无剑→不再是 block_hit。"""
        state = TeacherV4State()
        first = heuristic_action_v4(obs(2.0), {"has_sword": True, "state": state}, state)
        self.assertEqual(v4_intent_of(first).get("sprintReset"), "block_hit")
        second = heuristic_action_v4(obs(2.0), {"has_sword": False, "state": state}, state)
        self.assertNotEqual(v4_intent_of(second).get("sprintReset"), "block_hit")


class JumpCritTest(unittest.TestCase):
    """jump_crit：只在 ``vy<0`` 且冷却已好时出手（同帧 jump+attack）。"""

    def test_falling_and_ready_triggers_jump_attack(self) -> None:
        teacher = TeacherV4(has_sword=False)
        intent, action, reason = decide(obs(2.0, vy=-1.2), teacher)
        self.assertEqual(reason, "jump_crit")
        self.assertTrue(intent["jump"])
        self.assertTrue(intent["attack"])
        self.assertEqual(action[5], 1, "jump 头必须为 1（同帧组合，G-701）")

    def test_not_falling_no_jump_crit(self) -> None:
        teacher = TeacherV4(has_sword=False)
        _intent, _action, reason = decide(obs(2.0, vy=1.0), teacher)
        self.assertNotEqual(reason, "jump_crit")

    def test_cooldown_gate_blocks_jump_crit(self) -> None:
        teacher = TeacherV4(has_sword=False)
        _intent, action, reason = decide(
            obs(2.0, vy=-1.2, cooldown=0.45, since_attack=0.05), teacher)
        self.assertNotEqual(reason, "jump_crit")
        self.assertEqual(action[5], 0, "冷却没好时不该起跳")

    def test_jump_crit_respects_distance_window(self) -> None:
        teacher = TeacherV4(has_sword=False)
        _intent, _action, reason = decide(obs(3.0, vy=-1.2), teacher)
        self.assertNotEqual(reason, "jump_crit")

    def test_jump_arm_prepares_the_falling_window(self) -> None:
        """着地 + 冷却好 + 在带内 → 先只起跳（jump_arm），下一帧 vy<0 才是 jump_crit。

        这条是 2026-11-22 真机 12 局发现的：只按 ``vy<0`` 触发的话，vy 永远是 0
        （没人跳过），规则 0 次生效。起跳预备把这条规则「接上电」。
        """
        teacher = TeacherV4(has_sword=False)
        intent, action, reason = decide(obs(2.0), teacher)          # vy=0、着地
        self.assertEqual(reason, "jump_arm")
        self.assertTrue(intent["jump"])
        self.assertFalse(intent["attack"], "起跳那一帧先不出手，等下落窗口")
        self.assertEqual(action[5], 1)
        # 带着同一个 state 进入下一帧：正在下落 + 冷却还好的话立刻同帧 jump+attack
        intent2, _action2, reason2 = decide(obs(2.0, vy=-1.5), teacher)
        self.assertEqual(reason2, "jump_crit")
        self.assertTrue(intent2["attack"])

    def test_jump_arm_is_throttled(self) -> None:
        """1 秒内不重复起跳（否则会「一直跳、一直等不到下落」）。"""
        teacher = TeacherV4(has_sword=False)
        first = decide(obs(2.0), teacher)[2]
        second = decide(obs(2.0), teacher)[2]
        self.assertEqual(first, "jump_arm")
        self.assertNotEqual(second, "jump_arm")


class SpacingTest(unittest.TestCase):
    """spacing_hold：目标带 2.6–3.3 格，太近 S、太远 W。"""

    def test_too_far_moves_forward_with_sprint(self) -> None:
        teacher = TeacherV4(has_sword=False)
        intent, _action, reason = decide(obs(6.8), teacher)
        self.assertEqual(reason, "spacing_forward")
        self.assertEqual(intent["move"], "forward")
        self.assertTrue(intent["sprint"], "超过 2.5 格要疾跑才追得上")

    def test_too_close_moves_back(self) -> None:
        teacher = TeacherV4(has_sword=False)
        # 1.2 格：既在 2.6 的带内、又低于跳劈下界（1.4），所以是纯粹的「太近就退」
        intent, _action, reason = decide(obs(1.2, target_health=0.0), teacher)
        self.assertEqual(reason, "spacing_back")
        self.assertEqual(intent["move"], "back")


class StrafeTest(unittest.TestCase):
    """strafe：接战前 4–6 格用 poke（慢换向），交战线内用 combo（快换向）。"""

    def test_strafe_poke_before_engagement(self) -> None:
        teacher = TeacherV4(has_sword=False)
        intent, _action, reason = decide(obs(5.0), teacher)
        self.assertEqual(reason, "strafe_poke")
        self.assertIn(intent["move"], ("left", "right"))

    def test_strafe_alternates_direction(self) -> None:
        teacher = TeacherV4(has_sword=False)
        first = v4_intent_of(teacher(obs(5.0)))["move"]
        other = "right" if first == "left" else "left"
        # 换向节拍是 0.4–0.8s（frame_seconds 缺省 0.2 → 2–4 帧），这里连跑 6 帧
        moves = [v4_intent_of(teacher(obs(5.0)))["move"] for _ in range(6)]
        self.assertIn(other, moves, f"6 帧内应该换过向（实际 {moves}）")

    def test_strafe_combo_in_band(self) -> None:
        teacher = TeacherV4(has_sword=False)
        # 2.8 格在带内、够不到出手（in_reach 是 3.99，但冷却没好 → 不砍）→ 横移
        intent, _action, reason = decide(obs(2.8, cooldown=0.45, since_attack=0.05), teacher)
        self.assertEqual(reason, "strafe_combo")
        self.assertIn(intent["move"], ("left", "right"))


class StateTest(unittest.TestCase):
    def test_reset_episode_archives_and_clears(self) -> None:
        teacher = TeacherV4(has_sword=False)
        decide(obs(2.5), teacher)
        self.assertTrue(teacher.reasons)
        teacher.reset_episode()
        self.assertEqual(teacher.reasons, {})
        self.assertEqual(len(teacher.episode_reasons), 1)

    def test_default_module_state_can_be_reset(self) -> None:
        heuristic_action_v4(obs(2.5))          # 用模块级默认状态跑一帧
        reset_teacher_v4()                     # 不该抛
        self.assertEqual(TeacherV4State().frame, 0)

    def test_sprint_reset_values_cover_all_three(self) -> None:
        self.assertEqual(SPRINT_RESET_VALUES_V2, ("none", "release_w", "block_hit"))

class YawCorrectionTest(unittest.TestCase):
    """G-731：老师 v4 要输出「朝目标转」的 look 头，而不是恒 0。

    真机取证（2026-11-28）：``dist=2.8 yaw=-108.1 bearing=-71.9 yawErr=36.3``——
    被击退后朝向被甩开 36°，而旧老师从不转头（``dyaw`` 恒 0），蒸馏出来的策略自然也不会转。
    这些用例把「老师会转头」钉住：分档、死区、跨 ±180 的最短路、无目标/无朝向时不乱转。
    """

    @staticmethod
    def obs_at(yaw_deg: float, tx: float, tz: float, *, has_target: bool = True) -> np.ndarray:
        """目标在 ``(tx, tz)``、自身 yaw 为 ``yaw_deg``（度）的一帧观测。"""
        targets = [make_target(2, "Foe", x=tx, z=tz)] if has_target else []
        raw = make_observation("Glad_Bot_1", targets=targets)
        raw["self"]["yaw"] = math.radians(yaw_deg)
        return encode_observation(raw, "Glad_Bot_1", reach=REACH)

    def dyaw_of(self, yaw_deg: float, tx: float, tz: float, **kwargs) -> float:
        teacher = TeacherV4()
        intent, _action, _reason = decide(self.obs_at(yaw_deg, tx, tz, **kwargs), teacher)
        return float(intent["look"]["dyaw"])

    def test_facing_target_keeps_dyaw_zero(self) -> None:
        """正对目标（yaw=0、对手在 +Z）→ 不转，别抖。"""
        self.assertEqual(self.dyaw_of(0.0, 0.0, 4.0), 0.0)

    def test_back_target_turns_toward_it(self) -> None:
        """背身（对手在身后偏右，yawErr≈−135°）→ 输出最大一档反向转。"""
        dyaw = self.dyaw_of(0.0, 3.0, -3.0)
        self.assertNotEqual(dyaw, 0.0, "背对目标却不转头：G-731 的核心行为丢了")
        self.assertEqual(dyaw, -30.0)
        # 转完这一档之后 |yawErr| 必须变小（−135° → −105°），否则就是转反了。
        self.assertLess(abs(-135.0 - dyaw), 135.0)

    def test_seam_uses_shortest_path_not_sign_flip(self) -> None:
        """跨 ±180：+170° 与 −170° 的误差必须给出**相反**的转向（各走 10° 那一侧）。"""
        plus = self.dyaw_of(170.0, 0.0, 4.0)      # yawErr ≈ −170°
        minus = self.dyaw_of(-170.0, 0.0, 4.0)    # yawErr ≈ +170°
        self.assertEqual(plus, -30.0)
        self.assertEqual(minus, 30.0)
        self.assertEqual(plus, -minus, "跨缝时转向符号必须镜像，不能同向")

    def test_deadzone_is_silent_but_outside_turns(self) -> None:
        """|yawErr| ≤ 20° 不转；25° 就要转（死区边界）。"""
        self.assertEqual(self.dyaw_of(15.0, 0.0, 4.0), 0.0)
        self.assertEqual(self.dyaw_of(-15.0, 0.0, 4.0), 0.0)
        # yaw=+25°（朝 −X 偏一点）、对手在 +Z → yawErr = −25° → 往负方向转。
        self.assertNotEqual(self.dyaw_of(25.0, 0.0, 4.0), 0.0)
        self.assertEqual(self.dyaw_of(25.0, 0.0, 4.0), -30.0)

    def test_no_target_never_turns(self) -> None:
        """没有目标 → 观测 37/38 是 (0,0) → 不许乱转。"""
        self.assertEqual(self.dyaw_of(120.0, 3.0, 3.0, has_target=False), 0.0)

    def test_old_37_dim_vector_does_not_crash(self) -> None:
        """旧 37 维向量（没有 37/38 位）也要能跑：没有朝向信息就不转。"""
        legacy = self.obs_at(0.0, 3.0, -3.0)[:37]
        teacher = TeacherV4()
        intent, action, _reason = decide(legacy, teacher)
        self.assertEqual(len(action), 10)
        self.assertEqual(float(intent["look"]["dyaw"]), 0.0)

    def test_intent_reflects_the_correction(self) -> None:
        """``meta["intent"]`` 必须是修正**之后**的动作（复盘不能看到旧朝向）。"""
        teacher = TeacherV4()
        meta: dict = {}
        action = teacher(self.obs_at(0.0, 3.0, -3.0), meta)
        self.assertEqual(action[1], LOOK_YAW_V2.index(-30.0))
        self.assertEqual(float(meta["intent"]["look"]["dyaw"]), -30.0)
        self.assertEqual(action, list(teacher.state.last_action))


if __name__ == "__main__":
    unittest.main()
