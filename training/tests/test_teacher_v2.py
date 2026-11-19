"""老师 v2（``heuristic_action_v2``）的六条规则与退化情形。

背景：v1 的四条静态规则已经被 DAgger 克隆到一致率 1.0（见 docs/training-log.md），
说明老师没有「策略还学不会」的余量。v2 在 v1 之上加时序行为（间距 / 鱼竿 / 金苹果 /
暴击跳 / w-tap / 低血格挡），这些规则都需要跨帧记忆，所以用合成观测按帧驱动来验证。

覆盖：
- v1 默认行为不变（``TEACHER_VERSION == "v1"``，``heuristic_action`` 不受 meta 影响）；
- 间距控制的后撤闩锁（进 2.2 退 2.6）与「血量不落后就不后撤」；
- 鱼竿的窗口 / 3 秒冷却 / 放完切回剑；
- 金苹果的血量与距离双条件；
- 暴击跳的两帧节拍（jump → attack）与 w-tap 的 sprint off；
- 低血格挡；
- 退化：无目标 / 全零观测 / 未知版本号。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, OBS_FIELDS  # noqa: E402
from gladiator_train import demos  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    ACTION_APPROACH,
    ACTION_ATTACK,
    ACTION_BLOCK,
    ACTION_EAT,
    ACTION_IDLE,
    ACTION_JUMP,
    ACTION_RETREAT,
    ACTION_SWITCH_SWORD,
    ACTION_USE_ROD,
    TEACHER_VERSION,
    TeacherV2,
    collect_demos,
    heuristic_action,
    heuristic_action_v2,
    make_teacher,
    teacher_target,
)


def obs(
    *,
    health: float = 20.0,
    target_health: float = 20.0,
    distance: float = 2.0,
    in_reach: float | None = None,
    self_ground: float = 1.0,
    target_ground: float = 1.0,
    gapples: float = 3.0,
    since_attack: float = 5.0,
    has_target: float = 1.0,
) -> np.ndarray:
    """按字段名填 23 维观测（不依赖下标数字）。

    ``gapples`` 与 ``since_attack`` 用**原始量**（个数 / 秒），这里按编码口径归一化。
    """
    vector = np.zeros(OBS_DIM, dtype=np.float32)
    vector[OBS_FIELDS.index("self_health")] = health / 20.0
    vector[OBS_FIELDS.index("self_on_ground")] = self_ground
    vector[OBS_FIELDS.index("target_health")] = target_health / 20.0
    vector[OBS_FIELDS.index("target_distance")] = distance / 32.0
    vector[OBS_FIELDS.index("target_on_ground")] = target_ground
    vector[OBS_FIELDS.index("has_target")] = has_target
    vector[OBS_FIELDS.index("self_gapples")] = min(gapples, 5.0) / 5.0
    vector[OBS_FIELDS.index("self_since_attack")] = min(since_attack, 5.0) / 5.0
    vector[OBS_FIELDS.index("in_reach")] = (
        (1.0 if distance <= 3.99 else 0.0) if in_reach is None else in_reach)
    return vector


class TeacherVersionTest(unittest.TestCase):
    def test_default_version_is_v1(self):
        self.assertEqual(TEACHER_VERSION, "v1")
        self.assertIs(make_teacher(), heuristic_action)
        self.assertIsInstance(make_teacher("v2"), TeacherV2)

    def test_v1_behaviour_unchanged(self):
        """v1 仍是四条静态规则：射程内就砍，不看血量/苹果/鱼竿。"""
        self.assertEqual(heuristic_action(obs(distance=2.0, in_reach=1.0)), ACTION_ATTACK)
        self.assertEqual(heuristic_action(obs(health=5.0, distance=6.0)), ACTION_APPROACH)
        # 3.5 格：不在 reach 内、距离 < 4，v1 到这一步才看血量。
        self.assertEqual(heuristic_action(obs(health=5.0, distance=3.5, in_reach=0.0)),
                         ACTION_EAT)

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            make_teacher("v3")


class SpacingTest(unittest.TestCase):
    """规则 1：距离 < 2.2 且血量落后 → 后撤；退到 2.6 才解除。"""

    def test_retreats_when_out_traded(self):
        teacher = TeacherV2(rod_ready=False)
        action = teacher(obs(health=8.0, target_health=18.0, distance=2.0))
        self.assertEqual(action, ACTION_RETREAT)

    def test_retreat_latches_until_release_distance(self):
        teacher = TeacherV2(rod_ready=False)
        self.assertEqual(teacher(obs(health=8.0, target_health=18.0, distance=2.0)),
                         ACTION_RETREAT)
        # 2.2~2.6 之间还在闩锁里：继续退，不要急着贴回去。
        self.assertEqual(teacher(obs(health=8.0, target_health=18.0, distance=2.45)),
                         ACTION_RETREAT)
        # 退到 2.6 以上：闩锁解除，重新压上（此时应该挥刀，因为仍在 reach 内）。
        self.assertNotEqual(teacher(obs(health=8.0, target_health=18.0, distance=2.8)),
                            ACTION_RETREAT)

    def test_no_retreat_when_health_not_behind(self):
        teacher = TeacherV2(rod_ready=False)
        action = teacher(obs(health=18.0, target_health=10.0, distance=2.0))
        self.assertNotEqual(action, ACTION_RETREAT)


class RodTest(unittest.TestCase):
    """规则 2：2.5–4.2 格、鱼竿可用、距上次释放 > 3 秒。"""

    def test_uses_rod_in_window(self):
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(distance=3.5)), ACTION_USE_ROD)

    def test_switches_back_to_sword_next_frame(self):
        teacher = TeacherV2()
        teacher(obs(distance=3.5))
        self.assertEqual(teacher(obs(distance=3.5)), ACTION_SWITCH_SWORD)

    def test_respects_three_second_cooldown(self):
        teacher = TeacherV2()
        actions = []
        probe = obs(distance=3.5)
        for _ in range(45):
            actions.append(teacher(probe))
        rods = [index for index, action in enumerate(actions) if action == ACTION_USE_ROD]
        self.assertEqual(rods[0], 0)
        # 10Hz 控制：第 31 帧才过 3 秒（30 × 0.1 = 3.0 不算「超过」）。
        self.assertEqual(rods[1], 31, f"第二次放鱼竿在第 {rods[1]} 帧（应为 31）")

    def test_out_of_window_falls_through(self):
        teacher = TeacherV2()
        # 4.5 格超出鱼竿窗口：应该走「贴近」，不是放竿。
        self.assertNotEqual(teacher(obs(distance=4.5)), ACTION_USE_ROD)

    def test_rod_unavailable_via_meta(self):
        teacher = TeacherV2()
        action = teacher(obs(distance=3.5), {"rod_ready": False})
        self.assertNotEqual(action, ACTION_USE_ROD)


class GappleTest(unittest.TestCase):
    """规则 3：血量 ≤ 12 且距离 > 3 且背包有苹果。"""

    def test_eats_when_low_and_far(self):
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(health=10.0, target_health=18.0, distance=4.1)),
                         ACTION_EAT)

    def test_does_not_eat_too_close(self):
        teacher = TeacherV2()
        self.assertNotEqual(teacher(obs(health=10.0, target_health=18.0, distance=2.8)),
                            ACTION_EAT)

    def test_does_not_eat_without_gapple(self):
        teacher = TeacherV2()
        self.assertNotEqual(teacher(obs(health=10.0, target_health=18.0, distance=4.1,
                                        gapples=0.0)), ACTION_EAT)


class CritJumpAndWTapTest(unittest.TestCase):
    """规则 4/5：先起跳再补刀（两帧），挥刀后的下一帧只前进不疾跑。

    注意取 2.0 格而不是 2.5：2.5 已经进了鱼竿窗口（规则 2 优先于规则 4），
    那时老师会先放竿而不是起跳。
    """

    def test_jump_then_attack(self):
        teacher = TeacherV2()
        probe = obs(distance=2.0, in_reach=1.0)
        first = teacher(probe)
        second = teacher(probe)
        self.assertEqual(first, ACTION_JUMP)
        self.assertEqual(second, ACTION_ATTACK)

    def test_no_jump_when_target_airborne(self):
        teacher = TeacherV2()
        action = teacher(obs(distance=2.0, in_reach=1.0, target_ground=0.0))
        self.assertNotEqual(action, ACTION_JUMP)

    def test_no_jump_when_self_airborne(self):
        teacher = TeacherV2()
        action = teacher(obs(distance=2.0, in_reach=1.0, self_ground=0.0))
        self.assertNotEqual(action, ACTION_JUMP)

    def test_wtap_after_attack_reports_sprint_off(self):
        teacher = TeacherV2()
        probe = obs(distance=2.0, in_reach=1.0)
        sequence = []
        for _ in range(4):
            meta: dict = {}
            sequence.append((teacher(probe, meta), meta["sprint"]))
        self.assertEqual(sequence[0][0], ACTION_JUMP)
        self.assertEqual(sequence[1][0], ACTION_ATTACK)
        # w-tap：挥刀后的这一帧只前进、不疾跑，并明确把 sprint=False 写进 meta。
        self.assertEqual(sequence[2], (ACTION_APPROACH, False))
        # 再下一帧恢复疾跑（默认 sprint=True）。
        self.assertTrue(sequence[3][1])
        self.assertEqual(sequence[3][0], ACTION_ATTACK)

    def test_wtap_does_not_fire_without_previous_attack(self):
        teacher = TeacherV2()
        meta: dict = {}
        # 目标在空中 → 不跳，直接进入常规分支：第一帧就是 attack，不能带 sprint=False。
        teacher(obs(distance=2.0, in_reach=1.0, target_ground=0.0), meta)
        self.assertTrue(meta["sprint"])


class BlockTest(unittest.TestCase):
    """规则 6：血量 < 10、在 reach 内、对手刚攻击过 → 举剑格挡。"""

    def test_blocks_when_low_health_and_pressured(self):
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(health=6.0, target_health=18.0, distance=2.0,
                                     since_attack=0.2)), ACTION_BLOCK)

    def test_no_block_when_opponent_not_attacking(self):
        teacher = TeacherV2()
        action = teacher(obs(health=6.0, target_health=18.0, distance=2.0, since_attack=4.0))
        self.assertNotEqual(action, ACTION_BLOCK)

    def test_opponent_since_attack_from_meta_wins(self):
        teacher = TeacherV2()
        action = teacher(obs(health=6.0, target_health=18.0, distance=2.0, since_attack=4.0),
                         {"opponent_since_attack": 0.1})
        self.assertEqual(action, ACTION_BLOCK)


class DegenerateInputTest(unittest.TestCase):
    def test_no_target_idles(self):
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(has_target=0.0)), ACTION_IDLE)

    def test_zero_vector_does_not_crash(self):
        demos.reset_teacher_v2()
        self.assertEqual(heuristic_action_v2(np.zeros(OBS_DIM, dtype=np.float32)), ACTION_IDLE)

    def test_short_vector_does_not_crash(self):
        demos.reset_teacher_v2()
        self.assertEqual(heuristic_action_v2(np.zeros(3, dtype=np.float32)), ACTION_IDLE)

    def test_action_is_always_in_range(self):
        teacher = TeacherV2()
        for health in (2.0, 10.0, 20.0):
            for distance in (0.5, 2.0, 3.0, 5.0, 12.0):
                action = teacher(obs(health=health, distance=distance))
                self.assertIn(action, range(len(ACTION_NAMES)))


class StaticEnv:
    """固定观测的最小 env：只为验证 collect_demos 的版本派发与逐局重置。"""

    def __init__(self, observation: np.ndarray) -> None:
        self.observation = observation
        self.resets = 0

    def reset(self):
        self.resets += 1
        return self.observation.copy(), {}

    def step(self, _action):
        return self.observation.copy(), 0.0, False, False, {}


class CollectDemosVersionTest(unittest.TestCase):
    def test_v2_state_resets_each_episode(self):
        """不重置的话，第 2 局的第 1 帧会接着上一局的 pending_crit 直接挥刀。"""
        env = StaticEnv(obs(distance=2.0, in_reach=1.0))
        samples = collect_demos(env, episodes=2, max_steps=1, teacher_version="v2")
        first_actions = [action for _vector, action, _ret in samples]
        self.assertEqual(first_actions, [ACTION_JUMP, ACTION_JUMP])

    def test_default_version_is_v1(self):
        env = StaticEnv(obs(distance=2.0, in_reach=1.0))
        samples = collect_demos(env, episodes=1, max_steps=1)
        self.assertEqual(samples[0][1], ACTION_ATTACK)


class TeacherTargetTest(unittest.TestCase):
    """块 B 的软目标工具（放在这里是因为它与老师标签同源）。"""

    def test_temperature_one_is_onehot(self):
        q = teacher_target(2, 11, temperature=1.0)
        self.assertEqual(q[2], 1.0)
        self.assertEqual(float(np.sum(q)), 1.0)

    def test_higher_temperature_is_flatter(self):
        sharp = teacher_target(2, 11, temperature=1.0)
        flat = teacher_target(2, 11, temperature=2.0)
        self.assertGreater(float(flat[2]), 0.0)
        self.assertLess(float(flat[2]), float(sharp[2]))
        self.assertGreater(float(flat[0]), 0.0)

    def test_label_smoothing_mixes_uniform(self):
        q = teacher_target(2, 11, temperature=1.0, label_smoothing=0.5)
        self.assertAlmostEqual(float(q[2]), 0.5 + 0.5 / 11, places=9)
        self.assertAlmostEqual(float(q[0]), 0.5 / 11, places=9)


if __name__ == "__main__":
    unittest.main()
