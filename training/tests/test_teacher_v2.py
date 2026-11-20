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
        # v3 已上线（见 test_teacher_v3.py），这里改用仍未实现的版本号。
        with self.assertRaises(ValueError):
            make_teacher("v9")


class NoLongRetreatTest(unittest.TestCase):
    """规则 1（间距后撤）已删除的回归。

    2026-11-20 实测：v2 老师靠「血量落后就后撤」把出手率压到 9.6%，对 dagger-1118b
    0/6（同期 v1 老师 4/6）。这里用合成观测锁死「被追打时必须还手」。
    """

    def test_does_not_retreat_when_out_traded(self):
        teacher = TeacherV2(rod_ready=False)
        self.assertNotEqual(teacher(obs(health=8.0, target_health=18.0, distance=2.0)),
                            ACTION_RETREAT)

    def test_keeps_attacking_while_behind_on_health(self):
        """挨打 40 帧：至少 35 帧要挥刀，且不存在 spacing_retreat。"""
        teacher = TeacherV2(rod_ready=False)
        probe = obs(health=6.0, target_health=18.0, distance=1.8, in_reach=1.0)
        actions = [teacher(probe) for _ in range(40)]
        attacks = sum(1 for a in actions if a == ACTION_ATTACK)
        self.assertGreaterEqual(attacks, 35, f"只挥了 {attacks}/40 刀，说明又在长期后退")
        self.assertNotIn("spacing_retreat", teacher.reasons)

    def test_reason_stats_count_every_frame(self):
        teacher = TeacherV2(rod_ready=False)
        probe = obs(health=18.0, target_health=10.0, distance=2.0)
        for _ in range(5):
            teacher(probe)
        self.assertEqual(sum(teacher.reasons.values()), 5)


class RodTest(unittest.TestCase):
    """规则 2（鱼竿）已删除的回归。

    ``use_rod`` 解码不含位移，切回剑那帧也不动；只在「够不到、该往前压」时生效，
    等于白站 2 帧。保留 ``rod_ready`` 解析只为兼容旧调用方。
    """

    def test_never_uses_rod(self):
        teacher = TeacherV2()
        for distance in (2.6, 3.5, 4.1, 4.2):
            action = teacher(obs(distance=distance, in_reach=0.0))
            self.assertNotEqual(action, ACTION_USE_ROD, f"{distance} 格时仍在放鱼竿")

    def test_rod_ready_flag_is_still_accepted(self):
        teacher = TeacherV2(rod_ready=False)
        self.assertIsInstance(teacher(obs(distance=4.1, in_reach=0.0)), int)

    def test_out_of_reach_keeps_closing(self):
        """够不到时必须是 approach（往前压），不是放竿也不是站着。"""
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(distance=4.1, in_reach=0.0)), ACTION_APPROACH)


class GappleTest(unittest.TestCase):
    """规则 3（金苹果）**已删除**的回归。

    算术：吃一次 = 16 帧不能动/不能砍，换 4 点血（1 刀的伤害）；一局约 50 帧，
    16 帧 ≈ 6.4 点输出。原来还会因「每帧重发 use{gapple}」形成活锁（实测 207/436 帧）。
    """

    def test_never_eats(self):
        teacher = TeacherV2()
        for health in (6.0, 7.5, 12.0):
            action = teacher(obs(health=health, target_health=18.0, distance=4.1,
                                 in_reach=0.0))
            self.assertNotEqual(action, ACTION_EAT, f"{health} 血时还在吃苹果")

    def test_low_health_keeps_closing(self):
        teacher = TeacherV2()
        self.assertEqual(teacher(obs(health=6.0, target_health=18.0, distance=4.1,
                                     in_reach=0.0)), ACTION_APPROACH)

    def test_low_health_in_reach_attacks(self):
        teacher = TeacherV2()
        # 第一帧可能是跳劈（每局一次），之后必须是攻击。
        actions = [teacher(obs(health=6.0, target_health=18.0, distance=2.0, in_reach=1.0))
                   for _ in range(4)]
        self.assertNotIn(ACTION_EAT, actions)
        self.assertGreaterEqual(actions.count(ACTION_ATTACK), 3)


class CritJumpAndWTapTest(unittest.TestCase):
    """规则 4：先起跳再补刀（两帧）；规则 5（w-tap）已删除，挥刀后不再白丢一帧。"""

    def test_jump_then_attack(self):
        teacher = TeacherV2()
        probe = obs(distance=2.0, in_reach=1.0)
        self.assertEqual(teacher(probe), ACTION_JUMP)
        self.assertEqual(teacher(probe), ACTION_ATTACK)

    def test_no_jump_when_target_airborne(self):
        teacher = TeacherV2()
        self.assertNotEqual(teacher(obs(distance=2.0, in_reach=1.0, target_ground=0.0)),
                            ACTION_JUMP)

    def test_no_jump_when_self_airborne(self):
        teacher = TeacherV2()
        self.assertNotEqual(teacher(obs(distance=2.0, in_reach=1.0, self_ground=0.0)),
                            ACTION_JUMP)

    def test_no_wasted_frame_after_attack(self):
        """w-tap 删除后：跳劈补刀之后每一帧都应该继续砍，不再插入 ``approach``。"""
        teacher = TeacherV2()
        probe = obs(distance=2.0, in_reach=1.0)
        sequence = [teacher(probe) for _ in range(6)]
        self.assertEqual(sequence[0], ACTION_JUMP)
        self.assertEqual(sequence[1], ACTION_ATTACK)
        self.assertEqual(sequence[2:], [ACTION_ATTACK] * 4,
                         f"挥刀后出现了非攻击帧：{sequence}")

    def test_jump_only_once_per_episode(self):
        """``recent_attack_jumps`` 只在攻击后更新：跳劈每局最多一次（代价 1 帧）。"""
        teacher = TeacherV2()
        probe = obs(distance=2.0, in_reach=1.0)
        actions = [teacher(probe) for _ in range(10)]
        self.assertEqual(actions.count(ACTION_JUMP), 1)


class BlockTest(unittest.TestCase):
    """规则 6：只有拿到**对手**刚攻击的真信号才格挡；缺信号时优先还手。"""

    def test_blocks_when_low_health_and_pressured(self):
        teacher = TeacherV2()
        action = teacher(obs(health=6.0, target_health=18.0, distance=2.0),
                         {"opponent_since_attack": 0.2})
        self.assertEqual(action, ACTION_BLOCK)

    def test_low_health_in_reach_attacks_without_signal(self):
        """回归：没有对手攻击信号时，低血 + 贴身必须挥刀，不能连挡（实测 36 帧白丢）。

        第一帧允许是跳劈（每局一次），之后必须每帧都砍。
        """
        teacher = TeacherV2()
        probe = obs(health=6.0, target_health=18.0, distance=2.0, since_attack=0.1)
        actions = [teacher(probe) for _ in range(6)]
        self.assertNotIn(ACTION_BLOCK, actions)
        self.assertGreaterEqual(actions.count(ACTION_ATTACK), 5, f"实际序列 {actions}")

    def test_no_block_when_opponent_not_attacking(self):
        teacher = TeacherV2()
        action = teacher(obs(health=6.0, target_health=18.0, distance=2.0),
                         {"opponent_since_attack": 4.0})
        self.assertNotEqual(action, ACTION_BLOCK)

    def test_opponent_since_attack_from_meta_wins(self):
        teacher = TeacherV2()
        action = teacher(obs(health=6.0, target_health=18.0, distance=2.0, since_attack=4.0),
                         {"opponent_since_attack": 0.1})
        self.assertEqual(action, ACTION_BLOCK)


class V1ParityTest(unittest.TestCase):
    """核心回归：贴身、双方满血、无特殊情况时，v2 必须与 v1 逐帧同动作。

    v2 的定位是「v1 + 少量不亏节奏的时序行为」。一旦某些规则让它在能砍的时候不砍，
    出手率就会掉回 2026-11-20 那次 9.6% 的坑里（那次对 dagger-1118b 是 0/6）。
    """

    def test_same_actions_as_v1_when_in_reach(self):
        teacher = TeacherV2()
        probe = obs(health=20.0, target_health=20.0, distance=2.5, in_reach=1.0)
        v1_actions = [heuristic_action(probe) for _ in range(40)]
        v2_actions = [teacher(probe) for _ in range(40)]
        # 允许第一帧的跳劈（每局一次）不同，其余必须完全一致。
        self.assertEqual(v2_actions[0], ACTION_JUMP)
        self.assertEqual(v2_actions[1:], v1_actions[1:])

    def test_same_actions_as_v1_when_far(self):
        teacher = TeacherV2()
        probe = obs(health=20.0, target_health=20.0, distance=8.0, in_reach=0.0)
        self.assertEqual([teacher(probe) for _ in range(5)],
                         [heuristic_action(probe) for _ in range(5)])


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
