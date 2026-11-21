"""行为克隆热启动（P1.5）。

为什么需要它：PPO 从零学「贴身之后要挥剑」很慢——实测 24 训练 + 24 评估 0 胜，
策略塌成「approach 120 步 / attack 0 步」：因为 ``attack`` 只有命中才给正奖励，
「不挥剑」反而没有空挥惩罚，是稳定的局部最优。

做法：先用一个**只看观测向量**的脚本老师跑几局，把 (obs, 动作, 折扣回报) 存下来，
再用交叉熵把策略头拉到老师的行为上（只更新 W1/b1/W2/b2，价值头 Wv/bv 不动——
价值头属于 critic，必须由 PPO 自己学，否则会污染 GAE 的基线）。

老师刻意不读 env 内部状态：它只吃 23 维观测，所以离线、真机、单测三处行为完全一致。

两个版本（``TEACHER_VERSION`` 默认 ``"v1"``）：

- **v1** ``heuristic_action``：4 条静态规则。DAgger 一致率已顶到 1.0，说明它太好克隆、
  没有提升空间（见 ``docs/training-log.md``）；
- **v2** ``heuristic_action_v2``：v1 + 间距控制 / 鱼竿 / 金苹果 / 暴击跳 / w-tap / 低血格挡，
  带跨帧记忆（:class:`TeacherV2State`）。用 ``collect_demos(..., teacher_version="v2")``
  或 ``train.py --bc-teacher v2`` 选择。
- **v3** ``heuristic_action_v3``：把「攻击冷却」当资源，用死时间换位置与暴击（规则表见
  :func:`heuristic_action_v3` 的 docstring）。带跨帧记忆（:class:`TeacherV3State`），
  用 ``collect_demos(..., teacher_version="v3")`` 或 ``scripts/teacher_eval.py --teacher v3`` 选择。

``meta`` 是 v2/v3 的「观测之外的状态」入口（鱼竿是否可用、对手多久前攻击过、一帧多少秒、
对手是否正在挥刀），缺省值都能跑，所以离线采集不需要额外接线。

为什么需要 v3（2026-11-21）：v1/v2 的「射程内每帧都砍」在**宏动作空间**里其实有一半是白砍——
1.8 的受击无敌帧是 500ms，而控制频率是 20Hz，于是每 10 帧里最多只有 1 帧真的能造成伤害。
v2 把这 9 帧空转成了「同一个 attack 再发一次」，v3 把它们换成走位、跳劈准备与格挡，
因此单位时间的**有效伤害**更高（暴击 +50%），而不是靠出手率堆出来的。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, OBS_FIELDS

#: 默认老师版本（v1 = 规则老师，行为与 P1.5 完全一致）。
TEACHER_VERSION = "v1"
#: 可选老师版本。v2 在 v1 之上加间距/鱼竿/金苹果/暴击跳/w-tap/格挡六条规则；
#: v3 把攻击冷却当资源（死时间走位/跳劈/格挡），见 ``heuristic_action_v3``。
SUPPORTED_TEACHER_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")

# --- 观测下标：全部从 OBS_FIELDS 查，不写死数字 ---------------------------------


def field_index(name: str) -> int:
    try:
        return OBS_FIELDS.index(name)
    except ValueError as exc:  # pragma: no cover - 只在改观测定义时触发
        raise AssertionError(f"OBS_FIELDS 里没有 {name}，观测定义变了但 demos.py 没跟上") from exc


IDX_SELF_HEALTH = field_index("self_health")
IDX_SELF_ON_GROUND = field_index("self_on_ground")
IDX_TARGET_HEALTH = field_index("target_health")
IDX_TARGET_DISTANCE = field_index("target_distance")
IDX_TARGET_ON_GROUND = field_index("target_on_ground")
IDX_HAS_TARGET = field_index("has_target")
IDX_SELF_GAPPLES = field_index("self_gapples")
IDX_SELF_SINCE_ATTACK = field_index("self_since_attack")
IDX_IN_REACH = field_index("in_reach")
# v3 额外要用的字段（仍然按名字查，观测定义变了会立刻报错而不是静默错位）。
IDX_OPPONENT_SPEED = field_index("opponent_speed")
IDX_OPPONENT_VY = field_index("opponent_vertical_speed")

ACTION_IDLE = ACTION_NAMES.index("idle")
ACTION_APPROACH = ACTION_NAMES.index("approach")
ACTION_ATTACK = ACTION_NAMES.index("attack")
ACTION_RETREAT = ACTION_NAMES.index("retreat")
ACTION_JUMP = ACTION_NAMES.index("jump")
ACTION_USE_ROD = ACTION_NAMES.index("use_rod")
ACTION_EAT = ACTION_NAMES.index("eat_gapple")
ACTION_BLOCK = ACTION_NAMES.index("block")
ACTION_SWITCH_SWORD = ACTION_NAMES.index("switch_sword")
#: v3 的距离控制要用横移（v1/v2 没用过，所以这两个常量是新加的）。
ACTION_STRAFE_LEFT = ACTION_NAMES.index("strafe_left")
ACTION_STRAFE_RIGHT = ACTION_NAMES.index("strafe_right")

#: 观测里的归一化尺度（与 spaces.encode_observation 一致）。
HEALTH_SCALE = 20.0
DISTANCE_SCALE = 32.0
#: ``self_gapples`` 的归一化尺度（数量 / 5）。
GAPPLE_SCALE = 5.0

#: 老师认为「该继续靠近」的距离阈值（格）。
TEACHER_APPROACH_DISTANCE = 4.0
#: 老师认为「该回血」的血量阈值（点）。
TEACHER_HEAL_HEALTH = 8.0

# --- 老师 v2 的规则阈值（全部按「格 / 点 / 秒」写，单位见注释） -------------------
# 说明（2026-11-20 实测，见 docs 与提交说明）：v2 老师曾因四条「保命/节奏」规则把
# 出手率压到 9.6%（436 帧里只有 42 帧 attack），对 dagger-1118b 被打成 0/6，
# 而同期 v1 老师（纯「射程内就砍」）是 4/6。下列阈值保留的都是**加伤害**的规则，
# 被删掉的规则在各自的删除点写明了理由。
#: 暴击跳：距离不大于该值才起跳（格）。
TEACHER_V2_CRIT_DISTANCE = 3.0
#: 暴击跳：回看最近几次攻击的「是否跳劈」。
TEACHER_V2_CRIT_MEMORY = 2
#: 低血格挡：血量低于该值（点）。
TEACHER_V2_BLOCK_HEALTH = 10.0
#: 低血格挡：对手「刚攻击过」的窗口（秒）。
TEACHER_V2_BLOCK_WINDOW = 1.0
#: 控制频率倒数：v2 用帧数折算秒（10Hz 控制 → 0.1 秒/帧）。
#: 真机的控制频率不同时用 ``meta={"frame_seconds": ...}`` 覆盖。
TEACHER_V2_FRAME_SECONDS = 0.1

#: 编译期自检：字段顺序变了就立刻炸，而不是静默错位。
assert OBS_DIM == len(OBS_FIELDS), f"OBS_DIM={OBS_DIM} 与 OBS_FIELDS={len(OBS_FIELDS)} 不一致"
# G-702：观测 23 → 25，两个新字段追加在 ``in_reach`` **之后**。这里断言的是
# 「in_reach 没被挤走」（仍是 index 22），而不是「它是最后一个」——旧 23 维模型
# 与老师的下标必须继续指向同一个量，否则右侧补零续训会在无人察觉时错位。
assert IDX_IN_REACH == 22, "in_reach 的下标必须是 22（新字段只能追加在它之后）"
assert 0 <= IDX_SELF_HEALTH < OBS_DIM and 0 <= IDX_HAS_TARGET < OBS_DIM
assert 0 <= IDX_TARGET_DISTANCE < OBS_DIM
for _name, _index in (
    ("self_on_ground", IDX_SELF_ON_GROUND),
    ("target_health", IDX_TARGET_HEALTH),
    ("target_on_ground", IDX_TARGET_ON_GROUND),
    ("self_gapples", IDX_SELF_GAPPLES),
    ("self_since_attack", IDX_SELF_SINCE_ATTACK),
):
    assert 0 <= _index < OBS_DIM, f"{_name} 下标 {_index} 越界（OBS_DIM={OBS_DIM}）"


def _get(obs: np.ndarray, index: int) -> float:
    try:
        return float(np.asarray(obs, dtype=np.float64).reshape(-1)[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def heuristic_action(obs: np.ndarray) -> int:
    """脚本老师：只读观测向量，返回动作编号。

    规则（顺序即优先级）：
    1. 射程内（``in_reach==1``）且看得到目标 → ``attack``；
    2. 看不到目标 → ``idle``；
    3. 距离 > 4 格 → ``approach``（先贴上去）；
    4. 自己血量 < 8 → ``eat_gapple``；
    5. 其余 → ``approach``。
    """
    has_target = _get(obs, IDX_HAS_TARGET) >= 0.5
    if not has_target:
        return ACTION_IDLE
    if _get(obs, IDX_IN_REACH) >= 0.5:
        return ACTION_ATTACK
    if _get(obs, IDX_TARGET_DISTANCE) * DISTANCE_SCALE > TEACHER_APPROACH_DISTANCE:
        return ACTION_APPROACH
    if _get(obs, IDX_SELF_HEALTH) * HEALTH_SCALE < TEACHER_HEAL_HEALTH:
        return ACTION_EAT
    return ACTION_APPROACH


# --- 老师 v2：在 v1 之上加时序规则 --------------------------------------------


class TeacherV2State:
    """老师 v2 的**跨帧记忆**。

    规则 1/2/4/5 都需要「上一帧做过什么」，只吃单帧观测的纯函数表达不了。
    状态挂在老师实例上（每个 episode 一份），而不是模块全局，避免多局互相污染。

    字段含义：

    - ``frame``：调用计数（每调用一次 +1），用 ``frame × frame_seconds`` 折算秒；
    - ``backing_off``：间距控制的后撤闩锁（进入 2.2 格，退到 2.6 格才解除）；
    - ``pending_crit``：上一帧刚起跳，这一帧补刀（暴击跳的两帧节拍）；
    - ``pending_sword_switch``：上一帧刚放鱼竿，这一帧切回剑；
    - ``last_rod_frame``：上次放鱼竿的帧号（冷却判断）；
    - ``recent_attack_jumps``：最近若干次攻击是否「跳劈」；
    - ``last_attack_frame``：上次攻击的帧号（w-tap 判断「上一帧刚 attack」）。
    """

    __slots__ = ("frame", "pending_crit", "pending_sword_switch",
                 "last_rod_frame", "recent_attack_jumps", "last_attack_frame",
                 "last_action", "sprint", "reasons", "crit_used")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "TeacherV2State":
        self.frame = 0
        self.pending_crit = False
        self.pending_sword_switch = False
        self.last_rod_frame: int | None = None
        self.recent_attack_jumps: list[bool] = []
        self.last_attack_frame: int | None = None
        self.last_action: int | None = None
        #: 本帧建议的疾跑开关（``False`` = w-tap 的「只前进不疾跑」）。
        self.sprint = True
        #: 本局各条规则的命中次数（``reason -> count``），reset 时清零。
        #: 谁在长期生效、谁从未生效，看这个字典就够了。
        self.reasons: dict[str, int] = {}
        #: 本局是否已经用过跳劈（每局只值得试一次）。
        self.crit_used = False
        return self


def _finish(state: TeacherV2State, action: int, *, sprint: bool | None = None,
            jumped: bool = False, reason: str) -> tuple[int, dict]:
    """收尾：记账（攻击历史/w-tap/帧号）并返回 ``(动作, 诊断信息)``。"""
    frame = state.frame
    if action == ACTION_ATTACK:
        state.recent_attack_jumps.append(bool(jumped))
        # 只保留最近若干次：暴击跳只看最近两次，长历史没有信息量。
        del state.recent_attack_jumps[:-max(1, TEACHER_V2_CRIT_MEMORY)]
        state.last_attack_frame = frame
    state.last_action = action
    state.sprint = True if sprint is None else bool(sprint)
    state.frame = frame + 1
    state.reasons[reason] = state.reasons.get(reason, 0) + 1
    return action, {"reason": reason, "sprint": state.sprint, "jumped": bool(jumped)}


def _teacher_options(meta: Any) -> dict:
    """把 ``meta`` 规范成 v2 的决策上下文。

    可覆盖的键（都可缺省）：

    - ``rod_ready``：鱼竿是否在背包且能放（默认 ``True``，训练服初始套装含鱼竿）；
    - ``gapples``：金苹果数量（默认取观测 ``self_gapples``，观测只有 0..1 的归一化值）；
    - ``since_attack``：**自己**距上次攻击的秒数（默认取观测）；
    - ``opponent_since_attack``：**对手**距上次攻击的秒数（观测里没有这个字段，
      真机可由 bot-runtime 提供）。**缺省时不做低血格挡**——用自己的 since_attack 兜底
      会把「我刚砍完」误判成「对手在打我」（2026-11-20 实测的连挡问题）；
    - ``frame_seconds``：一帧多少秒（默认 0.1，即 10Hz 控制）；
    - ``state``：``TeacherV2State`` 实例（缺省用模块级默认状态）。
    """
    data: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    state = data.get("state")
    if not isinstance(state, TeacherV2State):
        state = _DEFAULT_V2_STATE
    frame_seconds = data.get("frame_seconds")
    return {
        "state": state,
        "rod_ready": bool(data.get("rod_ready", True)),
        "gapples": data.get("gapples"),
        "since_attack": data.get("since_attack"),
        # 缺省保持 None：`_decide_v2` 用它区分「有真信号」与「没有信号」。
        "opponent_since_attack": data.get("opponent_since_attack"),
        "frame_seconds": (TEACHER_V2_FRAME_SECONDS if frame_seconds is None
                          else max(1e-6, float(frame_seconds))),
    }


def _decide_v2(obs: np.ndarray, options: dict) -> tuple[int, dict]:
    """v2 的决策核心（纯函数 + 显式状态），返回 ``(动作, 诊断)``。

    判定顺序：格挡（仅真信号）→ 暴击补刀 → 起跳 → v1 兜底。

    2026-11-20 修复：v2 原来比 v1 弱得多（对 dagger-1118b：v2 老师 0/6、v1 老师 4/6），
    根因是四条「保命/节奏」规则把出手率压到 **9.6%**（436 帧里 42 帧 attack）：
    吃苹果 207 帧、后撤 43、格挡 36、w-tap 18。处理方式与理由逐条写在下面。
    """
    state: TeacherV2State = options["state"]
    frame = state.frame

    has_target = _get(obs, IDX_HAS_TARGET) >= 0.5
    if not has_target:
        # 看不到人：所有「上一帧刚做过什么」的节拍都作废，别把状态带进下一局。
        state.pending_crit = False
        state.pending_sword_switch = False
        return _finish(state, ACTION_IDLE, reason="no_target")

    health = _get(obs, IDX_SELF_HEALTH) * HEALTH_SCALE
    distance = _get(obs, IDX_TARGET_DISTANCE) * DISTANCE_SCALE
    in_reach = _get(obs, IDX_IN_REACH) >= 0.5
    self_ground = _get(obs, IDX_SELF_ON_GROUND) >= 0.5
    target_ground = _get(obs, IDX_TARGET_ON_GROUND) >= 0.5

    # ``gapples`` / ``since_attack`` 仍然被 :func:`_teacher_options` 解析（meta 契约不变），
    # 但当前没有规则消费它们：金苹果规则已删（见规则 3），而 ``since_attack`` 是**自己**
    # 的距上次攻击，不能当成「对手在打我」（见规则 6 的修复）。
    opponent_since_attack = options["opponent_since_attack"]

    # --- 规则 6：低血格挡（只在拿到**真**的「对手刚攻击」信号时才用） ---
    # 原来缺这个信号时回落到 `since_attack`（**自己**距上次攻击），于是「我刚挥完刀」
    # 被当成「对手正在打我」→ 血量 <10 时疯狂格挡（实测 36 帧/6 局，全是白丢的输出）。
    # 现在没有 meta 显式给 `opponent_since_attack` 就不格挡：宁可挨一刀也要换血。
    if (opponent_since_attack is not None and health < TEACHER_V2_BLOCK_HEALTH
            and in_reach and float(opponent_since_attack) <= TEACHER_V2_BLOCK_WINDOW):
        state.pending_crit = False
        return _finish(state, ACTION_BLOCK, reason="low_health_block")

    # --- 规则 4 第二帧：上一帧起跳了，这一帧补刀（够不到就作废，别打空气） ---
    if state.pending_crit:
        state.pending_crit = False
        if in_reach:
            return _finish(state, ACTION_ATTACK, jumped=True, reason="crit_attack")

    # --- 规则 3：金苹果（**已删除**） ---
    # ①先修的是活锁：一帧 0.1s、一次使用按住 1.6s，没有冷却会每帧重发 use{gapple}，
    #   实测 207/436 帧（47%）卡在吃苹果上（苹果数量在「吃完」之前不下降）。
    # ②加了冷却后仍然净亏：吃一次 = 16 帧不能动、不能砍，换回 4 点血（= 1 刀的伤害）。
    #   一局只有约 50 帧，16 帧 ≈ 1.6 刀 ≈ 6.4 点输出。算术上「吃」永远不如「砍」。
    # ③v1 的同类规则（<8 血吃）实际只在 3.84–4.0 格的窄缝里触发，本来也几乎不生效。
    # 结论：这个动作空间里吃苹果没有正收益，直接删；``gapples`` 仍被解析（meta 契约不变）。

    # --- 规则 2：鱼竿（**已删除**） ---
    # 理由：``use_rod`` 解码出来是 ``[look, use{rod}]``，**不含位移**；切回剑那一帧也不动。
    # 所以它只会在「够不到、正该往前压」的距离上白站 2 帧（实测 4–13 帧/6 局）。
    # 运行时默认 CPS=15、控制 10Hz，攻击帧本来就不受点击上限限制——省下的每一帧都是净输出。
    # ``rod_ready`` 仍然被解析（兼容旧调用方），但不再影响决策；``pending_sword_switch``
    # 也保留在状态里，避免外部读取时 AttributeError。

    # --- 规则 4：暴击跳（贴脸、双方在地面、本局还没跳劈过 → 起跳，下一帧补刀） ---
    # 保留但收紧到 in_reach 内：原来距离 ≤3 就起跳，会把「本来能砍的一帧」换成起跳。
    # 用显式 ``crit_used`` 闩锁，而不是靠 ``recent_attack_jumps`` 的两格窗口：
    # 后者的 True 会被后续两次普通攻击挤出去，于是「每 4 刀跳一次」，实测 10 帧里
    # 跳 3 次（约 12% 的帧被起跳顶掉）。跳劈每局试一次就够。
    if (not state.crit_used and in_reach and distance <= TEACHER_V2_CRIT_DISTANCE
            and self_ground and target_ground):
        state.crit_used = True
        state.pending_crit = True
        return _finish(state, ACTION_JUMP, reason="crit_jump")

    # --- 规则 5：w-tap（**已删除动作分支**） ---
    # v1 动作空间里 ``approach`` 的疾跑由 decode_action 按距离决定（>2.5 格才疾跑），
    # meta["sprint"]=False 没有消费方；所以「上一帧刚砍、这一帧只前进」实际就是
    # **白丢一帧输出**（实测 18 帧/6 局）。节奏信息仍写进 state.sprint 供将来真机使用。

    # --- 兜底：与 v1 完全同序（射程内就砍 → 太远就压上 → 低血吃苹果 → 压上） ---
    # 原来还有一条「距离 >3.0 先压进贴身带」，它让 3.0–3.99（in_reach 内）的帧变成
    # approach，实测 approach_band 62 帧/6 局——同样是在能砍的时候不砍，一并去掉。
    if in_reach:
        return _finish(state, ACTION_ATTACK, reason="attack")
    if distance > TEACHER_APPROACH_DISTANCE:
        return _finish(state, ACTION_APPROACH, reason="approach")
    if health < TEACHER_HEAL_HEALTH:
        return _finish(state, ACTION_EAT, reason="heal_low")
    return _finish(state, ACTION_APPROACH, reason="approach")


#: ``heuristic_action_v2(obs)``（不传 meta）时使用的默认状态。
#: 多局/多线程场景请用 :class:`TeacherV2` 实例，或显式传 ``meta={"state": ...}``。
_DEFAULT_V2_STATE = TeacherV2State()


def reset_teacher_v2() -> None:
    """清空模块级默认状态（测试 / 新 episode 之间调用）。"""
    _DEFAULT_V2_STATE.reset()


def heuristic_action_v2(obs: np.ndarray, meta: Mapping[str, Any] | None = None) -> int:
    """老师 v2：v1 的规则 + 「每局一次跳劈」（+ 有真信号时的低血格挡）。

    只读观测（+ 可选 ``meta``），返回动作编号。

    2026-11-20 大修：v2 原来比 v1 弱得多（对 dagger-1118b：v2 老师 0/6、v1 老师 1–5/6），
    根因是「保命/节奏」规则把出手率压到 9.6%（436 帧只有 42 帧 attack）。逐条处理：

    1. **间距后撤** —— **已删除**。原规则「自己血量落后且距离 <2.2 就后撤」，而血量落后
       恰好是被追打的时候，退出去等于放弃还手（实测 43 帧/6 局）。v1 没有这条。
    2. **鱼竿** —— **已删除**。``use_rod`` 解码不含位移、切回剑那帧也不动，只在
       「够不到、正该往前压」时生效 → 白站 2 帧（实测 4–13 帧/6 局）。
    3. **金苹果** —— **已删除**。先修活锁（每帧重发 use{gapple}，实测 207/436 帧 = 47%），
       再算账：吃一次 = 16 帧不能动/不能砍 ≈ 1.6 刀 ≈ 6.4 点输出，换回 4 点血 → 永远净亏。
    4. **暴击跳** —— 保留并收紧到 ``in_reach`` 内，且**每局只试一次**。原来靠
       ``recent_attack_jumps`` 的两格窗口，True 被后续攻击挤出去后变成「每 4 刀跳一次」，
       实测 10 帧跳 3 次（约 12% 的帧被起跳顶掉）。
    5. **w-tap** —— **已删除动作分支**。v1 动作空间没有「前进但不疾跑」这个原子动作
       （``decode_action`` 按距离决定疾跑），写 ``meta["sprint"]=False`` 没有消费方，
       实际只是白丢一帧输出（实测 18 帧/6 局）。``state.sprint`` 仍然留档。
    6. **低血格挡** —— 只在 ``meta`` **显式给出** ``opponent_since_attack`` 时生效。
       观测里没有「对手距上次攻击」，原来的兜底（用自己的 ``since_attack``）把
       「我刚砍完」当成「对手在打我」，血量 <10 时变成连挡不砍（实测 36 帧/6 局）。

    实际判定顺序：``6 格挡（需真信号）`` → ``4 补刀`` → ``4 起跳`` →
    ``v1 兜底（射程内就砍 / >4 格压上 / <8 血吃苹果）``。

    也就是说，v2 现在**等价于 v1 + 每局一次跳劈 + （有真信号时的）低血格挡**。
    这是刻意的：在这个动作空间里（控制 10Hz、运行时 CPS=15 不构成瓶颈），
    「射程内每帧都砍」就是上限，任何额外规则只要占一帧就是净亏。

    ``meta`` 支持的键见 :func:`_teacher_options`；写回 ``meta["sprint"]`` 与
    ``meta["reason"]``（决策依据，便于复盘）。
    """
    options = _teacher_options(meta)
    action, info = _decide_v2(obs, options)
    if isinstance(meta, dict):
        meta["sprint"] = info["sprint"]
        meta["reason"] = info["reason"]
    return int(action)


class TeacherV2:
    """可复用的 v2 老师实例（自己持有一份 :class:`TeacherV2State`）。

    ``teacher = TeacherV2(); [teacher(obs) for obs in frames]`` —— 状态跨帧，
    每个 episode 开头调 :meth:`reset`。``defaults`` 里的键会作为 ``meta`` 的默认值
    （例如 ``TeacherV2(rod_ready=False)`` 模拟没带鱼竿的局）。
    """

    version = "v2"

    def __init__(self, **defaults: Any) -> None:
        self.state = TeacherV2State()
        self.defaults = dict(defaults)
        #: 已结束 episode 的原因统计（每次 :meth:`reset` 归档上一局）。
        self.episode_reasons: list[dict[str, int]] = []

    def reset(self) -> "TeacherV2":
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
        self.state.reset()
        return self

    def reset_episode(self) -> None:
        """环境 ``reset()`` 时的回调（``GladiatorEnv`` 约定）：归档上一局并清零状态。"""
        self.reset()

    @property
    def reasons(self) -> dict[str, int]:
        """当前 episode 各条规则的命中次数（``reason -> count``）。"""
        return dict(self.state.reasons)

    def flush(self) -> list[dict[str, int]]:
        """把「还没归档的当前 episode」也收进 ``episode_reasons``（对局结束时调用）。"""
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
            self.state.reasons = {}
        return list(self.episode_reasons)

    def __call__(self, obs: np.ndarray, meta: Mapping[str, Any] | None = None) -> int:
        merged: dict[str, Any] = dict(self.defaults)
        if isinstance(meta, Mapping):
            merged.update(meta)
        merged["state"] = self.state
        action = heuristic_action_v2(obs, merged)
        if isinstance(meta, dict):
            # 调用方传进来的 dict 要能看到决策诊断（w-tap 的 sprint off 靠这个传出去）。
            meta["sprint"] = merged.get("sprint", True)
            meta["reason"] = merged.get("reason", "")
        return action


# ============================================================================
# 老师 v3：把「攻击冷却」当资源，用死时间换位置与暴击
# ============================================================================
#
# 出发点（这是 v3 与 v1/v2 的唯一本质差别）：
#
# 1.8 的受击无敌帧是 10 tick = 500ms，而控制频率是 20Hz（50ms/步）。也就是说每 10 帧里
# 最多只有 1 帧的 attack 能造成伤害，其余 9 帧的 attack 在服务端被直接吞掉。
#
#   - v1/v2 的做法是「在射程内就每帧发 attack」：能打中的那一刀确实没漏，但另外 9 帧
#     完全空转（动作被吞、位置不变）。它们的上限就是「每 0.5 秒一刀普通伤害」。
#   - v3 的做法是把这 9 帧当成**可支配的时间预算**：冷却期间去走位保持交战线、提前起跳
#     把下一刀变成暴击（+50%）、在对手出手的窗口里格挡。攻击仍然是默认行为，但只要
#     冷却没好就换成位移——**在冷却没好时发 attack 的信息量是 0**，换位移至少有位置收益。
#
# 于是 v3 的强度来源不是「出手更多」（出手频率反而被冷却锁死），而是：
#   a) 每一刀都踩在冷却刚好的那一帧（不空转、不漏刀）；
#   b) 约一半的刀是跳劈暴击（1.5 倍伤害）；
#   c) 击退后不会一路疾跑追出去（w-tap），对手留在自己射程内；
#   d) 鱼竿打断对手冲刺、把它拉回自己的交战线；
#   e) 格挡只花「本来打不出伤害的那些帧」，不占输出。
#
# 与 v2 教训的关系：v2 被「保命优先」的规则链害到出手率 9.6%，所以 v3 的铁律是
# **冷却一好、在射程内，就必须砍**——攻击判定永远排在所有规则之前（见 ``_decide_v3``
# 第 2 步）。所有防御/位移规则都只在「这一帧砍了也没有伤害」时才有机会生效。
#
# --- 2026-11-21 真机 12 局（1 胜 11 负）复盘出的三条修正 -----------------------
#
# 那次失败的 reason 直方图：eating=354 / wtap_strafe=172 / approach=114 / attack=109，
# 12 局平均只活 64 步，且 crit_jump **一次都没出现**。三个根因：
#
#   1. 金苹果门槛太低（血 ≤12 且 >4 格）。吃一次按住 1.6s，而它在 64 步的局里占 31 步：
#      等于半个回合站着不动被砍。现在改成「血 ≤6 + 对手 8 格外 + 没在接近 + 每局 1 个」，
#      并且**砍得到人就中断吃**。
#   2. 步长被写死 50ms。环境推进是 ``sleep(action_ticks / control_hz)``，实际一帧更慢，
#      于是「1.6s 锁存」锁了 3.2s、「0.30–0.50s 的下落窗口」挪到了落地之后（所以没有暴击）。
#      现在用 ``self_since_attack`` 的墙钟增量自校准步长。
#   3. 跳劈距离带 2.0–3.0 格。真机上双方稳定停在 3.5–3.9 格（每次命中把对手推开 0.75 格），
#      这个带永远不成立。现在放宽到 3.7，并把 w-tap 限制在「还够得着」时才生效，
#      避免击退后被 w-tap 白占两帧、压不回去。
#
# --- 2026-11-21 第二轮（仍 1 胜 11 负，但暴露了两个硬件级事实） -----------------
#
# ``frameSeconds = 0.2037``：``GladiatorEnv`` 的推进是 ``sleep(action_ticks/control_hz)``
# = 2/10 = 0.2s，不是 0.05s。所有「按 20Hz 写死的 tick 数」都偏了 4 倍：
#   - 0.5s 冷却 = 2.5 步 → 第 3 步砍（0.61s），不是第 10 帧；
#   - 跳劈滞空 0.6s = 3 步 → 起跳后**隔一步**砍（第 2 步 = 0.41s，正在下落），
#     不存在「第 6–10 帧」这种窗口；
#   - w-tap = 0.1s = 1 步；
#   - 横移换向 0.4–0.6s = 2–3 步。
# 现在所有常量都先写成**秒**，再 ``round(秒 / frame_seconds)`` 折成步数，
# 且 ``frame_seconds`` 优先用 ``self_since_attack`` 的墙钟增量自校准（跨局保留），
# 所以同一份老师在 20Hz / 10Hz / 5Hz 的 env 上行为一致。
#
# --- 2026-11-21 第三轮（逐刀掉血探针：普通 1.60 / 暴击 2.40，确认 crit 生效） ------
#
# 探针还暴露了两件事，都是**结构性**的，不是调参：
#
#   a) ``crit_jump`` 47 次只换到 4 次 ``crit_attack``。原因是起跳时机写成
#      「since_attack ≥ 0.1 就跳」，于是有一半的起跳发生在命中后**第 2 步**；
#      那一步跳完，下一刀落在起跳后第 1 步（0.204s）——**还在上升期，服务端不判暴击**。
#      现在改成按「还差几步到冷却」锚定（``steps_to_ready ∈ [2,3]``）：0.204s/步下
#      只可能在命中后的那一步起跳，跳完正好隔一步砍，落在下落期。
#   b) ``crit_wait``（为暴击多等一步）是**净亏**：等一步把 3 步循环变 4 步，
#      出手率 0.336 → 0.279 刀/步，伤害 0.70 → 0.57 hp/步。已删除——冷却一到就砍，
#      暴击靠「提前起跳」赚（起跳发生在冷却死时间里，本来就不占出手机会）。
#   c) ``w-tap`` 抢走的正是「命中后那一步」，而那一步是唯一能起跳的窗口。
#      这个动作空间里 w-tap 本来就是空操作（``approach`` 的 move 结束时会松掉疾跑），
#      所以它只在「跳劈起不来」时兜底，正常循环里不再出现。
#   d) tick 兜底时钟只在步长测准后启用：没测准时缺省 0.2 在 20Hz 环境上会把
#      「3 帧 = 0.6s」当成冷却好了，第 4 帧就砍第二刀（全被无敌帧吞掉）。
#
# 0.204s/步闭环（20 局）：出手 **0.336 刀/步**（上限 0.333，目标 0.3125）；暴击率
# 随击退强度 22–73%（0.75 格击退时 61%）；伤害 0.52–0.64 hp/步 = 对手的 1.12–1.38 倍。
#
# ``target_on_ground`` 是**死字段**：协议 ``entities[]`` 只有
# ``id/kind/name/pos/velocity/distance/heldItem``，没有 ``onGround``（见
# ``bot-runtime/src/observation.js``）。所以观测第 12 维在真机恒为 0，
# 「双方着地才起跳」这条把暴击率锁死在 **0**——第一轮 12 局 crit_jump 一次都没出现，
# 就是死在这一条上。现在起跳只看距离与冷却剩余；``self_on_ground`` 也不再作为硬条件
# （跳起来之后它才变 False，本来也拦不住）。

# --- v3 阈值（单位写在每行注释里；改这里就能调参） --------------------------

#: 一帧多少秒（**缺省值，实际会被自校准覆盖**）。
#: 真机实测 0.2037s/步（``action_ticks=2`` / ``control_hz=10``，见
#: ``artifacts/evals/20261121-0331-vs-teacher.json`` 的 ``frameSeconds``）——不是 0.05。
#: 用 ``meta={"frame_seconds": ...}`` 或 ``TeacherV3(frame_seconds=...)`` 可以钉死；
#: 20Hz 的环境上传 0.05 即可复用同一份老师。
TEACHER_V3_FRAME_SECONDS = 0.2
#: 步长自校准：用观测 ``self_since_attack``（墙钟秒）的增量反推「一帧到底多少秒」。
#: 2026-11-21 真机 12 局证明这个值不能写死：``GladiatorEnv`` 的推进是
#: ``sleep(action_ticks / control_hz)``，实测一帧远大于 50ms，于是「1.6s 的吃苹果锁存」
#: 实际锁了 3.2s（甚至更久），一局 64 步里它占 31 步——老师原地站着被砍死。
#: 自校准后，所有「秒 → tick」的换算（吃苹果锁存、跳劈窗口、w-tap、横移节拍）都自动对上。
TEACHER_V3_CLOCK_MIN_SAMPLES = 4
#: 自校准的合理区间（秒/帧）：超出这个范围的增量多半是观测抖动或重置，直接丢弃。
TEACHER_V3_CLOCK_MIN_SECONDS = 0.02
TEACHER_V3_CLOCK_MAX_SECONDS = 0.35
#: 自己的攻击距离（格）。与 ``BotProfile.reach`` 一致，``meta={"reach": ...}`` 可覆盖。
TEACHER_V3_REACH = 3.99
#: 出手余量（格）：观测的 ``in_reach`` 用的是**水平**距离，解码用的是实体 3D 距离，
#: 在空中时两者会差 0.2~0.4 格。贴边出手会被解码/运行时降级成「追击」，
#: 白白丢掉一个 0.6s 的冷却周期。
#:
#: 2026-11-21 第五轮：余量 **0.14（阈值 3.85 格 = 解码自己的 effective_reach）**。
#: 关键认识：``decode_action`` 在够不到时会把 attack **降级成「先靠近」**，
#: 也就是「朝对手走一步」——和 ``approach`` 做的事完全一样。所以
#: **在射程外选择 attack 是免费的**：它要么砍中，要么替我们靠近。
#: 而对手（BC）正是每步都按 attack，于是它从不漏掉任何一个贴身瞬间；
#: 我们前几版用 ``approach`` 顶替，等于把「尝试命中」的机会白扔了
#: （真机：出手 0.36/步 vs 对手 1.0/步，命中 0.176 vs 0.254/步）。
#: 逐刀探针实测的出手率只有 0.23–0.27 刀/步，而上限是 1/3 —— 差的那 25% 全是被
#: 「3.49 格以内才出手」挡掉的：0.204s/步下双方距离在 3.5–4.0 之间就没有停过。
#: 对手是每步都按 attack 的，我们少砍这 25% 就是输的直接原因。
#: 现在只要观测 ``in_reach`` 且冷却就绪就出手；真的打不出去（被解码/运行时降级成追击）
#: 时由 ``attack_dry`` 在下一帧自动收紧（见 TEACHER_V3_DRY_MARGIN），不用提前保守。
TEACHER_V3_ATTACK_MARGIN = 0.14
#: 「空挥」后的临时收紧（格）：上一刀被解码成追击时，说明当前距离在真机上打不出去。
#: 接下来这段时间里要求再压近这么多才出手（出手阈值 3.94 → 3.04），把边界上的空挥
#: 换成「压上」。0.9 是照着「解码阈值 3.84」往下留的余量：空挥说明 3.9 左右打不出去，
#: 那就退到 3.0 附近再说，反正只持续 0.4s。
TEACHER_V3_DRY_MARGIN = 0.9
#: 上面那道收紧持续多久（秒）。
TEACHER_V3_DRY_SECONDS = 0.4
#: 攻击冷却（秒）。1.8 受击无敌帧 500ms。
#: **按「真正命中」计时，而不是按「上一次攻击动作」**（2026-11-21 第四轮）：
#: 真机逐帧数据显示一半的刀在 3.7–3.9 格被运行时判成追击、一点伤害都没有，
#: 而 ``since_attack`` 照样归零 → 每次空挥后白等 0.5s。改成看观测里的对手血量
#: （``target_health`` 掉 ≥0.4 才算一次命中），空挥就不再浪费冷却窗口——
#: 这正是对手（每步都按 attack 的 BC）能抢到每个贴身窗口的原因。
TEACHER_V3_ATTACK_COOLDOWN = 0.5
#: 判定「掉血 = 一次命中」的最小血量差（点）。
TEACHER_V3_HIT_EPS = 0.4
#: 判定「这一刀是暴击」的血量差（点）：普通 1.6 / 暴击 2.4，取中间 2.0。
TEACHER_V3_CRIT_DAMAGE = 2.0

#: 理想交战线（格）：2.6–3.0 是「够得着、又不至于贴到被连击」的区间。
TEACHER_V3_IDEAL_MIN = 2.6
TEACHER_V3_IDEAL_MAX = 3.0
#: 区间内超过这个距离就压半步（而不是横移）：给「命中后的击退」留余量，
#: 否则下一次起跳会被推出射程，暴击就断了。
TEACHER_V3_HOLD_DISTANCE = 2.8
#: 超过这个距离就走 approach（格）。
TEACHER_V3_APPROACH_DISTANCE = 3.2
#: 走位横移区间（格）= 理想线向两侧各放宽一点。落在 3.0–3.2 时优先压半步进线（见
#: ``close_to_ideal``），只有落回区间内才横移。
TEACHER_V3_STRAFE_MIN = round(TEACHER_V3_IDEAL_MIN - 0.3, 2)
TEACHER_V3_STRAFE_MAX = round(TEACHER_V3_IDEAL_MAX + 0.2, 2)
#: 贴脸阈值（格）：小于它且对手在出手 → 退一步拉开跳劈距离。
TEACHER_V3_TOO_CLOSE = 2.0
#: 一次横移持续多少秒（在 0.4–0.6s 之间循环，避免每 tick 抖方向）。
#: 0.204s/步下等于 2–3 步；20Hz 下等于 8–12 tick。
TEACHER_V3_STRAFE_SPAN_SECONDS = (0.4, 0.6)

#: 跳劈：只在 2.0–3.7 格起跳。上界取到 ATTACK_MARGIN 的边界，是因为**真机上双方会
#: 稳定停在 3.5–3.9 格**（每次命中把对手推开 0.75 格，冷却期间压回来的距离正好抵掉），
#: 只允许 2.0–3.0 等于「永远不起跳」——2026-11-21 的 12 局里 crit_jump 一次都没出现。
TEACHER_V3_CRIT_MIN_DISTANCE = 2.0
TEACHER_V3_CRIT_MAX_DISTANCE = 3.7
#: 起跳时机不再用「冷却还剩多少秒」，而是**按还差几步到冷却**锚定（见 ``_v3_view`` 的
#: ``steps_to_ready``）：只有「现在起跳、等下落窗口打开时正好冷却好」才起跳。
#: 0.204s/步下这条恰好等价于「命中后的那一步起跳」，20Hz 下等价于「第 3–6 tick」。
#: （旧的 ``CRIT_PRE_JUMP_SECONDS`` 常量已删除：它在 0.204s 步长下会允许晚一步起跳，
#:  而晚一步起跳那一刀落在上升期，跳了等于白跳。）
#: 出手窗口（秒，相对起跳时刻）——**真机实测值**，不是推算值。
#:
#: ``scripts/tmp_crittiming.py`` 在真服上测了 delay=1/2/3（0.204s/步，各 10 次）：
#:   delay=1（跳后第 1 步砍）：6/7 出 2.40 = 暴击，且此时 vy = +3.7~+5.7（**还在上升**）
#:   delay=2（第 2 步）：0/7 暴击（vy ≈ 0，顶点附近）
#:   delay=3（第 3 步）：0/3 暴击（已经落地）
#: 也就是说「跳完立刻砍」才出暴击。物理上说不通（上升期怎么会有 fallDistance），
#: 唯一自洽的解释是 action 下发到真正发包之间还有约一步的延迟——所以**必须提前一步砍**。
#: 结论：0.204s/步下窗口 = 第 1 步（``round(0.10/0.204)=1``、``round(0.30/0.204)=1``）；
#: 20Hz 下自动变成第 2–6 tick。
TEACHER_V3_CRIT_ATTACK_MIN_SECONDS = 0.10
TEACHER_V3_CRIT_ATTACK_MAX_SECONDS = 0.30
#: 起跳允许「提前」多少格（在出手距离之外）。**实测 0 最好**：给 0.4 会让
#: 「跳完仍然够不到」的循环变多（出手率 0.28 → 0.20 刀/步），因为 0.204s 一步的
#: **空中** approach 只挪 0.2–0.4 格，补不上提前的量。留这个常量只是为了调参。
TEACHER_V3_CRIT_ARM_SLACK = 0.0

#: 两次跳劈的最小间隔（秒）：跳劈失败（对手跑开）时不至于连跳。
#: 0.204s/步下取 0.1（半 步）——这是个**相位敏感**的值：0.25（1.2 步）会把起跳推到
#: 下一个可用步，整条循环从 3 步变成 4 步，出手率掉 25%（实测 1.69 → 1.33 刀/秒）。
#: 0–0.20 都能落在同一条「3 步一循环」上，取中间值抗抖动。
TEACHER_V3_CRIT_COOLDOWN_SECONDS = 0.1

#: w-tap：每次 attack 之后这么多 tick 内不发 approach / forward（以 20Hz 为基准换算成秒）。
#: 疾跑攻击的击退更大，会把对手推出 3.99 格；松一下 W 再压上去能让对手留在射程内。
#: **只在「已经够得着」时生效**：被击退到 4 格外时压上去比摆架势重要得多。
TEACHER_V3_WTAP_TICKS = 2
#: w-tap 的真实时长（秒）。按 20Hz 的 2 tick 折算，因此换成 10Hz 也不会变成 0.2s。
TEACHER_V3_WTAP_SECONDS = TEACHER_V3_WTAP_TICKS * 0.05

#: 鱼竿窗口（格）：4.0–6.5 之间、够不到又不太远。
TEACHER_V3_ROD_MIN_DISTANCE = 4.0
TEACHER_V3_ROD_MAX_DISTANCE = 6.5
#: 鱼竿冷却（秒）。真机 ``releaseRod`` 之后要切回剑，所以额外占 1 tick。
TEACHER_V3_ROD_COOLDOWN_SECONDS = 3.0
#: 只在这两个条件之一满足时放竿：对手正在接近（距离每秒缩短 ≥ 该值，格/秒）。
TEACHER_V3_ROD_CLOSING_SPEED = 0.6
#: 对手至少要在这个水平速度（格/秒）以上，才算「它在动」。
#: 否则「距离在缩短」可能只是自己在走过去——那种情况该 approach，不该放竿。
TEACHER_V3_ROD_MIN_SPEED = 1.0

#: 金苹果：**只在绝对安全时吃**。血 ≤ 6、对手在 8 格外、且对手没有在接近。
#: 2026-11-21 真机 12 局的教训：原来的「血 ≤12 且 >4 格就吃」让老师 46% 的帧卡在吃苹果上，
#: 一局 64 步里有 31 步在原地被砍死。吃一次 = 整整 1.6s 不能动、不能砍，
#: 换 4 点血；被砍中两刀就是净亏 20 血。所以门槛必须高到「对手根本打不到」才允许吃。
TEACHER_V3_EAT_HEALTH = 6.0
TEACHER_V3_EAT_DISTANCE = 8.0
#: 吃之前要求对手「没有在接近」：距离变化率大于该值（格/秒，负 = 接近）。
#: -1.0 表示「对手每秒靠近超过 1 格就不吃」。
TEACHER_V3_EAT_CLOSING_SPEED = -1.0
#: 一次 ``use{gapple}`` 按住 1.6s（协议 ``holdMs``）。
TEACHER_V3_EAT_HOLD_SECONDS = 1.6
#: 一局最多吃 1 个。血少时的正确做法是打得更凶抢先手，不是回血。
TEACHER_V3_EAT_MAX_PER_EPISODE = 1
#: 吃苹果期间对手压进到这个距离（格）就**立刻放弃**这一口去打。
#: 这是安全阀：万一自校准失效（``since_attack`` 不推进）导致锁存被拉长，
#: 它也能保证「对手贴上来时不会继续站着吃」——2026-11-21 就是死在这一点上。
TEACHER_V3_EAT_ABORT_DISTANCE = 5.0
#: 两次吃之间的最小间隔（秒）：防止「苹果没吃完就每帧重发」的活锁（v2 实测踩过）。
TEACHER_V3_EAT_MIN_INTERVAL_SECONDS = 8.0

#: 格挡：只在对手 ≤ 2.6 格（一定打得到我们）且**自己的刀还在冷却**时。
#: 2026-11-21 的改法：原来要等「对手在挥刀」的真信号，而协议里没有这个字段
#: （``entities[]`` 连 ``onGround`` 都没有），12 局里只格挡了 4 帧。
#: 现在改成「死时间无条件减伤」——冷却期间那一刀本来就没伤害，挡一下是纯赚。
TEACHER_V3_BLOCK_DISTANCE = 2.6
#: 一次格挡事件最多连发几个 ``block`` 宏动作（1 个 = 按住右键 300ms）。
#: 只发 1 个，保证在下一次 attack 之前右键已经松开（0.204s/步下不会撞车）。
TEACHER_V3_BLOCK_MAX_TICKS = 1
#: 两次格挡事件的最小间隔（秒）：0.45s ≈ 一个攻击周期，保证一周期最多挡一次。
TEACHER_V3_BLOCK_COOLDOWN_SECONDS = 0.45
#: 一次 ``block`` 会按住右键这么久（``decode_action`` 里的 ``holdMs``）。
#: 只在「剩余冷却 ≥ 这个值」时才挡，保证右键在下一次 attack 之前松开。
TEACHER_V3_BLOCK_HOLD_SECONDS = 0.3
#: 「对手正在挥刀」的判定窗口（秒）。用 ``meta["opponent_since_attack"]`` 或
#: ``meta["opponent_swinging"]``；都没有时退化为「对手在空中且在下落（= 跳劈的出手帧）」。
TEACHER_V3_BLOCK_SIGNAL_SECONDS = 0.35

#: 追击：对手在逃跑时持续 approach，但自己不要压到这么近（格）以免被反打。
TEACHER_V3_CHASE_MIN_DISTANCE = 2.2

#: 反塌陷：最近多少 tick 的动作直方图。
TEACHER_V3_DIVERSITY_WINDOW = 20
#: 单个动作占比超过这个值 → 换一个合理动作。
TEACHER_V3_DIVERSITY_SHARE = 0.6
#: 少于这么多 tick 不做多样性干预（样本太小，占比没有意义）。
TEACHER_V3_DIVERSITY_MIN_TICKS = 8

#: 距离变化率的 EMA 系数（用来判断「对手在接近还是在逃」）。
TEACHER_V3_CLOSE_RATE_ALPHA = 0.4
#: 判定为「在逃」的距离变化率（格/秒，正 = 双方在拉开）。
TEACHER_V3_SEPARATING_SPEED = 1.0


class TeacherV3State:
    """老师 v3 的**跨帧记忆**（每局一份）。

    与 :class:`TeacherV2State` 的区别：v3 的大部分规则都要「时间预算」（冷却还剩多少、
    这一跳跳了多久、鱼竿还有多久好），所以状态里存的主要是**帧号**而不是布尔闩锁。

    字段：

    - ``frame``：调用计数（= 第几个决策 tick）；
    - ``ticks_since_attack``：距上次**自己选择** attack 过了几 tick（观测 ``since_attack``
      坏了/被吞时的兜底时钟）；
    - ``last_attack_frame`` / ``last_crit_frame``：上次攻击 / 上次跳劈的帧号；
    - ``crit_jump_frame``：本次起跳的帧号（下落窗口 = 起跳后 6–10 tick）；
    - ``last_rod_frame`` / ``pending_sword_switch``：鱼竿冷却与「放完竿切回剑」；
    - ``strafe_direction`` / ``strafe_until_frame`` / ``strafe_span_index``：横移换向节拍；
    - ``eat_until_frame`` / ``eats`` / ``last_eat_frame``：金苹果（只在绝对安全时用）；
    - ``block_ticks`` / ``last_block_frame``：格挡预算；
    - ``close_rate`` / ``last_distance``：对手相对距离变化率（EMA，格/秒）；
    - ``frame_seconds_est`` / ``clock_samples``：从 ``self_since_attack`` 反推的真实步长；
    - ``last_since`` / ``attack_dry``：上一帧的 ``since_attack`` 与「上一刀被解码成追击」；
    - ``history``：最近 ``TEACHER_V3_DIVERSITY_WINDOW`` 个动作（反塌陷直方图）；
    - ``reasons``：本局各条规则的命中次数。
    """

    __slots__ = (
        "frame", "last_action", "history", "reasons", "decisions",
        "ticks_since_attack", "last_attack_frame", "last_crit_frame", "crit_jump_frame",
        "last_rod_frame", "pending_sword_switch",
        "strafe_direction", "strafe_until_frame", "strafe_span_index",
        "eat_until_frame", "eats", "last_eat_frame",
        "block_ticks", "last_block_frame",
        "last_distance", "close_rate",
        "frame_seconds_est", "clock_samples", "last_since", "attack_dry",
        "dry_until_frame", "last_target_health", "last_hit_frame",
        "hits_landed", "crit_hits",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "TeacherV3State":
        # 步长是**环境属性**，不是对局属性：跨局保留，否则每局前 4 步又退回默认值。
        frame_seconds_est = getattr(self, "frame_seconds_est", None)
        clock_samples = getattr(self, "clock_samples", 0)
        self.frame = 0
        self.last_action: int | None = None
        self.history: list[int] = []
        self.reasons: dict[str, int] = {}
        self.decisions = 0
        # 初始视为「冷却早已好」：第一帧就该砍（since_attack 的缺省 5.0 也是这个意思）。
        self.ticks_since_attack = 10 ** 6
        self.last_attack_frame: int | None = None
        self.last_crit_frame: int | None = None
        self.crit_jump_frame: int | None = None
        self.last_rod_frame: int | None = None
        self.pending_sword_switch = False
        self.strafe_direction = 0          # 0 = 左，1 = 右
        self.strafe_until_frame = 0
        self.strafe_span_index = 0
        self.eat_until_frame: int | None = None
        self.eats = 0
        self.last_eat_frame: int | None = None
        self.block_ticks = 0
        self.last_block_frame: int | None = None
        self.last_distance: float | None = None
        self.close_rate = 0.0
        #: 从 ``self_since_attack`` 反推的真实步长（秒/帧）；``None`` = 还没测准。
        self.frame_seconds_est: float | None = frame_seconds_est
        self.clock_samples = clock_samples
        #: 上一帧观测到的 ``self_since_attack``（秒），用来量增量与识别「空挥」。
        self.last_since: float | None = None
        #: 上一刀被解码成了「追击」（没真挥出去）——这一帧不要急着重砍。
        self.attack_dry = False
        #: 「空挥」后的收紧窗口截止帧（见 TEACHER_V3_DRY_MARGIN）。
        self.dry_until_frame = 0
        #: 上一帧的对手血量 / 最近一次**真正命中**的帧号（按血量掉幅判定）。
        self.last_target_health: float | None = None
        self.last_hit_frame: int | None = None
        #: 本局真正命中次数与其中暴击次数（用来算真实暴击率，比 reason 里的标签可靠）。
        self.hits_landed = 0
        self.crit_hits = 0
        return self


def _v3_options(meta: Any, state: TeacherV3State | None) -> dict:
    """把 ``meta`` 规范成 v3 的决策上下文（全部可缺省）。"""
    data: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    chosen = state if isinstance(state, TeacherV3State) else data.get("state")
    if not isinstance(chosen, TeacherV3State):
        chosen = _DEFAULT_V3_STATE
    frame_seconds = data.get("frame_seconds")
    reach = data.get("reach")
    return {
        "state": chosen,
        "frame_seconds": (TEACHER_V3_FRAME_SECONDS if frame_seconds is None
                          else max(1e-6, float(frame_seconds))),
        #: 调用方显式给了 frame_seconds 就钉死，不让自校准覆盖（离线回放 / 单测要确定性）。
        "frame_seconds_pinned": frame_seconds is not None,
        "reach": TEACHER_V3_REACH if reach is None else float(reach),
        "rod_ready": bool(data.get("rod_ready", True)),
        "gapples": data.get("gapples"),
        "opponent_since_attack": data.get("opponent_since_attack"),
        "opponent_swinging": data.get("opponent_swinging"),
    }


def _v3_sync_clock(state: TeacherV3State, obs: np.ndarray, options: dict) -> float:
    """自校准「一帧多少秒」+ 识别「空挥」，返回本帧应当使用的 ``frame_seconds``。

    为什么需要校准（2026-11-21 真机 12 局的根因之一）：``GladiatorEnv`` 的推进是
    ``sleep(action_ticks / control_hz)``，实测一帧远大于 50ms。把 0.05 写死会让所有
    「秒 → tick」的换算一起错：1.6s 的吃苹果锁存实际锁 3.2s（甚至更久），
    跳劈窗口挪到落地之后（所以 12 局里 crit_jump 一次都没出现）。

    ``self_since_attack`` 是环境用 ``time.monotonic()`` 算的**墙钟秒**，与决策频率无关。
    上一帧没砍时，它的增量就是真实的一帧时长——几帧之内就能测准，之后用 EMA 抗抖动。

    顺带识别「空挥」：上一帧我们选了 attack、而这一帧 ``since_attack`` **变大**了，
    说明环境没把它记成攻击（``decode_action`` 在距离 > reach−0.15 时会把 attack 降级成
    「先靠近」）。此时 ``since_attack`` 不归零 → ``ready`` 永远为真 → 我们会每帧都发一刀
    空挥，并且每次都白占 2 帧 w-tap。``attack_dry`` 用来给下一帧加一道更严的距离门槛。
    """
    since = _get(obs, IDX_SELF_SINCE_ATTACK) * 5.0
    fallback = options["frame_seconds"]
    prev = state.last_since

    # --- 真正命中检测：对手血量掉幅（观测里的 target_health 是 /20 归一化）---
    target_health = _get(obs, IDX_TARGET_HEALTH) * HEALTH_SCALE
    prev_target = state.last_target_health
    if prev_target is not None and target_health > 0.0:
        drop = prev_target - target_health
        if drop >= TEACHER_V3_HIT_EPS:
            state.last_hit_frame = state.frame
            state.hits_landed += 1
            if drop >= TEACHER_V3_CRIT_DAMAGE:
                state.crit_hits += 1
    # 对手满血/被补满时不要把血量抬高当成命中；但重生了要重置基线。
    state.last_target_health = target_health if target_health > 0.0 else None

    state.attack_dry = bool(
        state.last_action == ACTION_ATTACK
        and prev is not None
        and since > prev + 1e-3)

    if (prev is not None and state.last_action != ACTION_ATTACK
            and 0.0 < since < 4.99 and 0.0 <= prev < 4.99):
        delta = since - prev
        if TEACHER_V3_CLOCK_MIN_SECONDS <= delta <= TEACHER_V3_CLOCK_MAX_SECONDS:
            state.clock_samples += 1
            if state.frame_seconds_est is None:
                state.frame_seconds_est = delta
            else:
                state.frame_seconds_est = 0.5 * state.frame_seconds_est + 0.5 * delta
    state.last_since = since

    if options.get("frame_seconds_pinned"):
        chosen = fallback                     # 调用方显式钉死（离线回放 / 单测）
    elif state.clock_samples >= TEACHER_V3_CLOCK_MIN_SAMPLES and state.frame_seconds_est:
        chosen = float(state.frame_seconds_est)
    else:
        chosen = fallback

    if state.attack_dry:
        # 上一刀被降级成「先靠近」：接下来这段时间要求再压近 TEACHER_V3_DRY_MARGIN 才出手。
        state.dry_until_frame = state.frame + max(
            1, int(round(TEACHER_V3_DRY_SECONDS / chosen)))
        # 空挥没有击退，也就没有 w-tap 的必要：撤掉上一刀的 w-tap，直接压上。
        if (state.last_attack_frame is not None
                and state.frame - state.last_attack_frame <= 1):
            state.last_attack_frame = None
    return chosen


def _v3_view(obs: np.ndarray, options: dict) -> dict:
    """把观测 + 状态折算成这一帧的**判定结果**（纯读，不改状态）。

    所有规则条件都在这里算完，``_decide_v3`` 只剩下「按优先级取第一个成立的」。
    这样单测可以直接断言某个条件，而不必从动作反推。
    """
    state: TeacherV3State = options["state"]
    frame_seconds: float = options["frame_seconds"]
    reach: float = options["reach"]
    frame = state.frame

    health = _get(obs, IDX_SELF_HEALTH) * HEALTH_SCALE
    distance = _get(obs, IDX_TARGET_DISTANCE) * DISTANCE_SCALE
    has_target = _get(obs, IDX_HAS_TARGET) >= 0.5
    in_reach = _get(obs, IDX_IN_REACH) >= 0.5
    self_ground = _get(obs, IDX_SELF_ON_GROUND) >= 0.5
    target_ground = _get(obs, IDX_TARGET_ON_GROUND) >= 0.5
    target_health = _get(obs, IDX_TARGET_HEALTH) * HEALTH_SCALE
    since_attack = _get(obs, IDX_SELF_SINCE_ATTACK) * 5.0        # 秒
    opponent_speed = _get(obs, IDX_OPPONENT_SPEED) * 10.0         # 格/秒
    opponent_vy = _get(obs, IDX_OPPONENT_VY) * 5.0                # 格/秒

    gapples = options["gapples"]
    if gapples is None:
        gapples = _get(obs, IDX_SELF_GAPPLES) * GAPPLE_SCALE

    # --- 冷却是否就绪：观测（墙钟秒）优先，自己数的 tick 兜底 ---
    # 两个信号取「或」而不是「与」：观测被吞（攻击被运行时降级成追击）时 tick 会误判成
    # 「已冷却」、观测会正确地说「还没挥过」——只要有一个说好了就砍，绝不因为计时分歧漏刀。
    cooldown = TEACHER_V3_ATTACK_COOLDOWN
    # 注意这里的 +0.5 帧：tick 兜底时钟只能**晚**不能早。早一帧会导致这一刀被无敌帧吞掉
    # （运行时仍会下发，环境仍会把 since_attack 归零），于是下一轮又早一帧 —— 永久性
    # 半速输出。宁可晚半帧。
    # 冷却判定：**优先用「距上次真正命中」**（血量掉幅），拿不到命中信号时退回
    # 「距上次攻击动作」（since_attack）。
    # 为什么要换：``since_attack`` 在空挥（被解码/运行时降级成追击）时也会归零，
    # 于是每次空挥都白等 0.5s；而对手是每步都按 attack 的，贴身窗口它一个不落。
    if state.last_hit_frame is not None:
        elapsed_hit = (frame - state.last_hit_frame) * frame_seconds
        # 血量掉幅要晚一帧才进观测，所以阈值减掉 0.75 帧补回来。
        need = max(0.0, cooldown - frame_seconds * 0.75)
        ready = elapsed_hit >= need
        steps_to_ready = max(0, int(math.ceil((need - elapsed_hit) / frame_seconds - 1e-6)))
    else:
        tick_ready = (state.clock_samples >= TEACHER_V3_CLOCK_MIN_SAMPLES
                      and state.ticks_since_attack * frame_seconds >= cooldown + frame_seconds * 0.5)
        ready = since_attack >= cooldown or tick_ready
        if since_attack >= 5.0:
            # 哨兵值 = 本局还没挥过刀：冷却相位未知，别在这个时刻规划跳劈（直接砍）。
            steps_to_ready = 99
        else:
            steps_to_ready = max(0, int(math.ceil((cooldown - since_attack) / frame_seconds - 1e-6)))
    #: 距离余量后的真实可打击范围（见 TEACHER_V3_ATTACK_MARGIN 的注释）。
    #: 上一刀被解码成「追击」时临时收紧 —— 说明 3.7 格左右这个边界在真机上打不出去。
    extra_margin = (TEACHER_V3_DRY_MARGIN if frame < state.dry_until_frame else 0.0)
    in_attack_range = bool(
        in_reach and distance <= reach - TEACHER_V3_ATTACK_MARGIN - extra_margin)

    # --- 对手「正在挥刀」：真信号优先，否则用「在空中且下落」当代用 ---
    swinging = options["opponent_swinging"]
    opponent_since = options["opponent_since_attack"]
    if swinging is not None:
        threat = bool(swinging)
    elif opponent_since is not None:
        threat = float(opponent_since) <= TEACHER_V3_BLOCK_SIGNAL_SECONDS
    else:
        # 没有真信号时只能猜：对手在空中且竖直速度为负 = 跳劈的下落期 = 它这帧会出手。
        # 这个代用信号偏保守（命中率不高但几乎不会误报），所以格挡还额外要求「冷却没好」，
        # 保证误判的代价只是「本来也打不出伤害的一帧」。
        threat = (not target_ground) and opponent_vy < 0.0

    # --- 跳劈窗口 ---
    crit_window_min = max(1, int(round(TEACHER_V3_CRIT_ATTACK_MIN_SECONDS / frame_seconds)))
    crit_window_max = max(crit_window_min,
                          int(round(TEACHER_V3_CRIT_ATTACK_MAX_SECONDS / frame_seconds)))
    crit_window_open = (
        state.crit_jump_frame is not None
        and crit_window_min <= frame - state.crit_jump_frame <= crit_window_max)
    crit_cooldown_ok = (
        state.last_crit_frame is None
        or (frame - state.last_crit_frame) * frame_seconds >= TEACHER_V3_CRIT_COOLDOWN_SECONDS)
    # 起跳时机：只在「下一步就是出手步」时起跳（真机实测暴击窗口就是第 1 步）。
    # 注意：**不要**依赖 ``target_on_ground``。协议里的 ``entities[]`` 没有 ``onGround``
    # 字段（只有 id/kind/name/pos/velocity/distance/heldItem），所以观测第 12 维在真机上
    # 恒为 0。2026-11-21 的 12 局里 crit_jump 一次都没出现，就是因为这里要求了它。
    # 起跳的**唯一**距离条件：这一帧本来就在出手距离内（只是刀还在冷却）。
    # 这样「跳了却打不到」= 0：跳完下一步一定还在原来那个距离附近，一定砍得出去。
    #: 起跳窗口：``steps_to_ready`` 为 0 或 1 都行——两者都保证「起跳后**下一步**就是出手步」，
    #: 而实测暴击窗口正是「跳后第 1 步」。允许 0 是必须的：命中后的击退会把双方推开，
    #: 我们通常要 2 步才重新进射程，而那时冷却**已经好了**（steps_to_ready = 0）。
    #: 只允许 1 的话，几乎永远等不到「进射程」和「还差 1 步」同时成立——这就是
    #: 前几版 crit_jump 55 次只换到 12–29 次 crit_attack 的原因。
    crit_arm_ok = bool(
        state.crit_jump_frame is None
        and crit_cooldown_ok
        and distance <= reach - TEACHER_V3_ATTACK_MARGIN + TEACHER_V3_CRIT_ARM_SLACK
        and distance >= TEACHER_V3_CRIT_MIN_DISTANCE
        and crit_window_min <= steps_to_ready <= crit_window_max)

    # --- 格挡预算 ---
    block_cooldown_ok = (
        state.last_block_frame is None
        or (frame - state.last_block_frame) * frame_seconds >= TEACHER_V3_BLOCK_COOLDOWN_SECONDS)
    # 死时间减伤：刀在冷却 + 对手够得到我们 → 挡（这一帧砍了也没有伤害，挡是纯赚）。
    # 不再要求「对手在挥刀」的真信号：协议里没有这个字段，等它等于永远不挡。
    block_ok = bool(
        distance <= TEACHER_V3_BLOCK_DISTANCE
        and not ready
        # 右键要按满 300ms，必须保证在下一刀之前松开；剩余冷却不足就别挡（免得和 attack 撞车）。
        and cooldown - since_attack >= TEACHER_V3_BLOCK_HOLD_SECONDS
        and state.block_ticks < TEACHER_V3_BLOCK_MAX_TICKS
        and block_cooldown_ok)

    # --- 金苹果：只在「对手根本打不到」时 ---
    eat_available = bool(gapples and float(gapples) > 0.0)
    eat_cooldown_ok = (
        state.last_eat_frame is None
        or (frame - state.last_eat_frame) * frame_seconds >= TEACHER_V3_EAT_MIN_INTERVAL_SECONDS)
    can_eat = bool(eat_available
                   and state.eats < TEACHER_V3_EAT_MAX_PER_EPISODE
                   and eat_cooldown_ok)
    # 三个条件缺一不可：血真的低了、对手在 8 格外、对手没有在接近。
    # 任何一条不满足就继续打——血少时抢输出比回血划算（吃一次 1.6s，被砍两刀就是净亏 20 血）。
    safe_to_eat = bool(
        can_eat
        and health <= TEACHER_V3_EAT_HEALTH
        and distance > TEACHER_V3_EAT_DISTANCE
        and state.close_rate >= TEACHER_V3_EAT_CLOSING_SPEED)

    # --- 鱼竿 ---
    rod_cooldown_ok = (
        state.last_rod_frame is None
        or (frame - state.last_rod_frame) * frame_seconds >= TEACHER_V3_ROD_COOLDOWN_SECONDS)
    rod_ok = bool(
        options["rod_ready"] and rod_cooldown_ok
        and TEACHER_V3_ROD_MIN_DISTANCE <= distance <= TEACHER_V3_ROD_MAX_DISTANCE
        # 对手正在接近：距离在缩短，且它自己在动（不是我们单方面走过去）。
        and state.close_rate <= -TEACHER_V3_ROD_CLOSING_SPEED
        and opponent_speed >= TEACHER_V3_ROD_MIN_SPEED)

    # --- w-tap / 追击 ---
    # w-tap 按秒折算：20Hz 下等于 2 tick，换到更慢的控制频率也还是 0.1s 的「松手」。
    wtap_ticks = max(1, int(round(TEACHER_V3_WTAP_SECONDS / frame_seconds)))
    wtap_active = bool(
        state.last_attack_frame is not None
        and 0 <= frame - state.last_attack_frame <= wtap_ticks)
    separating = state.close_rate >= TEACHER_V3_SEPARATING_SPEED

    return {
        "frame": frame,
        "frame_seconds": frame_seconds,
        "reach": reach,
        "health": health,
        "target_health": target_health,
        "distance": distance,
        "has_target": has_target,
        "in_reach": in_reach,
        "in_attack_range": in_attack_range,
        "self_ground": self_ground,
        "target_ground": target_ground,
        "since_attack": since_attack,
        "ready": ready,
        "swinging": threat,
        "gapples": float(gapples or 0.0),
        "attack_cooldown": cooldown,
        "crit_window_open": crit_window_open,
        "crit_window_min": crit_window_min,
        "crit_window_max": crit_window_max,
        "crit_arm_ok": crit_arm_ok,
        "block_ok": block_ok,
        "can_eat": can_eat,
        "safe_to_eat": safe_to_eat,
        "rod_ok": rod_ok,
        "wtap_active": wtap_active,
        "separating": separating,
        "opponent_speed": opponent_speed,
        "in_band": TEACHER_V3_STRAFE_MIN <= distance <= TEACHER_V3_STRAFE_MAX,
        "too_close": distance < TEACHER_V3_TOO_CLOSE,
    }


def _v3_strafe(state: TeacherV3State, view: dict) -> int:
    """横移换向节拍：每次持续 0.4–0.6s，到点换向（不要每帧抖）。

    0.204s/步时是 2–3 步；20Hz 时是 8–12 tick。用秒而不是帧，是为了同一份老师在
    两种控制频率下行为一致。
    """
    frame = view["frame"]
    frame_seconds = view["frame_seconds"]
    if frame >= state.strafe_until_frame:
        state.strafe_direction = 1 - state.strafe_direction
        low, high = TEACHER_V3_STRAFE_SPAN_SECONDS
        span_seconds = low + (high - low) * (state.strafe_span_index % 2)
        state.strafe_span_index += 1
        state.strafe_until_frame = frame + max(1, int(round(span_seconds / frame_seconds)))
    return ACTION_STRAFE_LEFT if state.strafe_direction == 0 else ACTION_STRAFE_RIGHT


def _v3_fallback_move(state: TeacherV3State, view: dict) -> int:
    """反塌陷时的「另一个合理动作」：远了就压上，近了就横移（换向）。"""
    if view["distance"] > TEACHER_V3_APPROACH_DISTANCE:
        return ACTION_APPROACH
    state.strafe_until_frame = 0            # 强制换向
    return _v3_strafe(state, view)


def _v3_action_is_legit(action: int, view: dict) -> bool:
    """这个动作在当前局面下「占着 60% 以上的帧」是不是合理的。

    只有「靠近」和「横移」有合理的长期占比：前者说明对手在逃、后者说明在交战线
    上绕圈——两者都是战术，不是塌陷。其余动作（idle / 连跳 / 连挡 / 反复放竿吃苹果）
    一旦占满窗口就是退化，必须换掉。
    """
    if action == ACTION_ATTACK:
        return True                          # 攻击是唯一有正收益的动作，永远不算塌陷
    if action == ACTION_APPROACH:
        # 压上永远合理：这是唯一既能进入射程、又能为跳劈留余量的动作。
        return True
    if action in (ACTION_STRAFE_LEFT, ACTION_STRAFE_RIGHT):
        return view["in_band"]
    return False


def _v3_diversify(state: TeacherV3State, action: int, view: dict) -> int:
    """规则 h：反塌陷——某个动作占最近窗口 60% 以上就换一个合理动作。

    为什么只换「不合理」的那些：v2 的教训是「任何一次把 attack 换成别的动作都是净亏」，
    所以 attack 永不参与替换；approach 在对手确实很远时也保留（追人本来就该一直压）。
    BC 策略那种「approach 120 步 / attack 0 步」的塌陷，在这里会被
    「approach 占比过高 + 已经够近了」这一条直接掐断。
    """
    if len(state.history) < TEACHER_V3_DIVERSITY_MIN_TICKS:
        return action
    recent = state.history[-TEACHER_V3_DIVERSITY_WINDOW:]
    share = recent.count(action) / float(len(recent))
    if share <= TEACHER_V3_DIVERSITY_SHARE:
        return action
    if _v3_action_is_legit(action, view):
        return action
    return _v3_fallback_move(state, view)


def _v3_finish(state: TeacherV3State, action: int, view: dict, *,
               reason: str, crit: bool = False) -> tuple[int, dict]:
    """收尾：更新所有时钟/直方图并返回 ``(动作, 诊断)``。"""
    frame = state.frame
    frame_seconds = view["frame_seconds"]

    state.ticks_since_attack += 1
    if action == ACTION_ATTACK:
        state.ticks_since_attack = 0
        state.last_attack_frame = frame
        if crit:
            state.last_crit_frame = frame
            state.crit_jump_frame = None
    elif action == ACTION_BLOCK:
        state.block_ticks += 1
        state.last_block_frame = frame
    else:
        state.block_ticks = 0

    # 跳劈窗口过了就作废（这一跳没打出暴击，等下一轮）。
    if (state.crit_jump_frame is not None
            and frame - state.crit_jump_frame > view["crit_window_max"]):
        state.crit_jump_frame = None

    state.history.append(action)
    if len(state.history) > TEACHER_V3_DIVERSITY_WINDOW:
        del state.history[:-TEACHER_V3_DIVERSITY_WINDOW]

    # 距离变化率（格/秒）：正 = 在拉开，负 = 在接近。EMA 抑制单帧抖动。
    if state.last_distance is not None and frame > 0:
        raw_rate = (view["distance"] - state.last_distance) / frame_seconds
        raw_rate = max(-20.0, min(20.0, raw_rate))
        alpha = TEACHER_V3_CLOSE_RATE_ALPHA
        state.close_rate = alpha * raw_rate + (1.0 - alpha) * state.close_rate
    state.last_distance = view["distance"]

    if action == ACTION_USE_ROD:
        state.last_rod_frame = frame
        state.pending_sword_switch = True     # 下一帧切回剑，否则下一刀是空手挥
    state.last_action = action
    state.frame = frame + 1
    state.decisions += 1
    state.reasons[reason] = state.reasons.get(reason, 0) + 1
    return action, {"reason": reason, "crit": bool(crit), "ready": view["ready"]}


def _v3_start_eat(state: TeacherV3State, view: dict, reason: str) -> tuple[int, dict]:
    """开始吃金苹果，并按 ``holdMs`` 锁住接下来 1.6s（防止每帧重发的活锁）。"""
    hold_ticks = max(1, int(round(TEACHER_V3_EAT_HOLD_SECONDS / view["frame_seconds"])))
    state.eats += 1
    state.last_eat_frame = view["frame"]
    state.eat_until_frame = view["frame"] + hold_ticks
    return _v3_finish(state, ACTION_EAT, view, reason=reason)


def _decide_v3(obs: np.ndarray, options: dict) -> tuple[int, dict]:
    """v3 的决策核心（纯函数 + 显式状态），返回 ``(动作, 诊断)``。

    判定顺序（顺序本身就是策略，改顺序会改变强度）：

    0. 看不到人 → idle；
    1. 上一帧放了鱼竿 → 切回剑（1 tick，否则下一刀是空手挥）；
    2. **攻击**：在射程内且冷却就绪 → attack。这条压在所有规则之上，连吃苹果锁存
       也会被打断（真机 12 局就是被「锁存期间继续站着吃」拖死的）。
       唯一例外是跳劈还在上升期时压 1–3 tick 等暴击窗口，见步骤 2 的注释；
    3. 吃苹果的锁存窗口内 → idle（此时砍不到人，按别的键只会打断 ``use{gapple}``）；
    4. 跳劈：**还差 2 步到冷却**时起跳（0.204s/步 = 命中后的那一步），再隔一步砍
       ——那一步正在下落，1.8 才认暴击；
    5. 贴脸（<2.0 格）且对手在下落 → 退一步拉开距离；
    6. 格挡：冷却没好 + 对手 ≤2.6 格 + 右键能在下一刀前松开；
    7. 金苹果：血 ≤6 **且**对手在 8 格外 **且**没在接近才吃（每局最多 1 个）；
    8. 鱼竿：4–6.5 格 + 冷却好 + 对手在接近；
    9. 距离控制：>3.2 压上 / 3.0–3.2 压半步进理想线 / 对手在逃就追 / 其余横移；
    10. 反塌陷：占比过高的动作换掉（攻击永不参与替换）。
    """
    state: TeacherV3State = options["state"]
    # 先校准时钟：后面所有「秒 → tick」的换算都用这一帧测出来的步长。
    options = dict(options)
    options["frame_seconds"] = _v3_sync_clock(state, obs, options)
    view = _v3_view(obs, options)
    frame = view["frame"]

    # --- 0. 没有目标 ---
    if not view["has_target"]:
        state.crit_jump_frame = None
        state.pending_sword_switch = False
        return _v3_finish(state, ACTION_IDLE, view, reason="no_target")

    # --- 1. 放完鱼竿切回剑 ---
    if state.pending_sword_switch:
        state.pending_sword_switch = False
        return _v3_finish(state, ACTION_SWITCH_SWORD, view, reason="rod_switch_back")

    # --- 2. 攻击（冷却就绪 + 在射程内）：唯一有正收益的动作 ---
    # 注意这一条**排在吃苹果锁存之前**：锁存期间只要进入射程且冷却好了，就中断吃苹果去砍。
    # 4 点血换一刀是划算的，而站在原地被砍是纯亏（真机 12 局就是这么输的）。
    if view["in_attack_range"] and view["ready"]:
        # **不在这里为暴击多等一步**（2026-11-21 第三轮）：实测「等一步换 1.5 倍」
        # 的账是错的——0.204s/步下等一步会让整个循环从 3 步变 4 步，出手率 0.336 → 0.279
        # 刀/步，净伤害反而下降（0.70 → 0.57 hp/步）。冷却一到就砍，暴击靠**提前起跳**
        # 赚（起跳发生在冷却的死时间里，本来就不占出手机会）。
        # 只按窗口判定是否「这一刀在下落期」：``self_on_ground`` 在真机上不可靠
        # （观测是上一帧快照，起跳后 0.2 步仍可能报 true），而窗口是按秒算的、可验证。
        crit = bool(view["crit_window_open"])
        state.eat_until_frame = None        # 中断吃苹果（这一刀比 4 点血值）
        action, info = _v3_finish(state, ACTION_ATTACK, view,
                                  reason="crit_attack" if crit else "attack", crit=crit)
        return action, info

    # --- 2.5 起跳（只在冷却的死步里，而且要够近才跳）---
    # 2026-11-21 第五轮实测：把起跳放到攻击之前时暴击率上到 46%，但出手量从 0.38 掉到
    # 0.26 刀/步、且一半的「暴击」因为距离被推开而砍空，净伤害反而下降。
    # 结论：**攻击优先，起跳只吃死时间**，并且只在近距离起跳（跳完那一步还够得着）。
    if view["crit_arm_ok"]:
        state.crit_jump_frame = frame
        return _v3_finish(state, ACTION_JUMP, view, reason="crit_jump")

    # --- 3. 吃苹果的锁存窗口（此时肯定够不到人，按别的键只会打断 use{gapple}）---
    if state.eat_until_frame is not None and frame < state.eat_until_frame:
        if view["distance"] <= TEACHER_V3_EAT_ABORT_DISTANCE:
            # 对手压上来了：这一口不要了，立刻转入迎战（安全阀，见常量注释）。
            state.eat_until_frame = None
        else:
            return _v3_finish(state, ACTION_IDLE, view, reason="eating")

    # --- 4.5 贴脸被压：先退出跳劈距离（优先级高于格挡） ---
    # 1.6 格以内站着挡是没有意义的：对手的每一刀必中，而我们的 3.99 格射程优势完全用不上。
    # 退一步既拉开到交战线、又让它的暴击落空，比原地格挡（只减伤不减节奏）更好。
    if view["too_close"] and view["swinging"]:
        return _v3_finish(state, ACTION_RETREAT, view, reason="retreat_too_close")

    # --- 5. 格挡（只在「这一帧砍了也没伤害」时） ---
    if view["block_ok"]:
        return _v3_finish(state, ACTION_BLOCK, view, reason="block")

    # --- 6. 金苹果（只在绝对安全时：血≤6 + 对手 8 格外 + 没在接近） ---
    if view["safe_to_eat"]:
        return _v3_start_eat(state, view, "eat_safe")

    # --- 7. 鱼竿 ---
    if view["rod_ok"]:
        return _v3_finish(state, ACTION_USE_ROD, view, reason="rod")

    # --- 8. 距离控制：不打人的时候**永远压上** ---
    # 2026-11-21 第五轮：真机数据表明横移/w-tap 是纯浪费——它们既不接近也不出手，
    # 每次占一步就等于把「进入射程的时间」推迟 0.2s。上一版有 17% 的步数花在
    # strafe / wtap_strafe 上，而对手（BC）是每步都在靠近或攻击的，于是它每局比我们
    # 多打 35% 的命中（13.2 : 9.8 刀）。现在除了「贴脸被压要后退」之外，一律 approach。
    if view["too_close"] and view["swinging"]:
        return _v3_finish(state, ACTION_RETREAT, view, reason="retreat_too_close")
    action = ACTION_APPROACH
    reason = "chase" if view["separating"] else "approach"
    return _v3_finish(state, action, view, reason=reason)


#: ``heuristic_action_v3(obs)``（不传 state）时使用的默认状态。
#: 多局/多线程场景请用 :class:`TeacherV3` 实例，或显式传 ``state=``/``meta={"state": ...}``。
_DEFAULT_V3_STATE = TeacherV3State()


def reset_teacher_v3() -> None:
    """清空模块级默认状态（测试 / 新 episode 之间调用）。"""
    _DEFAULT_V3_STATE.reset()


def heuristic_action_v3(
    obs: np.ndarray,
    meta: Mapping[str, Any] | None = None,
    state: TeacherV3State | None = None,
) -> int:
    """老师 v3：把攻击冷却当资源，用死时间换位置、暴击与格挡。

    只读观测（+ 可选 ``meta`` / ``state``），返回动作编号。``state`` 优先于
    ``meta["state"]``，都没有时用模块级默认状态（``reset_teacher_v3()`` 清空）。

    与 v1/v2 的本质差别：**冷却没好时不发 attack**。1.8 受击无敌 500ms，
    连发 attack 的 10 帧里只有 1 帧有伤害；v3 把其余 9 帧换成位移/跳劈准备/格挡，
    于是每一刀都踩在冷却刚好的那一帧，且约一半是暴击（1.5 倍）。

    "冷却还剩多少"用观测 ``self_since_attack``（墙钟秒），**不写死帧数**：
    真机实测一帧不是 50ms（``GladiatorEnv`` 推进 = ``sleep(action_ticks/control_hz)``），
    所以 v3 会用它的增量自校准步长，把「秒 → tick」的换算全部对齐。

    规则表（判定顺序即优先级）：

    ====== ==================== ==========================================
    序号   规则                 条件 → 动作
    ====== ==================== ==========================================
    0      无目标               看不到人 → idle
    1      切回剑               上一帧放了鱼竿 → switch_sword
    2      攻击                 ``in_reach`` 且冷却好 → attack（最高优先级，不许被
                               strafe / approach / w-tap 抢走；连吃苹果锁存也中断）
    3      吃苹果锁存           ``frame < eat_until_frame`` → idle
    4      跳劈                 ≥2.0 格 + **还差 2–3 步到冷却** → jump
                               （0.204s/步下 = 命中后的那一步；不看 onGround，
                               协议 entities[] 没有这个字段）
    5      贴脸被压             <2.0 格且对手在下落 → retreat
    6      格挡                 ≤2.6 格 + 冷却没好 + 右键来得及松开 → block（1 次）
    7      金苹果               血≤6 **且**对手 8 格外 **且**没在接近 → eat（每局 1 个）
    8      鱼竿                 4.0–6.5 格 + 冷却好 + 对手在接近 → use_rod
    9      距离控制             >3.2 压上 / 3.0–3.2 压半步 / 对手在逃就追 /
                               其余横移
    10     反塌陷               某动作占最近 20 tick >60% 且不合理 → 换掉
    ====== ==================== ==========================================

    ``meta`` 支持的键：``state`` / ``frame_seconds`` / ``reach`` / ``rod_ready`` /
    ``gapples`` / ``opponent_since_attack`` / ``opponent_swinging``。
    传入 dict 时会写回 ``meta["reason"]``（本帧依据）与 ``meta["crit"]``（是否跳劈），
    便于复盘「是不是一直在后撤 / 一直在格挡」。
    """
    options = _v3_options(meta, state)
    action, info = _decide_v3(obs, options)
    if isinstance(meta, dict):
        meta["reason"] = info["reason"]
        meta["crit"] = info["crit"]
    return int(action)


class TeacherV3:
    """可复用的 v3 老师实例（自己持有一份 :class:`TeacherV3State`）。

    ``teacher = TeacherV3(); [teacher(obs) for obs in frames]`` —— 状态跨帧，
    每个 episode 开头调 :meth:`reset`（环境 ``reset()`` 会通过 ``reset_episode`` 回调）。
    """

    version = "v3"

    def __init__(self, **defaults: Any) -> None:
        self.state = TeacherV3State()
        self.defaults = dict(defaults)
        #: 已结束 episode 的原因统计（每次 :meth:`reset` 归档上一局）。
        self.episode_reasons: list[dict[str, int]] = []

    def reset(self) -> "TeacherV3":
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
        self.state.reset()
        return self

    def reset_episode(self) -> None:
        """环境 ``reset()`` 时的回调（``GladiatorEnv`` 约定）：归档上一局并清零状态。"""
        self.reset()

    @property
    def reasons(self) -> dict[str, int]:
        """当前 episode 各条规则的命中次数（``reason -> count``）。"""
        return dict(self.state.reasons)

    def flush(self) -> list[dict[str, int]]:
        """把「还没归档的当前 episode」也收进 ``episode_reasons``（对局结束时调用）。"""
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
            self.state.reasons = {}
        return list(self.episode_reasons)

    def __call__(self, obs: np.ndarray, meta: Mapping[str, Any] | None = None) -> int:
        merged: dict[str, Any] = dict(self.defaults)
        if isinstance(meta, Mapping):
            merged.update(meta)
        action = heuristic_action_v3(obs, merged, self.state)
        if isinstance(meta, dict):
            meta["reason"] = merged.get("reason", "")
            meta["crit"] = merged.get("crit", False)
        return action


def make_teacher(version: str = TEACHER_VERSION) -> Callable[..., int]:
    """按版本号取老师。

    ``"v1"`` → :func:`heuristic_action`（无状态函数）；
    ``"v2"`` → :class:`TeacherV2`；``"v3"`` → :class:`TeacherV3`。
    """
    key = str(version or TEACHER_VERSION).strip().lower()
    if key == "v1":
        return heuristic_action
    if key == "v2":
        return TeacherV2()
    if key == "v3":
        return TeacherV3()
    raise ValueError(f"未知老师版本: {version!r}（允许 {', '.join(SUPPORTED_TEACHER_VERSIONS)}）")


def collect_demos(
    env,
    episodes: int = 4,
    max_steps: int = 120,
    teacher: Callable[[np.ndarray], int] | None = None,
    gamma: float = 0.99,
    teacher_version: str = TEACHER_VERSION,
) -> list[tuple[np.ndarray, int, float]]:
    """用老师跑 ``episodes`` 局，返回 ``[(obs, action, discounted_return)]``。

    ``teacher`` 显式给出时直接用它；否则按 ``teacher_version`` 取
    （``"v1"`` 默认 / ``"v2"`` 见 :func:`heuristic_action_v2`）。
    v2 是有状态的，所以每局 ``env.reset()`` 之后会调 ``teacher.reset()``（若有）。

    回报按 **该局后续折扣回报** 计算（末步之后为 0）；单步环境不做 bootstrap。
    只支持单智能体环境（``single_agent=True``），否则动作契约不同，直接报错而不是瞎猜。
    """
    teacher = teacher if teacher is not None else make_teacher(teacher_version)
    samples: list[tuple[np.ndarray, int, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, _info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_demos 只支持 single_agent=True 的环境（obs 应为向量）")
        _reset_teacher(teacher)
        steps: list[tuple[np.ndarray, int, float]] = []
        done = False
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            action = int(teacher(vector))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            steps.append((vector, action, float(reward)))
            obs = next_obs
            if terminated or truncated:
                done = True
                break
        # 反向累加折扣回报；局末之后视为 0（terminated/truncated 都按终止处理）。
        running = 0.0
        returns = [0.0] * len(steps)
        for index in range(len(steps) - 1, -1, -1):
            running = steps[index][2] + gamma * running
            returns[index] = running
        for (vector, action, _reward), value in zip(steps, returns):
            samples.append((vector, action, float(value)))
        if not done:
            # 截断的对局同样可用（回报就是到截断为止），这里不做额外处理。
            pass
    return samples


def _reset_teacher(teacher: Callable[..., int]) -> None:
    """老师带状态（v2）时每局清一次；无状态老师（v1 函数）无操作。"""
    reset = getattr(teacher, "reset", None)
    if callable(reset):
        reset()


def collect_dagger_samples(
    env,
    policy,
    episodes: int = 4,
    max_steps: int = 120,
    teacher: Callable[[np.ndarray], int] | None = None,
    rng: np.random.Generator | None = None,
    teacher_version: str = TEACHER_VERSION,
) -> list[tuple[np.ndarray, int, float]]:
    """DAgger 的一轮采样：用**当前策略**跑，每步让老师给动作标签（P2.3）。

    与 :func:`collect_demos` 的区别是「谁在跑」：BC 用老师跑（数据分布 = 老师的状态分布），
    策略自己跑久了会走到老师从没去过的状态（covariate shift），DAgger 每轮都用
    当前策略的状态分布重新打标签，于是「策略越强、标签越贴它的实际处境」。

    返回结构与 :func:`collect_demos` 一致（``(obs, action, 0.0)``，第三个字段不用），
    这样可以直接喂给 :func:`behavior_clone`。
    """
    teacher = teacher if teacher is not None else make_teacher(teacher_version)
    rng = rng if rng is not None else np.random.default_rng(0)
    samples: list[tuple[np.ndarray, int, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, _info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_dagger_samples 只支持 single_agent=True 的环境（obs 应为向量）")
        _reset_teacher(teacher)
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            label = int(teacher(vector))          # 老师对**这个状态**的标签
            action = int(policy.act(vector, rng))  # 策略自己走，才能采到自己的状态分布
            samples.append((vector, label, 0.0))
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    return samples


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(params["W1"] @ x + params["b1"])
    p = _softmax(params["W2"] @ h + params["b2"])
    return h, p


def _logits(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(hidden, 未除温度的 logits)``。"""
    h = np.tanh(params["W1"] @ x + params["b1"])
    return h, params["W2"] @ h + params["b2"]


#: 软目标里「老师分布」的底噪概率。老师只给硬标签（一个动作编号），把它当成
#: 一个「label 上质量为 1、其余动作 1e-3」的分布再按 ``1/T`` 幂次缩放：
#: ``q = softmax(log(p_teacher) / T)``。于是 T=1 退化为 one-hot（与旧实现逐位等价），
#: T>1 变平（非 label 动作拿到更多质量）、T<1 变尖（非 label 动作更小）。
TEACHER_PROB_FLOOR = 1e-3


def teacher_target(action: int, n_actions: int, temperature: float = 1.0,
                   label_smoothing: float = 0.0) -> np.ndarray:
    """构造 BC 的软目标 ``q``（长度 ``n_actions`` 的概率向量）。

    - ``temperature == 1`` 且无 smoothing → 直接返回 one-hot（**与旧实现逐位等价**）；
    - ``temperature > 1``：老师分布变平（``softmax(log p / T)``），BC 不再把策略压死；
    - ``label_smoothing > 0``：``q = (1-eps)·q + eps/|A|``（与均匀分布按 eps 混合）。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing 必须在 [0, 1) 内，收到 {label_smoothing}")
    n = int(n_actions)
    if temperature == 1.0:
        q = np.zeros(n, dtype=np.float64)
        q[int(action)] = 1.0
    else:
        # 老师只有硬标签：把它当「label 上质量 1、其余 1e-3」的分布，再按 1/T 幂次缩放。
        base = np.full(n, TEACHER_PROB_FLOOR, dtype=np.float64)
        base[int(action)] = 1.0
        q = _softmax(np.log(base) / float(temperature))
    if label_smoothing > 0.0:
        # 与均匀分布混合：保留「没有绝对正确动作」的先验，压制 softmax 饱和。
        q = (1.0 - float(label_smoothing)) * q + float(label_smoothing) / n
    return q


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p + 1e-12)))


def behavior_clone(
    policy,
    samples: Sequence[tuple[np.ndarray, int, float]],
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 64,
    entropy_coef: float = 0.0,
    seed: int = 0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> dict:
    """交叉熵监督训练策略头（W1/b1/W2/b2），**不动价值头 Wv/bv**。

    返回 ``{"loss": [...每轮平均交叉熵...], "accuracy": 最终 teacher 动作命中率,
    "samples": 样本数, "epochs": 实际轮数, "temperature": T,
    "labelSmoothing": eps, "teacherEntropy": 老师分布的平均熵}``。

    ``entropy_coef>0`` 时在损失里减去 ``β·H(π)``（鼓励保留探索）。

    温度/平滑的作用（P2.3 后续）：硬标签 + 30 轮 Adam 会把 softmax 压到熵 0.01~0.03，
    之后 PPO 的 ratio 与熵正则都推不动。``temperature>1`` 让目标变软（交叉熵
    ``-Σ q·log p``），``label_smoothing>0`` 混入均匀分布，二者都保留策略的初始熵。
    ``T=1`` 且 ``eps=0`` 时软目标就是 one-hot，梯度 ``p - q`` 与旧实现一致。

    口径说明：这里的 ``p`` 是 ``softmax(未除温度的 logits)``（与旧实现一致），
    ``policy.temperature`` 只影响采样/PPO 侧；BC 的 ``temperature`` 是**独立参数**，
    两者不要混用（要整体调平请在 BC 用 temperature，PPO 侧用 ``--logit-temp``）。
    ``weights`` 给每条样本一个非负权重（默认全 1）：用于「赢的局权重大、输的局权重小」
    与「交手帧权重大、跑位帧权重小」这类择优筛选。权重按批内和归一化，
    因此全 1 时与原实现逐位一致。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing 必须在 [0, 1) 内，收到 {label_smoothing}")
    if weights is not None and len(weights) != len(samples):
        raise ValueError(f"weights 长度 {len(weights)} 与样本数 {len(samples)} 不一致")
    data: list[tuple[np.ndarray, int, float]] = []
    for index, (x, a, _ret) in enumerate(samples):
        w = 1.0 if weights is None else float(weights[index])
        if w < 0 or not np.isfinite(w):
            raise ValueError(f"第 {index} 条样本权重非法: {w}")
        data.append((np.asarray(x, dtype=np.float64).reshape(-1), int(a), w))
    result = {"loss": [], "accuracy": 0.0, "samples": len(data), "epochs": 0,
              "temperature": float(temperature),
              "labelSmoothing": float(label_smoothing),
              "teacherEntropy": 0.0,
              "totalWeight": float(sum(w for _x, _a, w in data))}
    if not data:
        return result

    rng = np.random.default_rng(seed)
    names = ("W1", "b1", "W2", "b2")
    m = {k: np.zeros_like(policy.params[k]) for k in names}
    v = {k: np.zeros_like(policy.params[k]) for k in names}
    step = 0
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    n_actions = policy.n_actions
    # 软目标只与「动作编号」有关：11 个动作最多 11 种，算一遍缓存起来。
    targets: dict[int, np.ndarray] = {}

    for _epoch in range(int(max(0, epochs))):
        order = rng.permutation(len(data))
        epoch_loss = 0.0
        for start in range(0, len(data), max(1, int(batch))):
            chunk = [data[i] for i in order[start:start + max(1, int(batch))]]
            grads = {k: np.zeros_like(policy.params[k]) for k in names}
            for x, action, w in chunk:
                h, logits = _logits(policy.params, x)
                p = _softmax(logits)
                q = targets.get(action)
                if q is None:
                    q = targets[action] = teacher_target(
                        action, n_actions, temperature, label_smoothing)
                epoch_loss += w * float(-np.sum(q * np.log(p + 1e-12)))
                dlogits = (p - q) * w
                if entropy_coef:
                    entropy = _entropy(p)
                    dlogits = dlogits - entropy_coef * w * p * (np.log(p + 1e-12) + entropy)
                grads["W2"] += np.outer(dlogits, h)
                grads["b2"] += dlogits
                dz = (policy.params["W2"].T @ dlogits) * (1.0 - h * h)
                grads["W1"] += np.outer(dz, x)
                grads["b1"] += dz
            n = float(sum(w for _x, _a, w in chunk))
            if n <= 0:
                continue
            step += 1
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            for key in names:
                g = grads[key] / n  # 最小化方向（交叉熵）
                m[key] = b1 * m[key] + (1.0 - b1) * g
                v[key] = b2 * v[key] + (1.0 - b2) * (g * g)
                policy.params[key] -= lr * (m[key] / bc1) / (np.sqrt(v[key] / bc2) + eps)
        result["loss"].append(round(epoch_loss / max(1e-9, sum(w for _x, _a, w in data)), 6))
        result["epochs"] += 1

    correct = 0
    weight_sum = 0.0
    entropy_sum = 0.0
    for x, action, w in data:
        _h, logits = _logits(policy.params, x)
        p = _softmax(logits)
        correct += w * float(int(np.argmax(p)) == action)
        weight_sum += w
        q = targets.get(action)
        if q is None:
            q = targets[action] = teacher_target(action, n_actions, temperature, label_smoothing)
        entropy_sum += w * max(0.0, _entropy(q))
    result["accuracy"] = round(correct / max(1e-9, weight_sum), 4)
    result["teacherEntropy"] = round(entropy_sum / max(1e-9, weight_sum), 6)
    policy.updates += int(result["epochs"])  # 与 PPO 的更新计数分开记，但保持单调
    return result


def teacher_agreement(policy, samples: Iterable[tuple[np.ndarray, int, float]]) -> float:
    """老师动作与策略 argmax 的一致率（BC 前后对比用）。"""
    total = 0
    correct = 0
    for x, action, _ret in samples:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        _h, p = _forward(policy.params, x)
        correct += int(int(np.argmax(p)) == int(action))
        total += 1
    return round(correct / total, 4) if total else 0.0
