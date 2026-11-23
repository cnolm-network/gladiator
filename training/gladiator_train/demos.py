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
- **v4** ``heuristic_action_v4``：跑在 **v2 动作空间（10 头）** 上，按
  ``docs/knowledge/pvp-1.8-mechanics.md`` §3 的招式表实现 wtap / block_hit / spacing /
  strafe / jump_crit 的**组合**（有剑走 block_hit、无剑走 wtap，二者互斥）。
  它返回 10 头数组而不是动作编号，见 :class:`TeacherV4`。

``meta`` 是 v2/v3 的「观测之外的状态」入口（鱼竿是否可用、对手多久前攻击过、一帧多少秒、
对手是否正在挥刀），缺省值都能跑，所以离线采集不需要额外接线。

为什么需要 v3（2026-11-21）：v1/v2 的「射程内每帧都砍」在**宏动作空间**里其实有一半是白砍——
1.8 的受击无敌帧是 500ms，而控制频率是 20Hz，于是每 10 帧里最多只有 1 帧真的能造成伤害。
v2 把这 9 帧空转成了「同一个 attack 再发一次」，v3 把它们换成走位、跳劈准备与格挡，
因此单位时间的**有效伤害**更高（暴击 +50%），而不是靠出手率堆出来的。

为什么需要 v2 的 BC（2026-11-22，G-710）：老师 v4 跑在 **v2 十头动作空间**上，它的标签
不是一个动作编号，而是「每个头选哪一档」。于是 BC 的数据契约按动作空间分两种：

========================  ==============================  ====================
动作空间                 标签                            第三字段
========================  ==============================  ====================
v1（``Discrete``）       动作编号 ``int``                该局折扣回报（不使用）
v2（``MultiDiscrete``）  10 头下标元组（:data:`V2_HEADS`）  样本权重（默认 1.0）
========================  ==============================  ====================

损失也随之变化：v1 是单头交叉熵；v2 是**逐头交叉熵按头平均**（每个头权重相同），
返回里多一个 ``perHeadAccuracy``。``intent`` ↔ 下标元组的翻译只发生在
:func:`intent_to_tuple` 一处，避免「老师说的意图」和「BC 训练的下标」各写一遍。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from gladiator_env.spaces import (
    ACTION_HEADS_V2, ACTION_NAMES, LOOK_PITCH_V2, LOOK_YAW_V2, MULTI_DISCRETE_V2,
    MOVE_VALUES_V2, OBS_DIM, OBS_FIELDS, SPRINT_RESET_VALUES_V2, USE_VALUES_V2,
    decode_intent,
)

#: 默认老师版本（v1 = 规则老师，行为与 P1.5 完全一致）。
TEACHER_VERSION = "v1"
#: 可选老师版本。v2 在 v1 之上加间距/鱼竿/金苹果/暴击跳/w-tap/格挡六条规则；
#: v3 把攻击冷却当资源（死时间走位/跳劈/格挡），见 ``heuristic_action_v3``；
#: v4 用 v2 动作空间（10 头）实现 1.8 招式的**组合**（wtap / block_hit / spacing /
#: strafe / jump_crit），见 ``heuristic_action_v4``。
SUPPORTED_TEACHER_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3", "v4")

#: v2 十头的名字，顺序与 ``ACTION_HEADS_V2`` 的声明顺序逐位一致
#: （``move, look_yaw, look_pitch, attack, use, jump, sneak, sprint, slot, sprint_reset``）。
#: 蒸馏样本里的「标签」就是这个顺序的下标元组，改顺序等于作废旧数据集。
V2_HEADS: tuple[str, ...] = tuple(ACTION_HEADS_V2)


def _v2_index(head: str, value: Any, values: Sequence[Any]) -> int:
    """把 intent 里的一档取值翻成头下标（字符串按等值、浮点按 1e-6 容差）。

    ``int`` 输入按「已经是下标」处理并校验范围——日志里偶尔会有
    「人写死的下标」与「语义值」混用，两种都收但越界立刻报错。
    """
    if isinstance(value, bool):
        # bool 是 int 的子类，但 head 表里没有 True/False（二值头走 _v2_binary）。
        raise ValueError(f"intent 的 {head} 不接受布尔值: {value!r}")
    if value is None:
        for index, candidate in enumerate(values):
            if candidate is None:
                return index
        raise ValueError(f"intent 的 {head} 不接受 None（合法 {tuple(values)!r}）")
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if not 0 <= index < len(values):
            raise ValueError(f"intent 的 {head} 下标越界: {index}（合法 0..{len(values) - 1}）")
        return index
    for index, candidate in enumerate(values):
        if isinstance(candidate, str) or isinstance(value, str):
            if str(candidate) == str(value):
                return index
            continue
        try:
            if abs(float(candidate) - float(value)) <= 1e-6:
                return index
        except (TypeError, ValueError):
            continue
    raise ValueError(f"intent 的 {head} 取值 {value!r} 不在 {tuple(values)!r} 里")


def _v2_binary(head: str, value: Any) -> int:
    """二值头（attack/jump/sneak/sprint）：``True/False`` 或 ``0/1`` → 下标。"""
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"intent 的 {head} 必须是布尔或 0/1，收到 {value!r}") from exc
    if number not in (0, 1):
        raise ValueError(f"intent 的 {head} 只能是 0/1，收到 {number}")
    return number


def intent_to_tuple(intent: Mapping[str, Any] | None) -> tuple[int, ...]:
    """把 teacher 的 `intent` 字典翻成 **10 个头的下标元组**（顺序 = :data:`V2_HEADS`）。

    两种输入都收：

    - ``{"intent": {...}, "reason": ...}``：老师 v4 写回 ``meta`` 的形态（也兼容日志/JSON）；
    - 直接是 ``decode_intent`` 的输出（``{"move": ..., "look": {...}, "attack": ...}``）。

    缺省值与 ``decode_intent`` 的对称规则一致：``look`` 缺省为不转视角、``use`` 缺省
    ``None``、``sprintReset`` 缺省 ``none``、``slot`` 缺省 0。**越界/未知取值一律抛
    ``ValueError``**，不做「悄悄退化成 idle」的兜底——蒸馏标签错了会污染整个策略头。

    为什么需要它：BC 的标签必须是「每头一个下标」，而老师与日志说的是**语义**
    （``move="forward"`` / ``attack=True``）。这个函数就是两者之间唯一的翻译点。
    """
    data: Any = intent
    if isinstance(data, Mapping) and "intent" in data:
        inner = data.get("intent")
        if isinstance(inner, Mapping):
            data = inner
    if not isinstance(data, Mapping):
        raise ValueError(f"intent_to_tuple 需要 dict，收到 {type(intent).__name__}")
    look = data.get("look")
    if look is None:
        look = {}
    if not isinstance(look, Mapping):
        raise ValueError(f"intent 的 look 必须是 dict，收到 {look!r}")
    picked: dict[str, int] = {
        "move": _v2_index("move", data.get("move", "idle"), MOVE_VALUES_V2),
        "look_yaw": _v2_index("look_yaw", look.get("dyaw", 0.0), LOOK_YAW_V2),
        "look_pitch": _v2_index("look_pitch", look.get("dpitch", 0.0), LOOK_PITCH_V2),
        "attack": _v2_binary("attack", data.get("attack", False)),
        "use": _v2_index("use", data.get("use"), USE_VALUES_V2),
        "jump": _v2_binary("jump", data.get("jump", False)),
        "sneak": _v2_binary("sneak", data.get("sneak", False)),
        "sprint": _v2_binary("sprint", data.get("sprint", False)),
        "slot": _v2_index("slot", data.get("slot", 0),
                          tuple(range(ACTION_HEADS_V2["slot"]))),
        "sprint_reset": _v2_index("sprint_reset", data.get("sprintReset", "none"),
                                  SPRINT_RESET_VALUES_V2),
    }
    return tuple(picked[head] for head in V2_HEADS)


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
# v4 额外要用的字段：自身竖直速度（跳劈时机）与出手冷却（0=可打，1=刚打完）。
IDX_SELF_VY = field_index("self_vertical_speed")
IDX_ATTACK_COOLDOWN = field_index("self_attack_cooldown")

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
#: 老师假定的射程（**冻结**）：v3/v4 老师与现有蒸馏标签都是按 3.99 训练的，
#: 改成 3.0 会改变标签分布 → 必须先重训再换。G-708 只改 BotProfile / BotConfig 的默认值，
#: 这里留 3.99 并在报告里标为「待重训后回调」。
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
#: 疾跑攻击的击退更大，会把对手推出 3 格；松一下 W 再压上去能让对手留在射程内。
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
    # 1.6 格以内站着挡是没有意义的：对手的每一刀必中，而我们的 3 格射程优势完全用不上。
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


# ============================================================================
# 老师 v4：1.8 招式的**组合**（v2 动作空间 / 10 头）
# ============================================================================
#
# 为什么需要 v4（``docs/knowledge/pvp-1.8-mechanics.md`` §4 第 1 条）：
# v1–v3 都跑在 v1 的 11 个宏动作上，那里**没有「重置疾跑」这个维度**——``approach``
# 的疾跑由 ``decode_action`` 按距离决定，「松一帧 W」根本无法表达。于是：
#   - teacher 演示不了 wtap，BC 学不到；
#   - 策略也就永远学不到「命中帧非疾跑」这件事（1.8 的 KB 大小完全由它决定）。
#
# G-707 在 v2 动作空间里加了 ``sprint_reset`` 头（none / release_w / block_hit），
# v4 就是**唯一**能用它的老师：它输出 10 头动作，而不是 v1 的动作编号。
#
# 两条互斥的对刀路线（视频原话：有刀用防砍、无刀用 wtap，**不要同时用**）：
#   - 有剑 → ``block_hit``（按住 W + 举剑 ≥300ms，兼吃 0.5× 伤害）；
#   - 无剑 → ``release_w`` 的两种用法：命中后 ``wtap_keep_combo``（让对手持续吃大 KB）、
#     受击后 ``wtap_reduce_kb``（减小自己的 KB，快速回到攻击距离）；**永不按右键**。
# 这条分叉不是「优化」，是机制：没剑时右键只会在服务端拿到 ``ITEM_MISSING``。
#
# 观测里**没有手持物品字段**（25 维里只有耐久，区分不出剑和苹果），所以「有没有剑」
# 必须由 ``meta["has_sword"]`` 给出；缺省 ``True``（训练服初始套装必带剑）。
# 真机可由 bot-runtime 的 ``heldItem.name`` 直接喂进来。
#
# 与 v3 的关系：v4 **不**复用 v3 的「冷却预算」逻辑，而是按 KB §3 的招式表逐条实现
# （``jump_crit`` 只看 ``vy<0`` 这个真信号、``spacing_hold`` 用 2.6–3.3 的带）。
# 两者可以各自独立跑真机做对照（``teacher_eval.py --teacher v3/v4``）。

#: 目标带（格）。§3 ``spacing_hold``：太近 S、太远 W。
TEACHER_V4_SPACING_MIN = 2.6
TEACHER_V4_SPACING_MAX = 3.3
#: 有剑时用 ``block_hit`` 的上界（格）。§3 ``block_hit``：双方距离 ≤3.0 且自己有剑。
TEACHER_V4_BLOCK_DISTANCE = 3.0
#: ``jump_crit`` 的距离上界（格）。§3：距离 ≤2.4 且冷却已好。
TEACHER_V4_CRIT_MAX_DISTANCE = 2.4
#: 贴脸下界：太近（<1.4 格）时起跳会跳过对手，先退一步。
TEACHER_V4_CRIT_MIN_DISTANCE = 1.4
#: 「起跳预备」的最小间隔（秒）：只在着地且冷却已好时跳一次，防止「一直跳、一直等不到下落」。
TEACHER_V4_JUMP_ARM_COOLDOWN = 1.0
#: 接战前的横移区间（格）。§3 ``strafe_poke``：距离 4–6 格。
TEACHER_V4_STRAFE_POKE_MIN = 4.0
TEACHER_V4_STRAFE_POKE_MAX = 6.0
#: 横移换向区间（秒）。``strafe_poke`` 每 0.4–0.8s 换向（幅度大、频率低）。
TEACHER_V4_STRAFE_POKE_SPAN = (0.4, 0.8)
#: ``strafe_combo`` 的换向区间（秒）：幅度小、频率高（KB §3 原话）。
TEACHER_V4_STRAFE_COMBO_SPAN = (0.2, 0.4)
#: ``strafe_poke`` 期间每 N 帧插一帧纯前进。
#:
#: v2 的 ``move`` 是**单方向**的（forward / left / right 互斥），表达不了人类那种
#: 「按住 W 同时 A/D 横移」。纯横移的后果是实测出来的：4–6 格带里连站 58/103 帧
#: （2026-11-22 首轮冒烟），双方都不进带，等于把接战无限推迟。
#: 所以这里按「2 帧横移 + 1 帧前进」的节拍逼近：既保住了晃准星的意图，
#: 又保证每 0.6s 有一步是在缩短距离。``0`` 表示关闭（纯横移，用于对照实验）。
TEACHER_V4_POKE_PUSH_EVERY = 3
#: 攻击冷却（秒）。1.8 受击无敌帧 500ms。
TEACHER_V4_ATTACK_COOLDOWN = 0.5
#: 判定「掉血 = 一次命中 / 一次受击」的最小血量差（点）。
TEACHER_V4_HIT_EPS = 0.4
#: 一帧多少秒（缺省值；``meta["frame_seconds"]`` 可覆盖，真机实测约 0.2）。
TEACHER_V4_FRAME_SECONDS = 0.2
#: 热键栏槽位：训练套件的剑在第 0 格，因此固定 ``slot=0`` 等价于「不换槽」。
TEACHER_V4_SLOT = 0

#: 各头在 10 头向量里的下标（按 ``ACTION_HEADS_V2`` 的声明顺序查，不写死数字）。
_V4_HEADS = list(ACTION_HEADS_V2)
_V4_IDX = {name: index for index, name in enumerate(_V4_HEADS)}
#: ``look`` 的「不转」取值下标（0.0 度那一档）：老师不做瞄准，朝向交给运行时/策略。
_V4_LOOK_NEUTRAL = (LOOK_YAW_V2.index(0.0), LOOK_PITCH_V2.index(0.0))

#: 编译期自检：v4 输出的必须是 10 头（G-707 后的形状），否则 decode_intent 会拒收。
assert tuple(MULTI_DISCRETE_V2) == (5, 7, 5, 2, 3, 2, 2, 2, 9, 3), \
    f"v4 老师按 10 头写的，当前 nvec={tuple(MULTI_DISCRETE_V2)}"


def _v4_action(*, move: str = "idle", attack: bool = False, jump: bool = False,
               sprint: bool = False, reset: str = "none",
               slot: int = TEACHER_V4_SLOT) -> list[int]:
    """构造一个 v2 的 10 头动作（``MultiDiscrete`` 的取值，不是 one-hot）。"""
    action = [0] * len(_V4_HEADS)
    action[_V4_IDX["move"]] = MOVE_VALUES_V2.index(move)
    action[_V4_IDX["look_yaw"]] = _V4_LOOK_NEUTRAL[0]
    action[_V4_IDX["look_pitch"]] = _V4_LOOK_NEUTRAL[1]
    action[_V4_IDX["attack"]] = 1 if attack else 0
    action[_V4_IDX["use"]] = 0                     # 老师从不主动按右键（举剑走 sprint_reset）
    action[_V4_IDX["jump"]] = 1 if jump else 0
    action[_V4_IDX["sneak"]] = 0
    action[_V4_IDX["sprint"]] = 1 if sprint else 0
    action[_V4_IDX["slot"]] = max(0, min(8, int(slot)))
    action[_V4_IDX["sprint_reset"]] = SPRINT_RESET_VALUES_V2.index(reset)
    return action


def v4_intent_of(action: Sequence[int]) -> dict:
    """把 v4 的 10 头动作翻成 env 约定形状的 intent（``decode_intent`` 的输出）。

    调用方（真机探针、日志、单测）需要「这一帧到底想干什么」时用它，
    不必自己重复一遍 ``ACTION_HEADS_V2`` 的顺序。
    """
    return decode_intent(list(action))


class TeacherV4State:
    """老师 v4 的跨帧记忆（每局一份）。

    字段：

    - ``frame``：调用计数；
    - ``frame_seconds``：一帧秒数（``meta`` 可钉死，默认 0.2）；
    - ``last_target_health`` / ``last_self_health``：血量的上一帧值（判定命中/受击）；
    - ``last_hit_frame`` / ``last_hurt_frame``：最近一次「打中别人」/「被别人打中」的帧号；
    - ``last_arm_frame``：最近一次「起跳预备」的帧号（jump_crit 的起跳节流）；
    - ``strafe_direction`` / ``strafe_until_frame`` / ``strafe_span_index``：横移换向节拍；
    - ``reasons``：本局各条规则的命中次数；
    - ``hits_landed`` / ``hits_taken``：真实命中/受击次数（按血量掉幅统计）。
    """

    __slots__ = (
        "frame", "frame_seconds", "last_target_health", "last_self_health",
        "last_hit_frame", "last_hurt_frame", "last_arm_frame", "strafe_direction",
        "strafe_until_frame", "strafe_span_index", "reasons",
        "hits_landed", "hits_taken", "last_action",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "TeacherV4State":
        self.frame = 0
        self.frame_seconds = TEACHER_V4_FRAME_SECONDS
        self.last_target_health: float | None = None
        self.last_self_health: float | None = None
        self.last_hit_frame: int | None = None
        self.last_hurt_frame: int | None = None
        self.last_arm_frame: int | None = None
        self.strafe_direction = 0            # 0 = 左，1 = 右
        self.strafe_until_frame = 0
        self.strafe_span_index = 0
        self.reasons: dict[str, int] = {}
        self.hits_landed = 0
        self.hits_taken = 0
        self.last_action: list[int] | None = None
        return self


def _v4_options(meta: Any, state: TeacherV4State | None) -> dict:
    """把 ``meta`` 规范成 v4 的决策上下文（全部可缺省）。"""
    data: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    chosen = state if isinstance(state, TeacherV4State) else data.get("state")
    if not isinstance(chosen, TeacherV4State):
        chosen = _DEFAULT_V4_STATE
    frame_seconds = data.get("frame_seconds")
    return {
        "state": chosen,
        "frame_seconds": (TEACHER_V4_FRAME_SECONDS if frame_seconds is None
                          else max(1e-6, float(frame_seconds))),
        #: 有没有剑（观测里没有手持物字段，只能由调用方给；缺省 True）。
        "has_sword": bool(data.get("has_sword", True)),
        #: 热键栏目标槽（缺省 0 = 训练套件的剑位）。
        "slot": int(data.get("slot", TEACHER_V4_SLOT)),
    }


def _v4_track(state: TeacherV4State, obs: np.ndarray) -> None:
    """更新「上一刀打中了 / 刚被打了」这两个跨帧判定（按血量掉幅，不依赖信号缺失的字段）。"""
    target_health = _get(obs, IDX_TARGET_HEALTH) * HEALTH_SCALE
    self_health = _get(obs, IDX_SELF_HEALTH) * HEALTH_SCALE

    prev_target = state.last_target_health
    if prev_target is not None and target_health > 0.0 and prev_target - target_health >= TEACHER_V4_HIT_EPS:
        state.last_hit_frame = state.frame
        state.hits_landed += 1
    # 对手满血（重生/被补满）时不把血量抬升当成命中，但要把基线重置。
    state.last_target_health = target_health if target_health > 0.0 else None

    prev_self = state.last_self_health
    if prev_self is not None and self_health > 0.0 and prev_self - self_health >= TEACHER_V4_HIT_EPS:
        state.last_hurt_frame = state.frame
        state.hits_taken += 1
    state.last_self_health = self_health if self_health > 0.0 else None


def _v4_strafe(state: TeacherV4State, frame_seconds: float, span: tuple[float, float]) -> str:
    """横移换向节拍：``span`` 秒换一次向（不要每帧抖方向）。"""
    if state.frame >= state.strafe_until_frame:
        state.strafe_direction = 1 - state.strafe_direction
        low, high = span
        chosen = low + (high - low) * (state.strafe_span_index % 2)
        state.strafe_span_index += 1
        state.strafe_until_frame = state.frame + max(1, int(round(chosen / frame_seconds)))
    return "left" if state.strafe_direction == 0 else "right"


def _v4_finish(state: TeacherV4State, action: list[int], *, reason: str) -> tuple[list[int], dict]:
    """收尾：记账并返回 ``(10 头动作, 诊断)``。"""
    state.last_action = list(action)
    state.frame += 1
    state.reasons[reason] = state.reasons.get(reason, 0) + 1
    return action, {"reason": reason, "intent": decode_intent(list(action))}


def _decide_v4(obs: np.ndarray, options: dict) -> tuple[list[int], dict]:
    """v4 的决策核心：严格按 ``pvp-1.8-mechanics.md`` §3 的招式表。

    判定顺序（顺序即优先级）::

        0. 无目标                     → idle
        1. block_hit（有剑 + ≤3.0 格 + 冷却好）
                                      → move=forward, reset=block_hit, attack=1
        2. jump_crit（在 1.4–2.4 格、冷却好时先 jump_arm 起跳，
            vy<0 那一帧）→ jump=1 + attack=1（同一帧，G-701 组合）
        3. wtap_keep_combo（无剑 + 上一帧命中）→ move=forward, reset=release_w, attack=1
        4. wtap_reduce_kb（无剑 + 上一帧受击）→ move=forward, reset=release_w, attack=0
        5. 射程内且冷却好             → attack
        6. 距离阶梯（§3 的三条带，顺序即优先级）
           <2.6 格                    → S（spacing_back）
           >6.0 格                    → W（spacing_forward，追人）
           4–6 格                     → strafe_poke（慢换向）
           3.3–4.0 格                 → W（spacing_forward，压进带）
           2.6–3.3 格                 → strafe_combo（快换向）
        7. 兜底                       → forward

    有剑时**绝不**走第 3/4 条，无剑时**绝不**走第 1 条——两条 reset 路线互斥，
    这是视频里明确讲的（同时用失误率高），也是本模块最容易写错的地方。
    """
    state: TeacherV4State = options["state"]
    has_sword = options["has_sword"]
    slot = options["slot"]
    frame_seconds = options["frame_seconds"]
    state.frame_seconds = frame_seconds

    has_target = _get(obs, IDX_HAS_TARGET) >= 0.5
    if not has_target:
        return _v4_finish(state, _v4_action(slot=slot), reason="no_target")

    distance = _get(obs, IDX_TARGET_DISTANCE) * DISTANCE_SCALE
    in_reach = _get(obs, IDX_IN_REACH) >= 0.5
    vy = _get(obs, IDX_SELF_VY) * 5.0                     # 格/秒，下落为负
    cooldown = _get(obs, IDX_ATTACK_COOLDOWN) * 0.5       # 还差多少秒到 0.5s
    ready = cooldown <= frame_seconds * 0.25 or \
        _get(obs, IDX_SELF_SINCE_ATTACK) * 5.0 >= TEACHER_V4_ATTACK_COOLDOWN

    _v4_track(state, obs)
    hit_landed = state.last_hit_frame is not None and 0 <= state.frame - state.last_hit_frame <= 1
    hurt_now = state.last_hurt_frame is not None and 0 <= state.frame - state.last_hurt_frame <= 1

    # --- 6. 控距（任何时候都先修正站位，避免在够不到的带里空耗）-----------------
    spacing_forward = distance > TEACHER_V4_SPACING_MAX
    spacing_back = distance < TEACHER_V4_SPACING_MIN

    # --- 1. 有剑：防砍（reset=block_hit + 同一帧出手）---------------------------
    if has_sword and distance <= TEACHER_V4_BLOCK_DISTANCE:
        return _v4_finish(
            state,
            _v4_action(move="forward", attack=bool(ready), reset="block_hit", slot=slot),
            reason="block_hit")

    # --- 2. 跳劈：先「起跳预备」，再在**下落中**出手 ---------------------------
    # KB §3：距离 ≤2.4 且冷却已好 → 起跳 → 下落中挥刀。本环境把 jump 与 attack 放进
    # 同一帧（G-701 已验证会被服务端按跳劈结算），所以「vy<0」那一帧就是暴击窗口。
    #
    # 2026-11-22 真机 12 局暴露的问题：只按 `vy<0` 触发时这条规则**永远不可能生效**
    # ——vy 要变负得先起跳，而起跳又只在 vy<0 时发（自锁）。首轮 12 局的 reason 里
    # jump_crit 出现 0 次就是这个原因。现在拆成两拍：
    #   ① arm：着地 + 冷却已好 + 在距离带内 → 只起跳（reason=jump_arm，带 1s 节流）；
    #   ② crit：`vy<0` 且冷却仍好 → jump+attack 同帧（reason=jump_crit）。
    # 「只在 vy<0 时出手」这一条没有变：出手帧仍然要求正在下落。
    if (vy < 0.0 and ready
            and TEACHER_V4_CRIT_MIN_DISTANCE <= distance <= TEACHER_V4_CRIT_MAX_DISTANCE):
        return _v4_finish(
            state,
            _v4_action(move="forward", attack=True, jump=True, slot=slot),
            reason="jump_crit")

    arm_ready = (
        state.last_arm_frame is None
        or (state.frame - state.last_arm_frame) * frame_seconds >= TEACHER_V4_JUMP_ARM_COOLDOWN)
    if (ready and arm_ready and _get(obs, IDX_SELF_ON_GROUND) >= 0.5
            and TEACHER_V4_CRIT_MIN_DISTANCE <= distance <= TEACHER_V4_CRIT_MAX_DISTANCE):
        state.last_arm_frame = state.frame
        return _v4_finish(
            state,
            _v4_action(move="forward", jump=True, slot=slot),
            reason="jump_arm")

    # --- 3/4. 无剑：wtap 的两种用法（永不按右键）-------------------------------
    if not has_sword and (hit_landed or hurt_now):
        if hit_landed:
            # wtap_keep_combo：刚命中 → 松 W 再按回来 + 立刻挥刀，让对手持续吃大 KB。
            return _v4_finish(
                state,
                _v4_action(move="forward", attack=True, reset="release_w", slot=slot),
                reason="wtap_keep_combo")
        # wtap_reduce_kb：刚受击 → 松 W 再按回来，缩小自己的 KB、快速回到攻击距离。
        return _v4_finish(
            state,
            _v4_action(move="forward", attack=False, reset="release_w", slot=slot),
            reason="wtap_reduce_kb")

    # --- 5. 射程内且冷却好：砍 -------------------------------------------------
    if in_reach and ready and not spacing_back:
        return _v4_finish(state, _v4_action(move="forward", attack=True, slot=slot),
                          reason="attack")

    # --- 6. 距离阶梯：S → 追（>6）→ strafe_poke（4–6）→ W（3.3–4）→ combo（2.6–3.3）---
    # 顺序本身就是招式表（§3）：`strafe_poke` 的适用带是「接战前 4–6 格」，
    # 它**必须**排在「太远就压上」之前，否则 4–6 格永远被 approach 吃掉，
    # 那条规则在这份代码里就是死的。
    if spacing_back:
        return _v4_finish(state, _v4_action(move="back", slot=slot), reason="spacing_back")
    if distance > TEACHER_V4_STRAFE_POKE_MAX:
        return _v4_finish(
            state,
            _v4_action(move="forward", sprint=distance > 2.5, slot=slot),
            reason="spacing_forward")
    if TEACHER_V4_STRAFE_POKE_MIN <= distance <= TEACHER_V4_STRAFE_POKE_MAX:
        # 「边晃边压」：每 N 帧插一帧纯前进（单方向动作空间的折中，见常量注释）。
        if (TEACHER_V4_POKE_PUSH_EVERY > 0
                and state.frame % TEACHER_V4_POKE_PUSH_EVERY == TEACHER_V4_POKE_PUSH_EVERY - 1):
            return _v4_finish(
                state,
                _v4_action(move="forward", sprint=distance > 2.5, slot=slot),
                reason="strafe_poke")
        direction = _v4_strafe(state, frame_seconds, TEACHER_V4_STRAFE_POKE_SPAN)
        return _v4_finish(state, _v4_action(move=direction, slot=slot), reason="strafe_poke")
    if spacing_forward:
        return _v4_finish(
            state,
            _v4_action(move="forward", sprint=distance > 2.5, slot=slot),
            reason="spacing_forward")
    if TEACHER_V4_SPACING_MIN <= distance <= TEACHER_V4_SPACING_MAX:
        direction = _v4_strafe(state, frame_seconds, TEACHER_V4_STRAFE_COMBO_SPAN)
        return _v4_finish(state, _v4_action(move=direction, slot=slot), reason="strafe_combo")

    # --- 7. 兜底：压上（进带 / 进射程）-----------------------------------------
    return _v4_finish(state, _v4_action(move="forward", sprint=distance > 2.5, slot=slot),
                      reason="approach")


#: ``heuristic_action_v4(obs)``（不传 state）时使用的默认状态。
#: 多局/多线程场景请用 :class:`TeacherV4` 实例，或显式传 ``state=``。
_DEFAULT_V4_STATE = TeacherV4State()


def reset_teacher_v4() -> None:
    """清空模块级默认状态（测试 / 新 episode 之间调用）。"""
    _DEFAULT_V4_STATE.reset()


def heuristic_action_v4(
    obs: np.ndarray,
    meta: Mapping[str, Any] | None = None,
    state: TeacherV4State | None = None,
) -> list[int]:
    """老师 v4：按 1.8 招式表输出 **v2 的 10 头动作**（``list[int]``，可直接喂 env）。

    与 v1–v3 的差别：返回值不是动作编号，而是 ``MultiDiscrete`` 的取值向量
    （``[move, look_yaw, look_pitch, attack, use, jump, sneak, sprint, slot, sprint_reset]``）。
    想拿「这一帧的意图」用 :func:`v4_intent_of`（= ``decode_intent`` 的输出，
    与 env 下发给运行时的 ``{type:"intent", intent:{...}}`` 里的 intent 完全同形状）。

    招式表（``docs/knowledge/pvp-1.8-mechanics.md`` §3）与代码的对应关系：

    ================== ============================================ ==============
    招式               条件                                          动作
    ================== ============================================ ==============
    ``block_hit``      有剑 + ≤3.0 格                                W + 举剑 + 砍
    ``jump_crit``      ``vy<0`` + 冷却好 + 1.4–2.4 格                 jump + attack
    ``wtap_keep_combo`` 无剑 + 上一帧命中                            W + release_w + 砍
    ``wtap_reduce_kb`` 无剑 + 上一帧受击                            W + release_w
    ``spacing_hold``   >3.3 格 → W；<2.6 格 → S                    approach / retreat
    ``strafe_poke``    4–6 格                                        A/D，0.4–0.8s 换向
    ``strafe_combo``   2.6–3.3 格                                    A/D，0.2–0.4s 换向
    ``no_sword_fight`` 手上没剑                                    只用 wtap，**不按右键**
    ================== ============================================ ==============

    ``meta`` 支持的键：``has_sword``（缺省 ``True``；观测里没有手持物字段）、
    ``frame_seconds``（缺省 0.2）、``slot``（缺省 0）、``state``。
    传入 dict 时会写回 ``meta["reason"]`` 与 ``meta["intent"]``，便于复盘。
    """
    options = _v4_options(meta, state)
    action, info = _decide_v4(obs, options)
    if isinstance(meta, dict):
        meta["reason"] = info["reason"]
        meta["intent"] = info["intent"]
    return action


class TeacherV4:
    """可复用的 v4 老师实例（自己持有一份 :class:`TeacherV4State`）。

    ``teacher = TeacherV4(); teacher(obs)`` 返回 10 头动作；每个 episode 开头调
    :meth:`reset`（环境 ``reset()`` 会通过 ``reset_episode`` 回调）。
    ``is_v2 = True`` / ``action_space = "v2"`` 是给 ``GladiatorEnv`` 看的：
    它据此知道这个对手给的是意图向量而不是动作编号（见 env 的 ``_opponent_v2``）。
    """

    version = "v4"
    is_v2 = True
    action_space = "v2"

    def __init__(self, **defaults: Any) -> None:
        self.state = TeacherV4State()
        self.defaults = dict(defaults)
        #: 已结束 episode 的原因统计（每次 :meth:`reset` 归档上一局）。
        self.episode_reasons: list[dict[str, int]] = []

    def reset(self) -> "TeacherV4":
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
        self.state.reset()
        return self

    def reset_episode(self) -> None:
        """环境 ``reset()`` 时的回调：归档上一局并清零状态。"""
        self.reset()

    @property
    def reasons(self) -> dict[str, int]:
        return dict(self.state.reasons)

    def flush(self) -> list[dict[str, int]]:
        if self.state.reasons:
            self.episode_reasons.append(dict(self.state.reasons))
            self.state.reasons = {}
        return list(self.episode_reasons)

    def act(self, vector, rng=None, greedy: bool = True) -> list[int]:
        """与 ``Policy.act`` 同签名（env 的 ``opponent_policy`` 约定），但返回 10 头动作。"""
        merged: dict[str, Any] = dict(self.defaults)
        return heuristic_action_v4(vector, merged, self.state)

    def __call__(self, obs: np.ndarray, meta: Mapping[str, Any] | None = None) -> list[int]:
        merged: dict[str, Any] = dict(self.defaults)
        if isinstance(meta, Mapping):
            merged.update(meta)
        action = heuristic_action_v4(obs, merged, self.state)
        if isinstance(meta, dict):
            meta["reason"] = merged.get("reason", "")
            meta["intent"] = merged.get("intent")
        return action


def make_teacher(version: str = TEACHER_VERSION) -> Callable[..., int]:
    """按版本号取老师。

    ``"v1"`` → :func:`heuristic_action`（无状态函数）；
    ``"v2"`` → :class:`TeacherV2`；``"v3"`` → :class:`TeacherV3`；
    ``"v4"`` → :class:`TeacherV4`（**返回 10 头 v2 动作**，不是动作编号）。
    """
    key = str(version or TEACHER_VERSION).strip().lower()
    if key == "v1":
        return heuristic_action
    if key == "v2":
        return TeacherV2()
    if key == "v3":
        return TeacherV3()
    if key == "v4":
        return TeacherV4()
    raise ValueError(f"未知老师版本: {version!r}（允许 {', '.join(SUPPORTED_TEACHER_VERSIONS)}）")


def collect_demos(
    env,
    episodes: int = 4,
    max_steps: int = 120,
    teacher: Callable[..., Any] | None = None,
    gamma: float = 0.99,
    teacher_version: str = TEACHER_VERSION,
    v2_weight: float = 1.0,
) -> list[tuple[np.ndarray, Any, float]]:
    """用老师跑 ``episodes`` 局，返回 ``[(obs, 动作标签, 第三个字段)]``。

    两种动作空间，第三个字段的含义不同（**调用方必须按同一个口径解释**）：

    - **v1**（``env.action_space`` 是 ``Discrete``）：标签是动作编号 ``int``，
      第三字段是**该局后续折扣回报**（末步之后为 0，不做 bootstrap）；
    - **v2**（``env.action_space`` 是 ``MultiDiscrete``）：标签是 **10 个头的下标元组**
      （顺序 = :data:`V2_HEADS`），第三字段是**样本权重**（默认全 ``v2_weight``）。
      v2 没有「动作编号」，硬塞一个扁平下标会让 `decode_intent` 把 10 个头压成 1 个，
      所以这里显式换成「头下标 + 权重」，折扣回报在 v2 的逐头交叉熵里也没有用途。

    ``teacher`` 显式给出时直接用它；否则按 ``teacher_version`` 取
    （``"v1"`` 默认 / ``"v2"`` 见 :func:`heuristic_action_v2` / ``"v4"`` 见 :class:`TeacherV4`）。
    有状态的老师（v2/v3/v4）每局 ``env.reset()`` 之后会调 ``teacher.reset()``。

    v2 下老师返回的必须是 10 头动作（``TeacherV4`` 即如此）；返回 ``{"intent": {...}}``
    的老师也支持——经 :func:`intent_to_tuple` 翻成下标。若老师自己把 intent 写回
    ``meta["intent"]``（v4 的约定），这里还会做一次**交叉校验**：两种写法不一致
    说明老师内部与日志口径已经分叉，直接报错而不是把错标签喂进 BC。

    只支持单智能体环境（``single_agent=True``），否则动作契约不同，直接报错而不是瞎猜。
    """
    teacher = teacher if teacher is not None else make_teacher(teacher_version)
    v2 = _env_is_v2(env)
    samples: list[tuple[np.ndarray, Any, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, _info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_demos 只支持 single_agent=True 的环境（obs 应为向量）")
        _reset_teacher(teacher)
        steps: list[tuple[np.ndarray, Any, float]] = []
        done = False
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            action = _teacher_label(teacher, vector, v2=v2)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            steps.append((vector, action, float(reward)))
            obs = next_obs
            if terminated or truncated:
                done = True
                break
        if v2:
            # v2：权重守恒（默认 1.0），不做折扣回报——见 docstring。
            weight = float(v2_weight)
            if weight < 0 or not math.isfinite(weight):
                raise ValueError(f"v2_weight 必须是非负有限数，收到 {v2_weight}")
            for vector, action, _reward in steps:
                samples.append((vector, action, weight))
            continue
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


def _env_is_v2(env) -> bool:
    """环境是不是 v2 动作空间（``MultiDiscrete``）。

    优先读 ``env.action_space_name``（GladiatorEnv 的显式字段），退化到
    「``action_space`` 有 ``nvec``」——这样 FakeEnv/测试替身也能被正确识别。
    """
    name = getattr(env, "action_space_name", None)
    if name is not None:
        return str(name).strip().lower() == "v2"
    return hasattr(getattr(env, "action_space", None), "nvec")


def _v2_label(raw: Any) -> tuple[int, ...]:
    """把老师输出规范成 10 头下标元组（校验头数与取值范围）。"""
    if isinstance(raw, Mapping):
        return intent_to_tuple(raw)
    values = [int(value) for value in np.asarray(raw, dtype=np.int64).reshape(-1)]
    if len(values) != len(V2_HEADS):
        raise ValueError(
            f"v2 老师应返回 {len(V2_HEADS)} 个头，收到 {len(values)} 个：{values!r}")
    for index, (value, size) in enumerate(zip(values, MULTI_DISCRETE_V2)):
        if not 0 <= value < int(size):
            raise ValueError(
                f"v2 老师第 {index} 个头（{V2_HEADS[index]}）越界: {value}（0..{size - 1}）")
    return tuple(values)


def _teacher_label(teacher: Callable[..., Any], vector: np.ndarray, *, v2: bool) -> Any:
    """取一帧的老师标签（v1 → int；v2 → 头下标元组），并做 intent 交叉校验。"""
    if not v2:
        try:
            return int(teacher(vector))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "v1 环境的标签必须是动作编号；这个老师给的是 v2 十头动作，"
                "请用 --action-space v2（v2 下的 BC 只支持 teacher v4）") from exc
    meta: dict[str, Any] = {}
    # v4 约定 `teacher(obs, meta)` 会把本帧 intent 写回 meta；不带 meta 的老师（纯函数）
    # 不传第二参，避免用 TypeError 去猜签名。
    if bool(getattr(teacher, "is_v2", False)) or bool(getattr(teacher, "accepts_meta", False)):
        raw = teacher(vector, meta)
    else:
        raw = teacher(vector)
    action = _v2_label(raw)
    intent = meta.get("intent")
    if isinstance(intent, Mapping):
        from_intent = intent_to_tuple(intent)
        if from_intent != action:
            raise ValueError(
                "老师返回的动作与它写回的 intent 不一致："
                f"动作={action} intent→{from_intent}（老师的两个出口已经分叉）")
    return action


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
    sample_weight: float = 1.0,
) -> list[tuple[np.ndarray, Any, float]]:
    """DAgger 的一轮采样：用**当前策略**跑，每步让老师给动作标签（P2.3）。

    与 :func:`collect_demos` 的区别是「谁在跑」：BC 用老师跑（数据分布 = 老师的状态分布），
    策略自己跑久了会走到老师从没去过的状态（covariate shift），DAgger 每轮都用
    当前策略的状态分布重新打标签，于是「策略越强、标签越贴它的实际处境」。

    返回结构与 :func:`collect_demos` 一致，可以直接喂给 :func:`behavior_clone`：

    - **v1**：``(obs, 动作编号, 0.0)``（第三字段不被 BC 使用，保持旧值以免改变日志口径）；
    - **v2**：``(obs, 10 头下标元组, sample_weight)``（第三字段是**权重**，默认 1.0）。

    v2 下「策略自己走」用 :meth:`Policy.sample` 并按环境的 ``actionMask`` 掩码，
    与训练主循环同一口径（否则策略会走出被屏蔽的 slot/use，采到老师从没见过的分布）。
    标签仍然来自老师，所以「同帧取 v4 标签」这件事与 v1 完全一致。
    """
    teacher = teacher if teacher is not None else make_teacher(teacher_version)
    rng = rng if rng is not None else np.random.default_rng(0)
    v2 = _env_is_v2(env)
    samples: list[tuple[np.ndarray, Any, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_dagger_samples 只支持 single_agent=True 的环境（obs 应为向量）")
        _reset_teacher(teacher)
        mask = _mask_of(policy, info) if v2 else None
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            label = _teacher_label(teacher, vector, v2=v2)  # 老师对**这个状态**的标签
            if v2:
                # 策略自己走，才能采到自己的状态分布；v2 的动作是 (H,) 数组。
                action = policy.sample(vector, rng, mask=mask)
                samples.append((vector, label, float(sample_weight)))
            else:
                action = int(policy.act(vector, rng))
                samples.append((vector, label, 0.0))
            obs, _reward, terminated, truncated, info = env.step(action)
            if v2:
                mask = _mask_of(policy, info)
            if terminated or truncated:
                break
    return samples


def _mask_of(policy, info: Any):
    """从 ``info["actionMask"]`` 取 v2 掩码并规范成策略要的``list[list[bool]]``。

    任何异常都退化成 ``None``（不掩码）：DAgger 的标签由老师给，掩码只影响
    「策略走到哪些状态」，不值得为此中断一整轮采样。
    """
    if not isinstance(info, Mapping):
        return None
    raw = info.get("actionMask")
    if raw is None:
        return None
    normalize = getattr(policy, "normalize_mask", None)
    try:
        return normalize(raw) if callable(normalize) else raw
    except Exception:  # noqa: BLE001 - 见 docstring
        return None
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
    samples: Sequence[tuple[np.ndarray, Any, float]],
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

    两种策略结构，**样本第三字段的含义随之不同**：

    - **v1**（单头 softmax，11 宏动作）：标签是动作编号，第三字段是折扣回报（不使用）。
      返回 ``{"loss", "accuracy", "samples", "epochs", "temperature",
      "labelSmoothing", "teacherEntropy", "totalWeight"}``。
    - **v2**（10 个离散头的 factorized 策略）：标签是 10 个头下标组成的元组，
      第三字段是**权重**，返回在上面基础上多一个 ``perHeadAccuracy``（10 个头各自
      的加权命中率），且 ``accuracy`` = 各头准确率的平均。实现见
      :func:`_behavior_clone_v2`，**v1 的代码路径逐位未改**。

    ``entropy_coef>0`` 时在损失里减去 ``β·H(π)``（鼓励保留探索）；v2 的熵按各头平均。

    温度/平滑的作用（P2.3 后续）：硬标签 + 30 轮 Adam 会把 softmax 压到熵 0.01~0.03，
    之后 PPO 的 ratio 与熵正则都推不动。``temperature>1`` 让目标变软（交叉熵
    ``-Σ q·log p``），``label_smoothing>0`` 混入均匀分布，二者都保留策略的初始熵。
    ``T=1`` 且 ``eps=0`` 时软目标就是 one-hot，梯度 ``p - q`` 与旧实现一致。

    口径说明：这里的 ``p`` 是 ``softmax(未除温度的 logits)``（与旧实现一致），
    ``policy.temperature`` 只影响采样/PPO 侧；BC 的 ``temperature`` 是**独立参数**，
    两者不要混用（要整体调平请在 BC 用 temperature，PPO 侧用 ``--logit-temp``）。
    ``weights`` 给每条样本一个非负权重（默认全 1）：用于「赢的局权重大、输的局权重小」
    与「交手帧权重大、跑位帧权重小」这类择优筛选。权重按批内和归一化，
    因此全 1 时与原实现逐位一致。v2 下样本第三字段的权重与 ``weights`` **相乘**，
    所以两者都缺省（全 1）时与「只给一个全 1 权重」完全等价。
    """
    if getattr(policy, "is_v2", False):
        # v2 走独立实现：v1 的 W2/b2 是单张量，v2 是 W2_<head>/b2_<head>，
        # 混在一个函数里会把 v1 的逐位行为改坏（15 条权重测试就是判据）。
        return _behavior_clone_v2(
            policy, samples, epochs=epochs, lr=lr, batch=batch,
            entropy_coef=entropy_coef, seed=seed, temperature=temperature,
            label_smoothing=label_smoothing, weights=weights)
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


# --- G-705(c) AWR：用价值头给样本加权 ---------------------------------------

#: ``advantage_weights`` 里指数的上限：``exp(50) ≈ 5.2e21``，再大就没有数值意义，
#: 而且一条样本会在批内彻底吃掉梯度。超限时钳在边界（等比例缩放，不改变符号）。
AWR_EXP_LIMIT = 50.0


def advantage_weights(
    policy,
    samples: Sequence[tuple[np.ndarray, Any, float]],
    *,
    returns: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
    positive_only: bool = True,
    clip: "Optional[float | tuple[float, float]]" = None,
) -> np.ndarray:
    """G-705(c)：按「这步比价值头的预期好多少」给样本加权（AWR 的权重口径）。

    仅用**价值头** ``V(s)``，不改任何权重、不参与反向传播——它只是「同一批人类样本里
    哪几帧更值得学」的筛选器。三种加权模式的关系（见 ``docs/match.md`` §8.1）：

    - (a) 结果过滤：赢的局整局加权；
    - (b) 接触窗口：贴身的帧加权；
    - **(c) AWR**：不再按「局/距离」粗筛，而是问 critic「这一帧的即时收益比预期高吗」。

    口径（刻意写得死板，避免各处自己发明奖励）：

    - ``R`` 只来自调用方：``returns`` 与样本等长（``None`` → 全 0，即「没有回报」）；
    - ``A = R - V(s)``；``w = exp(A / temperature)``；
    - ``positive_only=True``（默认）把 ``A < 0`` 的样本权重置 0——AWR 的思想是
      「只向比平均更好的动作学」，负优势样本不该被当成正样本；
    - 整体按**均值归一到 1**（这样换模式不会悄悄改变 BC 的有效学习率）；
      若归一化前权重全为 0（例如全部样本都是负优势），退化为**全 1 均匀权重**
      ——否则 BC 会拿到一个总权重 0 的批次，等于这一轮什么都没学；
    - ``clip``：单个数字视为上界 ``(0, clip)``，也可给 ``(lo, hi)``；作用在**归一化之前**
      的原始权重上（目的是压住单点主导），因此归一化之后的具体数值可能略越界。

    :param policy: ``Policy``（v1/v2 都行；v2 用策略头之前的共享隐层 + ``Wv/bv``）
    :param samples: ``[(obs, action, weight), ...]``；第三字段此处不使用
    :param returns: 每条样本的单步回报 ``R``；``None`` 表示没有（按 0 处理）
    :param temperature: ``exp(A/T)`` 的温度；越小越尖（只留最优势的样本）
    :param positive_only: 负优势样本是否置 0
    :param clip: 权重上下界（见上）
    :returns: ``(N,)`` 的 float64 权重，均值 ≈ 1

    示例（与 (a)(b) 组合，脚本里的用法）::

        base = round_w * contact_w          # 已有的 (a)×(b)
        w = advantage_weights(policy, samples, returns=returns, temperature=1.0)
        weights = base * w
        weights *= len(weights) / weights.sum()   # 复合后重新归一到均值 1
    """
    count = len(samples)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    raw_temperature = float(temperature)
    if not np.isfinite(raw_temperature) or raw_temperature <= 0:
        raise ValueError(f"temperature 必须为正的有限数，收到 {temperature}")

    if returns is None:
        rewards = np.zeros(count, dtype=np.float64)
    else:
        rewards = np.asarray(list(returns), dtype=np.float64).reshape(-1)
        if rewards.size != count:
            raise ValueError(f"returns 长度 {rewards.size} 与样本数 {count} 不一致")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("returns 里有 NaN/Inf：先把数据修好再加权")

    values = np.asarray([float(policy.value(x)) for x, _action, _w in samples],
                        dtype=np.float64)
    advantage = rewards - values

    exponent = np.clip(advantage / raw_temperature, -AWR_EXP_LIMIT, AWR_EXP_LIMIT)
    weights = np.exp(exponent)

    if positive_only:
        weights = np.where(advantage < 0.0, 0.0, weights)

    if clip is not None:
        if isinstance(clip, (int, float)):
            lower, upper = 0.0, float(clip)
        else:
            lower, upper = float(clip[0]), float(clip[1])
        if not (np.isfinite(lower) and np.isfinite(upper)) or upper < lower:
            raise ValueError(f"clip 非法: {clip}")
        weights = np.clip(weights, lower, upper)

    mean = float(weights.mean()) if weights.size else 0.0
    if not np.isfinite(mean) or mean <= 1e-12:
        # 全负 / 全被剪成 0：退化为均匀，绝不让整批样本失去权重。
        return np.ones(count, dtype=np.float64)
    return weights / mean


def teacher_agreement(policy, samples: Iterable[tuple[np.ndarray, Any, float]]):
    """老师动作与策略 argmax 的一致率（BC 前后对比用）。

    - **v1**：返回一个 ``float``（单头 argmax 命中率），与旧口径逐位一致；
    - **v2**：返回 ``{"meanHeadAgreement": float, "exactMatch": float,
      "perHead": [float] * H, "heads": [...], "samples": int}``——十个头各算各的
      命中率再取平均（``meanHeadAgreement``，与 BC 的 ``accuracy`` 同口径），
      外加「十个头全部命中」的全体完全一致率（``exactMatch``）。
      只报一个数会骗人：8 个头对、`attack` 头全错时平均仍有 0.8，但那一帧根本不出手。

    v2 不做加权（与 v1 一致）：这里量的是「策略与老师的贴合度」，加权会把
    「样本多的局面」放大，反而看不出头与头之间的差别。
    """
    if getattr(policy, "is_v2", False):
        return _teacher_agreement_v2(policy, samples)
    total = 0
    correct = 0
    for x, action, _ret in samples:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        _h, p = _forward(policy.params, x)
        correct += int(int(np.argmax(p)) == int(action))
        total += 1
    return round(correct / total, 4) if total else 0.0


def _head_probs(policy, head_logits: np.ndarray) -> np.ndarray:
    """单头概率（优先用策略自己的 ``_head_probs``，含它配置的温度）。"""
    fn = getattr(policy, "_head_probs", None)
    if callable(fn):
        return np.asarray(fn(head_logits), dtype=np.float64)
    temperature = max(1e-9, float(getattr(policy, "temperature", 1.0)))
    return _softmax(np.asarray(head_logits, dtype=np.float64) / temperature)


def _teacher_agreement_v2(policy, samples: Iterable[tuple[np.ndarray, Any, float]]) -> dict:
    """:func:`teacher_agreement` 的 v2 实现（逐头一致率 + 全体完全一致率）。"""
    heads = list(policy.head_names())
    sizes = list(policy.head_sizes())
    per_head_correct = [0] * len(heads)
    exact = 0
    total = 0
    for x, action, _weight in samples:
        vector = np.asarray(x, dtype=np.float64).reshape(-1)
        picked = np.asarray(action, dtype=np.int64).reshape(-1)
        if picked.size != len(heads):
            raise ValueError(
                f"v2 样本的动作应有 {len(heads)} 个头，收到 {picked.size} 个")
        logits = policy.logits_all(vector)
        total += 1
        all_hit = True
        for index, head_logits in enumerate(logits):
            hit = int(np.argmax(_head_probs(policy, head_logits))) == int(picked[index])
            per_head_correct[index] += int(hit)
            all_hit = all_hit and hit
        exact += int(all_hit)
    if total == 0:
        return {"meanHeadAgreement": 0.0, "exactMatch": 0.0,
                "perHead": [0.0] * len(heads), "heads": heads, "samples": 0}
    per_head = [round(count / total, 4) for count in per_head_correct]
    return {
        "meanHeadAgreement": round(sum(per_head) / len(per_head), 4),
        "exactMatch": round(exact / total, 4),
        "perHead": per_head,
        "heads": heads,
        "samples": total,
    }


def _behavior_clone_v2(
    policy,
    samples: Sequence[tuple[np.ndarray, Sequence[int], float]],
    *,
    epochs: int,
    lr: float,
    batch: int,
    entropy_coef: float,
    seed: int,
    temperature: float,
    label_smoothing: float,
    weights: Optional[Sequence[float]],
) -> dict:
    """v2（多离散头）的 BC：**逐头交叉熵按头平均**，样本第三个字段当权重。

    损失口径（H = 头数，w_i = 样本权重）::

        L_i = (1/H) · Σ_h CE(q_h, p_h)          # 逐头平均：每个头权重相同
        L   = Σ_i w_i · L_i / Σ_i w_i

    为什么逐头平均、而不是「求和后再按样本归一化」：各头取值个数差 3 倍以上
    （``slot`` 9 档 vs ``attack`` 2 档），求和会让 slot 头的梯度压过 attack；
    平均保证每个头同等重要，也让 loss 的量级与 v1（单头 CE）可比。
    反传时 ``dlogits_h = (p_h - q_h) · w / H``，W1/b1 由各头之和更新。

    熵正则同样按头平均：``L_i -= β · (1/H) Σ_h H(p_h)``，梯度按每个头自己那一项减。

    返回与 :func:`behavior_clone` 相同，另加 ``perHeadAccuracy``（10 个头的加权命中率）；
    ``accuracy`` = 各头准确率的平均，与 :func:`teacher_agreement` 的
    ``meanHeadAgreement`` 同口径。``totalWeight`` 是**乘过**样本权重与 ``weights``
    之后的总权重。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing 必须在 [0, 1) 内，收到 {label_smoothing}")
    if weights is not None and len(weights) != len(samples):
        raise ValueError(f"weights 长度 {len(weights)} 与样本数 {len(samples)} 不一致")

    heads = list(policy.head_names())
    sizes = list(policy.head_sizes())
    n_heads = len(heads)
    data: list[tuple[np.ndarray, tuple[int, ...], float]] = []
    for index, (x, action, sample_weight) in enumerate(samples):
        picked = np.asarray(action, dtype=np.int64).reshape(-1)
        if picked.size != n_heads:
            raise ValueError(
                f"第 {index} 条 v2 样本的动作应有 {n_heads} 个头（{heads}），"
                f"收到 {picked.size} 个")
        for head_index, value in enumerate(picked):
            if not 0 <= int(value) < sizes[head_index]:
                raise ValueError(
                    f"第 {index} 条样本的 {heads[head_index]} 头越界: {int(value)}"
                    f"（0..{sizes[head_index] - 1}）")
        base = 1.0 if sample_weight is None else float(sample_weight)
        if not np.isfinite(base) or base < 0:
            raise ValueError(f"第 {index} 条样本权重非法: {sample_weight}")
        extra = 1.0 if weights is None else float(weights[index])
        if not np.isfinite(extra) or extra < 0:
            raise ValueError(f"第 {index} 条样本 weights 非法: {weights[index]}")
        data.append((np.asarray(x, dtype=np.float64).reshape(-1),
                     tuple(int(value) for value in picked), base * extra))

    result = {"loss": [], "accuracy": 0.0, "samples": len(data), "epochs": 0,
              "temperature": float(temperature),
              "labelSmoothing": float(label_smoothing),
              "teacherEntropy": 0.0,
              "totalWeight": float(sum(w for _x, _a, w in data)),
              "perHeadAccuracy": [0.0] * n_heads,
              "heads": heads}
    if not data:
        return result

    rng = np.random.default_rng(seed)
    names = (["W1", "b1"] + [f"W2_{head}" for head in heads]
             + [f"b2_{head}" for head in heads])
    m = {key: np.zeros_like(policy.params[key]) for key in names}
    v = {key: np.zeros_like(policy.params[key]) for key in names}
    step = 0
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    inv_heads = 1.0 / float(n_heads)
    # 软目标只与「标签元组」有关：每个头独立算一份，按标签元组缓存。
    targets: dict[tuple[int, ...], List[np.ndarray]] = {}

    def _q_of(label: tuple[int, ...]) -> List[np.ndarray]:
        cached = targets.get(label)
        if cached is None:
            cached = [teacher_target(value, size, temperature, label_smoothing)
                      for value, size in zip(label, sizes)]
            targets[label] = cached
        return cached

    total_weight = sum(w for _x, _a, w in data)

    for _epoch in range(int(max(0, epochs))):
        order = rng.permutation(len(data))
        epoch_loss = 0.0
        for start in range(0, len(data), max(1, int(batch))):
            chunk = [data[i] for i in order[start:start + max(1, int(batch))]]
            grads = {key: np.zeros_like(policy.params[key]) for key in names}
            for x, label, w in chunk:
                h = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
                logits = [policy.params[f"W2_{head}"] @ h + policy.params[f"b2_{head}"]
                          for head in heads]
                probs = [_softmax(head_logits) for head_logits in logits]
                qs = _q_of(label)
                sample_loss = 0.0
                # 隐层是各头共享的：每个头的 dlogits 各自回传到 h 再相加。
                dz = np.zeros_like(h)
                for head_index, (p, q) in enumerate(zip(probs, qs)):
                    sample_loss += float(-np.sum(q * np.log(p + 1e-12)))
                    dlogits = (p - q) * (w * inv_heads)
                    if entropy_coef:
                        entropy = _entropy(p)
                        dlogits = dlogits - (entropy_coef * w * inv_heads
                                             * p * (np.log(p + 1e-12) + entropy))
                    head = heads[head_index]
                    grads[f"W2_{head}"] += np.outer(dlogits, h)
                    grads[f"b2_{head}"] += dlogits
                    dz += policy.params[f"W2_{head}"].T @ dlogits
                sample_loss *= inv_heads
                epoch_loss += w * sample_loss
                dz = dz * (1.0 - h * h)
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
        result["loss"].append(round(epoch_loss / max(1e-9, total_weight), 6))
        result["epochs"] += 1

    correct = [0.0] * n_heads
    weight_sum = 0.0
    entropy_sum = 0.0
    for x, label, w in data:
        h = np.tanh(policy.params["W1"] @ x + policy.params["b1"])
        qs = _q_of(label)
        weight_sum += w
        for head_index, head in enumerate(heads):
            head_logits = policy.params[f"W2_{head}"] @ h + policy.params[f"b2_{head}"]
            p = _softmax(head_logits)
            correct[head_index] += w * float(int(np.argmax(p)) == int(label[head_index]))
            entropy_sum += w * max(0.0, _entropy(qs[head_index]))
    per_head = [round(value / max(1e-9, weight_sum), 4) for value in correct]
    result["perHeadAccuracy"] = per_head
    result["accuracy"] = round(sum(per_head) / float(n_heads), 4) if n_heads else 0.0
    result["teacherEntropy"] = round(
        entropy_sum / max(1e-9, weight_sum * n_heads), 6)
    policy.updates += int(result["epochs"])
    return result
