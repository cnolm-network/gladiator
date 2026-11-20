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

``meta`` 是 v2 的「观测之外的状态」入口（鱼竿是否可用、对手多久前攻击过、一帧多少秒），
缺省值都能跑，所以离线采集不需要额外接线。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, OBS_FIELDS

#: 默认老师版本（v1 = 规则老师，行为与 P1.5 完全一致）。
TEACHER_VERSION = "v1"
#: 可选老师版本。v2 在 v1 之上加间距/鱼竿/金苹果/暴击跳/w-tap/格挡六条规则。
SUPPORTED_TEACHER_VERSIONS: tuple[str, ...] = ("v1", "v2")

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

ACTION_IDLE = ACTION_NAMES.index("idle")
ACTION_APPROACH = ACTION_NAMES.index("approach")
ACTION_ATTACK = ACTION_NAMES.index("attack")
ACTION_RETREAT = ACTION_NAMES.index("retreat")
ACTION_JUMP = ACTION_NAMES.index("jump")
ACTION_USE_ROD = ACTION_NAMES.index("use_rod")
ACTION_EAT = ACTION_NAMES.index("eat_gapple")
ACTION_BLOCK = ACTION_NAMES.index("block")
ACTION_SWITCH_SWORD = ACTION_NAMES.index("switch_sword")

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
assert IDX_IN_REACH == OBS_DIM - 1, "in_reach 应是最后一个观测字段"
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


def make_teacher(version: str = TEACHER_VERSION) -> Callable[..., int]:
    """按版本号取老师：``"v1"`` → :func:`heuristic_action`，``"v2"`` → :class:`TeacherV2`。"""
    key = str(version or TEACHER_VERSION).strip().lower()
    if key == "v1":
        return heuristic_action
    if key == "v2":
        return TeacherV2()
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
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing 必须在 [0, 1) 内，收到 {label_smoothing}")
    data = [(np.asarray(x, dtype=np.float64).reshape(-1), int(a)) for x, a, _ret in samples]
    result = {"loss": [], "accuracy": 0.0, "samples": len(data), "epochs": 0,
              "temperature": float(temperature),
              "labelSmoothing": float(label_smoothing),
              "teacherEntropy": 0.0}
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
            for x, action in chunk:
                h, logits = _logits(policy.params, x)
                p = _softmax(logits)
                q = targets.get(action)
                if q is None:
                    q = targets[action] = teacher_target(
                        action, n_actions, temperature, label_smoothing)
                epoch_loss += float(-np.sum(q * np.log(p + 1e-12)))
                dlogits = p - q
                if entropy_coef:
                    entropy = _entropy(p)
                    dlogits = dlogits - entropy_coef * p * (np.log(p + 1e-12) + entropy)
                grads["W2"] += np.outer(dlogits, h)
                grads["b2"] += dlogits
                dz = (policy.params["W2"].T @ dlogits) * (1.0 - h * h)
                grads["W1"] += np.outer(dz, x)
                grads["b1"] += dz
            n = float(len(chunk))
            step += 1
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            for key in names:
                g = grads[key] / n  # 最小化方向（交叉熵）
                m[key] = b1 * m[key] + (1.0 - b1) * g
                v[key] = b2 * v[key] + (1.0 - b2) * (g * g)
                policy.params[key] -= lr * (m[key] / bc1) / (np.sqrt(v[key] / bc2) + eps)
        result["loss"].append(round(epoch_loss / max(1, len(data)), 6))
        result["epochs"] += 1

    correct = 0
    entropy_sum = 0.0
    for x, action in data:
        _h, logits = _logits(policy.params, x)
        p = _softmax(logits)
        correct += int(int(np.argmax(p)) == action)
        q = targets.get(action)
        if q is None:
            q = targets[action] = teacher_target(action, n_actions, temperature, label_smoothing)
        entropy_sum += max(0.0, _entropy(q))
    result["accuracy"] = round(correct / max(1, len(data)), 4)
    result["teacherEntropy"] = round(entropy_sum / max(1, len(data)), 6)
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
