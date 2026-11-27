"""观测编码与动作映射（训练环境 ↔ 通道 B 协议之间的适配层）。

设计原则：
- 训练环境只消费通道 B 的 `OBSERVATION` / 发送 `ACTION`，不接触 mineflayer；
- 观测固定长度、固定 dtype，缺失字段有默认值（G-402）；
- 动作空间与协议动作表一一映射（G-403），复合动作 = 协议动作的有序列表。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .items import default_table

# --- 观测 ---------------------------------------------------------------

#: 观测总维数。G-711：25 → 37（末尾追加 12 维跨帧信息）；G-731：37 → 39
#: （末尾追加朝向误差 sin/cos）。**两轮都只在末尾追加**，索引 0–36 一动不动，
#: 旧模型右侧补零即可继续用（``policy._pad_obs_columns``）。
OBS_DIM = 39
OBS_FIELDS = (
    "self_health",       # 0  自身血量 / 20
    "self_food",         # 1  饱食度 / 20
    "self_saturation",   # 2  饱和度 / 5
    "self_on_ground",    # 3  是否着地
    "self_yaw_sin",      # 4  朝向 sin（角度环绕友好）
    "self_yaw_cos",      # 5  朝向 cos
    "self_pitch",        # 6  pitch / 90
    "target_dx",         # 7  对手相对位置 x / 16
    "target_dy",         # 8  对手相对位置 y / 8
    "target_dz",         # 9  对手相对位置 z / 16
    "target_distance",   # 10 对手距离 / 32
    "target_health",     # 11 对手血量 / 20
    "target_on_ground",  # 12 对手是否着地
    "has_target",        # 13 是否有可见对手
    "step_fraction",     # 14 当前步数 / 最大步数
    "state_alive",       # 15 自身状态是否 Alive
    "self_held_durability",   # 16 手里物品剩余耐久（0..1；无耐久=1）
    "self_armor_remaining",   # 17 护甲平均剩余耐久（0..1；没穿=0）
    "self_gapples",           # 18 金苹果数量 / 5
    "opponent_speed",         # 19 对手水平速度 √(vx²+vz²)，clip0–10 后 /10
    "opponent_vertical_speed",  # 20 对手竖直速度 vy，clip-5..5 后 /5
    "self_since_attack",      # 21 距上次 attack 动作的秒数，clip0–5 后 /5（初始 5）
    "in_reach",               # 22 双方水平距离 ≤ 自己 reach
    "self_vertical_speed",    # 23 自身 vy，clip-5..5 后 /5（下落为负 → 跳劈时机）
    "self_attack_cooldown",   # 24 距上次 attack 还差多少到 0.5s，clip0–0.5 后 /0.5（1=刚打完）
    # --- G-711：跨帧信息（25..36），顺序 = ACTION_HEADS_V2 ---
    "prev_move",              # 25 上一帧 move 头 / 5
    "prev_look_yaw",          # 26 上一帧 look_yaw 头 / 7
    "prev_look_pitch",        # 27 上一帧 look_pitch 头 / 5
    "prev_attack",            # 28 上一帧 attack 头 / 2
    "prev_use",               # 29 上一帧 use 头 / 3
    "prev_jump",              # 30 上一帧 jump 头 / 2
    "prev_sneak",             # 31 上一帧 sneak 头 / 2
    "prev_sprint",            # 32 上一帧 sprint 头 / 2
    "prev_slot",              # 33 上一帧 slot 头 / 9
    "prev_sprint_reset",      # 34 上一帧 sprint_reset 头 / 3
    "frames_since_attack",    # 35 距上次 attack 的帧数，clip0–50 后 /50（0=本帧刚出手）
    "frames_since_hit",       # 36 距上次受伤的帧数，clip0–50 后 /50（0=本帧刚挨打）
    # --- G-731：朝向误差（37/38）。sin/cos 由「指向对手的 bearing − 自身 yaw」得来 ----
    "target_bearing_sin",     # 37 sin(yawErr)：正值 = 对手在自己左手边（1.8 yaw 逆时针）
    "target_bearing_cos",     # 38 cos(yawErr)：1 = 正对对手，-1 = 完全背对
)

#: G-711：「上一帧十头动作」在观测里的起始下标（25..34，顺序 = ``ACTION_HEADS_V2``）。
PREV_ACTION_OFFSET = 25
#: G-711：``frames_since_attack`` / ``frames_since_hit`` 的下标。
FRAMES_SINCE_ATTACK_INDEX = 35
FRAMES_SINCE_HIT_INDEX = 36
#: G-711：帧数特征的分母（50 帧 ≈ 20Hz 下的 2.5 秒；再久就该饱和，不必区分）。
FRAMES_NORMALIZER = 50.0

#: G-731：朝向误差 sin/cos 的下标。
BEARING_SIN_INDEX = 37
BEARING_COS_INDEX = 38
#: G-731：``self.yaw`` 缺失时这两维写 ``(0, 0)``——真实值恒满足 sin²+cos²=1，
#: 所以 ``(0,0)`` 可以无歧义地表示「没有朝向信息」，不必伪造一个角度。
BEARING_UNKNOWN = (0.0, 0.0)

#: 观测裁剪范围：协议字段理论上有限，但 NaN/异常值不应该进网络。
OBS_CLIP = 5.0

#: 1.8 受击无敌帧长度（秒）：两次有效伤害之间的最小间隔。
#: ``self_attack_cooldown`` 以它为归一化分母（1 = 刚打完，0 = 可以打）。
ATTACK_COOLDOWN_S = 0.5

NUM_ACTIONS = 11
ACTION_NAMES = (
    "idle",
    "approach",
    "attack",
    "strafe_left",
    "strafe_right",
    "retreat",
    "jump",
    "use_rod",       # 切鱼竿右键击退（中距离）
    "eat_gapple",    # 吃金苹果回血（需要 1.6s 按住）
    "block",         # 举剑格挡（自动防砍的显式版本）
    "switch_sword",  # 从其它物品切回剑（训练切物品能力）
)

#: 复合动作的移动持续时间（毫秒）。控制频率 10Hz 时略小于一帧，避免动作叠加。
MOVE_MS = 180

#: mineflayer 的 yaw 约定：yaw=0 面向 +z，正值逆时针（俯视）。目标方向为
#: `atan2(-dx, -dz)`；如真实服上标定相反，把该常量改成 ``"pos"``。
YAW_FORMULA = "neg"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def nearest_player(
    observation: dict,
    username: str,
    opponents: "set[str] | None" = None,
) -> dict | None:
    """在 OBSERVATION.entities 里找最近的其它玩家（排除自己）。

    ``opponents`` 非空时只接受名单内的玩家（小写比较）：训练时传入参赛 Bot 用户名，
    这样进入服务器的观战者/路人不会被当成目标，Bot 也不会去追它们。
    """
    allowed = {name.lower() for name in opponents} if opponents else None
    best = None
    best_distance = float("inf")
    for entity in observation.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if entity.get("kind") != "player":
            continue
        name = str(entity.get("name", ""))
        if name.lower() == username.lower():
            continue
        if allowed is not None and name.lower() not in allowed:
            continue
        distance = _f(entity.get("distance"), float("inf"))
        if distance < best_distance:
            best_distance = distance
            best = entity
    return best


# --- 对手着地推断（docs/observation-blindspots.md §2 #12） -----------------
#
# 为什么不能直接读协议里的 ``entities[].onGround``：
# mineflayer 的实体对象（prismarine-entity）在构造函数里把 ``onGround`` 置 **true**，
# 之后只有 ``lib/plugins/physics.js`` 会更新——而它只算 **Bot 自己** 的物理。
# 远端玩家实体的 ``onGround`` 因此是**构造默认值常真**，转发过来只会得到恒 1.0，
# 和恒 0 一样是死特征，而且更隐蔽（看起来「合理」）。所以训练侧自己从
# 「竖直速度 + y 是否变化」推，运行时应按同一套规则推导（见文档里的 JS diff）。

#: 判定「竖直速度归零」的阈值（格/tick）。1.8 站立时 vy 恒为 0，自由落体每 tick −0.08。
GROUND_VY_EPS = 0.02
#: 判定「这一帧 y 没动」的阈值（格）。静立时 y 逐位不变，0.001 只挡浮点抖动。
GROUND_Y_EPS = 0.001
#: 连续多少帧「vy≈0 且 y 未变」才认作着地。取 2 是为了滤掉跳跃顶点：
#: 顶点那一帧 vy≈0、y 位移也接近 0，但下一帧必然转为下落。
GROUND_CONFIRM_FRAMES = 2


def _float_or_none(value: Any) -> float | None:
    """宽容转 float；非数/NaN/Inf 一律 None（与 ``_f`` 的区别：不掩盖「没有」）。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def estimate_on_ground(
    entity: dict,
    previous_y: float | None = None,
    *,
    vy_eps: float = GROUND_VY_EPS,
    y_eps: float = GROUND_Y_EPS,
) -> bool | None:
    """单帧推断「这个实体是否着地」。

    @returns ``True`` 着地 / ``False`` 明确在空中 / ``None`` 无从判断

    判据（按优先级）：
    1. ``vy < -vy_eps`` → 在下落，无争议；
    2. ``|vy| > vy_eps`` → 在上升或快速运动，不是着地；
    3. 否则需要 ``previous_y``：``|y - previous_y| <= y_eps`` 才算着地。
       没有上一帧时返回 ``None``——单帧的「vy≈0」无法区分静立与跳跃顶点。
    """
    if not isinstance(entity, dict):
        return None
    pos = entity.get("pos") or {}
    velocity = entity.get("velocity") or {}
    y = _float_or_none(pos.get("y"))
    vy = _float_or_none(velocity.get("y"))
    if y is None:
        return None
    if vy is None:
        return None
    if vy < -vy_eps:
        return False
    if abs(vy) > vy_eps:
        return False
    if previous_y is None:
        return None
    return abs(y - previous_y) <= y_eps


class GroundEstimator:
    """逐观测者维护「对手是否着地」的推断状态。

    每个 Bot 一个实例：``update()`` 吃它自己那一帧的 ``entities``，``as_map()`` 交给
    :func:`encode_observation` 的 ``opponent_ground`` 参数。状态是「上一帧 y + 连续帧计数」，
    所以必须在**每步编码前**调一次 ``update()``（``GladiatorEnv._encode_all`` 是唯一入口）。

    首帧一律给 ``False``（没有上一帧 y，无法区分静立与跳跃顶点）；之后要连续
    ``confirm_frames`` 帧判为「可着地」才翻成 ``True``，即最快在**第 3 帧**（首帧 + 2 帧确认）
    报 ``True``——20Hz 下约 100ms。
    """

    def __init__(
        self,
        *,
        vy_eps: float = GROUND_VY_EPS,
        y_eps: float = GROUND_Y_EPS,
        confirm_frames: int = GROUND_CONFIRM_FRAMES,
    ) -> None:
        self.vy_eps = float(vy_eps)
        self.y_eps = float(y_eps)
        self.confirm_frames = max(1, int(confirm_frames))
        self.reset()

    def reset(self) -> None:
        self._last_y: dict[str, float] = {}
        self._streak: dict[str, int] = {}
        self._grounded: dict[str, bool] = {}
        self._frame: "object | None" = None

    def update(self, entities: "Iterable[dict] | None", frame: "object | None" = None) -> dict[str, bool]:
        """推进一帧；返回 ``{小写名字: 是否着地}``（沿用上一帧的名字集合）。

        ``frame`` 给「同一帧可能被多处编码」的场景用（学习方 ``_encode_all`` 与
        策略对手的观测在同一 step 内各编码一次）：**同一个 frame 只推进一次**，
        否则连续帧计数会被推快一倍，跳跃顶点的误判就漏出来了。
        传 ``None`` 表示不做幂等保护（纯函数式用法/单测）。
        """
        if frame is not None and frame == self._frame:
            return self._grounded
        if frame is not None:
            self._frame = frame
        for entity in entities or []:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).lower()
            if not name:
                continue
            y = _float_or_none((entity.get("pos") or {}).get("y"))
            state = estimate_on_ground(entity, self._last_y.get(name),
                                       vy_eps=self.vy_eps, y_eps=self.y_eps)
            if state is True:
                streak = self._streak.get(name, 0) + 1
                self._streak[name] = streak
                self._grounded[name] = streak >= self.confirm_frames
            else:
                # False（在空中）与 None（不知道）都清零：连续确认必须是真的连续。
                self._streak[name] = 0
                self._grounded[name] = False
            if y is not None:
                self._last_y[name] = y
        return self._grounded

    def as_map(self) -> dict[str, bool]:
        """当前推断结果（拷贝，避免调用方改到内部状态）。"""
        return dict(self._grounded)


def episode_fraction(steps: int, max_steps: int) -> float:
    """本局进度 0..1（观测 ``step_fraction``）。

    口径 = **本局已过时间 / 本局时长上限**：在训练里步长固定
    （``steps × 1/control_hz``），所以等价于 ``steps / max_steps``；运行时按
    ``elapsed / roundSeconds`` 算，两边是同一个量。``max_steps <= 0`` 视为「无上限」→ 0。
    """
    total = _float_or_none(max_steps)
    if total is None or total <= 0.0:
        return 0.0
    done = _float_or_none(steps)
    if done is None:
        return 0.0
    return _clamp(done / total, 0.0, 1.0)


def attack_cooldown_remaining(since_attack: float | None) -> float:
    """距上次出手还差多少秒到 0.5s 无敌帧结束（观测 ``self_attack_cooldown``）。

    ``0.0`` = 可以打、``ATTACK_COOLDOWN_S`` = 刚打完。既然口径是「没有出手就一定 ≥0.5s」，
    所以传 ``None``（不知道上次出手时间）按 **0.0**（= 视为可打），与
    ``encode_observation`` 的缺省语义、以及 ``since_attack`` 缺省 5s 保持一致。
    """
    value = _float_or_none(since_attack)
    if value is None:
        return 0.0
    return _clamp(ATTACK_COOLDOWN_S - value, 0.0, ATTACK_COOLDOWN_S)


def wrap_pi(angle: float) -> float:
    """把弧度角归一到 ``[-π, π)``（跨 ±180° 折返用）。

    ``(x + π) % 2π - π`` 是区间映射的标准写法：``190° → -170°``、``-190° → 170°``，
    ``180°`` 落在 ``-π``（左闭右开，避免「+π 与 −π 同一个方向却编码成两个值」）。
    """
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def wrap180(angle: float) -> float:
    """角度制版本的 :func:`wrap_pi`（结果落在 ``[-180, 180)``）。"""
    return math.degrees(wrap_pi(math.radians(float(angle))))


def yaw_error(dx: float, dz: float, self_yaw: float) -> float:
    """指向对手的 bearing 与自身 yaw 的差（弧度，已折返到 ``[-π, π)``）。

    ``bearing = atan2(-dx, dz)`` 是 1.8 的 yaw 约定（0 = +Z、+90° = −X、−90° = +X），
    与运行时 ``index.js::describeFacing`` 的 ``bearing`` / ``arena`` 的 ``yawTowards`` 同口径。
    正值 = 需要把 yaw 加大（逆时针）才对得上对手。
    """
    return wrap_pi(math.atan2(-dx, dz) - float(self_yaw))


def bearing_sin_cos(dx: float, dz: float, self_yaw: float) -> tuple[float, float]:
    """朝向误差的 ``(sin, cos)``（G-731 观测 37/38）。

    用 sin/cos 而不是直接给角度：角度在 ±180° 处不连续，网络学起来会看到「跳变」；
    sin/cos 是连续的，且两者合起来等价于角度（``atan2(sin, cos)`` 可还原）。
    """
    error = yaw_error(dx, dz, self_yaw)
    return math.sin(error), math.cos(error)


def self_yaw_of(observation: dict) -> float | None:
    """读观测里的自身朝向（弧度）；缺字段/非数返回 ``None``。

    单独一个入口是为了让「朝向到底有没有」这件事只有一种判法：编码器与环境的
    ``yawAvailable`` 诊断都走这里，避免两边对「0 是默认值还是真实读数」理解不同。
    """
    payload = observation.get("self") if isinstance(observation, dict) else None
    if not isinstance(payload, dict):
        return None
    return _float_or_none(payload.get("yaw"))


def encode_observation(
    observation: dict,
    username: str,
    step_fraction: float = 0.0,
    opponent_health: float | None = None,
    opponents: "set[str] | None" = None,
    reach: float = 3.0,
    since_attack: float = 5.0,
    self_velocity_y: float | None = None,
    attack_cooldown_remaining: float | None = None,
    prev_action: "Iterable[int] | None" = None,
    frames_since_attack: float | None = None,
    frames_since_hit: float | None = None,
    opponent_ground: "dict[str, bool] | None" = None,
    self_yaw: float | None = None,
) -> np.ndarray:
    """把一条 OBSERVATION 编码成固定长度的 float32 向量。

    ``opponent_health`` 由环境从对手自己的观测里取（协议实体列表不带血量）。
    ``opponents`` 见 :func:`nearest_player`。
    ``reach`` 是**自己**的攻击距离（P1 新增 ``in_reach`` 用）；
    ``since_attack`` 是距上次 attack 动作的秒数（P1 新增 ``self_since_attack`` 用，默认 5）。
    ``self_velocity_y`` 是自身竖直速度（G-702：跳劈时机的唯一直接信号）；
    不传则回落读 ``observation.self.velocity.y``（运行时从 mineflayer 补的字段）。
    ``attack_cooldown_remaining`` 是距上次 attack 还差多少秒到 0.5s 无敌帧结束（G-702）；
    环境按 ``0.5 - 距上次下发 attack 的秒数`` 算好后传入，不传按 0（= 可以打）。

    ``opponent_ground`` 是 ``{小写用户名: 是否着地}``（``GroundEstimator.as_map()``）。
    优先级：本参数 > 协议字段 ``target["onGround"]``（**只在它是真 bool 时**才采信）>
    :func:`estimate_on_ground` 的单帧推断（无历史，判不出着地 → 0）。

    G-711（跨帧信息，索引 25–36）：

    - ``prev_action``：上一帧的**十头下标**（顺序 = ``ACTION_HEADS_V2``），每维
      ``idx / head_size``。``None`` 表示「不知道」（v1 策略/新开一局），10 维全 0；
    - ``frames_since_attack`` / ``frames_since_hit``：距上次出手 / 受伤的**帧数**，
      clip 到 0–50 再 /50。``None`` 视为「很久以前」（= 1.0，与 ``since_attack`` 缺省 5s
      同口径），这样旧调用方不传时不会假装「刚打完」。

    G-731（朝向误差，索引 37/38）：``self_yaw`` 是自身朝向（弧度，mineflayer 口径）。
    不传则回落到 ``observation.self.yaw``；**两边都拿不到时写 ``(0, 0)`` 并在环境侧记
    ``yawAvailable=False``**（``(0,0)`` 不可能是真实的 sin/cos，所以是无歧义的「不知道」）。
    """
    self_payload = observation.get("self") or {}
    pos = self_payload.get("pos") or {}
    sx, sy, sz = _f(pos.get("x")), _f(pos.get("y")), _f(pos.get("z"))
    yaw = _f(self_payload.get("yaw"))
    pitch = _f(self_payload.get("pitch"))
    # G-731：朝向读没读到要单独判——``0.0`` 是合法角度（面向 +Z），不能用真假值区分。
    known_yaw = _float_or_none(self_yaw) if self_yaw is not None else self_yaw_of(observation)

    target = nearest_player(observation, username, opponents)
    state = str(observation.get("state", "Alive"))

    vector = np.zeros(OBS_DIM, dtype=np.float32)
    vector[0] = _f(self_payload.get("health"), 20.0) / 20.0
    vector[1] = _f(self_payload.get("food"), 20.0) / 20.0
    vector[2] = _f(self_payload.get("saturation")) / 5.0
    vector[3] = 1.0 if self_payload.get("onGround") else 0.0
    vector[4] = math.sin(yaw)
    vector[5] = math.cos(yaw)
    vector[6] = _clamp(pitch / (math.pi / 2.0), -1.0, 1.0)
    vector[13] = 0.0
    vector[14] = _clamp(step_fraction, 0.0, 1.0)
    vector[15] = 1.0 if state == "Alive" else 0.0
    held = self_payload.get("heldItem") or {}
    held_durability = (held.get("durability") or {}).get("remaining")
    vector[16] = _clamp(_f(held_durability, 1.0), 0.0, 1.0)
    vector[17] = _clamp(_f(self_payload.get("armorRemaining")), 0.0, 1.0)
    inventory = self_payload.get("inventory") or []
    gapples = sum(_f(item.get("count")) for item in inventory if item.get("name") == "golden_apple")
    vector[18] = _clamp(gapples / 5.0, 0.0, 1.0)

    # P1：距上次 attack 的时间（初始 5 秒 = 已经"冷却完毕"）。
    vector[21] = _clamp(_f(since_attack, 5.0), 0.0, 5.0) / 5.0

    # G-702（index 23）：自身竖直速度。跳劈的**唯一**直接时序信号——
    # 上升段 vy>0，到顶点 vy≈0，下落段 vy<0；「下落中且水平贴身」才是跳劈窗口。
    # 只在末尾追加，旧 23 维模型右侧补零即可续训。
    if self_velocity_y is None:
        self_velocity_y = ((self_payload.get("velocity") or {}).get("y"))
    vector[23] = _clamp(_f(self_velocity_y), -5.0, 5.0) / 5.0

    # G-702（index 24）：出手冷却还剩多少（1=刚打完，0=可以打）。
    # 与 index 21 的区别：21 是「过了多久」（0..5s，按 5s 归一），24 是「还差多久到 0.5s」，
    # 在 0.5s 这个真正决定「这一刀有没有伤害」的尺度上分辨得开。
    remaining = 0.0 if attack_cooldown_remaining is None else _f(attack_cooldown_remaining)
    vector[24] = _clamp(remaining, 0.0, ATTACK_COOLDOWN_S) / ATTACK_COOLDOWN_S

    # G-711（index 25..34）：上一帧的十头动作。BC 后 move/sprint 两个头一致率只有
    # 0.85/0.87，因为「上一帧走了哪一步」不在观测里——单帧静力学推不出换向节拍。
    # 归一化成 idx / head_size，落在 0..1，与其余维同量级。
    if prev_action is not None:
        picked = list(prev_action)
        for offset, (head, size) in enumerate(ACTION_HEADS_V2.items()):
            index = PREV_ACTION_OFFSET + offset
            if offset >= len(picked):
                break
            try:
                value = int(picked[offset])
            except (TypeError, ValueError):
                value = 0
            vector[index] = _clamp(value / float(size), 0.0, 1.0)

    # G-711（index 35/36）：跨帧计数。攻击/受伤都是「事件」，而 0.2s 一帧的采样
    # 经常整帧跳过事件本身（1.8 的无敌帧 500ms、死亡→重生 2 tick），所以给「距上次
    # 事件多少帧」，让策略能自己推断节拍。None = 不知道 → 按饱和值 1.0。
    frames_attack = FRAMES_NORMALIZER if frames_since_attack is None else _f(frames_since_attack)
    frames_hit = FRAMES_NORMALIZER if frames_since_hit is None else _f(frames_since_hit)
    vector[FRAMES_SINCE_ATTACK_INDEX] = (
        _clamp(frames_attack, 0.0, FRAMES_NORMALIZER) / FRAMES_NORMALIZER)
    vector[FRAMES_SINCE_HIT_INDEX] = (
        _clamp(frames_hit, 0.0, FRAMES_NORMALIZER) / FRAMES_NORMALIZER)

    if target is not None:
        tpos = target.get("pos") or {}
        dx = _f(tpos.get("x")) - sx
        dy = _f(tpos.get("y")) - sy
        dz = _f(tpos.get("z")) - sz
        horizontal = math.sqrt(dx * dx + dz * dz)
        distance = _f(target.get("distance"), math.sqrt(dx * dx + dy * dy + dz * dz))
        vector[7] = dx / 16.0
        vector[8] = dy / 8.0
        vector[9] = dz / 16.0
        vector[10] = distance / 32.0
        vector[11] = _clamp(
            (_f(opponent_health) if opponent_health is not None else _f(target.get("health"), 20.0)) / 20.0,
            0.0,
            1.0,
        )
        # 对手是否着地（原为恒 0 的死特征）。取数顺序见 ``opponent_ground`` 的说明：
        # 环境推断 > 协议真 bool 字段 > 单帧推断（判不出就当「未着地」）。
        grounded = None
        if opponent_ground:
            grounded = opponent_ground.get(str(target.get("name", "")).lower())
        if grounded is None:
            flag = target.get("onGround")
            if isinstance(flag, bool):
                grounded = flag
        if grounded is None:
            grounded = estimate_on_ground(target)
        vector[12] = 1.0 if grounded else 0.0
        vector[13] = 1.0
        # 对手速度（缺字段=0）：水平速度用 √(vx²+vz²)，竖直分量单独给。
        velocity = target.get("velocity") or {}
        vx, vy, vz = _f(velocity.get("x")), _f(velocity.get("y")), _f(velocity.get("z"))
        vector[19] = _clamp(math.sqrt(vx * vx + vz * vz), 0.0, 10.0) / 10.0
        vector[20] = _clamp(vy, -5.0, 5.0) / 5.0
        # 水平距离 ≤ reach 才算「这一刀打得中」；竖直差再大也不该判 false
        # （原版攻击判定是 3D 距离，但训练里跳劈的干扰远小于水平贴身/拉开）。
        vector[22] = 1.0 if horizontal <= float(reach) else 0.0
        # G-731（index 37/38）：朝向误差。索引 4/5 的 self_yaw_sin/cos 只说「朝哪」，
        # 不说「对手在哪」，模型因此学不出「先转向再走」——真机表现为被击退后一路背身跑。
        # 这里给「两者之差」的 sin/cos，让 7–9 维的相对位置与 4/5 维的朝向在同一空间对齐。
        if known_yaw is not None:
            vector[BEARING_SIN_INDEX], vector[BEARING_COS_INDEX] = bearing_sin_cos(
                dx, dz, known_yaw)
        # 无 yaw（协议没带）或无目标：保持 (0,0) = 「不知道」，不假装正对。

    return np.clip(vector, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# --- 动作 ---------------------------------------------------------------


def yaw_to_target(dx: float, dz: float) -> float:
    """返回朝向目标所需的 yaw（弧度，mineflayer 约定）。"""
    if YAW_FORMULA == "pos":
        return math.atan2(dx, dz)
    return math.atan2(-dx, -dz)


def pitch_to_target(dy: float, horizontal_distance: float) -> float:
    """返回朝向目标所需的 pitch（弧度；mineflayer 正值向下）。"""
    if horizontal_distance < 1e-6:
        return 0.0
    return math.atan2(-dy, horizontal_distance)


def look_at(target: dict, self_pos: dict) -> dict:
    """构造一个 `look` 协议动作，让 Bot 面朝 target。"""
    tpos = target.get("pos") or {}
    dx = _f(tpos.get("x")) - _f(self_pos.get("x"))
    dy = _f(tpos.get("y")) + 1.4 - (_f(self_pos.get("y")) + 1.62)
    dz = _f(tpos.get("z")) - _f(self_pos.get("z"))
    horizontal = math.sqrt(dx * dx + dz * dz)
    return {
        "type": "look",
        "params": {"yaw": yaw_to_target(dx, dz), "pitch": pitch_to_target(dy, horizontal)},
    }


# 观测距离与运行时 eyeDistance 的偏差余量（格）。
REACH_TOLERANCE = 0.15


def decode_action(
    index: int,
    observation: dict,
    username: str,
    opponents: "set[str] | None" = None,
    reach: float = 3.0,
) -> list[dict]:
    """把离散动作编号翻译成有序的协议动作列表（复合动作 = 多帧）。

    ``opponents`` 与 :func:`nearest_player` 一致：只把参赛 Bot 当作目标，
    观战者不会出现在目标里，也不会被 attack。
    """
    if index < 0 or index >= NUM_ACTIONS:
        raise ValueError(f"动作编号越界: {index}（合法范围 0..{NUM_ACTIONS - 1}）")
    name = ACTION_NAMES[index]
    self_payload = observation.get("self") or {}
    self_pos = self_payload.get("pos") or {}
    target = nearest_player(observation, username, opponents)
    distance = _f(target.get("distance")) if target else float("inf")

    if name == "idle":
        return [{"type": "idle", "params": {"durationMs": MOVE_MS}}]
    if name == "jump":
        return [{"type": "jump", "params": {}}]
    if name == "eat_gapple":
        # 血量还满时吃苹果是浪费，退化为待机（让模型学会“什么时候吃”）。
        health = _f(self_payload.get("health"), 20.0)
        if health >= 19.5:
            return [{"type": "idle", "params": {"durationMs": MOVE_MS}}]
        return [{"type": "use", "params": {"item": "gapple", "holdMs": 1600}}]
    if name == "block":
        return [{"type": "use", "params": {"item": "sword", "holdMs": 300}}]
    if name == "switch_sword":
        return [{"type": "use", "params": {"item": "sword", "holdMs": 50}}]

    if target is None:
        # 没有目标时退化为原地待机，避免把「前进」当成随机游走。
        return [{"type": "idle", "params": {"durationMs": MOVE_MS}}]

    look = look_at(target, self_pos)
    # 观测里的 distance 是「身体到身体」，运行时判 OUT_OF_REACH 用的是 eyeDistance
    # （+1.62 眼高 / +1.0 目标中心），同高度下会比观测值大 0.1~0.6 格。
    # 因此留 0.15 格余量：宁可多走一步靠近，也不要在边界上白白吃一次 OUT_OF_REACH。
    effective_reach = max(0.0, float(reach) - REACH_TOLERANCE)
    if name == "approach":
        # 距离超过 2.5 格就疾跑（必须与 forward 同时按才生效）；贴身时步行，避免冲过头。
        sprint = distance > 2.5
        duration = MOVE_MS if distance <= 4.0 else MOVE_MS * 2
        return [look, {"type": "move", "params": {
            "direction": "forward", "durationMs": duration, "sprint": sprint}}]
    if name == "retreat":
        return [look, {"type": "move", "params": {"direction": "back", "durationMs": MOVE_MS}}]
    if name == "strafe_left":
        return [look, {"type": "move", "params": {"direction": "left", "durationMs": MOVE_MS}}]
    if name == "strafe_right":
        return [look, {"type": "move", "params": {"direction": "right", "durationMs": MOVE_MS}}]
    if name == "use_rod":
        # 鱼竿在自己的攻击距离之外、但不超过 6 格时才有意义。
        if distance > 6.0 or distance < 2.0:
            return [look, {"type": "move", "params": {
                "direction": "forward", "durationMs": MOVE_MS, "sprint": True}}]
        return [look, {"type": "use", "params": {"item": "rod"}}]
    if name == "attack":
        entity_id = target.get("id")
        if entity_id is None or distance > effective_reach:
            # 够不到时先靠近：攻击距离由 BotProfile.reach 决定（默认原版 3.0），硬打只会空挥
            # （运行时回 OUT_OF_REACH，日志里大量 `目标距离 3.20 > reach 3.0` 就是这么来的）。
            return [look, {"type": "move", "params": {
                "direction": "forward", "durationMs": MOVE_MS, "sprint": distance > 2.5}}]
        # 近身：先转向目标、挥击、再补一次朝向，减少击退后的背对空挥。
        return [look, {"type": "attack", "params": {"targetEntityId": int(entity_id)}}, look]

    raise AssertionError(f"未处理的动作: {name}")


def describe_actions() -> Iterable[str]:
    return ACTION_NAMES


# --- 动作空间 v2（物品无关 + 可掩码） ---------------------------------------
#
# 背景见 ``docs/identity-and-action-space.md`` §2.2–2.3：v1 的 11 个宏动作把
# 「选槽 + 右键 + 按住 + 切回剑」写死在 ``decode_action`` 里，于是每加一个物品就要
# 加动作、改 ``NUM_ACTIONS``、重训。v2 把动作拆成**正交意图**：
#
#   移动 / 转视角 / 左键 / 右键 / 跳 / 蹲 / 疾跑 / 选槽
#
# 物品身份只进观测（category one-hot），不进动作；「这个物品怎么用」由
# ``items.json`` 的能力表回答（:mod:`gladiator_env.items`），因此新增物品零改代码。
#
# v1 原样保留：它是 stage 0 的冷启动基线，也是回归对照（规格 §2.5）。

#: ``move`` 头的取值 → 语义。0 是待机（松键），其余与 v1 的移动方向同名。
MOVE_VALUES_V2: tuple[str, ...] = ("idle", "forward", "back", "left", "right")
#: ``look_yaw`` 头：每 tick 的偏航增量（度）。粗调 ±30°、细调 ±3°。
LOOK_YAW_V2: tuple[float, ...] = (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0)
#: ``look_pitch`` 头：每 tick 的俯仰增量（度）。mineflayer 正值向下。
LOOK_PITCH_V2: tuple[float, ...] = (-15.0, -5.0, 0.0, 5.0, 15.0)
#: ``use`` 头：0=不按 / 1=单击 / 2=按住至松开（怎么按由能力表决定）。
USE_VALUES_V2: tuple[str | None, ...] = (None, "click", "hold")
#: ``attack`` / ``jump`` / ``sneak`` / ``sprint``：0/1。
BINARY_VALUES_V2: tuple[bool, ...] = (False, True)
#: ``sprint_reset`` 头（G-707）：**重置疾跑**的三种做法，顺序固定。
#:
#: 背景见 ``docs/knowledge/pvp-1.8-mechanics.md`` §3–§4：1.8 的击退大小由「命中帧是否疾跑」
#: 决定，而 v2 的 ``move`` 是**连续方向**（``forward`` 表示整帧按住），无法表达「松一帧 W」，
#: 于是 wtap 这类动作在旧九头里根本没有对应取值——老师演示不了，模型也学不到。
#:
#: - ``none``：不重置（默认，行为与九头完全一致）；
#: - ``release_w``：**同一帧内**先松 ``forward``、发出 attack、再按回 ``forward``
#:   （wtap / double tap；无刀时的对刀手段）；
#: - ``block_hit``：**同一帧内**先举剑（``use{sword,holdMs=300}``）再 attack
#:   （防砍 / block-hit；有刀时的对刀手段，兼吃 0.5× 伤害）。
#:
#: 组合权在模型手里（ADR-020）：环境与运行时都不做「命中后自动 reset」的隐式行为。
SPRINT_RESET_VALUES_V2: tuple[str, ...] = ("none", "release_w", "block_hit")
#: 热键栏槽位数（``slot`` 头的取值个数）。
HOTBAR_SLOTS = 9
#: 协议观测里热键栏第一个槽位的原始槽号（36..44）；``slot`` 头的取值是 ``槽号 - 36``。
HOTBAR_SLOT_BASE = 36

#: 十个头的取值个数。顺序即 ``head_offsets`` 的拼接顺序，**改动会作废旧模型**。
#:
#: G-707：末尾追加 ``sprint_reset``（3 档）→ nvec 从 9 头 / 37 维变成 10 头 / 40 维。
#: 追加在**末尾**是刻意的：一是旧 9 头 v2 模型可以只接管同名的 9 个头（见
#: ``gladiator_train.policy.adopt_weights``），二是 ``slot`` 的下标仍是 28，
#: 已有日志/探针脚本里写死的头下标不会错位。
ACTION_HEADS_V2: dict[str, int] = {
    "move": len(MOVE_VALUES_V2),          # 5
    "look_yaw": len(LOOK_YAW_V2),         # 7
    "look_pitch": len(LOOK_PITCH_V2),     # 5
    "attack": 2,                          # 2
    "use": len(USE_VALUES_V2),            # 3
    "jump": 2,                            # 2
    "sneak": 2,                           # 2
    "sprint": 2,                          # 2
    "slot": HOTBAR_SLOTS,                 # 9
    "sprint_reset": len(SPRINT_RESET_VALUES_V2),  # 3（G-707）
}

#: v2 动作向量总维度（各头取值个数之和）。
ACTION_DIM_V2: int = sum(ACTION_HEADS_V2.values())

#: 直接可喂给 ``gymnasium.spaces.MultiDiscrete`` 的 nvec。
MULTI_DISCRETE_V2: tuple[int, ...] = tuple(ACTION_HEADS_V2.values())


def head_offsets() -> dict[str, int]:
    """每个头在 ``ACTION_DIM_V2`` 维 one-hot / 扁平编号里的起始下标（按声明顺序累加）。

    ``{"move": 0, "look_yaw": 5, "look_pitch": 12, ...}``，最后一项 ``sprint_reset`` 是 37。
    新增头一律追加在末尾，已发布的下标（例如 ``slot=28``）因此保持稳定。
    """
    offsets: dict[str, int] = {}
    cursor = 0
    for head, size in ACTION_HEADS_V2.items():
        offsets[head] = cursor
        cursor += size
    return offsets


def _head_value(head: str, flat_value: object) -> int:
    """取出一个头的取值并校验范围（越界立刻报错，别让错误索引悄悄变成 idle）。"""
    try:
        value = int(flat_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"intent 的 {head} 必须是整数，收到 {flat_value!r}") from exc
    size = ACTION_HEADS_V2[head]
    if value < 0 or value >= size:
        raise ValueError(f"intent 的 {head} 越界: {value}（合法范围 0..{size - 1}）")
    return value


def decode_intent(flat: "int | Iterable[int] | np.ndarray") -> dict:
    """把 v2 的离散动作翻译成意图字典。

    三种输入都接受（都是「同一件事的三种写法」）：

    - 长度 10 的序列：``MultiDiscrete`` 每个头一个数（**训练主路径**）；
    - 长度 40 的序列：每个头一段的 one-hot / 二值展开（``ACTION_DIM_V2`` 就是它）；
    - ``int``：40 维展开里唯一那个 1 的下标（日志/BC 里压缩成一个数的写法）。

    返回形态固定（键顺序也固定，方便直接序列化成 JSON）::

        {"move": str, "look": {"dyaw": float, "dpitch": float},
         "attack": bool, "use": str|None, "slot": int,
         "jump": bool, "sneak": bool, "sprint": bool}

    注意 ``look`` 给的是**增量**（度），不是绝对朝向：v1 里「内部算朝向」的做法
    让策略学不到瞄准，v2 把瞄准交回给策略（规格 §2.2）。

    G-707：``sprint_reset`` 为 ``none`` 时**不写** ``sprintReset`` 键，因此旧代码/旧日志
    看到的 intent 形状与九头时代逐字节一致；只有真正要重置疾跑时才多出这个字段
    （运行时的 ``normalizeIntent`` 同样按「缺省 = none」处理）。
    """
    heads = list(ACTION_HEADS_V2)
    offsets = head_offsets()
    if isinstance(flat, (int, np.integer)):
        index = int(flat)
        if index < 0 or index >= ACTION_DIM_V2:
            raise ValueError(
                f"intent 扁平编号越界: {index}（合法范围 0..{ACTION_DIM_V2 - 1}）")
        one_hot = np.zeros(ACTION_DIM_V2, dtype=np.int64)
        one_hot[index] = 1
        ordered = _split_one_hot(one_hot, heads, offsets)
    else:
        values = [int(value) for value in flat]  # type: ignore[union-attr]
        if len(values) == len(heads):
            ordered = values
        elif len(values) == ACTION_DIM_V2:
            ordered = _split_one_hot(values, heads, offsets)
        else:
            raise ValueError(
                f"intent 需要 {len(heads)} 个头或 {ACTION_DIM_V2} 维展开，收到 {len(values)} 个")

    picked = {head: _head_value(head, ordered[index]) for index, head in enumerate(heads)}
    intent = {
        "move": MOVE_VALUES_V2[picked["move"]],
        "look": {
            "dyaw": LOOK_YAW_V2[picked["look_yaw"]],
            "dpitch": LOOK_PITCH_V2[picked["look_pitch"]],
        },
        "attack": BINARY_VALUES_V2[picked["attack"]],
        "use": USE_VALUES_V2[picked["use"]],
        "slot": picked["slot"],
        "jump": BINARY_VALUES_V2[picked["jump"]],
        "sneak": BINARY_VALUES_V2[picked["sneak"]],
        "sprint": BINARY_VALUES_V2[picked["sprint"]],
    }
    reset = SPRINT_RESET_VALUES_V2[picked["sprint_reset"]]
    if reset != "none":
        intent["sprintReset"] = reset
    return intent


def _split_one_hot(values: "Iterable[int]", heads: list[str], offsets: dict[str, int]) -> list[int]:
    """把 ``ACTION_DIM_V2`` 维展开按头切段，取每段的 argmax（全 0 段视为 0）。

    二值头（attack/jump/…）的展开段只有两个元素，argmax 与「1 在不在」等价；
    多值头必须是 one-hot，否则取第一个最大值的下标——训练里由策略采样保证合法。
    """
    flat = [int(value) for value in values]
    picked: list[int] = []
    for head in heads:
        start = offsets[head]
        size = ACTION_HEADS_V2[head]
        segment = flat[start:start + size]
        best = 0
        best_value = None
        for index, value in enumerate(segment):
            if best_value is None or value > best_value:
                best, best_value = index, value
        picked.append(best if best_value else 0)
    return picked



def action_mask(observation: dict, item_table: "Any | None" = None) -> dict[str, list[bool]]:
    """按当前观测/背包算 v2 的 action mask（``dict head -> list[bool]``）。

    规则（规格 §2.3「动作合法性由该表 + 背包状态算 action mask」）：

    - ``slot[i]``：热键栏第 i 格有物品才可用（空槽给策略点没意义，白占容量）；
    - ``use``：``0``（不按）恒可用；``1/2`` 需要**手持物品**在能力表里
      ``mode != none``——空手、材料、未知物品都会把这两档屏蔽掉；
    - 其余头（move / look / attack / jump / sneak / sprint）恒可用：v2 不做
      reach 屏蔽（那是运行时 attack-guard 的职责，环境侧不改 reward/termination）。
    - ``sprint_reset``（G-707）恒定**全开**：三种取值都不依赖物品——
      ``release_w`` 只动 ``forward`` 键；``block_hit`` 在运行时自己会先切剑，
      而「手上有没有剑」此刻可能只是观测晚了一帧（协议没有攻击中状态），
      在环境侧按物品屏蔽它只会让模型学到「有剑时才敢 reset」的伪相关。

    「手持物品」而不是「任意槽里的物品」是刻意的：意图里同时改槽与按右键时，
    运行时先选槽再按右键，但**本帧观测**只能看到旧手持物。保守一点不会让策略
    学到非法动作，代价只是「换槽那一帧不能同时按右键」。
    """
    table = item_table if item_table is not None else default_table()
    mask: dict[str, list[bool]] = {
        head: [True] * size for head, size in ACTION_HEADS_V2.items()
    }

    # --- slot：只看热键栏（36..44），背包里的东西要先挪过来 ---
    occupied: set[int] = set()
    self_payload = (observation or {}).get("self") or {}
    for item in self_payload.get("inventory") or []:
        if not isinstance(item, dict):
            continue
        try:
            slot = int(item.get("slot"))
        except (TypeError, ValueError):
            continue
        if HOTBAR_SLOT_BASE <= slot < HOTBAR_SLOT_BASE + HOTBAR_SLOTS:
            occupied.add(slot - HOTBAR_SLOT_BASE)
    for index in range(HOTBAR_SLOTS):
        mask["slot"][index] = index in occupied

    # --- use：手持物品必须「能用右键」 ---
    held = (self_payload.get("heldItem") or {}).get("name")
    held_name = str(held or "").strip()
    usable = bool(held_name) and held_name.lower() != "air" and table.is_usable(held_name)
    mask["use"][0] = True
    mask["use"][1] = usable
    mask["use"][2] = usable
    # --- sprint_reset：恒定全开（见 docstring；不引入物品条件） ---
    mask["sprint_reset"] = [True] * ACTION_HEADS_V2["sprint_reset"]
    return mask


# --- v1 宏动作 → v2 十头意图（G-720） ----------------------------------------
#
# 为什么需要：真人录像/旧日志里只有「一帧一个宏动作」的标签（``ACTION_NAMES`` 的
# 下标），而 v2 模型吃的是十个头。没有这层映射时 ``import_human_match.py`` 只能训 v1，
# 现役最强的 v2 模型（``v2-v4-dagger3`` 等）**一口真人数据都吃不下**（会抛
# ``ValueError: 第 0 帧 v2 动作应有 10 个头…收到 1 个``）。
#
# 口径：这是**有损**的标签翻译，只表达「这一帧想干什么」，不表达「怎么走位到那一步」。
# - 宏动作里的 ``approach``/``strafe``/``retreat`` 只有方向，没有「离多远」，所以
#   ``move`` 头直接照抄方向、``look`` 头给中性（0°, 0°）；
# - ``attack`` 宏动作默认带前压（v1 的 ``decode_action`` 就是这么做的），故 ``move=forward``
#   且 ``sprint=True``；``sprint_reset`` 一律 ``none``——真人那一刻按没按 W 我们看不到，
#   猜一个反而会教坏模型（知识库 §2.5 的结论是「KB 由命中帧是否疾跑决定」）；
# - ``use_rod``/``eat_gapple``/``block`` 都映射到 ``use`` 头（单击/按住），
#   具体用哪个物品由运行时按 ``slot`` + 能力表决定；
# - ``switch_sword`` 在十个头里没有对应维度（切物品是 ``slot`` 头 + 能力表），
#   这里退回 ``move=idle`` 并保持 ``slot=0``，只保证「不产生攻击」。
#
# 中性取值：``look_yaw``/``look_pitch`` 里 0.0 度的下标由元组 ``index`` 查得，不写死数字。

_NEUTRAL_YAW = LOOK_YAW_V2.index(0.0)
_NEUTRAL_PITCH = LOOK_PITCH_V2.index(0.0)
_MOVE_FORWARD = MOVE_VALUES_V2.index("forward")
_MOVE_BACK = MOVE_VALUES_V2.index("back")
_MOVE_LEFT = MOVE_VALUES_V2.index("left")
_MOVE_RIGHT = MOVE_VALUES_V2.index("right")
_MOVE_IDLE = MOVE_VALUES_V2.index("idle")
_USE_NONE = USE_VALUES_V2.index(None)
_USE_CLICK = USE_VALUES_V2.index("click")
_USE_HOLD = USE_VALUES_V2.index("hold")
_RESET_NONE = SPRINT_RESET_VALUES_V2.index("none")


def _intent(move: int = _MOVE_IDLE, *, yaw: int = _NEUTRAL_YAW,
            pitch: int = _NEUTRAL_PITCH, attack: int = 0, use: int = _USE_NONE,
            jump: int = 0, sneak: int = 0, sprint: int = 0, slot: int = 0,
            reset: int = _RESET_NONE) -> list[int]:
    """按 ``ACTION_HEADS_V2`` 的声明顺序拼一个十头向量（用具名常量，不写死魔法数）。"""
    return [move, yaw, pitch, attack, use, jump, sneak, sprint, slot, reset]


#: v1 宏动作下标 → v2 十头向量。键是 ``ACTION_NAMES`` 的名字，值是 10 个下标。
MACRO_TO_INTENT: dict[str, tuple[int, ...]] = {
    "idle": tuple(_intent()),
    "approach": tuple(_intent(_MOVE_FORWARD, sprint=1)),
    "attack": tuple(_intent(_MOVE_FORWARD, attack=1, sprint=1)),
    "strafe_left": tuple(_intent(_MOVE_LEFT, sprint=1)),
    "strafe_right": tuple(_intent(_MOVE_RIGHT, sprint=1)),
    "retreat": tuple(_intent(_MOVE_BACK)),
    "jump": tuple(_intent(_MOVE_FORWARD, jump=1, sprint=1)),
    "use_rod": tuple(_intent(use=_USE_CLICK)),
    "eat_gapple": tuple(_intent(use=_USE_HOLD)),
    "block": tuple(_intent(use=_USE_HOLD)),
    "switch_sword": tuple(_intent()),
}


def macro_to_intent(macro_index: "int | str") -> tuple[int, ...]:
    """把 v1 宏动作（下标或名字）翻成 v2 的十头下标元组。

    未登记的宏动作按 ``idle`` 处理并**不静默吞掉**：调用方（导入脚本）会把
    ``ACTION_NAMES[index]`` 记进报告，所以真出现新宏动作时能在报告里看到
    ``mapped="idle"`` 而不是悄悄变成待机。
    """
    if isinstance(macro_index, str):
        name = macro_index
    else:
        index = int(macro_index)
        if index < 0 or index >= len(ACTION_NAMES):
            raise ValueError(f"宏动作下标越界: {index}（0..{len(ACTION_NAMES) - 1}）")
        name = ACTION_NAMES[index]
    return MACRO_TO_INTENT.get(name, MACRO_TO_INTENT["idle"])


def macro_to_intent_array(macro_index: "int | str") -> "np.ndarray":
    """:func:`macro_to_intent` 的 ndarray 版本（``dtype=int64``，方便整列堆叠）。"""
    return np.asarray(macro_to_intent(macro_index), dtype=np.int64)
