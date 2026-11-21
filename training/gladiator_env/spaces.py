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

OBS_DIM = 25
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
)

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


def encode_observation(
    observation: dict,
    username: str,
    step_fraction: float = 0.0,
    opponent_health: float | None = None,
    opponents: "set[str] | None" = None,
    reach: float = 3.99,
    since_attack: float = 5.0,
    self_velocity_y: float | None = None,
    attack_cooldown_remaining: float | None = None,
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
    """
    self_payload = observation.get("self") or {}
    pos = self_payload.get("pos") or {}
    sx, sy, sz = _f(pos.get("x")), _f(pos.get("y")), _f(pos.get("z"))
    yaw = _f(self_payload.get("yaw"))
    pitch = _f(self_payload.get("pitch"))

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
        vector[12] = 1.0 if target.get("onGround") else 0.0
        vector[13] = 1.0
        # 对手速度（缺字段=0）：水平速度用 √(vx²+vz²)，竖直分量单独给。
        velocity = target.get("velocity") or {}
        vx, vy, vz = _f(velocity.get("x")), _f(velocity.get("y")), _f(velocity.get("z"))
        vector[19] = _clamp(math.sqrt(vx * vx + vz * vz), 0.0, 10.0) / 10.0
        vector[20] = _clamp(vy, -5.0, 5.0) / 5.0
        # 水平距离 ≤ reach 才算「这一刀打得中」；竖直差再大也不该判 false
        # （原版攻击判定是 3D 距离，但训练里跳劈的干扰远小于水平贴身/拉开）。
        vector[22] = 1.0 if horizontal <= float(reach) else 0.0

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
    reach: float = 3.99,
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
            # 够不到时先靠近：攻击距离由 BotProfile.reach 决定（默认原版 3.99），硬打只会空挥
            # （运行时回 OUT_OF_REACH，日志里大量 `目标距离 4.02 > reach 3.99` 就是这么来的）。
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
