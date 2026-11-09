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

# --- 观测 ---------------------------------------------------------------

OBS_DIM = 19
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
)

#: 观测裁剪范围：协议字段理论上有限，但 NaN/异常值不应该进网络。
OBS_CLIP = 5.0

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
) -> np.ndarray:
    """把一条 OBSERVATION 编码成固定长度的 float32 向量。

    ``opponent_health`` 由环境从对手自己的观测里取（协议实体列表不带血量）。
    ``opponents`` 见 :func:`nearest_player`。
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

    if target is not None:
        tpos = target.get("pos") or {}
        dx = _f(tpos.get("x")) - sx
        dy = _f(tpos.get("y")) - sy
        dz = _f(tpos.get("z")) - sz
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
        if entity_id is None or distance > reach:
            # 够不到时先靠近：攻击距离由 BotProfile.reach 决定（默认原版 3.99），硬打只会空挥。
            return [look, {"type": "move", "params": {
                "direction": "forward", "durationMs": MOVE_MS, "sprint": distance > 2.5}}]
        # 近身：先转向目标、挥击、再补一次朝向，减少击退后的背对空挥。
        return [look, {"type": "attack", "params": {"targetEntityId": int(entity_id)}}, look]

    raise AssertionError(f"未处理的动作: {name}")


def describe_actions() -> Iterable[str]:
    return ACTION_NAMES
