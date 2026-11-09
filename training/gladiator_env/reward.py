"""奖励接口（G-404）。

默认奖励只使用物理量：掉血、击杀、存活步数。它**不判断对局规则**（谁该赢、是否允许重生
属 Layer 1 / Mode），因此同一份奖励可以用于混战、死斗或带规则的比赛。
上层可以用 `reward_fn=` 注入自定义奖励（例如带时间惩罚的教练奖励）。

注意：`OBSERVATION.entities` 不携带对手血量（协议如此），所以「造成伤害」由环境从
**对手自身的 OBSERVATION.self.health** 计算，而不是从实体列表里读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RewardContext:
    """一次 `step()` 内单个 Bot 的奖励输入。"""

    username: str
    index: int
    previous: dict
    current: dict
    step: int
    terminated: bool
    truncated: bool
    opponent_previous: dict | None = None
    opponent_current: dict | None = None


def self_health(observation: dict) -> float:
    payload = observation.get("self") or {}
    try:
        return float(payload.get("health", 20.0))
    except (TypeError, ValueError):
        return 20.0


def self_state(observation: dict) -> str:
    return str(observation.get("state", "Alive"))


def is_out(observation: dict) -> bool:
    """Bot 是否已经退出本回合（死亡/淘汰）。"""
    return self_health(observation) <= 0.0 or self_state(observation) in ("Dead", "Eliminated")


def default_reward(ctx: RewardContext, *,
                   damage_scale: float = 0.1,
                   kill_bonus: float = 1.0,
                   death_penalty: float = 1.0,
                   step_penalty: float = 0.001) -> float:
    """掉血为主、击杀/死亡为首末奖励的通用奖励。"""
    damage_taken = max(0.0, self_health(ctx.previous) - self_health(ctx.current))
    damage_dealt = 0.0
    if ctx.opponent_previous is not None and ctx.opponent_current is not None:
        damage_dealt = max(0.0, self_health(ctx.opponent_previous) - self_health(ctx.opponent_current))

    reward = damage_scale * damage_dealt - damage_scale * damage_taken - step_penalty

    if ctx.terminated:
        if is_out(ctx.current):
            reward -= death_penalty
        elif ctx.opponent_current is not None and is_out(ctx.opponent_current):
            reward += kill_bonus
    return float(reward)


RewardFn = Callable[[RewardContext], float]
