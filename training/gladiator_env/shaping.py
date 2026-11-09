"""全方位奖励塑形（G-508）。

按参考文档的权重以“剑击 6 点伤害”为基准：

============  ==========================================
事件          奖励
============  ==========================================
击杀          +20.0
自身死亡      -20.0
对敌伤害      +2.5 / HP
承受伤害      -1.2 / HP
连击          +0.6 × combo，单次封顶 +3.0
威胁间距      命中后 1.5s 内保持 2.8–4.2 格，+0.3 / 秒
空挥          -0.3 / 次
长时间无交互  >3s 未出手，-0.1 / 秒
============  ==========================================

约束（与文档一致）：伤害:受伤 ≈ 2.1:1；连击奖励封顶；间距奖励只发生在命中后的窗口内。
所有状态按 Bot 独立维护，便于多 Bot 同时训练。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 单次连击奖励上限。
COMBO_BONUS_CAP = 3.0
#: 命中后的“威胁间距”窗口（秒）。
THREAT_WINDOW_SECONDS = 1.5
#: 威胁间距（格）。
THREAT_MIN = 2.8
THREAT_MAX = 4.2
#: 超过多久没出手算“无交互”（秒）。
INACTIVE_AFTER_SECONDS = 3.0
#: 连续这么久没有水平位移就算「站桩」（秒）。
STILL_AFTER_SECONDS = 3.0
#: 水平位移小于这个值视为没动（格）。
STILL_EPSILON = 0.05


@dataclass
class ShapingWeights:
    kill: float = 20.0
    death: float = -20.0
    damage_dealt: float = 2.5
    damage_taken: float = -1.2
    combo: float = 0.6
    combo_cap: float = COMBO_BONUS_CAP
    threat_window: float = 0.3
    #: “压迫位”奖励：不分窗口，只要站在剑距附近就给，专治双方畏首畏尾。
    pressure: float = 0.05
    #: 空挥惩罚。原来 -0.3 太重：模型宁可格挡/挥鱼竿也不出剑，改为 -0.1。
    whiff: float = -0.1
    #: 在剑距内出手就给的“敢打”奖励（不管中没中），抵消空挥惩罚的畏缩效应。
    attack_attempt: float = 0.05
    #: 离得远还格挡/挥鱼竿的“无意义动作”惩罚，专治只摆架势不打人。
    waste: float = -0.08
    #: 远距离主动缩短距离的奖励（每 step），专治站着不动/来回踱步。
    closing: float = 0.02
    #: 挨打后 0.6s 内反手还击的“不白挨”奖励，教模型对拼而不是一味格挡。
    counter: float = 0.15
    still: float = -0.1
    #: 兼容旧名：0 关闭「无交惩罚」，语义已改为按位移判定。
    inactive: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "kill": self.kill,
            "death": self.death,
            "damageDealt": self.damage_dealt,
            "damageTaken": self.damage_taken,
            "combo": self.combo,
            "comboCap": self.combo_cap,
            "threatWindowPerSecond": self.threat_window,
            "pressurePerSecond": self.pressure,
            "whiff": self.whiff,
            "attackAttempt": self.attack_attempt,
            "waste": self.waste,
            "closing": self.closing,
            "counter": self.counter,
            "stillPerSecond": self.still,
            "inactivePerSecond": self.inactive,
        }


@dataclass
class ShapingState:
    """单个 Bot 的塑形状态（跨 step 保留）。"""

    combo: int = 0
    max_combo: int = 0
    #: 当前连击串已经拿到的连击奖励（用于按串封顶）。
    combo_points: float = 0.0
    last_hit_at: float = -1e9
    threat_until: float = -1e9
    last_attack_at: float = -1e9
    last_taken_at: float = -1e9
    last_pos: tuple[float, float] | None = None
    still_seconds: float = 0.0
    totals: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + value


def _health(observation: dict[str, Any] | None) -> float | None:
    if not observation:
        return None
    self_block = observation.get("self") or {}
    value = self_block.get("health")
    return float(value) if isinstance(value, (int, float)) else None


def _pos(observation: dict[str, Any] | None) -> tuple[float, float] | None:
    """取自身水平坐标；缺失或非法时返回 None。"""
    if not observation:
        return None
    block = (observation.get("self") or {}).get("pos") or {}
    try:
        return float(block["x"]), float(block["z"])
    except (KeyError, TypeError, ValueError):
        return None


def _state(observation: dict[str, Any] | None) -> str:
    if not observation:
        return "Alive"
    return str(observation.get("state", "Alive"))


class ShapingReward:
    """有状态的全方位奖励。

    用法：``reward = shaping.compute(...)``；每回合开始调 ``reset(username)``。
    ``attacked`` 表示本 step 是否下发了 attack（用于空挥判定）。
    """

    def __init__(self, weights: ShapingWeights | None = None, terms: list | None = None) -> None:
        self.weights = weights or ShapingWeights()
        #: 预设里的附加项（见 ``reward_presets.Term``），按条件逐项结算。
        self.terms = list(terms or [])
        self._states: dict[str, ShapingState] = {}

    def state(self, username: str) -> ShapingState:
        return self._states.setdefault(username, ShapingState())

    def reset(self, username: str | None = None) -> None:
        if username is None:
            self._states.clear()
        else:
            self._states.pop(username, None)

    def compute(
        self,
        username: str,
        *,
        previous: dict[str, Any] | None,
        current: dict[str, Any] | None,
        opponent_previous: dict[str, Any] | None,
        opponent_current: dict[str, Any] | None,
        terminated: bool,
        attacked: bool,
        action: str | None = None,
        distance: float | None = None,
        previous_distance: float | None = None,
        now: float,
        dt: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        w = self.weights
        st = self.state(username)
        events: list[dict[str, Any]] = []
        reward = 0.0

        own_before = _health(previous)
        own_after = _health(current)
        opp_before = _health(opponent_previous)
        opp_after = _health(opponent_current)

        # --- 伤害交换 ---------------------------------------------------
        dealt = 0.0
        if opp_before is not None and opp_after is not None and opp_after < opp_before:
            dealt = opp_before - opp_after
            reward += w.damage_dealt * dealt
            st.add("damageDealt", w.damage_dealt * dealt)
            continued = (now - st.last_hit_at) < 1.5
            st.combo = st.combo + 1 if continued else 1
            if not continued:
                st.combo_points = 0.0
            st.max_combo = max(st.max_combo, st.combo)
            st.last_hit_at = now
            st.threat_until = now + THREAT_WINDOW_SECONDS
            if st.combo >= 2:
                # 连击奖励按“串”封顶：一条连击最多拿满 combo_cap。
                bonus = max(0.0, min(w.combo * st.combo, w.combo_cap - st.combo_points))
                if bonus > 0:
                    st.combo_points += bonus
                    reward += bonus
                    st.add("combo", bonus)
                    events.append({"type": "combo", "value": st.combo, "bonus": round(bonus, 3)})
            events.append({"type": "hit", "damage": round(dealt, 3), "combo": st.combo})

        taken = 0.0
        if own_before is not None and own_after is not None and own_after < own_before:
            taken = own_before - own_after
            reward += w.damage_taken * taken
            st.add("damageTaken", w.damage_taken * taken)
            st.last_taken_at = now
            events.append({"type": "taken", "damage": round(taken, 3)})

        # 空挥：出手但这一 step 没有造成伤害。
        if attacked:
            st.last_attack_at = now
            if dealt <= 0.0:
                reward += w.whiff
                st.add("whiff", w.whiff)
                events.append({"type": "whiff"})
            if now - st.last_taken_at <= 0.6:
                reward += w.counter
                st.add("counter", w.counter)
                events.append({"type": "counter"})
            # 只要在剑距内敢出手就先给一点，避免“打不到就完全不练”。
            if distance is not None and distance <= THREAT_MAX:
                reward += w.attack_attempt
                st.add("attackAttempt", w.attack_attempt)

        # 远距离主动贴近：每靠近一点就给一点，避免双方都缩在远处。
        if (
            previous_distance is not None
            and distance is not None
            and previous_distance > THREAT_MAX
            and distance < previous_distance - 0.05
            and not terminated
        ):
            reward += w.closing
            st.add("closing", w.closing)

        # 远距离摆架势（格挡/挥鱼竿）没有收益，反而拖时间，给个小惩罚。
        if action in ("block", "use_rod") and distance is not None and distance > THREAT_MAX:
            reward += w.waste
            st.add("waste", w.waste)
            events.append({"type": "waste", "action": action})

        # --- 威胁间距（只在命中窗口内） ----------------------------------
        if distance is not None and now <= st.threat_until and THREAT_MIN <= distance <= THREAT_MAX:
            bonus = w.threat_window * dt
            reward += bonus
            st.add("threatWindow", bonus)

        # --- 压迫位（不分窗口）：站在能打到的距离本身就该被鼓励 -----------
        if distance is not None and THREAT_MIN <= distance <= THREAT_MAX and not terminated:
            bonus = w.pressure * dt
            reward += bonus
            st.add("pressure", bonus)

        # --- 无交互惩罚 -------------------------------------------------
        # --- 站桩惩罚（按位移判定；走动本身不扣分，只有长时间不动才扣） ----------
        pos = _pos(current)
        if pos is not None:
            if st.last_pos is not None:
                moved = ((pos[0] - st.last_pos[0]) ** 2 + (pos[1] - st.last_pos[1]) ** 2) ** 0.5
                st.still_seconds = st.still_seconds + dt if moved < STILL_EPSILON else 0.0
            st.last_pos = pos
        if st.still_seconds > STILL_AFTER_SECONDS and not terminated:
            penalty = w.still * dt
            reward += penalty
            st.add("still", penalty)
        # 旧「无交互惩罚」只在显式配置非 0 时生效（默认关闭，避免误伤走位）。
        if w.inactive:
            idle_for = now - max(st.last_attack_at, st.last_hit_at, st.last_taken_at)
            if idle_for > INACTIVE_AFTER_SECONDS and not terminated:
                penalty = w.inactive * dt
                reward += penalty
                st.add("inactive", penalty)

        # --- 终局 -------------------------------------------------------
        if terminated:
            own_state = _state(current)
            opp_state = _state(opponent_current)
            if own_state in ("Dead", "Eliminated"):
                reward += w.death
                st.add("death", w.death)
                events.append({"type": "death"})
            elif opp_state in ("Dead", "Eliminated"):
                reward += w.kill
                st.add("kill", w.kill)
                events.append({"type": "kill"})

        # --- 预设附加项（条件式，见 reward_presets.Term） --------------------
        if self.terms:
            ctx = {
                "distance": distance,
                "action": action or "",
                "health": own_after,
                "opponentHealth": opp_after,
                "stillSeconds": st.still_seconds,
                "sinceHit": now - st.last_hit_at,
                "combo": st.combo,
                "damageDealt": dealt,
                "damageTaken": taken,
                "attacked": attacked,
            }
            for term in self.terms:
                value = term.evaluate(ctx, dt)
                if value:
                    reward += value
                    st.add(term.id, value)

        st.add("total", reward)
        st.add("steps", 1.0)
        return reward, events

    def summary(self, username: str) -> dict[str, Any]:
        st = self.state(username)
        return {
            "comboMax": st.max_combo,
            "totals": {k: round(v, 4) for k, v in sorted(st.totals.items())},
        }
