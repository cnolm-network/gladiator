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
    #: 胜利/失败：P2 起这是**唯一的大信号**（±50），kill/death 只作小额补充，
    #: 否则「打到人」和「赢下这局」在数值上混在一起，策略会满足于刷伤害。
    victory: float = 50.0
    kill: float = 2.0
    death: float = -2.0
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
    #: 「站进射程内真的挥刀」的一次性奖励（P1.5）。与 attack_attempt 的区别：
    #: 这一项用观测里的 ``in_reach``（水平距离 ≤ reach）判定，专治「贴身却不挥剑」
    #: ——实测策略塌成 approach×120 / attack×0，只有命中才给正奖励，「不挥」成了局部最优。
    attack_in_reach: float = 0.05
    #: 离得远还格挡/挥鱼竿的“无意义动作”惩罚，专治只摆架势不打人。
    waste: float = -0.08
    #: 远距离主动缩短距离的奖励（每 step），专治站着不动/来回踱步。
    closing: float = 0.02
    #: 挨打后 0.6s 内反手还击的“不白挨”奖励，教模型对拼而不是一味格挡。
    counter: float = 0.15
    still: float = -0.1
    #: 兼容旧名：0 关闭「无交惩罚」，语义已改为按位移判定。
    inactive: float = 0.0
    # --- PBRS（势能塑形，policy-invariant） ---
    #: 血量差势能权重。
    pbrs_alpha: float = 0.5
    #: 距离势能权重。
    pbrs_beta: float = 0.5
    #: 势能折扣：**默认 1.0（非折扣）**。回合动辄 400 步，γ<1 会让后半程的势能差分
    #: 衰减到无感，PBRS 就白加了。
    pbrs_gamma: float = 1.0
    #: dense 奖励每步封顶（除 victory 外）。victory 之外的大额奖励会让策略只刷过程分。
    shaping_cap: float = 5.0

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
            "attackInReachPerAttack": self.attack_in_reach,
            "waste": self.waste,
            "closing": self.closing,
            "counter": self.counter,
            "stillPerSecond": self.still,
            "inactivePerSecond": self.inactive,
            "victory": self.victory,
            "pbrsAlpha": self.pbrs_alpha,
            "pbrsBeta": self.pbrs_beta,
            "pbrsGamma": self.pbrs_gamma,
            "shapingCap": self.shaping_cap,
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
    #: 上一步的势能 Φ(s)；None 表示这一步是回合首步（Φ(s) 取 reset 时的状态）。
    last_phi: float | None = None
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


def _horizontal_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    """两个观测之间的水平距离（PBRS 的距离势能只看水平面）。"""
    pa, pb = _pos(a), _pos(b)
    if pa is None or pb is None:
        return None
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


class ShapingReward:
    """有状态的全方位奖励。

    用法：``reward = shaping.compute(...)``；每回合开始调 ``reset(username)``。
    ``attacked`` 表示本 step 是否下发了 attack（用于空挥判定）。
    """

    def __init__(self, weights: ShapingWeights | None = None, terms: list | None = None,
                 use_pbrs: bool = True) -> None:
        self.weights = weights or ShapingWeights()
        #: PBRS 开关（旧行为可通过 ``use_pbrs=False`` 完全关掉）。
        self.use_pbrs = bool(use_pbrs)
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

    def potential(self, observation: dict[str, Any] | None,
                  opponent_observation: dict[str, Any] | None) -> float:
        """势能 Φ(s) = α·phi_hp + β·phi_dist（只依赖状态，与动作无关）。

        - ``phi_hp = (20-opp_hp)/20 - (20-self_hp)/20``，裁剪到 [-1, 1]；
        - ``phi_dist = 1 - min(水平距离, 8)/8``（无目标时 0）。
        """
        w = self.weights
        own = _health(observation)
        opp = _health(opponent_observation)
        if own is None or opp is None:
            return 0.0
        alpha = float(w.pbrs_alpha)
        beta = float(w.pbrs_beta)
        phi_hp = (20.0 - opp) / 20.0 - (20.0 - own) / 20.0
        phi_hp = max(-1.0, min(1.0, phi_hp))
        distance = _horizontal_distance(observation, opponent_observation)
        phi_dist = 0.0 if distance is None else 1.0 - min(float(distance), 8.0) / 8.0
        return alpha * phi_hp + beta * phi_dist

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
        in_reach: bool | None = None,
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

        # --- PBRS：Φ(s) 只依赖状态（血量差 + 水平距离），与动作无关 -----------
        phi_next = self.potential(current, opponent_current)
        if st.last_phi is None:
            # 回合首步：Φ(s) 取 reset 时的状态（也就是 previous）。
            st.last_phi = self.potential(previous, opponent_previous)
        pbrs = 0.0
        if self.use_pbrs:
            pbrs = float(w.pbrs_gamma) * phi_next - float(st.last_phi)
        st.last_phi = phi_next

        own_before = _health(previous)
        own_after = _health(current)
        opp_before = _health(opponent_previous)
        opp_after = _health(opponent_current)

        if self.use_pbrs and pbrs:
            reward += pbrs
            st.add("pbrs", pbrs)
            events.append({"type": "pbrs", "value": round(pbrs, 4)})

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
            # P1.5：水平距离已在射程内还真的挥刀 —— 每次攻击结算一次，不是每 step 一次。
            if in_reach:
                reward += w.attack_in_reach
                st.add("attackInReach", w.attack_in_reach)
                events.append({"type": "attackInReach"})

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
        # P2：胜负是唯一大信号（±victory），kill/death 降为小额补充；
        # 二者都不参与 dense 封顶，否则「赢下这局」会被削平。
        victory_bonus = 0.0
        won = lost = False
        if terminated:
            own_state = _state(current)
            opp_state = _state(opponent_current)
            if own_state in ("Dead", "Eliminated"):
                lost = True
                reward += w.death
                st.add("death", w.death)
                events.append({"type": "death"})
            elif opp_state in ("Dead", "Eliminated"):
                won = True
                reward += w.kill
                st.add("kill", w.kill)
                events.append({"type": "kill"})
            if won:
                victory_bonus = w.victory
            elif lost:
                victory_bonus = -w.victory
            if victory_bonus:
                st.add("victory", victory_bonus)
                events.append({"type": "victory", "value": round(victory_bonus, 3)})

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
                "inReach": bool(in_reach),
                "won": won,
                "lost": lost,
            }
            for term in self.terms:
                value = term.evaluate(ctx, dt)
                if value:
                    reward += value
                    st.add(term.id, value)

        # --- dense 封顶（每步，不含 victory） ---------------------------------
        # 「打到人」的过程分可以很密（单步伤害 6 HP = +15），不封顶就会盖过 ±50 的胜负信号。
        cap = float(w.shaping_cap)
        if cap > 0.0:
            clipped = max(-cap, min(cap, reward))
            if clipped != reward:
                st.add("capped", clipped - reward)
                events.append({"type": "capped", "value": round(clipped - reward, 4)})
            reward = clipped
        reward += victory_bonus

        st.add("total", reward)
        st.add("steps", 1.0)
        return reward, events

    def summary(self, username: str) -> dict[str, Any]:
        st = self.state(username)
        return {
            "comboMax": st.max_combo,
            "totals": {k: round(v, 4) for k, v in sorted(st.totals.items())},
        }
