"""固定、可复现、带置信区间的评估套件（v2）。

为什么需要它：旧的 12 局评估波动就有 ±5 局，两个策略的差别完全被噪声吃掉；
同一模型换个 seed 能差 2~5 局，导致任何改动都不可判定。这里把评估变成契约：

1. **固定 seed 集**：``seed = config.seed + episodeIndex``，两个策略跑同一批 seed，
   配对比较才有意义（同一局的地图抖动/对手参数完全一致）；
2. **胜率给区间**：Wilson 95% 区间（小样本比正态近似可靠），而不是一个孤零零的百分比；
3. **配对检验**：McNemar 精确二项检验，只看「A 赢 B 输 / A 输 B 赢」的分歧局；
4. **按对手分层**：scripted / rusher / aimbot 的强弱差异很大，混在一起会互相抵消。

本模块只依赖标准库 + numpy，不 import gymnasium，也不改 ``gladiator_env`` 的任何语义。
``training/scripts/eval_policy.py`` 与 ``eval_suite.py`` 共用 :func:`reset_with_retry` 与
:func:`rollout`，避免两套「reset 重试 / 只用返回值」的坑再踩一遍。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

#: 评估套件默认配置（8 局 × 3 对手 = 24 局，正好是「可以下结论」的最低样本量）。
SUITE_DEFAULT = {
    "opponents": ("scripted", "rusher", "aimbot"),
    "episodes": 8,
    "seed": 1000,
    "maxSteps": 400,
}

#: 判定阈值：p < 0.05 才说「显著更好」。
SIGNIFICANCE = 0.05


# ---------------------------------------------------------------------------
# 统一口径的统计工具
# ---------------------------------------------------------------------------

def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple:
    """胜率的 Wilson 得分区间（默认 95%）。

    ``n=0`` 返回 ``(0.0, 1.0)``（没有任何信息，不假装 0% 或 100%）。
    小样本（n<30）比 ``p ± z·sqrt(p(1-p)/n)`` 更稳：全胜时上界仍是 1，
    但下界不会塌到 0。
    """
    n = int(n)
    wins = int(wins)
    if n <= 0:
        return (0.0, 1.0)
    wins = max(0, min(wins, n))
    p = wins / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact(b: int, c: int) -> float:
    """配对比较的**精确** McNemar 双侧 p 值。

    ``b`` =「A 赢、B 输」的局数，``c`` =「A 输、B 赢」的局数（两边都赢/都输的不参与）。
    在 H0（两者等价）下 ``b ~ Binomial(b+c, 0.5)``，双侧 p = 2·P(X ≤ min(b,c))。
    用 ``math.comb`` 精确求和，不用卡方近似——评估样本通常只有十几局，近似不可靠。
    """
    b = max(0, int(b))
    c = max(0, int(c))
    if b + c == 0:
        return 1.0
    n = b + c
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / float(2 ** n)
    return max(0.0, min(1.0, 2.0 * tail))


def _outcome_of(row: dict) -> str:
    """一局的胜负口径：优先 ``outcome``，其次 ``win``（兼容只给布尔的调用方）。"""
    outcome = row.get("outcome")
    if outcome in ("win", "loss", "draw"):
        return str(outcome)
    return "win" if row.get("win") else "loss"


def _damage_ratio(dealt: float, taken: float) -> float:
    """输出/承受伤害比；承受为 0 时退化为「输出值」，避免除零与 Infinity。"""
    if taken > 0:
        return float(dealt) / float(taken)
    return float(dealt) if dealt > 0 else 0.0


def summarize(rows: Sequence[dict]) -> dict:
    """汇总逐局 rows：按对手分组 + 总体。

    每组的字段固定为
    ``episodes/wins/losses/draws/winRate/winRateCI/meanReward/meanSteps/damageRatio``
    （另带 ``damageDealt``/``damageTaken`` 便于看伤害质量）。
    """
    rows = list(rows or [])
    by_opponent: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get("opponent") or "unknown")
        by_opponent.setdefault(key, []).append(row)
    return {
        "overall": _summarize_group(rows),
        "opponents": {name: _summarize_group(items) for name, items in by_opponent.items()},
    }


def _summarize_group(rows: Sequence[dict]) -> dict:
    episodes = len(rows)
    wins = sum(1 for row in rows if _outcome_of(row) == "win")
    losses = sum(1 for row in rows if _outcome_of(row) == "loss")
    draws = episodes - wins - losses
    rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
    steps = [float(row.get("steps", 0) or 0) for row in rows]
    dealt = sum(float(row.get("damageDealt", 0.0) or 0.0) for row in rows)
    taken = sum(float(row.get("damageTaken", 0.0) or 0.0) for row in rows)
    low, high = wilson_interval(wins, episodes)
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": (wins / episodes) if episodes else 0.0,
        "winRateCI": (low, high),
        "meanReward": (sum(rewards) / episodes) if episodes else 0.0,
        "meanSteps": (sum(steps) / episodes) if episodes else 0.0,
        "damageDealt": dealt,
        "damageTaken": taken,
        "damageRatio": _damage_ratio(dealt, taken),
    }


def _row_key(row: dict) -> tuple:
    try:
        return (str(row.get("opponent") or "unknown"),
                int(row["episodeIndex"]), int(row["seed"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"row 缺少配对键 (opponent, episodeIndex, seed): {row!r}"
        ) from exc


def pair_rows(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> dict:
    """按 ``(opponent, episodeIndex, seed)`` 对齐两个策略的逐局结果。

    返回 ``{"pairs", "b", "c", "p", "byOpponent"}``：``b`` 是「A 赢 B 输」、
    ``c`` 是「A 输 B 赢」，``p`` 为精确 McNemar。两个策略必须跑完全相同的 key 集合，
    否则抛 ``ValueError``（顺序无关：内部用字典索引，乱序也能配对）。
    """
    index_a: dict[tuple, dict] = {}
    for row in rows_a or []:
        key = _row_key(row)
        if key in index_a:
            raise ValueError(f"A 的结果里有重复的配对键: {key}")
        index_a[key] = row
    index_b: dict[tuple, dict] = {}
    for row in rows_b or []:
        key = _row_key(row)
        if key in index_b:
            raise ValueError(f"B 的结果里有重复的配对键: {key}")
        index_b[key] = row

    keys_a = set(index_a)
    keys_b = set(index_b)
    if keys_a != keys_b:
        missing_b = sorted(keys_a - keys_b)
        missing_a = sorted(keys_b - keys_a)
        raise ValueError(
            "两个策略的 seed 集不一致，无法配对："
            f"A 多出 {len(missing_b)} 局 {missing_b[:3]}，B 多出 {len(missing_a)} 局 {missing_a[:3]}"
        )

    b_count = 0
    c_count = 0
    both_win = 0
    both_lose = 0
    per_opponent: dict[str, dict] = {}
    for key in sorted(index_a):
        opponent, _episode, _seed = key
        a_win = _outcome_of(index_a[key]) == "win"
        b_win = _outcome_of(index_b[key]) == "win"
        bucket = per_opponent.setdefault(opponent, {"b": 0, "c": 0, "pairs": 0})
        bucket["pairs"] += 1
        if a_win and not b_win:
            b_count += 1
            bucket["b"] += 1
        elif b_win and not a_win:
            c_count += 1
            bucket["c"] += 1
        elif a_win and b_win:
            both_win += 1
        else:
            both_lose += 1

    for bucket in per_opponent.values():
        bucket["p"] = mcnemar_exact(bucket["b"], bucket["c"])
    return {
        "pairs": len(index_a),
        "b": b_count,
        "c": c_count,
        "p": mcnemar_exact(b_count, c_count),
        "bothWin": both_win,
        "bothLose": both_lose,
        "byOpponent": per_opponent,
    }


def verdict(comparison: dict, label_a: str = "A", label_b: str = "B") -> str:
    """把配对结果翻译成一句人话；p ≥ 0.05 一律「不显著」。"""
    p = float(comparison.get("p", 1.0))
    b = int(comparison.get("b", 0))
    c = int(comparison.get("c", 0))
    if p >= SIGNIFICANCE or b == c:
        return f"{label_a} 与 {label_b} 差异不显著（p={p:.3f} ≥ {SIGNIFICANCE}）"
    better = label_a if b > c else label_b
    return f"{better} 显著更好（b={b}, c={c}, p={p:.4f} < {SIGNIFICANCE}）"


# ---------------------------------------------------------------------------
# 配置与共用执行逻辑
# ---------------------------------------------------------------------------

@dataclass
class SuiteConfig:
    """一次评估套件的配置；字段名与 ``SUITE_DEFAULT`` 一一对应（``max_steps`` ↔ maxSteps）。"""

    opponents: tuple = SUITE_DEFAULT["opponents"]
    episodes: int = SUITE_DEFAULT["episodes"]
    seed: int = SUITE_DEFAULT["seed"]
    max_steps: int = SUITE_DEFAULT["maxSteps"]
    #: 贪心评估（默认）：同一 seed 的结果完全可复现；``False`` = 按策略采样。
    greedy: bool = True
    #: reset 超时重试（Bot 还没就绪时最常见）。
    reset_attempts: int = 4
    reset_delay: float = 3.0

    @property
    def maxSteps(self) -> int:  # noqa: N802 - 与 SUITE_DEFAULT 的键保持一致
        return self.max_steps

    @property
    def total_episodes(self) -> int:
        return len(self.opponents) * int(self.episodes)

    def to_dict(self) -> dict:
        return {
            "opponents": list(self.opponents),
            "episodes": int(self.episodes),
            "seed": int(self.seed),
            "maxSteps": int(self.max_steps),
            "greedy": bool(self.greedy),
            "resetAttempts": int(self.reset_attempts),
            "resetDelaySeconds": float(self.reset_delay),
            "totalEpisodes": self.total_episodes,
        }


def reset_with_retry(env, seed: int, *, attempts: int = 4, delay: float = 3.0,
                     sleeper: Callable[[float], None] = time.sleep,
                     logger: Optional[Callable[[str], None]] = None):
    """``env.reset`` 超时重试（同一个坑：Bot 未就绪时 reset 会 TimeoutError）。

    重试时 seed 用 ``seed + attempt``：不能反复用一个已经被消耗的 seed，
    否则同一局的地图抖动会重复，评估结果不可复现。
    """
    attempts = max(1, int(attempts))
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return env.reset(seed=int(seed) + attempt)
        except TimeoutError as exc:  # 只有「还没就绪」是可重试的
            last_error = exc
            if logger is not None:
                logger(f"[eval] reset 第 {attempt + 1}/{attempts} 次超时，{delay:g}s 后重试")
            if attempt + 1 < attempts:
                sleeper(delay)
    raise RuntimeError(f"Bot 始终未就绪：{last_error}") from last_error


def _default_username(env) -> Optional[str]:
    bots = getattr(env, "bots", None)
    if bots:
        return str(getattr(bots[0], "username", "") or "") or None
    return None


def _events_damage(env, username: str) -> Optional[tuple]:
    """优先用塑形事件里的 hit/taken 伤害；env 没有 ``last_events`` 时返回 None。"""
    getter = getattr(env, "last_events", None)
    if not callable(getter):
        return None
    try:
        events = getter() or {}
    except Exception:  # noqa: BLE001 - 评估不该因为诊断接口失败而中断
        return None
    mine = events.get(username) or []
    if not isinstance(mine, (list, tuple)):
        return None
    dealt = 0.0
    taken = 0.0
    for event in mine:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        try:
            damage = float(event.get("damage", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if kind == "hit":
            dealt += damage
        elif kind == "taken":
            taken += damage
    return dealt, taken


def rollout(env, policy, obs, *, max_steps: int, rng=None, greedy: bool = True,
            username: Optional[str] = None) -> dict:
    """跑完一局，返回 ``steps/reward/terminated/truncated/winner/damageDealt/damageTaken``。

    只用 ``reset``/``step`` 返回的 obs（不读 env 内部状态），与 ``eval_policy.py`` 同口径。
    伤害优先取塑形事件；env 没暴露事件时退化为双方血量的下降量（初始按满血 20 计）。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    username = username or _default_username(env)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    winner = None
    event_dealt = 0.0
    event_taken = 0.0
    event_samples = 0
    health_dealt = 0.0
    health_taken = 0.0
    health_prev: dict[str, float] = {}
    max_steps = max(1, int(max_steps))

    while steps < max_steps:
        action = policy.act(obs, rng, greedy=greedy)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        info = info or {}
        winner = info.get("winner", winner)

        if username:
            sample = _events_damage(env, username)
            if sample is not None:
                event_dealt += sample[0]
                event_taken += sample[1]
                event_samples += 1

        health = info.get("health") or {}
        if isinstance(health, dict) and health:
            for name, value in health.items():
                try:
                    current = float(value)
                except (TypeError, ValueError):
                    continue
                previous = health_prev.get(name)
                if previous is not None and previous > current:
                    drop = previous - current
                    if username and name == username:
                        health_taken += drop
                    elif username:
                        health_dealt += drop
                health_prev[name] = current

        if terminated or truncated:
            break

    if event_samples > 0:
        dealt, taken = event_dealt, event_taken
    else:
        dealt, taken = health_dealt, health_taken
    return {
        "steps": steps,
        "reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "winner": winner,
        "damageDealt": dealt,
        "damageTaken": taken,
    }


def _outcome(winner: Optional[str], username: Optional[str]) -> str:
    """赢家名字 → win/loss/draw；没有赢家（超时/平局）算 draw。"""
    if not winner:
        return "draw"
    if username and str(winner) == str(username):
        return "win"
    return "loss"


def run_suite(env_factory: Callable[[str], Any], policy, config: SuiteConfig,
              *, username: Optional[str] = None,
              sleeper: Callable[[float], None] = time.sleep,
              logger: Optional[Callable[[str], None]] = None) -> list:
    """按 ``config`` 跑完整个套件，返回逐局 rows。

    ``env_factory(opponent_mode)`` 每个对手创建一个 env（连接可以复用，但
    opponent_mode 在 env 构造时校验，重新创建最省心）；同一对手的 ``episodes`` 局
    共用这个 env，``seed = config.seed + episodeIndex`` 保证跨策略配对。
    """
    rows: list = []
    modes = tuple(config.opponents)
    for mode in modes:
        env = env_factory(mode)
        try:
            if hasattr(env, "opponent_mode"):
                env.opponent_mode = str(mode)
            name = username or _default_username(env)
            for index in range(int(config.episodes)):
                seed = int(config.seed) + index
                obs, _info = reset_with_retry(
                    env, seed,
                    attempts=config.reset_attempts,
                    delay=config.reset_delay,
                    sleeper=sleeper,
                    logger=logger,
                )
                rng = np.random.default_rng(seed)
                result = rollout(env, policy, obs, max_steps=config.max_steps,
                                 rng=rng, greedy=config.greedy, username=name)
                outcome = _outcome(result.get("winner"), name)
                rows.append({
                    "opponent": str(mode),
                    "episodeIndex": index,
                    "seed": seed,
                    "outcome": outcome,
                    "win": outcome == "win",
                    "reward": float(result["reward"]),
                    "steps": int(result["steps"]),
                    "damageDealt": float(result["damageDealt"]),
                    "damageTaken": float(result["damageTaken"]),
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "winner": result.get("winner"),
                })
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响已收集的结果
                    pass
    return rows


def format_interval(interval: Iterable[float]) -> str:
    """``(lo, hi)`` → ``"12.5%–87.5%"``（打印用，不参与统计）。"""
    low, high = tuple(interval)
    return f"{low * 100:.1f}%–{high * 100:.1f}%"
