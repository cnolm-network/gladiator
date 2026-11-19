"""每局复盘（G-509）。

训练循环每结束一个 episode 就生成一份结构化复盘 + Markdown：

- 结果（winner / terminated / truncated / 时长）
- 伤害交换（造成/承受/有效伤害差）
- 命中与空挥（命中率）
- 连击峰值、平均距离
- 动作分布（7 个离散动作各自的次数）
- 奖励分解（来自塑形统计）

只做统计，不做策略判断；数据来自观测与动作流，不需要额外查询服务端。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spaces import ACTION_NAMES


def _health(observation: dict[str, Any] | None) -> float | None:
    if not observation:
        return None
    value = (observation.get("self") or {}).get("health")
    return float(value) if isinstance(value, (int, float)) else None


def _pos(observation: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not observation:
        return None
    pos = (observation.get("self") or {}).get("pos") or {}
    try:
        return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def distance_between(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    pa, pb = _pos(a), _pos(b)
    if pa is None or pb is None:
        return None
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2) ** 0.5


@dataclass
class EpisodeReview:
    episode: int
    username: str
    steps: int
    duration_s: float
    total_reward: float
    winner: str | None
    terminated: bool
    truncated: bool
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    hits: int = 0
    #: 命中里「本 step 真的下发了 attack」的次数；
    #: ``environment_hits`` 是「对手掉血但本 step 没挥刀」——击退摔落/环境伤害等。
    #: 二者相加必须等于 ``hits``（见 EpisodeTracker.update）。
    attack_hits: int = 0
    environment_hits: int = 0
    whiffs: int = 0
    max_combo: int = 0
    attack_actions: int = 0
    distance_sum: float = 0.0
    distance_samples: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        attempts = self.hits + self.whiffs
        return 0.0 if attempts == 0 else self.hits / attempts

    @property
    def mean_distance(self) -> float | None:
        return None if self.distance_samples == 0 else self.distance_sum / self.distance_samples

    @property
    def damage_ratio(self) -> float | None:
        if self.damage_taken <= 0:
            return None
        return self.damage_dealt / self.damage_taken

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "username": self.username,
            "steps": self.steps,
            "durationSeconds": round(self.duration_s, 2),
            "totalReward": round(self.total_reward, 4),
            "winner": self.winner,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "damageDealt": round(self.damage_dealt, 3),
            "damageTaken": round(self.damage_taken, 3),
            "damageRatio": None if self.damage_ratio is None else round(self.damage_ratio, 3),
            "hits": self.hits,
            "attackHits": self.attack_hits,
            "environmentHits": self.environment_hits,
            "whiffs": self.whiffs,
            "hitRate": round(self.hit_rate, 3),
            "maxCombo": self.max_combo,
            "attackActions": self.attack_actions,
            "meanDistance": None if self.mean_distance is None else round(self.mean_distance, 3),
            "actionCounts": dict(self.action_counts),
            "rewardBreakdown": {k: round(v, 4) for k, v in sorted(self.reward_breakdown.items())},
            "events": self.events[:40],
        }


class EpisodeTracker:
    """在一个 episode 内累计统计；每个 step 调用一次 ``update``。"""

    def __init__(self, *, episode: int, username: str, opponent: str | None = None) -> None:
        self.review = EpisodeReview(
            episode=episode,
            username=username,
            steps=0,
            duration_s=0.0,
            total_reward=0.0,
            winner=None,
            terminated=False,
            truncated=False,
            action_counts={name: 0 for name in ACTION_NAMES},
        )
        self.opponent = opponent
        self._started = time.monotonic()
        self._last_hit_step = -1
        self._combo = 0
        self._opponent_alive = True

    def update(
        self,
        *,
        action_index: int,
        previous: dict[str, Any] | None,
        current: dict[str, Any] | None,
        opponent_previous: dict[str, Any] | None,
        opponent_current: dict[str, Any] | None,
        reward: float,
        events: list[dict[str, Any]] | None = None,
        attacked: bool | None = None,
        action_label: str | None = None,
    ) -> None:
        """累计一步统计。

        ``attacked`` 表示本 step **实际下发**了 attack（由环境给出）；不传时按
        「策略选的动作编号是不是 attack」推断（旧调用方兼容）。
        这样 ``hits`` 能拆成「真挥刀命中」与「非攻击来源掉血」，
        避免出现「attack=0 却有 6 点伤害」这种对不上的读数。

        ``action_label`` 给 v2（多离散头）用：动作不是单个下标，调用方传一个粗粒度
        标签（如 ``v2:attack``），``action_counts`` 才有可读的键。
        """
        r = self.review
        r.steps += 1
        r.total_reward += float(reward)
        # 下标与动作名同源：都来自 spaces.ACTION_NAMES（策略输出的就是它的下标）。
        if action_label is not None:
            name = str(action_label)
        else:
            name = (ACTION_NAMES[action_index] if 0 <= action_index < len(ACTION_NAMES)
                    else str(action_index))
        r.action_counts[name] = r.action_counts.get(name, 0) + 1
        if name == "attack":
            r.attack_actions += 1
        did_attack = (name == "attack") if attacked is None else bool(attacked)

        own_before, own_after = _health(previous), _health(current)
        opp_before, opp_after = _health(opponent_previous), _health(opponent_current)
        if own_before is not None and own_after is not None and own_after < own_before:
            r.damage_taken += own_before - own_after
        prior_opp = _health(opponent_previous)
        if opp_before is not None and opp_after is not None and opp_after < opp_before:
            r.damage_dealt += opp_before - opp_after
            r.hits += 1
            if did_attack:
                r.attack_hits += 1
            else:
                # 没挥刀却掉血：击退摔落、环境伤害等。单独计数，并在复盘里写明。
                r.environment_hits += 1
            if self._last_hit_step >= 0 and r.steps - self._last_hit_step <= 15:
                self._combo += 1
            else:
                self._combo = 1
            self._last_hit_step = r.steps
            r.max_combo = max(r.max_combo, self._combo)
            r.events.append({"step": r.steps, "type": "hit", "damage": round(opp_before - opp_after, 2)})
        elif name == "attack" and prior_opp is not None:
            r.whiffs += 1

        dist = distance_between(current, opponent_current)
        if dist is not None:
            r.distance_sum += dist
            r.distance_samples += 1

        for event in events or []:
            if event.get("type") in ("kill", "death", "combo"):
                r.events.append({"step": r.steps, **event})
        if len(r.events) > 200:
            del r.events[:-200]

    def finish(
        self,
        *,
        winner: str | None,
        terminated: bool,
        truncated: bool,
        reward_breakdown: dict[str, float] | None = None,
    ) -> EpisodeReview:
        r = self.review
        r.winner = winner
        r.terminated = bool(terminated)
        r.truncated = bool(truncated)
        r.duration_s = time.monotonic() - self._started
        r.reward_breakdown = dict(reward_breakdown or {})
        return r


def to_markdown(review: EpisodeReview) -> str:
    r = review
    lines = [
        f"# Episode {r.episode} · {r.username}",
        "",
        f"- 结果：{'胜 ' + r.winner if r.winner else '平局/未分胜负'}"
        f"（terminated={r.terminated}, truncated={r.truncated}）",
        f"- 步数 / 时长：{r.steps} 步 / {r.duration_s:.1f}s",
        f"- 总奖励：{r.total_reward:+.3f}",
        "",
        "## 伤害交换",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 造成伤害 | {r.damage_dealt:.1f} |",
        f"| 承受伤害 | {r.damage_taken:.1f} |",
        f"| 伤害比 | {'—' if r.damage_ratio is None else f'{r.damage_ratio:.2f}'} |",
        f"| 命中 / 空挥 | {r.hits} / {r.whiffs} |",
        f"| 其中：挥刀命中 / 非攻击掉血 | {r.attack_hits} / {r.environment_hits} |",
        f"| 命中率 | {r.hit_rate * 100:.0f}% |",
        f"| 连击峰值 | {r.max_combo} |",
        f"| 平均距离 | {'—' if r.mean_distance is None else f'{r.mean_distance:.2f} 格'} |",
        "",
        "## 动作分布",
        "",
        "| 动作 | 次数 |",
        "|---|---|",
    ]
    for name, count in r.action_counts.items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    if r.reward_breakdown:
        lines += ["## 奖励分解", "", "| 项 | 值 |", "|---|---|"]
        for key, value in r.reward_breakdown.items():
            lines.append(f"| {key} | {value:+.3f} |")
        lines.append("")
    if r.events:
        lines += ["## 关键事件", ""]
        for event in r.events[-15:]:
            step = event.get("step", "?")
            etype = event.get("type", "?")
            detail = ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("step", "type"))
            lines.append(f"- step {step}: {etype}" + (f"（{detail}）" if detail else ""))
        lines.append("")
    return "\n".join(lines)


def write_review(directory: str | Path, review: EpisodeReview) -> Path:
    """写 ``episode-XXX.md``（Markdown）与 ``episode-XXX.json``（结构化）。"""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"episode-{review.episode:03d}"
    base.with_suffix(".md").write_text(to_markdown(review), encoding="utf-8", newline="\n")
    base.with_suffix(".json").write_text(
        json.dumps(review.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return base.with_suffix(".md")
