"""训练循环：在 GladiatorEnv 上跑 REINFORCE。

约定：环境只提供观测/动作/奖励；这里负责采样、折扣回报、优势归一化与参数更新。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from gladiator_env.review import EpisodeTracker, write_review

from .policy import Policy, discounted_returns, normalize_advantages


@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    winner: Optional[str]
    duration_s: float
    started_at: str = ""
    started_epoch: float = 0.0
    mean_health: float = 0.0
    review: Optional[Dict[str, Any]] = None
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "totalReward": round(self.total_reward, 4),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "winner": self.winner,
            "durationSeconds": round(self.duration_s, 2),
            "startedAt": self.started_at,
            "startedAtEpoch": round(self.started_epoch, 3),
            "meanHealth": round(self.mean_health, 2),
            **({"review": self.review} if self.review else {}),
        }


class Trainer:
    """把 Gymnasium 环境的采样与策略更新串起来。"""

    def __init__(
        self,
        env: Any,
        policy: Policy,
        gamma: float = 0.99,
        seed: int | None = None,
        log: Callable[[str], None] = print,
        review_dir: str | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.log = log
        self.review_dir = review_dir
        self.history: List[Dict[str, Any]] = []
        self._episode_no = 0

    # ------------------------------------------------------------- one episode
    def run_episode(self, greedy: bool = False) -> EpisodeResult:
        started = time.time()
        started_wall = time.time()
        obs, _info = self.env.reset()
        result = EpisodeResult(0, 0.0, False, False, None, 0.0)
        healths: List[float] = []
        self._episode_no += 1
        username = self.env.bots[0].username
        opponent = self.env.bots[1].username if len(self.env.bots) > 1 else None
        tracker = EpisodeTracker(episode=self._episode_no, username=username, opponent=opponent)

        while True:
            previous_raw = self.env.raw_observations
            action = self.policy.act(obs, self.rng, greedy=greedy)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            current_raw = self.env.raw_observations
            tracker.update(
                action_index=int(action),
                previous=previous_raw[0] if previous_raw else None,
                current=current_raw[0] if current_raw else None,
                opponent_previous=previous_raw[1] if len(previous_raw) > 1 else None,
                opponent_current=current_raw[1] if len(current_raw) > 1 else None,
                reward=float(reward),
                events=self.env.last_events().get(username, []),
            )

            result.observations.append(np.asarray(obs, dtype=np.float64))
            result.actions.append(int(action))
            result.rewards.append(float(reward))
            result.total_reward += float(reward)
            result.steps += 1
            health = info.get("health", {})
            if health:
                healths.append(float(list(health.values())[0]))

            obs = next_obs
            if terminated or truncated:
                result.terminated = bool(terminated)
                result.truncated = bool(truncated)
                result.winner = info.get("winner")
                break

        result.duration_s = time.time() - started
        result.started_epoch = started_wall
        result.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_wall))
        result.mean_health = float(np.mean(healths)) if healths else 0.0
        breakdown = {}
        summary_fn = getattr(self.env, "shaping", None)
        if summary_fn is not None:
            breakdown = summary_fn.summary(username).get("totals", {})
        review = tracker.finish(
            winner=result.winner,
            terminated=result.terminated,
            truncated=result.truncated,
            reward_breakdown=breakdown,
        )
        if self.review_dir:
            path = write_review(self.review_dir, review)
            self.log(f"[review] 已写入 {path}")
        result.review = review.as_dict()
        return result

    # --------------------------------------------------------------- training
    def train(self, episodes: int, on_episode: Optional[Callable[[int, EpisodeResult], None]] = None) -> List[Dict[str, Any]]:
        started = time.time()
        start_index = len(self.history)
        for i in range(1, int(episodes) + 1):
            result = self.run_episode(greedy=False)
            returns = discounted_returns(result.rewards, self.gamma)
            advantages = normalize_advantages(returns)
            self.policy.update(result.observations, result.actions, advantages)

            record = {
                "episode": i,
                **result.summary(),
                "policyUpdates": self.policy.updates,
                "elapsedSeconds": round(time.time() - started, 1),
            }
            self.history.append(record)
            self.log(
                f"[train] episode {i}/{episodes} steps={result.steps} "
                f"reward={result.total_reward:.3f} winner={result.winner} "
                f"terminated={result.terminated} {result.duration_s:.1f}s"
            )
            if on_episode is not None:
                on_episode(i, result)
        return self.history[start_index:]

    # -------------------------------------------------------------- evaluation
    def evaluate(self, episodes: int = 1) -> List[Dict[str, Any]]:
        out = []
        for i in range(1, int(episodes) + 1):
            result = self.run_episode(greedy=True)
            record = {"episode": i, **result.summary()}
            out.append(record)
            self.log(
                f"[eval] episode {i}/{episodes} steps={result.steps} "
                f"reward={result.total_reward:.3f} winner={result.winner}"
            )
        return out


def save_report(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
