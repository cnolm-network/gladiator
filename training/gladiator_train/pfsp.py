"""PFSP（Prioritized Fictitious Self-Play）对手池：按「还没打过的对手」加权采样。

为什么需要：现在 `selfplay_train.py` 是**链式**自博弈（每轮只对上一轮快照），等价于「只跟同龄人打」，
一旦某一轮学歪，后面所有轮都会针对那一版学歪的风格收敛。PFSP 的经典做法是维护历史快照池，
按 `w_i ∝ (1 - win_rate_i)^p` 采样对手——**越打不过的对手权重越高**，逼策略补齐短板而不是刷同一个对手。

本模块只做「池 + 权重 + 采样 + 在线胜率」，不依赖 env，便于单测。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_EXPONENT = 2.0
DEFAULT_EPSILON = 0.02
DEFAULT_EMA = 0.5


def pfsp_weights(win_rates: Sequence[Optional[float]], *, exponent: float = DEFAULT_EXPONENT,
                 epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """把「对每个对手的胜率」换成采样权重 `(1 - wr + eps) ** p`。

    - `win_rate` 为 `None` 表示**还没打过**：按胜率 0 处理（优先打未知对手，信息增益最大）；
    - 全部胜率都为 1 时（打谁都赢），权重退化为 `eps ** p` 的等概率，此时由调用方决定要不要扩池；
    - 返回值归一化为概率；和为 0 的退化情形返回均匀分布，避免 `np.random.choice` 抛错。
    """
    if exponent < 0:
        raise ValueError("exponent 不能为负")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon 必须在 [0, 1]")

    scores = []
    for rate in win_rates:
        value = 0.0 if rate is None else float(rate)
        value = min(1.0, max(0.0, value))
        scores.append((1.0 - value + epsilon) ** exponent)
    weights = np.asarray(scores, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        if weights.size == 0:
            return weights
        return np.full(weights.shape, 1.0 / weights.size, dtype=np.float64)
    return weights / total


def update_win_rate(previous: Optional[float], wins: int, losses: int, *,
                    ema: float = DEFAULT_EMA) -> float:
    """把一次批次结果并进胜率估计（EMA，`ema` 越大越信新结果）。"""
    if wins < 0 or losses < 0:
        raise ValueError("wins/losses 不能为负")
    total = wins + losses
    if total == 0:
        return 0.0 if previous is None else float(previous)
    if not 0.0 < ema <= 1.0:
        raise ValueError("ema 必须在 (0, 1]")
    observed = wins / total
    if previous is None:
        return observed
    return (1.0 - ema) * float(previous) + ema * observed


class OpponentPool:
    """历史快照池：每个文件一个对手，附带「learner 对它」的胜率估计。"""

    def __init__(self, root: Optional[str] = None, *, exponent: float = DEFAULT_EXPONENT,
                 epsilon: float = DEFAULT_EPSILON, ema: float = DEFAULT_EMA) -> None:
        self.root = os.path.abspath(root) if root else None
        self.exponent = exponent
        self.epsilon = epsilon
        self.ema = ema
        self._win_rates: Dict[str, Optional[float]] = {}

    # --- 池维护 ---------------------------------------------------------
    def add(self, path: str) -> str:
        """把一个快照加入池；重复加入不重置已有胜率。"""
        key = os.path.abspath(path)
        self._win_rates.setdefault(key, None)
        return key

    def add_all(self, paths: Iterable[str]) -> None:
        for path in paths:
            self.add(path)

    def discover(self) -> List[str]:
        """在 `root` 下扫描 `*.npz` 并全部加入（用于续训时复用旧快照）。"""
        if not self.root or not os.path.isdir(self.root):
            return []
        found = sorted(
            os.path.join(self.root, name)
            for name in os.listdir(self.root)
            if name.lower().endswith(".npz")
        )
        self.add_all(found)
        return found

    # --- 采样 -----------------------------------------------------------
    def paths(self) -> List[str]:
        return sorted(self._win_rates)

    def win_rates(self) -> Dict[str, Optional[float]]:
        return {path: self._win_rates[path] for path in self.paths()}

    def weights(self) -> np.ndarray:
        return pfsp_weights([self._win_rates[path] for path in self.paths()],
                            exponent=self.exponent, epsilon=self.epsilon)

    def sample(self, rng: np.random.Generator) -> Optional[str]:
        paths = self.paths()
        if not paths:
            return None
        weights = self.weights()
        index = int(rng.choice(len(paths), p=weights))
        return paths[index]

    # --- 结果回流 -------------------------------------------------------
    def record(self, path: str, wins: int, losses: int) -> float:
        key = self.add(path)
        updated = update_win_rate(self._win_rates.get(key), wins, losses, ema=self.ema)
        self._win_rates[key] = updated
        return updated

    def summary(self) -> Dict[str, object]:
        paths = self.paths()
        weights = self.weights() if paths else np.asarray([])
        return {
            "size": len(paths),
            "exponent": self.exponent,
            "epsilon": self.epsilon,
            "ema": self.ema,
            "opponents": [
                {
                    "name": os.path.basename(path),
                    "winRate": None if self._win_rates[path] is None else round(float(self._win_rates[path]), 4),
                    "weight": round(float(weight), 4),
                }
                for path, weight in zip(paths, weights.tolist())
            ],
        }
