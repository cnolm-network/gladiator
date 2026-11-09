"""策略后端（Layer 3 之上的训练算法），不属于环境层。

环境契约见 `docs/training-env-api.md`；算法实现见 `gladiator_train.policy`。
"""

from .policy import Policy, discounted_returns, normalize_advantages
from .trainer import EpisodeResult, Trainer, save_report

__all__ = [
    "Policy",
    "Trainer",
    "EpisodeResult",
    "discounted_returns",
    "normalize_advantages",
    "save_report",
]
