"""Gladiator 训练环境（Layer 3）。

```python
from gladiator_env import GladiatorEnv, BotConfig, ArenaConfig

env = GladiatorEnv(
    [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
     BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
    arena=ArenaConfig(center=(0.5, 65.0, 0.5), spacing=6.0),
    rcon={"host": "127.0.0.1", "port": 25575, "password": "devpass"},
    auto_spawn=True,
)
obs, info = env.reset()
obs, rewards, terminated, truncated, info = env.step({"bot_0": 1, "bot_1": 1})
env.close()
```
"""

from __future__ import annotations

from .env import ArenaConfig, BotConfig, GladiatorEnv
from .reward import RewardContext, RewardFn, default_reward
from .spaces import ACTION_NAMES, NUM_ACTIONS, OBS_DIM, OBS_FIELDS, decode_action, encode_observation

__all__ = [
    "ArenaConfig",
    "BotConfig",
    "GladiatorEnv",
    "RewardContext",
    "RewardFn",
    "default_reward",
    "ACTION_NAMES",
    "NUM_ACTIONS",
    "OBS_DIM",
    "OBS_FIELDS",
    "decode_action",
    "encode_observation",
]

try:  # 允许 `gymnasium.make("GladiatorBrawl-v0")`（需先 import 本包）
    from gymnasium.envs.registration import register, registry

    if "GladiatorBrawl-v0" not in registry:
        register(
            id="GladiatorBrawl-v0",
            entry_point="gladiator_env.env:GladiatorEnv",
            order_enforce=False,
        )
except Exception:  # pragma: no cover - 注册失败不影响直接构造
    pass
