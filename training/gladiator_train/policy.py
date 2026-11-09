"""线性-softmax 策略（REINFORCE + 基线）。

这一层是**策略后端**，不是环境：环境只负责观测/动作/奖励，算法放在这里
（见 training/AGENTS.md "不将具体 RL 算法写入环境代码"）。

实现刻意保持无第三方依赖（只有 numpy）：
- 单隐藏层 tanh MLP + softmax；
- 手写反向传播（16 维观测、7 个动作，规模小到不值得引入 autograd）；
- Adam 优化器；
- 保存/加载 `.npz`，方便训练后拿到策略权重。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class Policy:
    """单隐藏层 softmax 策略，参数名与 `save()/load()` 一致。"""

    PARAM_NAMES = ("W1", "b1", "W2", "b2")

    def __init__(
        self,
        obs_dim: int = 16,
        n_actions: int = 7,
        hidden: int = 64,
        seed: int | None = None,
        lr: float = 3e-3,
        beta: float = 0.01,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0 or hidden <= 0:
            raise ValueError("obs_dim / n_actions / hidden 必须为正")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.beta = float(beta)
        self.adam_betas = adam_betas
        self.adam_eps = float(adam_eps)

        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / self.obs_dim), (self.hidden, self.obs_dim)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.n_actions, self.hidden)),
            "b2": np.zeros(self.n_actions),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._steps = 0
        self.updates = 0

    # ---------------------------------------------------------------- forward
    def forward(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (hidden, probs)。"""
        x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
        h = np.tanh(self.params["W1"] @ x + self.params["b1"])
        z = self.params["W2"] @ h + self.params["b2"]
        z = z - np.max(z)
        e = np.exp(z)
        return h, e / np.sum(e)

    def probs(self, x: Sequence[float]) -> np.ndarray:
        return self.forward(x)[1]

    def act(self, x: Sequence[float], rng: np.random.Generator, greedy: bool = False) -> int:
        p = self.probs(x)
        if greedy:
            return int(np.argmax(p))
        return int(rng.choice(p.shape[0], p=p))

    # --------------------------------------------------------------- backward
    def gradients(
        self,
        observations: Sequence[Sequence[float]],
        actions: Sequence[int],
        advantages: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """REINFORCE 目标 J = (1/N)·Σ A·logπ(a|x) + β·H(π) 的梯度（越大越好）。

        返回的是 **J 的梯度**（样本平均），调用方用 Adam 做上升。
        """
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        if len(observations) == 0:
            return grads
        for x, a, adv in zip(observations, actions, advantages):
            x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
            h, p = self.forward(x)
            # dJ/dlogits = (onehot - p)·A - β·p·(log p + H)
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
            dlogits = (_onehot(a, self.n_actions) - p) * float(adv)
            dlogits += -self.beta * p * (np.log(p + 1e-12) + entropy)

            grads["W2"] += np.outer(dlogits, h)
            grads["b2"] += dlogits
            dh = self.params["W2"].T @ dlogits
            dz = dh * (1.0 - h * h)
            grads["W1"] += np.outer(dz, x)
            grads["b1"] += dz
        n = float(len(observations))
        for k in grads:
            grads[k] /= n
        return grads

    def apply_gradients(self, grads: Dict[str, np.ndarray]) -> None:
        """Adam 上升一步（内部对梯度取负，复用最小化公式）。"""
        self._steps += 1
        b1, b2 = self.adam_betas
        bc1 = 1.0 - b1 ** self._steps
        bc2 = 1.0 - b2 ** self._steps
        for k in self.PARAM_NAMES:
            g = -grads[k]  # 下降方向
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)
            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2
            self.params[k] += self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        self.updates += 1

    def update(
        self,
        observations: Sequence[Sequence[float]],
        actions: Sequence[int],
        advantages: Sequence[float],
    ) -> None:
        self.apply_gradients(self.gradients(observations, actions, advantages))

    # ------------------------------------------------------------ persistence
    def save(self, path: str, extra: Dict[str, object] | None = None) -> None:
        payload = {k: v for k, v in self.params.items()}
        payload["_meta"] = np.array(
            json.dumps(
                {
                    "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions,
                    "hidden": self.hidden,
                    "updates": self.updates,
                    **(extra or {}),
                }
            )
        )
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "Policy":
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["_meta"]))
            policy = cls(
                obs_dim=int(meta["obs_dim"]),
                n_actions=int(meta["n_actions"]),
                hidden=int(meta["hidden"]),
            )
            for k in cls.PARAM_NAMES:
                policy.params[k] = np.array(data[k], dtype=np.float64)
        policy.updates = int(meta.get("updates", 0))
        return policy


def check_compat(policy: "Policy", *, obs_dim: int, n_actions: int, hidden: int) -> List[str]:
    """返回不匹配项（空列表=兼容）。供 ``--init-policy`` 的启动校验复用，便于单测。"""
    problems: List[str] = []
    if int(policy.obs_dim) != int(obs_dim):
        problems.append(f"obs_dim {policy.obs_dim} != {obs_dim}")
    if int(policy.n_actions) != int(n_actions):
        problems.append(f"n_actions {policy.n_actions} != {n_actions}")
    if int(policy.hidden) != int(hidden):
        problems.append(f"hidden {policy.hidden} != {hidden}")
    return problems


def validate_init_policy(path: str, *, obs_dim: int, n_actions: int,
                         hidden: int) -> Tuple[Policy | None, str | None]:
    """校验 ``--init-policy``；返回 ``(已加载策略, 中文错误)``。

    无路径时返回 ``(None, None)``；文件缺失/损坏/维度不符时返回 ``(None, 原因)``。
    调用方据此在创建任何环境之前退出（exit 2），避免跑了半天才发现模型用不了。
    """
    if not path:
        return None, None
    if not os.path.isfile(path):
        return None, f"初始策略不存在: {path}"
    try:
        loaded = Policy.load(path)
    except Exception as exc:  # noqa: BLE001 - npz 损坏/缺 _meta 都算不可用
        return None, f"初始策略无法加载: {exc}"
    problems = check_compat(loaded, obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    if problems:
        return None, "初始策略与当前环境不匹配: " + ", ".join(problems)
    return loaded, None


def adopt_weights(policy: Policy, loaded: Policy) -> Policy:
    """把已校验的权重装进本次策略：只接管参数，**不接管 optimizer / updates 状态**。"""
    policy.params = {key: np.array(loaded.params[key], dtype=np.float64)
                     for key in Policy.PARAM_NAMES}
    return policy


def _onehot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float64)
    vec[int(index)] = 1.0
    return vec


def discounted_returns(rewards: Iterable[float], gamma: float) -> np.ndarray:
    """从奖励序列计算折扣回报（无 bootstrap：episode 结束即终止）。"""
    out: List[float] = []
    running = 0.0
    for r in reversed(list(rewards)):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return np.asarray(out, dtype=np.float64)


def normalize_advantages(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if returns.size == 0:
        return returns
    std = float(np.std(returns))
    if std < eps:
        return returns - float(np.mean(returns))
    return (returns - float(np.mean(returns))) / (std + eps)
