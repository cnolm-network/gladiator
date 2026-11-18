"""线性-softmax 策略（actor-critic：REINFORCE 与 PPO 共用）。

这一层是**策略后端**，不是环境：环境只负责观测/动作/奖励，算法放在这里
（见 training/AGENTS.md "不将具体 RL 算法写入环境代码"）。

实现刻意保持无第三方依赖（只有 numpy）：
- 单隐藏层 tanh MLP + softmax 策略头 + **线性价值头**（PPO 的 critic）；
- 手写反向传播（20 多个观测、11 个动作，规模小到不值得引入 autograd）；
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

    PARAM_NAMES = ("W1", "b1", "W2", "b2", "Wv", "bv")

    def __init__(
        self,
        obs_dim: int = 16,
        n_actions: int = 7,
        hidden: int = 64,
        seed: int | None = None,
        #: PPO 的步长：REINFORCE 时代留下的 3e-3 对 PPO 太大（实测 entropy 两局就被压到 0.011）。
        lr: float = 3e-4,
        beta: float = 0.01,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        #: softmax 温度：``p = softmax(logits / T)``。T>1 让分布更平（熵更大）。
        #: 默认 1.0 = 旧行为；调参会写进 ``_meta``，续训时沿用。
        temperature: float = 1.0,
        #: 权重衰减（只作用于 W1/W2，不动 bias）；0 = 关闭。
        weight_decay: float = 0.0,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0 or hidden <= 0:
            raise ValueError("obs_dim / n_actions / hidden 必须为正")
        if temperature <= 0:
            raise ValueError("temperature 必须为正")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.beta = float(beta)
        self.adam_betas = adam_betas
        self.adam_eps = float(adam_eps)
        self.temperature = float(temperature)
        self.weight_decay = float(weight_decay)

        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / self.obs_dim), (self.hidden, self.obs_dim)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.n_actions, self.hidden)),
            "b2": np.zeros(self.n_actions),
            # 价值头：h → 标量。旧 npz 没有这两个参数，加载时随机初始化（见 load）。
            "Wv": rng.normal(0.0, np.sqrt(1.0 / self.hidden), (1, self.hidden)),
            "bv": np.zeros(1),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._steps = 0
        self.updates = 0

    # ---------------------------------------------------------------- forward
    def _hidden_and_logits(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (hidden, **未除温度**的 logits)。"""
        x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
        h = np.tanh(self.params["W1"] @ x + self.params["b1"])
        return h, self.params["W2"] @ h + self.params["b2"]

    def forward(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (hidden, probs)；``probs = softmax(logits / temperature)``。"""
        h, logits = self._hidden_and_logits(x)
        z = logits / self.temperature
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

    def hidden_of(self, x: Sequence[float]) -> np.ndarray:
        """单样本前向的隐层（PPO 的统计与测试要用）。"""
        x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
        return np.tanh(self.params["W1"] @ x + self.params["b1"])

    def value(self, x: Sequence[float]) -> float:
        """状态价值 V(s) = Wv·h + bv。"""
        h = self.hidden_of(x)
        return float(self.params["Wv"][0] @ h + self.params["bv"][0])

    def values(self, observations: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray([self.value(x) for x in observations], dtype=np.float64)

    def evaluate_actions(
        self,
        observations: Sequence[Sequence[float]],
        actions: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量算 ``(logp, entropy, values)``，供 PPO 算 ratio 与 GAE 用。"""
        n = len(observations)
        logp = np.zeros(n, dtype=np.float64)
        entropy = np.zeros(n, dtype=np.float64)
        values = np.zeros(n, dtype=np.float64)
        for i, (x, a) in enumerate(zip(observations, actions)):
            h, p = self.forward(x)
            logp[i] = float(np.log(p[int(a)] + 1e-12))
            entropy[i] = float(-np.sum(p * np.log(p + 1e-12)))
            values[i] = float(self.params["Wv"][0] @ h + self.params["bv"][0])
        return logp, entropy, values

    def mean_max_prob(self, observations: Sequence[Sequence[float]]) -> float:
        """批内「最大动作概率」的均值——接近 1 说明策略已经接近确定性。"""
        if len(observations) == 0:
            return 0.0
        total = 0.0
        for x in observations:
            total += float(np.max(self.probs(x)))
        return total / float(len(observations))

    def mean_prob_entropy(self, observations: Sequence[Sequence[float]]) -> float:
        """批内策略熵的均值（与 ``evaluate_actions`` 的 entropy 同口径，但不看动作）。"""
        if len(observations) == 0:
            return 0.0
        total = 0.0
        for x in observations:
            p = self.probs(x)
            total += float(-np.sum(p * np.log(p + 1e-12)))
        return total / float(len(observations))

    def logit_std(self, observations: Sequence[Sequence[float]]) -> float:
        """批内 **logits 标准差** 的均值（P2.3 饱和诊断）。

        为什么看它：``maxProb≈0.97`` 有两种成因——权重很大，或者观测特征互相一致
        导致 tanh 饱和、logits 本身被拉得很开。后者权重并不大（实测 |W1|=14、|W2|=6），
        梯度却会被 softmax 压到接近 0（策略梯度消失），所以必须和权重范数一起看。
        经验阈值：``logitStd > 4`` 基本可以认为已经饱和。
        """
        if len(observations) == 0:
            return 0.0
        total = 0.0
        for x in observations:
            _h, logits = self._hidden_and_logits(x)
            total += float(np.std(logits)) / self.temperature
        return total / float(len(observations))

    def reset_policy_head(self, seed: int | None = None, scale: float = 1.0) -> None:
        """重新初始化策略头 ``W2/b2``（Xavier），**保留 W1/b1 与价值头**（P2.3）。

        用途：BC 把策略头推到「几乎确定性」后，PPO 的 softmax 梯度会小到 1e-4，
        只靠 entropy_coef 拉不动。重设策略头能立刻把分布摊平（entropy 回到 ln|A| 附近），
        而隐层特征（W1/b1）与 critic 都保留，等于「保留表征、重开决策」。
        Adam 的一二阶矩也一并清零，否则旧动量会把新头又推回原处。
        """
        rng = np.random.default_rng(seed)
        fan_in, fan_out = self.hidden, self.n_actions
        std = float(scale) * np.sqrt(2.0 / (fan_in + fan_out))  # Xavier/Glorot
        self.params["W2"] = rng.normal(0.0, std, (self.n_actions, self.hidden))
        self.params["b2"] = np.zeros(self.n_actions)
        self._m["W2"] = np.zeros_like(self.params["W2"])
        self._v["W2"] = np.zeros_like(self.params["W2"])
        self._m["b2"] = np.zeros_like(self.params["b2"])
        self._v["b2"] = np.zeros_like(self.params["b2"])

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
            # 上面的导数是「对缩放后 logits（z = u/T）」求的；传到 W2/b2 要再除 T。
            dlogits = dlogits / self.temperature

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

    def ppo_gradients(
        self,
        observations: Sequence[Sequence[float]],
        actions: Sequence[int],
        old_logp: Sequence[float],
        advantages: Sequence[float],
        returns: Sequence[float],
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        old_values: Sequence[float] | None = None,
        value_clip: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """PPO（clip）目标对参数的梯度，**上升方向**（与 :meth:`gradients` 同约定）。

        ``J = mean( min(r·A, clip(r,1-ε,1+ε)·A) ) + β·H(π) - value_coef·0.5·(V-R)²``

        关键实现点：只有 ``min`` **选中**的那一支有梯度。``r`` 冲出裁剪区间时，
        ``min`` 选中的是 ``clip(r)·A``，它对 ``logits`` 的导数为 0——
        也就是「被 clip 的样本不再拉动策略」，这正是 PPO 的稳定来源。
        价值损失对 ``V`` 的导数是 ``-(V-R)·value_coef``（上升方向），并会经 ``h`` 回传到 W1/b1。

        ``value_clip>0`` 时（P2.2）对价值头做同样的裁剪：以行为策略的 ``old_values`` 为锚，
        ``v_clipped = old_v + clip(v − old_v, ±value_clip)``，损失取两支的最大值。
        ``v`` 已经跑得比裁剪值更差时，这一支的梯度为 0——抑制价值头被单局大回报拽飞。
        """
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        n = len(observations)
        if n == 0:
            return grads
        low = 1.0 - float(clip)
        high = 1.0 + float(clip)
        wv = self.params["Wv"][0]
        clip_value = float(value_clip or 0.0)
        old_v = None if old_values is None else np.asarray(old_values, dtype=np.float64).reshape(-1)
        for index, (x, a, old_lp, adv, ret) in enumerate(
                zip(observations, actions, old_logp, advantages, returns)):
            x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
            h, p = self.forward(x)
            logp = float(np.log(p[int(a)] + 1e-12))
            ratio = float(np.exp(logp - float(old_lp)))
            adv = float(adv)
            surr_unclipped = ratio * adv
            surr_clipped = min(max(ratio, low), high) * adv
            entropy = float(-np.sum(p * np.log(p + 1e-12)))

            if surr_unclipped <= surr_clipped:
                # min 取未裁剪那一支：d(r·A)/dlogits = A · r · (onehot − p)
                dlogits = -(adv * ratio) * (_onehot(a, self.n_actions) - p)
            else:
                # min 取裁剪那一支：clip 是常数，梯度为 0（只剩熵正则）
                dlogits = np.zeros(self.n_actions, dtype=np.float64)
            dlogits += -float(entropy_coef) * p * (np.log(p + 1e-12) + entropy)
            # dlogits 是对 z = logits/T 求的；回传到 W2/b2 要再除 T。
            dlogits = dlogits / self.temperature

            value = float(wv @ h + self.params["bv"][0])
            grad_scale = 1.0
            if clip_value > 0.0 and old_v is not None and index < old_v.size:
                anchor = float(old_v[index])
                clipped_value = anchor + float(np.clip(value - anchor, -clip_value, clip_value))
                loss_plain = (value - float(ret)) ** 2
                loss_clipped = (clipped_value - float(ret)) ** 2
                if loss_clipped > loss_plain:
                    # 裁剪支更差 → 被选中，梯度为 0（不再推价值头）。
                    grad_scale = 0.0
            dvalue = -float(value_coef) * (value - float(ret)) * grad_scale

            grads["W2"] += np.outer(dlogits, h)
            grads["b2"] += dlogits
            grads["Wv"][0] += dvalue * h
            grads["bv"][0] += dvalue
            dh = self.params["W2"].T @ dlogits + dvalue * wv
            dz = dh * (1.0 - h * h)
            grads["W1"] += np.outer(dz, x)
            grads["b1"] += dz
        for k in grads:
            grads[k] /= float(n)
        return grads

    def apply_gradients(self, grads: Dict[str, np.ndarray],
                        max_grad_norm: float = 0.0) -> float:
        """Adam **上升**一步：``θ += lr · Adam(∇J)``。

        注意这里不能再对梯度取负。旧实现写成 ``g = -grads`` 再 ``θ += lr·ĝ``，
        等价于沿 ``∇J`` 的**反方向**走（实测：给动作 1 正的 advantage，
        更新后 ``p(action=1)`` 从 0.328 掉到 0.207）——即策略在做「反向学习」。
        PPO 的裁剪/GAE 都建立在「按 ∇J 上升」之上，这个符号必须与 ``gradients`` 的
        「返回 J 的梯度」约定一致。
        """
        # 全局 L2 范数裁剪：PPO 单步梯度过大会把 softmax 推到近乎确定性，
        # 这是「entropy 塌到 0 就再也不学」的直接原因之一。
        norm = float(np.sqrt(sum(float(np.sum(np.square(g))) for g in grads.values())))
        scale = 1.0
        if max_grad_norm and max_grad_norm > 0.0 and norm > max_grad_norm:
            scale = max_grad_norm / (norm + 1e-12)
        self.last_grad_norm = norm

        self._steps += 1
        b1, b2 = self.adam_betas
        bc1 = 1.0 - b1 ** self._steps
        bc2 = 1.0 - b2 ** self._steps
        for k in self.PARAM_NAMES:
            g = grads[k] * scale  # 上升方向（按需缩放）
            if self.weight_decay > 0 and k in ("W1", "W2"):
                # 上升目标里带 -λ/2·||W||²，梯度是 -λ·W；bias 不衰减。
                g = g - self.weight_decay * self.params[k]
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)
            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2
            self.params[k] += self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        self.updates += 1
        return norm

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
                    # P2.3：温度随模型走（续训沿用），旧文件缺省 1.0。
                    "temperature": self.temperature,
                    **(extra or {}),
                }
            )
        )
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str, obs_dim: int | None = None) -> "Policy":
        """加载策略；``obs_dim`` 更大时把旧 W1 按**原索引**扩展（右侧补零列）。

        兼容性（老模型必须能直接接着练）：
        - 旧 ``.npz`` 没有价值头 ``Wv/bv`` → 随机初始化（critic 从零学，不污染 actor）；
        - 旧 ``obs_dim`` 小于当前环境 → ``W1`` 补零列，旧特征仍落在**原来的下标**上。
          观测新增字段一律追加在 ``OBS_FIELDS`` 末尾（见 spaces.py），所以补零只能补在右侧；
          若补在左侧，旧特征会整体右移，权重语义会被破坏。
        """
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["_meta"]))
            stored_dim = int(meta["obs_dim"])
            target_dim = int(obs_dim) if obs_dim is not None else stored_dim
            policy = cls(
                obs_dim=target_dim,
                n_actions=int(meta["n_actions"]),
                hidden=int(meta["hidden"]),
            )
            for k in cls.PARAM_NAMES:
                if k not in data:
                    # 旧文件没有价值头：保留构造函数里的随机初始化。
                    continue
                value = np.array(data[k], dtype=np.float64)
                if k == "W1" and value.shape[1] != target_dim:
                    value = _pad_obs_columns(value, target_dim)
                policy.params[k] = value
        policy.updates = int(meta.get("updates", 0))
        # P2.3：旧 npz 没有 temperature → 1.0（与旧行为一致）。
        policy.temperature = float(meta.get("temperature", 1.0) or 1.0)
        if policy.temperature <= 0:
            policy.temperature = 1.0
        policy.stored_obs_dim = stored_dim
        return policy


def check_compat(policy: "Policy", *, obs_dim: int, n_actions: int, hidden: int) -> List[str]:
    """返回不匹配项（空列表=兼容）。供 ``--init-policy`` 的启动校验复用，便于单测。"""
    problems: List[str] = []
    # 旧模型 obs_dim 更小是**允许**的：加载时右侧补零列即可接着练（P1 加的 4 个观测字段）。
    # 只有「模型比环境宽」无法自动降维，才报错。
    if int(policy.obs_dim) > int(obs_dim):
        problems.append(f"obs_dim {policy.obs_dim} > {obs_dim}（无法降维）")
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
    for key in Policy.PARAM_NAMES:
        value = np.array(loaded.params[key], dtype=np.float64)
        if key == "W1" and value.shape[1] != policy.obs_dim:
            value = _pad_obs_columns(value, policy.obs_dim)
        policy.params[key] = value
    # P2.3：温度也随模型走（--logit-temp 显式给了的话，调用方会在 adopt 之后覆盖）。
    policy.temperature = float(getattr(loaded, "temperature", 1.0) or 1.0)
    return policy


def _pad_obs_columns(matrix: np.ndarray, target_dim: int) -> np.ndarray:
    """把 ``W1``（hidden, old_dim）扩到（hidden, target_dim）：旧列保持原下标，右侧补零。

    新增观测字段追加在 ``OBS_FIELDS`` 末尾，所以旧列必须留在左侧。
    """
    hidden, old = matrix.shape
    if old == target_dim:
        return matrix.copy()
    if old > target_dim:
        raise ValueError(f"无法降维: W1 {old} > {target_dim}")
    out = np.zeros((hidden, target_dim), dtype=np.float64)
    out[:, :old] = matrix
    return out


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
