"""线性-softmax 策略（actor-critic：REINFORCE 与 PPO 共用）。

这一层是**策略后端**，不是环境：环境只负责观测/动作/奖励，算法放在这里
（见 training/AGENTS.md "不将具体 RL 算法写入环境代码"）。

实现刻意保持无第三方依赖（只有 numpy）：
- 单隐藏层 tanh MLP + softmax 策略头 + **线性价值头**（PPO 的 critic）；
- 手写反向传播（20 多个观测、11 个动作，规模小到不值得引入 autograd）；
- Adam 优化器；
- 保存/加载 `.npz`，方便训练后拿到策略权重。

两种动作空间（``action_space``）：

- ``v1``（默认，也是旧模型的形态）：**单头** Categorical，动作是 ``int``，
  参数名 ``W2`` / ``b2``；
- ``v2``：**多离散头（factorized）**，``h = tanh(W1·x + b1)`` 共享，
  每个头一套 ``W2_<head>`` / ``b2_<head>``，动作是长度 H 的 int 数组（当前 H=10，
  见 ``spaces.ACTION_HEADS_V2``），``log π(a|s) = Σ_head log π_head(a_head|s)``，
  熵也是逐头求和。头的数量与大小**一律从 ``head_sizes`` 推导**，本模块不假设具体头数：
  G-707 加 ``sprint_reset`` 时把 H 从 9 改成 10，除了默认值之外不需要改这里的逻辑。

v2 的头名与顺序来自 ``gladiator_env.spaces.ACTION_HEADS_V2``（延迟导入，
避免策略模块在 import 期就拉起 gymnasium）。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

#: 被 action mask 屏蔽的位置用的 logits（softmax 之后概率≈0，但仍是有限值，不会有 NaN）。
MASKED_LOGIT = -1e9

#: v1 的参数名（``Policy.PARAM_NAMES`` 在类上访问时就是这一组，兼容旧调用方）。
V1_PARAM_NAMES = ("W1", "b1", "W2", "b2", "Wv", "bv")

#: v2 头名/头大小缓存（避免每次构造都去 import gladiator_env）。
_HEADS_CACHE: Tuple[List[str], List[int]] | None = None


def default_action_heads() -> Tuple[List[str], List[int]]:
    """``ACTION_HEADS_V2`` 的 ``(头名列表, 头大小列表)``（顺序即头顺序）。"""
    global _HEADS_CACHE
    if _HEADS_CACHE is None:
        from gladiator_env.spaces import ACTION_HEADS_V2

        _HEADS_CACHE = (list(ACTION_HEADS_V2.keys()), [int(v) for v in ACTION_HEADS_V2.values()])
    return list(_HEADS_CACHE[0]), list(_HEADS_CACHE[1])


class _ParamNames(list):
    """既是列表又可调用：``policy.PARAM_NAMES`` 可迭代，``policy.PARAM_NAMES()`` 也可调用。"""

    def __call__(self) -> List[str]:
        return list(self)


class _ParamNamesDescriptor:
    """``Policy.PARAM_NAMES``：类上访问永远给 v1 名字，实例上访问给该实例的名字。

    这样 ``for key in Policy.PARAM_NAMES``（旧代码）与 ``policy.PARAM_NAMES()``（v2 需要）
    都能工作，不必把两种调用点全部翻一遍。
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return _ParamNames(V1_PARAM_NAMES)
        return _ParamNames(obj.param_names())



class Policy:
    """单隐藏层 softmax 策略，参数名与 `save()/load()` 一致。

    ``action_space="v1"`` 时与旧实现逐字节一致（单头 ``W2``/``b2``、动作为 ``int``）；
    ``action_space="v2"`` 时是 H 个离散头（当前 10）的因子化策略（每条 ``W2_<head>``/``b2_<head>``）。
    """

    #: 类上访问 → v1 参数名；实例上访问/调用 → 该实例的参数名（见 ``_ParamNamesDescriptor``）。
    PARAM_NAMES = _ParamNamesDescriptor()

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
        #: ``"v1"``（单头 Categorical）或 ``"v2"``（多离散头 factorized）。
        action_space: str = "v1",
        #: v2 的头大小/头名；缺省取 ``ACTION_HEADS_V2``（不需要调用方传）。
        head_sizes: Sequence[int] | None = None,
        head_names: Sequence[str] | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature 必须为正")
        self.action_space = str(action_space or "v1").strip().lower()
        if self.action_space not in ("v1", "v2"):
            raise ValueError(f"未知动作空间: {action_space}（允许 v1 / v2）")
        if hidden <= 0:
            raise ValueError("hidden 必须为正")
        self.is_v2 = self.action_space == "v2"
        if self.is_v2:
            names, sizes = default_action_heads()
            if head_names is not None:
                names = [str(name) for name in head_names]
            if head_sizes is not None:
                sizes = [int(size) for size in head_sizes]
            if len(names) != len(sizes):
                raise ValueError(f"head_names（{len(names)}）与 head_sizes（{len(sizes)}）数量不一致")
            if any(size <= 1 for size in sizes):
                raise ValueError(f"每个头至少要有 2 个取值，收到 {sizes}")
            self._head_names: List[str] = list(names)
            self._head_sizes: List[int] = list(sizes)
        else:
            if n_actions <= 0:
                raise ValueError("n_actions 必须为正")
            self._head_names = ["action"]
            self._head_sizes = [int(n_actions)]
        if obs_dim <= 0:
            raise ValueError("obs_dim 必须为正")
        self.obs_dim = int(obs_dim)
        #: v1 = 动作个数；v2 = 各头大小之和（仅供日志/旧代码读取，不代表单一 softmax）。
        self.n_actions = int(sum(self._head_sizes))
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
        }
        if self.is_v2:
            for name, size in zip(self._head_names, self._head_sizes):
                self.params[f"W2_{name}"] = rng.normal(
                    0.0, np.sqrt(1.0 / self.hidden), (size, self.hidden))
                self.params[f"b2_{name}"] = np.zeros(size)
        else:
            self.params["W2"] = rng.normal(
                0.0, np.sqrt(1.0 / self.hidden), (self.n_actions, self.hidden))
            self.params["b2"] = np.zeros(self.n_actions)
        # 价值头：h → 标量。旧 npz 没有这两个参数，加载时随机初始化（见 load）。
        self.params["Wv"] = rng.normal(0.0, np.sqrt(1.0 / self.hidden), (1, self.hidden))
        self.params["bv"] = np.zeros(1)
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._steps = 0
        self.updates = 0
        #: 某个头的 mask 全为 False 时退回「全开」的次数（空背包的 slot 头会这样）。
        self.mask_fallbacks = 0

    # ------------------------------------------------------------ 头/参数名
    def head_names(self) -> List[str]:
        """头名（v1 是 ``["action"]``，v2 是 H 个头，当前 10 个）。"""
        return list(self._head_names)

    def head_sizes(self) -> List[int]:
        """每个头的取值个数（v1 是 ``[n_actions]``）。"""
        return list(self._head_sizes)

    def param_names(self) -> List[str]:
        """该实例的参数名列表（v2 形如 ``W2_move``/``b2_slot``）。"""
        if not self.is_v2:
            return list(V1_PARAM_NAMES)
        names = ["W1", "b1"]
        names += [f"W2_{head}" for head in self._head_names]
        names += [f"b2_{head}" for head in self._head_names]
        names += ["Wv", "bv"]
        return names

    def normalize_mask(self, mask) -> List[List[bool]] | None:
        """把环境给的 actionMask 规范成 ``list[list[bool]]``（按**头顺序**）。

        接受三种形态：``dict head -> list[bool]``（单智能体）、
        ``{bot_key: dict}``（多智能体，取 ``bot_0``）、以及已经是列表的。
        缺某个头时按「全开」补，长度不符按头的真实大小截/补。
        """
        if mask is None:
            return None
        if isinstance(mask, dict):
            inner = mask.get("bot_0")
            if isinstance(inner, dict):
                mask = inner
            rows = []
            for name, size in zip(self._head_names, self._head_sizes):
                row = mask.get(name)
                if row is None:
                    rows.append([True] * size)
                    continue
                values = [bool(v) for v in np.asarray(row, dtype=bool).reshape(-1)]
                if len(values) < size:
                    values += [True] * (size - len(values))
                rows.append(values[:size])
            return rows
        if isinstance(mask, (list, tuple)):
            rows = []
            for index, size in enumerate(self._head_sizes):
                row = mask[index] if index < len(mask) else [True] * size
                values = [bool(v) for v in np.asarray(row, dtype=bool).reshape(-1)]
                if len(values) < size:
                    values += [True] * (size - len(values))
                rows.append(values[:size])
            return rows
        return None

    # ---------------------------------------------------------------- forward
    def _hidden_and_logits(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (hidden, **未除温度**的 logits)。**仅 v1**（v2 用 :meth:`logits_all`）。"""
        if self.is_v2:
            raise ValueError(
                f"v2 是 {len(self._head_sizes)} 个独立头，没有单一 logits："
                "请用 logits_all()/log_prob()")
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
        if self.is_v2:
            # v2 的动作是 (H,) 数组，用 int 表示会悄悄丢头——直接拒绝，避免调用方拿错接口。
            raise ValueError("v2 策略的动作是 (H,) 数组：请用 sample() / act_with_mask()")
        p = self.probs(x)
        if greedy:
            return int(np.argmax(p))
        return int(rng.choice(p.shape[0], p=p))

    # ---------------------------------------------------- v2：多离散头 API
    def logits_all(self, x: Sequence[float]) -> List[np.ndarray]:
        """每个头在 ``x`` 上的 logits（**未除温度**、**未加 mask**）。"""
        h = self.hidden_of(x)
        if not self.is_v2:
            return [self.params["W2"] @ h + self.params["b2"]]
        return [self.params[f"W2_{head}"] @ h + self.params[f"b2_{head}"]
                for head in self._head_names]

    def _masked(self, logits: List[np.ndarray],
                mask: Sequence[Sequence[bool]] | None) -> List[np.ndarray]:
        """把被屏蔽的 logits 置 ``MASKED_LOGIT``；某头 mask 全 False 时退回全开。

        全 False 是真实会发生的（热键栏空着时 ``slot`` 头就是 9 个 False）。
        那种情况没有「合法动作」可选，与其让 softmax 在全 -1e9 上退化，
        不如把这一头放开并计数（``mask_fallbacks``），训练才不会中途炸掉。
        """
        if mask is None:
            return list(logits)
        rows = list(mask)
        if len(rows) != len(logits):
            raise ValueError(f"mask 头数 {len(rows)} 与策略头数 {len(logits)} 不一致")
        out: List[np.ndarray] = []
        for index, head_logits in enumerate(logits):
            row = np.asarray(rows[index], dtype=bool).reshape(-1)
            if row.size != head_logits.size:
                raise ValueError(
                    f"第 {index} 个头的 mask 长度 {row.size} 与头大小 {head_logits.size} 不一致")
            if not bool(row.any()):
                self.mask_fallbacks += 1
                out.append(np.array(head_logits, dtype=np.float64, copy=True))
                continue
            masked = np.array(head_logits, dtype=np.float64, copy=True)
            masked[~row] = MASKED_LOGIT
            out.append(masked)
        return out

    def _head_probs(self, head_logits: np.ndarray) -> np.ndarray:
        """带温度、数值稳定的 softmax（屏蔽位概率≈0）。"""
        z = np.asarray(head_logits, dtype=np.float64) / self.temperature
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    @staticmethod
    def _head_log_softmax(head_logits: np.ndarray, temperature: float) -> np.ndarray:
        z = np.asarray(head_logits, dtype=np.float64) / temperature
        shift = z - np.max(z)
        return shift - np.log(np.sum(np.exp(shift)))

    def sample(self, x: Sequence[float], rng: np.random.Generator,
               mask: Sequence[Sequence[bool]] | None = None,
               greedy: bool = False) -> np.ndarray:
        """按（可选 mask 后的）各头分布采样，返回 ``(H,) int64``（v2 专用）。

        参数顺序与 :meth:`act` 一致（``x, rng``），掩码在第三位；``greedy=True`` 时
        每个头取 argmax（评估用，同一观测结果可复现）。
        """
        if not self.is_v2:
            raise ValueError("sample() 只用于 v2；v1 请用 act()")
        logits = self._masked(self.logits_all(x), mask)
        picked = np.zeros(len(logits), dtype=np.int64)
        for index, head_logits in enumerate(logits):
            p = self._head_probs(head_logits)
            if greedy:
                picked[index] = int(np.argmax(p))
            else:
                picked[index] = int(rng.choice(p.shape[0], p=p))
        return picked

    def log_prob(self, observations: Sequence[Sequence[float]],
                 actions: Sequence[Sequence[int]],
                 masks: Sequence[Sequence[Sequence[bool]]] | None = None) -> np.ndarray:
        """逐头的 log-softmax 之和，返回 ``(T,)``。"""
        n = len(observations)
        out = np.zeros(n, dtype=np.float64)
        for index, (x, action) in enumerate(zip(observations, actions)):
            out[index] = self._log_prob_one(x, action, None if masks is None else masks[index])
        return out

    def _log_prob_one(self, x: Sequence[float], action: Sequence[int],
                      mask: Sequence[Sequence[bool]] | None = None) -> float:
        if not self.is_v2:
            raise ValueError("_log_prob_one 只用于 v2")
        picked = np.asarray(action, dtype=np.int64).reshape(-1)
        if picked.size != len(self._head_sizes):
            raise ValueError(
                f"动作维度 {picked.size} 与头数 {len(self._head_sizes)} 不一致")
        logits = self._masked(self.logits_all(x), mask)
        total = 0.0
        for head_index, head_logits in enumerate(logits):
            value = int(picked[head_index])
            if value < 0 or value >= head_logits.size:
                raise ValueError(
                    f"第 {head_index} 个头的动作越界: {value}（合法范围 0..{head_logits.size - 1}）")
            total += float(self._head_log_softmax(head_logits, self.temperature)[value])
        return total

    def entropy(self, observations: Sequence[Sequence[float]],
                masks: Sequence[Sequence[Sequence[bool]]] | None = None) -> np.ndarray:
        """逐头熵之和，返回 ``(T,)``（mask 后的分布熵）。"""
        out = np.zeros(len(observations), dtype=np.float64)
        for index, x in enumerate(observations):
            mask = None if masks is None else masks[index]
            out[index] = self._entropy_one(x, mask)
        return out

    def _entropy_one(self, x: Sequence[float],
                     mask: Sequence[Sequence[bool]] | None = None) -> float:
        logits = self._masked(self.logits_all(x), mask)
        total = 0.0
        for head_logits in logits:
            p = self._head_probs(head_logits)
            total += float(-np.sum(p * np.log(p + 1e-12)))
        return total

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
        actions: Sequence,
        masks: Sequence[Sequence[Sequence[bool]]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量算 ``(logp, entropy, values)``，供 PPO 算 ratio 与 GAE 用。

        v2 下 ``actions`` 是 ``(T, H)``，``logp`` 是各头 log-softmax 之和、
        ``entropy`` 是各头熵之和（与 ``log_prob``/``entropy`` 同口径）。
        """
        n = len(observations)
        logp = np.zeros(n, dtype=np.float64)
        entropy = np.zeros(n, dtype=np.float64)
        values = np.zeros(n, dtype=np.float64)
        if self.is_v2:
            for i, (x, a) in enumerate(zip(observations, actions)):
                mask = None if masks is None else masks[i]
                logp[i] = self._log_prob_one(x, a, mask)
                entropy[i] = self._entropy_one(x, mask)
                h = self.hidden_of(x)
                values[i] = float(self.params["Wv"][0] @ h + self.params["bv"][0])
            return logp, entropy, values
        for i, (x, a) in enumerate(zip(observations, actions)):
            h, p = self.forward(x)
            logp[i] = float(np.log(p[int(a)] + 1e-12))
            entropy[i] = float(-np.sum(p * np.log(p + 1e-12)))
            values[i] = float(self.params["Wv"][0] @ h + self.params["bv"][0])
        return logp, entropy, values

    def mean_max_prob(self, observations: Sequence[Sequence[float]]) -> float:
        """批内「最大动作概率」的均值——接近 1 说明策略已经接近确定性。

        v2 是「各头最大概率的均值」：只要有一个头非常确定，数值就会被拉高，
        与 v1 的语义（是否接近确定性）一致，但**不**代表联合分布的 maxProb。
        """
        if len(observations) == 0:
            return 0.0
        total = 0.0
        for x in observations:
            if self.is_v2:
                probs = [np.max(self._head_probs(head)) for head in self.logits_all(x)]
                total += float(np.mean(probs)) if probs else 0.0
            else:
                total += float(np.max(self.probs(x)))
        return total / float(len(observations))

    def mean_prob_entropy(self, observations: Sequence[Sequence[float]]) -> float:
        """批内策略熵的均值（与 ``evaluate_actions`` 的 entropy 同口径，但不看动作）。"""
        if len(observations) == 0:
            return 0.0
        if self.is_v2:
            return float(np.mean(self.entropy(observations)))
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

        v2 先按头算标准差再取平均：各头里有的很平（move）有的很尖（attack），
        先平均会把尖的那个头的信息冲掉。
        """
        if len(observations) == 0:
            return 0.0
        total = 0.0
        for x in observations:
            if self.is_v2:
                per_head = [float(np.std(head)) / self.temperature for head in self.logits_all(x)]
                total += float(np.mean(per_head)) if per_head else 0.0
            else:
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
        if self.is_v2:
            # v2：逐个重设每个头的 W2_<head>/b2_<head>（Xavier 按该头的 fan_out 算）。
            for head, size in zip(self._head_names, self._head_sizes):
                key_w, key_b = f"W2_{head}", f"b2_{head}"
                std = float(scale) * np.sqrt(2.0 / (self.hidden + size))
                self.params[key_w] = rng.normal(0.0, std, (size, self.hidden))
                self.params[key_b] = np.zeros(size)
                self._m[key_w] = np.zeros_like(self.params[key_w])
                self._v[key_w] = np.zeros_like(self.params[key_w])
                self._m[key_b] = np.zeros_like(self.params[key_b])
                self._v[key_b] = np.zeros_like(self.params[key_b])
            return
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
        masks: Sequence[Sequence[Sequence[bool]]] | None = None,
    ) -> Dict[str, np.ndarray]:
        """REINFORCE 目标 J = (1/N)·Σ A·logπ(a|x) + β·H(π) 的梯度（越大越好）。

        返回的是 **J 的梯度**（样本平均），调用方用 Adam 做上升。
        v2 下 ``logπ`` 与 ``H`` 都是逐头求和（factorized），梯度按头分别回传。
        """
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        if len(observations) == 0:
            return grads
        if self.is_v2:
            for index, (x, a, adv) in enumerate(zip(observations, actions, advantages)):
                mask = None if masks is None else masks[index]
                self._accumulate_v2(grads, x, a, float(adv), mask,
                                    entropy_coef=self.beta)
            n = float(len(observations))
            for k in grads:
                grads[k] /= n
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

    def _accumulate_v2(self, grads: Dict[str, np.ndarray], x, action,
                       adv: float, mask, *, entropy_coef: float,
                       value_coef: float = 0.0,
                       clip: Tuple[float, float] | None = None,
                       old_logp: float | None = None,
                       value_term: Tuple[float, float, float] | None = None) -> float:
        """v2 的单样本梯度累加（REINFORCE 与 PPO 共用）。

        - ``clip is None`` → REINFORCE：``dlogits_i = A·(onehot_i − p_i)``；
        - ``clip = (low, high)`` → PPO：先按**联合** logp 算 ratio，再由 ``min`` 选中哪一支，
          选中裁剪支时所有头的策略梯度都为 0（只剩熵正则）。
        - ``value_term = (value, return_target, grad_scale)``，``grad_scale=0`` 表示价值支被裁掉。

        返回该样本的联合 ``logp``（PPO 外面算 ratio 用）。
        """
        x = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
        h = self.hidden_of(x)
        picked = np.asarray(action, dtype=np.int64).reshape(-1)
        if picked.size != len(self._head_sizes):
            raise ValueError(f"动作维度 {picked.size} 与头数 {len(self._head_sizes)} 不一致")
        logits = self._masked(self.logits_all(x), mask)

        # 第一遍：各头概率 + 联合 logp（PPO 的 ratio 必须建立在所有头之和上）。
        probs: List[np.ndarray] = []
        logp_total = 0.0
        for head_index, head_logits in enumerate(logits):
            value = int(picked[head_index])
            if value < 0 or value >= head_logits.size:
                raise ValueError(
                    f"第 {head_index} 个头的动作越界: {value}（合法范围 0..{head_logits.size - 1}）")
            p = self._head_probs(head_logits)
            probs.append(p)
            logp_total += float(np.log(p[value] + 1e-12))

        # 第二遍：按 PPO 的 min 分支决定策略梯度是否为零（每个头共用同一个 ratio）。
        policy_scale: float | None
        if clip is None:
            policy_scale = adv
        else:
            ratio = float(np.exp(logp_total - float(old_logp)))
            low, high = clip
            surr_unclipped = ratio * adv
            surr_clipped = min(max(ratio, low), high) * adv
            policy_scale = (adv * ratio) if surr_unclipped <= surr_clipped else 0.0

        dh = np.zeros(self.hidden, dtype=np.float64)
        for head_index, (head, p) in enumerate(zip(self._head_names, probs)):
            value = int(picked[head_index])
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
            dlogits = policy_scale * (_onehot(value, p.size) - p)
            dlogits += -float(entropy_coef) * p * (np.log(p + 1e-12) + entropy)
            dlogits = dlogits / self.temperature          # z = logits / T
            grads[f"W2_{head}"] += np.outer(dlogits, h)
            grads[f"b2_{head}"] += dlogits
            dh += self.params[f"W2_{head}"].T @ dlogits

        if value_term is not None:
            value, ret, grad_scale = value_term
            wv = self.params["Wv"][0]
            dvalue = -float(value_coef) * (value - float(ret)) * float(grad_scale)
            grads["Wv"][0] += dvalue * h
            grads["bv"][0] += dvalue
            dh += dvalue * wv

        dz = dh * (1.0 - h * h)
        grads["W1"] += np.outer(dz, x)
        grads["b1"] += dz
        return logp_total


    def ppo_gradients(
        self,
        observations: Sequence[Sequence[float]],
        actions: Sequence,
        old_logp: Sequence[float],
        advantages: Sequence[float],
        returns: Sequence[float],
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        old_values: Sequence[float] | None = None,
        value_clip: float = 0.0,
        masks: Sequence[Sequence[Sequence[bool]]] | None = None,
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

        v2：``actions`` 是 ``(T, H)``，策略梯度逐头独立（factorized 的乘积分布），
        熵项逐头求和；ratio 仍建立在**联合** logp 上（这才是 PPO 的裁剪对象）。
        """
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        n = len(observations)
        if n == 0:
            return grads
        low = 1.0 - float(clip)
        high = 1.0 + float(clip)
        clip_value = float(value_clip or 0.0)
        old_v = None if old_values is None else np.asarray(old_values, dtype=np.float64).reshape(-1)

        if self.is_v2:
            wv = self.params["Wv"][0]
            for index, (x, a, old_lp, adv, ret) in enumerate(
                    zip(observations, actions, old_logp, advantages, returns)):
                x_arr = np.asarray(x, dtype=np.float64).reshape(self.obs_dim)
                value = float(wv @ self.hidden_of(x_arr) + self.params["bv"][0])
                grad_scale = 1.0
                if clip_value > 0.0 and old_v is not None and index < old_v.size:
                    anchor = float(old_v[index])
                    clipped_value = anchor + float(np.clip(value - anchor, -clip_value, clip_value))
                    if (clipped_value - float(ret)) ** 2 > (value - float(ret)) ** 2:
                        grad_scale = 0.0
                self._accumulate_v2(
                    grads, x_arr, a, float(adv),
                    None if masks is None else masks[index],
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    clip=(low, high),
                    old_logp=float(old_lp),
                    value_term=(value, float(ret), grad_scale),
                )
            for k in grads:
                grads[k] /= float(n)
            return grads

        wv = self.params["Wv"][0]
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
                # 符号必须是**正的**（J 的上升方向）。旧实现写成 -(adv·ratio)·(onehot−p)，
                # 等价于沿 ∇J 反方向走（实测：正 advantage 会让被选中的动作概率下降）。
                dlogits = (adv * ratio) * (_onehot(a, self.n_actions) - p)
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
            if self.weight_decay > 0 and (k == "W1" or k.startswith("W2")):
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
        meta = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "updates": self.updates,
            # P2.3：温度随模型走（续训沿用），旧文件缺省 1.0。
            "temperature": self.temperature,
            **(extra or {}),
        }
        if self.is_v2:
            # 只给 v2 写这两个键：v1 的 _meta 保持与旧版逐字节一致（老文件里没有它们）。
            meta["actionSpace"] = "v2"
            meta["headSizes"] = list(self._head_sizes)
            meta["headNames"] = list(self._head_names)
        payload["_meta"] = np.array(json.dumps(meta))
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str, obs_dim: int | None = None) -> "Policy":
        """加载策略；``obs_dim`` 更大时把旧 W1 按**原索引**扩展（右侧补零列）。

        兼容性（老模型必须能直接接着练）：
        - 旧 ``.npz`` 没有价值头 ``Wv/bv`` → 随机初始化（critic 从零学，不污染 actor）；
        - 旧 ``obs_dim`` 小于当前环境 → ``W1`` 补零列，旧特征仍落在**原来的下标**上。
          观测新增字段一律追加在 ``OBS_FIELDS`` 末尾（见 spaces.py），所以补零只能补在右侧；
          若补在左侧，旧特征会整体右移，权重语义会被破坏；
        - 旧 ``.npz`` 没有 ``actionSpace`` → 视为 ``v1``（v1 是历史默认）；
          v2 模型带 ``actionSpace="v2"`` 与 ``headSizes``，加载时按它重建这批头
          （G-707 之后是 10 个；9 头的老 v2 模型按文件里的 ``headSizes`` 重建 9 个，
          再由 :func:`adopt_weights` 增量接管到 10 头）。
        """
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["_meta"]))
            stored_dim = int(meta["obs_dim"])
            target_dim = int(obs_dim) if obs_dim is not None else stored_dim
            action_space = str(meta.get("actionSpace") or "v1").strip().lower()
            head_sizes = meta.get("headSizes")
            head_names = meta.get("headNames")
            policy = cls(
                obs_dim=target_dim,
                n_actions=int(meta["n_actions"]),
                hidden=int(meta["hidden"]),
                action_space=action_space,
                head_sizes=head_sizes,
                head_names=head_names,
            )
            for k in policy.param_names():
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


def load_for_obs_dim(path: str, obs_dim: int) -> Policy:
    """加载策略，并把**旧观测维度**右侧补零扩到 ``obs_dim``。

    为什么需要单独一个入口：``Policy.load`` 默认保持文件里的 ``obs_dim``（续训语义），
    于是 G-702 之前的 23 维模型直接拿来评估会在 ``forward`` 里 reshape 失败
    （``cannot reshape array of size 25 into shape (23,)``）。评估脚本要的是「用当前
    观测喂旧模型」，口径与 ``--init-policy`` 一致：旧特征留在原下标，新列补零。
    """
    loaded = Policy.load(path)
    if loaded.obs_dim >= int(obs_dim):
        return loaded
    return Policy.load(path, obs_dim=int(obs_dim))


def check_compat(policy: "Policy", *, obs_dim: int, n_actions: int, hidden: int,
                 action_space: str = "v1") -> List[str]:
    """返回不匹配项（空列表=兼容）。供 ``--init-policy`` 的启动校验复用，便于单测。"""
    problems: List[str] = []
    # 旧模型 obs_dim 更小是**允许**的：加载时右侧补零列即可接着练（P1 加的 4 个观测字段）。
    # 只有「模型比环境宽」无法自动降维，才报错。
    if int(policy.obs_dim) > int(obs_dim):
        problems.append(f"obs_dim {policy.obs_dim} > {obs_dim}（无法降维）")
    want_space = str(action_space or "v1").strip().lower()
    if str(getattr(policy, "action_space", "v1")) != want_space:
        problems.append(
            f"动作空间 {getattr(policy, 'action_space', 'v1')} != {want_space}"
            "（v1/v2 的参数结构不同，无法互转：重训，或用 --action-space "
            f"{getattr(policy, 'action_space', 'v1')}）")
        return problems
    if int(policy.hidden) != int(hidden):
        problems.append(f"hidden {policy.hidden} != {hidden}")
    if policy.is_v2:
        # v2 的动作维度由头大小决定，n_actions 是求和值，只需确认头结构没变。
        from gladiator_env.spaces import ACTION_HEADS_V2

        want = list(ACTION_HEADS_V2.items())
        if policy.head_sizes() != [size for _name, size in want]:
            if _heads_extend(want, _head_pairs(policy)):
                # G-707：9 头的老 v2 模型可以增量接管到 10 头（新头 Xavier 初始化），
                # 所以这里**不算**不兼容；真正无法合并的是「头更多 / 头名不同 / 改了大小」。
                pass
            else:
                problems.append(
                    f"v2 头结构 {_head_pairs(policy)} != 当前 {want}"
                    "（不是可前缀扩展的结构，需重训）")
    elif int(policy.n_actions) != int(n_actions):
        problems.append(f"n_actions {policy.n_actions} != {n_actions}")
    return problems


def validate_init_policy(path: str, *, obs_dim: int, n_actions: int,
                         hidden: int, action_space: str = "v1") -> Tuple[Policy | None, str | None]:
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
    problems = check_compat(loaded, obs_dim=obs_dim, n_actions=n_actions, hidden=hidden,
                            action_space=action_space)
    if problems:
        return None, "初始策略与当前环境不匹配: " + ", ".join(problems)
    return loaded, None


def adopt_weights(policy: Policy, loaded: Policy) -> Policy:
    """把已校验的权重装进本次策略：只接管参数，**不接管 optimizer / updates 状态**。

    动作空间转换（ADR 之外的口径，见 ``docs/training.md``）：

    - ``v1 → v1`` / ``v2 → v2``：逐参数全量接管；
    - ``v1 → v2``：**允许**。``W1/b1``（表征）与 ``Wv/bv``（critic）继承，
      ``W2/b2`` 丢弃并重新随机初始化——v1 的单头 logits 和各头没有对应关系，
      硬塞只会得到垃圾动作分布，不如让新头从 Xavier 起点重新学；
    - ``v2(H=9) → v2(H=10)``：**允许**（G-707 的 ``sprint_reset`` 追加在末尾）。
      只搬**同名**的头，新加的头保留构造函数里的 Xavier 初始化，因此「加一个头」不需要重训
      已有部分；反过来（目标比模型少头，或头名对不上）仍然拒绝——那是真的结构不兼容；
    - ``v2 → v1``：**拒绝**。H 个头无法压回一个 11 路 softmax（信息不可逆），
      报错让调用方显式重训。
    """
    source_space = str(getattr(loaded, "action_space", "v1"))
    target_space = str(getattr(policy, "action_space", "v1"))
    if source_space == "v2" and target_space == "v1":
        raise ValueError(
            f"不能把 v2（{len(getattr(loaded, '_head_sizes', []))} 头）模型加载进 v1（单头）："
            f"{len(getattr(loaded, '_head_sizes', []))} 个离散头无法合并成一个 11 路 softmax。"
            "请用 --action-space v2 续训，或换一个 v1 模型。")
    if source_space == "v1" and target_space == "v2":
        for key in ("W1", "b1", "Wv", "bv"):
            if key not in loaded.params:
                continue
            value = np.array(loaded.params[key], dtype=np.float64)
            if key == "W1" and value.shape[1] != policy.obs_dim:
                value = _pad_obs_columns(value, policy.obs_dim)
            policy.params[key] = value
        policy.temperature = float(getattr(loaded, "temperature", 1.0) or 1.0)
        return policy
    if target_space == "v2" and policy.head_sizes() != loaded.head_sizes():
        if not _heads_extend(_head_pairs(policy), _head_pairs(loaded)):
            raise ValueError(
                f"v2 头结构不一致：模型 {loaded.head_sizes()} vs 当前 {policy.head_sizes()}"
                "（只有「目标 = 模型头 + 追加在末尾的新头」才能增量接管，否则需重训）")
    for key in policy.param_names():
        if key not in loaded.params:
            # 追加的新头（例如 G-707 的 sprint_reset）保留构造函数里的 Xavier 初始化。
            continue
        value = np.array(loaded.params[key], dtype=np.float64)
        if key == "W1" and value.shape[1] != policy.obs_dim:
            value = _pad_obs_columns(value, policy.obs_dim)
        policy.params[key] = value
    # P2.3：温度也随模型走（--logit-temp 显式给了的话，调用方会在 adopt 之后覆盖）。
    policy.temperature = float(getattr(loaded, "temperature", 1.0) or 1.0)
    return policy


def _head_pairs(owner: "Policy") -> List[Tuple[str, int]]:
    """``[(头名, 头大小), ...]``（顺序即头顺序）。"""
    return list(zip(owner.head_names(), owner.head_sizes()))


def _heads_extend(target: Sequence[Tuple[str, int]],
                  source: Sequence[Tuple[str, int]]) -> bool:
    """``source`` 是否是 ``target`` 的**严格前缀**（同名 + 同大小）。

    G-707 的 9→10 就是这种情形：``[("move",5),…,("slot",9)]`` 是
    ``[…, ("slot",9), ("sprint_reset",3)]`` 的前缀。前缀之外的排列（换头名、改大小、
    目标比模型还少）都不接受——那些情况下「按名字搬」得到的语义是错的。
    """
    target = list(target)
    source = list(source)
    if len(source) >= len(target):
        return False
    for index, item in enumerate(source):
        if target[index] != item:
            return False
    return True


def act_with_mask(policy: "Policy", obs: Sequence[float], rng: np.random.Generator,
                  mask: Sequence[Sequence[bool]] | None = None,
                  greedy: bool = False):
    """统一的动作选择：v1 → ``act()``（返回 int），v2 → ``sample()``（返回 (H,) 数组）。

    训练循环与评估脚本都走它，避免各处重复 ``if policy.is_v2``。
    """
    if getattr(policy, "is_v2", False):
        return policy.sample(obs, rng, mask=mask, greedy=greedy)
    return policy.act(obs, rng, greedy=greedy)


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
