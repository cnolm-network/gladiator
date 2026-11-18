"""权重合并 v2：对齐（REBASIN）→ 合并（TIES / Fisher）→ 门禁 → 蒸馏兜底。

**为什么需要对齐**：``Policy`` 是单隐藏层 ``tanh`` 网络，隐藏单元的顺序不影响网络
输出（把第 i 个单元与第 j 个单元的入边、偏置、出边一起换，函数完全不变），所以同一
个策略有 ``hidden!`` 种等价参数化（64 个单元时约 ``1.27e89`` 种）。不同场地各自
``Policy(seed=...)`` 随机初始化后训练，落到的是**不同坐标系**；逐元素平均会把
「左转」和「右转」的神经元加在一起，合并结果比任何单支都差。

处理链（``merge`` 的四个 method）：

1. ``mean``          —— 加权平均（旧行为，仅在**同一坐标系**下有意义）；
2. ``ties``          —— 逐张量 TIES：幅值裁剪 → 符号投票 → 同号项加权平均；
3. ``rebasin-ties``  —— 先用校准观测把每支的隐藏单元对齐到第一支，再做 TIES；
4. ``rebasin-fisher``—— 对齐后按 Fisher 对角加权平均（参数重要的地方权重更大）。

纯 numpy 实现，不引入 scipy：匈牙利算法自己写（64 个单元，O(n³) 足够快）。

约定：

- 所有函数都**不修改入参**（内部深拷贝），返回新的 ``dict[str, ndarray]``；
- 参数布局与 ``Policy`` 一致：``W1 (hidden, obs)``、``b1 (hidden,)``、
  ``W2 (n_actions, hidden)``、``b2 (n_actions,)``、``Wv (1, hidden)``、``bv (1,)``。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .eval_suite import SIGNIFICANCE

#: 与 ``Policy.PARAM_NAMES`` 同序，只用于遍历。
PARAM_ORDER = ("W1", "b1", "W2", "b2", "Wv", "bv")

#: 参与「隐藏单元置换」的参数：``b2``/``bv`` 与隐藏顺序无关，不能动。
_PERMUTED_KEYS = ("W1", "b1", "W2", "Wv")

MERGE_METHODS = ("mean", "ties", "rebasin-ties", "rebasin-fisher")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def params_of(obj: Any) -> Dict[str, np.ndarray]:
    """接受 ``Policy`` 或裸 params dict，统一取出参数表（不做拷贝）。"""
    if hasattr(obj, "params"):
        return dict(obj.params)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"无法从 {type(obj).__name__} 取参数（需要 Policy 或 dict）")


def copy_params(params: Any) -> Dict[str, np.ndarray]:
    """深拷贝参数表（float64），保证后续操作不碰输入。"""
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params_of(params).items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def hidden_features(params: Any, obs: Sequence[float]) -> np.ndarray:
    """``tanh(W1·x + b1)``：返回 ``(N, hidden)``。

    这是对齐用的「指纹」：同一套观测过网络时，坐标对齐的两个单元输出高度相关。
    ``obs`` 可以是单条 ``(obs_dim,)``，也可以是 ``(N, obs_dim)``。
    """
    table = params_of(params)
    w1 = np.asarray(table["W1"], dtype=np.float64)
    b1 = np.asarray(table["b1"], dtype=np.float64)
    matrix = np.asarray(obs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"obs 必须是 1D 或 2D，收到形状 {matrix.shape}")
    if matrix.shape[1] != w1.shape[1]:
        raise ValueError(
            f"obs 维度 {matrix.shape[1]} 与 W1 的输入维度 {w1.shape[1]} 不一致"
        )
    return np.tanh(matrix @ w1.T + b1)


def _standardize(feats: np.ndarray) -> np.ndarray:
    """按列去均值、除标准差；常数单元（std≈0）整列置 0（相似度恒为 0）。"""
    matrix = np.asarray(feats, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"feats 必须是 2D，收到形状 {matrix.shape}")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    out = np.zeros_like(matrix)
    live = std > 1e-12
    if np.any(live):
        out[:, live] = (matrix[:, live] - mean[live]) / std[live]
    return out


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行向量两两余弦相似度；零向量（常数单元）对任何单元的相似度记 0。"""
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = np.outer(norm_a, norm_b)
    sim = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    usable = denom > 1e-12
    if np.any(usable):
        sim[usable] = (a @ b.T)[usable] / denom[usable]
    return sim


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """经典 Kuhn–Munkres（JV 变体）最小代价指派，返回 ``row -> col``。

    纯 Python 实现，O(n³)；方阵、代价有限（这里来自余弦相似度，不会溢出）。
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"匈牙利算法需要方阵，收到形状 {matrix.shape}")
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    inf = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    owner = np.zeros(n + 1, dtype=int)   # owner[j] = 该列被哪一行占用（1-based）
    way = np.zeros(n + 1, dtype=int)
    for row in range(1, n + 1):
        owner[0] = row
        col0 = 0
        min_value = np.full(n + 1, inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = owner[col0]
            delta = inf
            next_col = -1
            for col in range(1, n + 1):
                if used[col]:
                    continue
                current = matrix[row0 - 1, col - 1] - u[row0] - v[col]
                if current < min_value[col]:
                    min_value[col] = current
                    way[col] = col0
                if min_value[col] < delta:
                    delta = min_value[col]
                    next_col = col
            if next_col < 0:  # 理论上不会发生：所有列都被访问过
                raise RuntimeError("匈牙利算法失败：找不到可增广列")
            for col in range(0, n + 1):
                if used[col]:
                    u[owner[col]] += delta
                    v[col] -= delta
                else:
                    min_value[col] -= delta
            col0 = next_col
            if owner[col0] == 0:
                break
        while True:  # 增广：沿 way 反向改派
            prev = way[col0]
            owner[col0] = owner[prev]
            col0 = prev
            if col0 == 0:
                break
    assignment = np.zeros(n, dtype=int)
    for col in range(1, n + 1):
        if owner[col] > 0:
            assignment[owner[col] - 1] = col - 1
    return assignment


def match_permutation(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """把 B 的隐藏单元匹配到 A 的下标：返回 ``perm``，``perm[k]`` = A 的第 k 个单元对应 B 的哪一列。

    用法是 ``apply_permutation(params_b, perm)`` 之后 B 就落在 A 的坐标系里。

    相似度前各自**按列去均值、除标准差**：tanh 单元常常是「几乎常开」或「几乎常关」，
    不去标准化的话，这类常数单元的绝对值会主导相似度矩阵，把真正的对应关系冲掉。

    校准样本至少 3 条（2 条时每个单元标准化后只剩 ±1 两种模式，匹配等于随机）；
    实测 4 条起就能 100% 恢复精确置换，但真实模型建议给 ≥ 32 条（或用 BC 数据）。
    """
    # 列才是「单元」：(N, hidden) → 转置成 (hidden, N)，按单元算相似度。
    a = _standardize(feats_a).T
    b = _standardize(feats_b).T
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"隐藏单元数不一致：A={a.shape[1]}，B={b.shape[1]}（无法对齐，需同结构模型）"
        )
    if a.shape[0] == 0:
        raise ValueError("隐藏单元为空，无法对齐")
    if a.shape[1] < 3:
        # 2 条样本时每个单元标准化后只剩 ±1/√2 两种模式，几乎全部相同 → 匹配变成随机。
        raise ValueError(
            f"校准样本只有 {a.shape[1]} 条：至少需要 3 条才能按单元标准化（建议 ≥ 32）"
        )
    similarity = _cosine_matrix(a, b)
    # 匈牙利求最小代价 → 用 -similarity 变成「相似度最大」的指派。
    return _hungarian(-similarity)


def apply_permutation(params: Any, perm: Sequence[int]) -> Dict[str, np.ndarray]:
    """按 ``perm`` 重排隐藏单元，返回**新**参数表（输入不被修改）。

    ``perm[k] = j`` 表示「新参数的第 k 个单元取原参数的第 j 个单元」。因此：

    - ``W1' = W1[perm, :]``、``b1' = b1[perm]``（入边与偏置按单元取行）；
    - ``W2' = W2[:, perm]``（出边按单元取列）；
    - ``Wv' = Wv[:, perm]``（**价值头必须同 perm 变换**：它是 ``h`` 的线性函数，
      h 换了顺序而 Wv 不换，价值预测就彻底错位——只影响 PPO 的 critic 与
      ``--init-policy`` 续训，不影响 ``act``，因此很容易被漏掉）；
    - ``b2``/``bv`` 与单元顺序无关，原样保留。
    """
    source = copy_params(params)
    indices = np.asarray(perm, dtype=int).reshape(-1)
    hidden = np.asarray(source["W1"]).shape[0]
    if indices.size != hidden:
        raise ValueError(f"perm 长度 {indices.size} 与隐藏单元数 {hidden} 不一致")
    if sorted(int(i) for i in indices) != list(range(hidden)):
        raise ValueError("perm 必须是 0..hidden-1 的一个排列（不能重复/越界）")
    out = dict(source)
    for key in _PERMUTED_KEYS:
        if key not in source:
            continue
        value = np.array(source[key], dtype=np.float64, copy=True)
        if key in ("W1", "b1"):
            out[key] = value[indices, ...]
        else:  # W2 / Wv：(..., hidden) → 按最后一维取列
            out[key] = value[..., indices]
    return out


# ---------------------------------------------------------------------------
# 合并算法
# ---------------------------------------------------------------------------

def _normalized_weights(count: int, weights: Optional[Sequence[float]]) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    values = np.asarray([float(w) for w in weights], dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError(f"weights 长度 {values.size} 与模型数量 {count} 不一致")
    if np.any(values < 0):
        raise ValueError("weights 不能为负")
    total = float(values.sum())
    if total <= 0:
        return np.full(count, 1.0 / count, dtype=np.float64)
    return values / total


def _stack(param_dicts: Sequence[Any], key: str) -> np.ndarray:
    values = [np.asarray(params_of(item)[key], dtype=np.float64) for item in param_dicts]
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{key} 的形状不一致: {sorted(shapes)}")
    return np.stack(values, axis=0)


def ties_merge(param_dicts: Sequence[Any], weights: Optional[Sequence[float]] = None,
               trim: float = 0.2) -> Dict[str, np.ndarray]:
    """TIES 合并：幅值裁剪 → 符号投票 → 同号项（加权）平均。

    ``trim=0`` 退化为加权平均（不裁剪、不做符号筛选）。

    三个模型在某元素上取 ``[1.0, 1.0, -0.2]``、``trim=0.2`` 时：幅值分位阈值恰为
    1.0，``-0.2`` 被裁掉，符号投票为 +1，结果 1.0 —— 这正是 TIES 想要的
    「少数派小扰动不参与平均」。
    """
    if not param_dicts:
        raise ValueError("ties_merge 至少需要一个模型")
    trim = float(trim)
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim 必须在 [0, 1)，收到 {trim}")
    weight = _normalized_weights(len(param_dicts), weights)
    reference = params_of(param_dicts[0])
    out: Dict[str, np.ndarray] = {}
    for key, value in reference.items():
        stacked = _stack(param_dicts, key)                       # (M, ...)
        if trim > 0:
            magnitude = np.abs(stacked)
            threshold = np.quantile(magnitude, 1.0 - trim, axis=0)
            keep = magnitude >= threshold
        else:
            keep = np.ones_like(stacked, dtype=bool)
        trimmed = np.where(keep, stacked, 0.0)
        signs = np.sign(trimmed)
        shape = (-1,) + (1,) * (stacked.ndim - 1)
        score = np.sum(signs * weight.reshape(shape), axis=0)
        chosen = np.sign(score)
        if trim <= 0:
            # 退化为加权平均：不做符号筛选（全负/正负混合都照原样平均）。
            out[key] = np.sum(stacked * weight.reshape(shape), axis=0)
            continue
        agree = keep & (signs == chosen)
        numer = np.sum(trimmed * weight.reshape(shape) * agree, axis=0)
        denom = np.sum(weight.reshape(shape) * agree, axis=0)
        merged = np.zeros_like(numer)
        live = denom > 0
        merged[live] = numer[live] / denom[live]
        out[key] = merged
    return out


def fisher_merge(param_dicts: Sequence[Any],
                 fisher_dicts: Sequence[Any]) -> Dict[str, np.ndarray]:
    """按 Fisher 对角加权平均：权重 = 各模型在该元素上的 Fisher 值。

    某元素上所有模型的 Fisher 都为 0（这类参数对似然没有影响）时退回等权平均。
    """
    if not param_dicts:
        raise ValueError("fisher_merge 至少需要一个模型")
    if len(fisher_dicts) != len(param_dicts):
        raise ValueError(
            f"fisher 数量 {len(fisher_dicts)} 与模型数量 {len(param_dicts)} 不一致"
        )
    reference = params_of(param_dicts[0])
    out: Dict[str, np.ndarray] = {}
    for key, value in reference.items():
        stacked = _stack(param_dicts, key)
        fisher = _stack(fisher_dicts, key)
        fisher = np.clip(np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        total = np.sum(fisher, axis=0)
        numer = np.sum(stacked * fisher, axis=0)
        merged = np.mean(stacked, axis=0)
        live = total > 0
        merged = np.where(live, numer / np.where(live, total, 1.0), merged)
        out[key] = np.asarray(merged, dtype=np.float64)
    return out


def fisher_information(policy: Any, samples: Sequence[Tuple[Sequence[float], int]],
                       obs_dim: Optional[int] = None,
                       n_actions: Optional[int] = None) -> Dict[str, np.ndarray]:
    """经验 Fisher 对角（softmax 对数似然的梯度平方均值）。

    ``samples`` 是 ``(obs, action)`` 序列；只算 actor 参数（``W1/b1/W2/b2``），
    价值头（``Wv/bv``）返回全 0（``fisher_merge`` 对 0 权重会退回等权）。

    对角 Fisher = ``E[(∂ log π(a|x) / ∂θ)²]``，逐元素估计，不需要完整 Hessian。
    """
    table = params_of(policy)
    w1 = np.asarray(table["W1"], dtype=np.float64)
    b1 = np.asarray(table["b1"], dtype=np.float64)
    w2 = np.asarray(table["W2"], dtype=np.float64)
    b2 = np.asarray(table["b2"], dtype=np.float64)
    dim = int(obs_dim) if obs_dim is not None else w1.shape[1]
    actions = int(n_actions) if n_actions is not None else w2.shape[0]
    temperature = float(getattr(policy, "temperature", 1.0) or 1.0)

    acc = {
        "W1": np.zeros_like(w1), "b1": np.zeros_like(b1),
        "W2": np.zeros_like(w2), "b2": np.zeros_like(b2),
    }
    count = 0
    for obs, action in samples:
        index = int(action)
        if index < 0 or index >= actions:
            raise ValueError(f"动作下标越界: {index}（动作数 {actions}）")
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        if x.size < dim:
            x = np.pad(x, (0, dim - x.size))
        elif x.size > dim:
            x = x[:dim]
        h = np.tanh(w1 @ x + b1)
        probs = _softmax((w2 @ h + b2) / temperature)
        ds = -probs
        ds[index] += 1.0
        ds /= temperature
        grad_w2 = np.outer(ds, h)
        grad_b2 = ds
        dz = (w2.T @ ds) * (1.0 - h * h)
        grad_w1 = np.outer(dz, x)
        grad_b1 = dz
        acc["W1"] += grad_w1 * grad_w1
        acc["b1"] += grad_b1 * grad_b1
        acc["W2"] += grad_w2 * grad_w2
        acc["b2"] += grad_b2 * grad_b2
        count += 1
    if count == 0:
        raise ValueError("Fisher 估计需要至少一个 (obs, action) 样本")
    out = {key: value / count for key, value in acc.items()}
    for key in ("Wv", "bv"):
        if key in table:
            out[key] = np.zeros_like(np.asarray(table[key], dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def merge(params_list: Sequence[Any], method: str = "mean",
          calibration_obs: Optional[Sequence[float]] = None,
          weights: Optional[Sequence[float]] = None,
          trim: float = 0.2,
          fisher: Optional[Sequence[Any]] = None,
          return_info: bool = False):
    """按 ``method`` 合并，返回新参数表（``return_info=True`` 时返回 ``(params, info)``）。

    ``method``：

    - ``mean``：加权平均（``weights`` 缺省等权）；
    - ``ties``：直接 TIES（要求各支已经在同一坐标系）；
    - ``rebasin-ties``：先用 ``calibration_obs`` 对齐到第一支，再 TIES；
    - ``rebasin-fisher``：对齐后按 Fisher 加权（需要 ``fisher`` 列表，与输入同序）。

    ``info`` 含 ``method``/``perms``（每支的置换，第一支恒为恒等）/``aligned``。
    """
    if method not in MERGE_METHODS:
        raise ValueError(
            f"未知 merge method: {method}（可选: {', '.join(MERGE_METHODS)}）"
        )
    if not params_list:
        raise ValueError("merge 至少需要一个模型")
    copies = [copy_params(item) for item in params_list]
    hidden = np.asarray(copies[0]["W1"]).shape[0]
    for index, item in enumerate(copies):
        if np.asarray(item["W1"]).shape[0] != hidden:
            raise ValueError(
                f"第 {index + 1} 支的隐藏单元数 {np.asarray(item['W1']).shape[0]} "
                f"与第一支 {hidden} 不一致"
            )
    identity = list(range(hidden))
    info: Dict[str, Any] = {"method": method, "perms": [identity], "aligned": False}

    if method == "mean":
        weight = _normalized_weights(len(copies), weights)
        out = {}
        for key in params_of(copies[0]):
            stacked = _stack(copies, key)
            shape = (-1,) + (1,) * (stacked.ndim - 1)
            out[key] = np.sum(stacked * weight.reshape(shape), axis=0)
        return (out, info) if return_info else out

    if method == "ties":
        out = ties_merge(copies, weights=weights, trim=trim)
        return (out, info) if return_info else out

    # ---- rebasin-*：先对齐到第一支 ----
    if calibration_obs is None:
        raise ValueError(f"{method} 需要 calibration_obs（校准观测）来估计隐藏单元对应关系")
    obs = np.asarray(calibration_obs, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    base_features = hidden_features(copies[0], obs)
    aligned = [copies[0]]
    perms: List[List[int]] = [identity]
    for index, item in enumerate(copies[1:], start=1):
        feats = hidden_features(item, obs)
        perm = match_permutation(base_features, feats)
        aligned.append(apply_permutation(item, perm))
        perms.append([int(v) for v in perm])
    info["perms"] = perms
    info["aligned"] = True

    if method == "rebasin-ties":
        out = ties_merge(aligned, weights=weights, trim=trim)
        return (out, info) if return_info else out

    # rebasin-fisher
    if not fisher:
        raise ValueError("rebasin-fisher 需要 fisher 列表（每支一个 Fisher 对角 dict）")
    if len(fisher) != len(copies):
        raise ValueError(f"fisher 数量 {len(fisher)} 与模型数量 {len(copies)} 不一致")
    aligned_fisher = [copy_params(fisher[0])]
    for index in range(1, len(fisher)):
        aligned_fisher.append(apply_permutation(fisher[index], perms[index]))
    out = fisher_merge(aligned, aligned_fisher)
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------
# 门禁判定（纯函数，便于单测）
# ---------------------------------------------------------------------------

def gate_verdict(baseline_summary: dict, merged_summary: dict, p_value: float,
                 allow_regression: bool = False) -> dict:
    """评估门禁：merged 相对 baseline 是否「没有变差」。

    规则（与 docs/training-log.md 的判定口径一致）：

    - ``p < 0.05`` 且 merged 胜率更高 → ``significant-better``（通过）；
    - ``p < 0.05`` 且 merged 胜率更低 → ``significant-worse``（不通过）；
    - 否则要求 ``merged 胜率 ≥ baseline 胜率 − 1/总局数``（一局容差）→ ``non-inferior``；
      达不到就是 ``regression``（不通过）。

    ``allow_regression=True`` 时任何结论都不阻断流程（``passed`` 恒为 True，
    ``status`` 仍如实记录）。
    """
    base = baseline_summary.get("overall", baseline_summary)
    merged = merged_summary.get("overall", merged_summary)
    n_base = int(base.get("episodes") or 0)
    n_merged = int(merged.get("episodes") or 0)
    total = max(n_base, n_merged, 1)
    base_rate = float(base.get("winRate") or 0.0)
    merged_rate = float(merged.get("winRate") or 0.0)
    delta = merged_rate - base_rate
    tolerance = 1.0 / total
    p = float(p_value)

    if p < SIGNIFICANCE and delta > 0:
        status, passed = "significant-better", True
        reason = (f"合并模型显著更好（p={p:.4f} < {SIGNIFICANCE}，"
                  f"胜率 {merged_rate:.1%} vs {base_rate:.1%}）")
    elif p < SIGNIFICANCE and delta < 0:
        status, passed = "significant-worse", False
        reason = (f"合并模型显著更差（p={p:.4f} < {SIGNIFICANCE}，"
                  f"胜率 {merged_rate:.1%} vs {base_rate:.1%}）")
    elif delta >= -tolerance - 1e-12:
        status, passed = "non-inferior", True
        reason = (f"合并模型非劣（胜率 {merged_rate:.1%} vs {base_rate:.1%}，"
                  f"容差 1/{total} = {tolerance:.1%}，p={p:.4f}）")
    else:
        status, passed = "regression", False
        reason = (f"合并模型掉点超过容差（胜率 {merged_rate:.1%} vs {base_rate:.1%}，"
                  f"差 {delta:+.1%}，容差 {tolerance:.1%}，p={p:.4f}）")

    if allow_regression:
        passed = True
        reason += "；已指定 --allow-regression，仅记录不阻断"
    return {
        "passed": bool(passed),
        "status": status,
        "reason": reason,
        "delta": delta,
        "tolerance": tolerance,
        "episodes": total,
        "pValue": p,
        "baselineWinRate": base_rate,
        "mergedWinRate": merged_rate,
        "allowRegression": bool(allow_regression),
    }


# ---------------------------------------------------------------------------
# 蒸馏兜底
# ---------------------------------------------------------------------------

def ensemble_probs(teachers: Sequence[Any], obs: Sequence[float],
                   temperature: float = 1.0) -> np.ndarray:
    """集成老师的概率：各模型 softmax **概率**取平均。

    注意：这里平均的是输出层的概率，**不需要对齐**——隐藏单元置换只改变中间层
    的编号，logits 是输出空间量，天然对置换不变。这也正是蒸馏能当兜底的原因。
    """
    matrix = np.asarray(obs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if not teachers:
        raise ValueError("ensemble 至少需要一个老师")
    total = None
    for teacher in teachers:
        table = params_of(teacher)
        h = np.tanh(matrix @ np.asarray(table["W1"]).T + np.asarray(table["b1"]))
        logits = h @ np.asarray(table["W2"]).T + np.asarray(table["b2"])
        probs = _softmax(logits / float(temperature))
        total = probs if total is None else total + probs
    return total / len(teachers)


def distill(teacher_params: Sequence[Any], obs: Sequence[float],
            init_params: Optional[Any] = None, epochs: int = 6,
            lr: float = 5e-3, batch_size: int = 64, seed: int = 0,
            temperature: float = 1.0, grad_floor: float = 1e-12,
            logger: Optional[Callable[[str], None]] = None) -> Tuple[Dict[str, np.ndarray], dict]:
    """用集成老师的软标签蒸馏一个学生（只训练 actor，价值头沿用初值）。

    ``epochs`` 对应 CLI 的 ``--distill-episodes``（在离线样本上过几遍）；
    目标是最小化 ``KL(p_teacher ‖ p_student)``，梯度 ``∂KL/∂s = (p_s − p_t)/T``。

    **梯度下限**（``grad_floor``）：Adam 的步长与梯度量级无关（``g/√v ≈ ±1``），
    当学生已经等于老师时残差只有 ``1e-17`` 量级的浮点噪声，却会被放大成 ``lr``
    量级的一步，于是「从最优点往外走」（实测 KL 0 → 0.0017）。所以梯度最大值的
    绝对值小于 ``grad_floor`` 时直接跳过该 batch。
    """
    matrix = np.asarray(obs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] == 0:
        raise ValueError("蒸馏需要非空样本")
    target = ensemble_probs(teacher_params, matrix, temperature=temperature)
    params = copy_params(init_params if init_params is not None else teacher_params[0])
    w1 = np.array(params["W1"], dtype=np.float64, copy=True)
    b1 = np.array(params["b1"], dtype=np.float64, copy=True)
    w2 = np.array(params["W2"], dtype=np.float64, copy=True)
    b2 = np.array(params["b2"], dtype=np.float64, copy=True)

    def _forward(batch: np.ndarray):
        h = np.tanh(batch @ w1.T + b1)
        probs = _softmax((h @ w2.T + b2) / float(temperature))
        return h, probs

    def _kl(batch: np.ndarray, probs_target: np.ndarray) -> float:
        _h, probs = _forward(batch)
        clipped = np.clip(probs, 1e-12, 1.0)
        return float(np.mean(np.sum(probs_target * (np.log(np.clip(probs_target, 1e-12, 1.0))
                                                   - np.log(clipped)), axis=1)))

    initial_loss = _kl(matrix, target)
    rng = np.random.default_rng(seed)
    state = {key: (np.zeros_like(arr), np.zeros_like(arr))
             for key, arr in (("W1", w1), ("b1", b1), ("W2", w2), ("b2", b2))}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    skipped = 0
    batch_size = max(1, int(batch_size))
    for epoch in range(max(1, int(epochs))):
        order = rng.permutation(matrix.shape[0])
        for start in range(0, order.size, batch_size):
            index = order[start:start + batch_size]
            batch = matrix[index]
            target_batch = target[index]
            h, probs = _forward(batch)
            ds = (probs - target_batch) / float(temperature)
            grad_w2 = ds.T @ h / len(index)
            grad_b2 = ds.mean(axis=0)
            dz = (ds @ w2) * (1.0 - h * h)
            grad_w1 = dz.T @ batch / len(index)
            grad_b1 = dz.mean(axis=0)
            if grad_floor > 0 and max(
                float(np.max(np.abs(grad))) for grad in (grad_w1, grad_b1, grad_w2, grad_b2)
            ) < float(grad_floor):
                # 已收敛（常见于「学生初值就是老师」）：别让 Adam 把浮点噪声放大成 lr 步。
                skipped += 1
                continue
            step += 1
            for key, grad, arr in (("W1", grad_w1, w1), ("b1", grad_b1, b1),
                                   ("W2", grad_w2, w2), ("b2", grad_b2, b2)):
                first, second = state[key]
                first *= beta1
                first += (1 - beta1) * grad
                second *= beta2
                second += (1 - beta2) * grad * grad
                m_hat = first / (1 - beta1 ** step)
                v_hat = second / (1 - beta2 ** step)
                arr -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if logger is not None:
            logger(f"[distill] epoch {epoch + 1}/{epochs} KL={max(0.0, _kl(matrix, target)):.6f}")

    final_loss = _kl(matrix, target)
    out = dict(params)
    out["W1"] = w1
    out["b1"] = b1
    out["W2"] = w2
    out["b2"] = b2
    info = {
        "teachers": len(teacher_params),
        "samples": int(matrix.shape[0]),
        "epochs": int(epochs),
        "lr": float(lr),
        "steps": int(step),
        "skippedSteps": int(skipped),
        # KL 可能因为浮点误差出现 -1e-9，展示时截到 0。
        "klBefore": max(0.0, initial_loss),
        "klAfter": max(0.0, final_loss),
        "improved": bool(final_loss <= initial_loss + 1e-12),
    }
    return out, info


def params_l2(a: Any, b: Any) -> float:
    """两份参数的 L2 距离（用于 summary 里「每支离合并多远」）。"""
    left = params_of(a)
    right = params_of(b)
    total = 0.0
    for key in left:
        if key not in right:
            continue
        diff = np.asarray(left[key], dtype=np.float64) - np.asarray(right[key], dtype=np.float64)
        total += float(np.sum(diff * diff))
    return math.sqrt(total)
