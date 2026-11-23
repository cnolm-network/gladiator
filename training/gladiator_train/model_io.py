"""模型加载与动作分发：v1 / v2 混打时唯一该看的地方（G-703）。

为什么单独抽一个模块：``league.py`` / ``mirror_eval.py`` / ``eval_suite.py`` 都要回答同样三个问题——
「这个 npz 是什么动作空间」「学习者该调 ``act`` 还是 ``sample``」「对手给 int 还是给 10 个头」。
写在三个脚本里必然漂移（v2 刚出来时 league/mirror 就是各自一份 ``!= "v1"`` 的硬拒，
导致最强的 v2 模型没法进联赛）。

口径（与 ``gladiator_env`` 一致，别在这里另立一套）：

- **v1**（单头 softmax，11 个宏动作）：``Policy.act()`` → ``int``（宏动作编号）；
- **v2**（十头 ``MultiDiscrete``）：``Policy.sample()`` → ``(H,)`` 数组，带 action mask
  （没给 mask 就不掩码，等同于「全开」）；
- 学习者的动作空间决定 ``GladiatorEnv(action_space=...)``；**对手的动作空间不需要另传**——
  env 自己读 ``opponent_policy.is_v2`` / ``action_space`` 得到 ``_opponent_v2``，
  再决定第 1 号 Bot 的动作走 ``decode_intent`` 还是 v1 宏动作。所以
  「v1 学习者 + v2 对手」「v2 学习者 + v1 对手」「v2 + v2」三种组合都不需要额外开关。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from gladiator_env.spaces import OBS_DIM

from .policy import Policy, act_with_mask

#: 已知的动作空间名。
ACTION_SPACES = ("v1", "v2")

#: 这些 stem 太通用（`.../models/<名字>/policy.npz` 全是 `policy`），标签改用父目录名。
GENERIC_STEMS = frozenset({"policy", "model", "weights", "final", "best"})


def model_label(path: str | Path) -> str:
    """模型的短标签：stem 太通用时用父目录名（``v2-v4-dagger3`` 而不是 ``policy``）。"""
    file = Path(path)
    stem = file.stem
    return stem if stem.lower() not in GENERIC_STEMS else (file.parent.name or stem)


class ModelLoadError(ValueError):
    """模型文件不存在 / 不是合法的 Policy.npz / 动作空间不认识。

    脚本层把它转成 ``SystemExit``（带路径与原因），测试层直接断言异常类型与消息。
    """


@dataclass(frozen=True)
class ModelRecord:
    """一个加载好的模型 + 它的动作空间元信息。"""

    path: str
    label: str
    policy: Any
    action_space: str
    head_sizes: Tuple[int, ...]
    obs_dim: int
    #: 文件里原本的观测维度（旧模型会比当前 ``OBS_DIM`` 窄，加载时右侧补零）。
    native_obs_dim: int = 0

    @property
    def is_v2(self) -> bool:
        return self.action_space == "v2"

    def obs_label(self) -> str:
        """``37`` 或 ``25→37``（补过零时把原维度也写出来，别让人以为模型是 37 维训练的）。"""
        if self.native_obs_dim and self.native_obs_dim != self.obs_dim:
            return f"{self.native_obs_dim}→{self.obs_dim}"
        return str(self.obs_dim)

    def describe(self) -> str:
        """一行摘要（日志/表格用）：``dagger2 [v2 5/7/5/2/3/2/2/2/9/3]``。"""
        sizes = "/".join(str(size) for size in self.head_sizes)
        return f"{self.label} [{self.action_space} {sizes}]"


def load_model(path: str | Path, *, label: str | None = None,
               obs_dim: int | None = None) -> ModelRecord:
    """加载一个模型并读出动作空间（旧观测维度的权重右侧补零到当前环境）。

    为什么默认就补零：环境现在的观测是 ``OBS_DIM``（37）维，而现役模型有 25/23 维的。
    ``Policy.load`` 默认**保持文件里的维度**（续训语义），直接喂 37 维观测会在
    reshape 处炸掉；评估/对局要的是「用当前观测喂旧模型」，所以这里用
    ``Policy.load(path, obs_dim=当前环境)``（= ``load_for_obs_dim`` 的口径：旧特征留在原下标，
    新列补零，只补不截）。

    :raises ModelLoadError: 文件不存在 / 不是合法 npz / ``actionSpace`` 不认识
    """
    file = Path(path)
    if not file.is_file():
        raise ModelLoadError(f"模型不存在: {file}")
    try:
        with np.load(str(file), allow_pickle=True) as data:
            meta = json.loads(str(data["_meta"]))
    except Exception as exc:  # noqa: BLE001 - 把底层 zip/npy/结构错误统一成一句话
        raise ModelLoadError(
            f"模型无法加载（不是 Policy.save 产出的 .npz？）: {file}: "
            f"{type(exc).__name__}: {exc}") from exc
    try:
        native_obs_dim = int(meta["obs_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"模型 _meta 里没有合法的 obs_dim: {file}") from exc
    target = OBS_DIM if obs_dim is None else int(obs_dim)
    try:
        # 只补不截：文件比环境宽时保持原样（那种情况下调用方该自己对齐环境）。
        policy = Policy.load(str(file), obs_dim=max(int(target), native_obs_dim))
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"模型无法加载（权重形状与 _meta 不一致？）: {file}: "
            f"{type(exc).__name__}: {exc}") from exc
    space = str(getattr(policy, "action_space", "v1") or "v1").strip().lower()
    if space not in ACTION_SPACES:
        raise ModelLoadError(
            f"模型动作空间不认识: {file} actionSpace={space!r}（允许 {ACTION_SPACES}）")
    sizes = tuple(int(size) for size in policy.head_sizes())
    if not sizes or any(size <= 0 for size in sizes):
        raise ModelLoadError(f"模型头结构非法: {file} headSizes={sizes}")
    obs_dim = int(getattr(policy, "obs_dim", 0) or 0)
    if obs_dim <= 0:
        raise ModelLoadError(f"模型 obsDim 非法: {file} obsDim={obs_dim}")
    return ModelRecord(path=str(file), label=label or file.stem, policy=policy,
                       action_space=space, head_sizes=sizes, obs_dim=obs_dim,
                       native_obs_dim=native_obs_dim)


def normalize_action_mask(policy, mask):
    """把 env 的 ``actionMask``（dict / 嵌套 dict / 已归一列表）规范成 ``list[list[bool]]``。

    没给 mask 或策略不支持（v1）时返回 ``None``——v1 没有 mask 概念，不要凭空造一个。
    """
    if mask is None:
        return None
    normalizer = getattr(policy, "normalize_mask", None)
    if normalizer is None:
        return None
    return normalizer(mask)


def policy_action_for_env(model, obs: Sequence[float], mask=None, rng=None,
                          greedy: bool = False):
    """按策略自己的动作空间给出 **env.step() 能直接吃** 的动作。

    - v1 → ``int``（宏动作编号，0..10）；
    - v2 → ``list[int]``（长度 = 头数，顺序 = ``head_sizes``）。

    ``model`` 可以是 :class:`ModelRecord`，也可以是裸 ``Policy``（测试/临时对象方便）；
    ``mask`` 只在 v2 下有意义，形态随便（``info["actionMask"]`` 原样传进来即可）。
    """
    policy = model.policy if isinstance(model, ModelRecord) else model
    if rng is None:
        rng = np.random.default_rng(0)
    is_v2 = bool(getattr(policy, "is_v2", False))
    if not is_v2:
        return int(policy.act(obs, rng, greedy=greedy))
    heads = act_with_mask(policy, obs, rng, mask=normalize_action_mask(policy, mask),
                          greedy=greedy)
    picked = [int(value) for value in np.asarray(heads).reshape(-1)]
    expected = len(policy.head_sizes())
    if len(picked) != expected:
        # 到这一步说明 mask/权重与动作空间对不上：宁可炸掉，也不要把错长度的数组发给 env。
        raise ValueError(
            f"v2 动作头数不符：策略给 {len(picked)} 个头，head_sizes={policy.head_sizes()}")
    return picked


def env_action_space(model) -> str:
    """学习者用哪个动作空间建 env（对手的由 env 自己判定）。"""
    if isinstance(model, ModelRecord):
        return model.action_space
    policy = model
    return str(getattr(policy, "action_space", "v1") or "v1").strip().lower()


def action_space_table(records) -> dict:
    """``label -> actionSpace``；给表格/JSON 用（顺序按传入顺序）。"""
    table: dict[str, str] = {}
    for record in records:
        table[record.label] = record.action_space
    return table
