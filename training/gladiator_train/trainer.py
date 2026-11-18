"""训练循环：在 GladiatorEnv 上跑 PPO（默认）或 REINFORCE。

约定：环境只提供观测/动作/奖励；这里负责采样、优势估计（GAE / 折扣回报）、
优势归一化与参数更新。两种算法的**环境交互完全一致**，只有更新方式不同：
- ``reinforce``：整回合蒙特卡洛回报 + 单次更新（旧行为，保持兼容）；
- ``ppo``：回合内 GAE + 裁剪的重要性采样，多轮（``epochs``）复用同一批数据。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from gladiator_env.review import EpisodeTracker, write_review

from .policy import Policy, discounted_returns, normalize_advantages


#: 目标熵控制器的系系数下限（P2.2）。
ENTROPY_COEF_MIN = 0.005


def explained_variance(values, returns) -> float:
    """``1 - Var(R - V) / Var(R)``：价值头拟合同口径（P2.2 诊断）。

    - 完美预测（V == R）→ 1.0；
    - V 是常数（等于 R 的均值）→ 0.0；
    - 比常数还差 → 负数。
    ``Var(R)`` 为 0（整局回报完全一样）时无定义，按 0.0 返回。
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if v.size == 0 or r.size == 0 or v.shape != r.shape:
        return 0.0
    var_r = float(np.var(r))
    if var_r < 1e-12:
        return 0.0
    return float(1.0 - np.var(r - v) / var_r)


class RunningStats:
    """Welford 在线均值/方差：跨局累计 return 的分布，用于 PPO 的目标归一化。

    为什么需要：单局回报从 -39 到 +201（P2 加 victory 后跨度更大），
    价值头直接回归原始量级会被极端局带崩；PPO 标准做法是把 return 标准化到 ~N(0,1)。
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values) -> None:
        import numpy as _np  # 局部导入：本模块其余部分不需要 numpy 之外的依赖

        arr = _np.asarray(values, dtype=float).reshape(-1)
        n_b = int(arr.size)
        if n_b == 0:
            return
        mean_b = float(arr.mean())
        m2_b = float(((arr - mean_b) ** 2).sum())
        total = self.count + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / total
        self.m2 += m2_b + delta * delta * self.count * n_b / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float((self.m2 / self.count) ** 0.5)

    def normalize(self, values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / max(self.std, eps)


def gae(
    rewards: List[float],
    values: List[float],
    dones: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """单回合的广义优势估计（GAE-λ）。

    ``delta_t = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)``，
    ``A_t = delta_t + γ·λ·(1-done)·A_{t+1}``（回合末 bootstrap=0）。
    返回 ``(advantages, returns)``，其中 ``returns = A + V``（critic 的回归目标）。
    """
    t_max = len(rewards)
    advantages = np.zeros(t_max, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    if t_max == 0:
        return advantages, advantages
    running = 0.0
    for step in reversed(range(t_max)):
        not_done = 1.0 - float(dones[step])
        next_value = 0.0 if step == t_max - 1 else float(values_arr[step + 1])
        delta = float(rewards[step]) + gamma * next_value * not_done - float(values_arr[step])
        running = delta + gamma * lam * not_done * running
        advantages[step] = running
    returns = advantages + values_arr
    return advantages, returns


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
    #: 本局对手策略（取自 env.opponent_mode）；用于「固定对手池胜率」分桶。
    opponent_mode: str = ""
    #: 学习方是否赢下本局（winner == 学习方用户名）；未分胜负（超时截断）为 False。
    win: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "totalReward": round(self.total_reward, 4),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "winner": self.winner,
            "opponentMode": self.opponent_mode,
            "win": self.win,
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
        algo: str = "reinforce",
        epochs: int = 4,
        clip: float = 0.2,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_returns: bool = True,
        lr_decay: bool = True,
        entropy_floor: float = 0.15,
        entropy_coef_max: float = 2.0,
        target_entropy: float = 0.30,
        minibatch: int = 256,
        max_grad_norm: float = 0.5,
        early_stop_kl: float = 0.03,
        value_clip: float = 0.2,
        saturation_logit_std: float = 4.0,
        saturation_max_prob: float = 0.95,
        saturation_boost: float = 1.5,
    ) -> None:
        self.env = env
        self.policy = policy
        self.gamma = float(gamma)
        # 默认仍是 reinforce：老调用方（含既有单测）语义不变；train.py 默认走 ppo。
        if algo not in ("ppo", "reinforce"):
            raise ValueError(f"未知算法: {algo}（ppo | reinforce）")
        self.algo = algo
        self.epochs = max(1, int(epochs))
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        #: PPO 标准：把 return / 优势按 running mean-std 归一化（跨局累计）。
        self.normalize_returns = bool(normalize_returns)
        self._ret_stats = RunningStats()
        # --- P3：PPO 稳定性 ---
        #: 初始学习率（线性衰减的基准，衰减到它的 10%）。
        self.base_lr = float(policy.lr)
        self.lr_decay = bool(lr_decay)
        self._lr_updates = 0
        self._lr_total = 1
        #: 单调衰减的保护：lr 只允许下降，任何情况下都不能回弹（P2.2）。
        self._lr_current = float(policy.lr)
        #: 熵下限与自适应系数（entropy 掉到 floor 以下就把 entropy_coef 翻倍，封顶 max）。
        self.entropy_floor = float(entropy_floor or 0.0)
        self.base_entropy_coef = float(entropy_coef)
        self.entropy_coef_max = float(entropy_coef_max)
        self._entropy_coef = float(entropy_coef)
        #: P2.2：目标熵控制器（0 = 关闭，回落到 P2.1 的翻倍逻辑）。
        self.target_entropy = float(target_entropy or 0.0)
        #: P2.2：价值裁剪（0 = 关闭）。
        self.value_clip = float(value_clip or 0.0)
        #: P2.3：饱和保护——logits 标准差或 maxProb 越界时额外放大熵系数。
        self.saturation_logit_std = float(saturation_logit_std or 0.0)
        self.saturation_max_prob = float(saturation_max_prob or 0.0)
        self.saturation_boost = float(saturation_boost or 1.0)
        #: minibatch<=0 表示「整批一次更新」（P2 旧行为）。
        self.minibatch = int(minibatch)
        self.max_grad_norm = float(max_grad_norm)
        self.early_stop_kl = float(early_stop_kl)
        self.rng = np.random.default_rng(seed)
        self.log = log
        self.review_dir = review_dir
        self.history: List[Dict[str, Any]] = []
        self._episode_no = 0

    def _attacked_this_step(self, username: str) -> bool | None:
        """本 step 是否真的下发了 attack；env 不支持时返回 None（回落到动作编号）。"""
        getter = getattr(self.env, "last_dispatch", None)
        if getter is None:
            return None
        try:
            actions = getter(username)
        except Exception:  # noqa: BLE001 - 诊断信息不该影响训练
            return None
        return any(a.get("type") == "attack" for a in actions or [])

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
                # 复盘口径：按「实际下发的协议动作」判定有没有真挥刀，
                # 而不是按策略选的动作编号（越界的 attack 会被 decode 成靠近）。
                attacked=self._attacked_this_step(username),
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
        # P1-1：分对手胜率需要「这局打的是谁 + 是否赢」，两者都从环境/结果里取，不依赖调用方补。
        result.opponent_mode = str(getattr(self.env, "opponent_mode", "") or "")
        result.win = bool(result.winner is not None and result.winner == username)
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

    # ------------------------------------------------------------------- ppo
    def ppo_update(self, result: EpisodeResult) -> Dict[str, Any]:
        """单个回合的 PPO 更新（老路径，语义不变）。

        多场地并行采样请用 :meth:`ppo_update_many`；单条轨迹走同一个实现，
        避免出现两套 GAE/归一化口径。
        """
        return self.ppo_update_many([result])

    def ppo_update_many(self, results: Sequence[EpisodeResult],
                        seed: Optional[int] = None) -> Dict[str, Any]:
        """多个**独立轨迹**的 PPO 更新：逐条算 GAE，拼成一个大 batch 只做一组更新。

        关键点（VecEnv 正确性的核心）：

        - 每条轨迹**单独** bootstrap（末步 ``done=1``），绝不把不同场地的 reward 拼成一条
          ——那会让 A 场的 ``V(s_{t+1})`` 串到 B 场；
        - ``returns`` 用同一份 running stats 归一化（跨轨迹统计更稳），优势在整批上
          ``normalize_advantages``（PPO 标准做法）；
        - ``epochs`` / ``minibatch`` / lr 调度 / 熵控制器全部沿用实例配置；
        - ``seed`` 只影响 minibatch 洗牌（``None`` = 旧行为：按 history 长度取种子）。
        """
        observations: List[np.ndarray] = []
        actions: List[int] = []
        old_logp_parts: List[np.ndarray] = []
        values_parts: List[np.ndarray] = []
        advantage_parts: List[np.ndarray] = []
        return_parts: List[np.ndarray] = []
        for result in results or []:
            if not result.observations:
                continue
            old_logp, _, values = self.policy.evaluate_actions(result.observations, result.actions)
            rewards = list(result.rewards)
            dones = [0.0] * len(rewards)
            if dones:
                # 回合已结束（terminated/truncated 都按终止处理）：末步 bootstrap=0。
                dones[-1] = 1.0
            advantages, returns = gae(rewards, list(values), dones,
                                      gamma=self.gamma, lam=self.gae_lambda)
            observations.extend(result.observations)
            actions.extend(result.actions)
            old_logp_parts.append(np.asarray(old_logp, dtype=np.float64))
            values_parts.append(np.asarray(values, dtype=np.float64))
            advantage_parts.append(advantages)
            return_parts.append(returns)
        if not observations:
            return {}
        old_logp = np.concatenate(old_logp_parts)
        values = np.concatenate(values_parts)
        advantages = np.concatenate(advantage_parts)
        returns = np.concatenate(return_parts)

        # P2：价值目标与优势都做 running 归一化；原始 reward 仍原样进 report。
        if self.normalize_returns:
            self._ret_stats.update(returns)
            returns_target = self._ret_stats.normalize(returns)
            advantages = advantages / max(self._ret_stats.std, 1e-6)
        else:
            returns_target = returns
        norm_adv = normalize_advantages(advantages)

        total = len(observations)
        # P2.2：价值头质量（用行为策略的 V 与 GAE 回报算，不受本次更新影响）。
        ev = explained_variance(values, returns)
        # minibatch<=0 → 整批一次更新（P2 行为）；否则按批打乱后逐批更新。
        batch_size = total if self.minibatch <= 0 else min(int(self.minibatch), total)
        stats: Dict[str, float] = {}
        early_stopped = False
        rng = np.random.default_rng(len(self.history) + 1 if seed is None else int(seed))

        for _epoch in range(self.epochs):
            order = rng.permutation(total) if batch_size < total else np.arange(total)
            for start in range(0, total, max(1, batch_size)):
                index = np.asarray(order[start:start + batch_size], dtype=int)
                grads = self.policy.ppo_gradients(
                    [observations[i] for i in index],
                    [actions[i] for i in index],
                    old_logp[index],
                    norm_adv[index],
                    returns_target[index],
                    clip=self.clip,
                    value_coef=self.value_coef,
                    entropy_coef=self._entropy_coef,
                    old_values=np.asarray(values, dtype=np.float64)[index],
                    value_clip=self.value_clip,
                )
                norm = self.policy.apply_gradients(grads, max_grad_norm=self.max_grad_norm)
                self._decay_lr()

                new_logp, entropy, new_values = self.policy.evaluate_actions(observations, actions)
                ratio = np.exp(new_logp - old_logp)
                clipped = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip)
                max_prob = float(self.policy.mean_max_prob(observations))
                # P2.3：策略饱和诊断（logits 拉太开 / 概率过于集中）。
                logit_std = float(self.policy.logit_std(observations))
                stats = {
                    "policyLoss": float(-np.mean(np.minimum(ratio * norm_adv, clipped * norm_adv))),
                    "valueLoss": float(np.mean(0.5 * (new_values - returns_target) ** 2)),
                    "entropy": float(np.mean(entropy)),
                    "approxKL": float(np.mean(old_logp - new_logp)),
                    "clipFraction": float(np.mean(np.abs(ratio - 1.0) > self.clip)),
                    "gradNorm": float(norm),
                    "lr": float(self.policy.lr),
                    "entropyCoef": float(self._entropy_coef),
                    "maxProb": max_prob,
                    "logitStd": logit_std,
                    "temperature": float(getattr(self.policy, "temperature", 1.0)),
                    "explainedVariance": float(ev),
                }
                self._adapt_entropy_coef(stats["entropy"])
                # 饱和时额外加码（只影响下一次 update 用的 coef）。
                saturated = self._is_saturated(logit_std, max_prob)
                if saturated:
                    self._entropy_coef = min(
                        self.entropy_coef_max, self._entropy_coef * self.saturation_boost)
                stats["saturationBoost"] = bool(saturated)
            # 早停：KL 已经跑太远，再更新只会把策略推得更偏（并且加速熵塌陷）。
            if self.early_stop_kl > 0 and stats.get("approxKL", 0.0) > self.early_stop_kl:
                early_stopped = True
                break

        # 布尔量不能走 round()（会变成 0/1），原样保留。
        out = {key: (value if isinstance(value, bool) else round(value, 6))
               for key, value in stats.items()}
        out["earlyStop"] = early_stopped
        return out

    def plan_lr_schedule(self, episodes: int, steps_per_episode: int = 1) -> int:
        """按**整体**训练计划设定 lr 衰减的「总更新数」（P2.2）。

        ``总更新数 = episodes × epochs × ceil(max(1,steps)/minibatch)``：
        minibatch 会把一局拆成多次参数更新，只按局数算会让 lr 提前触底。
        **只放大不回缩**：跨 ``train()`` 调用（分阶段训练）不会把计划改小，
        否则中途重设计划会让 lr 突然跳回去。
        """
        episodes = max(1, int(episodes))
        steps = max(1, int(steps_per_episode))
        if self.minibatch <= 0:
            per_episode = self.epochs
        else:
            per_episode = self.epochs * int(math.ceil(steps / float(self.minibatch)))
        self._lr_total = max(int(self._lr_total), episodes * max(1, per_episode))
        return self._lr_total

    def _decay_lr(self) -> None:
        """每个参数更新后把 lr 线性衰减到初始的 10%，**跨局单调、不回弹**。

        旧实现每次 ``train()`` 都重置 ``_lr_updates`` 并按本段步数重算，
        于是每局开头 lr 又跳回高位（run 20261118-0123 里 0.00003 → 0.00023 的锯齿）。
        现在计数器只增不减，并且额外夹一层「不得超过上一次的 lr」——
        ``_lr_total`` 只会在计划扩大时变大，理论上 frac 可能回退，这一层兜住它。
        """
        if not self.lr_decay or self.base_lr <= 0:
            return
        self._lr_updates += 1
        frac = min(1.0, self._lr_updates / float(max(1, self._lr_total)))
        new_lr = self.base_lr * (1.0 - 0.9 * frac)
        if new_lr > self._lr_current:
            new_lr = self._lr_current
        self._lr_current = new_lr
        self.policy.lr = new_lr

    def _is_saturated(self, logit_std: float, mean_max_prob: float) -> bool:
        """策略是否已经饱和（P2.3）：logits 拉太开或 maxProb 太高。

        这两种情况下 softmax 的梯度接近 0，``entropy_coef`` 的正常增长追不上
        熵塌陷速度（实测 entropy 从第 1 局就 0.008–0.023），需要额外加码。
        """
        if self.saturation_logit_std > 0 and logit_std > self.saturation_logit_std:
            return True
        if self.saturation_max_prob > 0 and mean_max_prob > self.saturation_max_prob:
            return True
        return False

    def _adapt_entropy_coef(self, entropy: float) -> None:
        """熵控制器（P2.2）：目标熵 + 旧的下限翻倍（二者取较大 coef）。

        目标熵形式是标准的「把熵当控制量」：``coef ← coef·exp(k·(target − H))``，
        H 低于目标就放大、高于目标就收缩，指数形式保证收敛到目标而不是无限翻倍。
        旧逻辑（H < floor 就翻倍）在 coef 顶到上限后完全失效——run 20261118-0123 里
        ``entropyCoef`` 从头到尾钉在 0.1，所以这里把上限抬到 0.5 并换成连续控制器。
        两者都关（``target_entropy=0`` 且 ``entropy_floor=0``）时退回 P2.1 行为（不调整）。

        P2.3：目标熵之外还有**饱和保护**——见 :meth:`_is_saturated`，命中时再乘
        ``saturation_boost``（仍受 ``entropy_coef_max`` 限制）。
        """
        proposal = self._entropy_coef
        if self.target_entropy > 0:
            proposal = self._entropy_coef * math.exp(
                0.5 * (self.target_entropy - float(entropy)))
            proposal = min(self.entropy_coef_max, max(ENTROPY_COEF_MIN, proposal))
        if self.entropy_floor > 0:
            if entropy < self.entropy_floor:
                # 下限保护：照旧翻倍，但不会把目标控制器算出的更高值压下去。
                proposal = max(proposal, min(self.entropy_coef_max, self._entropy_coef * 2.0))
            elif self.target_entropy <= 0 and entropy > 2.0 * self.entropy_floor:
                # 只有关闭目标熵时才用旧的回落分支，避免两个控制器互相拉扯。
                proposal = max(self.base_entropy_coef, self._entropy_coef / 1.1)
        self._entropy_coef = proposal

    # --------------------------------------------------------------- training
    def train(self, episodes: int, on_episode: Optional[Callable[[int, EpisodeResult], None]] = None) -> List[Dict[str, Any]]:
        started = time.time()
        start_index = len(self.history)
        # P2.2：lr 衰减的总更新数**跨调用连续**——这里只把计划「撑大」，不重置计数。
        # 调用方（train.py）应该在训练开始前用 plan_lr_schedule 按整份计划设一次；
        # 这里用「本段局数 × epochs」兜底，保证老的单次调用语义不变（每局至少一次更新）。
        if self.base_lr <= 0:
            self.base_lr = float(self.policy.lr)
            self._lr_current = float(self.policy.lr)
        self._lr_total = max(int(self._lr_total), self._lr_updates + int(episodes) * self.epochs)
        for i in range(1, int(episodes) + 1):
            updates_before = int(self.policy.updates)
            result = self.run_episode(greedy=False)
            if self.algo == "ppo":
                ppo_stats = self.ppo_update(result)
            else:
                returns = discounted_returns(result.rewards, self.gamma)
                advantages = normalize_advantages(returns)
                self.policy.update(result.observations, result.actions, advantages)
                ppo_stats = {}
            # minibatch 的真实拆分只有跑完一局才知道：用实测更新数把剩余局数补进计划，
            # 免得 lr 中途触底（计划只放大，因此不会引起回弹）。
            per_episode = int(self.policy.updates) - updates_before
            if per_episode > 0:
                self._lr_total = max(self._lr_total, self._lr_updates + (int(episodes) - i) * per_episode)

            record = {
                "episode": i,
                **result.summary(),
                "policyUpdates": self.policy.updates,
                "elapsedSeconds": round(time.time() - started, 1),
                # 只有 ppo 才写这几项：reinforce 的 history 键集保持原样（旧测试/旧报告依赖）。
                **({"algo": self.algo, **ppo_stats} if self.algo == "ppo" else {}),
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
    def evaluate(self, episodes: int = 1, greedy: bool = True) -> List[Dict[str, Any]]:
        """跑评估局。

        ``greedy=True`` 用 argmax（旧行为）；``False`` 用策略采样——训练时是采样，
        评估如果一直贪心，量到的是「另一个策略」，P1.6 要求两组都出。
        """
        out = []
        for i in range(1, int(episodes) + 1):
            result = self.run_episode(greedy=greedy)
            record = {"episode": i, **result.summary()}
            out.append(record)
            self.log(
                f"[eval{'/greedy' if greedy else ''}] episode {i}/{episodes} "
                f"steps={result.steps} reward={result.total_reward:.3f} winner={result.winner}"
            )
        return out


def save_report(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
