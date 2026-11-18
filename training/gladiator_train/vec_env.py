"""VecEnv：**一个策略 × M 个场地**并行采样（梯度聚合）。

与 ``multi_train.py`` 的区别（这是本项目最容易混淆的一点）：

| | multi_train.py | VecEnv / vec_train.py |
|---|---|---|
| 策略 | N 个独立 ``Policy``（各自随机初值、各自更新） | **1 个** ``Policy``，所有场地共享 |
| 并行的是什么 | 完整的训练实验 | 只是**采样**（每个场地各跑一局） |
| 权重合并 | 最后按 mean 逐元素平均 | 不做合并：梯度在同一个策略上累加 |
| 数学上是否成立 | 不成立：不同初值的 tanh 网络隐藏单元可以任意置换，逐元素平均会打碎隐藏层 | 成立：同一坐标系、同一组参数 |

所以「merge 之前必须先有 VecEnv」：先用并行采样把样本量堆起来、在一个策略上做更新，
才有资格谈「把多个场的经验合起来」；否则平均的是一堆互相不对齐的网络。

线程模型（重要）：

- 每个 worker 只碰**自己的** env（reset/step），不共享任何可变 numpy 状态；
- worker 内用 ``rng = default_rng(seed + batch*M + index)`` 选动作，结果可复现；
- ``Policy`` 的**参数更新只在主线程**：一批 M 局全部 join 之后才调用 ``ppo_update_many``；
- GAE / return 归一化**逐轨迹**计算（不能把不同场地的轨迹拼成一条，否则 bootstrap
  会把 A 场的未来奖励算到 B 场头上），拼成大 batch 只发生在参数更新阶段。

本模块只依赖标准库 + numpy；``GladiatorEnv`` 在 :func:`build_env` 里延迟导入，
因此单测可以用假 env，不需要 gymnasium/服务端。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .trainer import EpisodeResult, Trainer, gae

DEFAULT_SPACING = 6.0
DEFAULT_ARENA_GAP = 80.0
DEFAULT_CENTER = (0.5, 4.0, 0.5)


# ---------------------------------------------------------------------------
# 场地规划与环境构造（train.py 的「bot 配置 + ArenaConfig」抽到这里）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArenaSpec:
    """一个场地的身份：两个 Bot 的用户名/端口 + 竞技场中心。"""

    index: int
    users: Tuple[str, str]
    ports: Tuple[int, int]
    center: Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "users": list(self.users),
            "ports": list(self.ports),
            "center": list(self.center),
        }


def plan_arenas(count: int, *, users_prefix: str = "Bot_V", ports_base: int = 8770,
                arena_gap: float = DEFAULT_ARENA_GAP,
                center: Sequence[float] = DEFAULT_CENTER) -> List[ArenaSpec]:
    """把 M 个场地沿 X 轴排开：第 i 场用 ``<prefix><i>A/B``、端口 ``base+2i / base+2i+1``。

    ``arena_gap`` 取 80 格：两个 7×4×7 的观测窗口 + 追击距离都远小于它，
    同时 4 个场地仍在同一区块加载半径内（服务端不必反复加载区块）。
    """
    count = int(count)
    if count <= 0:
        raise ValueError("arenas 必须 ≥ 1")
    if len(center) != 3:
        raise ValueError("center 需要 x,y,z")
    cx, cy, cz = (float(v) for v in center)
    specs: List[ArenaSpec] = []
    for index in range(count):
        specs.append(ArenaSpec(
            index=index,
            users=(f"{users_prefix}{index}A", f"{users_prefix}{index}B"),
            ports=(int(ports_base) + 2 * index, int(ports_base) + 2 * index + 1),
            center=(cx + float(arena_gap) * index, cy, cz),
        ))
    return specs


def build_env(spec: ArenaSpec, *, host: str = "127.0.0.1", rcon_port: int = 25575,
              rcon_password: str = "devpass", max_steps: int = 400,
              action_ticks: int = 2, auto_spawn: bool = True,
              opponent_mode: str = "scripted", reward_preset: Optional[dict] = None,
              spacing: float = DEFAULT_SPACING, prep_seconds: float = 3.0,
              arena_jitter: Sequence[float] = (2.0, 1.0),
              victory: Optional[float] = None,
              opponent_policy: Any = None,
              opponent_epsilon: float = 0.0):
    """按 :class:`ArenaSpec` 构造一个 ``GladiatorEnv``（延迟导入 gymnasium 依赖）。

    ``opponent_policy``（自博弈快照：``Policy`` 或 ``.npz`` 路径）非空时，对手不再走
    ``opponent_mode`` 的手写控制器；见 ``GladiatorEnv`` 的同名参数。
    """
    from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv

    bots = [
        BotConfig(username=spec.users[0], env_url=f"ws://{host}:{spec.ports[0]}"),
        BotConfig(username=spec.users[1], env_url=f"ws://{host}:{spec.ports[1]}"),
    ]
    jitter = (float(arena_jitter[0]), float(arena_jitter[1])) if arena_jitter else (0.0, 0.0)
    arena = ArenaConfig(center=tuple(spec.center), spacing=float(spacing),
                        prep_seconds=float(prep_seconds), jitter=jitter)
    env = GladiatorEnv(
        bots,
        arena,
        rcon={"host": host, "port": int(rcon_port), "password": rcon_password},
        action_ticks=int(action_ticks),
        max_steps=int(max_steps),
        single_agent=True,
        auto_spawn=bool(auto_spawn),
        opponent_mode=str(opponent_mode),
        opponent_policy=opponent_policy,
        opponent_epsilon=float(opponent_epsilon),
        reward_preset=reward_preset,
    )
    if victory is not None:
        # 与 train.py 同口径：命令行覆盖优先级高于 reward preset。
        env.shaping.weights.victory = float(victory)
    return env


# ---------------------------------------------------------------------------
# VecEnv：M 个已构造好的环境
# ---------------------------------------------------------------------------

class VecEnv:
    """持有 M 个已构造好的 ``GladiatorEnv``；只负责批量 spawn/close。"""

    def __init__(self, envs: Sequence[Any], log: Callable[[str], None] = print) -> None:
        if not envs:
            raise ValueError("VecEnv 至少需要一个 env")
        self._envs = list(envs)
        self.log = log
        self._spawned = False
        self._closed = False
        self._close_errors: List[BaseException] = []

    @property
    def envs(self) -> List[Any]:
        """环境列表（返回副本，避免调用方误改内部顺序）。"""
        return list(self._envs)

    @property
    def count(self) -> int:
        return len(self._envs)

    def spawn_all(self) -> None:
        """把所有场地的 Bot 一起 spawn（幂等）。

        ``GladiatorEnv`` 没有公开的 spawn 方法，这里复用它的 ``_spawn_bots``；
        构造时 ``auto_spawn=True`` 的话 reset 也会兜底 spawn，所以失败可重试。
        任一场地失败都会先 close 全部再抛原异常（不留半开连接）。
        """
        if self._closed:
            raise RuntimeError("VecEnv 已关闭，不能再 spawn")
        if self._spawned:
            return
        try:
            for env in self._envs:
                spawner = getattr(env, "_spawn_bots", None)
                if callable(spawner):
                    spawner()
            self._spawned = True
        except Exception:
            self.close_all()
            raise

    def close_all(self) -> None:
        """关闭全部环境（幂等、尽力而为；单个 close 失败只记录不抛出）。"""
        if self._closed:
            return
        self._closed = True
        for index, env in enumerate(self._envs):
            closer = getattr(env, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - 关闭失败不能掩盖训练异常
                self._close_errors.append(exc)
                self.log(f"[vec] 关闭 env[{index}] 失败：{exc}")

    @property
    def close_errors(self) -> List[BaseException]:
        return list(self._close_errors)


# ---------------------------------------------------------------------------
# VecTrainer：一批 M 局 → 逐轨迹 GAE → 一个大 batch 更新
# ---------------------------------------------------------------------------

class VecTrainer:
    """单策略 × M 场地并行采样 + 梯度聚合。

    ``train(episodes)`` 里的 ``episodes`` 是**批数**：每批对每个 env 各跑 1 局，
    因此总对局数 = ``episodes × vec.count``（与 ``train.py`` 的「回合」语义对齐）。
    """

    def __init__(self, vec: VecEnv, policy, gamma: float = 0.99, gae_lambda: float = 0.95,
                 seed: int = 7, workers: Optional[int] = None, *,
                 algo: str = "ppo", log: Callable[[str], None] = print,
                 trainer_kwargs: Optional[dict] = None,
                 max_steps: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if str(algo) != "ppo":
            # 明确拒绝，而不是悄悄降级成「每批一次 REINFORCE」——两种算法的样本口径不同，
            # 降级会让报告里的 steps/s 与更新次数对不上。
            raise SystemExit("VecEnv 只支持 ppo")
        if vec.count <= 0:
            raise ValueError("VecEnv 不能为空")
        self.vec = vec
        self.policy = policy
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.seed = int(seed)
        self.workers = int(workers or vec.count)
        if self.workers <= 0:
            raise ValueError("workers 必须 ≥ 1")
        self.log = log
        self.max_steps = int(max_steps or getattr(vec.envs[0], "max_steps", 0) or 400)
        self._clock = clock
        self._trainer = Trainer(
            vec.envs[0], policy,
            gamma=self.gamma,
            seed=self.seed,
            log=log,
            algo="ppo",
            gae_lambda=self.gae_lambda,
            review_dir=None,
            **(trainer_kwargs or {}),
        )
        self.history: List[Dict[str, Any]] = []
        self._opp_stats: Dict[str, Dict[str, int]] = {}
        self._episodes_done = 0
        self._steps_done = 0
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None

    # --- 对外入口 -------------------------------------------------------
    def train(self, episodes: int) -> List[Dict[str, Any]]:
        """跑 ``episodes`` 个批次（每批 = 每个场地 1 局），返回逐批统计。"""
        batches = max(1, int(episodes))
        width = self.vec.count
        if self._started_at is None:
            self._started_at = self._clock()
        # lr 衰减按「整份计划」预先撑大：每批的样本数最多是 M×max_steps。
        if self.max_steps:
            self._trainer.plan_lr_schedule(batches, self.max_steps * width)

        for batch_index in range(batches):
            batch_started = self._clock()
            try:
                results = self._sample_batch(batch_index)
                update_stats = self._trainer.ppo_update_many(
                    results, seed=self.seed + batch_index * width + 1)
            except Exception:
                # 采样或更新任一环节失败：先关掉全部环境，再把原异常抛给调用方（不能吞）。
                self.vec.close_all()
                raise
            elapsed = max(self._clock() - batch_started, 1e-9)
            steps = sum(int(result.steps) for result in results)
            wins = sum(1 for result in results if result.win)
            self._episodes_done += len(results)
            self._steps_done += steps
            for result in results:
                bucket = self._opp_stats.setdefault(
                    result.opponent_mode or "unknown", {"episodes": 0, "wins": 0})
                bucket["episodes"] += 1
                bucket["wins"] += 1 if result.win else 0
            record = {
                "batch": batch_index + 1,
                "episodes": len(results),
                "steps": steps,
                "elapsedSeconds": round(elapsed, 3),
                "stepsPerSecond": round(steps / elapsed, 2),
                "wins": wins,
                "winRate": round(wins / max(1, len(results)), 4),
                "policyUpdates": int(self.policy.updates),
                **update_stats,
            }
            self.history.append(record)
            self.log(
                f"[vec] batch {batch_index + 1}/{batches} episodes={len(results)} "
                f"steps={steps} {elapsed:.2f}s {record['stepsPerSecond']:.1f} steps/s "
                f"win={wins}/{len(results)} updates={self.policy.updates}"
            )
        self._finished_at = self._clock()
        summary = self.summary()
        self.log(
            f"[vec] 完成：{summary['episodes']} episodes / {summary['steps']} steps / "
            f"{summary['durationSeconds']}s / {summary['stepsPerSecond']} steps/s"
        )
        if summary["byOpponent"]:
            detail = "；".join(
                f"{mode} {bucket['wins']}/{bucket['episodes']}"
                for mode, bucket in summary["byOpponent"].items()
            )
            self.log(f"[vec] 分对手胜率：{detail}")
        return self.history

    def summary(self) -> dict:
        elapsed = 0.0
        if self._started_at is not None:
            end = self._finished_at if self._finished_at is not None else self._clock()
            elapsed = max(end - self._started_at, 1e-9)
        by_opponent = {
            mode: {
                "episodes": bucket["episodes"],
                "wins": bucket["wins"],
                "winRate": (bucket["wins"] / bucket["episodes"]) if bucket["episodes"] else 0.0,
            }
            for mode, bucket in self._opp_stats.items()
        }
        wins = sum(bucket["wins"] for bucket in self._opp_stats.values())
        return {
            "batches": len(self.history),
            "episodes": int(self._episodes_done),
            "steps": int(self._steps_done),
            "durationSeconds": round(elapsed, 2),
            "stepsPerSecond": round(self._steps_done / elapsed, 2) if elapsed > 0 else 0.0,
            "wins": int(wins),
            "winRate": round(wins / self._episodes_done, 4) if self._episodes_done else 0.0,
            "policyUpdates": int(self.policy.updates),
            "byOpponent": by_opponent,
            "arenas": self.vec.count,
            "workers": self.workers,
        }

    # --- 采样 -----------------------------------------------------------
    def _sample_batch(self, batch_index: int) -> List[EpisodeResult]:
        """一批 = 每个场地各跑 1 局；线程池并发，join 之后才允许更新参数。"""
        envs = self.vec.envs
        width = len(envs)
        futures: List[Tuple[int, Any]] = []
        with ThreadPoolExecutor(max_workers=min(self.workers, width),
                                thread_name_prefix="vec") as pool:
            for index, env in enumerate(envs):
                worker_seed = self.seed + batch_index * width + index
                futures.append((index, pool.submit(self._run_episode, env, worker_seed)))
            results: List[Tuple[int, EpisodeResult]] = []
            for index, future in futures:
                results.append((index, future.result()))
        results.sort(key=lambda pair: pair[0])
        return [result for _index, result in results]

    def _run_episode(self, env, seed: int) -> EpisodeResult:
        """worker：只调 env.reset/step 与只读前向 ``policy.act``，不碰训练器状态。"""
        started = time.time()
        started_wall = time.time()
        rng = np.random.default_rng(seed)
        obs, _info = env.reset(seed=seed)
        username = str(env.bots[0].username)
        limit = self.max_steps or int(getattr(env, "max_steps", 400) or 400)
        observations: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        healths: List[float] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        winner = None
        steps = 0

        while steps < limit:
            action = int(self.policy.act(obs, rng, greedy=False))
            next_obs, reward, terminated, truncated, info = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float64))
            actions.append(action)
            rewards.append(float(reward))
            total_reward += float(reward)
            steps += 1
            info = info or {}
            winner = info.get("winner", winner)
            health = info.get("health") or {}
            if isinstance(health, dict) and health:
                value = health.get(username)
                if value is None:
                    value = list(health.values())[0]
                try:
                    healths.append(float(value))
                except (TypeError, ValueError):
                    pass
            obs = next_obs
            if terminated or truncated:
                break
        if steps >= limit and not (terminated or truncated):
            # 假 env / 没实现 max_steps 的 env 也要标成截断，统计口径才一致。
            truncated = True

        return EpisodeResult(
            steps=steps,
            total_reward=total_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            winner=winner,
            duration_s=time.time() - started,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_wall)),
            started_epoch=started_wall,
            mean_health=float(np.mean(healths)) if healths else 0.0,
            observations=observations,
            actions=actions,
            rewards=rewards,
            opponent_mode=str(getattr(env, "opponent_mode_label", None)
                              or getattr(env, "opponent_mode", "") or ""),
            win=bool(winner is not None and str(winner) == username),
        )


# ---------------------------------------------------------------------------
# 逐轨迹 GAE（单测/诊断用）
# ---------------------------------------------------------------------------

def trajectory_gae(rewards: Sequence[float], values: Sequence[float],
                   gamma: float = 0.99, lam: float = 0.95):
    """一条轨迹的 GAE，显式提醒：**不能跨场地拼接**。

    不同场地的回合在时间上不连续，拼成一条会让 A 场的 ``V(s_{t+1})`` 用到 B 场的状态，
    bootstrap 直接串味。``ppo_update_many`` 调用的就是本函数（逐条），不是整批。
    """
    rewards = list(rewards)
    dones = [0.0] * len(rewards)
    if dones:
        dones[-1] = 1.0
    return gae(rewards, list(values), dones, gamma=gamma, lam=lam)
