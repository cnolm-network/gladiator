"""自博弈训练 CLI：每轮把当前 learner 存成快照，下一轮用它当对手。

为什么需要它：``scripted`` / ``rusher`` / ``aimbot`` 都是手写控制器，行为固定、上限固定。
当 BC 一致率到 1.0、对三种对手全胜之后，训练就没有「更强的对手」可学了。自博弈把
**上一轮的自己**放上场当对手：对手随 learner 一起变强，难度自动跟着走。

一轮做三件事：

1. 把当前 learner 存成 ``<snapshot-dir>/round<k>.npz``；
2. 用 ``VecEnv`` 训练 ``--episodes-per-round`` 个批次，对手 = **上一轮**的快照
   （第 0 轮 = ``--init-policy`` 本身；不给就用当前策略）；
3. 用 ``eval_suite.run_suite``（固定 seed 集）跑 ``--eval-episodes`` 局 × 各对手，
   记录绝对强度（对手是手写模式，跨轮可比）。

用法::

    python scripts/selfplay_train.py --rounds 3 --episodes-per-round 8 --arenas 4 \\
        --init-policy ../dashboard/var/models/dagger-1118b/policy.npz --no-spawn

前置条件：``--arenas`` × 2 个已连上通道 B 的 **普通** ``bot-runtime``（不带 ``--brain``），
用户名/端口与 ``--users-prefix``/``--ports-base`` 对应。对手动作由 env 侧策略算，
所以第 2 个 Bot 同样只是「执行动作的运行时」，不需要自己的大脑进程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_train.eval_suite import (  # noqa: E402
    SUITE_DEFAULT,
    SuiteConfig,
    format_interval,
    run_suite,
    summarize,
)
from gladiator_train.policy import Policy, adopt_weights, validate_init_policy  # noqa: E402
from gladiator_train.runs import new_run_dir, run_paths  # noqa: E402
from gladiator_train.trainer import save_report  # noqa: E402
from gladiator_train.vec_env import (  # noqa: E402
    DEFAULT_ARENA_GAP,
    DEFAULT_CENTER,
    DEFAULT_SPACING,
    VecEnv,
    VecTrainer,
    build_env,
    plan_arenas,
)

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_ROOT = TRAINING_DIR / "artifacts" / "selfplay"


# ---------------------------------------------------------------------------
# 自博弈主循环（注入 build_vec / train / evaluate，便于脱离服务器单测）
# ---------------------------------------------------------------------------

def run_selfplay(
    policy: Any,
    *,
    rounds: int,
    episodes_per_round: int,
    snapshot_dir: str,
    build_vec: Callable[[Optional[str]], Any],
    train: Callable[[Any, int], list],
    evaluate: Callable[[Any], dict],
    init_policy: Optional[str] = None,
    pool: Optional[Any] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """跑 ``rounds`` 轮「存快照 → 用上一轮快照当对手训练 → 评估」。

    约定（与 docs/training.md 对应）：

    - 第 0 轮的对手是 ``init_policy``（没给就是刚存下的 ``round0.npz``，即当前策略本身）；
    - 第 k 轮（k ≥ 1）的对手是 ``round{k-1}.npz`` —— learner 已经在往前走，
      对手落后一轮，这正是自博弈想要的「稳定的、稍弱的靶子」；
    - 每轮结束前都会 ``vec.close_all()``（异常也关），不会把连接泄漏到下一轮。

    传入 ``pool``（`gladiator_train.pfsp.OpponentPool`）时切换成 **PFSP**：每轮的对手从历史快照池里
    按「还没打过 / 打不过」加权采样，而不是固定用上一轮；当轮打完把批次胜率回流进池子，
    池里再收录这一轮的新快照。池为空时退回链式行为（避免首轮没有对手可选）。
    """
    if rounds < 1:
        raise ValueError("rounds 必须 ≥ 1")
    if episodes_per_round < 1:
        raise ValueError("episodes_per_round 必须 ≥ 1")
    os.makedirs(snapshot_dir, exist_ok=True)
    records: list[dict] = []
    previous_snapshot: Optional[str] = None
    for index in range(int(rounds)):
        snapshot_path = os.path.join(snapshot_dir, f"round{index}.npz")
        policy.save(snapshot_path, extra={
            "source": "selfplay",
            "round": index,
            "opponent": previous_snapshot or init_policy,
        })
        sampled = pool.sample(_pool_rng(pool)) if pool is not None else None
        if sampled:
            opponent = sampled
            opponent_source = "pfsp"
        elif index == 0:
            opponent = init_policy or snapshot_path
            opponent_source = "init-policy" if init_policy else "self"
        else:
            opponent = previous_snapshot or snapshot_path
            opponent_source = "previous-snapshot"
        if pool is not None and index == 0 and init_policy:
            pool.add(init_policy)
        log(f"[selfplay] 第 {index + 1}/{rounds} 轮：快照 {os.path.basename(snapshot_path)}，"
            f"对手 {os.path.basename(str(opponent))}（{opponent_source}）")
        vec = None
        started = time.time()
        try:
            vec = build_vec(opponent)
            history = train(vec, int(episodes_per_round))
        finally:
            closer = getattr(vec, "close_all", None)
            if callable(closer):
                closer()
        evaluation = evaluate(policy) or {}
        summary = evaluation.get("summary") or {}
        overall = summary.get("overall") or {}
        record = {
            "round": index,
            "snapshot": snapshot_path,
            "snapshotSha256": _sha256(snapshot_path),
            "opponent": str(opponent),
            "opponentSource": opponent_source,
            "episodes": int(episodes_per_round),
            "batches": len(history or []),
            "trainWinRate": _train_win_rate(history),
            "trainSteps": sum(int(item.get("steps", 0) or 0) for item in history or []),
            "eval": evaluation,
            "evalWinRate": float(overall.get("winRate") or 0.0),
            "evalEpisodes": int(overall.get("episodes") or 0),
            "durationSeconds": round(time.time() - started, 2),
        }
        records.append(record)
        if record["evalEpisodes"]:
            log(f"[selfplay] 第 {index + 1} 轮评估："
                f"{overall.get('wins', 0)}/{record['evalEpisodes']} 胜 "
                f"({record['evalWinRate']:.1%}，"
                f"CI {format_interval(overall.get('winRateCI', (0.0, 1.0)))})")
        if pool is not None:
            wins, losses = _win_loss(history)
            pool.record(str(opponent), wins, losses)
            pool.add(snapshot_path)
            record["pfsp"] = {"wins": wins, "losses": losses,
                              "poolSize": len(pool.paths()),
                              "opponentWeight": _pool_weight(pool, str(opponent))}
        previous_snapshot = snapshot_path
    result = {"rounds": records}
    if pool is not None:
        result["pool"] = pool.summary()
    return result


def _pool_rng(pool: Any) -> Any:
    """池里只用来采样，独立随机源即可；保持可复现（按池大小派生种子）。"""
    import numpy as np

    state = getattr(pool, "_pfsp_rng", None)
    if state is None:
        state = np.random.default_rng(20261128 + len(pool.paths()))
        setattr(pool, "_pfsp_rng", state)
    return state


def _pool_weight(pool: Any, path: str) -> Optional[float]:
    """取某个对手当前的采样权重（只用于写报告，取不到就 None）。"""
    summary = pool.summary() if hasattr(pool, "summary") else {}
    for row in summary.get("opponents", []):
        if row.get("name") == os.path.basename(path):
            return row.get("weight")
    return None


def _win_loss(history: list) -> tuple:
    """从逐批结果里数胜负。

    训练侧每批回报的是汇总字段 `{"episodes": N, "wins": M, ...}`（不是逐局明细），
    所以这里用 `episodes - wins` 补出负场；字段缺失就当 0，不抛错。
    """
    wins = 0
    losses = 0
    for batch in history or []:
        if not isinstance(batch, dict):
            continue
        count = int(batch.get("episodes", 0) or 0)
        won = int(batch.get("wins", 0) or 0)
        wins += max(0, min(won, count))
        losses += max(0, count - won)
    return wins, losses


def _train_win_rate(history: list) -> Optional[float]:
    """从逐批统计里汇总胜率；拿不到胜负字段时返回 ``None``。"""
    wins = 0
    total = 0
    for item in history or []:
        if not isinstance(item, dict):
            continue
        episode_wins = item.get("wins")
        episode_count = item.get("episodes")
        if episode_wins is None or not episode_count:
            continue
        wins += int(episode_wins)
        total += int(episode_count)
    return (wins / total) if total else None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="自博弈训练：每轮拿上一轮快照当对手（VecEnv 并行采样 + 固定 seed 评估）",
    )
    p.add_argument("--rounds", type=int, default=3, help="轮数（默认 3）")
    p.add_argument("--episodes-per-round", type=int, default=8,
                   help="每轮每个场地的局数（= 批数，默认 8）")
    p.add_argument("--arenas", type=int, default=4, help="场地数（默认 4；建议 ≤ 4）")
    p.add_argument("--users-prefix", default="Bot_V", help="Bot 用户名前缀（<prefix><i>A/B>）")
    p.add_argument("--ports-base", type=int, default=8770, help="第 i 场用 base+2i / base+2i+1")
    p.add_argument("--arena-gap", type=float, default=DEFAULT_ARENA_GAP,
                   help=f"场地沿 X 轴间距（默认 {DEFAULT_ARENA_GAP:g} 格）")
    p.add_argument("--center", default="0.5,4,0.5", help="第 0 个场地的中心 x,y,z")
    p.add_argument("--init-policy", default="", help="初始策略 npz（同时充当第 0 轮对手）")
    p.add_argument("--snapshot-dir", default="", help="快照目录（默认 artifacts/selfplay/<时间>）")
    p.add_argument("--pfsp-pool", default="",
                   help="PFSP 对手池目录（给出则改为按历史快照加权采样对手，默认关闭=链式自博弈）")
    p.add_argument("--pfsp-exponent", type=float, default=2.0,
                   help="PFSP 权重指数 w ∝ (1 - winRate)^p，越大越偏向打不过的对手（默认 2.0）")
    p.add_argument("--save-dir", default="", help="run 目录（默认 artifacts/runs/<时间>-selfplay-<tag>）")
    p.add_argument("--max-steps", type=int, default=400, help="每局最多步数")
    p.add_argument("--eval-episodes", type=int, default=4, help="每轮评估：每个对手几局")
    p.add_argument("--eval-opponents", default=",".join(SUITE_DEFAULT["opponents"]),
                   help=f"评估用的手写对手（默认 {','.join(SUITE_DEFAULT['opponents'])}）")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--opponent-epsilon", type=float, default=0.02,
                   help="对手以该概率随机动作（默认 0.02，给对手一点多样性）")
    p.add_argument("--victory", type=float, default=50.0, help="胜负常数（默认 50）")
    p.add_argument("--workers", type=int, default=4, help="采样线程数（默认 4）")
    p.add_argument("--no-spawn", action="store_true", help="不 spawn（Bot 已经在跑）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", default="devpass")
    p.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    p.add_argument("--prep-seconds", type=float, default=3.0)
    p.add_argument("--arena-jitter", default="2,1", help="起始分布抖动 \"中心,间距\"")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--action-ticks", type=int, default=2)
    p.add_argument("--reward-preset", default="", help="reward 预设 JSON 路径（可选）")
    # --- PPO 超参（默认与 train.py / vec_train.py 一致） ---
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--target-entropy", type=float, default=0.30)
    p.add_argument("--entropy-floor", type=float, default=0.15)
    p.add_argument("--entropy-coef-max", type=float, default=2.0)
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    return p


def _parse_triple(text: str, flag: str) -> tuple:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit(f"{flag} 需要 x,y,z，收到 {text!r}")
    try:
        return tuple(float(value) for value in parts)
    except ValueError as exc:
        raise SystemExit(f"{flag} 需要数字: {exc}") from exc


def _parse_pair(text: str, flag: str) -> tuple:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"{flag} 需要两个数字，收到 {text!r}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise SystemExit(f"{flag} 需要数字: {exc}") from exc


def _parse_opponents(text: str) -> tuple:
    items = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not items:
        raise SystemExit("--eval-opponents 不能为空")
    allowed = {"random", "wander", "idle", "scripted", "rusher", "aimbot"}
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise SystemExit(f"未知对手模式: {', '.join(unknown)}（可选: {', '.join(sorted(allowed))}）")
    ordered, seen = [], set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds 必须 ≥ 1")
    if args.episodes_per_round < 1:
        raise SystemExit("--episodes-per-round 必须 ≥ 1")
    if args.arenas < 1:
        raise SystemExit("--arenas 必须 ≥ 1")
    if args.eval_episodes < 0:
        raise SystemExit("--eval-episodes 不能为负")
    if not 0.0 <= args.opponent_epsilon <= 1.0:
        raise SystemExit("--opponent-epsilon 必须在 [0, 1]")
    if args.arenas > 4:
        print(f"[selfplay] 提示：{args.arenas} 个场地会产生 {args.arenas * 2} 个在线 Bot，"
              "对手也吃服务端算力；建议 ≤ 4。", file=sys.stderr)

    center = _parse_triple(args.center, "--center")
    jitter = _parse_pair(args.arena_jitter, "--arena-jitter")

    from gladiator_env.reward_presets import load_preset
    from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM

    # 先校验初始策略：无效就直接退出，不建目录、不 spawn（与 vec_train 同顺序）。
    loaded, init_error = validate_init_policy(
        args.init_policy, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden)
    if init_error:
        print(f"[selfplay] {init_error}", file=sys.stderr)
        return 2

    if args.save_dir:
        run_dir = os.path.abspath(args.save_dir)
        os.makedirs(os.path.join(run_dir, "reviews"), exist_ok=True)
    else:
        run_dir = new_run_dir("selfplay", f"{args.arenas}x{args.rounds}")
    paths = run_paths(run_dir, "selfplay")
    if args.snapshot_dir:
        snapshot_dir = os.path.abspath(args.snapshot_dir)
    else:
        snapshot_dir = os.path.join(
            str(DEFAULT_SNAPSHOT_ROOT), time.strftime("%Y%m%d-%H%M", time.localtime()))
    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"[selfplay] run 目录 {run_dir}；快照目录 {snapshot_dir}")

    policy = Policy(obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden, seed=args.seed,
                    lr=args.lr)
    if loaded is not None:
        adopt_weights(policy, loaded)
        print(f"[selfplay] 初始策略 {args.init_policy}（第 0 轮对手 = 它）")

    preset = load_preset(args.reward_preset) if args.reward_preset else None
    specs = plan_arenas(args.arenas, users_prefix=args.users_prefix,
                        ports_base=args.ports_base, arena_gap=args.arena_gap, center=center)
    print("[selfplay] 场地：" + "；".join(
        f"#{spec.index} {spec.users[0]}/{spec.users[1]} :{spec.ports[0]}/:{spec.ports[1]}"
        for spec in specs))

    trainer_kwargs = {
        "epochs": args.ppo_epochs,
        "clip": args.clip,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "entropy_floor": args.entropy_floor,
        "entropy_coef_max": args.entropy_coef_max,
        "target_entropy": args.target_entropy,
        "minibatch": args.minibatch,
        "max_grad_norm": args.max_grad_norm,
        "value_clip": args.value_clip,
    }
    def build_vec(opponent_path: Optional[str]) -> VecEnv:
        envs = [
            build_env(
                spec,
                host=args.host,
                rcon_port=args.rcon_port,
                rcon_password=args.rcon_password,
                max_steps=args.max_steps,
                action_ticks=args.action_ticks,
                auto_spawn=not args.no_spawn,
                # opponent_mode 仅用于「没有快照」的兜底；有快照时被 env 忽略。
                opponent_mode="scripted",
                opponent_policy=opponent_path,
                opponent_epsilon=args.opponent_epsilon,
                reward_preset=preset,
                spacing=args.spacing,
                prep_seconds=args.prep_seconds,
                arena_jitter=jitter,
                victory=args.victory,
            )
            for spec in specs
        ]
        vec = VecEnv(envs)
        if not args.no_spawn:
            vec.spawn_all()
        return vec

    def train(vec: VecEnv, episodes: int) -> list:
        trainer = VecTrainer(
            vec, policy,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            seed=args.seed,
            workers=args.workers,
            trainer_kwargs=trainer_kwargs,
            max_steps=args.max_steps,
        )
        return trainer.train(episodes)

    eval_config = SuiteConfig(
        opponents=_parse_opponents(args.eval_opponents),
        episodes=int(args.eval_episodes),
        seed=int(args.seed) + 1000,
        max_steps=int(args.max_steps),
    )

    def evaluate(current: Policy) -> dict:
        if eval_config.episodes <= 0:
            return {"rows": [], "summary": {}}

        def factory(mode: str):
            return build_env(
                specs[0],
                host=args.host,
                rcon_port=args.rcon_port,
                rcon_password=args.rcon_password,
                max_steps=args.max_steps,
                action_ticks=args.action_ticks,
                auto_spawn=not args.no_spawn,
                opponent_mode=mode,
                reward_preset=preset,
                spacing=args.spacing,
                prep_seconds=args.prep_seconds,
                arena_jitter=jitter,
                victory=args.victory,
            )

        rows = run_suite(factory, current, eval_config, logger=print)
        return {"rows": rows, "summary": summarize(rows),
                "config": eval_config.to_dict()}

    pool = None
    if args.pfsp_pool:
        from gladiator_train.pfsp import OpponentPool  # noqa: PLC0415  (只有开了 PFSP 才需要)

        pool = OpponentPool(args.pfsp_pool, exponent=float(args.pfsp_exponent))
        os.makedirs(args.pfsp_pool, exist_ok=True)
        found = pool.discover()
        print(f"[selfplay] PFSP 对手池 {args.pfsp_pool}：已有 {len(found)} 个快照，p={args.pfsp_exponent}")

    started = time.time()
    result = run_selfplay(
        policy,
        rounds=int(args.rounds),
        episodes_per_round=int(args.episodes_per_round),
        snapshot_dir=snapshot_dir,
        build_vec=build_vec,
        train=train,
        evaluate=evaluate,
        init_policy=args.init_policy or None,
        pool=pool,
    )
    duration = time.time() - started

    final_path = paths["policy"]
    policy.save(final_path, extra={
        "selfplay": {
            "rounds": int(args.rounds),
            "episodesPerRound": int(args.episodes_per_round),
            "arenas": int(args.arenas),
            "opponentEpsilon": float(args.opponent_epsilon),
        },
        "seed": int(args.seed),
    })

    records = result["rounds"]
    meta = {
        "schema": "gladiator.selfplay/v1",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runDir": run_dir,
        "snapshotDir": snapshot_dir,
        "arenas": [spec.to_dict() for spec in specs],
        "initPolicy": args.init_policy or None,
        "finalPolicy": final_path,
        "pfspPool": result.get("pool"),
        "rounds": [
            {key: item.get(key) for key in
             ("round", "snapshot", "snapshotSha256", "opponent", "opponentSource",
              "episodes", "batches", "trainWinRate", "evalWinRate", "evalEpisodes",
              "durationSeconds", "pfsp")}
            for item in records
        ],
        "config": {
            "rounds": int(args.rounds),
            "episodesPerRound": int(args.episodes_per_round),
            "arenas": int(args.arenas),
            "maxSteps": int(args.max_steps),
            "seed": int(args.seed),
            "opponentEpsilon": float(args.opponent_epsilon),
            "victory": float(args.victory),
            "evalEpisodes": int(args.eval_episodes),
            "evalOpponents": list(eval_config.opponents),
            "evalSeed": int(eval_config.seed),
            "autoSpawn": not bool(args.no_spawn),
        },
        "durationSeconds": round(duration, 2),
    }
    _write_json(os.path.join(run_dir, "meta.json"), meta)
    save_report(paths["report"], {
        "kind": "selfplay",
        "runDir": run_dir,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "durationSeconds": round(duration, 2),
        "config": meta["config"],
        "arenas": meta["arenas"],
        "rounds": records,
        "summary": {
            "rounds": len(records),
            "finalEvalWinRate": records[-1]["evalWinRate"] if records else None,
            "firstEvalWinRate": records[0]["evalWinRate"] if records else None,
        },
        "policyPath": final_path,
        "metaPath": os.path.join(run_dir, "meta.json"),
    })

    print(f"[selfplay] 完成：{len(records)} 轮；最终策略 {final_path}")
    print(f"[selfplay] meta {os.path.join(run_dir, 'meta.json')}；report {paths['report']}")
    if records:
        print("[selfplay] 各轮评估胜率：" + "，".join(
            f"R{item['round'] + 1} {item['evalWinRate']:.1%}" for item in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
