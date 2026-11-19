"""VecEnv 训练 CLI：**一个策略 × M 个场地**并行采样，梯度聚合。

与 ``multi_train.py`` 的根本区别：

- ``multi_train.py``：每个场地一个独立 ``Policy``（各自随机初值），最后逐元素平均权重
  —— 这是并行跑 N 个实验，而且不同初值的 tanh 网络隐藏单元可以任意置换，
  平均会打碎隐藏层；
- ``vec_train.py``：**一个** ``Policy``，M 个场地同时跑 episode，把样本汇总成一个大 batch
  做一次 PPO 更新。吞吐来自并行采样，正确性来自「同一个策略、同一组参数」。

用法::

    # 4 场地 × 每场 8 局（总 32 局），Bot_V0A/B … Bot_V3A/B，端口 8770..8777
    python scripts/vec_train.py --arenas 4 --episodes 8 --workers 4

    # 从已有模型续训（只加载权重，不加载优化器状态）
    python scripts/vec_train.py --arenas 4 --init-policy ../dashboard/var/models/dagger-1118b/policy.npz

    # Bot 已由调用方 spawn（Dashboard / 手工）
    python scripts/vec_train.py --arenas 2 --no-spawn

默认超参与 ``train.py`` 的 PPO 一致：lr 3e-4、clip 0.2、4 个 PPO epoch、
目标熵 0.3、entropy_coef 0.02、value_clip 0.2、minibatch 256、victory 50（可用
``--victory`` 覆盖，优先级与 train.py 相同）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VecEnv 训练：一个策略 × M 个场地并行采样（PPO only）",
    )
    p.add_argument("--arenas", type=int, default=4, help="场地数 M（默认 4）")
    p.add_argument("--users-prefix", default="Bot_V", help="Bot 用户名前缀（<prefix><i>A/B>）")
    p.add_argument("--ports-base", type=int, default=8770, help="第 i 场用 base+2i / base+2i+1")
    p.add_argument("--arena-gap", type=float, default=DEFAULT_ARENA_GAP,
                   help=f"场地沿 X 轴的间距（默认 {DEFAULT_ARENA_GAP:g} 格）")
    p.add_argument("--center", default="0.5,4,0.5", help="第 0 个场地的中心 x,y,z")
    p.add_argument("--episodes", type=int, default=8,
                   help="每个场地的局数（= 批数）；总对局 = episodes × arenas")
    p.add_argument("--max-steps", type=int, default=400, help="每局最多步数")
    p.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"],
                   help="只支持 ppo；reinforce 会在 VecTrainer 处直接拒绝")
    p.add_argument("--init-policy", default="", help="初始策略 npz（只加载权重）")
    p.add_argument("--save-dir", default="", help="run 目录（默认 artifacts/runs/<时间>-vec-<tag>）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=4, help="采样线程数（默认 4）")
    p.add_argument("--no-spawn", action="store_true", help="不 spawn（Bot 已经在跑）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", default="devpass")
    p.add_argument("--action-space", default="v1", choices=["v1", "v2"],
                   help="v1 = 单头 11 宏动作（默认）；v2 = 9 头 MultiDiscrete")
    p.add_argument("--opponent-mode", default="scripted",
                   choices=["random", "wander", "idle", "scripted", "rusher", "aimbot"])
    p.add_argument("--reward-preset", default="", help="reward 预设 JSON 路径（可选）")
    p.add_argument("--victory", type=float, default=None,
                   help="覆盖胜负常数（默认取 ShapingWeights.victory=50）")
    p.add_argument("--action-ticks", type=int, default=2)
    p.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    p.add_argument("--prep-seconds", type=float, default=3.0)
    p.add_argument("--arena-jitter", default="2,1",
                   help="起始分布抖动 \"中心偏移,间距抖动\"；\"0,0\" = 完全确定")
    # --- PPO 超参（默认与 train.py 一致） ---
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.01, help="策略的熵正则系数（reinforce 用）")
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
    p.add_argument("--no-normalize-returns", dest="normalize_returns",
                   action="store_false", default=True)
    p.add_argument("--no-lr-decay", dest="lr_decay", action="store_false", default=True)
    return p


def _parse_center(text: str) -> tuple:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit(f"--center 需要 x,y,z，收到 {text!r}")
    try:
        return tuple(float(value) for value in parts)
    except ValueError as exc:
        raise SystemExit(f"--center 需要数字: {exc}") from exc


def _parse_jitter(text: str) -> tuple:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--arena-jitter 需要两个数字，收到 {text!r}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise SystemExit(f"--arena-jitter 需要数字: {exc}") from exc


def _config_dict(args: argparse.Namespace, center: tuple, jitter: tuple) -> dict:
    return {
        "arenas": int(args.arenas),
        "usersPrefix": args.users_prefix,
        "portsBase": int(args.ports_base),
        "arenaGap": float(args.arena_gap),
        "center": list(center),
        "episodes": int(args.episodes),
        "maxSteps": int(args.max_steps),
        "algo": args.algo,
        "seed": int(args.seed),
        "workers": int(args.workers),
        "lr": float(args.lr),
        "gamma": float(args.gamma),
        "hidden": int(args.hidden),
        "beta": float(args.beta),
        "ppoEpochs": int(args.ppo_epochs),
        "clip": float(args.clip),
        "gaeLambda": float(args.gae_lambda),
        "valueCoef": float(args.value_coef),
        "entropyCoef": float(args.entropy_coef),
        "targetEntropy": float(args.target_entropy),
        "entropyFloor": float(args.entropy_floor),
        "entropyCoefMax": float(args.entropy_coef_max),
        "valueClip": float(args.value_clip),
        "minibatch": int(args.minibatch),
        "maxGradNorm": float(args.max_grad_norm),
        "normalizeReturns": bool(args.normalize_returns),
        "lrDecay": bool(args.lr_decay),
        "actionSpace": args.action_space,
        "opponentMode": args.opponent_mode,
        "rewardPreset": args.reward_preset or None,
        "victory": (float(args.victory) if args.victory is not None else None),
        "arenaJitter": [jitter[0], jitter[1]],
        "spacing": float(args.spacing),
        "prepSeconds": float(args.prep_seconds),
        "actionTicks": int(args.action_ticks),
        "autoSpawn": not bool(args.no_spawn),
        "initPolicy": args.init_policy or None,
    }


def _write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.algo != "ppo":
        raise SystemExit("VecEnv 只支持 ppo")
    if args.arenas < 1:
        raise SystemExit("--arenas 必须 ≥ 1")
    if args.episodes < 1:
        raise SystemExit("--episodes 必须 ≥ 1")
    if args.workers < 1:
        raise SystemExit("--workers 必须 ≥ 1")

    center = _parse_center(args.center)
    jitter = _parse_jitter(args.arena_jitter)

    # 延迟导入 gymnasium 依赖：--help / 参数校验阶段不需要它们。
    from gladiator_env.reward_presets import load_preset
    from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM

    if args.save_dir:
        run_dir = os.path.abspath(args.save_dir)
        os.makedirs(os.path.join(run_dir, "reviews"), exist_ok=True)
    else:
        run_dir = new_run_dir("vec", f"{args.arenas}x{args.users_prefix}")
    paths = run_paths(run_dir, "vec")
    print(f"[vec] 运行目录 {run_dir}")

    # 初始策略先校验：无效直接退出，不建环境、不 spawn。
    loaded, init_error = validate_init_policy(
        args.init_policy, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden,
        action_space=args.action_space)
    if init_error:
        print(f"[vec] {init_error}", file=sys.stderr)
        return 2

    policy = Policy(
        obs_dim=OBS_DIM,
        n_actions=NUM_ACTIONS,
        hidden=args.hidden,
        seed=args.seed,
        lr=args.lr,
        beta=args.beta,
        action_space=args.action_space,
    )
    if loaded is not None:
        adopt_weights(policy, loaded)
        print(f"[vec] 已加载初始策略 {args.init_policy}"
              f"（obs={OBS_DIM} actions={policy.n_actions} hidden={args.hidden}"
              f" space={args.action_space}）")

    preset = load_preset(args.reward_preset) if args.reward_preset else None
    specs = plan_arenas(
        args.arenas,
        users_prefix=args.users_prefix,
        ports_base=args.ports_base,
        arena_gap=args.arena_gap,
        center=center,
    )
    print(f"[vec] 场地 {args.arenas} 个：" + "；".join(
        f"#{spec.index} {spec.users[0]}/{spec.users[1]} :{spec.ports[0]}/:{spec.ports[1]} "
        f"center={tuple(round(v, 1) for v in spec.center)}"
        for spec in specs
    ))

    envs = [
        build_env(
            spec,
            host=args.host,
            rcon_port=args.rcon_port,
            rcon_password=args.rcon_password,
            max_steps=args.max_steps,
            action_ticks=args.action_ticks,
            auto_spawn=not args.no_spawn,
            opponent_mode=args.opponent_mode,
            action_space=args.action_space,
            reward_preset=preset,
            spacing=args.spacing,
            prep_seconds=args.prep_seconds,
            arena_jitter=jitter,
            victory=args.victory,
        )
        for spec in specs
    ]
    vec = VecEnv(envs)
    started = time.time()
    try:
        if not args.no_spawn:
            # 先把所有场地 spawn 好：否则第 1 批 reset 会串行等 M 次登录。
            vec.spawn_all()
        trainer_kwargs = {
            "epochs": args.ppo_epochs,
            "clip": args.clip,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "normalize_returns": args.normalize_returns,
            "lr_decay": args.lr_decay,
            "entropy_floor": args.entropy_floor,
            "entropy_coef_max": args.entropy_coef_max,
            "target_entropy": args.target_entropy,
            "minibatch": args.minibatch,
            "max_grad_norm": args.max_grad_norm,
            "value_clip": args.value_clip,
        }
        trainer = VecTrainer(
            vec, policy,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            seed=args.seed,
            workers=args.workers,
            algo=args.algo,
            trainer_kwargs=trainer_kwargs,
            max_steps=args.max_steps,
        )
        print(
            f"[vec] 开始：{args.episodes} 批 × {vec.count} 场地 = "
            f"{args.episodes * vec.count} 局；PPO epochs={args.ppo_epochs} clip={args.clip} "
            f"lr={args.lr} targetEntropy={args.target_entropy}"
        )
        history = trainer.train(args.episodes)
        summary = trainer.summary()
    finally:
        vec.close_all()

    duration = time.time() - started
    policy_path = paths["policy"]
    policy.save(policy_path, extra={
        "vec": {
            "arenas": vec.count,
            "episodes": summary["episodes"],
            "steps": summary["steps"],
            "workers": args.workers,
        },
        "stepsPerSecond": summary["stepsPerSecond"],
        "episodes": summary["episodes"],
        "seed": args.seed,
    })

    config = _config_dict(args, center, jitter)
    meta = {
        "schema": "gladiator.vec-run/v1",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runDir": run_dir,
        "arenas": [spec.to_dict() for spec in specs],
        "config": config,
        "summary": summary,
        "policy": policy_path,
        "durationSeconds": round(duration, 2),
    }
    _write_json(os.path.join(run_dir, "meta.json"), meta)
    save_report(paths["report"], {
        "kind": "vec",
        "runDir": run_dir,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "durationSeconds": round(duration, 2),
        "config": config,
        "arenas": [spec.to_dict() for spec in specs],
        "history": history,
        "summary": summary,
        "policyPath": policy_path,
        "metaPath": os.path.join(run_dir, "meta.json"),
    })

    print(
        f"[vec] 保存 {policy_path}；meta {os.path.join(run_dir, 'meta.json')}；"
        f"report {paths['report']}"
    )
    print(
        f"[vec] steps/s={summary['stepsPerSecond']} episodes={summary['episodes']} "
        f"updates={summary['policyUpdates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
