"""多场地并行训练：一次拉起 N 组「学习方 + 对手」，场地沿 X 轴分开。

每个 Bot 仍然是一个独立的 `bot-runtime` 进程（通道 B 端口 8766, 8767, 8768, ...），
本脚本只是把 N 个 `GladiatorEnv` + `Trainer` 放进线程里同时跑，互不干扰。

前置（每组 2 个 Bot，共 2N 个）：

```powershell
# 端口 8766..8766+2N-1 各起一个运行时，用户名与 --users-prefix 对应
node src/index.js --username Bot_A1 --host 127.0.0.1 --mc-port 25565 --env-port 8766
...
# RCON 需开启（控制面：摆位/回血/清背包）
```

用法：

```powershell
python -m gladiator_train.multi_train --pairs 2 --episodes 4 --max-steps 200 --auto-spawn
```

每个场地中心默认沿 X 轴间隔 60 格（`--arena-gap`），避免两场互相看到、互相攻击。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.reward_presets import load_preset
from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM

from .policy import Policy, adopt_weights, validate_init_policy
from .runs import new_run_dir, run_paths
from .trainer import Trainer

OPPONENT_MODES = ("random", "wander", "idle", "scripted", "rusher", "aimbot")


@dataclass
class PairResult:
    pair: int
    users: List[str]
    episodes: int = 0
    steps: int = 0
    mean_reward: float = 0.0
    hits: int = 0
    whiffs: int = 0
    error: str | None = None
    history: List[dict] = field(default_factory=list)


def parse_stages(spec: str) -> list[tuple[str, int]]:
    """把 `idle:2,wander:3,rusher:5` 解析成 [(mode, episodes), ...]。"""
    plan: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        mode, _, count = chunk.partition(":")
        mode = mode.strip()
        if mode not in OPPONENT_MODES:
            raise ValueError(f"未知对手风格: {mode}")
        try:
            episodes = int(count) if count else 1
        except ValueError as exc:
            raise ValueError(f"回合数必须是整数: {chunk}") from exc
        plan.append((mode, episodes))
    if not plan:
        raise ValueError("--stages 为空")
    return plan


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_policies(args, results: list[PairResult], run_dir: str) -> dict:
    """把可用的多个场地权重做融合，写 ``run_dir/policy.npz``。

    返回写入 ``report.merge`` 的 dict：``method`` / ``pairs`` / ``mergedSha256`` 等。
    ``mean`` 等权平均；``weighted`` 按各场地 ``steps`` 加权；``none`` 不融合。
    """
    method = args.merge or "mean"
    info: dict = {
        "method": method,
        "pairs": [],
        "mergedPolicyPath": None,
        "mergedSha256": None,
        "available": False,
        "skipped": False,
    }
    if method == "none":
        info["error"] = "merge=none：未执行融合（dashboard 端也不会持久化写回）"
        info["skipped"] = True
        return info

    # 只取「该场地本身训练成功」且「权重文件存在」的 pair。
    ok = [r for r in results if r.error is None]
    if not ok:
        info["error"] = "没有可用的场地权重（全部失败），无法融合"
        info["skipped"] = True
        return info
    loaded = []
    for r in ok:
        path = os.path.join(args.save_dir, f"policy-pair{r.pair + 1}.npz")
        if not os.path.isfile(path):
            continue
        try:
            policy = Policy.load(path)
        except Exception as exc:  # noqa: BLE001 - 权重损坏不应拖垮整个 run
            print(f"[multi] 融合跳过 pair{r.pair + 1}：{type(exc).__name__}: {exc}")
            continue
        loaded.append((r, policy, path))

    if not loaded:
        info["error"] = "所有场地的权重文件都缺失/损坏，无法融合"
        info["skipped"] = True
        return info

    keys = Policy.PARAM_NAMES
    weights = []
    arrays = {k: [] for k in keys}
    for _r, policy, path in loaded:
        for k in keys:
            arrays[k].append(np.asarray(policy.params[k], dtype=np.float64))
        steps = _r.steps
        # weighted：按步数加权；steps<=0 时退化为等权，避免除零/负权。
        weights.append(float(steps) if (method == "weighted" and steps > 0) else 1.0)
    total_w = float(sum(weights)) or 1.0

    avg = {}
    for k in keys:
        stacked = np.stack(arrays[k])  # (n, *param_shape)
        # 把权重向量广播到 (n, *1...) 以对齐 param 维度数。
        weight_vec = np.asarray(weights, dtype=np.float64).reshape(-1, *([1] * (stacked.ndim - 1)))
        avg[k] = np.sum(stacked * weight_vec, axis=0) / total_w

    first = loaded[0][1]
    merged = Policy(obs_dim=int(first.obs_dim), n_actions=int(first.n_actions), hidden=int(first.hidden))
    for k in keys:
        merged.params[k] = avg[k]
    merged_path = os.path.join(run_dir, "policy.npz")
    merged.save(merged_path, extra={"merge": method, "pairs": [r.pair + 1 for r, _, _ in loaded]})

    info["pairs"] = [
        {"pair": r.pair + 1, "steps": r.steps, "sha256": _sha256_file(path)}
        for r, _p, path in loaded
    ]
    info["mergedPolicyPath"] = merged_path
    info["mergedSha256"] = _sha256_file(merged_path)
    info["available"] = True
    print(f"[multi] 融合完成：{method} × {len(loaded)} 个场地 → {merged_path} "
          f"(sha256 {info['mergedSha256'][:12]}…)")
    return info


def run_pair(args, pair: int, results: list[PairResult]) -> None:
    tag = f"[pair{pair}]"
    users = [f"{args.users_prefix}_{pair + 1}A", f"{args.users_prefix}_{pair + 1}B"]
    ports = [args.ports_base + pair * 2, args.ports_base + pair * 2 + 1]
    center = (args.center[0] + pair * args.arena_gap, args.center[1], args.center[2])
    result = PairResult(pair=pair, users=users)
    results.append(result)
    try:
        bots = [
            BotConfig(
                username=users[0],
                env_url=f"ws://{args.env_host}:{ports[0]}",
                cps=args.cps,
                cps_jitter=args.cps_jitter,
                reaction_ms=args.reaction_ms,
                reaction_jitter_ms=args.reaction_jitter_ms,
                reach=args.reach,
                model=args.model,
                wtap=args.wtap,
                auto_rod=args.auto_rod,
                auto_block=args.auto_block,
                aim_lock=args.aim_lock,
            ),
            BotConfig(
                username=users[1],
                env_url=f"ws://{args.env_host}:{ports[1]}",
                cps=args.opp_cps,
                cps_jitter=args.opp_cps_jitter,
                reaction_ms=args.opp_reaction_ms,
                reaction_jitter_ms=args.opp_reaction_jitter_ms,
                reach=args.reach,
                model="opponent",
                wtap=args.opp_assist,
                auto_rod=args.opp_assist,
                auto_block=args.opp_assist,
                aim_lock=args.opp_assist,
            ),
        ]
        env = GladiatorEnv(
            bots,
            ArenaConfig(center=center, spacing=args.spacing, prep_seconds=args.prep_seconds),
            rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
            action_ticks=args.action_ticks,
            max_steps=args.max_steps,
            single_agent=True,
            auto_spawn=args.auto_spawn,
            opponent_mode=args.opponent_mode,
            reward_preset=load_preset(args.reward_preset) if args.reward_preset else None,
        )
        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        policy = Policy(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden,
                        seed=args.seed + pair, lr=args.lr)
        init_path = getattr(args, "init_policy", "") or ""
        loaded, init_error = validate_init_policy(
            init_path, obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden)
        if init_error:
            raise RuntimeError(init_error)
        if loaded is not None:
            # 只接管权重，不接管 optimizer/updates。
            adopt_weights(policy, loaded)
            print(f"{tag} 已加载初始策略 {args.model_name or args.model_id or init_path}")
        review_dir = os.path.join(args.review_dir, f"pair-{pair + 1}")
        trainer = Trainer(env, policy, gamma=args.gamma, seed=args.seed + pair, review_dir=review_dir)
        print(f"{tag} 场地 center={center} users={users} ports={ports}")
        plan = args.plan if args.plan else [(args.opponent_mode, args.episodes)]
        epochs = max(1, int(args.epochs))
        history: list[dict] = []
        # 回合（epoch）→ 计划（plan）→ 轮次（count）：每轮次一局对局，结束即更新一次参数。
        for epoch in range(1, epochs + 1):
            print(f"{tag} 回合 {epoch}/{epochs}")
            for mode, count in plan:
                env.opponent_mode = mode
                print(f"{tag} 阶段 {mode} × {count} 轮次")
                for entry in trainer.train(count):
                    entry["opponentMode"] = mode
                    entry["epoch"] = epoch
                    entry["episode"] = len(history) + 1
                    history.append(entry)
        env.close()
        result.episodes = len(history)
        result.steps = sum(h["steps"] for h in history)
        result.mean_reward = sum(h["totalReward"] for h in history) / max(1, len(history))
        result.hits = sum((h.get("review") or {}).get("hits", 0) for h in history)
        result.whiffs = sum((h.get("review") or {}).get("whiffs", 0) for h in history)
        result.history = history
        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
        policy.save(os.path.join(args.save_dir, f"policy-pair{pair + 1}.npz"),
                    extra={"pair": pair + 1, "users": users})
        print(f"{tag} 完成：{result.episodes} episodes / {result.steps} steps / "
              f"mean reward {result.mean_reward:.3f} / 命中 {result.hits} 空挥 {result.whiffs}")
    except Exception as exc:  # 一个场地失败不影响其它场地
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"{tag} 失败：{result.error}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多场地并行训练")
    p.add_argument("--pairs", type=int, default=2, help="场地数（每场 2 个 Bot）")
    p.add_argument("--users-prefix", default="Bot_P")
    p.add_argument("--ports-base", type=int, default=8766)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--env-host", default="127.0.0.1")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", default="devpass")
    p.add_argument("--center", default="0.5,4,0.5")
    p.add_argument("--arena-gap", type=float, default=60.0, help="场地中心之间的 X 间距")
    p.add_argument("--spacing", type=float, default=6.0)
    p.add_argument("--prep-seconds", type=float, default=3.0, help="开局准备期（秒），期间双方不动")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1,
                   help="训练回合数：每回合完整按计划执行一遍（每轮次更新一次参数）")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--action-ticks", type=int, default=2)
    p.add_argument("--opponent-mode", default="rusher", choices=list(OPPONENT_MODES))
    p.add_argument("--stages", default="",
                   help="课程：mode:回合数 逗号分隔，例如 idle:2,wander:3,rusher:5（优先生效）")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--auto-spawn", action="store_true")
    p.add_argument("--cps", type=float, default=15.0)
    p.add_argument("--cps-jitter", type=float, default=2.0)
    p.add_argument("--reaction-ms", type=float, default=220.0)
    p.add_argument("--reaction-jitter-ms", type=float, default=40.0)
    p.add_argument("--reach", type=float, default=3.99)
    p.add_argument("--model", default="default")
    p.add_argument("--mc-port", type=int, default=25566, help="服务器端口（运行时已自行连接，仅记录）")
    p.add_argument("--wtap", action="store_true")
    p.add_argument("--auto-rod", action="store_true")
    p.add_argument("--auto-block", action="store_true")
    p.add_argument("--aim-lock", action="store_true")
    p.add_argument("--opp-assist", action="store_true")
    p.add_argument("--opp-cps", type=float, default=16.0)
    p.add_argument("--opp-cps-jitter", type=float, default=2.0)
    p.add_argument("--opp-reaction-ms", type=float, default=180.0)
    p.add_argument("--opp-reaction-jitter-ms", type=float, default=30.0)
    p.add_argument("--run-dir", default=None,
                   help="本次运行的产物目录；默认 artifacts/runs/<时间>-multi-<前缀>")
    p.add_argument("--save-dir", default=None, help="策略输出目录；默认 <run-dir>")
    p.add_argument("--init-policy", default="",
                   help="初始策略 npz（模型注册表导出）；每个场地用它初始化权重（不加载优化器状态）")
    p.add_argument("--model-id", default="", help="模型注册表 id（仅写入 report/config）")
    p.add_argument("--model-name", default="", help="模型显示名（仅写入 report/config）")
    p.add_argument("--merge", default="mean", choices=("mean", "weighted", "none"),
                   help="多场地权重融合：mean=等权平均，weighted=按 steps 加权，"
                        "none=不融合（不写 run/policy.npz，dashboard 端也不会持久化写回）")
    p.add_argument("--persist-to", default="",
                   help="持久化目标模型 id（仅写入 report/config；写回由 Dashboard 负责）")
    p.add_argument("--reward-preset", default=None,
                   help="reward 预设 JSON 路径（见 training/gladiator_env/reward_presets.py）；"
                        "缺省时使用内置默认权重")
    p.add_argument("--review-dir", default=None, help="复盘目录；默认 <run-dir>/reviews")
    p.add_argument("--report", default=None, help="报告；默认 <run-dir>/report.json")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 启动任何场地之前先验初始策略：无效直接 exit 2（不建目录、不起线程）。
    _loaded, init_error = validate_init_policy(
        args.init_policy, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden)
    if init_error:
        print(f"[multi] {init_error}", file=sys.stderr)
        return 2
    run_dir = args.run_dir or new_run_dir("multi", args.users_prefix)
    paths = run_paths(run_dir, "multi", pairs=args.pairs)
    args.save_dir = args.save_dir or run_dir
    args.review_dir = args.review_dir or paths["reviews"]
    args.report = args.report or paths["report"]
    print(f"[multi] 运行目录 {run_dir}")
    args.center = tuple(float(v) for v in args.center.split(","))
    if len(args.center) != 3:
        print("--center 需要 x,y,z")
        return 2
    args.plan = parse_stages(args.stages) if args.stages else []
    if args.plan:
        args.episodes = sum(count for _, count in args.plan)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.review_dir, exist_ok=True)

    started = time.time()
    results: list[PairResult] = []
    threads = [threading.Thread(target=run_pair, args=(args, i, results), name=f"pair{i}") for i in range(args.pairs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda r: r.pair)
    merge_info = _merge_policies(args, results, run_dir)
    report = {
        "runDir": run_dir,
        "kind": "multi",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "startedAtEpoch": round(started, 3),
        "durationSeconds": round(time.time() - started, 1),
        "config": {
            "pairs": args.pairs,
            "portsBase": args.ports_base,
            "center": list(args.center),
            "arenaGap": args.arena_gap,
            "episodes": args.episodes,
            "maxSteps": args.max_steps,
            "opponentMode": args.opponent_mode,
            "initPolicy": (args.init_policy or None),
            "modelId": (args.model_id or None),
            "modelName": (args.model_name or None),
            "persistTo": (args.persist_to or None),
        },
        "pairs": [
            {
                "pair": r.pair + 1,
                "users": r.users,
                "episodes": r.episodes,
                "steps": r.steps,
                "meanReward": round(r.mean_reward, 3),
                "hits": r.hits,
                "whiffs": r.whiffs,
                "error": r.error,
                "history": r.history,
            }
            for r in results
        ],
        "merge": merge_info,
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    total_steps = sum(r.steps for r in results)
    ok = sum(1 for r in results if r.error is None)
    print(f"[multi] {ok}/{args.pairs} 场地完成，{total_steps} steps，{report['durationSeconds']}s")
    print(f"[multi] 报告 {args.report}")
    return 0 if ok == args.pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
