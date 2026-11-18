"""镜像评估：新模型 vs 旧模型（两边都跑策略，交替左右位）。

固定对手课程已经被打到 100%（scripted/rusher/aimbot），再用它评估没有分辨力。
镜像对局把「谁更强」变成可测的：双方观测/动作空间一样，只有参数不同，
交替左右位抵消出生点偏差，最终用精确二项检验判断胜率是否显著高于 50%。

用法::

    .venv\\Scripts\\python.exe scripts\\mirror_eval.py \\
        --policy artifacts/branches/selfplay-final/policy.npz \\
        --policy-b ..\\dashboard\\var\\models\\dagger-1118b\\policy.npz \\
        --episodes 16 --seed 1200 --no-spawn --users Bot_V0A Bot_V0B
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.eval_suite import SUITE_DEFAULT, wilson_interval  # noqa: E402


def parse_rcon(spec: str) -> dict | None:
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--rcon 需要 host:port[:password]，收到 {spec!r}")
    return {"host": parts[0], "port": int(parts[1]),
            "password": parts[2] if len(parts) > 2 else "devpass"}


def binom_p_two_sided(wins: int, losses: int) -> float:
    """精确二项检验（p=0.5）的双侧 p 值；样本小的时候不 approximations。"""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="镜像评估：新模型 vs 旧模型（交替左右位）")
    parser.add_argument("--policy", required=True, help="新模型 .npz")
    parser.add_argument("--policy-b", required=True, help="旧模型 .npz（作为对手）")
    parser.add_argument("--episodes", type=int, default=SUITE_DEFAULT["episodes"],
                        help="总对局数（会交替左右位，默认 8 偶数）")
    parser.add_argument("--seed", type=int, default=SUITE_DEFAULT["seed"])
    parser.add_argument("--max-steps", type=int, default=SUITE_DEFAULT["maxSteps"])
    parser.add_argument("--users", nargs=2, default=["Bot_V0A", "Bot_V0B"])
    parser.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mc-port", type=int, default=25566)
    parser.add_argument("--rcon", default="127.0.0.1:25575:devpass")
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="两边都用 argmax 动作（去掉随机性）")
    parser.add_argument("--out", default="artifacts/evals")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bots = [
        BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
        BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
    ]
    rcon = parse_rcon(args.rcon)

    rows = []
    for episode in range(args.episodes):
        # 交替左右位：偶数局新模型当 A（学习者），奇数局旧模型当 A。
        new_is_a = episode % 2 == 0
        learner = args.policy if new_is_a else args.policy_b
        opponent = args.policy_b if new_is_a else args.policy
        env = GladiatorEnv(
            bots,
            ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
            single_agent=True,
            max_steps=args.max_steps,
            auto_spawn=not args.no_spawn,
            rcon=rcon,
            opponent_policy=opponent,
            opponent_epsilon=0.0,
        )
        from gladiator_train.policy import Policy

        policy = Policy.load(learner)
        rng = np.random.default_rng(args.seed + episode)
        try:
            observation, info = env.reset(seed=args.seed + episode)
            done = False
            steps = 0
            final_info = info
            while not done and steps < args.max_steps:
                action = policy.act(observation, rng, greedy=args.greedy)
                observation, _reward, terminated, truncated, final_info = env.step(action)
                done = bool(terminated or truncated)
                steps += 1
            winner = (final_info or {}).get("winner")
            learner_name = bots[0].username
            if winner == learner_name:
                outcome = "new" if new_is_a else "old"
            elif winner is None:
                outcome = "draw"
            else:
                outcome = "old" if new_is_a else "new"
            rows.append({"episode": episode, "newIsA": new_is_a, "steps": steps,
                         "winner": winner, "outcome": outcome,
                         "reward": (final_info or {}).get("reward")})
            print(f"[mirror] ep{episode + 1}/{args.episodes} newIsA={new_is_a} "
                  f"steps={steps} winner={winner} -> {outcome}", flush=True)
        finally:
            env.close()

    wins = sum(1 for row in rows if row["outcome"] == "new")
    losses = sum(1 for row in rows if row["outcome"] == "old")
    draws = sum(1 for row in rows if row["outcome"] == "draw")
    total = len(rows)
    win_rate = wins / total if total else 0.0
    p = binom_p_two_sided(wins, losses)
    summary = {
        "policy": str(args.policy),
        "baseline": str(args.policy_b),
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": win_rate,
        "winRateCI": wilson_interval(wins, total),
        "p": p,
        "significant": p < 0.05 and wins > losses,
        "rows": rows,
    }
    print()
    print(f"[mirror] 新模型 {wins}胜 {losses}负 {draws}平（共 {total} 局，胜率 {win_rate * 100:.1f}%，"
          f"CI {summary['winRateCI'][0] * 100:.1f}–{summary['winRateCI'][1] * 100:.1f}%，p={p:.4f}）")
    verdict = ("新模型显著更强" if summary["significant"]
               else ("旧模型显著更强" if p < 0.05 and losses > wins else "没有显著差异"))
    print(f"[mirror] 结论：{verdict}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{__import__('time').strftime('%Y%m%d-%H%M')}-mirror.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mirror] 结果写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
