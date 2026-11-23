r"""镜像评估：新模型 vs 旧模型（两边都跑策略，交替左右位）。

固定对手课程已经被打到 100%（scripted/rusher/aimbot），再用它评估没有分辨力。
镜像对局把「谁更强」变成可测的：只有参数不同，交替左右位抵消出生点偏差，
最终用精确二项检验判断胜率是否显著高于 50%。

**动作空间（G-703）**：两边可以是**不同**的动作空间（v1/v2 混打），各按自己的
``_meta.actionSpace`` 分发：

- 学习者（坐在 bots[0] 的那一方）：``action_space=`` 用它的（v1 → ``Discrete(11)``，
  v2 → ``MultiDiscrete``），取动作走 ``model_io.policy_action_for_env``（v2 带 action mask）；
- 对手：交给 ``GladiatorEnv(opponent_policy=...)``，env 自己按 ``is_v2`` 决定走
  ``decode_intent``（10 头）还是 v1 宏动作编号。

所以「v1 vs v2」「v2 vs v2」「v1 vs v1」都能跑；输出里会写明两边各是什么空间
——混打的胜率必须带着这个上下文看（否则会把「动作空间差异」读成「模型强弱」）。

用法::

    .venv\Scripts\python.exe scripts\mirror_eval.py \
        --policy ..\dashboard\var\models\v2-v4-dagger3\policy.npz \
        --policy-b ..\dashboard\var\models\v2-v4-dagger2\policy.npz \
        --episodes 16 --seed 1200 --no-spawn --users Bot_V0A Bot_V0B

    # 只校验模型与参数（服务器没开时先跑这个）
    .venv\Scripts\python.exe scripts\mirror_eval.py --policy A.npz --policy-b B.npz --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.eval_suite import SUITE_DEFAULT, wilson_interval  # noqa: E402
from gladiator_train.model_io import (  # noqa: E402
    ModelLoadError,
    ModelRecord,
    env_action_space,
    load_model,
    model_label,
    normalize_action_mask,
    policy_action_for_env,
)


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
    parser.add_argument("--policy", required=True, help="新模型 .npz（v1 / v2 均可）")
    parser.add_argument("--policy-b", required=True, help="旧模型 .npz（v1 / v2 均可）")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="只做模型加载与参数校验，不开对局（服务器没开时用）")
    return parser


def model_table(new: ModelRecord, old: ModelRecord) -> str:
    """两行小表：谁是什么动作空间（混打时必须先看这个）。"""
    lines = ["| 角色 | 模型 | 动作空间 | 头结构 | obsDim |",
             "|:--|:--|:--:|:--|---:|"]
    for role, record in (("新", new), ("旧", old)):
        sizes = "/".join(str(size) for size in record.head_sizes)
        lines.append(f"| {role} | {record.label} | {record.action_space} | {sizes} "
                     f"| {record.obs_label()} |")
    return "\n".join(lines)


def run_episode(bots, rcon, args, learner: ModelRecord, opponent: ModelRecord,
                seed: int) -> dict:
    """跑一局：``learner`` 坐 bots[0]（单智能体学习者），``opponent`` 当对手策略。"""
    env = GladiatorEnv(
        bots,
        ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
        single_agent=True,
        max_steps=args.max_steps,
        auto_spawn=not args.no_spawn,
        rcon=rcon,
        opponent_policy=opponent.policy,
        opponent_epsilon=0.0,
        action_space=env_action_space(learner),
    )
    rng = np.random.default_rng(seed)
    steps = 0
    try:
        observation, info = env.reset(seed=seed)
        mask = normalize_action_mask(learner.policy, (info or {}).get("actionMask"))
        done = False
        final_info = info
        while not done and steps < args.max_steps:
            action = policy_action_for_env(learner.policy, observation, mask, rng,
                                           greedy=args.greedy)
            observation, _reward, terminated, truncated, final_info = env.step(action)
            mask = normalize_action_mask(learner.policy, (final_info or {}).get("actionMask"))
            done = bool(terminated or truncated)
            steps += 1
        winner = (final_info or {}).get("winner")
    finally:
        env.close()
    return {"steps": steps, "winner": winner,
            "learnerSpace": learner.action_space, "opponentSpace": opponent.action_space}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes 至少为 1")
    try:
        new_model = load_model(args.policy, label=model_label(args.policy))
        old_model = load_model(args.policy_b, label=model_label(args.policy_b))
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[mirror] 新模型 {new_model.label}（{new_model.action_space}） vs "
          f"旧模型 {old_model.label}（{old_model.action_space}），"
          f"{args.episodes} 局（交替左右位）", flush=True)
    if args.dry_run:
        print()
        print(model_table(new_model, old_model))
        print()
        print(f"[mirror] dry-run 结束（未开对局；episodes={args.episodes} "
              f"seed={args.seed} maxSteps={args.max_steps}）")
        return 0

    bots = [
        BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
        BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
    ]
    rcon = parse_rcon(args.rcon)
    learner_name = bots[0].username

    rows = []
    for episode in range(args.episodes):
        # 交替左右位：偶数局新模型当学习者（bots[0]），奇数局旧模型当学习者。
        new_is_a = episode % 2 == 0
        learner = new_model if new_is_a else old_model
        opponent = old_model if new_is_a else new_model
        result = run_episode(bots, rcon, args, learner, opponent, args.seed + episode)
        winner = result["winner"]
        if winner == learner_name:
            outcome = "new" if new_is_a else "old"
        elif winner is None:
            outcome = "draw"
        else:
            outcome = "old" if new_is_a else "new"
        rows.append({"episode": episode, "newIsA": new_is_a, "steps": result["steps"],
                     "winner": winner, "outcome": outcome,
                     "learnerSpace": result["learnerSpace"],
                     "opponentSpace": result["opponentSpace"]})
        print(f"[mirror] ep{episode + 1}/{args.episodes} newIsA={new_is_a} "
              f"learner={learner.label}({learner.action_space}) "
              f"opponent={opponent.label}({opponent.action_space}) "
              f"steps={result['steps']} winner={winner} -> {outcome}", flush=True)

    wins = sum(1 for row in rows if row["outcome"] == "new")
    losses = sum(1 for row in rows if row["outcome"] == "old")
    draws = sum(1 for row in rows if row["outcome"] == "draw")
    total = len(rows)
    win_rate = wins / total if total else 0.0
    p = binom_p_two_sided(wins, losses)
    summary = {
        "policy": str(args.policy),
        "baseline": str(args.policy_b),
        # G-703：两边动作空间进产物，混打胜率必须带这个上下文读。
        "actionSpaces": {"new": new_model.action_space, "old": old_model.action_space},
        "models": [
            {"role": "new", "label": new_model.label, "path": new_model.path,
             "actionSpace": new_model.action_space,
             "headSizes": list(new_model.head_sizes), "obsDim": new_model.obs_dim},
            {"role": "old", "label": old_model.label, "path": old_model.path,
             "actionSpace": old_model.action_space,
             "headSizes": list(old_model.head_sizes), "obsDim": old_model.obs_dim},
        ],
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
    print(f"[mirror] 新模型 {wins}胜 {losses}负 {draws}平（共 {total} 局，"
          f"胜率 {win_rate * 100:.1f}%，"
          f"CI {summary['winRateCI'][0] * 100:.1f}–{summary['winRateCI'][1] * 100:.1f}%，"
          f"p={p:.4f}）")
    verdict = ("新模型显著更强" if summary["significant"]
               else ("旧模型显著更强" if p < 0.05 and losses > wins else "没有显著差异"))
    print(f"[mirror] 结论：{verdict}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{time.strftime('%Y%m%d-%H%M')}-mirror.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mirror] 结果写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
