"""固定评估套件 CLI（v2）：统计功效 + 配对比较。

它回答两个问题：

1. **这个策略有多强？** 每对手 + 总体胜率，带 Wilson 95% 区间（不是单点百分比）。
2. **新策略比基线更好吗？** 两个策略跑**同一批 seed**，按 (对手, 局号, seed) 配对，
   用精确 McNemar 看「A 赢 B 输 / A 输 B 赢」的分歧局；p < 0.05 才算显著更好。

用法::

    # 单策略（24 局 = 8 局 × 3 对手，可下结论的最低样本量）
    python scripts/eval_suite.py --policy ../dashboard/var/models/dagger-1118b/policy.npz \
        --no-spawn --users Bot_V50A Bot_V50B

    # 双策略配对比较（两个策略会用完全相同的 seed 集）
    python scripts/eval_suite.py --policy A.npz --policy-b B.npz --no-spawn

约定（沿用 ``eval_policy.py`` 已经踩过的坑）：
- Bot 由调用方先 spawn；``--no-spawn`` 时不再走 RCON spawn；
- reset 失败（Bot 未就绪）重试 4 次，seed 用 ``seed + attempt``；
- 只用 ``reset``/``step`` 返回的 obs，不读 env 内部状态。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.eval_suite import (  # noqa: E402
    SIGNIFICANCE,
    SUITE_DEFAULT,
    SuiteConfig,
    format_interval,
    pair_rows,
    run_suite,
    summarize,
    verdict,
)
from gladiator_train.policy import Policy  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = TRAINING_DIR / "artifacts" / "evals"
OPPONENT_MODES = ("random", "wander", "idle", "scripted", "rusher", "aimbot")


def parse_rcon(spec: str) -> dict:
    """``host:port[:password]`` → GladiatorEnv 的 rcon dict。"""
    raw = str(spec or "").strip()
    if not raw:
        return {}
    parts = raw.split(":")
    host = parts[0] or "127.0.0.1"
    port = int(parts[1]) if len(parts) > 1 and parts[1] else 25575
    password = ":".join(parts[2:]) if len(parts) > 2 else "devpass"
    return {"host": host, "port": port, "password": password}


def parse_opponents(text: str) -> tuple:
    items = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not items:
        raise SystemExit("--opponents 不能为空")
    unknown = [item for item in items if item not in OPPONENT_MODES]
    if unknown:
        raise SystemExit(f"未知对手模式: {', '.join(unknown)}（可选: {', '.join(OPPONENT_MODES)}）")
    # 去重但保持输入顺序
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定评估套件 v2：Wilson 区间 + 同 seed 配对 McNemar",
    )
    parser.add_argument("--policy", required=True, help="策略 A 的 .npz 路径")
    parser.add_argument("--policy-b", default=None, help="策略 B（给出则做配对比较）")
    parser.add_argument("--episodes", type=int, default=SUITE_DEFAULT["episodes"],
                        help=f"每个对手的局数（默认 {SUITE_DEFAULT['episodes']}）")
    parser.add_argument("--opponents", default=",".join(SUITE_DEFAULT["opponents"]),
                        help="逗号分隔的对手模式（默认 scripted,rusher,aimbot）")
    parser.add_argument("--seed", type=int, default=SUITE_DEFAULT["seed"],
                        help=f"起始 seed（默认 {SUITE_DEFAULT['seed']}）；第 i 局用 seed+i")
    parser.add_argument("--max-steps", type=int, default=SUITE_DEFAULT["maxSteps"],
                        help=f"每局最多步数（默认 {SUITE_DEFAULT['maxSteps']}）")
    parser.add_argument("--sampled", action="store_true",
                        help="按策略采样评估（默认贪心 argmax，可复现性更好）")
    parser.add_argument("--users", nargs=2, default=["Bot_V50A", "Bot_V50B"],
                        help="两个 Bot 的用户名（第一个是被评估方）")
    parser.add_argument("--ports", nargs=2, type=int, default=[8770, 8771],
                        help="两个 Bot 的通道 B 端口")
    parser.add_argument("--host", default="127.0.0.1", help="通道 B 主机")
    parser.add_argument("--mc-port", type=int, default=25566, help="Minecraft 端口")
    parser.add_argument("--rcon", default="127.0.0.1:25575:devpass",
                        help="RCON host:port:password")
    parser.add_argument("--no-spawn", action="store_true",
                        help="不自动 spawn（Bot 已经在跑）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录（默认 training/artifacts/evals）")
    return parser


def make_env_factory(args: argparse.Namespace, config: SuiteConfig):
    bots = [
        BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
        BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
    ]
    rcon = parse_rcon(args.rcon)

    def factory(mode: str):
        return GladiatorEnv(
            bots,
            ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
            single_agent=True,
            max_steps=config.max_steps,
            auto_spawn=not args.no_spawn,
            rcon=rcon or None,
            opponent_mode=mode,
        )

    return factory


def print_summary(title: str, summary: dict) -> None:
    print(f"\n[{title}]")
    header = (f"{'对手':<10}{'局数':>5}{'胜/负/平':>12}{'胜率':>8}"
              f"{'95% CI':>20}{'均 reward':>12}{'均步数':>8}{'伤害比':>8}")
    print(header)
    print("-" * 84)
    for name, stats in list(summary.get("opponents", {}).items()) + [("总体", summary.get("overall", {}))]:
        win_rate = f"{stats.get('winRate', 0.0) * 100:.1f}%"
        triple = f"{stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('draws', 0)}"
        print(
            f"{name:<10}{stats.get('episodes', 0):>5}{triple:>12}{win_rate:>8}"
            f"{format_interval(stats.get('winRateCI', (0.0, 1.0))):>20}"
            f"{stats.get('meanReward', 0.0):>12.2f}{stats.get('meanSteps', 0.0):>8.0f}"
            f"{stats.get('damageRatio', 0.0):>8.2f}"
        )


def print_comparison(label_a: str, label_b: str, comparison: dict) -> None:
    print(f"\n[配对比较] {label_a} vs {label_b}（同样 {comparison.get('pairs', 0)} 个 seed）")
    header = f"{'对手':<10}{'A 独赢':>8}{'B 独赢':>8}{'都赢/都输':>12}{'p 值':>10}  判定"
    print(header)
    rows = list(comparison.get("byOpponent", {}).items()) + [("总体", comparison)]
    for name, item in rows:
        p = float(item.get("p", 1.0))
        b = int(item.get("b", 0))
        c = int(item.get("c", 0))
        if name == "总体":
            both = f"{comparison.get('bothWin', 0)}/{comparison.get('bothLose', 0)}"
        else:
            both = "-"
        judgement = "不显著" if p >= SIGNIFICANCE or b == c else ("A 更好" if b > c else "B 更好")
        print(f"{name:<10}{b:>8}{c:>8}{both:>12}{p:>10.4f}  {judgement}")
    print(f"结论：{verdict(comparison, label_a, label_b)}")


def _load_policy(path: str) -> Policy:
    target = Path(path)
    if not target.is_file():
        raise SystemExit(f"策略文件不存在: {target}")
    return Policy.load(str(target))


def _write_json(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    opponents = parse_opponents(args.opponents)
    if args.episodes < 1:
        raise SystemExit("--episodes 必须 ≥ 1")
    config = SuiteConfig(
        opponents=opponents,
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        greedy=not args.sampled,
    )
    if config.total_episodes < 24:
        print(f"[提示] 总样本 {config.total_episodes} 局 < 24：按 docs/training-log.md 的口径，"
              "不足 24 局不下结论。")

    policy_a = _load_policy(args.policy)
    policy_b = _load_policy(args.policy_b) if args.policy_b else None
    factory = make_env_factory(args, config)
    started = time.strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    rows_a = run_suite(factory, policy_a, config, logger=print)
    summary_a = summarize(rows_a)
    print_summary(f"策略 A · {Path(args.policy).name}", summary_a)

    if policy_b is None:
        payload = {
            "schema": "gladiator.eval-suite/v1",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": config.to_dict(),
            "policy": str(Path(args.policy).resolve()),
            "rows": rows_a,
            "summary": summary_a,
        }
        path = _write_json(out_dir, f"{started}-suite.json", payload)
        print(f"\n已写入 {path}")
        return 0

    rows_b = run_suite(factory, policy_b, config, logger=print)
    summary_b = summarize(rows_b)
    print_summary(f"策略 B · {Path(args.policy_b).name}", summary_b)
    comparison = pair_rows(rows_a, rows_b)
    print_comparison(Path(args.policy).name, Path(args.policy_b).name, comparison)
    payload = {
        "schema": "gladiator.eval-compare/v1",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config.to_dict(),
        "policyA": str(Path(args.policy).resolve()),
        "policyB": str(Path(args.policy_b).resolve()),
        "rowsA": rows_a,
        "rowsB": rows_b,
        "summaryA": summary_a,
        "summaryB": summary_b,
        "comparison": comparison,
    }
    path = _write_json(out_dir, f"{started}-compare.json", payload)
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
