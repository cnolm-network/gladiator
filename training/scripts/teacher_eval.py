"""评估「脚本老师」本身有多强：让老师当对手，看被评估策略能否打过它。

背景：BC / DAgger 的上限就是老师。此前用老师演示训出来的策略镜像 0/24，
需要先回答「老师自己能不能打赢 dagger-1118b」，否则继续加演示量没有意义。

用法::

    cd training
    .venv\\Scripts\\python.exe scripts\\teacher_eval.py --policy ..\\dashboard\\var\\models\\dagger-1118b\\policy.npz \\
        --teacher v2 --episodes 8 --users Bot_V0A Bot_V0B --ports 8770 8771 --no-spawn

输出：被评估策略对老师的胜率（Wilson 区间 + 精确二项），写 ``artifacts/evals/<时间>-vs-teacher.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    TEACHER_VERSION,
    heuristic_action,
    make_teacher,
    reset_teacher_v2,
)
from gladiator_train.eval_suite import (  # noqa: E402
    SUITE_DEFAULT,
    SuiteConfig,
    format_interval,
    run_suite,
    summarize,
)
from gladiator_train.policy import Policy  # noqa: E402


class TeacherPolicy:
    """把脚本老师包成 ``Policy.act`` 的接口，供 env 的 ``opponent_policy`` 使用。

    v2 老师自带跨帧状态：这里用 :class:`TeacherV2` 实例持有它，并在环境 ``reset()``
    时通过 ``reset_episode()`` 归档上一局的规则命中统计——否则不同局的节拍会互相污染，
    也拿不到「谁一直在生效」的证据。
    """

    def __init__(self, version: str = TEACHER_VERSION) -> None:
        self.version = version
        self.meta = {"actionSpace": "v1"}
        self.decisions = 0
        self.teacher = make_teacher(version) if version == "v2" else None

    def act(self, vector, rng=None, greedy: bool = True) -> int:
        self.decisions += 1
        if self.version == "v2" and self.teacher is not None:
            return int(self.teacher(vector, self.meta))
        return int(heuristic_action(vector))

    def reset_episode(self) -> None:
        """环境每局 ``reset()`` 时回调：归档上一局的 reason 统计并清零跨帧状态。"""
        if self.teacher is not None:
            self.teacher.reset()

    def flush(self) -> None:
        """对局跑完后把最后一局也归档（否则它只有一次 ``reset_episode``，永远进不了列表）。"""
        if self.teacher is not None:
            self.teacher.flush()

    def reason_stats(self) -> dict:
        """聚合统计：``{"totals": {...}, "perEpisode": [...]}``。"""
        if self.teacher is None:
            return {"totals": {}, "perEpisode": []}
        episodes = list(self.teacher.episode_reasons)
        totals: dict[str, int] = {}
        for item in episodes:
            for key, value in item.items():
                totals[key] = totals.get(key, 0) + int(value)
        return {"totals": totals, "perEpisode": episodes}

    # 自博弈统计里只用到 act；其余接口留空避免 AttributeError。
    def value(self, x):  # noqa: D401 - 兼容性占位
        return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估脚本老师作为对手时的强度")
    parser.add_argument("--policy", required=True, help="被评估策略的 .npz 路径")
    parser.add_argument("--teacher", default=TEACHER_VERSION, choices=("v1", "v2"))
    parser.add_argument("--episodes", type=int, default=SUITE_DEFAULT["episodes"])
    parser.add_argument("--seed", type=int, default=SUITE_DEFAULT["seed"])
    parser.add_argument("--max-steps", type=int, default=SUITE_DEFAULT["maxSteps"])
    parser.add_argument("--sampled", action="store_true", help="按策略采样评估（默认贪心）")
    parser.add_argument("--users", nargs=2, default=["Bot_V50A", "Bot_V50B"])
    parser.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mc-port", type=int, default=25566)
    parser.add_argument("--rcon", default="127.0.0.1:25575:devpass")
    parser.add_argument("--no-spawn", action="store_true", help="Bot 已手动 spawn 时使用")
    parser.add_argument("--out", default=str(TRAINING_DIR / "artifacts" / "evals"))
    return parser


def parse_rcon(spec: str) -> dict:
    parts = str(spec or "").split(":")
    if not parts or not parts[0]:
        return {}
    return {
        "host": parts[0],
        "port": int(parts[1]) if len(parts) > 1 and parts[1] else 25575,
        "password": ":".join(parts[2:]) if len(parts) > 2 else "devpass",
    }


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy_path = Path(args.policy)
    if not policy_path.is_file():
        raise SystemExit(f"策略文件不存在: {policy_path}")
    policy = Policy.load(str(policy_path))
    rcon = parse_rcon(args.rcon)

    teacher = TeacherPolicy(args.teacher)

    def factory(_mode: str) -> GladiatorEnv:
        bots = [
            BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
            BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
        ]
        return GladiatorEnv(
            bots,
            ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
            single_agent=True,
            max_steps=args.max_steps,
            auto_spawn=not args.no_spawn,
            rcon=rcon or None,
            opponent_mode="scripted",
            opponent_policy=teacher,
        )

    config = SuiteConfig(
        opponents=("scripted",),  # factory 忽略 mode，对手固定是老师
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        greedy=not args.sampled,
    )
    reset_teacher_v2()
    rows = run_suite(factory, policy, config)
    teacher.flush()
    reasons = teacher.reason_stats()
    summary = summarize(rows)
    overall = summary.get("overall", {})
    print(f"\n[策略 vs 老师 {args.teacher}] {policy_path.name}")
    print(f"  局数 {overall.get('episodes', 0)}  胜/负/平 "
          f"{overall.get('wins', 0)}/{overall.get('losses', 0)}/{overall.get('draws', 0)}")
    print(f"  胜率 {overall.get('winRate', 0.0) * 100:.1f}%  "
          f"95% CI {format_interval(overall.get('winRateCI', (0.0, 1.0)))}")
    print(f"  均 reward {overall.get('meanReward', 0.0):.2f}  均步数 {overall.get('meanSteps', 0):.0f}")
    if reasons["totals"]:
        ordered = sorted(reasons["totals"].items(), key=lambda kv: -kv[1])
        print("  老师规则命中（全部局合计）：" + ", ".join(f"{k}={v}" for k, v in ordered))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    target = out_dir / f"{stamp}-vs-teacher.json"
    target.write_text(json.dumps({
        "policy": str(policy_path),
        "teacher": args.teacher,
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "greedy": not args.sampled,
        "summary": summary,
        "rows": rows,
        # 每条规则的命中次数：用来判断「是不是一直在后撤 / 一直在格挡」。
        "reasons": reasons["totals"],
        "episodeReasons": reasons["perEpisode"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
