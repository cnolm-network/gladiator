"""评估「脚本老师」本身有多强：让老师当对手，看被评估策略能否打过它。

背景：BC / DAgger 的上限就是老师。此前用老师演示训出来的策略镜像 0/24，
需要先回答「老师自己能不能打赢 dagger-1118b」，否则继续加演示量没有意义。

用法::

    cd training
    .venv\\Scripts\\python.exe scripts\\teacher_eval.py --policy ..\\dashboard\\var\\models\\dagger-1118b\\policy.npz \\
        --teacher v3 --episodes 8 --users Bot_V0A Bot_V0B --ports 8770 8771 --no-spawn

``--teacher`` 可选 ``v1`` / ``v2`` / ``v3`` / ``v4``（默认 ``v2``）：v1 是静态四条规则、
v2 在 v1 之上加跳劈、v3 改成「冷却感知」（冷却没好不空砍，改用走位/跳劈/格挡）、
v4 跑在 **v2 动作空间（10 头）** 上、按 1.8 招式表做 wtap / block_hit 的组合。
同一份策略对四个老师各跑一遍，就能看出老师之间的强度差。

注意 v4 是**唯一**会向环境发 ``{type:"intent"}`` 的老师（其余三个发 v1 宏动作）：
``GladiatorEnv`` 通过 ``opponent_policy.is_v2`` 识别，学习方仍然是原来的动作空间。

输出：被评估策略对老师的胜率（Wilson 区间 + 精确二项），写 ``artifacts/evals/<时间>-vs-teacher.json``
（JSON 里的 ``teacher`` 字段记录本局用的是哪个版本）。
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
from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.demos import (  # noqa: E402
    TEACHER_VERSION,
    heuristic_action,
    make_teacher,
    reset_teacher_v2,
    reset_teacher_v3,
    reset_teacher_v4,
)
from gladiator_train.eval_suite import (  # noqa: E402
    SUITE_DEFAULT,
    SuiteConfig,
    format_interval,
    run_suite,
    summarize,
)
from gladiator_train.policy import load_for_obs_dim  # noqa: E402


class TeacherPolicy:
    """把脚本老师包成 ``Policy.act`` 的接口，供 env 的 ``opponent_policy`` 使用。

    v2/v3 老师自带跨帧状态：这里用 :class:`TeacherV2` / :class:`TeacherV3` 实例持有它，
    并在环境 ``reset()`` 时通过 ``reset_episode()`` 归档上一局的规则命中统计——否则不同局的
    节拍会互相污染，也拿不到「谁一直在生效」的证据。

    v1 是无状态函数，直接调用即可（``teacher`` 为 ``None``，reason 统计为空）。

    v4 是 v2 意图策略：它返回 10 头数组而不是动作编号，这里用 ``is_v2`` / ``action_space``
    告诉 ``GladiatorEnv``（见 env 的 ``_opponent_v2``），让它走 ``decode_intent`` 那条路径。
    """

    def __init__(self, version: str = TEACHER_VERSION, **teacher_defaults) -> None:
        self.version = version
        self.decisions = 0
        #: v1 是无状态函数（``make_teacher("v1")`` 返回函数本身），只有 v2/v3/v4 需要实例。
        self.teacher = None if version == "v1" else make_teacher(version)
        #: v4 → True：env 据此把对手的动作当意图解释。
        self.is_v2 = bool(getattr(self.teacher, "is_v2", False))
        self.action_space = "v2" if self.is_v2 else "v1"
        self.meta: dict = {"actionSpace": self.action_space, **teacher_defaults}
        if self.is_v2 and hasattr(self.teacher, "defaults"):
            self.teacher.defaults.update(teacher_defaults)

    def act(self, vector, rng=None, greedy: bool = True):
        self.decisions += 1
        if self.teacher is None:
            return int(heuristic_action(vector))
        value = self.teacher(vector, self.meta)
        if self.is_v2:
            # env 侧会把它交给 decode_intent；返回普通 list 便于 np.asarray 归一。
            return [int(item) for item in value]
        return int(value)

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

    def frame_seconds(self) -> float | None:
        """v3 自校准出来的「一帧多少秒」（v1/v2 没有这个能力，返回 ``None``）。

        这个值本身就是重要诊断：它等于 ``GladiatorEnv`` 实际的 ``action_ticks/control_hz``。
        如果它明显大于 0.05，说明「按 20Hz 写死的常数」全都偏了一倍——v3 会自己纠正，
        但报告里记下来，才能解释别的脚本为什么节拍不对。
        """
        state = getattr(self.teacher, "state", None)
        # v3 自校准出 `frame_seconds_est`；v4 直接记 `frame_seconds`（不做自校准）。
        estimate = getattr(state, "frame_seconds_est", None)
        return estimate if estimate is not None else getattr(state, "frame_seconds", None)

    # 自博弈统计里只用到 act；其余接口留空避免 AttributeError。
    def value(self, x):  # noqa: D401 - 兼容性占位
        return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估脚本老师作为对手时的强度")
    parser.add_argument("--policy", required=True, help="被评估策略的 .npz 路径")
    parser.add_argument(
        "--teacher", default="v2", choices=("v1", "v2", "v3", "v4"),
        help="当对手的脚本老师版本（v1 静态四条 / v2 加跳劈 / v3 冷却感知 / "
             "v4 招式表+wtap+block_hit，默认 v2）")
    parser.add_argument(
        "--teacher-no-sword", action="store_true",
        help="让 v4 走「无剑」分支（只用 wtap、不按右键）。其它版本忽略这个开关。")
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
    # 旧 23 维模型（dagger-1118b）必须能直接评估当前 25 维环境：右侧补零，
    # 旧特征留在原下标（与 --init-policy 同口径）。
    policy = load_for_obs_dim(str(policy_path), OBS_DIM)
    rcon = parse_rcon(args.rcon)

    teacher = TeacherPolicy(
        args.teacher, **({"has_sword": False} if args.teacher_no_sword else {}))

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
            # G-710：被评估的策略可能是 v2（十头）——环境必须用同一个动作空间建，
            # 否则 run_suite 的空间校验会直接拒绝（v2 策略 + Discrete 环境）。
            action_space=str(getattr(policy, "action_space", "v1")),
        )

    config = SuiteConfig(
        opponents=("scripted",),  # factory 忽略 mode，对手固定是老师
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        greedy=not args.sampled,
    )
    reset_teacher_v2()
    reset_teacher_v3()
    reset_teacher_v4()
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
    frame_seconds = teacher.frame_seconds()
    if frame_seconds:
        print(f"  实测步长 {frame_seconds * 1000:.0f}ms/步（v3 自校准）")

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
        # v3 自校准出来的真实步长（秒/帧）；v1/v2 为 null。
        # 它等于 GladiatorEnv 的 action_ticks/control_hz，是解释节拍的关键证据。
        "frameSeconds": frame_seconds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
