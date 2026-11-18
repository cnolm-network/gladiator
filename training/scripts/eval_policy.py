"""离线评估一个策略文件：只跑评估局，不训练、不写回。

用法::

    python scripts/eval_policy.py --policy ../dashboard/var/models/default/policy.npz --episodes 12

约定：Bot 由调用方（Dashboard 或手工）先 spawn，脚本用 ``--ports`` 连各自的通道 B；
``--no-spawn`` 表示不通过 RCON 再 spawn 一次（默认就是 False，即会尝试 auto_spawn）。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.eval_suite import reset_with_retry, rollout  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估一个 .npz 策略文件（只跑评估局）")
    parser.add_argument("--policy", required=True, help="策略文件路径（.npz）")
    parser.add_argument("--users", nargs="+", default=["Bot_E1", "Bot_E2"], help="两个 Bot 的用户名")
    parser.add_argument("--ports", nargs="+", type=int, default=[8770, 8771], help="两个 Bot 的通道 B 端口")
    parser.add_argument("--host", default="127.0.0.1", help="通道 B 主机")
    parser.add_argument("--mc-port", type=int, default=25566, help="Minecraft 端口（RCON 用）")
    parser.add_argument("--episodes", type=int, default=12, help="采样评估局数")
    parser.add_argument("--greedy", type=int, default=12, help="贪心评估局数")
    parser.add_argument("--max-steps", type=int, default=400, help="每局最多步数")
    parser.add_argument("--seed", type=int, default=303, help="随机种子")
    parser.add_argument("--opponent-mode", default="scripted",
                        help="对手模式：scripted/rusher/aimbot/random/wander/idle")
    parser.add_argument("--reward-preset", default=None, help="reward 预设 JSON（可选）")
    parser.add_argument("--rcon-port", type=int, default=25575, help="RCON 端口")
    parser.add_argument("--rcon-password", default="devpass", help="RCON 密码")
    parser.add_argument("--no-spawn", action="store_true", help="不自动 spawn（Bot 已经在跑）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = Policy.load(args.policy)
    bots = [
        BotConfig(username=args.users[0], env_url=f"ws://{args.host}:{args.ports[0]}"),
        BotConfig(username=args.users[1], env_url=f"ws://{args.host}:{args.ports[1]}"),
    ]
    preset = None
    if args.reward_preset:
        from gladiator_env.reward_presets import load_preset

        preset = load_preset(args.reward_preset)

    env = GladiatorEnv(
        bots,
        ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
        single_agent=True,
        max_steps=args.max_steps,
        auto_spawn=not args.no_spawn,
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        opponent_mode=args.opponent_mode,
        reward_preset=preset,
    )
    try:
        rng = env.np_random
        for label, greedy in (("采样", False), ("贪心", True)):
            count = args.episodes if label == "采样" else args.greedy
            rows = []
            for _ in range(count):
                # 坑（与 eval_suite 共用同一份实现）：Bot 未就绪时 reset 会超时，
                # 重试要用新的 seed，否则同一局的地图抖动会重复。
                try:
                    obs, _info = reset_with_retry(env, args.seed, logger=print)
                except RuntimeError as exc:
                    raise SystemExit(str(exc)) from exc
                result = rollout(env, policy, obs, max_steps=args.max_steps,
                                 rng=rng, greedy=greedy)
                rows.append((result["steps"], result["reward"], result["terminated"]))
            wins = sum(1 for _s, _r, term in rows if term)
            print(
                f"[eval/{label}] {wins}/{count} 胜, mean reward {statistics.mean(r[1] for r in rows):.2f}, "
                f"mean steps {statistics.mean(r[0] for r in rows):.0f}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
