"""随机策略空手混战（G-406 的最小验收脚本）。

前置：本地 Paper 服务端已启动、开了 RCON、两个 bot-runtime 进程分别监听 8766/8767。
脚本会 spawn、摆位、清背包、回满，然后跑 --steps 步随机动作，打印血量/回合统计。

```bash
.venv/Scripts/python.exe scripts/random_brawl.py --steps 200
```
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="Glad_Bot_1,Glad_Bot_2")
    parser.add_argument("--ports", default="8766,8767")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rcon-port", type=int, default=25575)
    parser.add_argument("--rcon-password", default="devpass")
    parser.add_argument("--center", default="0.5,4,0.5")
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-spawn", action="store_true", help="Bot 已在服内时跳过 spawn")
    args = parser.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]
    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    if len(users) != len(ports):
        parser.error("--users 与 --ports 数量必须一致")
    center = tuple(float(v) for v in args.center.split(","))

    env = GladiatorEnv(
        [BotConfig(user, f"ws://{args.host}:{port}") for user, port in zip(users, ports)],
        arena=ArenaConfig(center=center, spacing=args.spacing, settle_timeout=20.0),
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        auto_spawn=not args.no_spawn,
        max_steps=args.steps,
        action_ticks=2,
    )
    env.action_space.seed(args.seed)

    started = time.time()
    obs, info = env.reset(seed=args.seed)
    print(f"reset 完成: {info}")

    episode = 1
    try:
        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            if step % 20 == 0 or terminated:
                health = " ".join(f"{name}={hp:.1f}" for name, hp in info["health"].items())
                reward = " ".join(f"{k}={v:+.2f}" for k, v in rewards.items())
                print(f"[step {step:3d}] {health} | {reward} | winner={info['winner']}")
            if terminated or truncated:
                print(f"episode {episode} 结束：terminated={terminated} truncated={truncated} "
                      f"steps={info['step']} winner={info['winner']}")
                if truncated or step >= args.steps:
                    break
                episode += 1
                _, info = env.reset()
    finally:
        env.close()

    print(f"完成 {episode} 个 episode / {args.steps} 步，用时 {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
