"""Gymnasium 契约自检（G-401）。

两种模式：

```bash
# 离线：用假通道 B 校验 spaces / reset / step 的 Gymnasium 形状（不需要服务端）
.venv/Scripts/python.exe scripts/check_env.py

# 真连：连本地 Paper 训练场，校验真实观测/动作
.venv/Scripts/python.exe scripts/check_env.py --live --users Glad_Bot_1,Glad_Bot_2 --ports 8766,8767
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymnasium.utils.env_checker import check_env

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.testing import FakeLink, make_observation, make_target


def build_fake_env() -> GladiatorEnv:
    links = [FakeLink("Glad_Bot_1"), FakeLink("Glad_Bot_2")]
    links[0].set_observation(make_observation(
        "Glad_Bot_1", x=-1.5, targets=[make_target(2, "Glad_Bot_2", x=1.5, distance=3.0)]))
    links[1].set_observation(make_observation(
        "Glad_Bot_2", x=1.5, targets=[make_target(1, "Glad_Bot_1", x=-1.5, distance=3.0)]))
    return GladiatorEnv(
        [BotConfig("Glad_Bot_1"), BotConfig("Glad_Bot_2")],
        links=links, control_hz=200.0, action_ticks=1, max_steps=10, connect=True,
        single_agent=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="连接真实训练场")
    parser.add_argument("--users", default="Glad_Bot_1,Glad_Bot_2")
    parser.add_argument("--ports", default="8766,8767")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rcon-port", type=int, default=25575)
    parser.add_argument("--rcon-password", default="devpass")
    parser.add_argument("--center", default="0.5,4,0.5")
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--skip-render-check", action="store_true", default=True)
    args = parser.parse_args()

    if not args.live:
        env = build_fake_env()
        check_env(env, skip_render_check=args.skip_render_check)
        env.close()
        print("Gymnasium check_env 通过（离线假通道 B）")
        return 0

    users = [u.strip() for u in args.users.split(",") if u.strip()]
    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    if len(users) != len(ports):
        parser.error("--users 与 --ports 数量必须一致")
    center = tuple(float(v) for v in args.center.split(","))
    env = GladiatorEnv(
        [BotConfig(user, f"ws://{args.host}:{port}") for user, port in zip(users, ports)],
        arena=ArenaConfig(center=center, spacing=args.spacing, settle_timeout=15.0),
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        max_steps=20,
    )
    try:
        check_env(env, skip_render_check=args.skip_render_check)
        print("Gymnasium check_env 通过（真实训练场）")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
