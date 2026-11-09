"""G-501/G-502 并发压测：N 个 Bot 持续随机混战，采样服务端 TPS 与插件侧堆占用。

用法（真实服务器）::

    python scripts/stress.py --minutes 30 --users Glad_Bot_1,Glad_Bot_2,Glad_Bot_3,Glad_Bot_4 \
        --ports 8766,8767,8768,8769

前提：服务端已开 RCON；每个用户名对应一个已启动的 bot-runtime（通道 B 端口见 --ports）。
脚本自身不做规则判定，只做「跑动 + 采样」，结果写入 JSON，便于对比不同机器/配置。
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv, default_reward
from gladiator_env.rcon import RconClient


def parse_center(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("center 需要 x,y,z")
    return parts[0], parts[1], parts[2]


def parse_float(text: str) -> float | None:
    """从 Paper 的 ``tps`` 输出里取最近 1 分钟的值。

    原始输出形如 ``§6TPS from last 1m, 5m, 15m: §a*20.0, §a19.98, §a19.99``。
    """
    import re

    cleaned = re.sub(r"§.", "", text)
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def parse_heap(text: str) -> tuple[int, int] | None:
    """从 ``/gladiator status`` 输出里取 ``heap used=NNMB max=MMMB``。"""
    import re

    match = re.search(r"heap used=(\d+)MB max=(\d+)MB", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gladiator 并发压测（G-501）")
    parser.add_argument("--minutes", type=float, default=30.0, help="压测时长（分钟）")
    parser.add_argument("--users", default="Glad_Bot_1,Glad_Bot_2,Glad_Bot_3,Glad_Bot_4")
    parser.add_argument("--ports", default="8766,8767,8768,8769")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rcon-port", type=int, default=25575)
    parser.add_argument("--rcon-password", default="devpass")
    parser.add_argument("--center", type=parse_center, default=(0.5, 4.0, 0.5))
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=400, help="单个 episode 最大步数")
    parser.add_argument("--sample-seconds", type=float, default=10.0, help="TPS 采样间隔")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", default="stress-report.json")
    parser.add_argument("--no-heap", action="store_true", help="跳过 /gladiator status 的堆采样")
    parser.add_argument("--no-spawn", action="store_true", help="Bot 已在线时跳过 spawn")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    users = [item.strip() for item in args.users.split(",") if item.strip()]
    ports = [int(item) for item in args.ports.split(",") if item.strip()]
    if len(users) != len(ports):
        raise SystemExit("--users 与 --ports 数量必须一致")

    bots = [
        BotConfig(username=user, env_url=f"ws://127.0.0.1:{port}")
        for user, port in zip(users, ports)
    ]
    arena = ArenaConfig(center=args.center, spacing=args.spacing)
    env = GladiatorEnv(
        bots,
        arena,
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        max_steps=args.max_steps,
        reward_fn=default_reward,
        auto_spawn=not args.no_spawn,
    )
    rcon = RconClient(host=args.host, port=args.rcon_port, password=args.rcon_password)
    try:
        rcon.connect()
    except Exception as exc:  # noqa: BLE001 - 压测仍可继续，只是没有 TPS 采样
        print(f"[warn] RCON 不可用，跳过 TPS 采样: {exc}")
        rcon = None

    started = time.time()
    deadline = started + args.minutes * 60.0
    next_sample = started
    tps_samples: list[float] = []
    heap_samples: list[int] = []
    episodes = 0
    steps = 0
    crashes = 0

    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    print(f"[stress] bots={users} 目标时长={args.minutes:.1f} 分钟")

    try:
        while time.time() < deadline:
            action = env.action_space.sample()
            try:
                _, _, terminated, truncated, _ = env.step(action)
            except Exception as exc:  # noqa: BLE001 - 压测记录后重建 episode
                crashes += 1
                print(f"[warn] step 异常，重建 episode: {exc}")
                env.reset()
                continue
            steps += 1

            now = time.time()
            if now >= next_sample:
                next_sample = now + args.sample_seconds
                if rcon is not None:
                    try:
                        value = parse_float(rcon.command("tps"))
                        if value is not None:
                            tps_samples.append(value)
                        if not args.no_heap:
                            heap = parse_heap(rcon.command("gladiator status"))
                            if heap is not None:
                                heap_samples.append(heap[0])
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warn] tps 采样失败: {exc}")

            if terminated or truncated:
                episodes += 1
                env.reset()

            if steps % 100 == 0:
                elapsed = now - started
                print(f"[stress] steps={steps} episodes={episodes} "
                      f"tps={tps_samples[-1] if tps_samples else 'n/a'} "
                      f"elapsed={elapsed:.0f}s")
    finally:
        env.close()
        if rcon is not None:
            rcon.close()

    duration = time.time() - started
    heap_start = median(heap_samples[:3]) if len(heap_samples) >= 3 else None
    heap_end = median(heap_samples[-3:]) if len(heap_samples) >= 3 else None
    heap_growth = None
    if heap_start and heap_end is not None:
        heap_growth = round((heap_end - heap_start) / heap_start * 100.0, 2)

    report = {
        "users": users,
        "durationSeconds": round(duration, 1),
        "steps": steps,
        "stepsPerSecond": round(steps / duration, 3) if duration else 0.0,
        "episodes": episodes,
        "crashes": crashes,
        "tpsSamples": len(tps_samples),
        "tpsAvg": round(sum(tps_samples) / len(tps_samples), 3) if tps_samples else None,
        "tpsMin": min(tps_samples) if tps_samples else None,
        "tpsMax": max(tps_samples) if tps_samples else None,
        "heapSamples": len(heap_samples),
        "heapStartMb": heap_start,
        "heapEndMb": heap_end,
        "heapMaxMb": max(heap_samples) if heap_samples else None,
        "heapGrowthPercent": heap_growth,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[stress] 完成: " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
