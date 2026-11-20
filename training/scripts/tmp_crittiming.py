"""临时诊断：确定「起跳后第几步砍」才出暴击（真机 0.204s/步）。

对每种延迟 k（1..4），重复若干次「jump → (k-1)×idle → attack」，
记录每次 attack 造成的掉血：1.60 = 普通，2.40 = 暴击。

与上一版的三处不同（上一版「没打到人」）：
1. **伤害晚一帧才进观测**：打完 attack 那一帧 `_raw` 里血量常常还没变，要再走一步再读。
   这个探针在 attack 之后多读一帧，取两者最小值。
2. **对手用 ``opponent_mode="idle"``**（站着不动的木桩）：不还手、不移动，距离稳定，
   不会把「没打中」和「打在无敌帧上」混在一起。
3. **校验跳跃真的生效**：读 ``self.pos.y``，起跳后 y 会 +0.3~1.2；顺便把
   「离地时机」和「掉血」对齐，才能看出服务端到底在哪一步认暴击。

用法::

    cd training
    .venv\\Scripts\\python -u scripts\\tmp_crittiming.py --repeats 6
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import ACTION_NAMES, OBS_FIELDS  # noqa: E402

IDX_DISTANCE = OBS_FIELDS.index("target_distance")
IDX_IN_REACH = OBS_FIELDS.index("in_reach")


def action_id(name: str) -> int:
    return list(ACTION_NAMES).index(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--max-delay", type=int, default=4)
    ap.add_argument("--seed", type=int, default=8100)
    ap.add_argument("--close-to", type=float, default=2.4, help="先走到这个距离以内再测")
    args = ap.parse_args()

    idle = action_id("idle")
    jump = action_id("jump")
    attack = action_id("attack")
    approach = action_id("approach")

    env = GladiatorEnv(
        [BotConfig("Bot_V0A", "ws://127.0.0.1:8770"), BotConfig("Bot_V0B", "ws://127.0.0.1:8771")],
        ArenaConfig(center=(-16.0, 4.0, 0.0), spacing=4.0, jitter=(0.0, 0.0), prep_seconds=1.0),
        # 木桩：站着不动、不还手。我们没有对手策略，距离完全由我们自己控制。
        opponent_mode="idle",
        rcon={"host": "127.0.0.1", "port": 25575, "password": "devpass"},
        auto_spawn=False,
        single_agent=True,
        max_steps=2000,
    )

    def opp_hp() -> float:
        try:
            return float(env._raw[1]["self"]["health"])
        except Exception:
            return float("nan")

    def my_y() -> float:
        try:
            return float(env._raw[0]["self"]["pos"]["y"])
        except Exception:
            return float("nan")

    def distance(obs) -> float:
        import numpy as np
        return float(np.asarray(obs).reshape(-1)[IDX_DISTANCE]) * 32.0

    def in_reach(obs) -> bool:
        import numpy as np
        return float(np.asarray(obs).reshape(-1)[IDX_IN_REACH]) >= 0.5

    rows: list[tuple[int, int, bool, float, float, float, float]] = []  # delay,rep,air,dist,dmg,dy,vy
    skipped = 0

    obs, _info = env.reset(seed=args.seed)
    for delay in range(1, args.max_delay + 1):
        for rep in range(args.repeats):
            # --- 1) 贴近到 close_to 以内（最多 10 步，避免被击退后一直追不上）---
            for _ in range(10):
                if distance(obs) <= args.close_to and in_reach(obs):
                    break
                obs, _r, term, trunc, _i = env.step(approach)
                if term or trunc:
                    obs, _info = env.reset(seed=args.seed + delay * 100 + rep)
            # --- 2) 等冷却：上一刀之后至少 3 步（0.6s > 500ms 无敌帧）---
            for _ in range(3):
                obs, _r, term, trunc, _i = env.step(idle)
                if term or trunc:
                    obs, _info = env.reset(seed=args.seed + delay * 100 + rep)

            dist_before = distance(obs)
            if dist_before > 3.6:
                skipped += 1
                continue

            hp0 = opp_hp()
            y0 = my_y()
            # --- 3) 起跳 ---
            obs, _r, term, trunc, _i = env.step(jump)
            # --- 4) 等到第 delay 步，同时记录竖直速度（下一步 y 减上一步 y）---
            y_prev = y0
            for _ in range(delay - 1):
                y_prev = my_y()
                obs, _r, term, trunc, _i = env.step(idle)
            y_now = my_y()
            vy = (y_now - y_prev) / 0.2037          # 格/秒；>0 上升、<0 下落
            airborne = y_now > y0 + 0.25
            dist_at_attack = distance(obs)
            # --- 5) 砍 ---
            obs, _r, term, trunc, _i = env.step(attack)
            hp_after_attack = opp_hp()
            obs, _r, term, trunc, _i = env.step(idle)   # 伤害常晚一帧才进观测
            hp_final = min(opp_hp(), hp_after_attack)

            dmg = (hp0 - hp_final) if hp0 == hp0 and hp_final == hp_final else 0.0
            rows.append((delay, rep, airborne, round(dist_at_attack, 2), round(max(dmg, 0.0), 2),
                         round(y_now - y0, 2), round(vy, 1)))
            if dmg <= 0.01:
                skipped += 1
            if term or trunc:
                obs, _info = env.reset(seed=args.seed + delay * 100 + rep)

    env.close()

    print("\ndelay  rep  离地?  砍时距离   掉血    Δy    vy(格/秒)")
    for delay, rep, air, dist, dmg, dy, vy in rows:
        print(f"{delay:>5}  {rep:>2}   {'YES' if air else ' no'}   {dist:>6}    {dmg:.2f}  {dy:>5}  {vy:>7}")

    by_delay: dict[int, list[float]] = defaultdict(list)
    by_air: dict[bool, list[float]] = defaultdict(list)
    for delay, _rep, air, _dist, dmg, _dy, _vy in rows:
        if dmg > 0.01:
            by_delay[delay].append(dmg)
            by_air[air].append(dmg)

    print("\n=== 按延迟汇总（只统计命中） ===")
    for delay in range(1, args.max_delay + 1):
        vals = by_delay.get(delay, [])
        if not vals:
            print(f"delay={delay}: 0 次命中")
            continue
        crits = sum(1 for v in vals if v >= 2.0)
        print(f"delay={delay}: n={len(vals)} 均值={sum(vals) / len(vals):.2f} "
              f"暴击={crits}/{len(vals)} ({crits / len(vals) * 100:.0f}%) 值={sorted(vals)}")

    print("\n=== 按是否离地汇总 ===")
    for air in (True, False):
        vals = by_air.get(air, [])
        if not vals:
            continue
        crits = sum(1 for v in vals if v >= 2.0)
        print(f"{'离地' if air else '未离地'}: n={len(vals)} 均值={sum(vals) / len(vals):.2f} "
              f"暴击={crits}/{len(vals)} ({crits / len(vals) * 100:.0f}%)")

    print(f"\n全部掉血取值: {dict(Counter(round(v, 2) for v in sum(by_delay.values(), [])))}")
    print(f"跳过/未命中: {skipped} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
