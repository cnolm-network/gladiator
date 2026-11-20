"""临时诊断：teacher v3 当学习者，逐刀看掉血，判断服务端是否认暴击。

用法：
    .venv\\Scripts\\python scripts\\tmp_critprobe.py --episodes 3 --max-steps 300
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
from gladiator_env.spaces import ACTION_NAMES  # noqa: E402
from gladiator_train.demos import make_teacher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--policy", default=str(ROOT.parent / "dashboard" / "var" / "models" / "dagger-1118b" / "policy.npz"))
    args = ap.parse_args()

    teacher = make_teacher("v3")
    meta = {"actionSpace": "v1"}
    env = GladiatorEnv(
        [BotConfig("Bot_V0A", "ws://127.0.0.1:8770"), BotConfig("Bot_V0B", "ws://127.0.0.1:8771")],
        ArenaConfig(center=(-16.0, 4.0, 0.0), spacing=8.0, jitter=(0.0, 0.0)),
        opponent_policy=args.policy,
        rcon={"host": "127.0.0.1", "port": 25575, "password": "devpass"},
        auto_spawn=False,
        single_agent=True,
        max_steps=args.max_steps,
    )

    hits: list[tuple[str, float]] = []
    for ep in range(args.episodes):
        teacher.reset()
        obs, info = env.reset(seed=args.seed + ep)
        try:
            prev_hp = float(env._raw[1]["self"]["health"])
        except Exception:
            prev_hp = info.get("opponentHealth")
        last_label = "?"
        for step in range(args.max_steps):
            action = teacher(obs, meta)
            last_label = ACTION_NAMES[action] if isinstance(action, int) else str(action)
            obs, reward, term, trunc, info = env.step(action)
            raw = getattr(env, "_raw", None)
            try:
                hp = float(raw[1]["self"]["health"])
            except Exception:
                hp = info.get("opponentHealth")
            if prev_hp is not None and hp is not None and hp < prev_hp - 1e-6:
                reason = meta.get("reason", last_label)
                crit = meta.get("crit", False)
                hits.append((f"{last_label}/{reason}{'/CRIT' if crit else ''}", round(prev_hp - hp, 2)))
            prev_hp = hp
            if term or trunc:
                break
        print(f"[ep{ep}] steps={step + 1} hits={len(hits)} reasons={teacher.reasons}", flush=True)
    env.close()

    if not hits:
        print("没有任何掉血记录")
        return 1
    by: dict[str, list[float]] = defaultdict(list)
    for label, dmg in hits:
        by[label].append(dmg)
    print("\n=== 每一刀掉血（按 动作/规则/是否暴击） ===")
    for label, vals in sorted(by.items(), key=lambda kv: -len(kv[1])):
        vals_sorted = sorted(vals)
        print(f"{label:>34}: n={len(vals):>3} mean={sum(vals) / len(vals):.2f} min={vals_sorted[0]:.2f} max={vals_sorted[-1]:.2f}")
    print(f"\n总刀数 {len(hits)}")
    print(f"动作分布 {Counter(l.split('/')[0] for l, _ in hits).most_common()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
