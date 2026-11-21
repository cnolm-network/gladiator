"""临时：v2 策略 vs v1 dagger 的跨动作空间镜像对决（dsh 自查用，不入库）。

为什么需要它：`mirror_eval.py` 明确拒绝 v2（它的对手路径只取 v1 动作），
而 G-710 的产物是 v2 十头策略，要和主线 `dagger-1118b` 比强弱只能自己搭一次。

做法：
- 学习方 = v2 策略（`bc-v4-v2-1/policy.npz`），发 `{type:"intent"}`；
- 对手 = v1 `dagger-1118b`，发 v1 宏动作（env 的 `use_intent` 只对 i==0 生效，
  所以同局混用两套动作是被支持的）；
- 交替左右位：前 N 局学习方站在 A 侧，后 N 局换到 B 侧，抵消出生点偏差。

用法（training/ 下）：:

    .venv\\Scripts\\python.exe scripts\\tmp_duel_v2.py \
        --v2 artifacts/runs/bc-v4-v2-1/policy.npz \
        --v1 ..\\dashboard\\var\\models\\dagger-1118b\\policy.npz \
        --episodes 12 --seed 9000 --max-steps 400 --no-spawn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import OBS_DIM  # noqa: E402
from gladiator_train.eval_suite import (  # noqa: E402
    format_interval,
    reset_with_retry,
    rollout,
    summarize,
)
from gladiator_train.policy import load_for_obs_dim  # noqa: E402


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
    ap = argparse.ArgumentParser(description="v2 vs v1 跨动作空间镜像")
    ap.add_argument("--v2", required=True, help="v2 策略 npz（学习方）")
    ap.add_argument("--v1", required=True, help="v1 策略 npz（对手，通常是 dagger-1118b）")
    ap.add_argument("--episodes", type=int, default=12, help="每侧局数（总局数 = 2×）")
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--users", nargs=2, default=["Bot_V0A", "Bot_V0B"])
    ap.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rcon", default="127.0.0.1:25575:devpass")
    ap.add_argument("--no-spawn", action="store_true")
    ap.add_argument("--out", default=str(TRAINING_DIR / "artifacts" / "evals"))
    args = ap.parse_args(argv)

    learner = load_for_obs_dim(str(Path(args.v2)), OBS_DIM)
    opponent = load_for_obs_dim(str(Path(args.v1)), OBS_DIM)
    assert getattr(learner, "action_space", "v1") == "v2", "学习方必须是 v2 模型"
    assert getattr(opponent, "action_space", "v1") == "v1", "对手必须是 v1 模型"
    rcon = parse_rcon(args.rcon)

    rows: list = []
    for side in ("A", "B"):
        users = [args.users[0], args.users[1]] if side == "A" else [args.users[1], args.users[0]]
        ports = list(args.ports) if side == "A" else list(reversed(args.ports))
        env = GladiatorEnv(
            [BotConfig(users[0], f"ws://{args.host}:{ports[0]}"),
             BotConfig(users[1], f"ws://{args.host}:{ports[1]}")],
            ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
            single_agent=True,
            max_steps=args.max_steps,
            auto_spawn=not args.no_spawn,
            rcon=rcon or None,
            opponent_mode="scripted",
            opponent_policy=opponent,
            action_space="v2",
        )
        try:
            for index in range(int(args.episodes)):
                seed = int(args.seed) + index
                obs, info = reset_with_retry(env, seed, attempts=4, delay=1.0)
                rng = __import__("numpy").random.default_rng(seed)
                normalizer = getattr(learner, "normalize_mask", None)
                raw_mask = (info or {}).get("actionMask")
                mask = normalizer(raw_mask) if (normalizer and raw_mask is not None) else None
                result = rollout(env, learner, obs, max_steps=args.max_steps,
                                 rng=rng, greedy=False, username=users[0], mask=mask)
                winner = result.get("winner")
                outcome = ("win" if winner == users[0] else
                           "loss" if winner == users[1] else "draw")
                rows.append({
                    "side": side,
                    "episodeIndex": index,
                    "seed": seed,
                    "outcome": outcome,
                    "win": outcome == "win",
                    "reward": float(result["reward"]),
                    "steps": int(result["steps"]),
                    "damageDealt": float(result["damageDealt"]),
                    "damageTaken": float(result["damageTaken"]),
                    "winner": winner,
                })
                print(f"[duel] side={side} ep={index} {outcome} steps={result['steps']} "
                      f"dmg={result['damageDealt']:.0f}:{result['damageTaken']:.0f}", flush=True)
        finally:
            env.close()

    summary = summarize(rows)
    overall = summary["overall"]
    print(f"\n[v2 vs v1] {overall['episodes']} 局  胜/负/平 "
          f"{overall['wins']}/{overall['losses']}/{overall['draws']}")
    print(f"  胜率 {overall['winRate']*100:.1f}%  95% CI "
          f"{format_interval(overall['winRateCI'])}")
    print(f"  均 reward {overall['meanReward']:.2f}  均步数 {overall['meanSteps']:.0f}")
    by_side = {}
    for row in rows:
        slot = by_side.setdefault(row["side"], {"wins": 0, "episodes": 0})
        slot["wins"] += int(row["win"])
        slot["episodes"] += 1
    for side, slot in by_side.items():
        print(f"  侧 {side}: {slot['wins']}/{slot['episodes']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    target = out_dir / f"{stamp}-duel-v2-vs-v1.json"
    target.write_text(json.dumps({
        "v2": str(args.v2), "v1": str(args.v1),
        "episodes": int(args.episodes), "seed": int(args.seed),
        "summary": summary, "rows": rows,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  产物 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
