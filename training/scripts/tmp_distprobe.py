"""临时诊断：逐帧打印 teacher v3 的距离/动作/掉血，用来看清真机的击退与贴回速度。"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
from gladiator_env.spaces import ACTION_NAMES, OBS_FIELDS
from gladiator_train.demos import make_teacher

IDX_DIST = OBS_FIELDS.index("target_distance")
IDX_REACH = OBS_FIELDS.index("in_reach")

def main() -> int:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    teacher = make_teacher("v3")
    meta = {"actionSpace": "v1"}
    env = GladiatorEnv(
        [BotConfig("Bot_V0A", "ws://127.0.0.1:8770"), BotConfig("Bot_V0B", "ws://127.0.0.1:8771")],
        ArenaConfig(center=(-16.0, 4.0, 0.0), spacing=4.0, jitter=(0.0, 0.0), prep_seconds=1.0),
        opponent_policy=str(ROOT.parent / "dashboard" / "var" / "models" / "dagger-1118b" / "policy.npz"),
        rcon={"host": "127.0.0.1", "port": 25575, "password": "devpass"},
        auto_spawn=False, single_agent=True, max_steps=200,
    )
    for ep in range(episodes):
        teacher.reset()
        obs, _ = env.reset(seed=5300 + ep)
        prev_hp = None
        print(f"\n--- ep{ep} ---")
        for step in range(200):
            d = float(np.asarray(obs).reshape(-1)[IDX_DIST]) * 32.0
            reach = float(np.asarray(obs).reshape(-1)[IDX_REACH])
            action = teacher(obs, meta)
            obs, _r, term, trunc, _i = env.step(action)
            try:
                hp = float(env._raw[1]["self"]["health"])
            except Exception:
                hp = prev_hp
            dmg = 0.0 if prev_hp is None or hp is None else max(0.0, prev_hp - hp)
            print(f"{step:3d} {ACTION_NAMES[action]:12s} {meta.get('reason',''):15s} "
                  f"d={d:4.2f} reach={reach:.0f} dmg={dmg:.2f}")
            prev_hp = hp
            if term or trunc:
                break
    env.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
