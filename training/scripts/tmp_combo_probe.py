"""临时诊断（G-701，不入库）：真机上「同一 tick 跳 + 砍」到底有没有生效。

为什么需要它：单测只能证明 `applyIntent` 把 `jump` 与 `attack` 都调用了，
**不能**证明服务端把这一帧当成一次有效的跳劈（伤害是否吃到暴击加成、
Bot 是否真的离地再下落）。这里直接看服务端判定的结果：

- 自己 `y` / `vy`：确认真的离地（`onGround=false`）且进入下落段（`vy<0`）；
- 对手掉血：普通一击 ≈1.60，暴击 ≈2.40。**出现 ≥2.2 的单次掉血**即证明
  「同帧起跳 + 出手」被服务端当成跳劈结算了（暴击需要下落中命中）。

用法（Bot 必须已在线；`--no-spawn` 表示不通过 RCON 再 spawn 一次）::

    cd training
    .venv\\Scripts\\python.exe scripts\\tmp_combo_probe.py --no-spawn --steps 80

先确认在线::

    node bot-runtime/tmp-rcon.mjs "gladiator list"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import MOVE_VALUES_V2  # noqa: E402

#: v2 九个头的取值下标（与 spaces.ACTION_HEADS_V2 的顺序一致）。
HEAD_MOVE = 0
HEAD_LOOK_YAW = 1
HEAD_LOOK_PITCH = 2
HEAD_ATTACK = 3
HEAD_USE = 4
HEAD_JUMP = 5
HEAD_SNEAK = 6
HEAD_SPRINT = 7
HEAD_SLOT = 8

#: 掉血判据：普通 ≈1.60，暴击 ≈2.40（钻石剑 + 钻石甲）。≥2.2 只可能是暴击。
CRIT_DAMAGE = 2.2
NORMAL_DAMAGE = 1.6


def combo_action(slot: int, move: str = "forward", attack: bool = True,
                 jump: bool = True) -> list[int]:
    """构造 `{move:'forward', jump:<bool>, attack:<bool>, use:0}` 的九头动作向量。

    ``jump`` 恒为 1 时：即使在半空下发也不会生效（原版只在着地时起跳），
    这正是「同一条意图里同时带 jump 与 attack」的写法——跳劈的组合权在模型手里。
    ``--no-jump`` 时把 jump 关掉，作为「同一距离、同一护甲」的落地对照组。
    """
    move_index = MOVE_VALUES_V2.index(move)
    action = [0] * 9
    action[HEAD_MOVE] = move_index
    action[HEAD_LOOK_YAW] = 3          # LOOK_YAW_V2 里的 0.0
    action[HEAD_LOOK_PITCH] = 2        # LOOK_PITCH_V2 里的 0.0
    action[HEAD_ATTACK] = 1 if attack else 0
    action[HEAD_USE] = 0               # 不按右键
    action[HEAD_JUMP] = 1 if jump else 0
    action[HEAD_SNEAK] = 0
    action[HEAD_SPRINT] = 0
    action[HEAD_SLOT] = max(0, min(8, slot))
    return action


def read_self(raw: dict) -> tuple[float, float, float, bool]:
    payload = (raw.get("self") or {})
    pos = payload.get("pos") or {}
    velocity = payload.get("velocity") or {}
    try:
        y = float(pos.get("y"))
    except (TypeError, ValueError):
        y = float("nan")
    try:
        vy = float(velocity.get("y"))
    except (TypeError, ValueError):
        vy = float("nan")
    try:
        health = float(payload.get("health"))
    except (TypeError, ValueError):
        health = float("nan")
    return y, vy, health, bool(payload.get("onGround"))


def _armor_remaining(raw: dict) -> float:
    try:
        return float((raw.get("self") or {}).get("armorRemaining"))
    except (TypeError, ValueError):
        return float("nan")


def nearest_distance(raw: dict) -> float | None:
    """观测里最近玩家的距离（entities 已按 distance 升序，取第一条 player）。"""
    for entity in raw.get("entities") or []:
        if not isinstance(entity, dict) or entity.get("kind") != "player":
            continue
        try:
            return float(entity.get("distance"))
        except (TypeError, ValueError):
            return None
    return None


def held_hotbar_slot(raw: dict) -> int:
    """当前手持物在热键栏的下标（0..8）；拿不到就 0。"""
    held = ((raw.get("self") or {}).get("heldItem") or {})
    slot = held.get("slot")
    try:
        slot = int(slot)
    except (TypeError, ValueError):
        return 0
    if 36 <= slot <= 44:
        return slot - 36
    if 0 <= slot <= 8:
        return slot
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80, help="下发帧数（每帧 0.2s @10Hz/2tick）")
    ap.add_argument("--users", nargs=2, default=["Bot_V0A", "Bot_V0B"])
    ap.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mc-port", type=int, default=25566)
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default="devpass")
    ap.add_argument("--center", default="-16,4,0")
    ap.add_argument("--spacing", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=701)
    ap.add_argument("--no-spawn", action="store_true", help="不自动 spawn（Bot 已在跑）")
    ap.add_argument("--all-frames", action="store_true", help="打印每一帧（默认只打变化帧）")
    ap.add_argument(
        "--attack-policy",
        choices=("always", "falling"),
        default="always",
        help="always=每帧都 attack（压力测试同帧组合）；"
             "falling=只在**下落段**（vy<0 且冷却已好）attack，用来验证跳劈暴击本身",
    )
    ap.add_argument("--no-jump", action="store_true",
                    help="不带 jump（落地对照组：同距离/同护甲下的普通伤害）")
    args = ap.parse_args()

    center = tuple(float(part) for part in args.center.split(","))
    if len(center) != 3:
        print(f"--center 需要 x,y,z 三个数，收到 {args.center}", file=sys.stderr)
        return 2

    env = GladiatorEnv(
        [
            BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
            BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
        ],
        ArenaConfig(center=center, spacing=args.spacing, jitter=(0.0, 0.0)),
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        auto_spawn=not args.no_spawn,
        single_agent=True,
        opponent_mode="idle",
        action_space="v2",
        max_steps=args.steps,
        connect=True,
        wait_action_results=False,
    )

    print(f"[probe] 动作空间=v2 single_agent opponent=idle center={center} "
          f"spacing={args.spacing} steps={args.steps}", flush=True)
    observation, info = env.reset(seed=args.seed)
    slot = held_hotbar_slot(env._raw[0])
    opp_armor = _armor_remaining(env._raw[1])
    print(f"[probe] 手持槽位={slot} 观测维度={observation.shape[0]} "
          f"对手护甲剩余={opp_armor:.2f}（判据要求满甲：普通 1.60 / 暴击 2.40）", flush=True)

    rows: list[dict] = []
    hits: list[float] = []
    start = time.monotonic()
    prev_hp = read_self(env._raw[1])[2]
    prev_vy: float | None = read_self(env._raw[0])[1]
    prev_cd = 1.0
    airborne_frames = 0
    falling_frames = 0
    combo_frames = 0

    for step in range(args.steps):
        # falling 模式：用**上一帧**观测判断「正在下落 + 冷却已好」，本帧仍然 jump=1。
        attack = True
        if args.attack_policy == "falling":
            attack = bool(prev_vy is not None and prev_vy < -0.01 and prev_cd <= 0.02)
        action = combo_action(slot, attack=attack, jump=not args.no_jump)
        observation, _reward, terminated, truncated, info = env.step(action)
        y, vy, own_hp, on_ground = read_self(env._raw[0])
        opp_hp = read_self(env._raw[1])[2]
        if not on_ground:
            airborne_frames += 1
            if vy < 0:
                falling_frames += 1
        if attack:  # 本帧确实同时下发了 jump=1 与 attack=1
            combo_frames += 1

        damage = 0.0
        if prev_hp is not None and opp_hp is not None and opp_hp < prev_hp - 1e-6:
            damage = round(prev_hp - opp_hp, 2)
            hits.append(damage)
        prev_hp = opp_hp

        rows.append({
            "step": step,
            "t": round(time.monotonic() - start, 2),
            "y": y,
            "vy": vy,
            "air": not on_ground,
            "dist": (lambda value: None if value is None else round(value, 2))(
                nearest_distance(env._raw[0])),
            "own_hp": own_hp,
            "opp_hp": opp_hp,
            "dmg": damage,
            "atk": attack,
            "obs_vy": float(observation[23]),
            "obs_cd": float(observation[24]),
        })
        prev_vy = vy
        prev_cd = float(observation[24])
        if terminated or truncated:
            print(f"[probe] 回合结束于 step={step + 1}（terminated={terminated} truncated={truncated}）",
                  flush=True)
            break

    env.close()

    print("\n=== 逐帧记录（每帧都带 jump=1；atk 列=本帧是否也带 attack=1） ===")
    header = (f"{'step':>4} {'t(s)':>6} {'y':>7} {'vy':>7} {'air':>4} {'atk':>4} {'dist':>5} "
              f"{'ownHp':>6} {'oppHp':>6} {'dmg':>5} {'o[23]':>6} {'o[24]':>6}")
    print(header)
    print("-" * len(header))
    for row in rows:
        if not args.all_frames and not (row["air"] or row["dmg"] or row["atk"]):
            continue
        dist = "-" if row["dist"] is None else f"{row['dist']:.2f}"
        print(f"{row['step']:>4} {row['t']:>6.2f} {row['y']:>7.3f} {row['vy']:>7.3f} "
              f"{'Y' if row['air'] else 'n':>4} {'1' if row['atk'] else '0':>4} {dist:>5} "
              f"{row['own_hp']:>6.1f} {row['opp_hp']:>6.1f} {row['dmg']:>5.2f} "
              f"{row['obs_vy']:>6.3f} {row['obs_cd']:>6.2f}")

    print("\n=== 小结 ===")
    print(f"策略={args.attack_policy} 帧数={len(rows)}  同帧 jump+attack 下发={combo_frames}  "
          f"离地帧={airborne_frames}  下落帧(vy<0)={falling_frames}")
    if hits:
        crits = [value for value in hits if value >= CRIT_DAMAGE]
        normals = [value for value in hits if abs(value - NORMAL_DAMAGE) <= 0.15]
        print(f"命中共 {len(hits)} 刀: {hits}")
        print(f"  普通(≈{NORMAL_DAMAGE})={len(normals)} 刀  暴击(≥{CRIT_DAMAGE})={len(crits)} 刀  "
              f"最大单刀={max(hits):.2f}")
        if crits:
            print(f"判据通过：出现 {len(crits)} 次 ≥{CRIT_DAMAGE} 的单刀，"
                  "证明同一帧的 jump+attack 被服务端按跳劈结算。")
            return 0
        print(f"判据未通过：所有单刀都 <{CRIT_DAMAGE}（没有吃到跳劈暴击）。"
              "可能原因：命中时机都在上升段/落地后，见上表 air/vy 列。")
        return 1
    print("判据未通过：整段没有造成任何掉血（先看 dist 是否进入 reach、Bot 是否握剑）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
