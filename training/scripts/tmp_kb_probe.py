"""临时诊断（G-707，不入库）：`release_w` 到底有没有让对手飞得更远。

为什么需要它：单测只能证明 `applyIntent` 在同一帧里「先松 W → attack → 再按 W」，
**不能**证明 1.8 的服务端把这一刀当成了「非疾跑命中 → 下一刀重新变成疾跑命中」。
判据只有一条，而且必须是真机数据：

    带 release_w 的 attack   的对手水平位移
    -------------------------------------  ≥ 1.5
    不带 reset 的 attack     的对手水平位移

固定场景（可复现）：对手 ``opponent_mode="idle"``（站着不动，位移全部来自击退）、
双方先贴到约 3 格、每帧都下发 attack（0.5s 无敌帧由服务端节流）。

如果比值 < 1.5，脚本还会额外打印「自己命中后的水平速度」（疾跑 5.6 格/s vs 走 4.3 格/s），
用来区分两种失败原因：

- **服务端不接受**：sprint 根本没有被重新触发 → 两段速度几乎一样（比值≈1.0），
  说明 mineflayer 的 ``setControlState('sprint')`` 在服务端取消疾跑后不会重发
  START_SPRINTING，光松/按 W 不够；
- **时序不对**：速度确实上去了（比值 > 1.15）但击退没变大 → 命中帧本身不是疾跑
  （松手发生在 attack 之前、但服务端把这一刀算在上一个 tick 的疾跑状态上），
  需要把 reset 提前/推后一帧。

用法（Bot 必须已在线）::

    cd training
    .venv\\Scripts\\python.exe scripts\\tmp_kb_probe.py --no-spawn --steps 60

先确认在线::

    node bot-runtime/tmp-rcon.mjs "gladiator list"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import (  # noqa: E402
    ACTION_HEADS_V2, MOVE_VALUES_V2, SPRINT_RESET_VALUES_V2,
)

#: 头下标（按 ACTION_HEADS_V2 的声明顺序查，避免写死数字——G-707 加头时不会错位）。
_HEADS = list(ACTION_HEADS_V2)
_IDX = {name: index for index, name in enumerate(_HEADS)}

#: 判据：带 reset 的位移至少是基线的多少倍。
PASS_RATIO = 1.5
#: 判定「掉血 = 一次命中」的最小血量差（点）。
HIT_EPS = 0.4
#: 一次击退要在多少帧内量完（0.2s/步，3 帧 ≈ 0.6s，足够速度衰减完）。
DISPLACEMENT_WINDOW = 3


def kb_action(slot: int, reset: str = "none", *, move: str = "forward",
              attack: bool = True, sprint: bool = True) -> list[int]:
    """构造一个 10 头动作：按住 W（+疾跑）+ 出手，可选 `sprint_reset`。"""
    action = [0] * len(_HEADS)
    action[_IDX["move"]] = MOVE_VALUES_V2.index(move)
    action[_IDX["look_yaw"]] = 3          # LOOK_YAW_V2 里的 0.0
    action[_IDX["look_pitch"]] = 2        # LOOK_PITCH_V2 里的 0.0
    action[_IDX["attack"]] = 1 if attack else 0
    action[_IDX["use"]] = 0
    action[_IDX["jump"]] = 0
    action[_IDX["sneak"]] = 0
    action[_IDX["sprint"]] = 1 if sprint else 0
    action[_IDX["slot"]] = max(0, min(8, int(slot)))
    action[_IDX["sprint_reset"]] = SPRINT_RESET_VALUES_V2.index(reset)
    return action


def _pos(payload: dict) -> tuple[float, float, float] | None:
    pos = (payload.get("self") or {}).get("pos") or {}
    try:
        return float(pos["x"]), float(pos["y"]), float(pos["z"])
    except (KeyError, TypeError, ValueError):
        return None


def _health(payload: dict) -> float | None:
    try:
        return float((payload.get("self") or {}).get("health"))
    except (TypeError, ValueError):
        return None


def _distance(a, b) -> float | None:
    if a is None or b is None:
        return None
    dx = a[0] - b[0]
    dz = a[2] - b[2]
    return (dx * dx + dz * dz) ** 0.5


def _held_hotbar_slot(payload: dict) -> int:
    held = ((payload.get("self") or {}).get("heldItem") or {})
    try:
        slot = int(held.get("slot"))
    except (TypeError, ValueError):
        return 0
    if 36 <= slot <= 44:
        return slot - 36
    return slot if 0 <= slot <= 8 else 0


class Phase:
    """一段采样：逐帧下发同一个 reset 策略，记录每次命中后的对手位移。

    血量变化要比坐标晚一帧才进观测，所以「命中帧」用**上一帧**的对手坐标当位移原点
    （那是被击退之前的最后位置），窗口取之后的 ``DISPLACEMENT_WINDOW`` 帧。
    """

    def __init__(self, name: str, reset: str) -> None:
        self.name = name
        self.reset = reset
        self.hits: list[dict] = []
        self.frames = 0
        self._pending: dict | None = None

    def _finalize(self) -> None:
        if self._pending is None:
            return
        speeds = self._pending.pop("speeds")
        self._pending["speedAfter"] = round(sum(speeds) / len(speeds), 2) if speeds else None
        self.hits.append(self._pending)
        self._pending = None

    def observe(self, frame: int, *, opp_health: float | None, prev_opp_health: float | None,
                opp_pos, prev_opp_pos, self_pos, prev_self_pos,
                frame_seconds: float) -> None:
        self.frames += 1
        # 1) 推进已有的位移窗口（含自身的速度采样，用来判断疾跑有没有被重新触发）
        if self._pending is not None:
            moved = _distance(opp_pos, self._pending["origin"])
            if moved is not None and moved > self._pending["displacement"]:
                self._pending["displacement"] = moved
            step = _distance(self_pos, prev_self_pos)
            if step is not None and frame_seconds > 0:
                self._pending["speeds"].append(step / frame_seconds)
            if frame - self._pending["frame"] >= DISPLACEMENT_WINDOW:
                self._finalize()
        # 2) 新命中：血量比上一帧低（≥ HIT_EPS），原点取上一帧的对手位置
        if (prev_opp_health is not None and opp_health is not None and opp_health > 0.0
                and prev_opp_health - opp_health >= HIT_EPS):
            self._finalize()                 # 上一窗口没走完就再来一刀：先收尾
            self._pending = {
                "frame": frame,
                "damage": round(prev_opp_health - opp_health, 2),
                "distance": (_distance(self_pos, prev_opp_pos)
                             if prev_opp_pos is not None else None),
                "origin": prev_opp_pos if prev_opp_pos is not None else opp_pos,
                "displacement": 0.0,
                "speeds": [],
            }

    def close(self) -> None:
        self._finalize()

    def summary(self) -> dict:
        displacements = [hit["displacement"] for hit in self.hits]
        speeds = [hit["speedAfter"] for hit in self.hits if hit.get("speedAfter")]
        distances = [hit["distance"] for hit in self.hits if hit.get("distance") is not None]
        return {
            "phase": self.name,
            "reset": self.reset,
            "frames": self.frames,
            "hits": len(self.hits),
            "meanDisplacement": (round(sum(displacements) / len(displacements), 3)
                                 if displacements else 0.0),
            "displacements": [round(value, 3) for value in displacements],
            "meanHitDistance": (round(sum(distances) / len(distances), 2)
                                if distances else None),
            "meanSpeedAfter": round(sum(speeds) / len(speeds), 2) if speeds else None,
        }


def run_phase(env, name: str, reset: str, steps: int, frame_seconds: float,
              warmup: int = 0, sprint_mode: str = "on",
              attack_every: int = 1) -> Phase:
    """跑一段。

    ``attack_every=3`` 时只在 ``index % 3 == 2`` 的帧挥刀：两次命中至少隔 0.6s，
    位移窗口（3 帧）正好装得下一次击退，不会被「第二刀」污染。
    """
    phase = Phase(name, reset)
    raw = env.raw_observations
    prev_opp_health = _health(raw[1]) if len(raw) > 1 else None
    prev_opp_pos = _pos(raw[1]) if len(raw) > 1 else None
    prev_self_pos = _pos(raw[0]) if raw else None
    slot = _held_hotbar_slot(raw[0]) if raw else 0

    # 贴脸预热：只前进不出手，把距离压到 3 格附近（"固定场景"的第 2 条）。
    for _ in range(max(0, warmup)):
        env.step(kb_action(slot, "none", attack=False))
        raw = env.raw_observations
        if len(raw) > 1 and (_distance(_pos(raw[0]), _pos(raw[1])) or 99.0) <= 3.2:
            break

    for index in range(steps):
        # sprint_mode="rearm"：3 帧一个循环 —— 第 0 帧关 sprint 并出手（非疾跑命中）、
        # 第 1 帧把 sprint 开回来（控制位 0→1，mineflayer 这才会重发 START_SPRINTING）、
        # 第 2 帧出手（此时若 sprint 位真的恢复了，这一刀就是**疾跑命中 = 大 KB**）。
        # 只有第 2 帧挥刀，所以每次采样都是一刀独立的击退。
        strike = (index % max(1, attack_every)) == (max(1, attack_every) - 1) or attack_every == 1
        if sprint_mode == "rearm":
            sprint = (index % 3) != 0
            strike = (index % 3) == 2
        else:
            sprint = sprint_mode == "on"
        _observation, _reward, terminated, truncated, _info = env.step(
            kb_action(slot, reset, sprint=sprint, attack=strike))
        raw = env.raw_observations
        self_pos = _pos(raw[0]) if raw else None
        opp_pos = _pos(raw[1]) if len(raw) > 1 else None
        opp_health = _health(raw[1]) if len(raw) > 1 else None
        phase.observe(index, opp_health=opp_health, prev_opp_health=prev_opp_health,
                      opp_pos=opp_pos, prev_opp_pos=prev_opp_pos,
                      self_pos=self_pos, prev_self_pos=prev_self_pos,
                      frame_seconds=frame_seconds)
        prev_opp_health = opp_health if opp_health else None
        prev_opp_pos = opp_pos
        prev_self_pos = self_pos
        if terminated or truncated:
            print(f"  [{name}] 回合结束于第 {index + 1} 帧（terminated={terminated} "
                  f"truncated={truncated}）", flush=True)
            break
    phase.close()
    return phase


def main() -> int:
    parser = argparse.ArgumentParser(description="wtap（release_w）是否真的加大击退")
    parser.add_argument("--steps", type=int, default=60, help="每段下发多少帧")
    parser.add_argument("--warmup", type=int, default=20, help="每段开始前最多用多少帧贴到 3 格")
    parser.add_argument("--users", nargs=2, default=["Bot_V0A", "Bot_V0B"])
    parser.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mc-port", type=int, default=25566)
    parser.add_argument("--rcon-port", type=int, default=25575)
    parser.add_argument("--rcon-password", default="devpass")
    parser.add_argument("--center", default="0.5,4,0.5")
    parser.add_argument("--spacing", type=float, default=4.0,
                        help="开局间距（格）；MIN_SPACING=4.0，贴到 3 格靠 warmup")
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--out", default="", help="可选：把结果写成 JSON")
    args = parser.parse_args()

    center = tuple(float(part) for part in args.center.split(","))
    if len(center) != 3:
        print(f"--center 需要 x,y,z，收到 {args.center}", file=sys.stderr)
        return 2

    env = GladiatorEnv(
        [
            BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
            BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
        ],
        ArenaConfig(center=center, spacing=args.spacing, jitter=(0.0, 0.0)),
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        single_agent=True,
        action_space="v2",
        opponent_mode="idle",
        max_steps=args.steps * 2 + 40,
        auto_spawn=not args.no_spawn,
        connect=True,
        wait_action_results=False,
    )
    frame_seconds = float(env.action_ticks) / float(env._control_hz or 10.0)
    print(f"[kb] 动作空间=v2 对手=idle 间距={args.spacing} 帧长≈{frame_seconds:.3f}s "
          f"每段 {args.steps} 帧 判据 ratio≥{PASS_RATIO}", flush=True)

    env.reset(seed=args.seed)
    raw = env.raw_observations
    print(f"[kb] 起手距离={_distance(_pos(raw[0]), _pos(raw[1])):.2f} 格 "
          f"手持槽={_held_hotbar_slot(raw[0])}", flush=True)

    # 三相同一个出手节奏（每 3 帧一刀，间隔 0.6s > 0.5s 无敌帧），差异只在「有没有 reset」
    # 与「有没有把 sprint 控制位重新点亮」——这样每次命中都是一次独立的击退采样。
    baseline = run_phase(env, "baseline(no reset)", "none", args.steps, frame_seconds,
                         warmup=args.warmup, attack_every=3)
    print(f"  baseline: 命中 {baseline.summary()['hits']} 次，"
          f"平均位移 {baseline.summary()['meanDisplacement']:.3f} 格", flush=True)

    env.reset(seed=args.seed)
    wtap = run_phase(env, "release_w", "release_w", args.steps, frame_seconds,
                     warmup=args.warmup, attack_every=3)
    print(f"  release_w: 命中 {wtap.summary()['hits']} 次，"
          f"平均位移 {wtap.summary()['meanDisplacement']:.3f} 格", flush=True)

    env.reset(seed=args.seed)
    toggle = run_phase(env, "release_w + sprint rearm", "release_w", args.steps,
                       frame_seconds, warmup=args.warmup, sprint_mode="rearm")
    print(f"  release_w+sprint 重新点亮: 命中 {toggle.summary()['hits']} 次，"
          f"平均位移 {toggle.summary()['meanDisplacement']:.3f} 格", flush=True)

    env.close()

    base = baseline.summary()
    experiment = wtap.summary()
    toggled = toggle.summary()
    ratio = (experiment["meanDisplacement"] / base["meanDisplacement"]
             if base["meanDisplacement"] > 1e-9 else float("inf"))
    ratio_toggle = (toggled["meanDisplacement"] / base["meanDisplacement"]
                    if base["meanDisplacement"] > 1e-9 else float("inf"))
    print("\n=== 逐次命中位移（格） ===")
    print(f"  baseline            : {base['displacements']}")
    print(f"  release_w           : {experiment['displacements']}")
    print(f"  release_w + sprint  : {toggled['displacements']}")
    print(f"\n=== 小结 ===\n  基线均值 {base['meanDisplacement']:.3f} 格  "
          f"release_w 均值 {experiment['meanDisplacement']:.3f} 格  比值 {ratio:.2f}  "
          f"| rearm 均值 {toggled['meanDisplacement']:.3f} 格  比值 {ratio_toggle:.2f}")
    if ratio_toggle >= PASS_RATIO and ratio < PASS_RATIO:
        print(f"  关键结论：单靠 release_w 不够（{ratio:.2f}x），但**同一帧内把 sprint 控制位"
              f"关再开**就能让击退放大到 {ratio_toggle:.2f}x → 1.8 的疾跑位必须在客户端"
              "重发 START_SPRINTING，模型需要自己组合 sprint 头 + sprint_reset 头。")

    verdict = "pass" if ratio >= PASS_RATIO else "fail"
    if verdict == "pass":
        print(f"  判据通过：比值 {ratio:.2f} ≥ {PASS_RATIO}，wtap 在服务端确实放大了击退。")
    elif ratio_toggle >= PASS_RATIO:
        # 已经能定性了：单靠 release_w 不够，必须把 sprint 控制位重新点亮。
        # 这里不再打印「服务端不接受」那套猜测性提示，避免两段结论互相打架。
        print(f"  判据未通过（release_w 单独用只有 {ratio:.2f}x < {PASS_RATIO}），"
              f"但 **release_w + sprint 重新点亮** 达到 {ratio_toggle:.2f}x。")
        print("  定性结论：1.8 的疾跑位在命中后由服务端清掉，只有客户端把 sprint 控制位"
              "0→1 重发 START_SPRINTING 才会恢复；因此 wtap 必须由**两个头共同表达**"
              "（sprint_reset + sprint），不能只靠 release_w。")
    else:
        speed_a = base.get("meanSpeedAfter")
        speed_b = experiment.get("meanSpeedAfter")
        hint = "样本不足，先确认命中了人"
        if speed_a and speed_b:
            speed_ratio = speed_b / speed_a
            print(f"  命中后自身水平速度：baseline {speed_a:.2f} 格/s → "
                  f"release_w {speed_b:.2f} 格/s（比值 {speed_ratio:.2f}）")
            if speed_ratio >= 1.15:
                hint = ("疾跑确实被重新触发了，但击退没变大 → **时序问题**："
                        "命中帧仍被服务端算作疾跑前的状态，需要把 reset 提前/推后一帧")
            else:
                hint = ("疾跑没有被重新触发（速度几乎不变）→ **服务端不接受**："
                        "光松/按 W 不够，需要在 release_w 里同时把 sprint 控制位关再开"
                        "（重发 START_SPRINTING）")
        print(f"  判据未通过：比值 {ratio:.2f} < {PASS_RATIO}。{hint}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frameSeconds": frame_seconds,
            "baseline": base,
            "release_w": experiment,
            "release_w_sprint_toggle": toggled,
            "ratio": round(ratio, 3),
            "ratioSprintToggle": round(ratio_toggle, 3),
            "pass": verdict == "pass",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {args.out}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
