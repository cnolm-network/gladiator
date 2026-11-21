"""把人机对战录制（`dashboard` 编排器产出的 frames.jsonl）转成训练数据，可选做一轮 BC。

用法::

    python scripts/import_human_match.py <录制目录> [--out dataset.npz]
                                        [--bc] [--base-model <policy.npz>]
                                        [--epochs 10] [--run-dir <目录>] [--seed 0]

录制目录由 `dashboard/match_runner.py` 生成，结构::

    <时间>-<玩家>-vs-<Bot>/
      meta.json      # 对局配置（plan / brain / 录制频率）
      frames.jsonl   # 10Hz 采样：{"t":秒, "round":局号, "player":{...}, "bot":{...}}
      summary.json   # 比分与每局结果

**动作标签推断规则**（人类玩家没有动作日志，只能从状态变化反推；顺序即优先级）：

1. `attack`：下一帧对手掉血 ≥ 0.5 且双方水平距离 ≤ 4.5 格——这一刀是命中的；
2. `eat_gapple`：自己血量单帧回升 ≥ 1.5 点（金苹果回 4 点，且不受其它治疗影响）；
3. `block`：该帧 `blocking=true`（举剑）；
4. `jump`：竖直位移 > 0.25 格，或从着地变为离地；
5. `approach` / `retreat`：把位移投影到**朝向轴**上，前向速度 > 0.2 格/s / < -0.2；
6. `strafe_left` / `strafe_right`：位移投影到**右向轴**上，> 0.2 / < -0.2；
7. `idle`：以上都不满足。

`switch_sword` 与 `use_rod` 无法从只读遥测里区分（录制不含手持物与右键事件），
因此这两类永远不产生标签——训练数据只覆盖可观测的 9 种动作。这一点是刻意的：
宁可少标，也不要把「没观察到」硬猜成某个动作。

观测向量直接复用 `gladiator_env.spaces.encode_observation`（23 维），保证与在线
运行时/训练环境逐位对齐：把每帧双方 stats 还原成一条 OBSERVATION 字典再编码。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, encode_observation  # noqa: E402

TRAINING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = TRAINING_ROOT / "artifacts"

#: 标签推断阈值（改这里就等于改数据分布，文档里同步写清楚）。
ATTACK_DAMAGE_MIN = 0.5
ATTACK_RANGE = 4.5
HEAL_MIN = 1.5
FORWARD_SPEED_MIN = 0.2
JUMP_DY_MIN = 0.25

#: 择优权重（a 结果过滤 / b 接触窗口）。权重只影响 BC 的交叉熵，不改数据集本身。
WIN_WEIGHT = 1.0
LOSS_WEIGHT = 0.3
DRAW_WEIGHT = 0.5
CONTACT_RANGE = 5.0
CONTACT_WEIGHT = 1.0
FAR_WEIGHT = 0.2

ATTACK = ACTION_NAMES.index("attack")
APPROACH = ACTION_NAMES.index("approach")
EAT_GAPPLE = ACTION_NAMES.index("eat_gapple")
BLOCK = ACTION_NAMES.index("block")
JUMP = ACTION_NAMES.index("jump")
RETREAT = ACTION_NAMES.index("retreat")
STRAFE_LEFT = ACTION_NAMES.index("strafe_left")
STRAFE_RIGHT = ACTION_NAMES.index("strafe_right")
IDLE = ACTION_NAMES.index("idle")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人机对战录制 → 训练数据集（可选 BC）")
    parser.add_argument("record_dir", help="录制目录（含 frames.jsonl）")
    parser.add_argument("--out", default=None, help="数据集输出路径（默认 <record_dir>/dataset.npz）")
    parser.add_argument("--bc", action="store_true", help="用数据集做一轮行为克隆")
    parser.add_argument("--base-model", default=None, help="BC 的起点策略（.npz）；缺省则随机初始化")
    parser.add_argument("--epochs", type=int, default=10, help="BC 轮数")
    parser.add_argument("--min-samples", type=int, default=20, help="少于该样本数时跳过 BC")
    parser.add_argument("--run-dir", default=None, help="BC 产物目录（默认 artifacts/runs/<时间>-human-bc）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--win-weight", type=float, default=WIN_WEIGHT,
                        help=f"玩家赢的局的样本权重（默认 {WIN_WEIGHT}）")
    parser.add_argument("--loss-weight", type=float, default=LOSS_WEIGHT,
                        help=f"玩家输的局的样本权重（默认 {LOSS_WEIGHT}）")
    parser.add_argument("--draw-weight", type=float, default=DRAW_WEIGHT,
                        help=f"平局/未知结果的样本权重（默认 {DRAW_WEIGHT}）")
    parser.add_argument("--contact-range", type=float, default=CONTACT_RANGE,
                        help=f"交手距离阈值（默认 {CONTACT_RANGE} 格）")
    parser.add_argument("--far-weight", type=float, default=FAR_WEIGHT,
                        help=f"超出交手距离的样本权重（默认 {FAR_WEIGHT}）")
    parser.add_argument("--no-prefer-wins", action="store_true",
                        help="关闭择优权重（所有样本等权，等价旧行为）")
    return parser.parse_args(argv)


def load_round_winners(record_dir: Path) -> dict[int, str]:
    """读 ``summary.json`` 的每局胜者；缺文件时返回空字典（全部按未知处理）。"""
    path = record_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    winners: dict[int, str] = {}
    for item in summary.get("rounds") or []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("round"))
        except (TypeError, ValueError):
            continue
        winners[number] = str(item.get("winner") or "unknown")
    return winners


def round_weight(winner: str, *, win_weight: float, loss_weight: float,
                 draw_weight: float) -> float:
    """(a) 结果过滤：赢的局满权重、输的局降权、平局居中。"""
    if winner == "player":
        return float(win_weight)
    if winner == "bot":
        return float(loss_weight)
    return float(draw_weight)


def contact_window_weight(frame: dict, *, player_name: str, bot_name: str,
                   contact_range: float, contact_weight: float,
                   far_weight: float) -> float:
    """(b) 接触窗口：双方水平距离在 ``contact_range`` 内才按满权重学。"""
    player = frame.get("player") or {}
    bot = frame.get("bot") or {}
    if not player or not bot:
        return float(far_weight)
    dx = _num(player.get("x")) - _num(bot.get("x"))
    dz = _num(player.get("z")) - _num(bot.get("z"))
    distance = math.hypot(dx, dz)
    return float(contact_weight if distance <= contact_range else far_weight)


# --- 读取与标签 -------------------------------------------------------------

def load_frames(record_dir: Path) -> list[dict]:
    path = record_dir / "frames.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"找不到录制文件: {path}")
    frames: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict) and frame.get("player"):
                frames.append(frame)
    return frames


def load_meta(record_dir: Path) -> dict:
    path = record_dir / "meta.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def infer_action(prev: dict, cur: dict, dt: float) -> int:
    """按模块 docstring 的规则，从两帧之间的人类状态变化推断宏动作。"""
    dt = max(dt, 0.05)
    before = prev.get("player") or {}
    now = cur.get("player") or {}
    bot_before = prev.get("bot") or {}
    bot_now = cur.get("bot") or {}

    dx = _num(now.get("x")) - _num(before.get("x"))
    dz = _num(now.get("z")) - _num(before.get("z"))
    dy = _num(now.get("y")) - _num(before.get("y"))
    yaw = math.radians(_num(before.get("yaw")))
    forward = -math.sin(yaw) * dx + math.cos(yaw) * dz
    right = -math.cos(yaw) * dx - math.sin(yaw) * dz
    forward_speed = forward / dt
    right_speed = right / dt

    damage = _num(bot_before.get("health"), 20.0) - _num(bot_now.get("health"), 20.0)
    distance = math.hypot(_num(now.get("x")) - _num(bot_now.get("x")),
                          _num(now.get("z")) - _num(bot_now.get("z")))
    heal = _num(now.get("health")) - _num(before.get("health"))

    if damage >= ATTACK_DAMAGE_MIN and distance <= ATTACK_RANGE:
        return ATTACK
    if heal >= HEAL_MIN:
        return EAT_GAPPLE
    if bool(now.get("blocking")):
        return BLOCK
    if dy > JUMP_DY_MIN or (bool(before.get("onGround")) and not bool(now.get("onGround"))):
        return JUMP
    if forward_speed > FORWARD_SPEED_MIN:
        return APPROACH
    if forward_speed < -FORWARD_SPEED_MIN:
        return RETREAT
    if right_speed > FORWARD_SPEED_MIN:
        return STRAFE_RIGHT
    if right_speed < -FORWARD_SPEED_MIN:
        return STRAFE_LEFT
    return IDLE


def _observation(frame: dict, prev: Optional[dict], player_name: str, bot_name: str,
                 step_fraction: float, since_attack: float, reach: float) -> np.ndarray:
    """把一帧 stats 还原成 OBSERVATION 字典，再交给官方编码器（保证 23 维一致）。"""
    player = frame.get("player") or {}
    bot = frame.get("bot") or {}
    previous_bot = (prev or {}).get("bot") or bot
    dt = max(_num(frame.get("t")) - _num((prev or {}).get("t")), 0.05)
    vx = (_num(bot.get("x")) - _num(previous_bot.get("x"))) / dt
    vz = (_num(bot.get("z")) - _num(previous_bot.get("z"))) / dt
    vy = (_num(bot.get("y")) - _num(previous_bot.get("y"))) / dt

    self_payload = {
        "pos": {"x": _num(player.get("x")), "y": _num(player.get("y")), "z": _num(player.get("z"))},
        "yaw": math.radians(_num(player.get("yaw"))),
        "pitch": math.radians(_num(player.get("pitch"))),
        "onGround": bool(player.get("onGround")),
        "health": _num(player.get("health"), 20.0),
        "food": _num(player.get("food"), 20.0),
        "saturation": _num(player.get("saturation")),
        # 录制不含背包细节：手持物按满耐久、有甲就按满耐久，避免把 0 当成"空手"。
        "heldItem": {"durability": {"remaining": 1.0}},
        "armorRemaining": 1.0 if _num(player.get("armor")) > 0 else 0.0,
        "inventory": [],
    }
    entities = [{
        "id": 1,
        "kind": "player",
        "name": bot_name,
        "pos": {"x": _num(bot.get("x")), "y": _num(bot.get("y")), "z": _num(bot.get("z"))},
        "velocity": {"x": vx, "y": vy, "z": vz},
        "distance": math.sqrt(
            (_num(bot.get("x")) - _num(player.get("x"))) ** 2
            + (_num(bot.get("y")) - _num(player.get("y"))) ** 2
            + (_num(bot.get("z")) - _num(player.get("z"))) ** 2
        ),
        "onGround": bool(bot.get("onGround")),
        "health": _num(bot.get("health"), 20.0),
    }]
    observation = {
        "state": "Alive" if _num(player.get("health"), 20.0) > 0 else "Dead",
        "self": self_payload,
        "entities": entities,
    }
    return encode_observation(
        observation, player_name,
        step_fraction=step_fraction,
        opponent_health=_num(bot.get("health"), 20.0),
        opponents={bot_name},
        reach=reach,
        since_attack=since_attack,
    )


def build_dataset(frames: list[dict], *, player_name: str, bot_name: str,
                  round_seconds: float = 120.0, reach: float = 3.99,
                  winners: Optional[dict[int, str]] = None,
                  win_weight: float = WIN_WEIGHT, loss_weight: float = LOSS_WEIGHT,
                  draw_weight: float = DRAW_WEIGHT, contact_range: float = CONTACT_RANGE,
                  contact_weight: float = CONTACT_WEIGHT,
                  far_weight: float = FAR_WEIGHT) -> dict:
    """返回 ``{"obs","actions","times","rounds","weights"}``（obs 为 23 维）。

    ``weights`` 是 (a)(b) 两步择优的结果：``局结果权重 × 接触窗口权重``。
    全部参数取默认值时，赢局接触帧为 1.0、输局远处帧为 0.06，权重并未丢弃任何样本。
    """
    observations: list[np.ndarray] = []
    actions: list[int] = []
    times: list[float] = []
    rounds: list[int] = []
    weights: list[float] = []
    winners = winners or {}
    since_attack = 5.0
    previous: Optional[dict] = None
    for frame in frames:
        player = frame.get("player") or {}
        t = _num(frame.get("t"))
        if previous is not None:
            dt = max(t - _num(previous.get("t")), 0.05)
            action = infer_action(previous, frame, dt)
            since_attack = 0.0 if action == ATTACK else min(5.0, since_attack + dt)
        else:
            action = IDLE
            since_attack = min(5.0, since_attack + 0.1)
        step_fraction = 0.0 if round_seconds <= 0 else min(1.0, max(0.0, t / round_seconds))
        observations.append(_observation(frame, previous, player_name, bot_name,
                                         step_fraction, since_attack, reach))
        actions.append(int(action))
        times.append(t)
        round_number = int(_num(frame.get("round"), 1))
        rounds.append(round_number)
        weights.append(
            round_weight(winners.get(round_number, "unknown"),
                         win_weight=win_weight, loss_weight=loss_weight,
                         draw_weight=draw_weight)
            * contact_window_weight(frame, player_name=player_name, bot_name=bot_name,
                             contact_range=contact_range,
                             contact_weight=contact_weight, far_weight=far_weight))
        if _num(player.get("health"), 20.0) <= 0:
            since_attack = 5.0
        previous = frame

    obs = np.asarray(observations, dtype=np.float32) if observations else np.zeros((0, OBS_DIM), np.float32)
    return {
        "obs": obs,
        "actions": np.asarray(actions, dtype=np.int64),
        "times": np.asarray(times, dtype=np.float32),
        "rounds": np.asarray(rounds, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float32),
    }


# --- BC ---------------------------------------------------------------------

def run_bc(dataset: dict, *, base_model: Optional[str], epochs: int,
           run_dir: Optional[Path], meta: dict, seed: int = 0,
           min_samples: int = 20) -> Optional[dict]:
    """在现有模型上做一轮行为克隆，产物写到新 run 目录（**不覆盖**模型权重）。"""
    from gladiator_train.demos import behavior_clone
    from gladiator_train.policy import Policy

    samples = [(dataset["obs"][i], int(dataset["actions"][i]), 0.0)
               for i in range(len(dataset["actions"]))]
    weight_list = dataset.get("weights")
    if weight_list is not None and len(weight_list) != len(samples):
        return {"skipped": True,
                "reason": f"权重长度 {len(weight_list)} 与样本数 {len(samples)} 不一致"}
    if len(samples) < int(min_samples):
        return {"skipped": True, "reason": f"样本太少（{len(samples)} < {min_samples}）"}

    if base_model:
        policy = Policy.load(base_model)
        if policy.obs_dim != OBS_DIM:
            return {"skipped": True,
                    "reason": f"模型 obsDim={policy.obs_dim} 与数据集 {OBS_DIM} 不一致"}
    else:
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=64, seed=seed)

    result = behavior_clone(policy, samples, epochs=max(1, int(epochs)), seed=seed,
                            weights=None if weight_list is None else list(weight_list))
    target = run_dir or (DEFAULT_ARTIFACTS / "runs" /
                         f"{time.strftime('%Y%m%d-%H%M')}-human-bc-{meta.get('player', 'player')}")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    policy.save(str(target / "policy.npz"), extra={
        "source": "human-match",
        "recordDir": str(meta.get("recordDir") or ""),
        "samples": len(samples),
    })
    report = {
        "kind": "human-bc",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseModel": base_model,
        "samples": len(samples),
        "epochs": result["epochs"],
        "loss": result["loss"][-1] if result["loss"] else None,
        "accuracy": result["accuracy"],
        "policy": str(target / "policy.npz"),
        "recordDir": meta.get("recordDir"),
    }
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    record_dir = Path(args.record_dir).resolve()
    if not record_dir.is_dir():
        print(f"录制目录不存在: {record_dir}", file=sys.stderr)
        return 2
    try:
        frames = load_frames(record_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not frames:
        print(f"录制里没有有效帧: {record_dir / 'frames.jsonl'}", file=sys.stderr)
        return 2

    meta = load_meta(record_dir)
    plan = meta.get("plan") or {}
    player_name = str(plan.get("player") or "player")
    bot_name = str(plan.get("bot") or "bot")
    round_seconds = float(plan.get("roundSeconds") or 120)

    dataset = build_dataset(
        frames, player_name=player_name, bot_name=bot_name, round_seconds=round_seconds,
        winners=load_round_winners(record_dir),
        win_weight=args.win_weight, loss_weight=args.loss_weight,
        draw_weight=args.draw_weight, contact_range=args.contact_range,
        contact_weight=1.0, far_weight=args.far_weight,
    )
    if args.no_prefer_wins:
        dataset["weights"] = np.ones_like(dataset["weights"])
    out_path = Path(args.out) if args.out else record_dir / "dataset.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {ACTION_NAMES[index]: int((dataset["actions"] == index).sum())
              for index in range(len(ACTION_NAMES))}
    np.savez(
        out_path,
        obs=dataset["obs"],
        actions=dataset["actions"],
        times=dataset["times"],
        rounds=dataset["rounds"],
        weights=dataset["weights"],
        meta=np.array(json.dumps({
            "schema": "gladiator.human-match-dataset/v1",
            "recordDir": str(record_dir),
            "player": player_name,
            "bot": bot_name,
            "frames": len(frames),
            "samples": int(len(dataset["actions"])),
            "obsDim": OBS_DIM,
            "actionNames": list(ACTION_NAMES),
            "labelRules": "见 import_human_match.py 模块 docstring",
            "preferWins": not args.no_prefer_wins,
            "weightRules": {
                "win": args.win_weight, "loss": args.loss_weight,
                "draw": args.draw_weight, "contactRange": args.contact_range,
                "contact": 1.0, "far": args.far_weight,
            },
        }, ensure_ascii=False)),
    )

    bc_report = None
    if args.bc:
        bc_report = run_bc(
            dataset,
            base_model=args.base_model,
            epochs=args.epochs,
            run_dir=Path(args.run_dir) if args.run_dir else None,
            meta={**meta, "recordDir": str(record_dir), "player": player_name},
            seed=args.seed,
            min_samples=args.min_samples,
        )

    print(json.dumps({
        "ok": True,
        "recordDir": str(record_dir),
        "dataset": str(out_path),
        "frames": len(frames),
        "samples": int(len(dataset["actions"])),
        "actions": counts,
        "bc": bc_report,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
