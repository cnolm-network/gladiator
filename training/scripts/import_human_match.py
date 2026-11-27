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

观测向量直接复用 `gladiator_env.spaces.encode_observation`（维度随 `OBS_DIM` 走，当前 39 维），
保证与在线运行时/训练环境逐位对齐：把每帧双方 stats 还原成一条 OBSERVATION 字典再编码。

**G-736：可选的权威伤害来源（`--records`）**。上面的 `rewards` 是「相邻两帧血量差」推的，
10Hz 采样会把同一 tick 的多段伤害合并、也分不清谁打的。`arena` 会把每局导出到
`server/plugins/Arena/records/<uuid>-<ts>.jsonl`，里面有逐条 `ENTITY_ATTACK` 伤害与 `KILL`，
所以给了 `--records <目录|文件>` 就改用事件求和：

- `rewards[i] = 我方打出的伤害 - 我方吃到的伤害`（局末击杀再加 `--win-bonus`）；
- 额外产出 `damageDealt` / `damageTaken`（逐帧，供诊断与分对方统计）；
- `meta.rewardSource` 标明本数据集用的是 `arena-record` 还是回退后的 `health-delta`；
- 目录模式下会按「时间窗 + 参与者」自动挑出属于这一场的录制文件（多局对局 = 多个文件）。

时间基准：`meta.boardSession`（形如 `flat-1v1-1795807974138`）末尾的毫秒时间戳就是
**t=0 的绝对时刻**（实测比录制首个事件早 3ms）；只有它缺失时才退到 `meta.startedAt`
（ISO、秒精度，最坏有 1 秒误差）。两者都没有就无法对齐，会告警并回退 `health-delta`。
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

from gladiator_env.spaces import (ACTION_NAMES, OBS_DIM, encode_observation,  # noqa: E402
                                 macro_to_intent)

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
    parser.add_argument("record_dir", nargs="?", default=None,
                        help="录制目录（含 frames.jsonl）")
    #: G-736：兼容 `--record <dir>` 写法（README/任务书里都这么用），与位置参数二选一。
    parser.add_argument("--record", dest="record_opt", default=None,
                        help="同位置参数（--record <录制目录>）")
    parser.add_argument("--out", default=None,
                        help="数据集输出路径；给目录（或无扩展名的路径）则写 <dir>/dataset.npz"
                             "（默认 <record_dir>/dataset.npz）")
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
    # --- G-705(c)：AWR（用价值头给样本加权） ---
    parser.add_argument("--weighting", choices=["round", "contact", "awr"], default="contact",
                        help="权重模式：round = 只用 (a) 结果过滤；"
                             "contact = (a)×(b) 接触窗口（默认）；"
                             "awr = 在 contact 之上再乘 (c) 价值加权（需要 --base-model）")
    parser.add_argument("--awr-temperature", type=float, default=1.0,
                        help="AWR 温度：w = exp((R − V)/T)，越小越尖（默认 1.0）")
    parser.add_argument("--awr-clip", type=float, default=None,
                        help="AWR 权重上界（默认不剪）；用于压住单帧主导")
    parser.add_argument("--awr-strict", action="store_true",
                        help="数据集没有 rewards 时直接报错（默认只在 stderr 提示，"
                             "此时 AWR 退化为均匀权重、等于没开）")
    parser.add_argument("--win-bonus", type=float, default=0.0,
                        help="局末胜局额外加进 rewards 的量（默认 0 = 只用血量差；"
                             "想让 AWR 更看重「赢」可给正数）")
    # --- G-720：目标动作空间（v1 宏动作 / v2 十头） ---
    # --- G-736：权威伤害来源（Arena Recorder） ---
    parser.add_argument("--records", default=None,
                        help="Arena 录制目录或单个 .jsonl；给了就用 ENTITY_ATTACK/KILL 事件"
                             "求和作为 rewards（解析不出会告警并回退 health-delta）")
    parser.add_argument("--target-action-space", choices=["auto", "v1", "v2"], default="auto",
                        help="数据集的动作空间：auto = 跟 --base-model 走（无模型则 v1）；"
                             "v2 会把宏动作有损翻译成十头（见 macro_to_intent）")
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


def compose_awr_weights(dataset: dict, policy, *, temperature: float,
                        clip: Optional[float], strict: bool = False) -> tuple[np.ndarray, dict]:
    """(c) AWR：把已有的 (a)×(b) 权重再乘上「价值优势」权重，并重新归一到均值 1。

    <p>为什么乘而不是替换：AWR 回答的是「这一帧比预期好吗」，它对**局结果 / 距离**
    一无所知；而 (a)(b) 恰好只管这两件事。两者是正交的筛选维度，相乘才不丢信息。

    <p>***回报从哪来***：只认 ``dataset["rewards"]``（录制侧以后带上来才有）；
    没有就按 0 处理——**绝不在这里发明奖励函数**（见 ``advantage_weights`` 的 docstring）。
    全 0 时 ``A = −V``，负优势会被 ``positive_only`` 置 0。因此返回的 ``info`` 里会标出：

    - ``awrReturns``：回报来源（``dataset.rewards`` / ``zeros（录制暂无 rewards）``）；
    - ``awrEffective``：本次加权**是否真的改变了样本权重**。为 ``False`` 时说明
      数据集里没有回报、且所有优势都被置 0 → 权重退化成均匀，``--weighting awr``
      等价于不带 (c)。只看 ``awrReturns`` 会误判，必须看这个字段；
    - ``awrDistinct``：归一化后不同权重值的个数（1 = 全等，等于没加权）；
    - ``awrZeroed``：被置 0 的样本数。

    <p>``strict=True`` 时，**没有回报就直接报错**——避免「以为开了 AWR、其实什么都没做」。

    @returns ``(weights, info)``：``weights`` 与样本等长、均值 1；``info`` 直接并进
    ``meta.weightRules``，便于复算。
    """
    from gladiator_train.demos import advantage_weights

    base = np.asarray(dataset["weights"], dtype=np.float64)
    returns = dataset.get("rewards")
    if returns is None and strict:
        raise ValueError(
            "数据集里没有 rewards，AWR 无从计算（A = -V，负优势会被全部置 0，"
            "等于没加权）。要么让录制侧带上单步回报，要么去掉 --awr-strict 接受"
            "--weighting contact 的效果。")
    samples = [(dataset["obs"][i], _action_label(dataset["actions"][i]), 0.0)
               for i in range(len(base))]
    awr = advantage_weights(
        policy, samples,
        returns=None if returns is None else np.asarray(returns, dtype=np.float64),
        temperature=temperature,
        positive_only=True,
        clip=None if clip is None else (0.0, float(clip)),
    )
    combined = base * awr
    mean = float(combined.mean()) if combined.size else 0.0
    if not np.isfinite(mean) or mean <= 1e-12:
        # 理论上进不来（advantage_weights 已退化保护），留着防「总权重 0 的批次」。
        combined = np.ones_like(base)
    else:
        combined = combined / mean
    distinct = int(np.unique(np.round(awr, 9)).size) if awr.size else 0
    info = {
        "mode": "awr",
        "awrTemperature": float(temperature),
        "awrClip": None if clip is None else float(clip),
        "awrReturns": "dataset.rewards" if returns is not None else "zeros（录制暂无 rewards）",
        # 只有「有回报」时 A 才是 advantage。没有回报时 A = −V，权重**不是**噪声而是
        # 系统性偏向 critic 认为差的状态（与 AWR 原意相反），所以这里必须为 False。
        "awrEffective": returns is not None,
        "awrDistinct": distinct,
        "awrZeroed": int(np.sum(awr == 0.0)),
        "awrMean": round(float(combined.mean()), 6),
    }
    return combined, info


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
    """把一帧 stats 还原成 OBSERVATION 字典，再交给官方编码器（维度随 OBS_DIM，逐位一致）。"""
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
                  round_seconds: float = 120.0, reach: float = 3.0,
                  winners: Optional[dict[int, str]] = None,
                  win_weight: float = WIN_WEIGHT, loss_weight: float = LOSS_WEIGHT,
                  draw_weight: float = DRAW_WEIGHT, contact_range: float = CONTACT_RANGE,
                  contact_weight: float = CONTACT_WEIGHT,
                  far_weight: float = FAR_WEIGHT, win_bonus: float = 0.0,
                  action_space: str = "v1",
                  arena: Optional[dict] = None) -> dict:
    """返回 ``{"obs","actions","times","rounds","weights","rewards"}``（obs 为 ``OBS_DIM`` 维）。

    ``actions`` 的形状取决于 ``action_space``：

    - ``"v1"``（默认）：``(N,)`` 的宏动作下标（``ACTION_NAMES``）；
    - ``"v2"``：``(N, 10)`` 的十头下标，由 :func:`gladiator_env.spaces.macro_to_intent`
      从宏动作**有损翻译**而来（见该函数的注释；只表达「这一帧想干什么」）。

    ``weights`` 是 (a)(b) 两步择优的结果：``局结果权重 × 接触窗口权重``。
    全部参数取默认值时，赢局接触帧为 1.0、输局远处帧为 0.06，权重并未丢弃任何样本。
    (c) AWR 不在这里做：它需要模型的价值头，由 :func:`compose_awr_weights` 在
    ``--weighting awr`` 时叠加（见 ``docs/match.md`` §8.1）。

    ``rewards`` 是**观测回报**（不外挂 shaping）。两条来源，二选一：

    - 默认：由录制帧的**血量差**推出，每步 ``(对手掉血 - 自己掉血)``，跨局时把
      ``win_bonus`` 加到新局首帧（仅当上局赢，默认 0）；
    - ``arena`` 非空（``--records`` 解析成功）：直接用 :func:`align_record_rewards`
      按**伤害事件**逐帧求和的值，此时 ``damageDealt`` / ``damageTaken`` 也一并返回。

    它给 AWR 一个 ``A = R - V(s)`` 的 R；没有它时 AWR 只能退化成 ``A = -V``。
    """
    from gladiator_env.spaces import MULTI_DISCRETE_V2, ACTION_HEADS_V2

    mode = str(action_space or "v1").strip().lower()
    if mode not in ("v1", "v2"):
        raise ValueError(f"action_space 只能是 v1 / v2，收到 {action_space!r}")

    observations: list[np.ndarray] = []
    actions: list[int] = []
    intents: list[tuple[int, ...]] = []
    times: list[float] = []
    rounds: list[int] = []
    weights: list[float] = []
    rewards: list[float] = []
    #: G-736：``--records`` 生效时按事件求和；否则留空（数据集里也不会出现这两个键）。
    arena_mode = arena is not None
    damage_dealt = np.asarray(arena["damageDealt"], dtype=np.float32) if arena_mode else None
    damage_taken = np.asarray(arena["damageTaken"], dtype=np.float32) if arena_mode else None
    if arena_mode and (len(damage_dealt) != len(frames) or len(damage_taken) != len(frames)):
        raise ValueError("arena.damageDealt/damageTaken 长度与帧数不一致")
    #: G-736：预对齐好的逐帧回报（含击杀 bonus）。
    arena_rewards = np.asarray(arena["rewards"], dtype=np.float64) if arena_mode else None
    prev_bot_health: Optional[float] = None
    prev_player_health: Optional[float] = None
    prev_round: Optional[int] = None
    winners = winners or {}
    since_attack = 5.0
    previous: Optional[dict] = None
    for frame_index, frame in enumerate(frames):
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
        if mode == "v2":
            intents.append(macro_to_intent(int(action)))
        times.append(t)
        round_number = int(_num(frame.get("round"), 1))
        rounds.append(round_number)
        bot = frame.get("bot") or {}
        bot_health = _num(bot.get("health"), 20.0)
        player_health = _num(player.get("health"), 20.0)
        if arena_mode:
            # G-736：权威来源——伤害事件已经按帧求和，击杀 bonus 也落在末帧，这里不再算血量差。
            reward = float(arena_rewards[frame_index])
        else:
            reward = 0.0
            if prev_bot_health is not None and prev_player_health is not None:
                reward += max(0.0, prev_bot_health - bot_health)  # 打出去的伤害
                reward -= max(0.0, prev_player_health - player_health)  # 吃到的伤害
            if prev_round is not None and round_number != prev_round:
                # 上一局的收尾帧：按局结果补 terminal（默认 0，等于不加）。
                if winners.get(prev_round) == "player":
                    reward += win_bonus
        rewards.append(reward)
        prev_bot_health, prev_player_health, prev_round = bot_health, player_health, round_number
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
    if mode == "v2":
        action_array = (np.asarray(intents, dtype=np.int64) if intents
                        else np.zeros((0, len(MULTI_DISCRETE_V2)), np.int64))
    else:
        action_array = np.asarray(actions, dtype=np.int64)
    dataset = {
        "obs": obs,
        "actions": action_array,
        "times": np.asarray(times, dtype=np.float32),
        "rounds": np.asarray(rounds, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "actionSpace": mode,
        "headSizes": list(ACTION_HEADS_V2.values()) if mode == "v2" else [],
        "macroActions": np.asarray(actions, dtype=np.int64),
    }
    if arena_mode:
        # 只在真正用事件来源时才加这两个键：不给 --records 时数据集结构与旧版逐位一致。
        dataset["damageDealt"] = damage_dealt
        dataset["damageTaken"] = damage_taken
    return dataset


# --- Arena Recorder（G-736）-------------------------------------------------
#
# 录制文件（每局一份，`arena` 的 `Recorder` 导出）是一行一条 JSONL：
#
#   {"seq":1,"at":1795807974138,"category":"event",
#    "from":"EventHandler:...FlatManagerSimplified",
#    "content":"victim=Glad_P9 attacker=Glad_M3 damage=1.5999999046325684 cause=ENTITY_ATTACK"}
#
# `at` 是 epoch 毫秒；`content` 是 `key=value` 的扁平文本（**不是** JSON，别用 json.loads）。
# 我们只关心两类：带 `damage=` 的伤害事件、带 `cause=KILL` 的击杀事件。

#: 录制目录的默认位置（只用于错误提示；实际路径由 `--records` 给）。
DEFAULT_RECORDS_DIR = TRAINING_ROOT.parent / "server" / "plugins" / "Arena" / "records"
#: 按时间窗挑录制文件时的前后容差（毫秒）。对局起止与文件首末事件本来就有几百毫秒差。
RECORD_WINDOW_SLACK_MS = 2000


def parse_record_content(content: str) -> dict[str, str]:
    """把 `victim=A attacker=B damage=1.6 cause=ENTITY_ATTACK` 拆成字典。

    只做最简单的 ``split(" ")`` + ``k=v``：录制内容里的值都是无空格标识符/数字。
    没有 ``=`` 的碎片（例如消息里的自然语言）直接丢弃，**不要**因此报错——录制格式
    以后加字段时，这里必须保持宽容。
    """
    fields: dict[str, str] = {}
    for token in str(content or "").split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if key:
            fields[key] = value
    return fields


def load_record_events(path: Path) -> list[dict]:
    """读一个 ``records/*.jsonl``，返回伤害 / 击杀事件（按文件顺序，即时间序）。

    每条事件形如 ``{"at": 毫秒, "kind": "damage"|"kill", "victim": str,
    "attacker": str, "damage": float, "cause": str}``。

    ``attacker`` 只在伤害事件里可能缺失（摔伤/虚空等无攻击者的伤害）——保留为空字符串，
    由调用方计入 ``unattributed``；击杀事件把 ``killer`` 当攻击者。
    """
    events: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("category") != "event":
                continue
            fields = parse_record_content(row.get("content"))
            at = _num(row.get("at"), -1.0)
            if at < 0:
                continue
            if "damage" in fields and "victim" in fields:
                events.append({
                    "at": at,
                    "kind": "damage",
                    "victim": fields.get("victim", ""),
                    "attacker": fields.get("attacker", ""),
                    "damage": _num(fields.get("damage")),
                    "cause": fields.get("cause", ""),
                })
            elif fields.get("cause") == "KILL" and "victim" in fields:
                events.append({
                    "at": at,
                    "kind": "kill",
                    "victim": fields.get("victim", ""),
                    # 录制里击杀写 `killer=`；没有 killer 的 KILL 是环境死亡（摔死/虚空）。
                    "attacker": fields.get("killer", ""),
                    "damage": 0.0,
                    "cause": "KILL",
                })
    return events


def resolve_record_origin(meta: dict) -> tuple[Optional[float], str]:
    """解析「t=0」的绝对时刻（epoch 毫秒），返回 ``(origin_ms, 来源标签)``。

    优先级（2026-11-28 真机核对，见模块 docstring）：

    1. ``meta.boardSession`` = ``flat-1v1-<epochMs>``：**精确到毫秒**，且比录制首个事件只早 3ms；
    2. ``meta.startedAt``：ISO 字符串，**只有秒精度**（最坏差 1 秒，对 850ms 的伤害节奏是灾难性的）；
    3. 都没有 → ``(None, "missing")``，调用方必须告警并回退。

    ``frames.jsonl`` 只有相对秒 ``t``，没有绝对时间，所以这里不能拿帧当基准。
    """
    session = str(meta.get("boardSession") or "").strip()
    if session:
        tail = session.rsplit("-", 1)[-1]
        if tail.isdigit() and len(tail) >= 12:
            return float(tail), "boardSession"

    started = str(meta.get("startedAt") or "").strip()
    if started:
        for parser in (_parse_iso_local_ms, _parse_epoch_ms):
            value = parser(started)
            if value is not None:
                return value, "startedAt"
    return None, "missing"


def _parse_iso_local_ms(text: str) -> Optional[float]:
    """ISO 时间 → epoch 毫秒（无时区时按本地时间，与 ``startedAt`` 的写法一致）。"""
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.timestamp() * 1000.0


def _parse_epoch_ms(text: str) -> Optional[float]:
    """纯数字时间戳（秒或毫秒）→ 毫秒。"""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value > 1e12:            # 已经是毫秒
        return value
    if value > 1e9:             # 秒
        return value * 1000.0
    return None


def select_record_files(target: Path, events_by_file: dict[Path, list[dict]],
                        *, origin_ms: float, end_ms: float,
                        names: tuple[str, ...]) -> list[Path]:
    """挑出属于这一场的录制文件（目录模式）。单文件模式由调用方直接给。

    两个条件同时满足才算：

    - **时间窗**：文件的首末事件与 ``[origin_ms, end_ms]`` 有交叠（前后各留 2 秒）；
    - **参与者**：文件里出现过的 ``victim``/``attacker`` 至少命中一个 ``names``
      （大小写不敏感）。多局对局会产出多个文件，按首个事件时间排序后返回。
    """
    wanted = {str(name).lower() for name in names if name}
    lower_bound = origin_ms - RECORD_WINDOW_SLACK_MS
    upper_bound = end_ms + RECORD_WINDOW_SLACK_MS
    chosen: list[tuple[float, Path]] = []
    for path, events in events_by_file.items():
        if not events:
            continue
        first = min(event["at"] for event in events)
        last = max(event["at"] for event in events)
        if last < lower_bound or first > upper_bound:
            continue
        seen = set()
        for event in events:
            for key in ("victim", "attacker"):
                value = str(event.get(key) or "").lower()
                if value:
                    seen.add(value)
        if wanted and not (seen & wanted):
            continue
        chosen.append((first, path))
    return [path for _first, path in sorted(chosen)]


def frame_seconds(frames: list[dict]) -> list[float]:
    """帧的**对局绝对时间**（秒）。

    多局对局时 `frames.jsonl` 里的 `t` 是「本局已过秒数」，每局都会归零（真机实测：
    3 局 637 帧，`t` 在 20.11→0.07、21.38→0.08 处重置）；拿它跟 `arena-record` 的
    绝对毫秒对齐就会把第 2、3 局的事件全算到第 1 局头上。所以录制侧同时写 `tAbs`
    （整场累计秒数），这里优先用它，老录制没有 `tAbs` 时退回 `t`。
    """
    if frames and all("tAbs" in frame for frame in frames):
        return [_num(frame.get("tAbs")) for frame in frames]
    return [_num(frame.get("t")) for frame in frames]


def align_record_rewards(frames: list[dict], events: list[dict], *,
                         origin_ms: float, player_name: str, bot_name: str,
                         win_bonus: float = 0.0) -> dict:
    """把事件按时间对齐到帧，返回逐帧 ``rewards`` / ``damageDealt`` / ``damageTaken``。

    对齐规则（任务书原话）：一帧覆盖 ``[t, t+dt)``，事件落在哪段就计入哪帧。
    ``dt`` 用相邻帧差（10Hz 采样下约 0.1s）；最后一帧覆盖 ``[t_last, +∞)``，
    首帧之前的事件钳到首帧——都不丢，但要分别计数，便于发现「时间基准整体偏了」。

    伤害归属：``victim`` 是我方 → ``damageTaken`` + 伤害、``rewards`` − 伤害；
    ``victim`` 是对手 → ``damageDealt`` + 伤害、``rewards`` + 伤害。
    ``attacker`` 不是两者之一（或缺失）的伤害事件计入 ``unattributed``，**不进 rewards**。
    """
    import bisect

    count = len(frames)
    times = frame_seconds(frames)
    rounds = [int(_num(frame.get("round"), 1)) for frame in frames]
    rewards = np.zeros(count, dtype=np.float64)
    dealt = np.zeros(count, dtype=np.float64)
    taken = np.zeros(count, dtype=np.float64)

    player_key = player_name.strip().lower()
    bot_key = bot_name.strip().lower()

    #: 每局最后一帧的下标（击杀奖励落在这一帧）。
    last_frame_of_round: dict[int, int] = {}
    for index, round_number in enumerate(rounds):
        last_frame_of_round[round_number] = index

    stats = {
        "damageEvents": 0,
        "killEvents": 0,
        "unattributed": 0,
        "unattributedDamage": 0.0,
        "unknownVictim": 0,
        "beforeFirstFrame": 0,
        "afterLastFrame": 0,
        "killBonus": 0.0,
        "causes": {},
    }

    for event in events:
        relative = (_num(event.get("at")) - origin_ms) / 1000.0
        index = bisect.bisect_right(times, relative) - 1
        if index < 0:
            index = 0
            stats["beforeFirstFrame"] += 1
        elif index >= count - 1:
            index = count - 1
            if relative > times[-1]:
                stats["afterLastFrame"] += 1

        if event["kind"] == "kill":
            stats["killEvents"] += 1
            killer = str(event.get("attacker") or "").strip().lower()
            if killer == player_key and win_bonus:
                target = last_frame_of_round.get(rounds[index], index)
                rewards[target] += float(win_bonus)
                stats["killBonus"] += float(win_bonus)
            continue

        stats["damageEvents"] += 1
        cause = str(event.get("cause") or "?")
        stats["causes"][cause] = stats["causes"].get(cause, 0) + 1
        amount = float(event.get("damage") or 0.0)
        victim = str(event.get("victim") or "").strip().lower()
        attacker = str(event.get("attacker") or "").strip().lower()

        if victim == player_key:
            # 我方挨打：伤害来自对手才算「被对手打」。
            if attacker != bot_key:
                stats["unattributed"] += 1
                stats["unattributedDamage"] += amount
                continue
            taken[index] += amount
            rewards[index] -= amount
        elif victim == bot_key:
            if attacker != player_key:
                stats["unattributed"] += 1
                stats["unattributedDamage"] += amount
                continue
            dealt[index] += amount
            rewards[index] += amount
        else:
            stats["unknownVictim"] += 1

    stats["damageDealt"] = float(dealt.sum())
    stats["damageTaken"] = float(taken.sum())
    return {"rewards": rewards, "damageDealt": dealt, "damageTaken": taken, "stats": stats}


def _warn(message: str) -> None:
    """``[records]`` 前缀的告警（统一走 stderr，便于测试与 CI 抓取）。"""
    print(f"[records] {message}", file=sys.stderr)


def load_arena_records(target: str | Path, *, frames: list[dict], meta: dict,
                       player_name: str, bot_name: str, win_bonus: float = 0.0,
                       warn=_warn) -> Optional[dict]:
    """总入口：``--records`` 给了路径时调用；解析不出可用事件就告警并返回 ``None``（调用方回退）。

    返回 ``{"rewards","damageDealt","damageTaken","stats","files","origin","originSource"}``。
    **绝不静默产出全 0**：只要一条可归属的伤害都没有，就当解析失败。
    """
    path = Path(target)
    origin_ms, origin_source = resolve_record_origin(meta)
    if origin_ms is None:
        warn(f"meta 里没有可用的时间基准（boardSession / startedAt 都缺），"
             f"无法把录制对齐到帧；回退 health-delta（{path}）")
        return None
    if not path.exists():
        warn(f"录制路径不存在: {path}；回退 health-delta")
        return None

    candidates = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not candidates:
        warn(f"目录里没有 *.jsonl 录制文件: {path}；回退 health-delta")
        return None

    events_by_file: dict[Path, list[dict]] = {}
    for candidate in candidates:
        try:
            events_by_file[candidate] = load_record_events(candidate)
        except OSError as exc:
            warn(f"读不了录制文件 {candidate}: {exc}")
            events_by_file[candidate] = []

    times = frame_seconds(frames)
    end_ms = origin_ms + (times[-1] if times else 0.0) * 1000.0
    if path.is_dir():
        chosen = select_record_files(path, events_by_file, origin_ms=origin_ms,
                                     end_ms=end_ms, names=(player_name, bot_name))
        if not chosen:
            warn(f"{path} 里没有与本场（{player_name} vs {bot_name}，"
                 f"{len(frames)} 帧）匹配的录制文件；回退 health-delta")
            return None
    else:
        chosen = [path]

    events: list[dict] = []
    for item in chosen:
        events.extend(events_by_file.get(item, []))
    events.sort(key=lambda event: event["at"])
    if not events:
        warn(f"录制文件里没有解析出伤害/击杀事件: {', '.join(str(p) for p in chosen)}；"
             "回退 health-delta")
        return None

    aligned = align_record_rewards(
        frames, events, origin_ms=origin_ms,
        player_name=player_name, bot_name=bot_name, win_bonus=win_bonus)
    stats = aligned["stats"]
    attributed = stats["damageDealt"] + stats["damageTaken"]
    if attributed <= 0.0:
        warn(f"解析出 {stats['damageEvents']} 条伤害事件，但没有一条能归属到 "
             f"{player_name} / {bot_name}（unattributed={stats['unattributed']}）；"
             "回退 health-delta")
        return None

    aligned.update({
        "files": [str(item) for item in chosen],
        "origin": float(origin_ms),
        "originSource": origin_source,
    })
    return aligned


# --- BC ---------------------------------------------------------------------

def _action_label(value: Any) -> Any:
    """v1 标签是动作号（int），v2 标签是十头元组；``behavior_clone`` 自己按策略分流。"""
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array)
    return tuple(int(item) for item in array.reshape(-1))


def run_bc(dataset: dict, *, base_model: Optional[str], epochs: int,
           run_dir: Optional[Path], meta: dict, seed: int = 0,
           min_samples: int = 20) -> Optional[dict]:
    """在现有模型上做一轮行为克隆，产物写到新 run 目录（**不覆盖**模型权重）。"""
    from gladiator_train.demos import behavior_clone
    from gladiator_train.policy import Policy

    actions = dataset["actions"]
    # v2 的样本第三字段是**权重**（不是 v1 的回报），必须给 1.0，否则整批样本权重为 0。
    default_weight = 0.0
    samples = [(dataset["obs"][i], _action_label(actions[i]), default_weight)
               for i in range(len(actions))]
    weight_list = dataset.get("weights")
    if weight_list is not None and len(weight_list) != len(samples):
        return {"skipped": True,
                "reason": f"权重长度 {len(weight_list)} 与样本数 {len(samples)} 不一致"}
    if len(samples) < int(min_samples):
        return {"skipped": True, "reason": f"样本太少（{len(samples)} < {min_samples}）"}

    if base_model:
        # 与 AWR 同一条口径：旧模型右侧补零到当前 OBS_DIM，否则现役 23/25 维模型
        # 一律被「维度不一致」拒掉，真人录像 BC 根本跑不起来（走 model_io）。
        from gladiator_train.model_io import ModelLoadError, load_model

        try:
            policy = load_model(base_model, obs_dim=OBS_DIM).policy
        except (OSError, ValueError, ModelLoadError) as exc:
            return {"skipped": True, "reason": f"模型加载失败: {exc}"}
        if policy.obs_dim != OBS_DIM:
            return {"skipped": True,
                    "reason": f"模型 obsDim={policy.obs_dim} 与数据集 {OBS_DIM} 不一致"}
    else:
        policy = Policy(obs_dim=OBS_DIM, n_actions=len(ACTION_NAMES), hidden=64, seed=seed)

    if getattr(policy, "is_v2", False):
        # v2 把第三字段当权重用（v1 当回报且不使用），统一改成 1.0 再交给 behavior_clone。
        samples = [(obs, label, 1.0) for obs, label, _ in samples]
        if dataset.get("actionSpace") != "v2":
            return {"skipped": True,
                    "reason": "起点是 v2 模型，但数据集动作不是 v2（加 --target-action-space v2）"}

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


def _resolve_action_space(target: str, base_model: Optional[str]) -> str:
    """``--target-action-space auto`` 时按起点模型判 v1 / v2（无模型按 v1）。"""
    choice = (target or "auto").strip().lower()
    if choice in ("v1", "v2"):
        return choice
    if not base_model:
        return "v1"
    try:
        from gladiator_train.model_io import load_model

        return load_model(base_model, obs_dim=OBS_DIM).action_space
    except (OSError, ValueError):
        # 模型读不出来时不要在这里报错：后面 run_bc / AWR 会给更具体的失败原因。
        return "v1"


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    record_arg = args.record_dir or args.record_opt
    if not record_arg:
        print("缺少录制目录（位置参数或 --record）", file=sys.stderr)
        return 2
    record_dir = Path(record_arg).resolve()
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

    action_space = _resolve_action_space(args.target_action_space, args.base_model)

    arena = None
    if args.records:
        arena = load_arena_records(
            args.records, frames=frames, meta=meta,
            player_name=player_name, bot_name=bot_name, win_bonus=args.win_bonus)
    dataset = build_dataset(
        frames, player_name=player_name, bot_name=bot_name, round_seconds=round_seconds,
        winners=load_round_winners(record_dir),
        win_weight=args.win_weight, loss_weight=args.loss_weight,
        draw_weight=args.draw_weight, contact_range=args.contact_range,
        contact_weight=1.0,
        # round 模式只要 (a)：把接触窗口关掉（远/近都按 1.0），不另写一条权重路径。
        far_weight=1.0 if args.weighting == "round" else args.far_weight,
        win_bonus=args.win_bonus,
        action_space=action_space,
        arena=arena,
    )
    reward_source = "arena-record" if arena else "health-delta"
    if args.no_prefer_wins:
        dataset["weights"] = np.ones_like(dataset["weights"])
    weight_rules: dict[str, Any] = {
        "mode": "uniform" if args.no_prefer_wins else args.weighting,
        "win": args.win_weight, "loss": args.loss_weight,
        "draw": args.draw_weight, "contactRange": args.contact_range,
        "contact": 1.0, "far": 1.0 if args.weighting == "round" else args.far_weight,
        "winBonus": args.win_bonus,
    }
    if args.weighting == "awr" and not args.no_prefer_wins:
        if not args.base_model:
            print("[awr] 需要 --base-model（价值头来自模型，不能凭空造）", file=sys.stderr)
            return 2
        try:
            # 旧模型（23/25 维）在 37 维数据集上算 V(s) 会 reshape 失败：走 model_io 统一
            # 的「右侧补零到当前 OBS_DIM」口径（与 league/mirror 同一条规则，见 G-703）。
            from gladiator_train.model_io import load_model

            awr_record = load_model(args.base_model, obs_dim=OBS_DIM)
            awr_policy = awr_record.policy
        except (OSError, ValueError) as exc:
            print(f"[awr] 模型加载失败: {exc}", file=sys.stderr)
            return 2
        if awr_policy.obs_dim != OBS_DIM:
            print(f"[awr] 模型 obsDim={awr_policy.obs_dim} 与数据集 {OBS_DIM} 不一致"
                  "（先用同名观测维度的模型算 V(s)）", file=sys.stderr)
            return 2
        try:
            weights, awr_info = compose_awr_weights(
                dataset, awr_policy,
                temperature=args.awr_temperature, clip=args.awr_clip,
                strict=args.awr_strict)
        except ValueError as exc:
            print(f"[awr] {exc}", file=sys.stderr)
            return 2
        dataset["weights"] = weights.astype(np.float32)
        weight_rules.update(awr_info)
        if not weight_rules["awrReturns"].startswith("dataset.rewards"):
            print("[awr] 注意：数据集没有 rewards，A = -V；"
                  "权重会系统性偏向 critic 认为**差**的状态（与 AWR 原意相反），"
                  "先跑录制侧带 rewards 更可靠", file=sys.stderr)
        if not weight_rules["awrEffective"]:
            print("[awr] 警告：本次 AWR 不是真优势加权"
                  f"（awrReturns={weight_rules['awrReturns']}、"
                  f"distinct={weight_rules['awrDistinct']}、"
                  f"zeroed={weight_rules['awrZeroed']}）——"
                  "别拿它当「人类比模型强」的证据；要真开 AWR 得让数据里带 rewards",
                  file=sys.stderr)
    out_path = Path(args.out) if args.out else record_dir / "dataset.npz"
    #: G-736：`--out` 给目录（含「无扩展名」的路径）时写 <dir>/dataset.npz——
    #: 任务书/README 里都是 `--out ..\training\artifacts\tmp-g736` 这种写法。
    if args.out and (out_path.is_dir() or out_path.suffix == ""):
        out_path = out_path / "dataset.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    macro = dataset.get("macroActions")
    if macro is None:
        macro = dataset["actions"]
    counts = {ACTION_NAMES[index]: int((macro == index).sum())
              for index in range(len(ACTION_NAMES))}
    np.savez(
        out_path,
        obs=dataset["obs"],
        actions=dataset["actions"],
        times=dataset["times"],
        rounds=dataset["rounds"],
        weights=dataset["weights"],
        rewards=dataset["rewards"],
        macroActions=np.asarray(macro, dtype=np.int64),
        **({"damageDealt": dataset["damageDealt"],
            "damageTaken": dataset["damageTaken"]} if arena else {}),
        meta=np.array(json.dumps({
            "schema": "gladiator.human-match-dataset/v1",
            "recordDir": str(record_dir),
            "player": player_name,
            "bot": bot_name,
            "frames": len(frames),
            "samples": int(len(dataset["actions"])),
            "obsDim": OBS_DIM,
            "actionSpace": action_space,
            "headSizes": list(dataset.get("headSizes") or []),
            "actionNames": list(ACTION_NAMES),
            "labelRules": "见 import_human_match.py 模块 docstring",
            "preferWins": not args.no_prefer_wins,
            "weightRules": weight_rules,
            "rewardSource": reward_source,
            "arenaRecord": None if not arena else {
                "files": arena["files"],
                "origin": arena["origin"],
                "originSource": arena["originSource"],
                "stats": arena["stats"],
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
        "rewardSource": reward_source,
        "rewards": {
            "sum": float(np.sum(dataset["rewards"])),
            "nonZeroFrames": int(np.count_nonzero(dataset["rewards"])),
        },
        "arenaRecord": None if not arena else {
            "files": arena["files"],
            "originSource": arena["originSource"],
            "stats": arena["stats"],
        },
        "bc": bc_report,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
