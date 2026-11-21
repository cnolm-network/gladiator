"""人机对战编排器（块 C）。

把一个「人类玩家 vs 指定 Bot」的对局跑起来并跑完：选/拉起运行时 → 经 RCON 让插件 spawn →
每局摆位发装备 → 倒计时 → 采样判定 → 记分 → 可选录制与赛后导入训练。

设计边界：
- **规则不进 Java**：传送/清包/发装备/回血/判胜负全部在这里用 RCON 命令编排，
  插件只提供观测（`/gladiator stats`）与生命周期（`spawn`/`list`）；
- **不重复启动**：`attach` 模式优先复用已存在的运行时（含用户手动启动的外部进程），
  只有需要大脑而进程没带 `--brain` 时才重启（外部进程不允许重启，直接报冲突）；
- **可测试**：RCON、时钟、sleep 都可注入，单测用脚本化的假连接在毫秒级跑完一整场。

状态机（`phase`）：
``idle → preparing → countdown → fighting → roundEnd → … → finished``
异常/取消落在 ``failed`` / ``stopped``。
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from config import DashboardConfig
from errors import BadRequest, Conflict, Forbidden
from ports import is_port_free, is_port_open
from rcon_client import RconClient, RconError, run_command, strip_formatting
from runtime_manager import RuntimeManager
from server_manager import ServerManager
from validation import (parse_bool, parse_center, parse_int, parse_soft_spec,
                        validate_username)

_RE_LIST_BOT = re.compile(r"^\s*-\s+\S+\s+name=(\S+).*?ready=(true|false)", re.MULTILINE)

PHASES = ("idle", "preparing", "countdown", "fighting", "roundEnd", "finished", "stopped", "failed")

#: 每局准备阶段的倒计时秒数（与训练场 `--prep-seconds` 的语义一致）。
COUNTDOWN_SECONDS = 3

#: 采样周期：录制时 10Hz（任务要求的录制频率），不录制时 5Hz（足够抓住死亡瞬间）。
RECORD_PERIOD = 0.1
JUDGE_PERIOD = 0.2

#: 套件（`give` 的参数；带空格表示数量）。
KITS: dict[str, tuple[str, ...]] = {
    "diamond": (
        "diamond_sword",
        "diamond_helmet",
        "diamond_chestplate",
        "diamond_leggings",
        "diamond_boots",
        "golden_apple 5",
        "fishing_rod",
    ),
    "none": (),
}

#: 给双方回血/回饱食度用的原版效果 id（1.8：6=瞬间治疗，23=饱和）。
HEAL_EFFECTS = ("6", "23")

#: 倒计时免伤：1.8 的抗性（11）放大器 4 = 抗性 V = 完全免伤。
SHIELD_EFFECT_ID = 11
SHIELD_AMPLIFIER = 4
SHIELD_SECONDS = COUNTDOWN_SECONDS + 2


def parse_ready_bots(raw: str) -> dict[str, bool]:
    """从 ``gladiator list`` 输出解析 {username: ready}。"""
    text = strip_formatting(raw or "")
    result: dict[str, bool] = {}
    for match in _RE_LIST_BOT.finditer(text):
        result[match.group(1)] = match.group(2) == "true"
    return result


def parse_stats_line(raw: str) -> Optional[dict]:
    """解析 ``/gladiator stats <player>`` 的单行输出；失败/离线返回 None。"""
    text = strip_formatting(raw or "").strip()
    if not text or "stats 失败" in text:
        return None
    result: dict[str, Any] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value in ("true", "false"):
            result[key] = value == "true"
            continue
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result if ("health" in result or "name" in result) else None


def judge_round(player: Optional[dict], bot: Optional[dict], elapsed: float,
                round_seconds: float, prev_player: Optional[dict] = None,
                prev_bot: Optional[dict] = None) -> Optional[dict]:
    """判定当前局是否结束；未结束返回 ``None``。

    规则（全部是"读数→结论"，不含游戏规则）：
    1. 某一方读不到 stats（离线/被踢）→ 对方胜；
    2. 血量 ≤ 0 → 对方胜（kill）；
    3. 血量从低位突然回到接近满值 → 判为"死亡后重生"，对方胜
       （1.8 死亡到重生只有 2 tick，5~10Hz 采样会漏掉 health=0 的那一帧）；
    4. 到达局时长 → 血量低者判负，相等为平局（timeout）。
    """
    if player is None:
        return {"winner": "bot", "reason": "player-offline"}
    if bot is None:
        return {"winner": "player", "reason": "bot-offline"}
    player_health = float(player.get("health", 20.0))
    bot_health = float(bot.get("health", 20.0))
    if player_health <= 0:
        return {"winner": "bot", "reason": "kill"}
    if bot_health <= 0:
        return {"winner": "player", "reason": "kill"}
    if _looks_respawned(prev_player, player):
        return {"winner": "bot", "reason": "player-respawn"}
    if _looks_respawned(prev_bot, bot):
        return {"winner": "player", "reason": "bot-respawn"}
    if elapsed >= float(round_seconds):
        if player_health > bot_health:
            return {"winner": "player", "reason": "timeout"}
        if bot_health > player_health:
            return {"winner": "bot", "reason": "timeout"}
        return {"winner": "draw", "reason": "timeout"}
    return None


def _looks_respawned(previous: Optional[dict], current: dict) -> bool:
    if not previous:
        return False
    try:
        before = float(previous.get("health", 20.0))
        after = float(current.get("health", 20.0))
    except (TypeError, ValueError):
        return False
    return before <= 12.0 and after >= 19.5


#: 默认出生点相对中点的横向偏移：玩家 -4 格、Bot +4 格（面对面 8 格）。
DEFAULT_SPAWN_OFFSET = 4.0

#: 死亡后回到站位的判定半径（格）。1.8 的 tp 落点会有 ±0.5 的实体碰撞修正。
SPAWN_TOLERANCE = 2.0

#: 强制重生后最多等多久确认双方回到站位（秒）。
RESPAWN_RECOVER_TIMEOUT = 5.0

#: 恢复窗口内重发 ``tp`` 的间隔（秒）：客户端重生后的本地位置预测会覆盖一次 tp。
RESPAWN_PLACE_REFRESH = 0.4

#: 触发「强制重生 + 重新摆位」的判定原因（离线不属于此列：人不在线重生也没用）。
DEATH_REASONS = ("kill", "player-respawn", "bot-respawn")

#: 单局内最多强制重生几次：防止异常状态（一直掉血/一直读不到位置）把这一局拖死。
MAX_RESPAWNS_PER_ROUND = 20


def derive_default_spawns(player_name: str, player_stats: Optional[dict],
                          bot_name: str, bot_stats: Optional[dict]) -> Optional[dict]:
    """未指定 ``spawns`` 时按双方当前位置推导出生点。

    规则（任务书 bug1-2）：中点 ``x/z`` 为场地中心、``y`` 取玩家当前 ``y``；
    玩家落在中心 ``-4`` 格、Bot 落在 ``+4`` 格。任一方的 stats 读不到时返回 ``None``
    （调用方记一条告警并放弃自动出生点，而不是拿 0,0,0 乱传）。

    返回 ``{player_name: (x, y, z), bot_name: (x, y, z)}``，坐标保留 2 位小数，
    与 RCON 命令里的格式一致。
    """
    if not player_stats or not bot_stats:
        return None
    try:
        px = float(player_stats["x"])
        pz = float(player_stats["z"])
        py = float(player_stats.get("y", 4.0))
        bx = float(bot_stats["x"])
        bz = float(bot_stats["z"])
    except (KeyError, TypeError, ValueError):
        return None
    center_x = (px + bx) / 2.0
    center_z = (pz + bz) / 2.0
    return {
        player_name: (round(center_x - DEFAULT_SPAWN_OFFSET, 2), round(py, 2), round(center_z, 2)),
        bot_name: (round(center_x + DEFAULT_SPAWN_OFFSET, 2), round(py, 2), round(center_z, 2)),
    }


def rekit_commands(names: list, kit: str) -> list[str]:
    """死亡后的补装命令（只补死掉的那一方，不打扰还活着的对手）。

    1.8 死亡会清空背包并掉落物品；不补装的话，剩下半局只能用拳头，
    而 brain 会因为「背包里没有剑」一直空挥（ITEM_MISSING）。
    不含倒计时免伤（抗性 V）：重生后无敌 5 秒会破坏对局。
    """
    commands: list[str] = []
    for name in names:
        commands.append(f"clear {name}")
    for name in names:
        commands.append(f"gamemode 0 {name}")
    for name in names:
        for item in KITS.get(kit, ()):
            commands.append(f"give {name} {item}")
    for name in names:
        commands.append(f"effect {name} clear")
    for name in names:
        for effect in HEAL_EFFECTS:
            commands.append(f"effect {name} {effect} 1 20 true")
    return commands


def _dead_sides(plan: MatchPlan, player: Optional[dict], bot: Optional[dict],
                reason: Optional[str]) -> list:
    """这一刀到底谁死了（用于只给死者补装）。

    血量能判断时以血量为准；「重生启发式」命中时血量已经回到满值，只能看判定原因。
    """
    dead: list = []
    player_health = 20.0 if player is None else float(player.get("health", 20.0))
    bot_health = 20.0 if bot is None else float(bot.get("health", 20.0))
    if reason == "player-respawn" or player_health <= 0:
        dead.append(plan.player)
    if reason == "bot-respawn" or bot_health <= 0:
        dead.append(plan.bot)
    return dead or [plan.player, plan.bot]


def _at_spawn(stats: Optional[dict], point: Optional[tuple],
              tolerance: float = SPAWN_TOLERANCE) -> bool:
    """该玩家是否「活着且已经回到出生点」（只看水平距离，y 由服务器落地决定）。"""
    if not stats or not point:
        return False
    if float(stats.get("health", 0.0)) <= 0:
        return False
    try:
        dx = float(stats.get("x", 0.0)) - float(point[0])
        dz = float(stats.get("z", 0.0)) - float(point[2])
    except (TypeError, ValueError):
        return False
    return abs(dx) <= tolerance and abs(dz) <= tolerance


def round_place_commands(player: str, bot: str,
                         spawns: Optional[dict]) -> list[str]:
    """开局摆位命令：双方传送到起点、重生点设到起点、解除倒计时保护。

    只在倒计时结束后执行：运行时大脑从启动就开始决策，若在准备阶段就把 Bot 放到
    玩家身边，它会在倒计时里把玩家打死（真机冒烟 2026-11-18 22:28 就是这种情况）。
    重生点一并设到起点，否则 1.8 的默认重生点是世界出生点，死后会跑到几千格外。
    ``effect clear`` 必须无条件下发：倒计时里给的是抗性 V（免伤），不清就会整局无敌。
    """
    commands: list[str] = []
    for name in (player, bot):
        point = (spawns or {}).get(name)
        if not point:
            continue
        commands.append(f"tp {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
        commands.append(f"spawnpoint {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
    for name in (player, bot):
        commands.append(f"effect {name} clear")
    return commands


def round_setup_commands(player: str, bot: str, kit: str,
                         spawns: Optional[dict], place: bool = True) -> list[str]:
    """一局的全部准备命令（顺序固定，便于复盘与单测）。

    ``place=True`` 时把 ``tp`` 放在最前面（历史行为）；对局编排用 ``place=False``，
    摆位改由 :func:`round_place_commands` 在倒计时之后执行。
    """
    commands: list[str] = []
    if spawns and place:
        for name in (player, bot):
            point = spawns.get(name)
            if point:
                commands.append(f"tp {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
    for name in (player, bot):
        commands.append(f"clear {name}")
    for name in (player, bot):
        commands.append(f"gamemode 0 {name}")
    for name in (player, bot):
        for item in KITS.get(kit, ()):
            commands.append(f"give {name} {item}")
    for name in (player, bot):
        commands.append(f"effect {name} clear")
    for name in (player, bot):
        for effect in HEAL_EFFECTS:
            commands.append(f"effect {name} {effect} 1 20 true")
    # 倒计时期间的免伤：Bot 大脑从运行时启动就在决策，倒计时里可能已经把玩家打死。
    # 抗性 V（放大器 4）在 1.8 里等于完全免伤；倒计时结束后由摆位命令 clear 掉。
    for name in (player, bot):
        commands.append(f"effect {name} {SHIELD_EFFECT_ID} {SHIELD_SECONDS} {SHIELD_AMPLIFIER} true")
    return commands


def build_import_command(python_exe: str, script: Path, record_dir: Path,
                         *, bc: bool, base_model: Optional[str],
                         dataset: Optional[Path] = None) -> list[str]:
    """赛后导入训练的命令行（纯函数，单测直接断言参数）。"""
    command = [str(python_exe), str(script), str(record_dir)]
    if dataset is not None:
        command.extend(["--out", str(dataset)])
    if bc:
        command.append("--bc")
        if base_model:
            command.extend(["--base-model", str(base_model)])
    return command


def argv_brain_spec(argv: list) -> Optional[str]:
    """从运行时命令行里取 ``--brain`` 的值；没带则 None。"""
    items = [str(item) for item in (argv or [])]
    for index, item in enumerate(items):
        if item == "--brain" and index + 1 < len(items):
            return items[index + 1]
    return None


def choose_runtime(runtimes: list, bot: str, env_port: int,
                   brain_spec: Optional[str]) -> dict:
    """决定复用/重启/新起运行时。

    返回 ``{"action": "reuse"|"restart"|"start"|"blocked", "runtime": dict|None, "reason": str}``。
    优先按用户名匹配（含 external），其次按通道 B 端口匹配。
    """
    alive = [item for item in (runtimes or []) if item.get("alive")]
    by_name = [item for item in alive
               if str(item.get("username") or "").lower() == bot.lower()]
    by_port = [item for item in alive if item.get("envPort") == env_port]
    target = (by_name or by_port or [None])[0]
    if target is None:
        return {"action": "start", "runtime": None, "reason": "没有匹配的运行时"}
    has_brain = argv_brain_spec(target.get("command") or [])
    if brain_spec:
        if has_brain == brain_spec:
            return {"action": "reuse", "runtime": target, "reason": "运行时已带同一模型"}
        if target.get("external"):
            return {
                "action": "blocked",
                "runtime": target,
                "reason": (
                    f"外部运行时 :{target.get('envPort')} 未带 --brain {brain_spec}，"
                    "Dashboard 不会停止外部进程；请手动重启它并加上 --brain，或改用 spawn 模式"
                ),
            }
        return {
            "action": "restart",
            "runtime": target,
            "reason": "将以大脑模式重启" if not has_brain else f"模型变更，将以 {brain_spec} 重启",
        }
    if has_brain:
        return {
            "action": "reuse",
            "runtime": target,
            "reason": "运行时带大脑，本次不指定模型（沿用进程内模型）",
        }
    return {"action": "reuse", "runtime": target, "reason": "复用已存在的运行时"}


def _username(value: Any, field: str) -> str:
    """用户名必填校验（与 `/gladiator spawn` 的名字规则一致）。"""
    text = str(value or "").strip()
    if not text:
        raise BadRequest(f"{field} 不能为空")
    return validate_username(text, field)


@dataclass
class MatchPlan:
    """一次人机对战的全部配置（默认值即推荐用法）。"""

    mode: str = "spawn"
    bot: str = "Glad_Bot_1"
    env_port: int = 8770
    player: str = ""
    rounds: int = 3
    round_seconds: float = 120.0
    kit: str = "diamond"
    spawns: Optional[dict] = None
    brain: Optional[str] = None
    record: bool = True
    train_after: bool = False
    brain_path: Optional[str] = None
    #: 强制重生后等多久再传送（毫秒）：等待客户端完成重生、实体重新进入世界。
    respawn_delay_ms: int = 800
    #: 软约束串（运行时参数，默认 "" = 全开）：``cps=12,noRod``。
    #: 只透传给 bot-runtime 的 ``--soft``，**不写进模型 / 模型清单**。
    soft: str = ""
    #: 本次是否因为「勾了赛后训练」而强制打开录制（前端据此提示用户）。
    record_forced: bool = False

    @classmethod
    def from_payload(cls, payload: dict, cfg: DashboardConfig) -> "MatchPlan":
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        mode = str(payload.get("mode") or "spawn").strip().lower()
        if mode not in ("attach", "spawn"):
            raise BadRequest("mode 只能是 attach（接入已有运行时）或 spawn（新建运行时）")
        bot = _username(payload.get("bot") or "Glad_Bot_1", "Bot 用户名")
        player = _username(payload.get("player"), "玩家名")
        env_port = parse_int(payload.get("envPort", cfg.runtime_port_min), "通道 B 端口",
                             minimum=1024, maximum=65535)
        if env_port in (cfg.mc_port, cfg.bridge_port, cfg.rcon_port, cfg.port):
            raise BadRequest(f"通道 B 端口不能使用保留端口 {env_port}")
        rounds = parse_int(payload.get("rounds", 3), "局数", minimum=1, maximum=20)
        round_seconds = parse_int(payload.get("roundSeconds", 120), "每局时长（秒）",
                                  minimum=10, maximum=1800)
        kit = str(payload.get("kit") or "diamond").strip().lower()
        if kit not in KITS:
            raise BadRequest(f"kit 只能是 {'/'.join(sorted(KITS))}")
        spawns = _parse_spawns(payload.get("spawns"), player, bot)
        brain = payload.get("brain")
        brain = str(brain).strip() if brain not in (None, "") else None
        record = parse_bool(payload.get("record", True), "record")
        train_after = parse_bool(payload.get("trainAfter", False), "trainAfter")
        # 赛后训练的数据来自录制：没有录制就没有 frames.jsonl，导入只会拿到空数据集。
        # 这里静默补上录制（而不是 400），并在快照里标注 recordForced，前端负责提示。
        record_forced = False
        if train_after and not record:
            record = True
            record_forced = True
        respawn_delay_ms = parse_int(payload.get("respawnDelayMs", 800), "重生等待（毫秒）",
                                     minimum=0, maximum=5000)
        # 软约束是运行参数：语法错就在这里 400，别等拉起运行时才炸。
        soft = parse_soft_spec(payload.get("soft", ""), "软约束（soft）")
        return cls(mode=mode, bot=bot, env_port=env_port, player=player,
                   rounds=rounds, round_seconds=float(round_seconds), kit=kit,
                   spawns=spawns, brain=brain, record=record, train_after=train_after,
                   respawn_delay_ms=respawn_delay_ms, soft=soft,
                   record_forced=record_forced)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "bot": self.bot,
            "envPort": self.env_port,
            "player": self.player,
            "rounds": self.rounds,
            "roundSeconds": self.round_seconds,
            "kit": self.kit,
            "spawns": self.spawns,
            "brain": self.brain,
            "record": self.record,
            "trainAfter": self.train_after,
            "respawnDelayMs": self.respawn_delay_ms,
            "soft": self.soft,
            "recordForced": self.record_forced,
        }


def _parse_spawns(value: Any, player: str, bot: str) -> Optional[dict]:
    """``spawns`` 允许缺省/单侧；两侧都给时才真正传送。"""
    if value in (None, "", {}):
        return None
    if not isinstance(value, dict):
        raise BadRequest("spawns 需要 {player:{x,y,z}, bot:{x,y,z}} 或 null")
    out: dict[str, tuple] = {}
    for key, name in (("player", player), ("bot", bot)):
        raw = value.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, dict):
            # 前端也可能直接发 {x,y,z}；统一成 [x,y,z] 再走公共校验。
            raw = [raw.get("x"), raw.get("y"), raw.get("z")]
        out[name] = parse_center(raw, f"spawns.{key}")
    return out or None


class MatchRunner:
    """人机对战编排器（每个 Dashboard 进程一个实例）。"""

    def __init__(self, cfg: DashboardConfig, server: ServerManager, runtimes: RuntimeManager,
                 models=None, *, rcon_factory: Optional[Callable[[], Any]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 sleeper: Optional[Callable[[float], None]] = None) -> None:
        self.cfg = cfg
        self.server = server
        self.runtimes = runtimes
        self.models = models
        self._rcon_factory = rcon_factory
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._lock = threading.RLock()
        self._match: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        #: `gladiator list` 的 ready 快照（前端下拉要标 ready），2 秒节流。
        self._ready_map: dict[str, bool] = {}
        self._ready_at = 0.0

    # --- 查询 -----------------------------------------------------------
    def current(self, refresh: bool = False) -> dict:
        """返回对局快照。

        ``refresh=True`` 时顺带刷新 Bot 的 ready 列表（2 秒节流、失败静默），
        供前端「Bot 下拉标 ready」使用；对局内部的 ready 由编排线程自己维护。
        """
        ready = self._refresh_ready() if refresh else self._ready_map
        available = self._available_bots(ready)
        with self._lock:
            match = self._match
            if match is None:
                return {
                    **self._base_snapshot(),
                    "status": "idle",
                    "phase": "idle",
                    "running": False,
                    "count": 0,
                    "bots": [],
                    "availableBots": available,
                    "round": 0,
                    "rounds": 0,
                    "roundSeconds": 0,
                    "score": {"player": 0, "bot": 0, "draws": 0},
                    "roundsDetail": [],
                    "brain": None,
                    "defaultSpawns": None,
                    "error": None,
                    "log": [],
                    "startedAt": None,
                    "startedAtEpoch": 0,
                    "durationSeconds": 0,
                    "countdown": None,
                }
            snapshot = dict(match["snapshot"])
            snapshot["startedAtEpoch"] = match["started_at"]
            snapshot["startedAt"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(match["started_at"]))
            snapshot["durationSeconds"] = round(
                (match["finished_at"] or time.time()) - match["started_at"], 1)
            snapshot["log"] = list(match["log"])[-120:]
            snapshot["roundsDetail"] = [dict(item) for item in match["rounds"]]
            snapshot["score"] = dict(match["score"])
            snapshot["bots"] = [dict(item) for item in match["bots"]]
            snapshot["availableBots"] = available
            return snapshot

    def _refresh_ready(self) -> dict[str, bool]:
        now = time.time()
        if now - self._ready_at < 2.0:
            return self._ready_map
        self._ready_at = now
        if not is_port_open(self.cfg.rcon_host, self.cfg.rcon_port, timeout=0.2):
            self._ready_map = {}
            return self._ready_map
        try:
            raw = run_command(self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                              "gladiator list", timeout=1.5)
            self._ready_map = parse_ready_bots(raw)
        except (RconError, OSError):
            # 服务器没开/RCON 抽风都不该让状态接口失败，保留上一次结果。
            pass
        return self._ready_map

    def _available_bots(self, ready: dict[str, bool]) -> list[dict]:
        """可选的 Bot：运行时里的用户名 ∪ 插件已知的 Bot（带 ready 标记）。"""
        entries: dict[str, dict] = {}
        try:
            runtimes = self.runtimes.list()
        except Exception:  # noqa: BLE001 - 扫描外部进程失败不影响下拉
            runtimes = []
        for runtime in runtimes:
            name = str(runtime.get("username") or "").strip()
            if not name:
                continue
            entries[name.lower()] = {
                "username": name,
                "envPort": runtime.get("envPort"),
                "external": bool(runtime.get("external")),
                "alive": bool(runtime.get("alive")),
                "ready": bool((ready or {}).get(name)),
            }
        for name, is_ready in (ready or {}).items():
            key = str(name).lower()
            if key in entries:
                entries[key]["ready"] = bool(is_ready)
                continue
            entries[key] = {
                "username": name,
                "envPort": None,
                "external": False,
                "alive": True,
                "ready": bool(is_ready),
            }
        return sorted(entries.values(), key=lambda item: str(item["username"]).lower())

    def _base_snapshot(self) -> dict:
        return {
            "address": f"{self.cfg.mc_host}:{self.cfg.mc_port}",
            "version": "1.8.9",
            "onlineMode": "offline",
            "mcRunning": is_port_open(self.cfg.mc_host, self.cfg.mc_port),
            "rconReachable": is_port_open(self.cfg.rcon_host, self.cfg.rcon_port),
            # 空闲时也给出 soft 字段，前端不用区分「没有这个键」与「空串」。
            "soft": "",
            "commands": {
                "join": f"{self.cfg.mc_host}:{self.cfg.mc_port}",
                "spectate": "/gladiator spectate <bot>",
                "unspectate": "/gladiator unspectate",
            },
        }

    # --- 启动 -----------------------------------------------------------
    def start(self, payload: dict) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        with self._lock:
            if self._match is not None and self._match["snapshot"]["running"]:
                raise Conflict("已有一场人机对战在进行中，请先结束")

        plan = MatchPlan.from_payload(payload, self.cfg)
        if plan.brain:
            plan.brain_path = self._resolve_brain(plan.brain)
        if not is_port_open(self.cfg.mc_host, self.cfg.mc_port):
            raise Conflict("Paper 未运行：请先在「服务管理」启动服务器")
        if not is_port_open(self.cfg.rcon_host, self.cfg.rcon_port):
            raise Conflict("RCON 端口未监听，无法执行对局命令；请确认 server.properties 已开启 RCON")
        if plan.mode == "spawn" and not is_port_free(self.cfg.mc_host, plan.env_port):
            raise Conflict(f"通道 B 端口 {plan.env_port} 已被占用；attach 模式可复用该运行时")
        # attach + 大脑 + 外部进程这类"必须重启但不能重启"的情况在启动前就拒绝，
        # 否则用户只会看到一个已经 failed 的对局。
        decision = choose_runtime(self.runtimes.list(), plan.bot, plan.env_port, plan.brain)
        if decision["action"] == "blocked":
            raise Conflict(decision["reason"])

        match = {
            "snapshot": {
                **self._base_snapshot(),
                "status": "starting",
                "phase": "preparing",
                "running": True,
                **plan.to_dict(),
                "brain": plan.brain,
                "round": 0,
                "count": 1,
                "countdown": None,
                "recordDir": None,
                "runtime": None,
                # plan.spawns 为 null 时，开局会用双方的 stats 推导一组默认出生点写在这里。
                "defaultSpawns": None,
                "error": None,
                "strategyLabel": plan.brain or "脚本策略",
            },
            "plan": plan,
            "started_at": time.time(),
            "finished_at": None,
            "rounds": [],
            "score": {"player": 0, "bot": 0, "draws": 0},
            "bots": [{"username": plan.bot, "envPort": plan.env_port, "ready": False}],
            "log": [],
            "cancel": False,
            "started_runtime_ids": [],
            "frames_file": None,
            "training": None,
        }
        with self._lock:
            self._match = match
            self._cancel.clear()
        self._log(match, f"对局配置：{plan.player} vs {plan.bot}（{plan.rounds} 局 × "
                         f"{int(plan.round_seconds)}s，kit={plan.kit}，模式={plan.mode}）")
        self._thread = threading.Thread(target=self._worker, args=(match,), name="match", daemon=True)
        self._thread.start()
        return self.current()

    def _resolve_brain(self, model_id: str) -> str:
        """把模型 id 解析成绝对 npz 路径（校验存在），供 `--brain` 使用。"""
        if self.models is None:
            raise BadRequest("Dashboard 没有装配模型注册表，无法使用 brain")
        try:
            path = self.models.policy_path(model_id)
        except Exception as exc:  # noqa: BLE001 - gmod 缺失等统一转成中文 400
            raise BadRequest(f"模型 {model_id} 不可用：{exc}") from exc
        if path is None or not Path(path).is_file():
            raise BadRequest(f"模型 {model_id} 还没有权重文件（policy.npz），请先训练或导入模型")
        return str(Path(path).resolve())

    # --- 工具 -----------------------------------------------------------
    def _log(self, match: dict, text: str, ok: bool = True) -> None:
        with self._lock:
            match["log"].append({
                "at": time.strftime("%H:%M:%S"),
                "text": text,
                "ok": ok,
            })
            if len(match["log"]) > 300:
                del match["log"][:-300]

    def _set_phase(self, match: dict, phase: str, round_no: Optional[int] = None,
                   countdown: Optional[float] = None) -> None:
        with self._lock:
            match["snapshot"]["phase"] = phase
            if round_no is not None:
                match["snapshot"]["round"] = round_no
            match["snapshot"]["countdown"] = countdown

    def _cancelled(self, match: dict) -> bool:
        return self._cancel.is_set() or bool(match.get("cancel"))

    def _connect(self):
        if self._rcon_factory is not None:
            return self._rcon_factory()
        client = RconClient(self.cfg.rcon_host, self.cfg.rcon_port,
                            self.cfg.rcon_password, timeout=2.0)
        client.connect()
        return client

    def _command(self, conn, text: str) -> str:
        try:
            return conn.command(text)
        except (RconError, OSError):
            return ""

    def _stats(self, conn, name: str) -> Optional[dict]:
        return parse_stats_line(self._command(conn, f"gladiator stats {name}"))

    def _bot_ready(self, conn, name: str) -> bool:
        return bool(parse_ready_bots(self._command(conn, "gladiator list")).get(name))

    # --- 运行线程 -------------------------------------------------------
    def _worker(self, match: dict) -> None:
        plan: MatchPlan = match["plan"]
        conn = None
        try:
            conn = self._connect()
            self._ensure_runtime(match, plan)
            self._ensure_spawned(match, plan, conn)
            self._ensure_spawns(match, plan, conn)
            self._begin_recording(match, plan)
            for index in range(1, plan.rounds + 1):
                if self._cancelled(match):
                    raise _Cancelled()
                self._set_phase(match, "preparing", index)
                self._log(match, f"第 {index}/{plan.rounds} 局：摆位与发装备")
                self._prepare_round(match, plan, conn)
                self._countdown(match, plan, index)
                self._set_phase(match, "fighting", index, countdown=None)
                self._log(match, f"第 {index} 局开始")
                result = self._fight_round(match, plan, index, conn)
                self._set_phase(match, "roundEnd", index)
                self._finish_round(match, plan, result)
                self._sleep(1.0)
            self._set_phase(match, "finished")
            with self._lock:
                match["snapshot"]["running"] = False
                match["snapshot"]["status"] = "finished"
                match["finished_at"] = time.time()
            self._write_summary(match)
            score = match["score"]
            self._log(match, f"对局结束：玩家 {score['player']} : {score['bot']} Bot"
                             f"（平 {score['draws']}）")
            if plan.train_after:
                self._start_import(match, plan)
        except _Cancelled:
            self._close_recording(match)
            with self._lock:
                match["snapshot"]["running"] = False
                match["snapshot"]["status"] = "stopped"
                match["snapshot"]["phase"] = "stopped"
                match["finished_at"] = time.time()
            self._log(match, "对局已终止")
        except Exception as exc:  # noqa: BLE001 - 汇总为对局失败
            self._close_recording(match)
            with self._lock:
                match["snapshot"]["running"] = False
                match["snapshot"]["status"] = "failed"
                match["snapshot"]["phase"] = "failed"
                match["snapshot"]["error"] = str(exc)
                match["finished_at"] = time.time()
            self._log(match, f"对局失败：{exc}", ok=False)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cleanup(match, plan)

    def _cleanup(self, match: dict, plan: MatchPlan) -> None:
        """只清理本次为 spawn 模式拉起的运行时；attach 的进程留给用户。"""
        if plan.mode != "spawn":
            return
        for runtime_id in list(match.get("started_runtime_ids") or []):
            try:
                self.runtimes.stop(runtime_id)
            except Exception:  # noqa: BLE001 - 已退出/外部进程
                pass

    # --- 运行时 ---------------------------------------------------------
    def _ensure_runtime(self, match: dict, plan: MatchPlan) -> None:
        candidates = self.runtimes.list()
        decision = choose_runtime(candidates, plan.bot, plan.env_port, plan.brain)
        action = decision["action"]
        target = decision.get("runtime") or {}
        if action == "blocked":
            raise Conflict(decision["reason"])
        if action == "reuse":
            self._log(match, f"复用运行时 {target.get('id')}"
                             f"（PID {target.get('pid')}）：{decision['reason']}")
            with self._lock:
                match["snapshot"]["runtime"] = _runtime_view(target)
            return
        if action == "restart":
            self._log(match, f"重启运行时 {target.get('id')}：{decision['reason']}")
            try:
                self.runtimes.stop(target["id"])
            except Exception as exc:  # noqa: BLE001
                raise Conflict(f"重启运行时失败：{exc}") from exc
            self._sleep(0.5)
        runtime = self.runtimes.start(plan.bot, plan.env_port,
                                     extra_args=self._brain_argv(plan))
        with self._lock:
            match["started_runtime_ids"].append(runtime.get("id"))
            match["snapshot"]["runtime"] = _runtime_view(runtime)
        self._log(match, f"已启动运行时 {runtime.get('id')}（PID {runtime.get('pid')}）")
        self._wait_listening(plan.env_port)

    def _brain_argv(self, plan: MatchPlan) -> list[str]:
        if not plan.brain_path:
            return []
        argv = [
            "--brain", plan.brain_path,
            "--models-dir", str((self.cfg.var_dir / "models").resolve()),
            # 20Hz 与 BC/训练的控制步长对齐；5Hz 时出手频率只有 1/4（真机 2026-11-18 实测）。
            "--brain-hz", "20",
            "--rcon", f"{self.cfg.rcon_host}:{self.cfg.rcon_port}:{self.cfg.rcon_password}",
            "--target-player", plan.player,
        ]
        if plan.soft:
            # 运行参数：只决定「这一个进程怎么用这个模型」，模型文件一字不动。
            argv.extend(["--soft", plan.soft])
        return argv

    def _wait_listening(self, env_port: int, timeout: float = 20.0) -> None:
        deadline = self._clock() + timeout
        while self._clock() < deadline:
            if is_port_open(self.cfg.mc_host, env_port, timeout=0.2):
                return
            self._sleep(0.25)

    def _ensure_spawned(self, match: dict, plan: MatchPlan, conn) -> None:
        # 通道 A 会话数（spawn 前至少要有一个运行时连上来）
        deadline = self._clock() + 20.0
        while self._clock() < deadline:
            if self._cancelled(match):
                raise _Cancelled()
            raw = strip_formatting(self._command(conn, "gladiator status"))
            found = re.search(r"sessions=(\d+)", raw)
            if found and int(found.group(1)) >= 1:
                break
            self._sleep(0.5)
        if not self._bot_ready(conn, plan.bot):
            text = strip_formatting(self._command(conn, f"gladiator spawn {plan.bot}")).strip()
            self._log(match, f"spawn {plan.bot}: {text or '(无输出)'}")
            if "失败" in text and "已就绪" not in text:
                raise RuntimeError(f"gladiator spawn 失败：{text}")
        deadline = self._clock() + 40.0
        while self._clock() < deadline:
            if self._cancelled(match):
                raise _Cancelled()
            ready = self._bot_ready(conn, plan.bot)
            with self._lock:
                match["bots"] = [{"username": plan.bot, "envPort": plan.env_port, "ready": ready}]
            if ready:
                self._log(match, f"{plan.bot} 已就绪")
                return
            self._sleep(1.0)
        raise RuntimeError(f"等待 {plan.bot} 就绪超时（40s）；可查看运行时日志")

    def _ensure_spawns(self, match: dict, plan: MatchPlan, conn) -> None:
        """``plan.spawns`` 为空时，用开局第一次 stats 推导默认出生点。

        为什么要推导：真机自测（2026-11-18 22:53/22:56/22:58 三场）都没填 spawns，
        结果整局没有摆位、也没有重生点，玩家死后回到世界出生点，Bot 连人都找不到。
        """
        if plan.spawns:
            with self._lock:
                match["snapshot"]["spawns"] = {name: list(point)
                                               for name, point in plan.spawns.items()}
            return
        player_stats = bot_stats = None
        for _ in range(4):
            if self._cancelled(match):
                raise _Cancelled()
            player_stats = self._stats(conn, plan.player)
            bot_stats = self._stats(conn, plan.bot)
            if player_stats and bot_stats:
                break
            self._sleep(0.5)
        spawns = derive_default_spawns(plan.player, player_stats, plan.bot, bot_stats)
        if spawns is None:
            self._log(match, "无法读取双方坐标，跳过自动出生点（死后可能回到世界出生点）", ok=False)
            return
        plan.spawns = spawns
        with self._lock:
            match["snapshot"]["spawns"] = {name: list(point) for name, point in spawns.items()}
            match["snapshot"]["defaultSpawns"] = {name: list(point)
                                                  for name, point in spawns.items()}
        px, py, pz = spawns[plan.player]
        bx, _, bz = spawns[plan.bot]
        self._log(match, "未指定出生点，使用自动出生点："
                         f"玩家({px:.1f}, {py:.1f}, {pz:.1f}) / Bot({bx:.1f}, {py:.1f}, {bz:.1f})")

    # --- 每局 -----------------------------------------------------------
    def _prepare_round(self, match: dict, plan: MatchPlan, conn) -> None:
        for command in round_setup_commands(plan.player, plan.bot, plan.kit, plan.spawns,
                                           place=False):
            if self._cancelled(match):
                raise _Cancelled()
            output = strip_formatting(self._command(conn, command)).strip()
            lowered = output.lower()
            if output and ("unknown command" in lowered or "未知的命令" in output):
                self._log(match, f"命令不可用（已跳过）：{command} → {output}", ok=False)
        # 摆位/发装备是异步生效的，给服务器 1 秒把实体位置与背包同步到客户端。
        self._sleep(1.0)

    def _countdown(self, match: dict, plan: MatchPlan, index: int) -> None:
        self._set_phase(match, "countdown", index, countdown=float(COUNTDOWN_SECONDS))
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            if self._cancelled(match):
                raise _Cancelled()
            self._log(match, f"第 {index} 局倒计时 {remaining}")
            self._set_phase(match, "countdown", index, countdown=float(remaining))
            self._sleep(1.0)

    def _period(self, plan: MatchPlan) -> float:
        return RECORD_PERIOD if plan.record else JUDGE_PERIOD

    def _place_sides(self, match: dict, plan: MatchPlan, conn) -> None:
        """倒计时结束后再摆位，避免 Bot 在倒计时里就开打。"""
        commands = round_place_commands(plan.player, plan.bot, plan.spawns)
        if not commands:
            return
        for command in commands:
            if self._cancelled(match):
                raise _Cancelled()
            output = strip_formatting(self._command(conn, command)).strip()
            if output and "unknown command" in output.lower():
                self._log(match, f"命令不可用（已跳过）：{command} → {output}", ok=False)
        # 传送是异步生效的：等客户端位置回包，再取基线血量。
        self._sleep(0.4)

    def _recover_after_death(self, match: dict, plan: MatchPlan, conn,
                             dead: Optional[list] = None) -> bool:
        """死亡后：强制重生 → 补装 → 传送回出生点 → 确认站位。

        1.8 的死亡界面必须客户端点「重生」，服务端用 `Player#spigot().respawn()` 代理这一步
        （插件命令 `gladiator respawn <player>`）。对双方都发一次：活着的那一方返回「跳过」，
        这是预期行为，不是错误。

        ``dead`` 是确实死掉的一方（用于补装）：只补死者，不动还活着的对手。
        返回 ``True`` 表示双方 5 秒内回到站位、本局可以继续；``False`` 表示按死亡判负。
        """
        spawns = plan.spawns or {}
        targets = list(dead or [plan.player, plan.bot])
        self._log(match, f"检测到死亡（{', '.join(targets)}）：已强制重生 + 重新摆位")
        for name in (plan.player, plan.bot):
            output = strip_formatting(self._command(conn, f"gladiator respawn {name}")).strip()
            self._log(match, f"respawn {name}: {output or '(无输出)'}",
                      ok=not output.startswith("respawn 失败"))
        # 1.8 重生后客户端要重新加载世界，立刻 tp 会被拉回原位置。
        self._sleep(max(0.0, plan.respawn_delay_ms / 1000.0))
        # 死亡清空背包：补装，否则剩下半局只能空手，brain 会一直 ITEM_MISSING。
        for command in rekit_commands(targets, plan.kit):
            if self._cancelled(match):
                raise _Cancelled()
            self._command(conn, command)
        for name in (plan.player, plan.bot):
            point = spawns.get(name)
            if not point:
                continue
            self._command(conn, f"tp {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
            self._command(conn,
                          f"spawnpoint {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
        # 重生后的客户端会先做一次「回到世界出生点」的本地预测，再被服务端拉回：
        # 只发一次 tp 很容易被这次本地预测覆盖，因此窗口内按 REFRESH 周期重发，
        # 直到死者真的站在出生点。存活的一方不要求站定——它会自己追击，站定反而
        # 会把正常对局判成恢复失败（真机 23:44 就是这样：bot 追出去 → 判负）。
        deadline = self._clock() + RESPAWN_RECOVER_TIMEOUT
        next_refresh = self._clock() + RESPAWN_PLACE_REFRESH
        while self._clock() < deadline:
            if self._cancelled(match):
                raise _Cancelled()
            player = self._stats(conn, plan.player)
            bot = self._stats(conn, plan.bot)
            if (self._sides_recovered(plan, targets, spawns, player, bot)):
                self._log(match, "双方已回到站位，本局继续")
                return True
            if self._clock() >= next_refresh:
                next_refresh = self._clock() + RESPAWN_PLACE_REFRESH
                for name in targets:
                    point = spawns.get(name)
                    if not point:
                        continue
                    self._command(
                        conn, f"tp {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
                    self._command(
                        conn, f"spawnpoint {name} {point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
            self._sleep(0.2)
        self._log(match, f"{RESPAWN_RECOVER_TIMEOUT:.0f} 秒内未回到站位，按死亡判负", ok=False)
        return False

    @staticmethod
    def _sides_recovered(plan: MatchPlan, dead: list, spawns: dict,
                         player: Optional[dict], bot: Optional[dict]) -> bool:
        """死者必须「活着且站在出生点」；存活方只要求活着（它正在打，不该站定）。"""
        for name, stats in ((plan.player, player), (plan.bot, bot)):
            if name in dead:
                if not _at_spawn(stats, spawns.get(name)):
                    return False
            elif not stats or float(stats.get("health", 0.0)) <= 0:
                return False
        return True

    def _fight_round(self, match: dict, plan: MatchPlan, index: int, conn) -> dict:
        self._place_sides(match, plan, conn)
        start_stats = {
            "player": self._stats(conn, plan.player),
            "bot": self._stats(conn, plan.bot),
        }
        started = self._clock()
        previous = {"player": start_stats["player"], "bot": start_stats["bot"]}
        # 防止时钟被冻结（例如测试注入的假时钟不前进）导致死循环。
        max_iterations = int(plan.round_seconds / self._period(plan)) + 100
        # 死亡不再直接结束这一局：强制重生 + 摆位后继续，直到时间上限或恢复失败。
        recoveries = 0
        verdict = None
        for _ in range(max_iterations):
            if self._cancelled(match):
                raise _Cancelled()
            player = self._stats(conn, plan.player)
            bot = self._stats(conn, plan.bot)
            elapsed = self._clock() - started
            if plan.record:
                self._write_frame(match, index, elapsed, player, bot)
            verdict = judge_round(player, bot, elapsed, plan.round_seconds,
                                  previous["player"], previous["bot"])
            previous = {"player": player, "bot": bot}
            if verdict is not None:
                if (verdict.get("reason") in DEATH_REASONS
                        and plan.spawns
                        and recoveries < MAX_RESPAWNS_PER_ROUND):
                    recoveries += 1
                    dead = _dead_sides(plan, player, bot, verdict.get("reason"))
                    if self._recover_after_death(match, plan, conn, dead):
                        previous = {
                            "player": self._stats(conn, plan.player),
                            "bot": self._stats(conn, plan.bot),
                        }
                        continue
                verdict["elapsed"] = elapsed
                break
            self._sleep(self._period(plan))
        if verdict is None:
            verdict = {"winner": "draw", "reason": "budget", "elapsed": self._clock() - started}
        verdict["playerBefore"] = start_stats["player"]
        verdict["playerAfter"] = previous["player"]
        verdict["botBefore"] = start_stats["bot"]
        verdict["botAfter"] = previous["bot"]
        return verdict

    def _finish_round(self, match: dict, plan: MatchPlan, verdict: dict) -> None:
        winner = str(verdict.get("winner") or "draw")
        entry = {
            "round": len(match["rounds"]) + 1,
            "winner": winner,
            "reason": verdict.get("reason"),
            "duration": round(float(verdict.get("elapsed") or 0.0), 1),
            "player": _health_view(verdict.get("playerBefore"), verdict.get("playerAfter")),
            "bot": _health_view(verdict.get("botBefore"), verdict.get("botAfter")),
        }
        with self._lock:
            match["rounds"].append(entry)
            key = "draws" if winner == "draw" else winner
            match["score"][key] = match["score"].get(key, 0) + 1
        self._log(match, f"第 {entry['round']} 局结束：{_winner_label(winner, plan)}"
                         f"（{entry['reason']}，{entry['duration']}s，玩家血量 "
                         f"{entry['player']['healthAfter']:.1f} / Bot "
                         f"{entry['bot']['healthAfter']:.1f}）")

    # --- 录制 -----------------------------------------------------------
    def _record_dir(self, plan: MatchPlan) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_player = re.sub(r"[^A-Za-z0-9_-]", "_", plan.player)
        safe_bot = re.sub(r"[^A-Za-z0-9_-]", "_", plan.bot)
        return self.cfg.artifacts_dir / "matches" / f"{stamp}-{safe_player}-vs-{safe_bot}"

    def _begin_recording(self, match: dict, plan: MatchPlan) -> None:
        if not plan.record:
            return
        directory = self._record_dir(plan)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(match, f"录制目录创建失败（继续对局不录制）：{exc}", ok=False)
            return
        meta = {
            "schema": "gladiator.match/v1",
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "plan": plan.to_dict(),
            "brain": plan.brain,
            "brainPath": plan.brain_path,
            "botRuntime": match["snapshot"].get("runtime"),
            "address": match["snapshot"]["address"],
            "frameHz": round(1.0 / RECORD_PERIOD, 2),
        }
        _write_json(directory / "meta.json", meta)
        try:
            handle = (directory / "frames.jsonl").open("a", encoding="utf-8")
        except OSError as exc:
            self._log(match, f"录制文件打不开（继续对局不录制）：{exc}", ok=False)
            return
        with self._lock:
            match["frames_file"] = handle
            match["snapshot"]["recordDir"] = str(directory)
        self._log(match, f"录制中：{directory}（{int(1 / RECORD_PERIOD)}Hz）")

    def _write_frame(self, match: dict, round_no: int, elapsed: float,
                     player: Optional[dict], bot: Optional[dict]) -> None:
        handle = match.get("frames_file")
        if handle is None:
            return
        frame = {
            "t": round(float(elapsed), 2),
            "round": round_no,
            "player": _frame_of(player),
            "bot": _frame_of(bot),
        }
        try:
            handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
            if int(elapsed * 10) % 10 == 0:
                handle.flush()
        except OSError:
            with self._lock:
                match["frames_file"] = None

    def _close_recording(self, match: dict) -> None:
        handle = match.get("frames_file")
        if handle is None:
            return
        with self._lock:
            match["frames_file"] = None
        try:
            handle.close()
        except OSError:
            pass

    def _write_summary(self, match: dict) -> None:
        self._close_recording(match)
        directory = match["snapshot"].get("recordDir")
        if not directory:
            return
        summary = {
            "schema": "gladiator.match-summary/v1",
            "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "durationSeconds": round((match["finished_at"] or time.time()) - match["started_at"], 1),
            "score": dict(match["score"]),
            "rounds": [dict(item) for item in match["rounds"]],
            "plan": match["plan"].to_dict(),
        }
        try:
            _write_json(Path(directory) / "summary.json", summary)
        except OSError as exc:
            self._log(match, f"写 summary.json 失败：{exc}", ok=False)

    # --- 赛后导入训练 ---------------------------------------------------
    def _start_import(self, match: dict, plan: MatchPlan) -> None:
        directory = match["snapshot"].get("recordDir")
        if not directory:
            self._log(match, "没有录制目录，跳过赛后导入", ok=False)
            return
        script = self.cfg.training_dir / "scripts" / "import_human_match.py"
        if not script.is_file():
            self._log(match, f"找不到导入脚本 {script}，跳过", ok=False)
            return
        dataset = Path(directory) / "dataset.npz"
        command = build_import_command(
            self.cfg.python_exe, script, Path(directory),
            bc=bool(plan.brain_path), base_model=plan.brain_path, dataset=dataset,
        )
        training = {"status": "running", "command": command, "startedAt": time.time(),
                    "output": "", "dataset": str(dataset)}
        with self._lock:
            match["training"] = training
        self._log(match, "赛后导入开始：" + " ".join(command[-6:]))

        def run() -> None:
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           cwd=str(self.cfg.training_dir), timeout=1800)
                output = (completed.stdout or "") + (completed.stderr or "")
                status = "finished" if completed.returncode == 0 else "failed"
                with self._lock:
                    training["status"] = status
                    training["returncode"] = completed.returncode
                    training["output"] = output[-4000:]
                    training["finishedAt"] = time.time()
                self._log(match, f"赛后导入{('成功' if status == 'finished' else '失败')}"
                                 f"（退出码 {completed.returncode}）", ok=status == "finished")
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    training["status"] = "failed"
                    training["output"] = str(exc)
                    training["finishedAt"] = time.time()
                self._log(match, f"赛后导入失败：{exc}", ok=False)

        threading.Thread(target=run, name="match-import", daemon=True).start()

    # --- 停止 -----------------------------------------------------------
    def stop(self) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        with self._lock:
            match = self._match
        if match is None:
            raise Conflict("当前没有人机对战")
        with self._lock:
            match["cancel"] = True
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        plan: MatchPlan = match["plan"]
        ids = list(match.get("started_runtime_ids") or [])
        if ids:
            from concurrent.futures import ThreadPoolExecutor

            def stop_one(runtime_id: str) -> None:
                try:
                    self.runtimes.stop(runtime_id)
                except Exception:  # noqa: BLE001
                    pass

            with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
                list(pool.map(stop_one, ids))
        self._close_recording(match)
        with self._lock:
            match["snapshot"]["running"] = False
            match["snapshot"]["status"] = "stopped"
            match["snapshot"]["phase"] = "stopped"
            match["finished_at"] = match["finished_at"] or time.time()
            match["bots"] = [{"username": plan.bot, "envPort": plan.env_port, "ready": False}]
        self._log(match, "对战已结束并清理")
        return self.current()


class _Cancelled(Exception):
    pass


# --- 纯函数辅助 -------------------------------------------------------------

def _runtime_view(runtime: dict) -> dict:
    return {
        "id": runtime.get("id"),
        "pid": runtime.get("pid"),
        "envPort": runtime.get("envPort"),
        "external": runtime.get("external"),
        "username": runtime.get("username"),
        "command": list(runtime.get("command") or []),
    }


def _health_view(before: Optional[dict], after: Optional[dict]) -> dict:
    before_health = 20.0 if before is None else float(before.get("health", 20.0))
    after_health = 0.0 if after is None else float(after.get("health", 0.0))
    before_armor = 0.0 if before is None else float(before.get("armor", 0.0))
    after_armor = 0.0 if after is None else float(after.get("armor", 0.0))
    return {
        "healthBefore": round(before_health, 1),
        "healthAfter": round(after_health, 1),
        "healthLost": round(max(0.0, before_health - after_health), 1),
        "armorBefore": round(before_armor, 1),
        "armorAfter": round(after_armor, 1),
    }


def _frame_of(stats: Optional[dict]) -> Optional[dict]:
    if stats is None:
        return None
    keep = ("health", "food", "saturation", "armor", "x", "y", "z",
            "yaw", "pitch", "sprinting", "blocking", "onGround", "hurt")
    return {key: stats[key] for key in keep if key in stats}


def _winner_label(winner: str, plan: MatchPlan) -> str:
    if winner == "player":
        return f"{plan.player} 胜"
    if winner == "bot":
        return f"{plan.bot} 胜"
    return "平局"


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
