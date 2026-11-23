"""人机对战编排器（块 C）。

把一个「人类玩家 vs 指定 Bot」的对局跑起来并跑完。**判定归板子**：

    起运行时 → `/gladiator spawn` 让 Bot 上线 → `/gladiator board start` 开局
    → 每 500ms 读 `/gladiator board status` 同步阶段 / 比分 / 逐局结果
    → 录制帧（含 rewards）+ 赛后导入 → 只清理自己拉起的运行时

设计边界（G-715 之后）：

- **规则不在 dashboard，也不在 RCON 命令里**：清场、套件、倒计时、摆位、死亡重生、判胜负、
  计分全部由服务器侧的板子（`flat-1v1`，见 `arena/`）用 Bukkit 事件驱动（`PlayerDeathEvent`
  → `ArenaKillEvent` → `RoundOutcome`）。dashboard 只发一条 start，然后**只读** status。
- **判死不轮询血量**：1.8 死亡到重生只有约 2 tick，任何 1Hz/2Hz 采样都会跨过去（真机三局
  全 `timeout` 收尾）。结束判定只认 status 的 `phase=ended`（或 `session=none`）+ 比分。
- **玩家必须自己进服**：dashboard 只 spawn Bot；玩家由人自己登录（地址见 `commands.join`）。
- **录制只做观测**：每 500ms 两条 `/gladiator stats` + 一帧 `frames.jsonl`（带本帧伤害回报）。
- **可测试**：RCON、时钟、sleep 可注入，单测用脚本化的假连接在毫秒级跑完一整场。

状态机（`phase`）是板子阶段的投影：

``idle → preparing → countdown → fighting → finished``；异常/取消落在 ``failed`` / ``stopped``。
板子的 `starting`/`idle` 映射为 `preparing`，`ended` 映射为 `finished`。
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
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

#: `/gladiator board status` 的一行输出（键序固定，允许 `board=`/`player=`/`bot=` 为空）。
_RE_BOARD_STATUS = re.compile(
    r"session=(?P<session>\S*)\s+board=(?P<board>\S*)\s+round=(?P<round>\d+)"
    r"\s+rounds=(?P<rounds>\d+)\s+phase=(?P<phase>\S+)"
    r"\s+score=(?P<player_score>\d+)\s*:\s*(?P<bot_score>\d+)\s+draws=(?P<draws>\d+)"
    r"\s+reason=(?P<reason>.*?)\s+player=(?P<player>\S*)\s+bot=(?P<bot>\S*)\s*$",
)

PHASES = ("idle", "preparing", "countdown", "fighting", "roundEnd", "finished", "stopped", "failed")

#: 板子自己声明的阶段（`BoardContext#phase`）与 dashboard 阶段名的映射来源。
BOARD_PHASES = ("idle", "starting", "countdown", "fighting", "ended")

#: 对局板子的 id（服务器侧 `Board#id()`，见 `arena/src/.../FlatBoard.java`）。
BOARD_ID = "flat-1v1"

#: status 轮询周期（秒）：契约规定 500ms。同一节奏写一帧 frames.jsonl（≈2Hz）。
STATUS_PERIOD = 0.5
# 板子引擎的采样周期：RCON 修好之后单次往返是毫秒级（旧实现在认证后丢了 data 监听，
# 每次 command() 都要等满 3s 超时），所以可以直接按 10Hz 采样——训练数据的帧密度
# 取决于它，0.5s 一帧会让 20 秒的一局只剩 40 个样本。
BOARD_STATUS_PERIOD = 0.1
# status 探测周期：阶段/比分变化是秒级，不需要每帧都问。
STATUS_PROBE_PERIOD = 0.5
# 每帧之间让出一小口气，避免把 RCON 连接打满；实际帧率由 stats 往返决定（~9Hz）。
FRAME_SLEEP = 0.02
FRAME_PERIOD = STATUS_PERIOD

#: 连续解析失败多少次后认为服务端没实现/输出坏了（12 × 500ms = 6s）。
MAX_UNPARSED_STATUS = 12

#: 刚 start 后允许 status 短暂报 `session=none` 的次数（宽限 2s，避免开局瞬间漏判为结束）。
SESSION_GRACE = 4

#: 服务端在对局中把 reason 填成 `running`（见 `GladiatorCommand#boardStatusLine`）：
#: 它不是「这一局为什么结束」，记录逐局原因时要当噪声丢掉。
RUNNING_REASON = "running"

#: 等玩家进服的秒数（超过就失败并给出进服提示，而不是让板子 `start` 报一个含糊的错误）。
PLAYER_JOIN_TIMEOUT = 30.0

#:  watchdog 余量：每局多给 20s（板子自己的倒计时/换局），整场再多给 60s。
ROUND_SLACK_SECONDS = 20.0
MATCH_SLACK_SECONDS = 60.0

#: `kit` 的请求体取值（旧前端只有 diamond/none），映射到板子的套件 id 后随 start 下发。
KIT_IDS: tuple[str, ...] = ("diamond", "none", "classic", "gapple", "sword", "empty")

#: 请求体的 kit → 板子 `kit=` 参数（`arena/ArenaKits#normalize` 认得这些名字）。
BOARD_KIT_BY_KIT: dict[str, str] = {
    "diamond": "classic",
    "classic": "classic",
    "gapple": "gapple",
    "sword": "sword",
    "none": "empty",
    "empty": "empty",
}


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


def parse_board_status(raw: str) -> Optional[dict]:
    """解析 ``/gladiator board status`` 的一行 key=value。

    契约（服务端 G-715 起）：

    ```
    session=<id|none> board=<id> round=<n> rounds=<n> phase=<idle|countdown|fighting|ended>
    score=<player>:<bot> draws=<n> reason=<...> player=<名字> bot=<名字>
    ```

    返回::

        {"session": str|None, "board": str, "round": int, "rounds": int,
         "phase": str, "score": {"player": int, "bot": int, "draws": int},
         "reason": str|None, "player": str, "bot": str}

    ``session=none``（无对局）→ ``session`` 为 ``None``；``reason=none`` → ``reason`` 为
    ``None``。输出不可解析时返回 ``None``（调用方据此判断服务端未支持/输出异常）。
    """
    text = strip_formatting(raw or "")
    for line in text.splitlines():
        parsed = _parse_board_status_line(line)
        if parsed is not None:
            return parsed
    return None


def _parse_board_status_line(line: str) -> Optional[dict]:
    match = _RE_BOARD_STATUS.search(line.strip())
    if match is not None:
        return {
            "session": _empty_to_none(match.group("session")),
            "board": _placeholder(match.group("board")),
            "round": int(match.group("round")),
            "rounds": int(match.group("rounds")),
            "phase": (match.group("phase") or "").strip().lower(),
            "score": {
                "player": int(match.group("player_score")),
                "bot": int(match.group("bot_score")),
                "draws": int(match.group("draws")),
            },
            "reason": _empty_to_none(match.group("reason")),
            "player": _placeholder(match.group("player")),
            "bot": _placeholder(match.group("bot")),
        }
    # 兜底：键序变了也尽量读出来（reason 里有空格时会截断，只在前面的正则失败时才走这里）。
    tokens: dict[str, str] = {}
    for token in line.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            tokens.setdefault(key.lower(), value)
    if "session" not in tokens or "phase" not in tokens:
        return None
    try:
        score_raw = tokens.get("score", "0:0")
        player_score, _, bot_score = score_raw.partition(":")
        return {
            "session": _empty_to_none(tokens.get("session", "")),
            "board": _placeholder(tokens.get("board", "")),
            "round": int(tokens.get("round") or 0),
            "rounds": int(tokens.get("rounds") or 0),
            "phase": (tokens.get("phase") or "").strip().lower(),
            "score": {
                "player": int(player_score or 0),
                "bot": int(bot_score or 0),
                "draws": int(tokens.get("draws") or 0),
            },
            "reason": _empty_to_none(tokens.get("reason", "")),
            "player": _placeholder(tokens.get("player", "")),
            "bot": _placeholder(tokens.get("bot", "")),
        }
    except (TypeError, ValueError):
        return None


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    """``none`` / 空串 / ``-`` 统一成 ``None``（无对局、无 reason）。"""
    text = str(value or "").strip()
    if not text or text.lower() in ("none", "-", "null"):
        return None
    return text


def _placeholder(value: Optional[str]) -> str:
    """服务端用 ``-`` 占位（`board=-`、`player=-`、`bot=-`）：dashboard 侧统一成空串。"""
    text = str(value or "").strip()
    return "" if text == "-" else text


def map_board_phase(phase: Optional[str]) -> str:
    """板子阶段 → dashboard 阶段名（未知阶段原样透出，前端按原文显示）。"""
    text = (phase or "").strip().lower()
    if text in ("", "idle", "starting", "prepare", "preparing"):
        return "preparing"
    if text in ("countdown", "prep"):
        return "countdown"
    if text in ("fighting", "battle", "running"):
        return "fighting"
    if text in ("ended", "finished", "done"):
        return "finished"
    return text


def board_finished(status: Optional[dict]) -> bool:
    """对局是否已结束：只认 ``session=none`` 或 ``phase=ended``。"""
    if not status:
        return False
    if status.get("session") is None:
        return True
    return (status.get("phase") or "").strip().lower() in ("ended", "finished")


def board_start_command(plan: "MatchPlan", board_id: str = BOARD_ID) -> str:
    """拼 `/gladiator board start`（纯函数，单测直接断言）。

    ``spawns`` 两侧都有坐标时才带上 6 个数（``px py pz bx by bz``）；缺一侧就不带，
    由板子自己决定出生点——摆位是板子的事，dashboard 不再替它猜。
    """
    parts = ["gladiator", "board", "start", board_id, plan.player, plan.bot,
             str(int(plan.rounds)), _seconds_arg(plan.round_seconds)]
    spawns = plan.spawns or {}
    player_point = spawns.get(plan.player)
    bot_point = spawns.get(plan.bot)
    if player_point and bot_point:
        for point in (player_point, bot_point):
            parts.extend(f"{float(value):.2f}" for value in point[:3])
    board_kit = BOARD_KIT_BY_KIT.get((plan.kit or "").strip().lower())
    if board_kit:
        parts.append(f"kit={board_kit}")
    return " ".join(parts)


def _seconds_arg(seconds: float) -> str:
    """局时长参数：服务端按整数解析（`GladiatorCommand#parseInt`），这里统一取整。"""
    return str(int(round(float(seconds))))


def _damage(before: Optional[dict], after: Optional[dict]) -> float:
    """血量下降量（伤害，正值）；回血/重生/读不到 = 0。"""
    if not before or not after:
        return 0.0
    try:
        old = float(before.get("health"))
        new = float(after.get("health"))
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return round(max(0.0, old - new), 2)


def frame_rewards(previous_player: Optional[dict], previous_bot: Optional[dict],
                  player: Optional[dict], bot: Optional[dict]) -> dict:
    """用相邻两帧的血量差算「本帧对对方造成的伤害」（AWR 要的逐帧回报）。

    返回 ``{"playerDamage": 本帧对 Bot 造成的伤害, "botDamage": 本帧对 Player 造成的伤害}``。
    伤害记正值；血量上升（回血/重生/换局重置）或某一侧读不到 stats 时写 0。
    """
    return {
        "playerDamage": _damage(previous_bot, bot),
        "botDamage": _damage(previous_player, player),
    }


def _infer_round_reason(winner: str, seconds: float, round_seconds: float,
                        session_reason: Optional[str]) -> str:
    """推断**逐局**结束原因（status 只给会话级 reason）。

    - 没有秒表（跨局被合并采样，`seconds<=0`）就不猜，回退到会话级原因；
    - 有胜负且没打满时长 → ``kill``（板子里除「超时」就是「击杀」/掉线判负）；
    - 平局且打满时长 → ``timeout``。
    """
    if not round_seconds or seconds <= 0:
        return session_reason or "unknown"
    full = seconds >= float(round_seconds) - 1.5
    if winner == "draw":
        return "timeout" if full else (session_reason or "unknown")
    return "kill" if not full else (session_reason or "eliminated")


@dataclass
class BoardTracker:
    """把连续的 status 采样聚合成「阶段 + 比分 + 逐局结果」。

    板子的 status 只给**累计**比分（`score=<玩家>:<bot>` 与 `draws`），没有逐局历史，
    因此逐局结果靠下面两条推断，并用 `docs/match.md` §4.4 写清楚：

    - 赢家：相邻两次采样之间的比分增量（玩家 +1 / Bot +1 / 平局 +1）；
    - `seconds`：该局从板子报 `round=N` 起算到比分变化为止的墙钟（500ms 轮询，
      量化误差 ±0.5s）；跨局被合并采样时记 0；
    - `reason`：取**最近一次**非 `running` / `none` 的 `reason`。服务端对局中固定填
      `running`，只有结束那一刻才是 `lastReason()`（会话级，如 `rounds-finished`），
      所以除末局可能拿到会话级原因外，其余记 `unknown`——逐局原因要服务端补字段。
    """

    entries: list[dict] = field(default_factory=list)
    score: dict = field(default_factory=lambda: {"player": 0, "bot": 0, "draws": 0})
    round_no: int = 0
    round_started: Optional[float] = None
    reason: Optional[str] = None
    seen_session: bool = False
    # 每局时长上限（秒）：用来把「这一局到底是打死还是超时」判出来。
    # 服务端 status 的 reason 是**会话级**的（只有末局能拿到 rounds-finished），
    # 逐局原因只能靠时长推断——见 _entry。
    round_seconds: float = 0.0

    def update(self, status: dict, now: float) -> list[dict]:
        """吸收一条 status，返回本次新增的逐局记录（0~n 条）。"""
        if status.get("session"):
            self.seen_session = True
        # 服务端在对局中是 `reason=running`，只有结束时才是真正的 lastReason；
        # 把 running 当噪声丢掉，否则整场每局都会被记成 running。
        reason = status.get("reason")
        if reason and reason.strip().lower() != RUNNING_REASON:
            self.reason = reason
        previous_started = self.round_started
        round_no = int(status.get("round") or 0)
        if round_no != self.round_no:
            self.round_no = round_no
            self.round_started = now if round_no >= 1 else None
        raw_score = status.get("score") or {}
        new_score = {
            "player": int(raw_score.get("player") or 0),
            "bot": int(raw_score.get("bot") or 0),
            "draws": int(raw_score.get("draws") or 0),
        }
        deltas = (
            ("player", max(0, new_score["player"] - int(self.score.get("player") or 0))),
            ("bot", max(0, new_score["bot"] - int(self.score.get("bot") or 0))),
            ("draw", max(0, new_score["draws"] - int(self.score.get("draws") or 0))),
        )
        created: list[dict] = []
        for winner, count in deltas:
            for _ in range(count):
                started = previous_started if not created else self.round_started
                created.append(self._entry(winner, now, started))
        self.score = new_score
        # 跨局合并采样：比分已记完但轮次号早就变到下一局了（`round_no > len(entries)`），
        # 这时秒表要在**同一次采样**里补上新一局的起点，否则下一局会记成 0 秒。
        if self.round_started is None and round_no >= 1 and len(self.entries) < round_no:
            self.round_started = now
        return created

    def _entry(self, winner: str, now: float, started: Optional[float]) -> dict:
        seconds = 0.0 if started is None else round(max(0.0, now - started), 1)
        entry = {
            "round": len(self.entries) + 1,
            "winner": winner,
            "reason": _infer_round_reason(winner, seconds, self.round_seconds, self.reason),
            "seconds": seconds,
            # 向后兼容：旧字段 duration 与 seconds 同义（前端表格与既有消费者读它）。
            "duration": seconds,
        }
        self.entries.append(entry)
        # 一局结束 → 下一局的秒表从下一次采样起算（板子可能已经在倒计时）。
        self.round_started = None
        return entry

    def round_elapsed(self, now: float) -> float:
        """当前局已进行的秒数（帧里的 `t`）。"""
        if self.round_started is None:
            return 0.0
        return round(max(0.0, now - self.round_started), 2)


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
    #: 套件标识：**仅记录**（G-715 后套件由板子按自身配置发放，dashboard 不再 `give`）。
    kit: str = "diamond"
    spawns: Optional[dict] = None
    brain: Optional[str] = None
    record: bool = True
    train_after: bool = False
    brain_path: Optional[str] = None
    #: 旧字段（G-715 前用于「强制重生后等多久再传送」）。重生归板子，保留只为请求体兼容。
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
        if kit not in KIT_IDS:
            raise BadRequest(f"kit 只能是 {'/'.join(sorted(KIT_IDS))}")
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
    """``spawns`` 允许缺省/单侧；两侧都给时才随 board start 带上坐标。"""
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
                    "boardSession": None,
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
                # G-715 后不再推导出生点（摆位归板子）；字段保留为 null 以兼容前端。
                "defaultSpawns": None,
                # 板子会话 id（`/gladiator board status` 的 session）；开局后才有值。
                "boardSession": None,
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
            "record_dir": None,
            "meta": None,
            "board_stopped": False,
            "training": None,
        }
        with self._lock:
            self._match = match
            self._cancel.clear()
        self._log(match, f"对局配置：{plan.player} vs {plan.bot}（{plan.rounds} 局 × "
                         f"{int(plan.round_seconds)}s，模式={plan.mode}，板子={BOARD_ID}）")
        self._log(match, "判定与摆位由服务器板子负责；请让玩家自己进入服务器 "
                         f"{self.cfg.mc_host}:{self.cfg.mc_port}")
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

    def _set_phase(self, match: dict, phase: str, round_no: Optional[int] = None) -> None:
        with self._lock:
            match["snapshot"]["phase"] = phase
            if round_no is not None:
                match["snapshot"]["round"] = round_no
            # 板子不回报倒计时剩余秒数：字段保留（前端读它），恒为 null。
            match["snapshot"]["countdown"] = None

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
            self._ensure_player_online(match, plan, conn)
            self._begin_recording(match, plan)
            self._board_start(match, plan, conn)
            self._follow_board(match, plan, conn)
            # 板子自己收尾时不重复发（`_follow_board` 已置位）；watchdog 兜底退出时才真停。
            self._board_stop(match, conn)
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
            self._board_stop(match, conn)
            self._close_recording(match)
            with self._lock:
                match["snapshot"]["running"] = False
                match["snapshot"]["status"] = "stopped"
                match["snapshot"]["phase"] = "stopped"
                match["finished_at"] = time.time()
            self._log(match, "对局已终止")
        except Exception as exc:  # noqa: BLE001 - 汇总为对局失败
            # 失败时板子可能还在跑：先停板子，否则下一场 start 会被「已有对局」拒绝。
            self._board_stop(match, conn)
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
        """让 Bot 上线并 ready（玩家不归 dashboard 管：人自己进服）。

        板子的 `ctx.player(bot)` 要求 Bot 已是真实 `Player`，所以必须在 board start
        之前把 Bot 拉起来并等到 ready，否则板子开局时找不到参与者。
        """
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

    def _ensure_player_online(self, match: dict, plan: MatchPlan, conn) -> None:
        """等玩家自己进服：dashboard 不拉真玩家，人必须自己登录。

        板的 `start` 在玩家离线时只会回一句「启动失败」，所以这里先给出明确提示。
        """
        deadline = self._clock() + PLAYER_JOIN_TIMEOUT
        hinted = False
        while self._clock() < deadline:
            if self._cancelled(match):
                raise _Cancelled()
            if self._stats(conn, plan.player):
                self._log(match, f"玩家 {plan.player} 已在线，准备开局")
                return
            if not hinted:
                self._log(match, f"等待玩家 {plan.player} 进入服务器 "
                                 f"{self.cfg.mc_host}:{self.cfg.mc_port}（{int(PLAYER_JOIN_TIMEOUT)}s 上限）",
                          ok=False)
                hinted = True
            self._sleep(1.0)
        raise RuntimeError(
            f"玩家 {plan.player} 未在 {int(PLAYER_JOIN_TIMEOUT)} 秒内进入服务器"
            f"（{self.cfg.mc_host}:{self.cfg.mc_port}）：请先自己进服再开始对战")

    # --- 板子 -----------------------------------------------------------
    def _board_start(self, match: dict, plan: MatchPlan, conn) -> None:
        """发一条 `/gladiator board start`；套件/倒计时/摆位/判死从此全归板子。"""
        command = board_start_command(plan)
        text = strip_formatting(self._command(conn, command)).strip()
        self._log(match, f"board start: {text or '(无输出)'}")
        lowered = text.lower()
        if not text:
            raise RuntimeError("板子启动无输出：服务端可能未实现 /gladiator board（需要 G-715 起的版本）")
        if "未知的命令" in text or "unknown command" in lowered or "不存在的命令" in text:
            raise RuntimeError(f"服务端未实现 /gladiator board：{text}")
        if "失败" in text:
            raise RuntimeError(f"板子启动失败：{text}")
        session = _parse_session_id(text)
        with self._lock:
            match["snapshot"]["boardSession"] = session
        if session:
            self._log(match, f"板子已开局（session={session}，{BOARD_ID}）")
        else:
            self._log(match, "板子已开局（未在输出里找到 session=，以 status 为准）", ok=False)
        # meta.json 记录 boardSession：录制文件开工时板子还没开，这里补写一次。
        self._update_meta(match)

    def _follow_board(self, match: dict, plan: MatchPlan, conn) -> None:
        """跟一场板子对局：status 每 500ms 探一次，录制帧每次循环都采。

        为什么分开：`gladiator stats` 在主线程算，一次往返约 50ms（20 TPS 的刻度），
        一帧要采双方 = ~110ms。把 status 也塞进同一循环，帧密度就退化成 ~3Hz；
        而 status 只需知道阶段/比分（0.5s 足够）。
        """
        tracker = BoardTracker()
        tracker.round_seconds = float(plan.round_seconds or 0.0)
        deadline = (self._clock() + plan.rounds * (plan.round_seconds + ROUND_SLACK_SECONDS)
                    + MATCH_SLACK_SECONDS)
        previous = {"round": 0, "player": None, "bot": None}
        unparsed = 0
        missing_session = 0
        next_status_at = 0.0
        status = None
        while True:
            if self._cancelled(match):
                raise _Cancelled()
            now = self._clock()
            if now >= next_status_at:
                raw = self._command(conn, "gladiator board status")
                parsed = parse_board_status(raw)
                next_status_at = now + STATUS_PROBE_PERIOD
                if parsed is None:
                    unparsed += 1
                    if unparsed >= MAX_UNPARSED_STATUS:
                        raise RuntimeError(
                            "无法解析 /gladiator board status 的输出："
                            f"{strip_formatting(raw).strip() or '(空)'}")
                    self._sleep(BOARD_STATUS_PERIOD)
                    continue
                unparsed = 0
                if parsed["session"] is None and not tracker.seen_session:
                    # 刚 start 完的几百毫秒内可能还没进对局，宽限几次再判定结束。
                    missing_session += 1
                    if missing_session < SESSION_GRACE:
                        self._sleep(BOARD_STATUS_PERIOD)
                        continue
                self._apply_status(match, plan, tracker, parsed, now)
                status = parsed
            if plan.record and tracker.round_no >= 1:
                player = self._stats(conn, plan.player)
                bot = self._stats(conn, plan.bot)
                same_round = tracker.round_no == previous["round"]
                rewards = frame_rewards(
                    previous["player"] if same_round else None,
                    previous["bot"] if same_round else None,
                    player, bot)
                self._write_frame(match, tracker.round_no, tracker.round_elapsed(self._clock()),
                                  player, bot, rewards)
                previous = {"round": tracker.round_no, "player": player, "bot": bot}
            if status is not None and board_finished(status):
                # 面板/板子已经自己收尾：stop() 不必再发一次 board stop。
                with self._lock:
                    match["board_stopped"] = True
                self._log(match, f"板子报对局结束（phase={status['phase'] or '-'}，"
                                 f"比分 {tracker.score['player']}:{tracker.score['bot']}"
                                 f"，平 {tracker.score['draws']}，"
                                 f"reason={status['reason'] or '-'}）")
                break
            if now >= deadline:
                self._log(match, f"板子在 {int(deadline)}s 内未结束，按最后一次 status 收尾"
                                 "（比分可能不是终局）", ok=False)
                break
            self._sleep(FRAME_SLEEP)

    def _apply_status(self, match: dict, plan: MatchPlan, tracker: BoardTracker,
                      status: dict, now: float) -> None:
        """把一条 status 投影到快照，并把新增的逐局结果写进 roundsDetail。"""
        entries = tracker.update(status, now)
        phase = "finished" if status["session"] is None else map_board_phase(status["phase"])
        with self._lock:
            snapshot = match["snapshot"]
            snapshot["boardSession"] = status["session"]
            snapshot["phase"] = phase
            snapshot["round"] = int(status["round"])
            if int(status["rounds"]) > 0:
                snapshot["rounds"] = int(status["rounds"])
            snapshot["score"] = dict(status["score"])
            match["rounds"] = list(tracker.entries)
            match["score"] = dict(status["score"])
            # 板子一旦进入 countdown/fighting，对局就已经真的在跑：把顶层 status 从
            # "starting" 推进到 "running"，否则前端整场都显示「准备中」。
            if phase in ("countdown", "fighting") and match["snapshot"].get("status") == "starting":
                match["snapshot"]["status"] = "running"
        for entry in entries:
            self._log(match, f"第 {entry['round']} 局结束：{_winner_label(entry['winner'], plan)}"
                             f"（{entry['reason']}，{entry['seconds']}s）")

    def _board_stop(self, match: dict, conn=None) -> None:
        """发一条 `/gladiator board stop`（幂等；板子已收尾时不再发）。"""
        with self._lock:
            if match.get("board_stopped"):
                return
            match["board_stopped"] = True
        own = conn is None
        try:
            if own:
                conn = self._connect()
            text = strip_formatting(self._command(conn, "gladiator board stop")).strip()
            self._log(match, f"board stop: {text or '(无输出)'}")
        except Exception as exc:  # noqa: BLE001 - 停板子是尽力而为
            self._log(match, f"board stop 失败（已忽略）：{exc}", ok=False)
        finally:
            if own and conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

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
            "board": BOARD_ID,
            # 板子还没开局：session 在 `_board_start` 之后补写（见 `_update_meta`）。
            "boardSession": None,
            "frameHz": round(1.0 / FRAME_PERIOD, 2),
        }
        try:
            handle = (directory / "frames.jsonl").open("a", encoding="utf-8")
        except OSError as exc:
            self._log(match, f"录制文件打不开（继续对局不录制）：{exc}", ok=False)
            return
        with self._lock:
            match["frames_file"] = handle
            match["record_dir"] = directory
            match["meta"] = meta
            match["snapshot"]["recordDir"] = str(directory)
        self._update_meta(match)
        self._log(match, f"录制中：{directory}（{round(1 / FRAME_PERIOD, 1)}Hz）")

    def _update_meta(self, match: dict) -> None:
        """把内存里的 meta（含最新 boardSession）落盘。"""
        directory = match.get("record_dir")
        meta = match.get("meta")
        if directory is None or meta is None:
            return
        with self._lock:
            meta["boardSession"] = match["snapshot"].get("boardSession")
        try:
            _write_json(Path(directory) / "meta.json", meta)
        except OSError as exc:
            self._log(match, f"写 meta.json 失败：{exc}", ok=False)

    def _write_frame(self, match: dict, round_no: int, elapsed: float,
                     player: Optional[dict], bot: Optional[dict],
                     rewards: Optional[dict] = None) -> None:
        handle = match.get("frames_file")
        if handle is None:
            return
        frame = {
            "t": round(float(elapsed), 2),
            "round": round_no,
            "player": _frame_of(player),
            "bot": _frame_of(bot),
            # 逐帧伤害回报（AWR 的输入）：本帧对对方造成的伤害，无伤害/读不到为 0。
            "rewards": dict(rewards or {"playerDamage": 0.0, "botDamage": 0.0}),
        }
        try:
            handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
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
            "board": BOARD_ID,
            "boardSession": match["snapshot"].get("boardSession"),
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
        # 编排线程已在 _Cancelled 分支里停过板子；线程卡住时这里兜底停一次（幂等）。
        if not match.get("board_stopped"):
            self._board_stop(match)
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

def _parse_session_id(text: str) -> Optional[str]:
    """从 `board 已开始 session=<id> ...` 里取 session id；没有返回 None。"""
    found = re.search(r"session=(\S+)", strip_formatting(text or ""))
    if not found:
        return None
    return _empty_to_none(found.group(1))


def _runtime_view(runtime: dict) -> dict:
    return {
        "id": runtime.get("id"),
        "pid": runtime.get("pid"),
        "envPort": runtime.get("envPort"),
        "external": runtime.get("external"),
        "username": runtime.get("username"),
        "command": list(runtime.get("command") or []),
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
