"""人机对战：启动 N 个 bot-runtime，通过 RCON 请求插件 spawn，并给出观战指引。

流程（后台线程，避免 HTTP 请求长时间阻塞）：
1. 确认 Paper 已在运行（游戏端口监听）；
2. 逐个启动 bot-runtime（通道 B 端口从 8770 起）；
3. 等待运行时连上通道 A，RCON 执行 ``gladiator spawn <username>``；
4. 轮询 ``gladiator list``，直到 Bot ``ready=true`` 或超时；
5. 返回进服地址与 ``/gladiator spectate <bot>`` 提示。

只启动本模块创建的运行时；停止时不会碰用户手动启动的进程。
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

from config import DashboardConfig
from errors import BadRequest, Conflict, Forbidden
from ports import is_port_free, is_port_open
from rcon_client import RconError, run_command, strip_formatting
from runtime_manager import RuntimeManager
from server_manager import ServerManager
from validation import parse_int, validate_username

_RE_LIST_BOT = re.compile(r"^\s*-\s+\S+\s+name=(\S+).*?ready=(true|false)", re.MULTILINE)


def parse_ready_bots(raw: str) -> dict[str, bool]:
    """从 ``gladiator list`` 输出解析 {username: ready}。"""
    text = strip_formatting(raw or "")
    result: dict[str, bool] = {}
    for match in _RE_LIST_BOT.finditer(text):
        result[match.group(1)] = match.group(2) == "true"
    return result


class MatchRunner:
    def __init__(
        self,
        cfg: DashboardConfig,
        server: ServerManager,
        runtimes: RuntimeManager,
    ) -> None:
        self.cfg = cfg
        self.server = server
        self.runtimes = runtimes
        self._lock = threading.RLock()
        self._match: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()

    # --- 查询 -----------------------------------------------------------
    def current(self, refresh: bool = False) -> dict:
        with self._lock:
            match = self._match
        snapshot = self._empty_snapshot()
        if match is None:
            return {
                **snapshot,
                "status": "idle",
                "running": False,
                "count": 0,
                "bots": [],
                "error": None,
                "log": [],
                "startedAt": None,
                "durationSeconds": 0,
            }
        if refresh and match["status"] in ("starting", "running") and snapshot["rconReachable"]:
            try:
                ready = parse_ready_bots(run_command(
                    self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                    "gladiator list", timeout=1.5,
                ))
                with self._lock:
                    if self._match is match:
                        for bot in match["bots"]:
                            if bot["username"] in ready:
                                bot["ready"] = ready[bot["username"]]
            except (RconError, OSError):
                pass
        with self._lock:
            return {
                **snapshot,
                "status": match["status"],
                "running": match["status"] in ("starting", "running"),
                "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(match["started_at"])),
                "startedAtEpoch": match["started_at"],
                "durationSeconds": round(
                    (match["finished_at"] or time.time()) - match["started_at"], 1
                ),
                "count": len(match["bots"]),
                "bots": [dict(bot) for bot in match["bots"]],
                "error": match["error"],
                "log": list(match["log"])[-80:],
            }

    def _empty_snapshot(self) -> dict:
        return {
            "address": f"{self.cfg.mc_host}:{self.cfg.mc_port}",
            "version": "1.8.9",
            "onlineMode": "offline",
            "mcRunning": is_port_open(self.cfg.mc_host, self.cfg.mc_port),
            "rconReachable": is_port_open(self.cfg.rcon_host, self.cfg.rcon_port),
            "commands": {
                "join": f"{self.cfg.mc_host}:{self.cfg.mc_port}",
                "spectate": "/gladiator spectate <bot>",
                "unspectate": "/gladiator unspectate",
            },
        }

    # --- 启动 -----------------------------------------------------------
    def start(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        with self._lock:
            if self._match is not None and self._match["status"] in ("starting", "running"):
                raise Conflict("已有一场人机对战在进行中，请先结束")

        count = parse_int(payload.get("count", 2), "Bot 数量", minimum=1, maximum=8)
        prefix = validate_username(payload.get("usersPrefix") or "Glad_Bot", "用户名前缀")
        base = parse_int(payload.get("portsBase", 8770), "起始端口", minimum=1024, maximum=65000)
        ports = list(range(base, base + count))
        reserved = {self.cfg.mc_port, self.cfg.bridge_port, self.cfg.rcon_port, self.cfg.port}
        if any(port in reserved for port in ports):
            raise BadRequest("端口范围与保留端口冲突")
        if not is_port_open(self.cfg.mc_host, self.cfg.mc_port):
            raise Conflict("Paper 未运行：请先在「服务管理」启动服务器")
        if not is_port_open(self.cfg.rcon_host, self.cfg.rcon_port):
            raise Conflict("RCON 端口未监听，无法执行 gladiator spawn；请确认 server.properties 已开启 RCON")
        for port in ports:
            if not is_port_free(self.cfg.mc_host, port):
                raise Conflict(f"通道 B 端口 {port} 已被占用")

        bots = []
        for index, port in enumerate(ports, start=1):
            username = validate_username(f"{prefix}_{index}", f"Bot 用户名[{index}]")
            bots.append({
                "username": username,
                "envPort": port,
                "runtimeId": f"runtime-{port}",
                "ready": False,
                "started": False,
                "error": None,
            })
        match = {
            "status": "starting",
            "started_at": time.time(),
            "finished_at": None,
            "bots": bots,
            "error": None,
            "log": [],
            "cancel": False,
        }
        with self._lock:
            self._match = match
            self._cancel.clear()
        self._log(match, f"准备启动 {count} 个 Bot（{', '.join(b['username'] for b in bots)}）")
        self._thread = threading.Thread(
            target=self._worker, args=(match,), name="match-start", daemon=True
        )
        self._thread.start()
        return self.current()

    def _log(self, match: dict, text: str, ok: bool = True) -> None:
        with self._lock:
            match["log"].append({
                "at": time.strftime("%H:%M:%S"),
                "text": text,
                "ok": ok,
            })
            if len(match["log"]) > 200:
                del match["log"][:-200]

    def _worker(self, match: dict) -> None:
        started_runtime_ids: list[str] = []
        try:
            for bot in match["bots"]:
                if self._cancel.is_set() or match.get("cancel"):
                    raise _Cancelled()
                try:
                    runtime = self.runtimes.start(bot["username"], bot["envPort"])
                except Exception as exc:  # noqa: BLE001 - 汇总为对战失败
                    raise RuntimeError(f"启动运行时 {bot['username']} 失败: {exc}") from exc
                started_runtime_ids.append(bot["runtimeId"])
                bot["started"] = True
                bot["pid"] = runtime.get("pid")
                self._log(match, f"运行时 {bot['runtimeId']} 已启动（PID {runtime.get('pid')}）")

            self._log(match, "等待运行时连接通道 A（最多 15 秒）")
            deadline = time.monotonic() + 15
            sessions = 0
            while time.monotonic() < deadline:
                if self._cancel.is_set() or match.get("cancel"):
                    raise _Cancelled()
                try:
                    raw = run_command(
                        self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                        "gladiator status", timeout=2.0,
                    )
                    found = re.search(r"sessions=(\d+)", strip_formatting(raw))
                    if found:
                        sessions = int(found.group(1))
                        if sessions >= len(match["bots"]):
                            break
                except (RconError, OSError) as exc:
                    self._log(match, f"查询通道 A 会话失败: {exc}", ok=False)
                time.sleep(0.8)
            self._log(match, f"通道 A 当前会话数 {sessions}")

            for bot in match["bots"]:
                if self._cancel.is_set() or match.get("cancel"):
                    raise _Cancelled()
                try:
                    raw = run_command(
                        self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                        f"gladiator spawn {bot['username']}", timeout=3.0,
                    )
                except (RconError, OSError) as exc:
                    bot["error"] = str(exc)
                    self._log(match, f"spawn {bot['username']} 失败: {exc}", ok=False)
                    continue
                text = strip_formatting(raw).strip()
                self._log(match, f"spawn {bot['username']}: {text}")
                if "失败" in text:
                    bot["error"] = text

            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                if self._cancel.is_set() or match.get("cancel"):
                    raise _Cancelled()
                try:
                    raw = run_command(
                        self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                        "gladiator list", timeout=2.0,
                    )
                    ready = parse_ready_bots(raw)
                    with self._lock:
                        for bot in match["bots"]:
                            if bot["username"] in ready:
                                bot["ready"] = ready[bot["username"]]
                        ready_count = sum(1 for bot in match["bots"] if bot["ready"])
                    if ready_count >= len(match["bots"]):
                        break
                except (RconError, OSError):
                    pass
                time.sleep(1.0)

            with self._lock:
                ready_count = sum(1 for bot in match["bots"] if bot["ready"])
                match["status"] = "running" if ready_count else "failed"
                match["finished_at"] = time.time() if ready_count == 0 else None
                if ready_count < len(match["bots"]):
                    match["error"] = (
                        f"仅 {ready_count}/{len(match['bots'])} 个 Bot 就绪；"
                        "可查看运行时日志或稍后重试"
                    )
            self._log(
                match,
                f"对战就绪：{ready_count}/{len(match['bots'])} 个 Bot ready" if ready_count
                else "没有 Bot 就绪，对战启动失败",
                ok=ready_count > 0,
            )
        except _Cancelled:
            with self._lock:
                match["status"] = "stopped"
                match["finished_at"] = time.time()
            self._log(match, "对战启动已取消")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                match["status"] = "failed"
                match["finished_at"] = time.time()
                match["error"] = str(exc)
            self._log(match, f"对战启动失败：{exc}", ok=False)
        finally:
            if match["status"] not in ("running",):
                for runtime_id in started_runtime_ids:
                    try:
                        self.runtimes.stop(runtime_id)
                    except Exception:  # noqa: BLE001 - 清理失败不覆盖原始错误
                        pass

    # --- 停止 -----------------------------------------------------------
    def _safe_stop_runtime(self, runtime_id: str) -> None:
        try:
            self.runtimes.stop(runtime_id)
        except Exception:  # noqa: BLE001 - 可能已退出
            pass

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
        runtime_ids = [bot["runtimeId"] for bot in list(match["bots"]) if bot.get("started")]
        if runtime_ids:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(8, len(runtime_ids))) as pool:
                list(pool.map(self._safe_stop_runtime, runtime_ids))
        for bot in list(match["bots"]):
            bot["ready"] = False
        with self._lock:
            match["status"] = "stopped"
            match["finished_at"] = time.time()
        self._log(match, "对战已结束，Bot 已下线")
        return self.current()


class _Cancelled(Exception):
    pass
