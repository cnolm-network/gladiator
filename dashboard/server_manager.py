"""Paper 服务器进程管理 + `gladiator status` 摘要解析。"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from config import DashboardConfig
from errors import Conflict, Forbidden
from ports import is_port_free, is_port_open, pid_listening_on
from processes import ManagedProcess, ProcessManager
from rcon_client import RconError, run_commands, strip_formatting

PROC_ID = "paper"

_RE_VERSION = re.compile(r"version=(\S+)\s+api=(\S+)\s+ready=(true|false)\s+worldguard=(\S+)")
_RE_COUNTS = re.compile(r"\[Gladiator\]\s+bots=(\d+)\s+games=(\d+)\s+training=(\d+)")
_RE_LINK = re.compile(r"\[Gladiator\]\s+link=(\S+)\s+sessions=(\d+)")
_RE_HEAP = re.compile(r"heap used=(\d+)MB max=(\d+)MB")
_RE_TPS = re.compile(r"(?:TPS[^:]*:)?\s*\*?\s*(\d+(?:\.\d+)?)")


def parse_gladiator_status(raw: str) -> dict | None:
    """把 RCON 返回的多行状态解析成结构化 dict；无法识别时返回 None。"""
    text = strip_formatting(raw or "")
    result: dict = {
        "raw": text.strip(),
        "version": None,
        "api": None,
        "ready": None,
        "worldguard": None,
        "bots": None,
        "games": None,
        "training": None,
        "sessions": None,
        "link": None,
        "heapUsedMb": None,
        "heapMaxMb": None,
    }
    found = False
    match = _RE_VERSION.search(text)
    if match:
        found = True
        result.update(
            version=match.group(1),
            api=match.group(2),
            ready=match.group(3) == "true",
            worldguard=match.group(4),
        )
    match = _RE_COUNTS.search(text)
    if match:
        found = True
        result.update(bots=int(match.group(1)), games=int(match.group(2)), training=int(match.group(3)))
    match = _RE_LINK.search(text)
    if match:
        found = True
        result.update(link=match.group(1), sessions=int(match.group(2)))
    match = _RE_HEAP.search(text)
    if match:
        found = True
        result.update(heapUsedMb=int(match.group(1)), heapMaxMb=int(match.group(2)))
    return result if found else None


def parse_tps(raw: str) -> float | None:
    text = strip_formatting(raw or "")
    if ":" in text:
        text = text.split(":", 1)[1]
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


class ServerManager:
    def __init__(self, cfg: DashboardConfig, procs: ProcessManager) -> None:
        self.cfg = cfg
        self.procs = procs
        self._lock = threading.Lock()

    # --- 查询 -----------------------------------------------------------
    def status(self, tail_lines: int = 60) -> dict:
        proc = self.procs.get(PROC_ID)
        managed = proc is not None and proc.alive
        ports = {
            "game": is_port_open(self.cfg.mc_host, self.cfg.mc_port),
            "bridge": is_port_open(self.cfg.mc_host, self.cfg.bridge_port),
            "rcon": is_port_open(self.cfg.rcon_host, self.cfg.rcon_port),
        }
        running = managed or ports["game"]
        gladiator = None
        tps = None
        rcon_error = None
        if ports["rcon"]:
            try:
                results = run_commands(
                    self.cfg.rcon_host,
                    self.cfg.rcon_port,
                    self.cfg.rcon_password,
                    ["gladiator status", "tps"],
                    timeout=2.5,
                )
                raw_status = results[0][1]
                gladiator = parse_gladiator_status(raw_status)
                if gladiator is None:
                    rcon_error = "RCON 已连接，但未识别到 gladiator status 输出"
                tps = parse_tps(results[1][1])
            except (RconError, OSError) as exc:
                rcon_error = f"RCON 查询失败: {exc}"

        snapshot = proc.snapshot(tail_lines) if proc is not None else None
        external = bool(ports["game"] and not managed)
        pid = snapshot["pid"] if snapshot else None
        if external:
            pid = pid_listening_on(self.cfg.mc_port)
        result = {
            "running": running,
            "managed": managed,
            "external": external,
            "pid": pid,
            "uptimeSeconds": snapshot["uptimeSeconds"] if snapshot else None,
            "startedAt": snapshot["startedAt"] if snapshot else None,
            "exitCode": snapshot["exitCode"] if snapshot else None,
            "ports": ports,
            "tps": tps,
            "gladiator": gladiator,
            "rconError": rcon_error,
            "logTail": snapshot["tail"] if snapshot else [],
            "logPath": snapshot["logPath"] if snapshot else None,
            "jarExists": self.cfg.server_jar_path.is_file(),
            "javaExists": self.cfg.java_available,
            "javaVersion": self.cfg.java_version,
            "javaReady": self.cfg.java_ready,
            "javaHint": "" if self.cfg.java_ready else self.cfg.java_hint,
            "jar": str(self.cfg.server_jar_path),
            "java": self.cfg.java_exe if self.cfg.java_ready else None,
            "allowProcessControl": self.cfg.allow_process_control,
            "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return result

    # --- 启动/停止 ------------------------------------------------------
    def start(self) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        with self._lock:
            proc = self.procs.get(PROC_ID)
            if proc is not None and proc.alive:
                raise Conflict(f"Paper 已在运行（PID {proc.pid}）")
            if not self.cfg.server_jar_path.is_file():
                raise Conflict(f"未找到服务端 jar: {self.cfg.server_jar_path}")
            if not self.cfg.java_ready:
                # 不尝试启动：Paper 1.8.8 在 Java 17 上必然失败，提前给可操作的错误。
                raise Conflict(
                    "未找到 Java 8，请配置 javaPath"
                    + (f"（{self.cfg.java_hint}）" if self.cfg.java_hint else "")
                )
            if not is_port_free(self.cfg.mc_host, self.cfg.mc_port):
                raise Conflict(f"端口 {self.cfg.mc_port} 已被占用，无法启动 Paper")

            argv = [
                self.cfg.java_exe,
                "-Dfile.encoding=UTF-8",
                f"-Xms{max(256, self.cfg.server_memory_mb // 2)}M",
                f"-Xmx{self.cfg.server_memory_mb}M",
                "-jar",
                self.cfg.server_jar,
                "nogui",
            ]
            self.procs.spawn(
                proc_id=PROC_ID,
                kind="server",
                label="Paper 1.8.8",
                argv=argv,
                cwd=self.cfg.server_dir,
                tail_lines=1200,
            )
        return self.status()

    def stop(self) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        proc = self.procs.get(PROC_ID)
        if proc is None or not proc.alive:
            if is_port_open(self.cfg.mc_host, self.cfg.mc_port):
                raise Conflict(
                    f"端口 {self.cfg.mc_port} 上有服务器在监听，但它不是由 Dashboard 启动的，"
                    "请到原终端停止"
                )
            return self.status()
        self.procs.stop(PROC_ID, graceful_line="stop\n", timeout=40.0)
        return self.status()

    def write_console(self, line: str) -> dict:
        """向 Paper 控制台写一行（例如 stop / say ...）。"""
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        proc = self.procs.get(PROC_ID)
        if proc is None or not proc.alive or proc.popen.stdin is None:
            raise Conflict("Paper 未由 Dashboard 托管运行")
        try:
            proc.popen.stdin.write(line.rstrip("\n") + "\n")
            proc.popen.stdin.flush()
        except OSError as exc:
            raise Conflict(f"写入控制台失败: {exc}") from exc
        return {"ok": True, "line": line}

    def wait_ready(self, timeout: float = 120.0, interval: float = 1.0) -> bool:
        """等待游戏端口 + RCON 都就绪。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_port_open(self.cfg.mc_host, self.cfg.mc_port) and is_port_open(
                self.cfg.rcon_host, self.cfg.rcon_port
            ):
                return True
            time.sleep(interval)
        return False

    def managed_snapshot(self) -> ManagedProcess | None:
        return self.procs.get(PROC_ID)

    def log_path(self) -> Path | None:
        proc = self.procs.get(PROC_ID)
        return proc.log_path if proc else None
