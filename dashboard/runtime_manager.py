"""bot-runtime（Layer 2）进程管理。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DashboardConfig
from errors import Conflict, Forbidden
from ports import is_port_free, is_port_open, listening_pids
from processes import ProcessManager
from validation import parse_port, validate_username


class RuntimeManager:
    def __init__(self, cfg: DashboardConfig, procs: ProcessManager) -> None:
        self.cfg = cfg
        self.procs = procs
        self._lock = threading.Lock()
        # id -> {username, envPort}
        self._meta: dict[str, dict] = {}

    # --- 查询 -----------------------------------------------------------
    def _reserved_ports(self) -> set[int]:
        return {self.cfg.mc_port, self.cfg.bridge_port, self.cfg.rcon_port, self.cfg.port}

    def list(self) -> list[dict]:
        out = []
        for snapshot in self.procs.list(kind="runtime", tail_lines=60):
            meta = self._meta.get(snapshot["id"], {})
            env_port = meta.get("envPort")
            listening = bool(env_port) and is_port_open(self.cfg.mc_host, int(env_port))
            out.append({
                **snapshot,
                "username": meta.get("username"),
                "envPort": env_port,
                "listening": listening,
                "external": False,
                "state": "running" if snapshot["alive"] else "stopped",
            })

        if self.cfg.scan_external:
            managed_ports = {item.get("envPort") for item in out if item.get("alive")}
            for port, pid in sorted(
                listening_pids(self.cfg.runtime_port_min, self.cfg.runtime_port_max).items()
            ):
                if port in managed_ports:
                    continue
                out.append({
                    "id": f"external-{port}",
                    "kind": "runtime",
                    "label": f"外部 bot-runtime :{port}",
                    "pid": pid,
                    "alive": True,
                    "exitCode": None,
                    "startedAt": None,
                    "startedAtEpoch": 0,
                    "uptimeSeconds": None,
                    "command": [],
                    "cwd": str(self.cfg.bot_runtime_dir),
                    "logPath": None,
                    "error": None,
                    "tail": [],
                    "username": None,
                    "envPort": port,
                    "listening": True,
                    "external": True,
                    "state": "external",
                })
        out.sort(key=lambda item: (item.get("envPort") or 0, item.get("id") or ""))
        return out

    def defaults(self) -> dict:
        """给前端一个不冲突的默认用户名 + 通道 B 端口（从配置的下限开始）。"""
        used = {item.get("envPort") for item in self.list() if item.get("alive")}
        for port in range(self.cfg.runtime_port_min, self.cfg.runtime_port_max + 1):
            if port in used or port in self._reserved_ports():
                continue
            if is_port_free(self.cfg.mc_host, port):
                return {"username": "Glad_Bot_1", "envPort": port}
        return {"username": "Glad_Bot_1", "envPort": self.cfg.runtime_port_min}

    def get(self, runtime_id: str) -> dict | None:
        for item in self.list():
            if item["id"] == runtime_id:
                return item
        return None

    def live_port_owner(self, env_port: int) -> dict | None:
        for item in self.list():
            if item.get("envPort") == env_port and item.get("alive"):
                return item
        return None

    # --- 启动/停止 ------------------------------------------------------
    def start(self, username: str, env_port: int, *, mc_host: str | None = None,
              mc_port: int | None = None) -> dict:
        username = validate_username(username)
        env_port = parse_port(env_port, "通道 B 端口", self._reserved_ports())
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        if not (self.cfg.bot_runtime_dir / "src" / "index.js").is_file():
            raise Conflict(f"未找到 bot-runtime 入口: {self.cfg.bot_runtime_dir / 'src' / 'index.js'}")

        with self._lock:
            owner = self.live_port_owner(env_port)
            if owner is not None:
                raise Conflict(
                    f"端口 {env_port} 已被运行时 {owner.get('username') or owner.get('id')} 占用"
                )
            if not is_port_free(self.cfg.mc_host, env_port):
                raise Conflict(f"端口 {env_port} 已被其他程序占用")
            runtime_id = f"runtime-{env_port}"
            existing = self.procs.get(runtime_id)
            if existing is not None and existing.alive:
                raise Conflict(f"运行时 {runtime_id} 已在运行（PID {existing.pid}）")

            argv = [
                self.cfg.node_exe,
                "src/index.js",
                "--username",
                username,
                "--host",
                mc_host or self.cfg.mc_host,
                "--mc-port",
                str(mc_port or self.cfg.mc_port),
                "--env-port",
                str(env_port),
            ]
            self.procs.spawn(
                proc_id=runtime_id,
                kind="runtime",
                label=f"Bot {username} :{env_port}",
                argv=argv,
                cwd=self.cfg.bot_runtime_dir,
                tail_lines=500,
            )
            self._meta[runtime_id] = {"username": username, "envPort": env_port}

        # 给 node 一点启动时间，状态里就能看到监听结果
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            if is_port_open(self.cfg.mc_host, env_port, timeout=0.1):
                break
            time.sleep(0.1)
        return self.get(runtime_id) or {}

    def stop(self, runtime_id: str) -> dict:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        if runtime_id.startswith("external-"):
            raise Conflict("该 bot-runtime 由外部启动，Dashboard 不会停止它")
        proc = self.procs.get(runtime_id)
        if proc is None:
            raise Conflict(f"未找到运行时 {runtime_id}")
        # terminate 后最多等 2 秒再 kill，不再先白等 8 秒。
        self.procs.stop(runtime_id, timeout=2.0, terminate_timeout=2.0)
        return self.get(runtime_id) or proc.snapshot()

    def stop_all(self) -> list[dict]:
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        ids = [item["id"] for item in self.list() if item["alive"] and not item.get("external")]
        if not ids:
            return []
        results = []
        with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
            futures = [pool.submit(self.stop, runtime_id) for runtime_id in ids]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Conflict:
                    continue
        return results

    def log_path(self, runtime_id: str) -> Path | None:
        proc = self.procs.get(runtime_id)
        return proc.log_path if proc else None

    def wait_listening(self, env_port: int, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_port_open(self.cfg.mc_host, env_port):
                return True
            time.sleep(0.25)
        return is_port_open(self.cfg.mc_host, env_port)
