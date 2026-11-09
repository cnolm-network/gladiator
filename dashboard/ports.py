"""端口探测工具（只用于回环地址）。"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_NETSTAT_LOCK = threading.Lock()
_NETSTAT_CACHE: tuple[float, dict[int, int]] = (0.0, {})
_NETSTAT_TTL = 2.0
_RE_NETSTAT = re.compile(
    r"^\s*TCP\s+\S*:(\d+)\s+\S+\s+(?:LISTENING|LISTEN)\s+(\d+)\s*$", re.IGNORECASE
)


def _netstat_listeners() -> dict[int, int]:
    """返回 {监听端口: PID}；Windows 用 netstat 解析，其他平台返回空。"""
    global _NETSTAT_CACHE
    now = time.monotonic()
    with _NETSTAT_LOCK:
        cached_at, cached = _NETSTAT_CACHE
        if now - cached_at < _NETSTAT_TTL:
            return cached
    result: dict[int, int] = {}
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=4,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            for line in (completed.stdout or "").splitlines():
                match = _RE_NETSTAT.match(line)
                if not match:
                    continue
                port = int(match.group(1))
                pid = int(match.group(2))
                if pid > 0 and port not in result:
                    result[port] = pid
        except (OSError, subprocess.SubprocessError, ValueError):
            result = {}
    with _NETSTAT_LOCK:
        _NETSTAT_CACHE = (time.monotonic(), result)
    return result


def listening_pids(port_min: int = 1, port_max: int = 65535) -> dict[int, int]:
    """列出指定端口范围内处于 LISTEN 的端口及 PID（读不到时返回空字典）。"""
    return {
        port: pid
        for port, pid in _netstat_listeners().items()
        if int(port_min) <= port <= int(port_max)
    }


def pid_listening_on(port: int) -> int | None:
    return _netstat_listeners().get(int(port))


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """能否建立 TCP 连接（用于判断服务是否在监听）。"""
    if not (1 <= int(port) <= 65535):
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def is_port_free(host: str, port: int) -> bool:
    """能否在 host:port 上 bind（判断端口是否被占用）。"""
    if not (1 <= int(port) <= 65535):
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_port_open(host: str, port: int, timeout: float = 30.0, interval: float = 0.3) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if is_port_open(host, port):
            return True
        time.sleep(interval)
    return is_port_open(host, port)


def wait_port_free(host: str, port: int, timeout: float = 30.0, interval: float = 0.3) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if is_port_free(host, port):
            return True
        time.sleep(interval)
    return is_port_free(host, port)
