"""极简 RCON 客户端（Source RCON 协议）。

Dashboard 需要读 `gladiator status`、`tps`，以及在「人机对战」里执行
`gladiator spawn <user>`。这里放一份只依赖标准库的小实现，避免 import
`training/gladiator_env`（那会连带要求 gymnasium/numpy，且 Dashboard 不应耦合训练层内部）。
协议实现与 `training/gladiator_env/rcon.py` 保持一致。
"""

from __future__ import annotations

import re
import socket
import struct
import time
from typing import Iterable

SERVERDATA_AUTH = 3
SERVERDATA_AUTH_RESPONSE = 2
SERVERDATA_EXECCOMMAND = 2
SERVERDATA_RESPONSE_VALUE = 0

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_MINECRAFT_COLOR_RE = re.compile(r"[§\u00a7].?")


class RconError(RuntimeError):
    """RCON 连接、认证或命令失败。"""


class RconClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25575,
        password: str = "devpass",
        timeout: float = 3.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.password = password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._request_id = 0

    def __enter__(self) -> "RconClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- 连接 -----------------------------------------------------------
    def connect(self) -> "RconClient":
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._authenticate()
        return self

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _authenticate(self) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._write(SERVERDATA_AUTH, request_id, self.password)
        deadline = time.monotonic() + self.timeout
        while True:
            body = self._read()
            if body.response_id == -1:
                raise RconError("RCON 认证失败：密码错误")
            if body.discard_or_auth:
                return
            if time.monotonic() > deadline:
                raise RconError("RCON 认证超时")

    # --- 命令 -----------------------------------------------------------
    def command(self, command: str) -> str:
        if self._sock is None:
            raise RconError("RCON 尚未连接")
        self._request_id += 1
        self._write(SERVERDATA_EXECCOMMAND, self._request_id, command)
        chunks: list[str] = []
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                packet = self._read()
            except socket.timeout:
                break
            chunks.append(packet.body)
            if self._sock is not None:
                self._sock.settimeout(0.15)
        if self._sock is not None:
            self._sock.settimeout(self.timeout)
        return "".join(chunks)

    # --- 帧 -------------------------------------------------------------
    def _write(self, kind: int, request_id: int, body: str) -> None:
        assert self._sock is not None
        payload = struct.pack("<ii", request_id, kind) + body.encode("utf-8") + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(payload)) + payload)

    def _read(self) -> "_Packet":
        assert self._sock is not None
        raw_size = self._recv_exact(4)
        (size,) = struct.unpack("<i", raw_size)
        if size < 10 or size > 1024 * 1024:
            raise RconError(f"RCON 响应长度非法: {size}")
        data = self._recv_exact(size)
        request_id, kind = struct.unpack("<ii", data[:8])
        body = data[8:-2].decode("utf-8", errors="replace")
        return _Packet(request_id, kind == SERVERDATA_AUTH_RESPONSE, body)

    def _recv_exact(self, size: int) -> bytes:
        assert self._sock is not None
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(size - len(chunks))
            if not chunk:
                raise RconError("RCON 连接被对端关闭")
            chunks.extend(chunk)
        return bytes(chunks)


class _Packet:
    __slots__ = ("response_id", "discard_or_auth", "body")

    def __init__(self, response_id: int, discard_or_auth: bool, body: str) -> None:
        self.response_id = response_id
        self.discard_or_auth = discard_or_auth
        self.body = body


def strip_formatting(text: str) -> str:
    """去掉 ANSI 转义与 Minecraft 的 § 颜色码。"""
    return _MINECRAFT_COLOR_RE.sub("", _ANSI_RE.sub("", text))


def run_command(host: str, port: int, password: str, command: str, timeout: float = 3.0) -> str:
    with RconClient(host, port, password, timeout=timeout) as client:
        return client.command(command)


def run_commands(
    host: str,
    port: int,
    password: str,
    commands: Iterable[str],
    timeout: float = 3.0,
) -> list[tuple[str, str]]:
    """在同一个连接里顺序执行多条命令，返回 (命令, 原始输出)。"""
    out: list[tuple[str, str]] = []
    with RconClient(host, port, password, timeout=timeout) as client:
        for command in commands:
            out.append((command, client.command(command)))
    return out
