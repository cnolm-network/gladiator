"""极简 RCON 客户端（Source RCON 协议）。

训练环境（Layer 3）不直接调用 Bukkit API；「创建普通世界 + 空手混战」这类**场地准备**
属于调用方职责，本地训练场用 RCON 驱动服务端控制台完成：

- `gladiator spawn <user>` 拉起 Bot（走公开 API，等价于调用方 `spawnBot`）
- `gladiator train <user>` 把 Bot 纳入训练会话（等价于 `pushPlace`）
- `tp` / `clear` / `effect` / `spawnpoint` 摆放与回满

生产环境可以把 `RconClient` 换成调用方自己的编排接口（同一 `command()` 形态即可）。
"""

from __future__ import annotations

import socket
import struct
import time

SERVERDATA_AUTH = 3
SERVERDATA_AUTH_RESPONSE = 2
SERVERDATA_EXECCOMMAND = 2
SERVERDATA_RESPONSE_VALUE = 0


class RconError(RuntimeError):
    """RCON 连接、认证或命令失败。"""


class RconClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 25575, password: str = "devpass",
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._request_id = 0

    # --- 连接 -----------------------------------------------------------

    def __enter__(self) -> "RconClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

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
        # 认证可能先回一个空包，再回 AUTH_RESPONSE
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
            # 服务端对长输出会拆包；没有更多数据时 read 会超时退出
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
