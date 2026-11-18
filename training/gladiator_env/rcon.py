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


#: 建连重试的间隔（秒）：Paper 的 RCON 对短时间大量新建/断开很敏感，
#: 首连失败多半是它还没准备好接受新连接，等一下再试即可。总耗时 < 1s。
CONNECT_RETRY_DELAYS: tuple[float, ...] = (0.25, 0.5)


class RconClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 25575, password: str = "devpass",
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._request_id = 0
        #: 建连尝试次数（1 次首试 + len(CONNECT_RETRY_DELAYS) 次重试 = 3 次）。
        self.connect_attempts = 1 + len(CONNECT_RETRY_DELAYS)

    # --- 连接 -----------------------------------------------------------

    def __enter__(self) -> "RconClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def connect(self) -> "RconClient":
        """建立并认证连接；失败自动重试（0.25s / 0.5s），第三次仍失败才抛原始异常。

        保留最后一次的原始异常（含 "RCON 连接被对端关闭" 这类可读信息），
        不让重试把真实原因吃掉。
        """
        last_error: Exception | None = None
        for attempt in range(self.connect_attempts):
            if attempt:
                time.sleep(CONNECT_RETRY_DELAYS[attempt - 1])
            self.close()  # 上一次可能留下半条连接，先清干净
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                sock.settimeout(self.timeout)
                self._sock = sock
                self._authenticate()
                return self
            except (RconError, OSError) as exc:
                last_error = exc
                self.close()
        assert last_error is not None
        raise last_error

    def is_connected(self) -> bool:
        """是否已有连接（只看本地状态；真正的探活由 ``command`` 的重连兜底）。"""
        return self._sock is not None

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
        """执行一条命令；连接坏了就重连一次再重试（只重试一次，避免放大故障）。"""
        try:
            return self._command_once(command)
        except RconError:
            if not self._reconnect_for_retry():
                raise
            return self._command_once(command)

    def _reconnect_for_retry(self) -> bool:
        """为「命令重试」重建连接；失败返回 False（调用方抛出原始错误）。"""
        self.close()
        try:
            self.connect()
        except (RconError, OSError):
            return False
        return True

    def _command_once(self, command: str) -> str:
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
