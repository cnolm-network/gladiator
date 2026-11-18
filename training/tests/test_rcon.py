"""RCON 建连重试与连接复用（P1.6 收尾）。

真实故障：训练到第 5 局 `env.reset()` 挂 `RCON 连接被对端关闭`。Paper 对「短时间大量
新建/断开 RCON」很敏感（日志里每局好几行 `Rcon connection from`），而旧代码每局
reset 都 connect 一次、`_prepare_arena` 结束就断，一局能新建 3–4 条。

修法两条：建连失败按 0.25s/0.5s 重试（共 3 次）；整局只连一次、只有断线才重连，
命令失败自动重连一次再重试。
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.rcon import CONNECT_RETRY_DELAYS, RconClient, RconError  # noqa: E402
from gladiator_env.testing import FakeLink, make_observation, make_target  # noqa: E402


def auth_packet(request_id: int = 1, body: str = "") -> bytes:
    """构造一个 AUTH_RESPONSE 帧（serverdata 2）。"""
    payload = struct.pack("<ii", request_id, 2) + body.encode("utf-8") + b"\x00\x00"
    return struct.pack("<i", len(payload)) + payload


class FakeSocket:
    """按脚本回放 recv 的假 socket；脚本用尽时返回 b""（= 对端关闭）。"""

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.chunks = list(chunks or [])
        self.sent: list[bytes] = []
        self.timeouts: list[float] = []
        self.closed = False

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        # 按 TCP 语义「最多返回 size 字节」：_recv_exact 依赖这个行为。
        while self.chunks and not self.chunks[0]:
            self.chunks.pop(0)
        if not self.chunks:
            return b""
        head = self.chunks[0]
        piece, rest = head[:size], head[size:]
        if rest:
            self.chunks[0] = rest
        else:
            self.chunks.pop(0)
        return piece

    def close(self) -> None:
        self.closed = True


class ConnectRetryTest(unittest.TestCase):
    def test_retry_after_socket_closed_then_succeed(self):
        """第一次认证读到「对端关闭」→ 重试一次成功。"""
        broken = FakeSocket([])             # 没有任何数据 → _recv_exact 直接抛 RconError
        good = FakeSocket([auth_packet()])
        sockets = [broken, good]

        with mock.patch("gladiator_env.rcon.socket.create_connection",
                        side_effect=lambda *a, **k: sockets.pop(0)) as connect, \
                mock.patch("gladiator_env.rcon.time.sleep") as sleep:
            client = RconClient(timeout=0.5)
            client.connect()

        self.assertEqual(connect.call_count, 2, "首连失败后应重试一次")
        self.assertTrue(client.is_connected())
        sleep.assert_called_once_with(CONNECT_RETRY_DELAYS[0])
        client.close()

    def test_three_attempts_then_raise_original_error(self):
        """三次都失败才抛，且保留原始错误信息（"对端关闭"）。"""
        sockets = [FakeSocket([]) for _ in range(3)]
        with mock.patch("gladiator_env.rcon.socket.create_connection",
                        side_effect=lambda *a, **k: sockets.pop(0)) as connect, \
                mock.patch("gladiator_env.rcon.time.sleep"):
            client = RconClient(timeout=0.5)
            with self.assertRaises(RconError) as ctx:
                client.connect()

        self.assertEqual(connect.call_count, 3)
        self.assertIn("对端关闭", str(ctx.exception))
        self.assertFalse(client.is_connected())

    def test_connect_calls_create_connection_once_when_healthy(self):
        with mock.patch("gladiator_env.rcon.socket.create_connection",
                        return_value=FakeSocket([auth_packet()])) as connect:
            RconClient(timeout=0.5).connect()
        self.assertEqual(connect.call_count, 1)


class CommandReconnectTest(unittest.TestCase):
    def test_command_reconnects_and_retries_once(self):
        class FlakyClient(RconClient):
            def __init__(self) -> None:
                super().__init__(timeout=0.5)
                self.calls = 0
                self.reconnects = 0

            def connect(self):  # 不真的建 socket
                self.reconnects += 1
                self._sock = object()
                return self

            def close(self) -> None:
                self._sock = None

            def _command_once(self, command: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RconError("RCON 连接被对端关闭")
                return "ok"

        client = FlakyClient()
        self.assertEqual(client.command("list"), "ok")
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.reconnects, 1)

    def test_command_failure_after_reconnect_is_raised(self):
        class DeadClient(RconClient):
            def connect(self):
                raise RconError("RCON 连接被对端关闭")

            def _command_once(self, command: str) -> str:
                raise RconError("RCON 连接被对端关闭")

        with self.assertRaises(RconError):
            DeadClient().command("list")


class EnvConnectionReuseTest(unittest.TestCase):
    """GladiatorEnv 整局只连一次：两次 reset 累计 connect 调用数 = 1。"""

    def build(self):
        instances: list["FakeRcon"] = []

        class FakeRcon:
            def __init__(self, **_kwargs) -> None:
                self.connected = False
                self.connect_calls = 0
                self.close_calls = 0
                self.commands: list[str] = []
                instances.append(self)

            def connect(self):
                self.connect_calls += 1
                self.connected = True
                return self

            def is_connected(self) -> bool:
                return self.connected

            def close(self) -> None:
                self.close_calls += 1
                self.connected = False

            def command(self, text: str) -> str:
                self.commands.append(text)
                return ""

        links = [FakeLink("Bot_T1"), FakeLink("Bot_T2")]
        for index, link in enumerate(links):
            other = "Bot_T2" if index == 0 else "Bot_T1"
            link.set_observation(make_observation(
                link.username, x=-1.5 if index == 0 else 1.5, z=0.0,
                targets=[make_target(2 if index == 0 else 1, other,
                                     x=1.5 if index == 0 else -1.5, z=0.0, distance=3.0)]))
        with mock.patch("gladiator_env.env.RconClient", FakeRcon):
            env = GladiatorEnv(
                [BotConfig("Bot_T1", "ws://127.0.0.1:8766"),
                 BotConfig("Bot_T2", "ws://127.0.0.1:8767")],
                ArenaConfig(center=(0.0, 4.0, 0.0), spacing=6.0, prep_seconds=0.0,
                            give_kit=False),
                rcon={"host": "127.0.0.1", "port": 25575, "password": "x"},
                links=links, connect=True, control_hz=200.0, action_ticks=1,
                single_agent=True,
            )
        return env, instances

    def test_reset_reuses_the_single_connection(self):
        env, instances = self.build()
        # 跳过摆位（FakeLink 不会真的走到 tp 点），这里只验连接生命周期。
        env.reset(options={"prepare_arena": False})
        env.reset(options={"prepare_arena": False})
        client = instances[0]
        self.assertEqual(len(instances), 1, "整局只应创建一个 RconClient")
        self.assertEqual(client.connect_calls, 1, "两次 reset 只该连一次")
        self.assertEqual(client.close_calls, 0, "reset 不应关闭连接")
        env.close()
        self.assertEqual(client.close_calls, 1, "close() 才真正断开")

    def test_reset_reconnects_only_when_dropped(self):
        env, instances = self.build()
        client = instances[0]
        env.reset(options={"prepare_arena": False})
        client.connected = False  # 模拟服务端把连接关了
        env.reset(options={"prepare_arena": False})
        self.assertEqual(client.connect_calls, 2)
        env.close()


if __name__ == "__main__":
    unittest.main()
