"""通道 B（Layer 2 ↔ Layer 3）同步客户端。

训练环境是同步 Gymnasium 接口，而 Layer 2 的通道 B 是 WebSocket。这里用「后台事件循环 +
线程安全队列」把它包成同步对象：`start()` / `wait_welcome()` / `latest()` / `send_actions()`。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from websockets.asyncio.client import connect


class LinkError(RuntimeError):
    """通道 B 连接或协议错误。"""


class LinkClient:
    """单个 Bot 的通道 B 客户端。

    参数
    ----
    url:      `ws://127.0.0.1:8766` 之类的地址，由该 Bot 的 bot-runtime 进程监听。
    bot_id:   `glad-<小写用户名>`，必须与服务端 `BotRegistry.botIdFor` 一致。
    timeout:  连接与单次等待的默认超时（秒）。
    """

    def __init__(self, url: str, bot_id: str, timeout: float = 10.0) -> None:
        self.url = url
        self.bot_id = bot_id
        self.timeout = timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._closed = False
        self._error: BaseException | None = None
        self._last_error: str | None = None
        self._stopped = False

        self._seq = 0
        self._seq_lock = threading.Lock()
        self._welcome: dict | None = None
        self._observation: dict | None = None
        self._results: dict[int, dict] = {}
        self._idle_events: list[dict] = []
        self._cond = threading.Condition()

    # --- 生命周期 -------------------------------------------------------

    def start(self) -> "LinkClient":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"link-{self.bot_id}", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - 后台线程里的错误要转交主线程
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()
            self._loop.close()

    async def _main(self) -> None:
        """连接 + 自动重连：训练常常跑几十分钟，运行时重启/掉线后要能自己回来。"""
        attempt = 0
        while not self._stopped:
            try:
                async with connect(self.url, open_timeout=self.timeout, max_size=8 * 1024 * 1024,
                                   ping_interval=None) as ws:
                    self._ws = ws
                    await ws.send(json.dumps({
                        "type": "HELLO",
                        "payload": {"botId": self.bot_id, "v": 1, "protocol": "gladiator-env"},
                    }))
                    attempt = 0
                    async for raw in ws:
                        self._handle(raw)
            except BaseException as exc:  # noqa: BLE001 - 连接断开的异常走重连
                if self._stopped:
                    raise
                with self._cond:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._cond.notify_all()
            finally:
                self._ws = None
            if self._stopped:
                return
            attempt += 1
            delay = min(5.0, 0.5 * (2 ** min(attempt - 1, 4)))
            time.sleep(delay)

    def _handle(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LinkError(f"通道 B 收到非法 JSON: {raw!r}") from exc
        kind = message.get("type")
        payload = message.get("payload") or {}
        with self._cond:
            if kind == "WELCOME":
                self._welcome = payload
                self._cond.notify_all()
            elif kind == "OBSERVATION":
                self._observation = payload
                self._cond.notify_all()
            elif kind == "ACTION_RESULT":
                reply_to = message.get("replyTo")
                if isinstance(reply_to, int):
                    self._results[reply_to] = payload
                self._cond.notify_all()
            elif kind == "REJECT":
                self._error = LinkError(f"通道 B 拒绝: {payload}")
                self._cond.notify_all()
            elif kind == "IDLE":
                # G-405：空闲信号只做记录，训练循环不会因此发起任何外部查询。
                self._idle_events.append(payload)
                del self._idle_events[:-32]
            # PONG / 其他帧当前不参与训练循环

    def stop(self) -> None:
        self._stopped = True
        loop = self._loop
        ws = self._ws
        if loop is not None and ws is not None and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(ws.close(), loop)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # --- 等待 -----------------------------------------------------------

    def wait_welcome(self, timeout: float | None = None) -> dict:
        deadline = time.monotonic() + (timeout or self.timeout)
        with self._cond:
            while self._welcome is None:
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LinkError(f"{self.bot_id}: 等待 WELCOME 超时（{self.url}）")
                self._cond.wait(min(remaining, 0.25))
            return self._welcome

    def wait_observation(self, timeout: float | None = None, after: dict | None = None) -> dict:
        """等到一条新的 OBSERVATION（或现有观测已不同于 ``after``）。"""
        deadline = time.monotonic() + (timeout or self.timeout)
        with self._cond:
            while True:
                self._raise_if_failed()
                observation = self._observation
                if observation is not None and observation is not after:
                    # 优先等「下一帧」：tick 与上一帧不同才算新鲜观测。
                    if isinstance(after, dict) and "tick" in after:
                        if observation.get("tick") != after.get("tick"):
                            return observation
                    else:
                        return observation
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LinkError(f"{self.bot_id}: 等待 OBSERVATION 超时")
                self._cond.wait(min(remaining, 0.05))

    def latest(self) -> dict | None:
        with self._cond:
            return self._observation

    def idle_events(self) -> list[dict]:
        """取走当前累积的 IDLE 通知（G-405：仅记录，不触发查询）。"""
        with self._cond:
            events = list(self._idle_events)
            self._idle_events.clear()
            return events

    def control_hz(self) -> float:
        with self._cond:
            welcome = self._welcome or {}
        try:
            return float(welcome.get("controlHz") or 10.0)
        except (TypeError, ValueError):
            return 10.0

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise LinkError(str(error)) from error

    # --- 发送 -----------------------------------------------------------

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    async def _send_action(self, seq: int, action: dict) -> None:
        ws = self._ws
        if ws is None:
            raise LinkError(f"{self.bot_id}: 通道 B 尚未连接")
        await ws.send(json.dumps({
            "type": "ACTION",
            "seq": seq,
            "payload": {"type": action["type"], "params": action.get("params") or {}},
        }))

    def send_actions(self, actions: list[dict], timeout: float = 2.0) -> list[dict]:
        """按顺序发送一组协议动作，返回对应的 ACTION_RESULT（可能为空）。"""
        loop = self._loop
        if loop is None or loop.is_closed() or self._ws is None:
            raise LinkError(f"{self.bot_id}: 通道 B 尚未连接"
                            + (f"（{self._last_error}）" if getattr(self, "_last_error", None) else ""))
        results = []
        for action in actions:
            seq = self._next_seq()
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_action(seq, action), loop)
            except RuntimeError as exc:
                raise LinkError(f"{self.bot_id}: 通道 B 事件循环已关闭: {exc}") from exc
            try:
                future.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                raise LinkError(f"{self.bot_id}: 发送动作失败: {exc}") from exc
            results.append(self._collect_result(seq, timeout))
        return results

    def send_actions_nowait(self, actions: list[dict], timeout: float = 2.0) -> list[int]:
        """只发送、不等结果（训练控制循环用：动作在运行时异步执行）。

        返回已分配的 seq 列表；结果通过 :meth:`poll_results` 异步取回，
        这样 ``step()`` 不会被每个动作的执行耗时串行拖慢。
        """
        loop = self._loop
        if loop is None or loop.is_closed() or self._ws is None:
            raise LinkError(f"{self.bot_id}: 通道 B 尚未连接"
                            + (f"（{self._last_error}）" if getattr(self, "_last_error", None) else ""))
        seqs: list[int] = []
        for action in actions:
            seq = self._next_seq()
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_action(seq, action), loop)
            except RuntimeError as exc:
                raise LinkError(f"{self.bot_id}: 通道 B 事件循环已关闭: {exc}") from exc
            try:
                future.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                raise LinkError(f"{self.bot_id}: 发送动作失败: {exc}") from exc
            seqs.append(seq)
        return seqs

    def poll_results(self) -> list[dict]:
        """取走已到达、尚未消费的 ACTION_RESULT。"""
        with self._cond:
            pending = list(self._results.values())
            self._results.clear()
            return pending

    def _collect_result(self, seq: int, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        with self._cond:
            while seq not in self._results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {}
                self._cond.wait(min(remaining, 0.05))
            return self._results.pop(seq)
