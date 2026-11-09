"""只读实时日志（SSE）。

三种来源：

* ``server``   —— ``server/console.out.log``（最后 200 行 + 跟随新行）
* ``training`` —— 当前/最近一次训练的 ``runs/<id>/run.log``；没有该文件时退化为
                  Dashboard 捕获的 stdout（``var/logs/training.log``）
* ``runtime``  —— 指定 bot-runtime 的 stdout 日志；外部进程拿不到 stdout 时退化为
                  该运行时的状态与最近事件

所有路径都由服务端枚举/白名单确定，客户端只能传 source 与已存在的 name，
不存在任意文件读取。只读，不执行任何命令。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from artifacts import ArtifactStore, RUN_ID_RE
from config import DashboardConfig
from errors import BadRequest, NotFound
from processes import ProcessManager
from runtime_manager import RuntimeManager
from server_manager import ServerManager
from training_runner import TrainingRunner

MAX_INITIAL_LINES = 200
TAIL_BYTES = 256 * 1024
MAX_LINE_CHARS = 4000
FOLLOW_INTERVAL = 0.5
HEARTBEAT_EVERY = 10.0
FALLBACK_REFRESH = 3.0


@dataclass
class TerminalSource:
    source: str
    label: str
    path: Path | None = None
    initial_lines: list[str] = field(default_factory=list)
    fallback: bool = False
    note: str = ""
    offset: int = 0
    meta: dict = field(default_factory=dict)


def _read_tail(path: Path, max_lines: int = MAX_INITIAL_LINES) -> tuple[list[str], int]:
    """返回最后若干行和读取后的字节偏移；文件不存在时返回空。"""
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    start = max(0, size - TAIL_BYTES)
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read()
            offset = fh.tell()
    except OSError:
        return [], 0
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # 丢弃可能被截断的首行
    return [line[:MAX_LINE_CHARS] for line in lines[-max_lines:]], offset


def _sse(data: str, event: str | None = None) -> str:
    payload = data.replace("\r", " ").replace("\n", " ")
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


class TerminalService:
    def __init__(self, cfg: DashboardConfig, server: ServerManager, runtimes: RuntimeManager,
                 training: TrainingRunner, artifacts: ArtifactStore, procs: ProcessManager) -> None:
        self.cfg = cfg
        self.server = server
        self.runtimes = runtimes
        self.training = training
        self.artifacts = artifacts
        self.procs = procs

    # --- 解析 -----------------------------------------------------------
    def _training_source(self, name: str) -> TerminalSource:
        current = self.training.current()
        job = current.get("job") or {}
        run_id = (name or "").strip()
        if not run_id:
            run_id = str(job.get("runId") or "")
        if run_id and (not RUN_ID_RE.match(run_id) or ".." in run_id):
            raise BadRequest("training 的 name 必须是合法的 run id")
        if not run_id:
            runs = self.artifacts.runs()
            if runs:
                run_id = str(runs[0]["id"])
        if run_id and run_id != "legacy":
            path = self.artifacts.runs_dir / run_id / "run.log"
            if path.is_file() and self.artifacts.runs_dir.resolve() in path.resolve().parents:
                lines, offset = _read_tail(path)
                return TerminalSource("training", f"训练 {run_id}", path, lines,
                                      note="跟随 run.log", offset=offset)
        # 指定了历史 run 且没有 run.log：明确回 noLog，不要退化成当前任务的 stdout（会串场）。
        if run_id and run_id != "legacy" and not (job and run_id == job.get("runId")):
            return TerminalSource("training", f"训练 {run_id}", None, [],
                                  fallback=True,
                                  note="本次运行没有日志（历史数据或启动即失败）",
                                  meta={"runId": run_id, "logState": "no-log", "noLog": True})
        # 训练脚本暂未写 run.log，或刚启动还没 flush：退回 Dashboard 捕获的 stdout。
        if job and (not run_id or run_id == job.get("runId")):
            proc = self.procs.get("training")
            if proc is not None and proc.log_path:
                lines, offset = _read_tail(proc.log_path)
                if not lines:
                    lines = list(job.get("logTail") or [])
                return TerminalSource("training", f"训练 {job.get('runId') or '当前任务'}",
                                      proc.log_path, lines, note="run.log 尚未生成，显示 Dashboard 捕获的 stdout",
                                      offset=offset)
            if job.get("logTail"):
                return TerminalSource("training", "训练（内存日志）", None,
                                      list(job.get("logTail") or []),
                                      fallback=True,
                                      note="暂无输出流：显示 Dashboard 内存中的最近日志")
        return TerminalSource("training", "训练", None, [],
                              fallback=True, note="暂无输出流：还没有训练任务或 run.log")

    def _runtime_source(self, name: str) -> TerminalSource:
        raw = (name or "").strip()
        if not raw:
            raise BadRequest("source=runtime 时 name 不能为空")
        items = self.runtimes.list()
        target = None
        for item in items:
            if raw in (item.get("id"), item.get("username")):
                target = item
                break
        if target is None:
            # 允许传端口，方便外部运行时
            for item in items:
                if str(item.get("envPort")) == raw:
                    target = item
                    break
        if target is None:
            raise NotFound(f"未找到运行时: {raw}")
        label = f"运行时 {target.get('username') or target.get('id')}"
        if target.get("external"):
            return TerminalSource("runtime", label, None, self._runtime_fallback(target),
                                  fallback=True, note="暂无输出流：外部运行时无法读取 stdout，显示状态与最近事件",
                                  meta={"runtimeId": target.get("id")})
        proc = self.procs.get(str(target.get("id")))
        if proc is not None and proc.log_path and proc.log_path.is_file():
            lines, offset = _read_tail(proc.log_path)
            return TerminalSource("runtime", label, proc.log_path, lines, note="跟随 bot-runtime stdout",
                                  offset=offset, meta={"runtimeId": target.get("id")})
        return TerminalSource("runtime", label, None, self._runtime_fallback(target),
                              fallback=True, note="暂无输出流：该运行时没有可读日志",
                              meta={"runtimeId": target.get("id")})

    def _runtime_fallback(self, target: dict) -> list[str]:
        lines = [
            f"[dashboard] {target.get('username') or target.get('id')} · state={target.get('state')} "
            f"· envPort={target.get('envPort')} · pid={target.get('pid')}",
        ]
        if target.get("error"):
            lines.append(f"[error] {target.get('error')}")
        tail = target.get("tail") or []
        if tail:
            lines.append("[recent] 最近事件：")
            lines.extend(str(line) for line in tail[-20:])
        return lines

    def resolve(self, source: str, name: str = "") -> TerminalSource:
        source = (source or "").strip().lower()
        if source == "server":
            path = self.cfg.server_dir / "console.out.log"
            lines, offset = _read_tail(path)
            note = "跟随 server/console.out.log" if path.is_file() else "服务端日志尚未生成"
            return TerminalSource("server", "Paper 服务端", path, lines, note=note, offset=offset)
        if source == "training":
            return self._training_source(name)
        if source == "runtime":
            return self._runtime_source(name)
        raise BadRequest("source 必须是 server / training / runtime")

    # --- SSE ------------------------------------------------------------
    def stream(self, spec: TerminalSource) -> Iterator[str]:
        """生成 SSE 文本块。调用方需先 resolve 以在发送 200 前校验参数。"""
        yield "retry: 2000\n\n"
        meta = {
            "source": spec.source,
            "label": spec.label,
            "fallback": spec.fallback,
            "note": spec.note,
            "path": str(spec.path) if spec.path else None,
            **(spec.meta or {}),
        }
        yield _sse(json.dumps(meta, ensure_ascii=False), "meta")
        for line in spec.initial_lines:
            yield _sse(line)
        if spec.path is None:
            yield from self._fallback_stream(spec)
            return
        yield from self._follow(spec.path, spec.offset)

    def _follow(self, path: Path, offset: int = 0) -> Iterator[str]:
        buffer = b""
        last_beat = time.monotonic()
        while True:
            try:
                size = path.stat().st_size
            except OSError:
                yield _sse("[dashboard] 日志文件已不可读，等待恢复…")
                time.sleep(FOLLOW_INTERVAL)
                continue
            if size < offset:
                offset = 0
                buffer = b""
                yield _sse("[dashboard] 日志已轮转/截断，从头继续")
            if size > offset:
                try:
                    with path.open("rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        offset = fh.tell()
                except OSError:
                    time.sleep(FOLLOW_INTERVAL)
                    continue
                buffer += chunk
                *complete, buffer = buffer.split(b"\n")
                for raw in complete:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r")[:MAX_LINE_CHARS]
                    yield _sse(line)
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_EVERY:
                last_beat = now
                yield ": ping\n\n"
            time.sleep(FOLLOW_INTERVAL)

    def _fallback_stream(self, spec: TerminalSource) -> Iterator[str]:
        """没有可跟随文件时，周期性发送状态；训练结束后也不会突然 404。"""
        last_beat = 0.0
        while True:
            if spec.source == "runtime" and spec.meta.get("runtimeId"):
                current = self.runtimes.get(str(spec.meta["runtimeId"]))
                if current:
                    yield _sse(f"[status] state={current.get('state')} alive={current.get('alive')} "
                               f"pid={current.get('pid')} envPort={current.get('envPort')}")
            now = time.monotonic()
            if now - last_beat >= FALLBACK_REFRESH:
                last_beat = now
                yield ": ping\n\n"
            time.sleep(FOLLOW_INTERVAL)
