"""可追踪的子进程管理。

设计要点：
- 只管理 Dashboard 自己 `Popen` 出来的进程，不接管外部启动的 java/node；
- 每个进程有稳定 id、PID、启动时间、完整命令行、日志文件与内存环形日志；
- 停止时优先优雅（Paper 走控制台 ``stop``），超时再 terminate / kill；
- `var/managed.json` 落盘一份审计记录（重启 Dashboard 不自动接管旧进程，见 README）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Optional

from errors import Conflict, NotFound

_LOG_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: 单个进程日志超过这个大小就归档（默认 5MB）。
LOG_ROTATE_BYTES = 5 * 1024 * 1024
#: 归档保留份数（含当前文件之外的）。
LOG_KEEP = 5

# Windows: 不弹控制台窗口；独立进程组，避免 Ctrl+C 传播。
_CREATE_FLAGS = 0
if os.name == "nt":  # pragma: no cover - 平台相关
    _CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def should_rotate_log(path: Path, *, now: float | None = None,
                      max_bytes: int = LOG_ROTATE_BYTES) -> bool:
    """判定是否需要在启动新进程前归档：超过 ``max_bytes`` 或跨天。

    纯函数（时间可注入），测试不依赖真实时钟。
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return False
    if stat.st_size > max_bytes:
        return True
    moment = time.time() if now is None else now
    try:
        return time.strftime("%Y%m%d", time.localtime(stat.st_mtime)) != time.strftime(
            "%Y%m%d", time.localtime(moment)
        )
    except (OverflowError, OSError):  # pragma: no cover - 极端时间戳
        return False


def rotate_log(path: Path, *, now: float | None = None, keep: int = LOG_KEEP,
               max_bytes: int = LOG_ROTATE_BYTES) -> Path | None:
    """把超限/跨天的日志改名为 ``<stem>-YYYYmmdd-HHMMSS.log`` 并只保留最近 ``keep`` 份。

    返回归档文件路径；不需要轮转时返回 ``None``。轮转只发生在启动新进程之前，
    因此**当前 run 的增量已经写进新文件**（从 0 开始），不会丢失。
    """
    target = Path(path)
    if not should_rotate_log(target, now=now, max_bytes=max_bytes):
        return None
    moment = time.time() if now is None else now
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(moment))
    archived = target.with_name(f"{target.stem}-{stamp}{target.suffix}")
    index = 2
    while archived.exists():
        archived = target.with_name(f"{target.stem}-{stamp}-{index}{target.suffix}")
        index += 1
    try:
        os.replace(target, archived)
    except OSError:
        return None
    _prune_archives(target, keep=keep)
    return archived


def _prune_archives(path: Path, *, keep: int) -> None:
    """只删本进程 id 的归档（``<stem>-<时间戳>.log``），保留最近 ``keep`` 份。"""
    pattern = f"{Path(path).stem}-*.log"
    try:
        archives = [item for item in Path(path).parent.glob(pattern)
                    if item.name != Path(path).name]
        archives.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        return
    for stale in archives[max(0, keep):]:
        try:
            stale.unlink()
        except OSError:
            continue


@dataclass
class ManagedProcess:
    id: str
    kind: str
    label: str
    argv: list[str]
    cwd: str
    pid: int
    started_at: float
    log_path: Path
    popen: subprocess.Popen
    tail: Deque[str] = field(default_factory=lambda: deque(maxlen=800))
    exit_code: Optional[int] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    on_line: Optional[Callable[[str], None]] = None
    on_exit: Optional[Callable[["ManagedProcess"], None]] = None
    #: 本次进程在 ``log_path`` 里的起始字节偏移（header 之前的位置）。
    #: 训练结束按它增量切片生成 run.log，不再全量读文件。
    log_offset: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def alive(self) -> bool:
        return self.popen.poll() is None

    def snapshot(self, tail_lines: int = 80) -> dict:
        with self.lock:
            lines = list(self.tail)[-max(1, tail_lines):]
        uptime = (time.time() if self.exit_code is None else (self.finished_at or time.time())) - self.started_at
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "pid": self.pid,
            "alive": self.alive,
            "exitCode": self.exit_code if self.exit_code is not None else self.popen.poll(),
            "startedAt": _iso(self.started_at),
            "startedAtEpoch": self.started_at,
            "uptimeSeconds": round(max(0.0, uptime), 1),
            "command": list(self.argv),
            "cwd": self.cwd,
            "logPath": str(self.log_path),
            "logOffset": self.log_offset,
            "error": self.error,
            "tail": lines,
        }


class ProcessManager:
    def __init__(self, logs_dir: Path, state_file: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.state_file = Path(state_file)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, ManagedProcess] = {}
        self._lock = threading.RLock()

    # --- 查询 -----------------------------------------------------------
    def get(self, proc_id: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self._procs.get(proc_id)

    def require(self, proc_id: str) -> ManagedProcess:
        proc = self.get(proc_id)
        if proc is None:
            raise NotFound(f"未找到进程: {proc_id}")
        return proc

    def list(self, kind: Optional[str] = None, tail_lines: int = 40) -> list[dict]:
        with self._lock:
            procs = list(self._procs.values())
        snapshots = [p.snapshot(tail_lines) for p in procs if kind is None or p.kind == kind]
        snapshots.sort(key=lambda item: item["startedAtEpoch"])
        return snapshots

    def alive_count(self, kind: Optional[str] = None) -> int:
        with self._lock:
            return sum(1 for p in self._procs.values() if p.alive and (kind is None or p.kind == kind))

    # --- 启动 -----------------------------------------------------------
    def spawn(
        self,
        *,
        proc_id: str,
        kind: str,
        label: str,
        argv: list[str],
        cwd: Path | str,
        env: Optional[dict[str, str]] = None,
        tail_lines: int = 800,
        on_line: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[ManagedProcess], None]] = None,
    ) -> ManagedProcess:
        cwd = str(cwd)
        with self._lock:
            existing = self._procs.get(proc_id)
            if existing is not None and existing.alive:
                raise Conflict(f"进程 {proc_id} 已在运行（PID {existing.pid}）")
            log_path = self.logs_dir / f"{_LOG_NAME_RE.sub('_', proc_id)}.log"
            # 启动前归档：超过 5MB 或跨天时改名，当前进程从空文件写起（增量不丢）。
            rotate_log(log_path)
            child_env = os.environ.copy()
            child_env.update({
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            })
            if env:
                child_env.update(env)

            log_file = open(log_path, "ab", buffering=0)
            log_offset = log_file.tell()
            started_at = time.time()
            log_file.write(
                f"\n=== {_iso(started_at)} 启动 {proc_id} ===\n$ {' '.join(argv)}\n".encode("utf-8")
            )
            try:
                popen = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=child_env,
                    creationflags=_CREATE_FLAGS,
                )
            except OSError as exc:
                log_file.write(f"启动失败: {exc}\n".encode("utf-8"))
                log_file.close()
                raise Conflict(f"无法启动 {label}: {exc}") from exc

            proc = ManagedProcess(
                id=proc_id,
                kind=kind,
                label=label,
                argv=list(argv),
                cwd=cwd,
                pid=popen.pid,
                started_at=started_at,
                log_path=log_path,
                popen=popen,
                tail=deque(maxlen=tail_lines),
                on_line=on_line,
                on_exit=on_exit,
                log_offset=log_offset,
                lock=threading.Lock(),
            )
            self._procs[proc_id] = proc
            self._persist()

        threading.Thread(target=self._pump, args=(proc, log_file), name=f"pump-{proc_id}", daemon=True).start()
        return proc

    def _pump(self, proc: ManagedProcess, log_file) -> None:
        stream = proc.popen.stdout
        try:
            if stream is not None:
                for raw in stream:
                    line = raw.rstrip("\r\n")
                    with proc.lock:
                        proc.tail.append(line)
                    try:
                        log_file.write((line + "\n").encode("utf-8"))
                    except OSError:
                        pass
                    if proc.on_line is not None:
                        try:
                            proc.on_line(line)
                        except Exception:  # 回调异常不能影响进程读取
                            pass
        finally:
            try:
                code = proc.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况下流未关闭
                code = proc.popen.poll()
            proc.exit_code = code
            proc.finished_at = time.time()
            log_file.write(f"=== 退出码 {code} ===\n".encode("utf-8"))
            try:
                log_file.close()
            except OSError:
                pass
            self._persist()
            if proc.on_exit is not None:
                try:
                    proc.on_exit(proc)
                except Exception:
                    pass

    # --- 停止 -----------------------------------------------------------
    def stop(
        self,
        proc_id: str,
        graceful_line: Optional[str] = None,
        timeout: float = 15.0,
        terminate_timeout: float = 2.0,
    ) -> dict:
        """停止进程。

        - 传 ``graceful_line``（Paper 的 ``stop``）时先等它自己退出；
        - 其余（node bot-runtime / 训练进程）先 ``terminate``，最多等 2 秒再 ``kill``，
          避免旧实现先 ``wait(8s)`` 导致网页上“停止 Bot”长时间转圈。
        """
        proc = self.require(proc_id)
        if not proc.alive:
            return proc.snapshot()
        graceful = bool(graceful_line and proc.popen.stdin is not None)
        if graceful:
            try:
                proc.popen.stdin.write(graceful_line or "")
                proc.popen.stdin.flush()
            except (OSError, ValueError):
                graceful = False
        try:
            if graceful:
                try:
                    proc.popen.wait(timeout=max(1.0, timeout))
                except subprocess.TimeoutExpired:
                    pass
            if proc.popen.poll() is None:
                try:
                    proc.popen.terminate()
                except OSError as exc:
                    proc.error = f"终止失败: {exc}"
                try:
                    proc.popen.wait(timeout=max(0.5, terminate_timeout))
                except subprocess.TimeoutExpired:
                    try:
                        proc.popen.kill()
                        proc.popen.wait(timeout=5)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        proc.error = "进程未能退出"
        except OSError as exc:
            proc.error = f"停止失败: {exc}"
        time.sleep(0.05)
        return proc.snapshot()

    def stop_all(self, kind: Optional[str] = None, timeout: float = 10.0) -> list[dict]:
        """并发停止多个进程；单个进程最多等 2 秒，避免串行累计。"""
        with self._lock:
            ids = [p.id for p in self._procs.values() if p.alive and (kind is None or p.kind == kind)]
        if not ids:
            return []
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
            futures = {pool.submit(self.stop, proc_id, None, timeout): proc_id for proc_id in ids}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:  # noqa: BLE001 - 单个失败不影响其它进程
                    continue
        return results

    # --- 审计落盘 -------------------------------------------------------
    def _persist(self) -> None:
        with self._lock:
            payload = [
                {
                    "id": p.id,
                    "kind": p.kind,
                    "label": p.label,
                    "pid": p.pid,
                    "alive": p.alive,
                    "exitCode": p.exit_code if p.exit_code is not None else p.popen.poll(),
                    "startedAt": _iso(p.started_at),
                    "command": list(p.argv),
                    "cwd": p.cwd,
                    "logPath": str(p.log_path),
                }
                for p in self._procs.values()
            ]
        tmp = self.state_file.with_suffix(".tmp")
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_file)
        except OSError:
            pass
