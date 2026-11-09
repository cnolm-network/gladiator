"""持久化训练的模型写锁（G-624）。

同一个模型同时只允许一个「持久化训练」持有写锁，避免两个训练同时写回
``var/models/<id>/policy.npz`` 造成覆盖/计数错乱。

锁文件：``dashboard/var/models/<id>/.lock``，内容::

    {"pid": 1234, "runId": "20261109-0010-train-Bot_A", "at": "...", "modelId": "default"}

- 锁的 pid 是 **Dashboard 进程**（写回由 Dashboard 负责，不是训练进程）；
- 启动时发现锁但持锁进程已不存在 → 回收并给一条中文 warning，进 job.notes；
- 训练进入任何终态（finished / failed / stopped / 启动前失败）都释放锁，且只释放自己 runId 的锁。

进程存活检查用 ``ctypes`` 的 ``OpenProcess``（Windows）或 ``/proc``（POSIX）；
测试可注入 ``pid_check``，不依赖本机真实进程。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

LOCK_NAME = ".lock"


def default_pid_alive(pid: Any) -> bool:
    """进程是否仍存在；Windows 上不能用 ``os.kill(pid, 0)``（会 TerminateProcess）。"""
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - 平台相关
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, value)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        except Exception:  # noqa: BLE001 - 检查失败按“不存活”处理
            return False
    return Path(f"/proc/{value}").exists()


class ModelLock:
    def __init__(self, root: Path | str, *,
                 pid_check: Callable[[Any], bool] | None = None) -> None:
        self.root = Path(root)
        self._pid_alive = pid_check or default_pid_alive
        self._held: dict[str, dict] = {}

    def path_for(self, model_id: str) -> Path:
        return self.root / str(model_id) / LOCK_NAME

    def peek(self, model_id: str) -> dict | None:
        try:
            data = json.loads(self.path_for(model_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def acquire(self, model_id: str, run_id: str) -> dict:
        """尝试持锁；返回 ``{ok, warning, error, recovered}``。"""
        ident = str(model_id)
        target = self.path_for(ident)
        existing = self.peek(ident)
        recovered = False
        warning: str | None = None
        if existing is not None:
            if (str(existing.get("runId")) == str(run_id)
                    and int(existing.get("pid") or 0) == os.getpid()):
                return {"ok": True, "warning": None, "error": None, "recovered": False}
            if self._pid_alive(existing.get("pid")):
                return {
                    "ok": False,
                    "warning": None,
                    "error": (f"模型 {ident} 正在被 run {existing.get('runId')} 的持久化训练占用"
                              f"（pid={existing.get('pid')}）；请等它结束、或换一个模型 / 关闭持久化训练"),
                    "recovered": False,
                }
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                return {"ok": False, "warning": None,
                        "error": f"无法回收过期模型锁 {target}: {exc}", "recovered": False}
            recovered = True
            warning = (f"发现并回收了模型 {ident} 的过期锁"
                       f"（原 run {existing.get('runId')}, pid={existing.get('pid')} 已不存在）")
        payload = {
            "pid": os.getpid(),
            "runId": str(run_id),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "modelId": ident,
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(LOCK_NAME + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            return {"ok": False, "warning": warning,
                    "error": f"创建模型锁失败: {exc}", "recovered": recovered}
        self._held[ident] = payload
        return {"ok": True, "warning": warning, "error": None, "recovered": recovered}

    def release(self, model_id: str, run_id: str) -> bool:
        """只释放本 runId 持有的锁；返回是否真的删掉了文件。"""
        ident = str(model_id)
        self._held.pop(ident, None)
        existing = self.peek(ident)
        target = self.path_for(ident)
        if existing is None:
            return False
        if str(existing.get("runId")) != str(run_id):
            return False
        try:
            target.unlink(missing_ok=True)
            return True
        except OSError:
            return False
