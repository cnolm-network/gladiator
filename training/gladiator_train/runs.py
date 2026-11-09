"""一次训练运行（run）的产物目录。

每次启动训练都新建一个带时间戳与模式的目录，避免 `episode-001` 之类的文件名
在多次运行之间互相覆盖::

    training/artifacts/runs/20261107-2320-multi-Bot_C/
        run.log
        report.json
        policy.npz            # train
        policy-pair1.npz ...  # multi
        reviews/
            episode-001.md
            episode-001.json
            pair-1/...

目录名里的时间取自启动时刻（本地时区），排序即时间序。
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

#: `gladiator_train` 的上两级就是 training/。
TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_ROOT = os.path.join(TRAINING_ROOT, "artifacts")
RUNS_ROOT = os.path.join(ARTIFACTS_ROOT, "runs")

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str, fallback: str = "run") -> str:
    """把用户名/模式等清洗成可做目录名的片段。"""
    text = _SLUG.sub("-", (value or "").strip()).strip("-._")
    return text[:48] or fallback


def new_run_dir(kind: str, tag: str, when: Optional[float] = None) -> str:
    """创建并返回 `artifacts/runs/<YYYYMMDD-HHMM>-<kind>-<tag>`。

    同一分钟内重复启动时追加 `-2`、`-3`，保证目录唯一（不会覆盖上一次的数据）。
    """
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(when if when is not None else time.time()))
    base = os.path.join(RUNS_ROOT, f"{stamp}-{slugify(kind, 'train')}-{slugify(tag, 'run')}")
    candidate = base
    suffix = 2
    while os.path.exists(candidate):
        candidate = f"{base}-{suffix}"
        suffix += 1
    os.makedirs(os.path.join(candidate, "reviews"), exist_ok=True)
    return candidate


def run_paths(run_dir: str, kind: str, pairs: int = 1) -> dict:
    """返回一次运行的标准产物路径。"""
    return {
        "runDir": run_dir,
        "report": os.path.join(run_dir, "report.json"),
        "log": os.path.join(run_dir, "run.log"),
        "reviews": os.path.join(run_dir, "reviews"),
        "policy": os.path.join(run_dir, "policy.npz"),
        "policies": [os.path.join(run_dir, f"policy-pair{index + 1}.npz") for index in range(max(1, pairs))],
    }
