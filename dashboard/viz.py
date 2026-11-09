"""模型可视化后端：``GET /api/models/viz?id=<model>`` 的全部数据。

设计要点（与 dashboard 其它模块一致，**不依赖 numpy**，纯标准库）：

- ``.npz`` 是 zip，里面每个成员是 ``.npy``；复用 ``models`` 的 ``_npy_header`` 解析
  header，再手动把 float 字节解出来（见 ``_read_npy_array`` / ``_read_npz_params``）。
- 数据来源：
    - 基本信息卡：``ModelStore.get`` 返回的模型元数据 + ``training_stats``；
    - 权重直方图 / 范数 / 与最近备份的差异：解析 ``policy.npz``（及 ``models-trash`` 里的备份）；
    - 累计 reward 曲线 / 效果指标 / 对手分组 / 动作分布 / 累计统计表：汇总该模型
      ``training.json`` 里记录的各次 run，逐一把 ``artifacts/runs/<runId>/report.json``
      的 episode 历史拉起来聚合。
- 任何一段数据缺失时，对应段返回 ``available: false``，前端据此显示「暂无数据」。
"""

from __future__ import annotations

import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from errors import NotFound
from models import NPY_MAGIC, REQUIRED_ARRAYS, _npy_header, _sha256

# 11 个动作（与 gladiator_env 的 action space 一致，顺序也一致）。
ACTION_KEYS = [
    "idle", "approach", "attack", "strafe_left", "strafe_right", "retreat",
    "jump", "use_rod", "eat_gapple", "block", "switch_sword",
]


# --------------------------------------------------------------------------- #
# npy / npz 手动解析（无 numpy）
# --------------------------------------------------------------------------- #
def _parse_descr(descr: str) -> tuple[str, int]:
    """把 npy 的 ``descr``（如 ``'<f8'`` / ``'|f4'`` / ``'f8'``）解析成 (struct fmt, 字节数)。"""
    match = re.match(r"^([<>=\|]?)([a-zA-Z])(\d*)$", (descr or "").strip())
    if not match:
        raise ValueError(f"无法解析 dtype: {descr!r}")
    byteorder, kind, size = match.group(1), match.group(2), match.group(3)
    if kind != "f":
        raise ValueError(f"仅支持浮点权重，得到 dtype: {descr!r}")
    nbytes = int(size) if size else 8
    little = byteorder in ("<", "=", "|", "")
    fmt = ("<" if little else ">") + ("d" if nbytes == 8 else "f")
    return fmt, nbytes


def _read_npy_array(blob: bytes) -> list[float]:
    """从单个 ``.npy`` 的字节里解出全部 float，返回扁平 list。"""
    if not blob.startswith(NPY_MAGIC):
        raise ValueError("不是合法的 npy 数据")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", blob[8:12])[0]
        start = 12
    else:
        raise ValueError(f"不支持的 npy 版本: {major}")
    shape, descr = _npy_header(blob)
    fmt, nbytes = _parse_descr(descr)
    # fmt 形如 "<d" / ">f"：字节序前缀 + 类型字符；struct 不接受重复前缀，
    # 必须是「前缀 + 类型字符 × count」。
    prefix = fmt[0] if fmt[0] in "<>=!" else ""
    type_char = fmt[-1]
    data = blob[start + header_len:]
    count = 1
    for dim in shape:
        count *= int(dim)
    if len(data) < count * nbytes:
        raise ValueError("npy 数据长度不足")
    return list(struct.unpack(f"{prefix}{type_char * count}", data[: count * nbytes]))


def _read_npz_params(path: Path) -> dict[str, list[float]]:
    """读 ``policy.npz`` 的 W1/b1/W2/b2，返回 key -> 扁平 float list。"""
    out: dict[str, list[float]] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for key in REQUIRED_ARRAYS:
            member = f"{key}.npy"
            if member not in names:
                raise ValueError(f"模型缺少参数数组 {key}")
            with archive.open(member) as handle:
                out[key] = _read_npy_array(handle.read())
    return out


def _flatten(params: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for key in REQUIRED_ARRAYS:
        values.extend(params.get(key, []))
    return values


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


# --------------------------------------------------------------------------- #
# 基本信息
# --------------------------------------------------------------------------- #
def _basic_info(model: dict, training: dict) -> dict:
    training = training or {}
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "description": model.get("description") or "",
        "builtin": bool(model.get("builtin")),
        "kind": model.get("kind"),
        "obsDim": model.get("obsDim"),
        "numActions": model.get("numActions"),
        "hidden": model.get("hidden"),
        "episodes": model.get("episodes"),
        "steps": model.get("steps"),
        "meanReward": model.get("meanReward"),
        "sha256": model.get("sha256"),
        "updatedAt": model.get("updatedAt"),
        "source": model.get("source") or {},
        "training": {
            "lastTrainedAt": training.get("lastTrainedAt"),
            "totalEpisodes": training.get("totalEpisodes"),
            "totalSteps": training.get("totalSteps"),
            "writeCount": training.get("writeCount"),
            "lastMeanReward": training.get("lastMeanReward"),
        },
    }


# --------------------------------------------------------------------------- #
# 权重：直方图 / 范数
# --------------------------------------------------------------------------- #
def _weights_section(models, model_id: str) -> dict:
    policy = models.policy_path(model_id)
    if policy is None or not Path(policy).is_file():
        return {"available": False, "note": "该模型没有 policy.npz（例如内置 default 尚未训练写回）"}
    try:
        params = _read_npz_params(Path(policy))
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        return {"available": False, "note": f"权重解析失败：{exc}"}

    norms = {key: _l2_norm(params.get(key, [])) for key in REQUIRED_ARRAYS}
    flat = _flatten(params)
    hist = _histogram(flat, bins=24)
    return {
        "available": True,
        "obsDim": models.get(model_id).get("obsDim") if models.get(model_id) else None,
        "numActions": models.get(model_id).get("numActions") if models.get(model_id) else None,
        "hidden": models.get(model_id).get("hidden") if models.get(model_id) else None,
        "sizeBytes": Path(policy).stat().st_size,
        "sha256": _sha256(Path(policy)),
        "norms": norms,
        "totalNorm": _l2_norm(flat),
        "count": len(flat),
        "histogram": hist,
    }


def _histogram(values: list[float], bins: int = 24) -> dict:
    if not values:
        return {"available": False, "min": None, "max": None, "binEdges": [], "counts": [], "total": 0}
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        hi = lo + 1.0  # 避免除零（所有权重相同的情况）
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    edges = [lo + i * width for i in range(bins + 1)]
    return {
        "available": True,
        "min": lo,
        "max": hi,
        "binEdges": edges,
        "counts": counts,
        "total": len(values),
    }


# --------------------------------------------------------------------------- #
# 与最近备份的权重差异
# --------------------------------------------------------------------------- #
def _backup_diff(models, model_id: str) -> dict:
    policy = models.policy_path(model_id)
    if policy is None or not Path(policy).is_file():
        return {"available": False, "note": "当前模型没有 policy.npz，无可比较的备份"}
    trash = getattr(models, "trash", None)
    if trash is None or not Path(trash).is_dir():
        return {"available": False, "note": "未找到备份目录（models-trash）"}
    pattern = re.compile(rf"^{re.escape(model_id)}-\d{{8}}-\d{{6}}(?:-\d+)?\.npz$")
    candidates = [p for p in Path(trash).iterdir()
                  if p.is_file() and pattern.match(p.name)]
    if not candidates:
        return {"available": False, "note": "没有历史备份（首次写回后才会产生备份）"}
    # 文件名里的时间戳是 ISO 排序的，取最新的一份。
    backup = sorted(candidates)[-1]
    try:
        cur = _read_npz_params(Path(policy))
        bak = _read_npz_params(backup)
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        return {"available": False, "note": f"备份或当前权重解析失败：{exc}"}
    a = _flatten(cur)
    b = _flatten(bak)
    if len(a) != len(b):
        return {"available": False, "note": "备份与当前权重维度不一致，无法比较"}
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    rel = (_l2_norm([x - y for x, y in zip(a, b)]) / norm_b) if norm_b else None
    cos = (dot / (norm_a * norm_b)) if (norm_a and norm_b) else None
    return {
        "available": True,
        "backupPath": str(backup),
        "backupSha256": _sha256(backup),
        "l2RelativeChange": rel,
        "cosineSimilarity": cos,
    }


# --------------------------------------------------------------------------- #
# 训练 run 聚合：曲线 / 指标 / 对手分组 / 动作分布 / 累计统计表
# --------------------------------------------------------------------------- #
def _load_run_history(artifacts_dir: Path, run_id: str, at: str | None) -> list[dict]:
    """读一次 run 的所有 episode，统一成扁平结构（train / multi 都覆盖）。"""
    report_path = artifacts_dir / "runs" / run_id / "report.json"
    try:
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(report, dict):
        return []
    kind = report.get("kind")
    episodes: list[dict] = []

    def collect(history, learner):
        for idx, entry in enumerate(history or [], start=1):
            if not isinstance(entry, dict):
                continue
            review = entry.get("review") or {}
            if not isinstance(review, dict):
                review = {}
            winner = entry.get("winner")
            action_counts = review.get("actionCounts") or {}
            episodes.append({
                "runId": run_id,
                "at": at or "",
                "index": idx,
                "opponentMode": entry.get("opponentMode") or review.get("opponentMode") or "unknown",
                "totalReward": float(entry.get("totalReward") or 0),
                "winner": winner,
                "learner": learner,
                "steps": int(entry.get("steps") or 0),
                "duration": float(review.get("durationSeconds") or entry.get("durationSeconds") or 0),
                "hits": int(review.get("hits") or 0),
                "whiffs": int(review.get("whiffs") or 0),
                "hitRate": review.get("hitRate"),
                "damageDealt": float(review.get("damageDealt") or 0),
                "damageTaken": float(review.get("damageTaken") or 0),
                "damageRatio": review.get("damageRatio"),
                "actionCounts": action_counts if isinstance(action_counts, dict) else {},
            })

    if kind == "multi":
        for pair in report.get("pairs") or []:
            learner = (pair.get("users") or [None, None])[0]
            collect(pair.get("history"), learner)
    else:
        users = (report.get("config") or {}).get("users") or [None, None]
        collect(report.get("history"), users[0])
    return episodes


def _runs_section(models, artifacts_dir: Path, model_id: str) -> dict:
    training = models.training_stats(model_id)
    runs = training.get("runs") or []
    # 只保留有 runId 的，并去重（保留最近一份）。
    seen = set()
    run_ids = []
    for run in runs:
        rid = run.get("runId")
        if rid and rid not in seen:
            seen.add(rid)
            run_ids.append(rid)

    if not run_ids:
        return {"available": False, "note": "该模型还没有被持久化训练写回的记录（training.json 为空）"}

    episodes: list[dict] = []
    per_run: dict[str, list[dict]] = {}
    for rid in run_ids:
        at = next((r.get("at") for r in runs if r.get("runId") == rid), "")
        items = _load_run_history(artifacts_dir, rid, at)
        per_run[rid] = items
        episodes.extend(items)

    if not episodes:
        return {"available": False,
                "note": "training.json 记录了 run，但对应报告里没有 episode 历史（可能 run 未正常完成）",
                "runCount": len(run_ids)}

    return {
        "available": True,
        "runCount": len(run_ids),
        "episodeCount": len(episodes),
        "byRun": _by_run_series(per_run),
        "cumulative": _cumulative(episodes),
        "metrics": _overall_metrics(episodes),
        "opponents": _opponent_groups(episodes),
        "actions": _action_distribution(episodes),
        "table": _stats_table(per_run, episodes),
    }


def _by_run_series(per_run: dict[str, list[dict]]) -> dict:
    series = []
    for run_id, items in per_run.items():
        points = [{"i": item["index"], "reward": item["totalReward"]} for item in items]
        series.append({"runId": run_id, "points": points})
    return {"labels": list(per_run.keys()), "series": series}


def _cumulative(episodes: list[dict]) -> dict:
    ordered = sorted(episodes, key=lambda e: (e["at"] or "", e["runId"] or "", e["index"]))
    running = 0.0
    points = []
    for i, item in enumerate(ordered, start=1):
        running += item["totalReward"]
        points.append({"i": i, "reward": item["totalReward"], "cumulative": running})
    return {"labels": [str(i) for i in range(1, len(points) + 1)], "points": points}


def _overall_metrics(episodes: list[dict]) -> dict:
    n = len(episodes)
    wins = sum(1 for e in episodes if e["winner"] is not None and e["winner"] == e["learner"])
    hits = sum(e["hits"] for e in episodes)
    whiffs = sum(e["whiffs"] for e in episodes)
    dealt = sum(e["damageDealt"] for e in episodes)
    taken = sum(e["damageTaken"] for e in episodes)
    durations = [e["duration"] for e in episodes if e["duration"] > 0]
    denom = hits + whiffs
    return {
        "available": n > 0,
        "episodeCount": n,
        "winRate": (wins / n) if n else None,
        "hitRate": (hits / denom) if denom else None,
        "whiffRate": (whiffs / denom) if denom else None,
        "damageRatio": (dealt / taken) if taken > 0 else None,
        "avgDuration": (sum(durations) / len(durations)) if durations else None,
    }


def _opponent_groups(episodes: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for e in episodes:
        groups.setdefault(e["opponentMode"], []).append(e)
    out = []
    for mode, items in sorted(groups.items()):
        n = len(items)
        wins = sum(1 for e in items if e["winner"] == e["learner"])
        hits = sum(e["hits"] for e in items)
        whiffs = sum(e["whiffs"] for e in items)
        dealt = sum(e["damageDealt"] for e in items)
        taken = sum(e["damageTaken"] for e in items)
        rewards = [e["totalReward"] for e in items]
        durations = [e["duration"] for e in items if e["duration"] > 0]
        denom = hits + whiffs
        out.append({
            "opponentMode": mode,
            "nEpisodes": n,
            "winRate": (wins / n) if n else None,
            "hitRate": (hits / denom) if denom else None,
            "damageRatio": (dealt / taken) if taken > 0 else None,
            "avgReward": (sum(rewards) / len(rewards)) if rewards else None,
            "avgDuration": (sum(durations) / len(durations)) if durations else None,
        })
    return out


def _action_distribution(episodes: list[dict]) -> list[dict]:
    totals: dict[str, int] = {key: 0 for key in ACTION_KEYS}
    for e in episodes:
        counts = e["actionCounts"] or {}
        for key, value in counts.items():
            if key in totals:
                totals[key] += int(value or 0)
            else:
                totals[key] = int(value or 0)
    grand = sum(totals.values()) or 1
    out = []
    for key in ACTION_KEYS:
        out.append({"action": key, "count": totals[key], "fraction": totals[key] / grand})
    return out


def _stats_table(per_run: dict[str, list[dict]], episodes: list[dict]) -> dict:
    rows = []
    for run_id, items in per_run.items():
        n = len(items)
        if n == 0:
            continue
        wins = sum(1 for e in items if e["winner"] == e["learner"])
        hits = sum(e["hits"] for e in items)
        whiffs = sum(e["whiffs"] for e in items)
        dealt = sum(e["damageDealt"] for e in items)
        taken = sum(e["damageTaken"] for e in items)
        denom = hits + whiffs
        rewards = [e["totalReward"] for e in items]
        durations = [e["duration"] for e in items if e["duration"] > 0]
        rows.append({
            "runId": run_id,
            "episodes": n,
            "steps": sum(e["steps"] for e in items),
            "meanReward": (sum(rewards) / len(rewards)) if rewards else None,
            "winRate": (wins / n) if n else None,
            "hitRate": (hits / denom) if denom else None,
            "damageRatio": (dealt / taken) if taken > 0 else None,
            "avgDuration": (sum(durations) / len(durations)) if durations else None,
        })
    n = len(episodes)
    wins = sum(1 for e in episodes if e["winner"] == e["learner"])
    hits = sum(e["hits"] for e in episodes)
    whiffs = sum(e["whiffs"] for e in episodes)
    dealt = sum(e["damageDealt"] for e in episodes)
    taken = sum(e["damageTaken"] for e in episodes)
    denom = hits + whiffs
    rewards = [e["totalReward"] for e in episodes]
    durations = [e["duration"] for e in episodes if e["duration"] > 0]
    totals = {
        "episodes": n,
        "steps": sum(e["steps"] for e in episodes),
        "meanReward": (sum(rewards) / len(rewards)) if rewards else None,
        "winRate": (wins / n) if n else None,
        "hitRate": (hits / denom) if denom else None,
        "damageRatio": (dealt / taken) if taken > 0 else None,
        "avgDuration": (sum(durations) / len(durations)) if durations else None,
    }
    return {"columns": ["run", "episodes", "steps", "meanReward", "winRate", "hitRate",
                        "damageRatio", "avgDuration"], "rows": rows, "totals": totals}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_viz(models, artifacts_dir: Path, model_id: str) -> dict:
    """构造 ``GET /api/models/viz?id=<model>`` 的完整响应。"""
    model = models.get(model_id)  # 不存在时抛出 NotFound，由路由层转成 404
    if model is None:
        raise NotFound(f"模型不存在: {model_id}")
    training = model.get("training") or {}
    return {
        "ok": True,
        "id": model_id,
        "model": _basic_info(model, training),
        "weights": _weights_section(models, model_id),
        "backupDiff": _backup_diff(models, model_id),
        "runs": _runs_section(models, Path(artifacts_dir), model_id),
    }
