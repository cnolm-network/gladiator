"""训练产物读取：按运行目录（runs/）组织，同时兼容旧的扁平文件。

新结构（由训练脚本或 Dashboard 创建）::

    training/artifacts/runs/<YYYYMMDD-HHMM>-<kind>-<tag>/
        report.json          # train-report.json / multi-report.json 的同构内容
        policy*.npz
        reviews/episode-XXX.md|json
        run.log

旧结构仍然可读：``train-report.json``、``multi-report.json``、``reviews/``。

只读，不写 training/artifacts；所有文件读取失败都会降级为 ``{exists: true, error: ...}``，
避免一个坏文件把整个 Dashboard 打挂。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from errors import BadRequest, NotFound

MAX_REVIEW_FILES = 500
MAX_MARKDOWN_BYTES = 2 * 1024 * 1024
MAX_RUNS = 100
#: 运行目录名：<日期>-<时分>-<kind>-<tag>；tag 由 Dashboard 限制为字母数字下划线。
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
_RUN_PREFIX_RE = re.compile(r"^(\d{8})-(\d{4})-(train|multi)(?:-(.*))?$")
#: 没有 report.json 时，最近 5 分钟仍在写入的目录视为 running，否则 interrupted。
RUNNING_WINDOW_SECONDS = 300


def _run_kind(name: str, meta: Any) -> str:
    """优先 run.json 的 kind；缺失时从目录名推断（脚本自己建的 run 可能没有 run.json）。"""
    if isinstance(meta, dict) and meta.get("kind") in ("train", "multi"):
        return str(meta["kind"])
    match = _RUN_PREFIX_RE.match(name)
    if match:
        return match.group(3)
    return "multi" if "multi" in name else "train"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _finished_at(payload: dict) -> str | None:
    """报告里有 finishedAt 就用；否则用 startedAt + durationSeconds 推算。"""
    raw = payload.get("finishedAt")
    if raw:
        return str(raw)
    started = payload.get("startedAt")
    duration = payload.get("durationSeconds")
    if started and isinstance(duration, (int, float)):
        try:
            end = datetime.fromisoformat(str(started)) + timedelta(seconds=float(duration))
            return end.isoformat(timespec="seconds")
        except (TypeError, ValueError):
            return None
    return None


def _read_json(path: Path) -> dict | list | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _slim_history(history: Any) -> list[dict]:
    if not isinstance(history, list):
        return []
    out = []
    for item in history:
        if not isinstance(item, dict):
            continue
        review = item.get("review") if isinstance(item.get("review"), dict) else {}
        out.append({
            "episode": item.get("episode"),
            "steps": item.get("steps"),
            "totalReward": item.get("totalReward"),
            "startedAt": item.get("startedAt"),
            "startedAtEpoch": item.get("startedAtEpoch"),
            "finishedAt": item.get("finishedAt"),
            "durationSeconds": item.get("durationSeconds"),
            "winner": item.get("winner"),
            "terminated": item.get("terminated"),
            "truncated": item.get("truncated"),
            "meanHealth": item.get("meanHealth"),
            "opponentMode": item.get("opponentMode"),
            "policyUpdates": item.get("policyUpdates"),
            "hits": review.get("hits"),
            "whiffs": review.get("whiffs"),
            "hitRate": review.get("hitRate"),
            "maxCombo": review.get("maxCombo"),
            "damageDealt": review.get("damageDealt"),
            "damageTaken": review.get("damageTaken"),
        })
    return out


def _history_summary(history: list[dict]) -> dict:
    rewards = [float(item["totalReward"]) for item in history if isinstance(item.get("totalReward"), (int, float))]
    steps = sum(int(item.get("steps") or 0) for item in history)
    hits = sum(int(item.get("hits") or 0) for item in history)
    whiffs = sum(int(item.get("whiffs") or 0) for item in history)
    decided = sum(1 for item in history if item.get("winner"))
    return {
        "episodes": len(history),
        "totalSteps": steps,
        "meanReward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "bestReward": round(max(rewards), 4) if rewards else None,
        "worstReward": round(min(rewards), 4) if rewards else None,
        "totalReward": round(sum(rewards), 4) if rewards else None,
        "finalReward": round(rewards[-1], 4) if rewards else None,
        "hits": hits,
        "whiffs": whiffs,
        "hitRate": round(hits / (hits + whiffs), 4) if (hits + whiffs) else None,
        "decidedEpisodes": decided,
    }


def _plan_from_config(config: dict, kind: str) -> list[dict]:
    if not isinstance(config, dict):
        return []
    plan = config.get("plan")
    if isinstance(plan, list):
        out = []
        for item in plan:
            if isinstance(item, dict):
                out.append({
                    "opponentMode": item.get("opponentMode") or item.get("mode"),
                    "episodes": item.get("episodes"),
                })
        if out:
            return out
    stages = config.get("stages")
    if isinstance(stages, str) and stages.strip():
        out = []
        for part in stages.split(","):
            mode, _, count = part.partition(":")
            try:
                parsed = int(count)
            except ValueError:
                parsed = None
            out.append({"opponentMode": mode.strip(), "episodes": parsed})
        if out:
            return out
    mode = config.get("opponentMode") or config.get("opponent_mode")
    episodes = config.get("episodes")
    if mode or episodes is not None:
        return [{"opponentMode": mode, "episodes": episodes}]
    return []


class ArtifactStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self.dir = Path(artifacts_dir)
        self.reviews_dir = self.dir / "reviews"
        self.runs_dir = self.dir / "runs"

    # --- train-report ---------------------------------------------------
    def train_report(self, path: Path | None = None) -> dict:
        explicit = path is not None
        path = Path(path) if path else self.dir / "train-report.json"
        if not explicit and not path.is_file():
            # 新训练全部落在 runs/<时间>-train-<tag>/report.json，不再写扁平文件；
            # 这里回落到最近一次 train run，否则「数据可视化」永远只显示旧数据。
            fallback = self._latest_run_report("train")
            if fallback is not None:
                path = fallback
        if not path.is_file():
            return {"exists": False, "path": str(path)}
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return {"exists": True, "path": str(path), "error": "train-report.json 不是合法 JSON"}
        history = _slim_history(payload.get("history"))
        return {
            "exists": True,
            "path": str(path),
            "modifiedAt": _iso(path.stat().st_mtime),
            "startedAt": payload.get("startedAt"),
            "finishedAt": _finished_at(payload),
            "durationSeconds": payload.get("durationSeconds"),
            "config": payload.get("config"),
            "plan": _plan_from_config(payload.get("config") or {}, "train"),
            "summary": _history_summary(history),
            "history": history,
            "evaluation": _slim_history(payload.get("evaluation")),
            "policyPath": payload.get("policyPath"),
        }

    # --- multi-report ---------------------------------------------------
    def multi_report(self, path: Path | None = None) -> dict:
        explicit = path is not None
        path = Path(path) if path else self.dir / "multi-report.json"
        if not explicit and not path.is_file():
            fallback = self._latest_run_report("multi")
            if fallback is not None:
                path = fallback
        if not path.is_file():
            return {"exists": False, "path": str(path)}
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return {"exists": True, "path": str(path), "error": "multi-report.json 不是合法 JSON"}
        pairs = []
        for item in payload.get("pairs") or []:
            if not isinstance(item, dict):
                continue
            history = _slim_history(item.get("history"))
            pairs.append({
                "pair": item.get("pair"),
                "users": item.get("users"),
                "episodes": item.get("episodes"),
                "steps": item.get("steps"),
                "meanReward": item.get("meanReward"),
                "hits": item.get("hits"),
                "whiffs": item.get("whiffs"),
                "error": item.get("error"),
                "startedAt": item.get("startedAt"),
                "durationSeconds": item.get("durationSeconds"),
                "history": history,
                "summary": _history_summary(history),
            })
        history = []
        for pair in pairs:
            for item in pair["history"]:
                entry = dict(item)
                entry["pair"] = pair.get("pair")
                history.append(entry)
        return {
            "exists": True,
            "path": str(path),
            "modifiedAt": _iso(path.stat().st_mtime),
            "startedAt": payload.get("startedAt"),
            "finishedAt": _finished_at(payload),
            "durationSeconds": payload.get("durationSeconds"),
            "config": payload.get("config"),
            "plan": _plan_from_config(payload.get("config") or {}, "multi"),
            "pairs": pairs,
            "history": history,
            "summary": _history_summary(history),
        }

    # --- run 目录 -------------------------------------------------------
    def _latest_run_dir(self, kind: str | None = None) -> Path | None:
        """按目录名（时间前缀）倒序找最近一次运行，可选按 kind 过滤。"""
        if not self.runs_dir.is_dir():
            return None
        for directory in sorted(self.runs_dir.iterdir(), key=lambda item: item.name, reverse=True):
            if not directory.is_dir():
                continue
            if kind is not None and _run_kind(directory.name, _read_json(directory / "run.json")) != kind:
                continue
            return directory
        return None

    def _latest_run_report(self, kind: str) -> Path | None:
        """最近一次带 report.json 的运行（跳过仍在进行、还没写报告的）。"""
        if not self.runs_dir.is_dir():
            return None
        for directory in sorted(self.runs_dir.iterdir(), key=lambda item: item.name, reverse=True):
            if not directory.is_dir():
                continue
            meta_path = directory / "run.json"
            meta = _read_json(meta_path) if meta_path.is_file() else None
            if _run_kind(directory.name, meta) != kind:
                continue
            report = directory / "report.json"
            if report.is_file():
                return report
        return None

    def runs(self) -> list[dict]:
        """按目录名倒序返回每次运行；包含一个旧扁平数据的虚拟 run。"""
        items: list[dict] = []
        if self.runs_dir.is_dir():
            directories = sorted(
                (path for path in self.runs_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )[:MAX_RUNS]
            for directory in directories:
                try:
                    items.append(self._run_meta(directory))
                except OSError:
                    continue
        legacy = self._legacy_meta()
        if legacy is not None:
            items.append(legacy)
        items.sort(key=lambda item: (item.get("createdAt") or "", item.get("id") or ""), reverse=True)
        return items

    def _legacy_meta(self) -> dict | None:
        # 只认旧的扁平文件；train_report() 默认会回落到 runs/，这里必须显式传路径，
        # 否则新旧数据会互相引用、虚拟 run 重复出现。
        train = self.train_report(self.dir / "train-report.json")
        multi = self.multi_report(self.dir / "multi-report.json")
        if not train.get("exists") and not multi.get("exists"):
            return None
        kind = "multi" if multi.get("exists") and not train.get("exists") else "train"
        report = multi if kind == "multi" else train
        history = report.get("history") or []
        # legacy 只对应旧的扁平 reviews/，不能把最新 run 的复盘混进来。
        reviews = self._reviews_in(self.reviews_dir, "")
        summary = report.get("summary") or _history_summary(history)
        if kind == "multi":
            summary = dict(summary)
            summary.setdefault("episodes", sum(int(pair.get("episodes") or 0) for pair in report.get("pairs") or []))
            summary.setdefault("totalSteps", sum(int(pair.get("steps") or 0) for pair in report.get("pairs") or []))
            summary.setdefault("meanReward", _mean([
                float(pair["meanReward"]) for pair in report.get("pairs") or []
                if isinstance(pair.get("meanReward"), (int, float))
            ]))
        timestamps = [path.stat().st_mtime for path in (self.dir / "train-report.json", self.dir / "multi-report.json")
                      if path.is_file()]
        created = _iso(max(timestamps)) if timestamps else None
        return {
            "id": "legacy",
            "legacy": True,
            "name": "旧版扁平数据",
            "kind": kind,
            "tag": "legacy",
            "createdAt": created,
            "modifiedAt": report.get("modifiedAt"),
            "startedAt": report.get("startedAt"),
            "finishedAt": _finished_at(report),
            "durationSeconds": report.get("durationSeconds"),
            "status": "finished",
            "summary": summary,
            "episodes": summary.get("episodes"),
            "steps": summary.get("totalSteps"),
            "meanReward": summary.get("meanReward"),
            "finalReward": summary.get("finalReward"),
            "plan": report.get("plan") or [],
            "reviews": reviews,
            "reviewCount": len(reviews),
            "logPath": None,
            "reportPath": report.get("path"),
            "reportFile": Path(str(report.get("path") or "")).name,
            "error": report.get("error"),
        }

    def _run_meta(self, directory: Path) -> dict:
        name = directory.name
        match = _RUN_PREFIX_RE.match(name)
        if match:
            kind = match.group(3)
            tag = match.group(4) or ""
            created = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}T{match.group(2)[:2]}:{match.group(2)[2:]}"
        else:
            kind = "multi" if "multi" in name else "train"
            tag = ""
            created = _iso(directory.stat().st_mtime)

        report_path = directory / "report.json"
        if not report_path.is_file():
            for candidate in ("train-report.json", "multi-report.json"):
                if (directory / candidate).is_file():
                    report_path = directory / candidate
                    break
        payload = _read_json(report_path) if report_path.is_file() else None
        if isinstance(payload, dict) and "pairs" in payload:
            kind = "multi"

        # run.json 由 Dashboard 在启动时写入（训练脚本只写 report.json），
        # 里面记录了本次 run 用了哪份 reward 预设。
        run_meta = _read_json(directory / "run.json")
        reward_preset = run_meta.get("rewardPreset") if isinstance(run_meta, dict) else None
        run_model = None
        if isinstance(run_meta, dict) and run_meta.get("modelId"):
            run_model = {
                "id": str(run_meta.get("modelId")),
                "name": str(run_meta.get("modelName") or run_meta.get("modelId")),
            }

        report: dict = {}
        if isinstance(payload, dict):
            if kind == "multi":
                report = self.multi_report(report_path)
            else:
                report = self.train_report(report_path)
        summary = report.get("summary") or {}
        reviews = self._reviews_in(directory / "reviews", f"runs/{name}/reviews")
        log_path = directory / "run.log"
        if not log_path.is_file():
            log_path = None

        mtimes = [directory.stat().st_mtime]
        for path in (report_path, log_path):
            if path is not None and path.is_file():
                mtimes.append(path.stat().st_mtime)
        modified_ts = max(mtimes)
        if report.get("exists"):
            status = "finished"
        elif time.time() - modified_ts < RUNNING_WINDOW_SECONDS:
            status = "running"
        else:
            status = "interrupted"

        return {
            "id": name,
            "legacy": False,
            "name": name,
            "kind": kind,
            "tag": tag,
            "runDir": str(directory),
            "createdAt": created,
            "modifiedAt": _iso(modified_ts),
            "startedAt": report.get("startedAt"),
            "finishedAt": _finished_at(report),
            "durationSeconds": report.get("durationSeconds"),
            "status": status,
            "summary": summary,
            "episodes": summary.get("episodes"),
            "steps": summary.get("totalSteps"),
            "meanReward": summary.get("meanReward"),
            "finalReward": summary.get("finalReward"),
            "plan": report.get("plan") or [],
            "reviews": reviews,
            "reviewCount": len(reviews),
            "rewardPreset": reward_preset,
            "model": run_model,
            # G-624：run.json 里的持久化写回结果（enabled/ok/modelId/error）。
            "persist": run_meta.get("persist") if isinstance(run_meta, dict) else None,
            "persistWarning": run_meta.get("persistWarning") if isinstance(run_meta, dict) else None,
            "logPath": str(log_path) if log_path else None,
            # 历史 run（本次改造之前）没有 run.log：前端显式显示「本次运行没有日志（历史数据）」。
            "logState": "ok" if log_path else "no-log",
            "reportPath": str(report_path) if report_path.is_file() else None,
            "reportFile": report_path.name if report_path.is_file() else None,
            "error": report.get("error"),
        }

    def run_detail(self, run_id: str) -> dict:
        if run_id == "legacy":
            return self._legacy_detail()
        if not run_id or not RUN_ID_RE.match(run_id) or ".." in run_id:
            raise BadRequest("run id 非法")
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise NotFound(f"运行不存在: {run_id}")
        meta = self._run_meta(directory)
        report_path = meta.get("reportPath")
        if report_path:
            report = self.multi_report(Path(report_path)) if meta.get("kind") == "multi" else self.train_report(Path(report_path))
            meta["history"] = report.get("history") or []
            meta["evaluation"] = report.get("evaluation") or []
            meta["config"] = report.get("config")
            meta["pairs"] = report.get("pairs") or []
            meta["policyPath"] = report.get("policyPath")
        else:
            meta["history"] = []
            meta["evaluation"] = []
            meta["config"] = None
            meta["pairs"] = []
        meta["detail"] = True
        return meta

    def _legacy_detail(self) -> dict:
        meta = self._legacy_meta()
        if meta is None:
            raise NotFound("没有旧的扁平训练产物")
        # 必须显式传路径：train_report() 默认会回落到最新 run，会把 legacy 和新 run 混在一起。
        train = self.train_report(self.dir / "train-report.json")
        multi = self.multi_report(self.dir / "multi-report.json")
        meta["train"] = train
        meta["multi"] = multi
        meta["history"] = (multi.get("history") if meta.get("kind") == "multi" else train.get("history")) or []
        meta["evaluation"] = train.get("evaluation") or []
        meta["config"] = (multi if meta.get("kind") == "multi" else train).get("config")
        return meta

    def run_report_at(self, path: Path | str | None) -> dict:
        """按 report 路径读取（训练任务结束时用），自动识别 train/multi。"""
        if not path:
            return {"exists": False}
        target = Path(path)
        payload = _read_json(target) if target.is_file() else None
        if isinstance(payload, dict) and "pairs" in payload:
            return self.multi_report(target)
        return self.train_report(target)

    # --- reviews --------------------------------------------------------
    def _reviews_in(self, directory: Path, prefix: str) -> list[dict]:
        if not directory.is_dir():
            return []
        items = []
        for path in sorted(directory.rglob("*.json"))[:MAX_REVIEW_FILES]:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            relative = path.relative_to(directory).as_posix()
            markdown = path.with_suffix(".md")
            markdown_path = None
            if markdown.is_file():
                markdown_path = f"{prefix}/{relative[:-5]}.md" if prefix else relative[:-5] + ".md"
            items.append({
                "path": relative,
                "markdownPath": markdown_path,
                "episode": payload.get("episode"),
                "username": payload.get("username"),
                "winner": payload.get("winner"),
                "terminated": payload.get("terminated"),
                "truncated": payload.get("truncated"),
                "steps": payload.get("steps"),
                "startedAt": payload.get("startedAt"),
                "durationSeconds": payload.get("durationSeconds"),
                "totalReward": payload.get("totalReward"),
                "hits": payload.get("hits"),
                "whiffs": payload.get("whiffs"),
                "hitRate": payload.get("hitRate"),
                "maxCombo": payload.get("maxCombo"),
                "damageDealt": payload.get("damageDealt"),
                "damageTaken": payload.get("damageTaken"),
                "actionCounts": payload.get("actionCounts"),
                "rewardBreakdown": payload.get("rewardBreakdown"),
            })
        items.sort(key=lambda item: (str(item["path"]).split("/")[0], item.get("episode") or 0))
        return items

    def episode_detail(self, run_id: str, episode: int) -> dict:
        """单局详情：report.history[episode-1] + reviews/episode-XXX.{json,md}。

        ``episode`` 是 1 起始的全局局号（与 ``report.history`` 的 ``episode`` 字段一致）。
        """
        if not run_id or not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or ".." in run_id:
            raise BadRequest("run id 非法")
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise NotFound(f"运行不存在: {run_id}")
        try:
            index = int(episode)
        except (TypeError, ValueError) as exc:
            raise BadRequest("episode 需要整数") from exc

        meta = self._run_meta(directory)
        report_path = meta.get("reportPath")
        history: list = []
        if report_path:
            path = Path(report_path)
            report = self.multi_report(path) if meta.get("kind") == "multi" else self.train_report(path)
            history = report.get("history") or []
        total = len(history)
        if index < 1 or index > total:
            raise BadRequest(f"episode 超出范围：有效范围 1..{total}（该 run 共 {total} 局）")
        item = history[index - 1]
        if not isinstance(item, dict):
            raise BadRequest("该局的 history 数据损坏")

        review = None
        base = directory / "reviews" / f"episode-{index:03d}"
        markdown_path = base.with_suffix(".md")
        data_path = base.with_suffix(".json")
        if markdown_path.is_file() or data_path.is_file():
            markdown = ""
            if markdown_path.is_file():
                try:
                    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    markdown = ""
            data = _read_json(data_path) if data_path.is_file() else None
            review = {
                "markdown": markdown,
                "data": data if isinstance(data, dict) else None,
                "path": f"runs/{run_id}/reviews/{markdown_path.name}" if markdown_path.is_file() else None,
            }

        model = None
        run_meta = _read_json(directory / "run.json")
        if isinstance(run_meta, dict) and run_meta.get("modelId"):
            model = {
                "id": str(run_meta.get("modelId")),
                "name": str(run_meta.get("modelName") or run_meta.get("modelId")),
            }

        return {
            "runId": run_id,
            "runKind": meta.get("kind"),
            "episode": index,
            "episodeCount": total,
            "startedAt": item.get("startedAt"),
            "durationSeconds": item.get("durationSeconds"),
            "steps": item.get("steps"),
            "totalReward": item.get("totalReward"),
            "winner": item.get("winner"),
            "terminated": item.get("terminated"),
            "truncated": item.get("truncated"),
            "opponentMode": item.get("opponentMode"),
            "pairKey": item.get("pairKey"),
            "model": model,
            "review": review,
        }

    def reviews(self, run_id: str | None = None) -> list[dict]:
        """复盘列表。

        指定 ``run_id`` 时只返回该运行的复盘；未指定时保持旧行为
        （旧目录 + 最近一次 run），供独立的 ``/api/artifacts/reviews`` 使用。
        """
        if run_id:
            return self.run_detail(run_id).get("reviews") or []
        items = self._reviews_in(self.reviews_dir, "")
        latest = self._latest_run_dir("train") or self._latest_run_dir("multi")
        if latest is not None:
            items = self._reviews_in(latest / "reviews", f"runs/{latest.name}/reviews") + items
        return items

    def review_markdown(self, name: str) -> str:
        if not name or not isinstance(name, str):
            raise BadRequest("缺少复盘文件名")
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise BadRequest("复盘路径非法")
        if relative.suffix.lower() != ".md":
            raise BadRequest("只允许读取 .md 复盘")
        root = self.dir.resolve()
        target = (self.dir / relative).resolve()
        if target != root and root not in target.parents:
            raise BadRequest("复盘路径越界")
        parts = relative.parts
        allowed = (
            len(parts) >= 2 and parts[0] == "reviews"
        ) or (
            len(parts) >= 4 and parts[0] == "runs" and parts[2] == "reviews"
        )
        # 兼容旧的直接文件名（review_markdown("episode-001.md")）。
        if not allowed and len(parts) == 1:
            allowed = True
            target = (self.reviews_dir / relative).resolve()
            if target != self.reviews_dir.resolve() and self.reviews_dir.resolve() not in target.parents:
                raise BadRequest("复盘路径越界")
        if not allowed:
            raise BadRequest("复盘路径必须是 reviews/ 或 runs/<id>/reviews/ 下的文件")
        if not target.is_file():
            raise NotFound(f"复盘不存在: {name}")
        try:
            if target.stat().st_size > MAX_MARKDOWN_BYTES:
                raise BadRequest("复盘文件过大")
            return target.read_text(encoding="utf-8")
        except OSError as exc:
            raise NotFound(f"读取复盘失败: {exc}") from exc

    # --- stress ---------------------------------------------------------
    def stress(self, run_dir: Path | None = None) -> dict:
        """读取压测报告。

        ``run_dir`` 非空时只看该 run（选中 run 的数据必须跟着切换）；
        否则先看 artifacts 根目录，再看最近一次 run。
        """
        candidates: list[Path] = []
        if run_dir is not None:
            candidates.append(Path(run_dir) / "stress-report.json")
        else:
            candidates.append(self.dir / "stress-report.json")
            latest = self._latest_run_dir()
            if latest is not None:
                candidates.append(latest / "stress-report.json")
        path = next((item for item in candidates if item.is_file()), candidates[0])
        if not path.is_file():
            return {"exists": False, "path": str(path)}
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return {"exists": True, "path": str(path), "error": "stress-report.json 不是合法 JSON"}
        metrics = [
            {"key": "tpsAvg", "label": "TPS 均值", "value": payload.get("tpsAvg"), "unit": ""},
            {"key": "tpsMin", "label": "TPS 最低", "value": payload.get("tpsMin"), "unit": ""},
            {"key": "tpsMax", "label": "TPS 最高", "value": payload.get("tpsMax"), "unit": ""},
            {"key": "stepsPerSecond", "label": "步频", "value": payload.get("stepsPerSecond"), "unit": "step/s"},
        ]
        return {
            "exists": True,
            "derived": False,
            "path": str(path),
            "modifiedAt": _iso(path.stat().st_mtime),
            "raw": payload,
            "metrics": metrics,
        }

    def _derived_stress(self, run: dict, report: dict) -> dict:
        """没有 stress-report 时，从选中 run 派生步频/回合数/平均 reward。

        TPS 不在后端伪造；前端使用 ``/api/server`` 的实时 TPS 并标注“实时”。
        """
        summary = report.get("summary") or run.get("summary") or {}
        duration = report.get("durationSeconds") or run.get("durationSeconds")
        steps = summary.get("totalSteps") or run.get("steps")
        episodes = summary.get("episodes") or run.get("episodes")
        mean_reward = summary.get("meanReward")
        if run.get("kind") == "multi" and report.get("pairs"):
            pairs = report.get("pairs") or []
            if not steps:
                steps = sum(int(pair.get("steps") or 0) for pair in pairs)
            if not episodes:
                episodes = sum(int(pair.get("episodes") or 0) for pair in pairs)
        metrics = []
        if isinstance(steps, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
            metrics.append({
                "key": "stepsPerSecond",
                "label": "步频（由 run 派生）",
                "value": round(float(steps) / float(duration), 2),
                "unit": "step/s",
            })
        if episodes is not None:
            metrics.append({"key": "episodes", "label": "回合数", "value": episodes, "unit": ""})
        if isinstance(mean_reward, (int, float)):
            metrics.append({"key": "meanReward", "label": "平均 reward", "value": round(mean_reward, 2), "unit": ""})
        return {
            "exists": False,
            "derived": True,
            "sourceRun": run.get("id"),
            "metrics": metrics,
            "note": "未找到 stress-report.json，步频/回合数/平均 reward 由选中 run 派生；TPS 使用服务端实时值。",
        }

    # --- 选中 run / 聚合 -------------------------------------------------
    def _select_run(self, run_id: str | None, runs: list[dict]) -> dict | None:
        if run_id:
            if not RUN_ID_RE.match(run_id) or ".." in run_id:
                raise BadRequest("run id 非法")
            selected = next((item for item in runs if item.get("id") == run_id), None)
            if selected is None:
                raise NotFound(f"运行不存在: {run_id}")
            return selected
        if not runs:
            return None
        # 默认选“最近一次真实运行”；legacy 只在没有真实 run 时兜底，避免旧数据抢占默认视图。
        return next((item for item in runs if not item.get("legacy")), runs[0])

    @staticmethod
    def _public_run(run: dict | None) -> dict | None:
        if not run:
            return None
        keys = (
            "id", "legacy", "name", "kind", "tag", "runDir", "createdAt", "modifiedAt",
            "startedAt", "finishedAt", "durationSeconds", "status", "summary", "episodes",
            "steps", "meanReward", "finalReward", "plan", "reviewCount", "logPath",
            "reportPath", "reportFile", "error",
        )
        return {key: run.get(key) for key in keys}

    def summary(self, run_id: str | None = None) -> dict:
        """数据可视化聚合：默认按最近一次 run，可用 ``run`` 参数切换。"""
        runs = self.runs()
        selected = self._select_run(run_id, runs)
        train: dict = {"exists": False}
        multi: dict = {"exists": False}
        reviews: list[dict] = []
        stress: dict = {"exists": False}
        detail: dict | None = None

        if selected is not None and selected.get("legacy"):
            detail = self._legacy_detail()
            train = self.train_report(self.dir / "train-report.json")
            multi = self.multi_report(self.dir / "multi-report.json")
            reviews = self._reviews_in(self.reviews_dir, "")
            stress = self.stress()
            primary = multi if selected.get("kind") == "multi" else train
            if not stress.get("exists"):
                stress = self._derived_stress(detail, primary)
        elif selected is not None:
            detail = self.run_detail(selected["id"])
            report_path = detail.get("reportPath")
            if detail.get("kind") == "multi":
                multi = self.multi_report(Path(report_path)) if report_path else {"exists": False}
                primary = multi
            else:
                train = self.train_report(Path(report_path)) if report_path else {"exists": False}
                primary = train
            reviews = detail.get("reviews") or []
            run_dir = Path(detail["runDir"]) if detail.get("runDir") else None
            stress = self.stress(run_dir)
            if not stress.get("exists"):
                stress = self._derived_stress(detail, primary)

        return {
            "artifactsDir": str(self.dir),
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "selectedRun": self._public_run(detail),
            "runs": runs,
            "train": train,
            "multi": multi,
            "reviews": reviews,
            "stress": stress,
        }

    def merge(self, run_ids: list[str]) -> dict:
        """把多次运行合并成一份只读视图，用于数据页的多选/时间区间展示。"""
        if not run_ids:
            raise BadRequest("至少选择一个运行")
        if len(run_ids) > MAX_RUNS:
            raise BadRequest(f"一次最多合并 {MAX_RUNS} 个运行")

        all_runs = {str(item.get("id")): item for item in self.runs()}
        selected: list[dict] = []
        seen: set[str] = set()
        for raw in run_ids:
            run_id = str(raw or "").strip()
            if not run_id or run_id in seen:
                continue
            if not RUN_ID_RE.match(run_id) or ".." in run_id:
                raise BadRequest("run id 非法")
            run = all_runs.get(run_id)
            if run is None:
                raise NotFound(f"运行不存在: {run_id}")
            seen.add(run_id)
            selected.append(run)
        if not selected:
            raise BadRequest("至少选择一个运行")

        histories: list[dict] = []
        pairs: list[dict] = []
        reviews: list[dict] = []
        sources: list[dict] = []
        durations: list[float] = []
        started: list[str] = []
        for run in selected:
            run_id = str(run.get("id"))
            name = str(run.get("name") or run_id)
            detail = self._legacy_detail() if run.get("legacy") else self.run_detail(run_id)
            for item in detail.get("history") or []:
                entry = dict(item)
                entry["runId"] = run_id
                entry["runName"] = name
                entry["runKind"] = run.get("kind")
                entry["legacy"] = bool(run.get("legacy"))
                histories.append(entry)
            for pair in detail.get("pairs") or []:
                entry = {key: value for key, value in pair.items() if key != "history"}
                entry["runId"] = run_id
                entry["runName"] = name
                entry["pairKey"] = f"{run_id}::pair-{pair.get('pair')}"
                pairs.append(entry)
            for review in detail.get("reviews") or []:
                entry = dict(review)
                entry["runId"] = run_id
                entry["runName"] = name
                match = re.search(r"(?:^|/)pair-([^/]+)/", str(entry.get("path") or ""))
                entry["pairKey"] = f"{run_id}::pair-{match.group(1)}" if match else None
                reviews.append(entry)
            duration = run.get("durationSeconds")
            if isinstance(duration, (int, float)):
                durations.append(float(duration))
            if run.get("startedAt"):
                started.append(str(run["startedAt"]))
            sources.append({
                "id": run_id,
                "name": name,
                "kind": run.get("kind"),
                "status": run.get("status"),
                "episodes": run.get("episodes"),
                "steps": run.get("steps"),
                "meanReward": run.get("meanReward"),
                "startedAt": run.get("startedAt"),
                "legacy": bool(run.get("legacy")),
            })

        histories.sort(key=lambda item: (str(item.get("startedAt") or ""), int(item.get("episode") or 0)))
        reviews.sort(key=lambda item: (str(item.get("startedAt") or ""), int(item.get("episode") or 0)))
        reviews = reviews[:1000]
        summary = _history_summary(histories)
        if not histories:
            episodes = sum(int(run.get("episodes") or 0) for run in selected)
            steps = sum(int(run.get("steps") or 0) for run in selected)
            rewards = [
                (float(run["meanReward"]), int(run.get("episodes") or 0))
                for run in selected
                if isinstance(run.get("meanReward"), (int, float))
            ]
            total_episodes = sum(weight for _value, weight in rewards)
            summary = {
                "episodes": episodes,
                "totalSteps": steps,
                "meanReward": round(
                    sum(value * weight for value, weight in rewards) / total_episodes, 4
                ) if total_episodes else None,
                "bestReward": None,
                "worstReward": None,
                "totalReward": None,
                "finalReward": None,
                "hits": sum(int((run.get("summary") or {}).get("hits") or 0) for run in selected),
                "whiffs": sum(int((run.get("summary") or {}).get("whiffs") or 0) for run in selected),
                "hitRate": None,
                "decidedEpisodes": None,
            }

        kind = selected[0].get("kind") if len({run.get("kind") for run in selected}) == 1 else "mixed"
        duration = round(sum(durations), 2) if durations else None
        report = {
            "exists": True,
            "merged": True,
            "path": f"{len(selected)} 次运行合并",
            "startedAt": min(started) if started else None,
            "durationSeconds": duration,
            "summary": summary,
            "history": histories,
            "pairs": pairs,
        }
        has_train = any(run.get("kind") == "train" for run in selected)
        has_multi = any(run.get("kind") == "multi" for run in selected)
        train = dict(report) if has_train else {"exists": False}
        multi = dict(report) if has_multi else {"exists": False}
        selected_run = {
            "id": "merged",
            "merged": True,
            "legacy": False,
            "name": f"合并 {len(selected)} 次运行",
            "kind": kind,
            "status": "merged",
            "createdAt": max((str(run.get("createdAt") or "") for run in selected), default=""),
            "startedAt": min(started) if started else None,
            "finishedAt": max((str(run.get("finishedAt") or "") for run in selected), default="") or None,
            "durationSeconds": duration,
            "summary": summary,
            "episodes": summary.get("episodes"),
            "steps": summary.get("totalSteps"),
            "meanReward": summary.get("meanReward"),
            "finalReward": summary.get("finalReward"),
            "plan": [],
            "reviewCount": len(reviews),
            "runDir": None,
            "sources": sources,
        }
        stress = self._derived_stress(selected_run, report)
        return {
            "artifactsDir": str(self.dir),
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "merged": True,
            "selectedRun": selected_run,
            "selectedRunIds": [str(run.get("id")) for run in selected],
            "runs": list(all_runs.values()),
            "train": train,
            "multi": multi,
            "reviews": reviews,
            "stress": stress,
            "sources": sources,
        }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
