"""训练任务（gladiator_train.train / multi_train）的后台运行与进度解析。

训练进程的 stdout 是唯一的实时进度来源；这里按行解析关键事件：
- ``[train] === 阶段 idle：2 回合 ===``
- ``[train] episode 1/2 steps=200 reward=38.72 winner=None terminated=False 40.4s``
- ``[pair1] 完成：...`` / ``[multi] 2/2 场地完成 ...``
进程退出后读取 ``train-report.json`` / ``multi-report.json`` 附上报告摘要。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from artifacts import ArtifactStore
from config import DashboardConfig
from errors import BadRequest, Conflict, Forbidden, NotFound
from ports import is_port_open
from processes import ManagedProcess, ProcessManager
from rcon_client import run_command
from models import DEFAULT_ID as DEFAULT_MODEL_ID, spaces_dimensions
from model_locks import ModelLock
from reward_presets import DEFAULT_ID as DEFAULT_REWARD_PRESET_ID
from reward_presets import schema_document
from validation import (
    STAGE_MODES,
    format_plan,
    parse_bool,
    parse_center,
    parse_float,
    parse_int,
    parse_plan,
    parse_ports,
    parse_stages,
    pick,
    split_list,
    validate_username,
)

OPPONENT_MODES = ("random", "wander", "idle", "scripted", "rusher", "aimbot")

#: 多场地开启「融合（mean/weighted）」后，多个场地的权重会被平均成唯一一份写回模型，
#: 此时允许持久化；merge=none 或融合失败时自动降级为 persist=false + persistWarning。
MULTI_PERSIST_WARNING = (
    "多场地训练需开启「融合」才能持久化：融合会把多个场地的权重平均成唯一一份写回模型；"
    "若融合为 none 或融合失败，将自动降级为不写回模型"
)
GMOD_PERSIST_WARNING = "文件模型（.gmod）只读，可先「导入为策略模型」再开启持久化训练"

_RE_EPOCH = re.compile(r"^\[train\] === 回合 (\d+)/(\d+) ===")
# 新格式：--- 阶段 rusher：5 轮次 ---；同时兼容旧格式 === 阶段 idle：2 回合 ===
_RE_STAGE = re.compile(r"^\[train\] (?:===|---) ?阶段 (\w+)[：:] ?(\d+) (?:回合|轮次)")
_RE_EPISODE = re.compile(
    r"^\[train\] episode (\d+)/(\d+) steps=(\d+) reward=(-?\d+(?:\.\d+)?) "
    r"winner=(\S+) terminated=(\w+) ([\d.]+)s"
)
_RE_EVAL = re.compile(
    r"^\[eval\] episode (\d+)/(\d+) steps=(\d+) reward=(-?\d+(?:\.\d+)?) winner=(\S+)"
)
_RE_PAIR_DONE = re.compile(
    r"^\[pair(\d+)\] 完成：(\d+) episodes / (\d+) steps / mean reward (-?\d+(?:\.\d+)?)"
)
_RE_PAIR_FAIL = re.compile(r"^\[pair(\d+)\] 失败：(.*)$")
_RE_MULTI_DONE = re.compile(r"^\[multi\] (\d+)/(\d+) 场地完成")
_RE_TRAIN_DONE = re.compile(
    r"^\[train\] 完成：(\d+) episodes / (\d+) steps / mean reward (-?\d+(?:\.\d+)?)"
)


@dataclass
class TrainingJob:
    id: str
    kind: str
    params: dict
    total_episodes: int
    stage_counts: list[int] = field(default_factory=list)
    status: str = "starting"  # starting | running | finished | failed | stopped
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    epoch_index: int = 0
    epoch_count: int = 0
    stage_index: int = 0
    current_stage: Optional[str] = None
    stage_completed: int = 0
    episodes_completed: int = 0
    pairs_completed: int = 0
    steps: int = 0
    rewards: list[float] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    last_episode: Optional[dict] = None
    evaluation: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    report: Optional[dict] = None
    stop_requested: bool = False
    stop_worker_running: bool = False
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    report_path: Optional[str] = None
    auto_cleanup: bool = True
    owned_runtimes: list[str] = field(default_factory=list)
    cleanup_done: bool = False
    cleanup_count: int = 0
    cleanup_message: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    #: 本次训练的真实命令行，失败路径也要能写进 run.log。
    argv: list[str] = field(default_factory=list)
    #: 持久化训练写回结果：``{ok, modelId, error?, path?, training?}``。
    persist: dict = field(default_factory=dict)
    #: 为什么本次不写回（gmod 只读 / 多场地不支持）。
    persist_warning: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self, log_tail: list[str]) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "params": dict(self.params),
                "status": self.status,
                "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
                "startedAtEpoch": self.started_at,
                "finishedAt": (
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.finished_at))
                    if self.finished_at else None
                ),
                "durationSeconds": round((self.finished_at or time.time()) - self.started_at, 1),
                "pid": self.pid,
                "exitCode": self.exit_code,
                "epochIndex": self.epoch_index,
                "epochCount": self.epoch_count,
                "stageIndex": self.stage_index,
                "stageCount": len(self.stage_counts),
                "currentStage": self.current_stage,
                "stageCompleted": self.stage_completed,
                "episodesCompleted": self.episodes_completed,
                "totalEpisodes": self.total_episodes,
                "pairsCompleted": self.pairs_completed,
                "steps": self.steps,
                "rewards": list(self.rewards),
                "episodes": list(self.episodes),
                "lastEpisode": dict(self.last_episode) if self.last_episode else None,
                "evaluation": list(self.evaluation),
                "error": self.error,
                "report": self.report,
                "reportPath": self.report_path,
                "runId": self.run_id,
                "runDir": self.run_dir,
                "autoCleanup": self.auto_cleanup,
                "cleanup": {
                    "done": self.cleanup_done,
                    "count": self.cleanup_count,
                    "message": self.cleanup_message,
                },
                "notes": list(self.notes),
                "persist": dict(self.persist),
                "persistWarning": self.persist_warning,
                "logTail": list(log_tail),
            }


class TrainingRunner:
    def __init__(self, cfg: DashboardConfig, procs: ProcessManager, artifacts: ArtifactStore,
                 runtimes=None, reward_presets=None, scripts=None, models=None) -> None:
        self.cfg = cfg
        self.procs = procs
        self.artifacts = artifacts
        #: 可选：训练启动时自动补齐缺失的 bot-runtime（web 一键启动训练）。
        self.runtimes = runtimes
        #: reward 预设（``dashboard/var/reward-presets``）；为 None 时禁用该参数。
        self.reward_presets = reward_presets
        #: 脚本注册表（``dashboard/var/scripts.json``）：脚本列表与颜色来源。
        self.scripts = scripts
        #: 模型注册表（``dashboard/var/models``）：训练初始化权重来源。
        self.models = models
        #: 持久化训练写锁（同一模型同时只允许一个训练写回）。
        self.model_locks = ModelLock(cfg.var_dir / "models")
        self._job: Optional[TrainingJob] = None
        self._lock = threading.RLock()

    # --- 查询 -----------------------------------------------------------
    def running(self) -> bool:
        with self._lock:
            job = self._job
        if job is None:
            return False
        if job.status == "starting":
            return True
        if job.status in ("running", "stopping"):
            proc = self.procs.get(job.id)
            return bool(proc and proc.alive)
        return False

    def current(self) -> dict:
        with self._lock:
            job = self._job
        if job is None:
            return {"job": None, "options": self.options(), "running": False}
        proc = self.procs.get(job.id)
        tail = proc.snapshot(160)["tail"] if proc else []
        return {"job": job.snapshot(tail), "options": self.options(), "running": self.running()}

    def _script_items(self) -> list:
        """脚本列表：注册表合并结果；注册表不可用时退回内置 6 项。"""
        if self.scripts is not None:
            try:
                items = self.scripts.items()
                if items:
                    return items
            except (OSError, ValueError):
                pass
        return [
            {"id": mode, "name": mode.capitalize(), "builtin": True,
             "color": {"scripted": "§e", "rusher": "§c", "random": "§7",
                       "wander": "§a", "idle": "§8", "aimbot": "§d"}.get(mode, "§f")}
            for mode in OPPONENT_MODES
        ]

    def _model_items(self) -> list:
        if self.models is None:
            return [{"id": "default", "name": "default（全新初始化）", "builtin": True,
                     "kind": "policy", "valid": True, "error": None, "color": None,
                     "obsDim": None, "numActions": None, "episodes": 0, "meanReward": None}]
        try:
            items = self.models.list()
        except (OSError, ValueError):
            items = []
        return [
            {
                "id": item["id"],
                "name": item["name"],
                "builtin": bool(item.get("builtin")),
                "kind": item.get("kind") or "policy",
                "valid": bool(item.get("valid", True)),
                "error": item.get("error"),
                "warning": item.get("warning"),
                "color": item.get("color"),
                "source": item.get("source"),
                "sizeBytes": item.get("sizeBytes"),
                "sha256": item.get("sha256"),
                "obsDim": item.get("obsDim"),
                "numActions": item.get("numActions"),
                "episodes": item.get("episodes"),
                "meanReward": item.get("meanReward"),
                "hasPolicy": bool(item.get("hasPolicy")),
                "training": item.get("training"),
            }
            for item in items
        ]

    def _model_from_payload(self, payload: dict) -> dict:
        """解析 ``modelId`` → ``{id, name, kind, path, builtin}``。

        ``path=None`` 表示这次全新初始化（default 首次训练、没有 policy.npz）。
        G-624 起 default 是**可持久化**模型：写回过一次后 ``policy_path("default")``
        就指向 ``var/models/default/policy.npz``，下次选它会自动接着练。
        """
        raw = pick(payload, "modelId", "initPolicy", default=None)
        ident = str(raw).strip() if raw is not None else ""
        if not ident or ident == DEFAULT_MODEL_ID:
            ident = DEFAULT_MODEL_ID
        if self.models is None:
            if ident == DEFAULT_MODEL_ID:
                return {"id": ident, "name": "default（持久化训练）", "kind": "policy",
                        "path": None, "builtin": True}
            raise BadRequest("模型注册表未启用")
        try:
            model = self.models.get(ident)
        except NotFound as exc:
            raise BadRequest(f"模型不存在: {ident}") from exc
        if model.get("kind") == "gmod" and not model.get("valid", True):
            raise BadRequest(f"文件模型 {ident} 不可用：{model.get('error')}")
        expected_obs, expected_actions = spaces_dimensions()
        if model.get("obsDim") not in (None, expected_obs):
            raise BadRequest(
                f"模型 {ident} 的 obsDim={model.get('obsDim')} 与当前观察维度 {expected_obs} 不一致"
            )
        if model.get("numActions") not in (None, expected_actions):
            raise BadRequest(
                f"模型 {ident} 的 numActions={model.get('numActions')} 与当前动作空间 {expected_actions} 不一致"
            )
        path = self.models.policy_path(ident)
        if path is not None and Path(path).is_file():
            return {"id": ident, "name": str(model.get("name") or ident),
                    "kind": str(model.get("kind") or "policy"), "path": str(path),
                    "builtin": bool(model.get("builtin"))}
        if ident == DEFAULT_MODEL_ID:
            return {"id": ident, "name": str(model.get("name") or ident),
                    "kind": str(model.get("kind") or "policy"), "path": None,
                    "builtin": True}
        raise BadRequest(f"模型 {ident} 没有策略文件（policy.npz），请重新导入")

    def _persist_options(self, payload: dict, model: dict, *, kind: str, merge: str | None = None) -> dict:
        """算出本次是否写回、写回哪个模型，以及为什么不写回。

        multi 在 ``merge != "none"`` 时允许持久化（写回的是融合后的权重）；
        merge=none 或模型为只读 .gmod 时强制 persist=false 并给出原因。
        """
        requested = parse_bool(pick(payload, "persist", default=True), "persist")
        ident = model["id"]
        warning: str | None = None
        enabled = requested
        if model.get("kind") == "gmod":
            enabled = False
            warning = GMOD_PERSIST_WARNING
        elif kind == "multi":
            merge_enabled = bool(merge) and merge != "none"
            if merge_enabled:
                enabled = requested
                warning = None
            else:
                enabled = False
                warning = MULTI_PERSIST_WARNING
        return {
            "persist": bool(enabled),
            "persistModelId": ident if enabled else None,
            "persistModelKind": model.get("kind") if enabled else None,
            "persistWarning": warning,
            "persistRequested": bool(requested),
        }

    def _sync_role_colors(self, job: TrainingJob) -> None:
        """训练前把 scripts.json 的颜色同步到服务端标签（RCON ``caller tagcolor``）。

        训练环境在 Bot 进服后用 ``caller tag <user> <role>`` 设标签，所以只要先把
        role → 颜色写进调用方插件，标签颜色就会与 Web 配置一致。失败不阻断训练，
        原因写进 ``job.notes``（进度卡可见）。
        """
        if self.scripts is None:
            return
        roles: list[str] = []
        for item in (job.params.get("plan") or []):
            mode = str(item.get("opponentMode") or "").strip().lower()
            if mode and mode not in roles:
                roles.append(mode)
        # 两种任务里第一个 Bot 都是「被训练方」，角色名固定为 trained（见 training/env.py）。
        roles.insert(0, "trained")
        applied: list[str] = []
        for role in roles:
            color = self.scripts.color_for(role)
            if not color:
                continue
            try:
                run_command(self.cfg.rcon_host, self.cfg.rcon_port, self.cfg.rcon_password,
                            f"caller tagcolor {role} {color}", timeout=2.0)
                applied.append(f"{role}={color}")
            except Exception:  # noqa: BLE001 - 服务端/调用方插件不在时不阻断训练
                continue
        if applied:
            with job.lock:
                job.notes.append("已同步标签颜色: " + ", ".join(applied))

    def _reward_preset_summaries(self) -> list:
        if self.reward_presets is None:
            return []
        try:
            return self.reward_presets.list()
        except OSError:
            return []

    def options(self) -> dict:
        return {
            "opponentModes": list(OPPONENT_MODES),
            "stageModes": list(STAGE_MODES),
            "kinds": [
                {"value": "train", "label": "单场地训练（train）"},
                {"value": "multi", "label": "多场地并行（multi_train）"},
            ],
            # 脚本列表来自 dashboard/var/scripts.json（内置可改色、自定义可增删）。
            "scripts": self._script_items(),
            # 模型注册表：default 是可持久化默认模型，其余项带 policy.npz / .gmod。
            "models": self._model_items(),
            "modelSupport": True,
            "persistSupported": True,
            # reward 预设：列表来自 dashboard/var/reward-presets，schema 元数据供编辑 UI 渲染。
            "rewardPresets": self._reward_preset_summaries(),
            "rewardPresetSchema": schema_document(),
            "defaults": {
                "kind": "train",
                "rewardPreset": DEFAULT_REWARD_PRESET_ID,
                "modelId": DEFAULT_MODEL_ID,
                "persist": True,
                "users": ["Bot_T1", "Bot_T2"],
                "ports": [8770, 8771],
                "episodes": 4,
                "epochs": 1,
                "maxSteps": 200,
                "plan": [{"opponentMode": "scripted", "episodes": 4}],
                "stages": "",
                "opponentMode": "scripted",
                "seed": 7,
                "evalEpisodes": 1,
                "evalEpochs": 1,
                "autoSpawn": True,
                "autoCleanup": True,
                "center": "0.5,4,0.5",
                "spacing": 6.0,
                "prepSeconds": 3.0,
                "pairs": 2,
                "usersPrefix": "Bot_P",
                "portsBase": 8770,
            },
            "game": {"host": self.cfg.mc_host, "port": self.cfg.mc_port},
        }

    # --- 启动 -----------------------------------------------------------
    def _reserved_ports(self) -> set[int]:
        return {self.cfg.mc_port, self.cfg.bridge_port, self.cfg.rcon_port, self.cfg.port}

    @staticmethod
    def _safe_tag(value: str) -> str:
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
        return tag[:40] or "run"

    def _prepare_run_dir(self, kind: str, tag: str) -> tuple[str, Path]:
        """为本次训练创建独立目录，避免 episode-001 在不同 run 之间互相覆盖。

        目录名与训练脚本约定一致：<YYYYMMDD-HHMM>-<kind>-<tag>。
        """
        stamp = time.strftime("%Y%m%d-%H%M")
        base = f"{stamp}-{kind}-{self._safe_tag(tag)}"
        runs_dir = self.cfg.artifacts_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        candidate = runs_dir / base
        index = 2
        while candidate.exists():
            candidate = runs_dir / f"{base}-{index}"
            index += 1
        candidate.mkdir(parents=True, exist_ok=False)
        try:
            (candidate / "run.json").write_text(
                json.dumps({
                    "runId": candidate.name,
                    "kind": kind,
                    "tag": tag,
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "startedBy": "dashboard",
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return candidate.name, candidate

    def _record_run_meta(self, run_dir: Path | str | None, extra: dict) -> None:
        """把 Dashboard 已知、训练脚本不记录的元数据补进 run.json。

        训练侧只写 ``report.json``，不记录 ``--reward-preset`` 用了哪一份；
        这里补 ``plan`` / ``rewardPreset`` 等字段，数据页才能回显口径。
        """
        if not run_dir:
            return
        path = Path(run_dir) / "run.json"
        data: dict = {}
        try:
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
        data.update(extra)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return

    def _reward_preset_from_payload(self, payload: dict) -> dict | None:
        """解析请求里的 ``rewardPreset``（预设 id）→ ``{id, name, path}``。"""
        raw = pick(payload, "rewardPreset", "rewardPresetId", default=None)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise BadRequest("rewardPreset 需要预设 id 字符串")
        ident = raw.strip()
        if not ident:
            return None
        store = self.reward_presets
        if store is None:
            raise BadRequest("reward 预设存储未启用")
        try:
            preset = store.get(ident)
            path = store.path_for(ident)
        except NotFound as exc:
            raise BadRequest(f"reward 预设不存在: {ident}") from exc
        return {"id": ident, "name": str(preset.get("name") or ident), "path": path}

    def _build_train(self, payload: dict) -> tuple[list[str], dict, int, list[int]]:
        defaults = self.options()["defaults"]
        raw_users = pick(payload, "users", default=defaults["users"])
        users = [
            validate_username(item, f"用户名[{index}]")
            for index, item in enumerate(split_list(raw_users, "users"))
        ]
        ports = parse_ports(pick(payload, "ports", default=defaults["ports"]), "ports", self._reserved_ports())
        if len(users) != len(ports):
            raise BadRequest("users 与 ports 数量必须一致")
        if not 2 <= len(users) <= 16:
            raise BadRequest("训练至少需要 2 个 Bot（学习方 + 对手）")

        episodes = parse_int(
            pick(payload, "episodes", default=defaults["episodes"]), "episodes", minimum=1, maximum=500
        )
        max_steps = parse_int(
            pick(payload, "maxSteps", default=defaults["maxSteps"]), "maxSteps", minimum=1, maximum=5000
        )
        mode = str(pick(payload, "opponentMode", default=defaults["opponentMode"])).strip().lower()
        if mode not in OPPONENT_MODES:
            raise BadRequest(f"未知 opponentMode: {mode}")
        # plan 是唯一的回合来源：数组优先；缺省时兼容旧的 stages 字符串，最后兜底单段。
        raw_plan = pick(payload, "plan", default=None)
        if raw_plan is None:
            raw_plan = pick(payload, "stages", default=None)
        plan = parse_plan(raw_plan, fallback_mode=mode, fallback_episodes=episodes)
        plan_text = format_plan(plan)
        total_episodes = sum(count for _mode, count in plan)
        seed = parse_int(
            pick(payload, "seed", default=defaults["seed"]), "seed", minimum=0, maximum=2**31 - 1
        )
        epochs = parse_int(
            pick(payload, "epochs", default=defaults.get("epochs", 1)),
            "epochs", minimum=1, maximum=200,
        )
        eval_epochs = parse_int(
            pick(payload, "evalEpochs", "evalEpisodes", default=defaults.get("evalEpochs", 1)),
            "evalEpochs", minimum=0, maximum=50,
        )
        train_total = epochs * total_episodes
        reward_preset = self._reward_preset_from_payload(payload)
        if reward_preset:
            reward_preset = {**reward_preset, "applied": True}
        model = self._model_from_payload(payload)
        persist_opts = self._persist_options(payload, model, kind="train")
        auto_spawn = parse_bool(pick(payload, "autoSpawn", default=defaults["autoSpawn"]), "autoSpawn")
        center = parse_center(pick(payload, "center", default=defaults["center"]))
        spacing = parse_float(
            pick(payload, "spacing", default=defaults["spacing"]), "spacing", minimum=1.0, maximum=20.0
        )
        prep_seconds = parse_float(
            pick(payload, "prepSeconds", default=defaults["prepSeconds"]),
            "prepSeconds",
            minimum=0.0,
            maximum=30.0,
        )

        auto_cleanup = parse_bool(
            pick(payload, "autoCleanup", default=defaults.get("autoCleanup", True)), "autoCleanup"
        )
        run_id, run_dir = self._prepare_run_dir("train", users[0])
        report_path = run_dir / "report.json"
        review_dir = run_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.cfg.python_exe, "-m", "gladiator_train.train",
            "--users", *users,
            "--ports", *[str(port) for port in ports],
            "--host", self.cfg.mc_host,
            "--mc-port", str(self.cfg.mc_port),
            "--episodes", str(plan[0][1]),
            "--max-steps", str(max_steps),
            "--center", ",".join(str(value) for value in center),
            "--spacing", str(spacing),
            "--prep-seconds", str(prep_seconds),
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--eval-epochs", str(eval_epochs),
            "--run-dir", str(run_dir),
        ]
        # 始终用 --stages 表达计划：单段与多段走同一条路径，避免“回合数是否生效”歧义。
        argv += ["--stages", plan_text]
        if reward_preset:
            argv += ["--reward-preset", reward_preset["path"]]
        # model-id / model-name 只写进 report.config；init-policy 才决定权重起点。
        if model["path"]:
            argv += ["--init-policy", model["path"]]
        argv += ["--model-id", model["id"], "--model-name", model["name"]]
        if persist_opts["persist"]:
            # 写回统一由 Dashboard 负责；训练侧只把这个 id 记进 report.config。
            argv += ["--persist-to", persist_opts["persistModelId"]]
        if auto_spawn:
            argv += ["--auto-spawn"]

        stage_counts = [count for _mode, count in plan]
        params = {
            "users": users,
            "ports": ports,
            "plan": [{"opponentMode": item[0], "episodes": item[1]} for item in plan],
            "planText": plan_text,
            "episodes": plan[0][1],
            "episodesPerEpoch": total_episodes,
            "epochs": epochs,
            "evalEpochs": eval_epochs,
            # 口径分离：evalRounds = 每个评估回合的对局数（= Σ计划轮次）；
            # evalEpisodes = 评估总对局数 = evalEpochs × evalRounds。
            "evalRounds": total_episodes,
            "evalEpisodes": eval_epochs * total_episodes,
            "evalSupported": True,
            "totalEpisodes": train_total,
            "rewardPreset": reward_preset,
            "modelId": model["id"],
            "modelName": model["name"],
            "modelKind": model["kind"],
            "initPolicy": model["path"],
            "persist": persist_opts["persist"],
            "persistModelId": persist_opts["persistModelId"],
            "persistModelKind": persist_opts["persistModelKind"],
            "persistWarning": persist_opts["persistWarning"],
            "persistSupported": True,
            "maxSteps": max_steps,
            "seed": seed,
            "autoSpawn": auto_spawn,
            "autoCleanup": auto_cleanup,
            "center": list(center),
            "spacing": spacing,
            "prepSeconds": prep_seconds,
            "runId": run_id,
            "runDir": str(run_dir),
            "reportPath": str(report_path),
            "reviewDir": str(review_dir),
        }
        self._record_run_meta(run_dir, {
            "plan": params["plan"],
            "planText": plan_text,
            "epochs": epochs,
            "evalEpochs": eval_epochs,
            "evalRounds": total_episodes,
            "evalEpisodes": params["evalEpisodes"],
            "totalEpisodes": train_total,
            "rewardPreset": reward_preset,
            "modelId": params["modelId"],
            "modelName": params["modelName"],
            "initPolicy": params["initPolicy"],
            "persist": {"enabled": params["persist"], "modelId": params["persistModelId"],
                        "ok": None, "error": None},
            "persistWarning": params["persistWarning"],
        })
        return argv, params, train_total, stage_counts

    def _build_multi(self, payload: dict) -> tuple[list[str], dict, int, list[int]]:
        defaults = self.options()["defaults"]
        pairs = parse_int(
            pick(payload, "pairs", default=defaults["pairs"]), "pairs", minimum=1, maximum=8
        )
        prefix = validate_username(
            pick(payload, "users-prefix", "usersPrefix", default=defaults["usersPrefix"]), "usersPrefix"
        )
        base = parse_int(
            pick(payload, "ports-base", "portsBase", default=defaults["portsBase"]),
            "portsBase",
            minimum=1024,
            maximum=65535,
        )
        episodes = parse_int(
            pick(payload, "episodes", default=defaults["episodes"]), "episodes", minimum=1, maximum=200
        )
        max_steps = parse_int(
            pick(payload, "maxSteps", default=defaults["maxSteps"]), "maxSteps", minimum=1, maximum=5000
        )
        mode = str(pick(payload, "opponentMode", default="rusher")).strip().lower()
        if mode not in OPPONENT_MODES:
            raise BadRequest(f"未知 opponentMode: {mode}")
        raw_plan = pick(payload, "plan", default=None)
        if raw_plan is None:
            raw_plan = pick(payload, "stages", default=None)
        plan = parse_plan(raw_plan, fallback_mode=mode, fallback_episodes=episodes)
        plan_text = format_plan(plan)
        per_pair = sum(count for _mode, count in plan)
        epochs = parse_int(
            pick(payload, "epochs", default=defaults.get("epochs", 1)),
            "epochs", minimum=1, maximum=200,
        )
        per_pair_total = per_pair * epochs
        # 多场地训练侧（multi_train）没有 --eval-epochs，评估回合只存在于单场地 train。
        # 明确拒绝而不是静默忽略：否则表单里填了评估回合却毫无效果。
        eval_epochs = parse_int(
            pick(payload, "evalEpochs", "evalEpisodes", default=0),
            "evalEpochs", minimum=0, maximum=50,
        )
        if eval_epochs:
            raise BadRequest(
                "多场地训练不支持评估回合：evalEpochs 必须为 0（评估只在单场地 train 中提供）"
            )
        reward_preset = self._reward_preset_from_payload(payload)
        if reward_preset:
            reward_preset = {**reward_preset, "applied": True}
        model = self._model_from_payload(payload)
        merge = str(pick(payload, "merge", default="mean") or "mean").strip().lower()
        if merge not in ("mean", "weighted", "none"):
            merge = "mean"
        persist_opts = self._persist_options(payload, model, kind="multi", merge=merge)
        seed = parse_int(
            pick(payload, "seed", default=defaults["seed"]), "seed", minimum=0, maximum=2**31 - 1
        )
        auto_spawn = parse_bool(pick(payload, "autoSpawn", default=True), "autoSpawn")
        auto_cleanup = parse_bool(
            pick(payload, "autoCleanup", default=defaults.get("autoCleanup", True)), "autoCleanup"
        )
        spacing = parse_float(
            pick(payload, "spacing", default=defaults["spacing"]), "spacing", minimum=1.0, maximum=20.0
        )
        prep_seconds = parse_float(
            pick(payload, "prepSeconds", default=defaults["prepSeconds"]),
            "prepSeconds",
            minimum=0.0,
            maximum=30.0,
        )

        ports = list(range(base, base + pairs * 2))
        users = []
        for index in range(1, pairs + 1):
            users.append(f"{prefix}_{index}A")
            users.append(f"{prefix}_{index}B")
        reserved = self._reserved_ports()
        if any(port in reserved for port in ports):
            raise BadRequest(f"端口范围 {base}..{base + pairs * 2 - 1} 与保留端口冲突")
        if any(port > 65535 for port in ports):
            raise BadRequest("端口范围超出 65535")
        users = [validate_username(name) for name in users]

        run_id, run_dir = self._prepare_run_dir("multi", prefix)
        report_path = run_dir / "report.json"
        review_dir = run_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.cfg.python_exe, "-m", "gladiator_train.multi_train",
            "--pairs", str(pairs),
            "--users-prefix", prefix,
            "--ports-base", str(base),
            "--host", self.cfg.mc_host,
            "--mc-port", str(self.cfg.mc_port),
            "--episodes", str(per_pair),
            "--epochs", str(epochs),
            "--max-steps", str(max_steps),
            "--opponent-mode", plan[0][0],
            "--spacing", str(spacing),
            "--prep-seconds", str(prep_seconds),
            "--seed", str(seed),
            "--run-dir", str(run_dir),
        ]
        argv += ["--stages", plan_text]
        if reward_preset:
            argv += ["--reward-preset", reward_preset["path"]]
        argv += ["--merge", merge]
        if model["path"]:
            argv += ["--init-policy", model["path"]]
        argv += ["--model-id", model["id"], "--model-name", model["name"]]
        if persist_opts["persist"]:
            argv += ["--persist-to", persist_opts["persistModelId"]]
        if auto_spawn:
            argv += ["--auto-spawn"]
        params = {
            "pairs": pairs,
            "usersPrefix": prefix,
            "portsBase": base,
            "ports": ports,
            "users": users,
            "episodes": per_pair_total,
            "episodesPerEpoch": per_pair,
            "epochs": epochs,
            "maxSteps": max_steps,
            "opponentMode": plan[0][0],
            "plan": [{"opponentMode": item[0], "episodes": item[1]} for item in plan],
            "planText": plan_text,
            # 多场地没有评估阶段：显式写 0，前端/数据页据此不显示评估口径。
            "evalEpochs": 0,
            "evalRounds": 0,
            "evalEpisodes": 0,
            "evalSupported": False,
            "totalEpisodes": pairs * per_pair_total,
            "rewardPreset": reward_preset,
            "modelId": model["id"],
            "modelName": model["name"],
            "modelKind": model["kind"],
            "initPolicy": model["path"],
            "persist": persist_opts["persist"],
            "persistModelId": persist_opts["persistModelId"],
            "persistModelKind": persist_opts["persistModelKind"],
            "persistWarning": persist_opts["persistWarning"],
            "merge": merge,
            "mergeEnabled": merge != "none",
            "persistSupported": (merge != "none") and model.get("kind") != "gmod",
            "seed": seed,
            "autoSpawn": auto_spawn,
            "autoCleanup": auto_cleanup,
            "spacing": spacing,
            "prepSeconds": prep_seconds,
            "runId": run_id,
            "runDir": str(run_dir),
            "reportPath": str(report_path),
            "reviewDir": str(review_dir),
        }
        self._record_run_meta(run_dir, {
            "plan": params["plan"],
            "planText": plan_text,
            "epochs": epochs,
            "evalEpochs": 0,
            "evalSupported": False,
            "pairs": pairs,
            "totalEpisodes": params["totalEpisodes"],
            "rewardPreset": reward_preset,
            "modelId": params["modelId"],
            "modelName": params["modelName"],
            "initPolicy": params["initPolicy"],
            "persist": {"enabled": params["persist"], "modelId": params["persistModelId"],
                        "ok": None, "error": None},
            "persistWarning": params["persistWarning"],
        })
        return argv, params, pairs * per_pair_total, [per_pair] * pairs

    def _ensure_runtimes(self, params: dict, job: TrainingJob | None = None) -> tuple[list[str], list[str]]:
        """并发补齐缺失的 bot-runtime，统一 7 秒 deadline。

        返回 ``(本次拉起的 runtime id, 错误信息)``。id 会在启动成功后立刻写入
        ``job.owned_runtimes``，这样即使训练在启动阶段被停止，也能按 id 清理。
        """
        started_ids: list[str] = []
        errors: list[str] = []
        if self.runtimes is None or not params.get("autoSpawn", True):
            return started_ids, errors
        users = list(params.get("users") or [])
        ports = [int(port) for port in (params.get("ports") or [])]
        if not ports:
            return started_ids, errors

        deadline = time.monotonic() + 7.0

        def start_one(index: int, port: int) -> tuple[str | None, str | None]:
            if is_port_open(self.cfg.mc_host, port, timeout=0.2):
                return None, None
            username = users[index] if index < len(users) else f"Bot_{index + 1}"
            try:
                runtime = self.runtimes.start(username, port)
                runtime_id = (runtime or {}).get("id") or f"runtime-{port}"
            except Exception as exc:  # noqa: BLE001 - 统一收集，训练启动失败时一次性报出
                return None, f"{username}:{port} → {exc}"
            if job is not None:
                with job.lock:
                    if runtime_id not in job.owned_runtimes:
                        job.owned_runtimes.append(runtime_id)
            # 所有 runtime 并行等待，最长只等统一的 deadline，而不是每个端口 20 秒。
            while time.monotonic() < deadline:
                if is_port_open(self.cfg.mc_host, port, timeout=0.2):
                    break
                time.sleep(0.2)
            return runtime_id, None

        workers = min(8, len(ports))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(start_one, index, port) for index, port in enumerate(ports)]
            for future in as_completed(futures):
                try:
                    runtime_id, error = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                    continue
                if runtime_id:
                    started_ids.append(runtime_id)
                if error:
                    errors.append(error)
        return started_ids, errors

    def _stop_owned(self, runtime_ids: list[str], reason: str = "") -> int:
        """只停止明确记录为“本次训练拉起”的 runtime id（并发，单项超时很短）。"""
        if self.runtimes is None:
            return 0
        targets: list[str] = []
        for runtime_id in runtime_ids:
            if not runtime_id or str(runtime_id).startswith("external-"):
                continue
            try:
                current = self.runtimes.get(runtime_id)
            except Exception:  # noqa: BLE001
                continue
            if current is not None and current.get("alive"):
                targets.append(runtime_id)
        if not targets:
            return 0

        def stop_one(runtime_id: str) -> int:
            try:
                self.runtimes.stop(runtime_id)
                return 1
            except Exception:  # noqa: BLE001 - 清理失败不应掩盖训练结果
                return 0

        stopped = 0
        with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
            for value in pool.map(stop_one, targets):
                stopped += value
        return stopped

    def _cleanup_owned(self, job: TrainingJob) -> None:
        """训练进入终态后清理本次自动拉起的运行时；幂等，可由多个路径调用。"""
        with job.lock:
            if job.cleanup_done:
                return
            job.cleanup_done = True
            ids = list(job.owned_runtimes)
            enabled = job.auto_cleanup
        if not enabled:
            message = f"autoCleanup 已关闭，保留 {len(ids)} 个本次拉起的运行时"
            stopped = 0
        else:
            stopped = self._stop_owned(ids, reason="训练结束")
            message = f"已清理 {stopped} 个运行时" if ids else "本次训练没有自动拉起运行时"
        with job.lock:
            job.cleanup_count = stopped
            job.cleanup_message = message
            job.notes.append(message)

    def _attach_run_log(self, job: TrainingJob, proc: ManagedProcess) -> None:
        """把本次训练新增的 stdout 落一份到 run.log（增量读取，不整份读日志）。

        ``proc.log_offset`` 是本次进程 header 之前的字节偏移（由 ProcessManager 记录）；
        日志轮转只发生在启动新进程之前，所以这个偏移在新文件里对应 0，不会丢增量。
        """
        if not job.run_dir or not proc.log_path:
            return
        try:
            run_dir = Path(job.run_dir)
            target = run_dir / "run.log"
            if target.is_file():
                return
            source = Path(proc.log_path)
            start = max(0, int(getattr(proc, "log_offset", 0) or 0))
            try:
                size = source.stat().st_size
            except OSError:
                return
            if size < start:  # 运行中发生过轮转/截断：从头读
                start = 0
            with source.open("rb") as fh:
                fh.seek(start)
                data = fh.read()
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            # 去掉 Dashboard 写的 header 与 "$ 命令行"，run.log 只保留真正的输出。
            if lines and lines[0].startswith("=== "):
                lines = lines[1:]
            if lines and lines[0].startswith("$ "):
                lines[0] = lines[0][2:]
            text = "".join(lines)
            run_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError:
            return

    def _write_failure_log(self, job: TrainingJob, reason: str) -> None:
        """启动校验失败/启动前被停止时也留一份 run.log，写明原因与命令行。"""
        if not job.run_dir:
            return
        try:
            run_dir = Path(job.run_dir)
            target = run_dir / "run.log"
            if target.is_file():
                return
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            lines = [
                f"=== {stamp} 训练未启动 ===",
                f"原因: {reason}",
                "",
            ]
            argv = list(getattr(job, "argv", None) or [])
            if argv:
                lines.append("$ " + " ".join(str(item) for item in argv))
            notes = list(getattr(job, "notes", None) or [])
            for note in notes:
                lines.append(f"[dashboard] {note}")
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            return

    def start(self, payload: dict) -> dict:
        """创建训练任务后立即返回；补齐 runtime / 校验端口 / 拉起进程在后台线程完成。"""
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        kind = str(payload.get("kind") or "train").strip().lower()
        if kind == "train":
            argv, params, total, stage_counts = self._build_train(payload)
            listen_ports = params["ports"]
        elif kind == "multi":
            argv, params, total, stage_counts = self._build_multi(payload)
            listen_ports = params["ports"]
        else:
            raise BadRequest(f"未知训练类型: {kind}")

        with self._lock:
            previous = self._job
            if previous is not None and (
                previous.status in ("starting", "running", "stopping") or self.running()
            ):
                raise Conflict("已有训练任务在运行，请先停止")
            # 持久化训练：同一模型同时只允许一个训练持有写锁（冲突直接 409）。
            lock_result = None
            if params.get("persist") and params.get("persistModelId"):
                lock_result = self.model_locks.acquire(
                    params["persistModelId"], params["runId"])
                if not lock_result["ok"]:
                    raise Conflict(lock_result["error"] or "模型写锁获取失败")
            job = TrainingJob(
                id="training",
                kind=kind,
                params=params,
                total_episodes=total,
                stage_counts=stage_counts,
                epoch_count=int(params.get("epochs", 1) or 1),
                status="starting",
                run_id=params.get("runId"),
                run_dir=params.get("runDir"),
                report_path=params.get("reportPath"),
                auto_cleanup=bool(params.get("autoCleanup", True)),
                argv=list(argv),
                persist={"enabled": bool(params.get("persist")),
                         "modelId": params.get("persistModelId"),
                         "ok": None, "error": None},
                persist_warning=params.get("persistWarning"),
            )
            if lock_result and lock_result.get("warning"):
                job.notes.append(lock_result["warning"])
            self._job = job

        thread = threading.Thread(
            target=self._launch,
            args=(job, argv, listen_ports),
            name="training-launch",
            daemon=True,
        )
        thread.start()
        return self.current()

    def _launch(self, job: TrainingJob, argv: list[str], listen_ports: list[int]) -> None:
        """后台启动：并发补齐 runtime → 校验端口 → spawn 训练进程。"""
        try:
            _owned, errors = self._ensure_runtimes(job.params, job)
            with job.lock:
                cancelled = job.stop_requested
            if cancelled:
                self._finish_before_spawn(job)
                return
            missing = [
                port for port in listen_ports
                if not is_port_open(self.cfg.mc_host, int(port), timeout=0.2)
            ]
            if missing:
                detail = ""
                if errors:
                    detail = "；自动启动失败: " + "；".join(errors[:4])
                self._fail_before_spawn(
                    job,
                    "以下通道 B 端口没有 bot-runtime 在监听: "
                    + ", ".join(str(port) for port in missing)
                    + detail,
                )
                return

            # 把 Web 里配置的脚本颜色写进调用方插件（失败不阻断，原因进 notes）。
            self._sync_role_colors(job)

            def on_line(line: str) -> None:
                self._handle_line(job, line)

            def on_exit(proc: ManagedProcess) -> None:
                self._handle_exit(job, proc)

            # spawn 与 stop_requested 检查放在同一把锁里，避免“刚检查完就被停止”的竞态。
            with job.lock:
                if job.stop_requested:
                    cancelled = True
                else:
                    cancelled = False
                    proc = self.procs.spawn(
                        proc_id="training",
                        kind="training",
                        label=f"训练 {job.kind}",
                        argv=argv,
                        cwd=self.cfg.training_dir,
                        env={"PYTHONPATH": self._python_path()},
                        tail_lines=2000,
                        on_line=on_line,
                        on_exit=on_exit,
                    )
                    job.pid = proc.pid
                    if job.status == "starting":
                        job.status = "running"
            if cancelled:
                self._finish_before_spawn(job)
        except Exception as exc:  # noqa: BLE001 - 后台线程必须把失败写回 job
            with job.lock:
                was_stopping = job.stop_requested
            if was_stopping:
                self._finish_before_spawn(job)
            else:
                self._fail_before_spawn(job, str(exc))

    def _finish_before_spawn(self, job: TrainingJob) -> None:
        with job.lock:
            if job.status in ("starting", "running", "stopping"):
                job.status = "stopped" if job.stop_requested else "failed"
                job.finished_at = time.time()
        self._write_failure_log(job, "训练在启动前被停止（尚未 spawn 训练进程）")
        self._mark_persist_skipped(job, "训练在启动前被停止，未写回模型")
        self._release_persist_lock(job)
        self._cleanup_owned(job)

    def _fail_before_spawn(self, job: TrainingJob, message: str) -> None:
        with job.lock:
            if job.status in ("starting", "running", "stopping"):
                job.status = "failed"
                job.error = message
                job.finished_at = time.time()
        self._write_failure_log(job, message)
        self._mark_persist_skipped(job, f"训练启动失败（{message}），未写回模型")
        self._release_persist_lock(job)
        self._cleanup_owned(job)

    def _mark_persist_skipped(self, job: TrainingJob, reason: str) -> None:
        params = job.params or {}
        if not (params.get("persist") and params.get("persistModelId")):
            return
        with job.lock:
            job.persist = {"enabled": True, "modelId": params["persistModelId"],
                           "ok": False, "skipped": True, "error": reason}
            result = dict(job.persist)
        self._record_run_meta(job.run_dir, {"persist": result})

    def _python_path(self) -> str:
        existing = os.environ.get("PYTHONPATH", "")
        return str(self.cfg.training_dir) + (os.pathsep + existing if existing else "")

    def stop(self) -> dict:
        """请求停止后立即返回；进程收尾与 autoCleanup 放到后台线程。"""
        if not self.cfg.allow_process_control:
            raise Forbidden("Dashboard 以只读模式运行（DASHBOARD_ALLOW_PROCESS_CONTROL=0）")
        with self._lock:
            job = self._job
        if job is None or job.status in ("finished", "failed", "stopped"):
            raise Conflict("没有正在运行的训练任务")
        with job.lock:
            job.stop_requested = True
            if job.status in ("starting", "running"):
                job.status = "stopping"
            already_stopping = job.stop_worker_running
            job.stop_worker_running = True
            proc = self.procs.get(job.id)
        if proc is not None and proc.alive:
            if not already_stopping:
                threading.Thread(
                    target=self._stop_process,
                    args=(job,),
                    name="training-stop",
                    daemon=True,
                ).start()
        elif job.status == "stopping" and job.pid is None and not already_stopping:
            # 还没 spawn：_launch 在 ensure 结束后会看到 stop_requested 并收尾。
            pass
        elif not already_stopping:
            self._finish_before_spawn(job)
        return self.current()

    def _stop_process(self, job: TrainingJob) -> None:
        try:
            self.procs.stop(job.id, timeout=2.0, terminate_timeout=2.0)
        except Exception:  # noqa: BLE001 - 进程可能已经自行退出
            pass
        with job.lock:
            if job.status in ("starting", "running", "stopping"):
                job.status = "stopped"
                job.finished_at = time.time()
        self._cleanup_owned(job)
        proc = self.procs.get(job.id)
        if proc is not None:
            self._attach_run_log(job, proc)

    # --- stdout 解析 ----------------------------------------------------
    def _handle_line(self, job: TrainingJob, line: str) -> None:
        with job.lock:
            match = _RE_EPOCH.match(line)
            if match:
                job.epoch_index = int(match.group(1))
                job.epoch_count = int(match.group(2))
                # 新回合重新按计划从第 1 段开始计数
                job.stage_index = 0
                job.stage_completed = 0
                return
            match = _RE_STAGE.match(line)
            if match:
                job.current_stage = match.group(1)
                job.stage_index += 1
                job.stage_completed = 0
                return
            match = _RE_EPISODE.match(line)
            if match:
                index = int(match.group(1))
                job.stage_completed = index
                # multi 的每个 pair 都从 1 开始编号且并发输出，不能按 pair 内序号推全局进度；
                # 每收到一条 episode 行就是完成了一局。
                job.episodes_completed = min(job.total_episodes, job.episodes_completed + 1)
                job.steps += int(match.group(3))
                try:
                    job.rewards.append(float(match.group(4)))
                except ValueError:
                    pass
                duration = float(match.group(7))
                started_epoch = time.time() - duration
                entry = {
                    "episode": index,
                    "steps": int(match.group(3)),
                    "totalReward": float(match.group(4)),
                    "reward": float(match.group(4)),
                    "winner": None if match.group(5) in ("None", "null") else match.group(5),
                    "terminated": match.group(6) == "True",
                    "durationSeconds": duration,
                    "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_epoch)),
                    "startedAtEpoch": round(started_epoch, 3),
                }
                job.last_episode = entry
                job.episodes.append(entry)
                if len(job.episodes) > 1000:
                    job.episodes = job.episodes[-1000:]
                return
            match = _RE_EVAL.match(line)
            if match:
                job.evaluation.append({
                    "episode": int(match.group(1)),
                    "steps": int(match.group(3)),
                    "reward": float(match.group(4)),
                    "winner": None if match.group(5) in ("None", "null") else match.group(5),
                })
                return
            match = _RE_PAIR_DONE.match(line)
            if match:
                # pair 的 steps/mean reward 是各 episode 的汇总，不能再次累加；
                # 否则 multi 的步数和 reward 曲线会翻倍、并混入 pair 均值点。
                job.pairs_completed = max(job.pairs_completed, int(match.group(1)))
                return
            match = _RE_PAIR_FAIL.match(line)
            if match:
                job.error = f"场地 {match.group(1)} 失败：{match.group(2)}"
                return
            match = _RE_MULTI_DONE.match(line)
            if match:
                job.pairs_completed = int(match.group(1))
                return
            match = _RE_TRAIN_DONE.match(line)
            if match:
                job.episodes_completed = int(match.group(1))
                return

    def _handle_exit(self, job: TrainingJob, proc: ManagedProcess) -> None:
        with job.lock:
            job.exit_code = proc.exit_code
            job.finished_at = time.time()
            stopped = job.stop_requested
            code = proc.exit_code
        if stopped:
            with job.lock:
                job.status = "stopped"
        elif code == 0:
            report = self._load_report(job)
            with job.lock:
                job.status = "finished"
                job.report = report
            self._sync_job_from_report(job, report)
        else:
            tail = proc.snapshot(120)["tail"]
            error = None
            for line in reversed(tail):
                text = line.strip()
                if not text or text.startswith("==="):
                    continue
                if any(key in text for key in ("Traceback", "Error", "error", "错误", "失败", "无法导入", "Exception")):
                    error = text
                    break
            with job.lock:
                job.status = "failed"
                job.error = error or f"训练进程退出码 {code}"
        # 无论成功、失败还是手动停止，都在这里收尾；autoCleanup 只停本次拉起的 runtime。
        self._attach_run_log(job, proc)
        # 写回必须在 cleanup.done 之前完成：测试/前端以 cleanup 作为「收尾结束」信号。
        self._finalize_persist(job)
        self._cleanup_owned(job)

    def _finalize_persist(self, job: TrainingJob) -> None:
        """训练进入终态：只有 ``finished`` 才写回；任何终态都释放模型写锁。"""
        params = job.params or {}
        enabled = bool(params.get("persist"))
        model_id = params.get("persistModelId")
        if not (enabled and model_id):
            return
        try:
            with job.lock:
                status = job.status
            if status == "finished":
                # 多场地：写回的是融合后的 run/policy.npz；融合失败（无可用权重）时自动降级。
                if job.kind == "multi":
                    merge = self._read_multi_merge(job.run_dir)
                    if merge is None or not merge.get("available"):
                        msg = ("多场地融合失败（无可用权重），已自动降级为不写回模型；"
                               "请在报告里查看 merge 详情")
                        result = {"enabled": True, "modelId": model_id, "ok": False,
                                  "skipped": True, "degraded": True, "error": msg}
                        self._record_run_meta(job.run_dir, {"persistWarning": msg})
                        with job.lock:
                            job.persist_warning = msg
                    else:
                        result = self._write_back_policy(job, model_id)
                        result.update({
                            "merged": True,
                            "mergePairCount": len(merge.get("pairs") or []),
                            "mergeMethod": merge.get("method"),
                        })
                else:
                    result = self._write_back_policy(job, model_id)
            else:
                result = {
                    "enabled": True, "modelId": model_id, "ok": False, "skipped": True,
                    "error": f"训练状态为 {status}（只有正常 finished 才写回），本次未写回模型",
                }
            with job.lock:
                job.persist = result
                if result.get("error") and not result.get("skipped"):
                    job.notes.append(f"模型写回失败：{result['error']}")
            self._record_run_meta(job.run_dir, {"persist": result})
        finally:
            self.model_locks.release(model_id, job.run_id or "")

    @staticmethod
    def _read_multi_merge(run_dir) -> dict | None:
        """读训练报告的 ``merge`` 段；缺失或非 dict 时返回 None。"""
        if not run_dir:
            return None
        report_path = Path(run_dir) / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        merge = report.get("merge")
        return merge if isinstance(merge, dict) else None

    def _write_back_policy(self, job: TrainingJob, model_id: str) -> dict:
        """从 run 目录取最终权重，交给 ModelStore 原子写回并累加元数据。"""
        if self.models is None:
            return {"enabled": True, "modelId": model_id, "ok": False,
                    "error": "模型注册表未启用，无法写回"}
        policy_file = Path(job.run_dir) / "policy.npz" if job.run_dir else None
        with job.lock:
            episodes = len(job.episodes) or int(job.episodes_completed)
            steps = int(job.steps)
            rewards = list(job.rewards)
        mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else None
        stats = {
            "runId": job.run_id,
            "episodes": episodes,
            "steps": steps,
            "meanReward": mean_reward,
        }
        result = self.models.write_back(model_id, policy_file, stats)
        result.update({"enabled": True, "modelId": model_id})
        return result

    def _release_persist_lock(self, job: TrainingJob) -> None:
        params = job.params or {}
        if params.get("persist") and params.get("persistModelId"):
            self.model_locks.release(params["persistModelId"], job.run_id or "")

    def _sync_job_from_report(self, job: TrainingJob, report: dict) -> None:
        """以 report.json 为最终口径校准进度，避免多 pair 并发输出造成计数漂移。"""
        if not isinstance(report, dict) or not report.get("exists"):
            return
        history = list(report.get("history") or [])
        pairs = list(report.get("pairs") or [])
        with job.lock:
            if history:
                job.episodes = list(history)
                job.rewards = [
                    float(item.get("totalReward", item.get("reward", 0.0)) or 0.0)
                    for item in history
                ]
                job.steps = sum(int(item.get("steps") or 0) for item in history)
                job.episodes_completed = len(history)
                job.last_episode = history[-1]
            if job.kind == "multi" and pairs:
                job.pairs_completed = len([pair for pair in pairs if not pair.get("error")])

    def _load_report(self, job: TrainingJob) -> dict:
        try:
            report = self.artifacts.run_report_at(job.report_path)
            if not report.get("exists"):
                return {"exists": False, "path": job.report_path}
            return {
                "exists": True,
                "path": report.get("path"),
                "runId": job.run_id,
                "kind": job.kind,
                "startedAt": report.get("startedAt"),
                "finishedAt": report.get("finishedAt"),
                "durationSeconds": report.get("durationSeconds"),
                "plan": report.get("plan") or [],
                "summary": report.get("summary"),
                "history": report.get("history") or [],
                "pairs": report.get("pairs") or [],
                "evaluation": report.get("evaluation") or [],
            }
        except Exception as exc:  # noqa: BLE001 - 报告读取失败不应影响任务状态
            return {"exists": False, "error": str(exc), "path": job.report_path}
