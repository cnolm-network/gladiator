"""模型注册表：``dashboard/var/models/<id>/model.json`` + ``policy.npz``。

schema ``gladiator.model/v1``::

    {
      "format": "gladiator.model/v1",
      "id": "rush-v3",
      "name": "Rush v3",
      "description": "从 run 20261108-1900-train-Bot_A 导入",
      "createdAt": "2026-11-08T22:50:00",
      "updatedAt": "2026-11-08T22:50:00",
      "source": {"runId": "20261108-1900-train-Bot_A", "runName": "...", "pair": null},
      "obsDim": 19, "numActions": 11, "hidden": 64,
      "episodes": 12, "steps": 2400, "meanReward": 12.3,
      "sha256": "…", "note": "", "builtin": false
    }

设计要点：

- **不依赖 numpy**：``.npz`` 是 zip，里面每个成员是 ``.npy``。这里只解析 npy 的 header
  （magic + 版本 + header dict），拿到 shape/dtype，用来校验 ``W1/b1/W2/b2`` 与维度；
- 内置 ``default`` 表示「全新初始化」，没有 ``policy.npz``，不可删、不可改 id，可改名称/说明；
- 覆盖与删除前把整个目录备份到 ``dashboard/var/models-trash/``；
- ``obsDim`` / ``numActions`` 必须与 ``training/gladiator_env/spaces.py`` 的
  ``OBS_DIM`` / ``NUM_ACTIONS`` 一致（从源码读取常量，不 import，避免引入 numpy）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

from config import ROOT
from errors import BadRequest, Conflict, NotFound

if TYPE_CHECKING:  # 避免与 gmod.py 循环导入（gmod 复用本模块的 npy 解析）
    from gmod import GmodScanner

FORMAT = "gladiator.model/v1"
TRAINING_FORMAT = "gladiator.model-training/v1"
DEFAULT_ID = "default"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: 列表/接口里 .gmod 的 id 前缀：``gmod:<文件名 stem>``。
#: 这样同名 .gmod 与可持久化策略模型可以并存，不会互相顶掉（G-626 回归修复）。
GMOD_PREFIX = "gmod:"


def gmod_stem(model_id: Any) -> str | None:
    """``gmod:rush-v3`` → ``rush-v3``；不是 gmod 引用时返回 None。"""
    text = str(model_id or "").strip()
    if text.startswith(GMOD_PREFIX) and len(text) > len(GMOD_PREFIX):
        return text[len(GMOD_PREFIX):]
    return None
NAME_MAX = 60
DESCRIPTION_MAX = 300
NOTE_MAX = 300
REQUIRED_ARRAYS = ("W1", "b1", "W2", "b2")

SPACES_FILE = Path(ROOT) / "training" / "gladiator_env" / "spaces.py"
_ARTIFACTS_ROOT = Path(ROOT) / "training" / "artifacts"

NPY_MAGIC = b"\x93NUMPY"


def spaces_dimensions(path: Path | None = None) -> tuple[int, int]:
    """从 ``spaces.py`` 源码里读 ``OBS_DIM`` / ``NUM_ACTIONS``（正则，不 import）。"""
    target = Path(path) if path is not None else SPACES_FILE
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return 19, 11
    obs = re.search(r"^OBS_DIM\s*=\s*(\d+)", text, re.MULTILINE)
    actions = re.search(r"^NUM_ACTIONS\s*=\s*(\d+)", text, re.MULTILINE)
    return (int(obs.group(1)) if obs else 19, int(actions.group(1)) if actions else 11)


def _npy_header(blob: bytes) -> tuple[tuple[int, ...], str]:
    """解析 ``.npy`` 的 header，返回 (shape, descr)。"""
    if not blob.startswith(NPY_MAGIC):
        raise BadRequest("不是合法的 npy 数据")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", blob[8:12])[0]
        start = 12
    else:  # pragma: no cover - 未知版本
        raise BadRequest(f"不支持的 npy 版本: {major}")
    header = blob[start:start + header_len].decode("latin1", errors="replace")
    shape_match = re.search(r"'shape':\s*\(([^)]*)\)", header)
    descr_match = re.search(r"'descr':\s*'([^']+)'", header)
    if shape_match is None:
        raise BadRequest("npy header 缺少 shape")
    parts = [part.strip() for part in shape_match.group(1).split(",") if part.strip()]
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise BadRequest(f"npy shape 非法: {shape_match.group(1)}") from exc
    return shape, (descr_match.group(1) if descr_match else "")


def inspect_policy(path: Path) -> dict:
    """检查 ``policy.npz``：必需数组、形状一致、与当前环境维度一致。"""
    target = Path(path)
    if not target.is_file():
        raise BadRequest(f"模型文件不存在: {target}")
    shapes: dict[str, tuple[int, ...]] = {}
    try:
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            missing = [key for key in REQUIRED_ARRAYS if f"{key}.npy" not in names]
            if missing:
                raise BadRequest(f"模型缺少参数数组: {', '.join(missing)}（需要 W1/b1/W2/b2）")
            if "_meta.npy" not in names:
                # 训练侧 Policy.load() 靠 _meta 还原 obs_dim/n_actions/hidden；
                # 缺它会在训练启动时才崩溃，必须在导入时挡掉。
                raise BadRequest("模型缺少 _meta.npy，无法被训练侧加载；请使用训练产物 policy.npz")
            with archive.open("_meta.npy") as handle:
                _npy_header(handle.read())
            for key in REQUIRED_ARRAYS:
                with archive.open(f"{key}.npy") as handle:
                    blob = handle.read()
                shapes[key], _descr = _npy_header(blob)
    except zipfile.BadZipFile as exc:
        raise BadRequest("模型文件不是合法的 npz（zip）归档") from exc

    w1, b1, w2, b2 = (shapes[key] for key in REQUIRED_ARRAYS)
    if len(w1) != 2 or len(w2) != 2 or len(b1) != 1 or len(b2) != 1:
        raise BadRequest("参数形状非法：W1/W2 应为二维、b1/b2 应为一维")
    hidden, obs_dim = w1
    num_actions, hidden2 = w2
    if hidden2 != hidden:
        raise BadRequest(f"W1/W2 的 hidden 不一致: {hidden} vs {hidden2}")
    if b1[0] != hidden:
        raise BadRequest(f"b1 长度 {b1[0]} 与 hidden {hidden} 不一致")
    if b2[0] != num_actions:
        raise BadRequest(f"b2 长度 {b2[0]} 与 numActions {num_actions} 不一致")

    expected_obs, expected_actions = spaces_dimensions()
    if obs_dim != expected_obs:
        raise BadRequest(f"模型 obsDim={obs_dim} 与当前环境观察维度 {expected_obs} 不一致")
    if num_actions != expected_actions:
        raise BadRequest(f"模型 numActions={num_actions} 与当前动作空间 {expected_actions} 不一致")

    return {
        "obsDim": int(obs_dim),
        "numActions": int(num_actions),
        "hidden": int(hidden),
        "sha256": _sha256(target),
        "sizeBytes": target.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_model() -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "format": FORMAT,
        "id": DEFAULT_ID,
        "name": "default（持久化训练）",
        "description": ("默认可持久化模型。首次训练写回后生成 policy.npz，"
                        "之后选它即接着上次继续练；关闭「持久化训练」则只读起点、产物仅留在 run 里。"),
        "createdAt": now,
        "updatedAt": now,
        "source": {"runId": None, "runName": None, "pair": None},
        "obsDim": spaces_dimensions()[0],
        "numActions": spaces_dimensions()[1],
        "hidden": 64,
        "episodes": 0,
        "steps": 0,
        "meanReward": None,
        "sha256": None,
        "note": "",
        "builtin": True,
    }


class ModelStore:
    """``dashboard/var/models`` 的读写；每次覆盖/删除都留 ``models-trash`` 副本。"""

    def __init__(self, root: Path, artifacts_dir: Path | None = None,
                 gmod: GmodScanner | None = None) -> None:
        self.root = Path(root)
        self.trash = self.root.parent / "models-trash"
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _ARTIFACTS_ROOT
        #: .gmod 文件模型扫描器（None 时只有策略产物与内置 default）。
        self.gmod = gmod

    # --- 查询 -----------------------------------------------------------
    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        default_file = self.root / DEFAULT_ID / "model.json"
        if not default_file.is_file():
            self._write(DEFAULT_ID, _default_model())
        else:
            # G-624 迁移：老仓库里 default 的 name/description 还写着「全新初始化」，
            # 与现在「可持久化模型」的语义冲突；只改这两个展示字段，不动训练数据。
            data = self._read_file(default_file)
            fresh = _default_model()
            if isinstance(data, dict) and (
                data.get("name") == "default（全新初始化）"
                or data.get("description", "").startswith("不加载任何权重")
            ):
                data["name"] = fresh["name"]
                data["description"] = fresh["description"]
                self._write(DEFAULT_ID, data)
        if self.gmod is not None:
            self.gmod.ensure()

    def payload(self) -> dict:
        """``GET /api/models`` 的完整响应（含热扫描统计）。"""
        items = self.list()
        scan = self.gmod.last_scan if self.gmod is not None else {"scanned": 0, "invalid": []}
        return {
            "ok": True,
            "models": items,
            "scanned": scan.get("scanned", 0),
            "invalid": scan.get("invalid", []),
            "gmodDir": str(self.gmod.directory) if self.gmod is not None else None,
        }

    def reload_payload(self) -> dict:
        """``POST /api/models/reload``：强制重扫 + 报告新增/失败条目。"""
        result = (self.gmod.reload() if self.gmod is not None
                  else {"scanned": 0, "added": 0, "invalid": []})
        return {
            "ok": True,
            "scanned": result["scanned"],
            "added": result["added"],
            "invalid": result["invalid"],
            "models": self.list(),
            "gmodDir": str(self.gmod.directory) if self.gmod is not None else None,
        }

    def _gmod_items(self) -> list[dict]:
        if self.gmod is None:
            return []
        try:
            return self.gmod.scan().get("items") or []
        except (OSError, ValueError):
            return []

    def list(self) -> list[dict]:
        """策略产物 + 内置 default + .gmod 文件模型；同名 id 以文件名（gmod）为准。"""
        items: list[dict] = []
        for path in sorted(self.root.glob("*/model.json")):
            data = self._read_file(path)
            if data is None:
                continue
            items.append(self._public(path.parent.name, data))
        if not any(item["id"] == DEFAULT_ID for item in items):
            items.insert(0, self._public(DEFAULT_ID, _default_model()))
        items.sort(key=lambda item: (0 if item["id"] == DEFAULT_ID else 1, item["id"]))
        for item in items:
            # 直接用注册表内的路径，避免 id 与 .gmod 同名时误触发文件模型的缓存解包。
            # default 自 G-624 起也可能有 policy.npz（首次持久化训练写回后）。
            policy = self.root / item["id"] / "policy.npz"
            item["hasPolicy"] = bool(policy.is_file())
            item["sizeBytes"] = policy.stat().st_size if item["hasPolicy"] else 0
            item["valid"] = True
            item["error"] = None
            item["warning"] = None
            item["color"] = None
            item["kind"] = "policy"
            item["training"] = self.training_stats(item["id"])
            source = dict(item.get("source") or {})
            source["type"] = "builtin" if item["id"] == DEFAULT_ID else "policy"
            item["source"] = source

        # .gmod 与策略模型**并存**：列表 id 加 ``gmod:`` 前缀，同 id 时两行都在，
        # 只读副本行标 conflict=true + warning，绝不顶掉可持久化模型。
        policy_ids = {item["id"] for item in items}
        gmod_items: list[dict] = []
        for entry in self._gmod_items():
            item = dict(entry)
            stem = str(item.get("fileId") or item.get("id") or "").strip()
            item["fileId"] = stem
            item["id"] = f"{GMOD_PREFIX}{stem}"
            item["conflict"] = stem in policy_ids
            if item["conflict"]:
                note = "与可持久化模型同名，已并列为只读副本"
                item["warning"] = (item["warning"] + "；" + note) if item.get("warning") else note
            gmod_items.append(item)
        merged: list[dict] = items + gmod_items
        merged.sort(key=lambda item: (
            0 if item["id"] == DEFAULT_ID else 1,
            0 if item.get("kind") == "policy" else 1,
            item["id"],
        ))
        for item in merged:
            # gmod 的 training 来自包内 meta.json（G-626），旧包没有就是 None。
            item.setdefault("training", self.training_stats(item["id"])
                            if item.get("kind") != "gmod" else None)
        return merged

    def _decorate_policy(self, item: dict) -> dict:
        """给「策略产物 / 内置 default」补上与 gmod 条目一致的字段。"""
        item.setdefault("kind", "policy")
        item.setdefault("valid", True)
        item.setdefault("error", None)
        item.setdefault("warning", None)
        item.setdefault("color", None)
        item.setdefault("hasPolicy", False)
        item.setdefault("sizeBytes", 0)
        item.setdefault("training", self.training_stats(item["id"]))
        source = dict(item.get("source") or {})
        source["type"] = "builtin" if item["id"] == DEFAULT_ID else "policy"
        item["source"] = source
        return item

    def get(self, model_id: Any) -> dict:
        # ``gmod:<stem>`` 只指向文件模型（列表层用的明确引用）。
        stem = gmod_stem(model_id)
        if stem is not None:
            item = self.gmod.get(stem) if self.gmod is not None else None
            if item is None:
                raise NotFound(f"文件模型不存在: {stem}")
            return item
        ident = self._check_id(model_id)
        # 同名时 plain id **优先策略模型**；只有注册表里确实没有它时才回落到 .gmod
        # （兼容旧链接与用户手填的裸 id）。
        path = self.root / ident / "model.json"
        if path.is_file():
            data = self._read_file(path)
            if data is None:
                raise BadRequest(f"模型元数据损坏: {ident}")
            return self._decorate_policy(self._public(ident, data))
        if ident == DEFAULT_ID:
            return self._decorate_policy(self._public(DEFAULT_ID, _default_model()))
        if self.gmod is not None:
            item = self.gmod.get(ident)
            if item is not None:
                return item
        raise NotFound(f"模型不存在: {ident}")

    def detail(self, model_id: Any) -> dict:
        model = self.get(model_id)
        if model.get("kind") == "gmod":
            model["policyPath"] = None
            model["packagePath"] = str(
                self.gmod.directory / model["source"]["file"]
            ) if self.gmod is not None else None
            # run.json 里记的是文件 stem（也可能是旧的裸 id），用 stem 匹配。
            stem = model.get("fileId") or gmod_stem(model_id) or model["id"]
            model["recentRuns"] = self._recent_runs(stem)
            # 文件模型只读，但包内 meta.json 可能带导出时的训练累计（G-626）。
            model["training"] = model.get("training")
            return model
        policy = self.root / model["id"] / "policy.npz"
        model["hasPolicy"] = bool(policy.is_file())
        model["sizeBytes"] = policy.stat().st_size if model["hasPolicy"] else 0
        model["policyPath"] = str(policy) if model["hasPolicy"] else None
        model["recentRuns"] = self._recent_runs(model["id"])
        model["training"] = self.training_stats(model["id"])
        return model

    def policy_path(self, model_id: Any) -> Path | None:
        """训练侧 ``--init-policy`` 用的绝对路径。

        .gmod 会把包内 ``policy.npz`` 解到 ``var/gmod-cache/<sha>.npz``；
        策略产物（**含 default**）直接返回注册表里的副本路径——default 首次
        训练写回后这里就有文件，下次选它即接着上次继续练。
        """
        stem = gmod_stem(model_id)
        if stem is not None:
            if self.gmod is None:
                raise BadRequest("没有启用 .gmod 扫描器")
            return self.gmod.cache_policy(stem)
        ident = self._check_id(model_id)
        # 策略产物优先：同名 .gmod 不应改变可持久化模型的训练起点。
        path = self.root / ident / "policy.npz"
        if path.is_file():
            return path
        # 已注册的策略模型（含 default）即使还没写回权重，也不回落同名 .gmod：
        # 否则「全新模型」会被误认为已有权重（viz 会显示 .gmod 的直方图，
        # 首次持久化训练也会被 gmod 只读逻辑挡掉）。
        if (self.root / ident / "model.json").is_file():
            return path
        if self.gmod is not None and self.gmod.get(ident) is not None:
            return self.gmod.cache_policy(ident)
        return path

    def _recent_runs(self, model_id: str, limit: int = 5) -> list[dict]:
        """扫 artifacts/runs/*/run.json，找用过该模型的 run（新的在前）。"""
        runs_dir = self.artifacts_dir / "runs"
        if not runs_dir.is_dir():
            return []
        found: list[dict] = []
        try:
            entries = sorted(runs_dir.iterdir(), reverse=True)
        except OSError:
            return []
        for directory in entries:
            if not directory.is_dir():
                continue
            meta_path = directory / "run.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            if str(meta.get("modelId") or "") != model_id:
                continue
            found.append({
                "runId": directory.name,
                "modelName": meta.get("modelName"),
                "createdAt": meta.get("createdAt"),
                "kind": meta.get("kind"),
            })
            if len(found) >= limit:
                break
        return found

    # --- 持久化训练（G-624） ---------------------------------------------
    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def training_stats(self, model_id: Any) -> dict:
        """读 ``<id>/training.json``；不存在时返回全零结构（不抛异常）。

        ``gmod:<stem>`` 的文件模型没有 training.json，直接返回包里 meta 的快照。
        """
        stem = gmod_stem(model_id)
        if stem is not None:
            item = self.gmod.get(stem) if self.gmod is not None else None
            if item is None:
                raise NotFound(f"文件模型不存在: {stem}")
            snapshot = item.get("training") or {}
            return {
                "format": TRAINING_FORMAT,
                "modelId": f"{GMOD_PREFIX}{stem}",
                "lastRunId": None,
                "lastTrainedAt": snapshot.get("lastTrainedAt"),
                "totalEpisodes": self._as_int(snapshot.get("totalEpisodes")),
                "totalSteps": self._as_int(snapshot.get("totalSteps")),
                "lastMeanReward": snapshot.get("lastMeanReward"),
                "runs": [],
                "writeCount": self._as_int(snapshot.get("writeCount")),
            }
        ident = self._check_id(model_id)
        try:
            data = json.loads((self.root / ident / "training.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        runs = data.get("runs")
        return {
            "format": TRAINING_FORMAT,
            "modelId": ident,
            "lastRunId": data.get("lastRunId"),
            "lastTrainedAt": data.get("lastTrainedAt"),
            "totalEpisodes": self._as_int(data.get("totalEpisodes")),
            "totalSteps": self._as_int(data.get("totalSteps")),
            "lastMeanReward": data.get("lastMeanReward"),
            "runs": list(runs)[-20:] if isinstance(runs, list) else [],
            "writeCount": self._as_int(data.get("writeCount")),
        }

    def export_gmod(self, model_id: Any, *, overwrite: bool = False) -> dict:
        """把策略模型导出为 ``models/<id>.gmod``（权重 + 维度 + 训练统计，G-626）。

        包里**只有** ``meta.json`` 与 ``policy.npz``：不包含优化器动量、复盘明细与 run 目录。
        目标已存在且 ``overwrite=False`` 时 409；未知 id 404；没有权重 400。
        """
        if self.gmod is None:
            raise BadRequest(".gmod 目录未配置，无法导出")
        stem = gmod_stem(model_id)
        if stem is not None:
            raise BadRequest(f"{stem} 已经是 .gmod 文件模型，无需导出；可直接复制 models/{stem}.gmod")
        ident = self._check_id(model_id)
        # 只按「注册表里的策略模型」判断，不能被同名的 .gmod 抢走（否则无法覆盖导出）。
        model_json = self.root / ident / "model.json"
        if not model_json.is_file():
            raise NotFound(f"模型不存在: {ident}")
        policy = self.root / ident / "policy.npz"
        if not policy.is_file():
            raise BadRequest(
                f"模型 {ident} 没有 policy.npz（先做一次持久化训练写回，或从 run 导入后再导出）"
            )
        info = inspect_policy(policy)
        raw = self._read_file(model_json) or {}
        model = {"name": raw.get("name"), "description": raw.get("description"),
                 "createdAt": raw.get("createdAt")}
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        training = self.training_stats(ident)
        meta: dict[str, Any] = {
            "format": FORMAT,
            "id": ident,
            "name": str(raw.get("name") or model.get("name") or ident),
            "description": str(raw.get("description") or model.get("description") or ""),
            "obsDim": info["obsDim"],
            "actDim": info["numActions"],
            "hidden": info["hidden"],
            "version": 1,
            "createdAt": str(raw.get("createdAt") or model.get("createdAt") or now),
            "exportedAt": now,
            # policy.npz 自身的内容 sha256（包文件 sha256 另由响应返回）。
            "sha256": info["sha256"],
            "training": {
                "format": TRAINING_FORMAT,
                "totalEpisodes": training["totalEpisodes"],
                "totalSteps": training["totalSteps"],
                "lastMeanReward": training["lastMeanReward"],
                "lastTrainedAt": training["lastTrainedAt"],
                "writeCount": training["writeCount"],
                "runs": training["runs"][-20:],
            },
        }
        color = raw.get("color") or model.get("color")
        # 字段恒存在（没有就写 null），便于跨机器读取端无分支处理。
        meta["color"] = color.strip() if isinstance(color, str) and color.strip() else None
        preset = raw.get("rewardPreset")
        if preset:
            meta["rewardPreset"] = preset
        return self.gmod.write_package(ident, meta, policy, overwrite=overwrite)

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def write_back(self, model_id: Any, policy_file: Path | str, stats: dict) -> dict:
        """把一次**正常完成**的训练权重写回模型，返回 ``{ok, modelId, error?}``。

        原子性：先写 ``policy.npz.tmp`` 再 ``os.replace``；覆盖前把旧权重复制到
        ``models-trash/<id>-<时间戳>.npz``。任何失败都只体现在返回值里，绝不抛出
        （写回失败不能让 job 变 failed）。
        """
        ident = self._check_id(model_id)
        result: dict[str, Any] = {"ok": False, "modelId": ident}
        try:
            if self.gmod is not None and self.gmod.get(ident) is not None:
                raise BadRequest(".gmod 是只读文件包，请先「导入为策略模型」再持久化训练")
            source = Path(policy_file)
            if not source.is_file():
                raise BadRequest(f"训练产物不存在: {source}")
            info = inspect_policy(source)
            directory = self.root / ident
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "policy.npz"
            backup_path: str | None = None
            if target.is_file():
                self.trash.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup = self.trash / f"{ident}-{stamp}.npz"
                index = 2
                while backup.exists():
                    backup = self.trash / f"{ident}-{stamp}-{index}.npz"
                    index += 1
                shutil.copy2(target, backup)
                backup_path = str(backup)
            tmp = directory / "policy.npz.tmp"
            try:
                shutil.copy2(source, tmp)
                os.replace(tmp, target)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            episodes = self._as_int((stats or {}).get("episodes"))
            steps = self._as_int((stats or {}).get("steps"))
            mean_reward = (stats or {}).get("meanReward")
            run_id = str((stats or {}).get("runId") or "")
            training = self.training_stats(ident)
            training.update({
                "lastRunId": run_id or None,
                "lastTrainedAt": now,
                # 次数与步数必须累加，不是覆盖。
                "totalEpisodes": training["totalEpisodes"] + episodes,
                "totalSteps": training["totalSteps"] + steps,
                "lastMeanReward": mean_reward,
                "writeCount": training["writeCount"] + 1,
                "runs": (training["runs"] + [{
                    "runId": run_id, "at": now,
                    "episodes": episodes, "steps": steps, "meanReward": mean_reward,
                }])[-20:],
            })
            self._write_json_atomic(directory / "training.json", training)

            model_data = self._read_file(directory / "model.json")
            if model_data is None:
                model_data = _default_model() if ident == DEFAULT_ID else {
                    "format": FORMAT, "id": ident, "name": ident, "description": "",
                    "createdAt": now, "updatedAt": now,
                    "source": {"runId": None, "runName": None, "pair": None},
                    "obsDim": info["obsDim"], "numActions": info["numActions"],
                    "hidden": info["hidden"], "episodes": 0, "steps": 0,
                    "meanReward": None, "sha256": None, "note": "", "builtin": False,
                }
            model_data.update({
                "episodes": training["totalEpisodes"],
                "steps": training["totalSteps"],
                "meanReward": mean_reward,
                "obsDim": info["obsDim"],
                "numActions": info["numActions"],
                "hidden": info["hidden"],
                "sha256": info["sha256"],
                "updatedAt": now,
            })
            self._write(ident, model_data)
            result.update({"ok": True, "path": str(target), "backup": backup_path,
                           "training": training})
        except (OSError, BadRequest, ValueError, TypeError) as exc:
            result["error"] = f"模型写回失败: {exc}"
        return result

    # --- 变更 -----------------------------------------------------------
    def create(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise BadRequest("模型 name 不能为空")
        if len(name) > NAME_MAX:
            raise BadRequest(f"模型 name 最多 {NAME_MAX} 个字符")
        description = str(payload.get("description") or "").strip()[:DESCRIPTION_MAX]
        note = str(payload.get("note") or "").strip()[:NOTE_MAX]

        run_id = str(payload.get("sourceRunId") or "").strip()
        raw_path = str(payload.get("policyPath") or "").strip()
        gmod_id = str(payload.get("gmodId") or "").strip()
        given = [item for item in (run_id, raw_path, gmod_id) if item]
        if not given:
            raise BadRequest("需要 sourceRunId / policyPath / gmodId（从 run 导入 / 指定 npz / 导入 .gmod）")
        if len(given) > 1:
            raise BadRequest("sourceRunId / policyPath / gmodId 只能给一个")

        source: dict[str, Any] = {"runId": None, "runName": None, "pair": None}
        stats = {"episodes": 0, "steps": 0, "meanReward": None}
        if gmod_id:
            # 「导入为策略模型并继续训练」：把只读 .gmod 解出的 npz 复制成可写回的策略产物。
            if self.gmod is None:
                raise BadRequest("没有启用 .gmod 扫描器，无法导入文件模型")
            stem = gmod_stem(gmod_id) or gmod_id
            gmod_item = self.gmod.get(stem)
            if gmod_item is None:
                raise NotFound(f"文件模型不存在: {stem}")
            if not gmod_item.get("valid"):
                raise BadRequest(f"文件模型 {stem} 不可用：{gmod_item.get('error')}")
            policy_file = self.gmod.cache_policy(stem)
            source = {"runId": None, "runName": None, "pair": None,
                      "gmodId": f"{GMOD_PREFIX}{stem}", "gmodFile": gmod_item["source"]["file"]}
        elif run_id:
            policy_file, source, stats = self._policy_from_run(run_id, payload.get("pair"))
        else:
            policy_file = self._policy_from_path(raw_path)

        info = inspect_policy(policy_file)
        ident = self._unique_id(payload.get("id") or self._slug(
            (gmod_stem(gmod_id) or gmod_id) or run_id or Path(policy_file).parent.name))

        created = time.strftime("%Y-%m-%dT%H:%M:%S")
        data = {
            "format": FORMAT,
            "id": ident,
            "name": name,
            "description": description,
            "createdAt": created,
            "updatedAt": created,
            "source": source,
            "obsDim": info["obsDim"],
            "numActions": info["numActions"],
            "hidden": info["hidden"],
            "episodes": stats["episodes"],
            "steps": stats["steps"],
            "meanReward": stats["meanReward"],
            "sha256": info["sha256"],
            "note": note,
            "builtin": False,
        }
        directory = self.root / ident
        if directory.exists():
            self._backup(ident)
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(policy_file, directory / "policy.npz")
        except OSError as exc:
            raise BadRequest(f"复制策略文件失败: {exc}") from exc
        self._write(ident, data)
        return self.detail(ident)

    def update(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        raw_id = payload.get("id")
        stem = gmod_stem(raw_id)
        if stem is not None:
            raise Conflict(
                f".gmod 由文件系统管理，请修改 models/{stem}.gmod 的 meta.json 后刷新（不能在此改名）")
        ident = self._check_id(raw_id)
        if (not (self.root / ident / "model.json").is_file()
                and self.gmod is not None and self.gmod.get(ident) is not None):
            raise Conflict(".gmod 由文件系统管理，请修改包内 meta.json 后刷新（不能在此改名）")
        if ident == DEFAULT_ID:
            data = _default_model()
            data["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            path = self.root / ident / "model.json"
            if not path.is_file():
                raise NotFound(f"模型不存在: {ident}")
            loaded = self._read_file(path)
            if loaded is None:
                raise BadRequest(f"模型元数据损坏: {ident}")
            data = loaded

        if payload.get("name") is not None:
            name = str(payload.get("name") or "").strip()
            if not name:
                raise BadRequest("模型 name 不能为空")
            if len(name) > NAME_MAX:
                raise BadRequest(f"模型 name 最多 {NAME_MAX} 个字符")
            data["name"] = name
        if payload.get("description") is not None:
            data["description"] = str(payload.get("description") or "").strip()[:DESCRIPTION_MAX]
        if payload.get("note") is not None:
            data["note"] = str(payload.get("note") or "").strip()[:NOTE_MAX]
        data["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if ident != DEFAULT_ID:
            self._backup(ident)
        self._write(ident, data)
        return self.detail(ident)

    def delete(self, model_id: Any) -> str:
        stem = gmod_stem(model_id)
        if stem is not None:
            # 用户的文件不碰：只告诉它去哪里删。
            if self.gmod is None or self.gmod.get(stem) is None:
                raise NotFound(f"文件模型不存在: {stem}")
            raise Conflict(self.gmod.failure_message(stem))
        ident = self._check_id(model_id)
        if ident == DEFAULT_ID:
            raise BadRequest("内置 default 模型不能删除（它表示「全新初始化」）")
        directory = self.root / ident
        if not directory.is_dir():
            # 没有同名策略模型时，裸 id 可能指 .gmod（兼容旧调用）。
            if self.gmod is not None and self.gmod.get(ident) is not None:
                raise Conflict(self.gmod.failure_message(ident))
            raise NotFound(f"模型不存在: {ident}")
        self._backup(ident, move=True)
        return ident

    # --- 内部 -----------------------------------------------------------
    def _policy_from_run(self, run_id: str, pair: Any) -> tuple[Path, dict, dict]:
        from_artifacts = self.artifacts_dir / "runs" / str(run_id)
        if not re.match(r"^[A-Za-z0-9_.\-]{1,80}$", str(run_id)) or ".." in str(run_id):
            raise BadRequest("runId 非法")
        if not from_artifacts.is_dir():
            raise NotFound(f"运行不存在: {run_id}")

        candidates: list[Path] = []
        if pair not in (None, ""):
            try:
                index = int(pair)
            except (TypeError, ValueError) as exc:
                raise BadRequest("pair 需要整数（从 1 开始）") from exc
            candidates.append(from_artifacts / f"policy-pair{index}.npz")
        candidates.append(from_artifacts / "policy.npz")
        candidates.extend(sorted(from_artifacts.glob("policy-pair*.npz")))
        policy_file = next((item for item in candidates if item.is_file()), None)
        if policy_file is None:
            raise BadRequest(f"该 run 没有 policy.npz / policy-pairN.npz: {run_id}")

        run_name = run_id
        stats = {"episodes": 0, "steps": 0, "meanReward": None}
        report_path = from_artifacts / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        if isinstance(report, dict):
            summary = report.get("summary")
            if not isinstance(summary, dict):
                history = report.get("history") or []
                rewards = [float(item.get("totalReward") or 0) for item in history if isinstance(item, dict)]
                summary = {
                    "episodes": len(history),
                    "totalSteps": sum(int(item.get("steps") or 0) for item in history if isinstance(item, dict)),
                    "meanReward": (sum(rewards) / len(rewards)) if rewards else None,
                }
            stats = {
                "episodes": int(summary.get("episodes") or 0),
                "steps": int(summary.get("totalSteps") or 0),
                "meanReward": summary.get("meanReward"),
            }
        pair_value = None
        match = re.search(r"policy-pair(\d+)", policy_file.name)
        if match:
            pair_value = int(match.group(1))
        return policy_file, {"runId": str(run_id), "runName": run_name, "pair": pair_value}, stats

    def _policy_from_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise BadRequest("policyPath 需要绝对路径")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise BadRequest(f"policyPath 不存在: {raw_path}") from exc
        root = self.artifacts_dir.resolve()
        if root not in resolved.parents:
            raise BadRequest(f"policyPath 必须位于 training/artifacts 内: {resolved}")
        if resolved.suffix.lower() != ".npz":
            raise BadRequest("policyPath 需要 .npz 文件")
        return resolved

    def _unique_id(self, base: str) -> str:
        # ID_RE 上限 32 位：不要把常见的 runId（如 20261108-2200-train-bot_a，25 位）截断。
        limit = 32
        slug = re.sub(r"[^a-z0-9_-]+", "-", str(base).strip().lower()).strip("-_")[:limit] or "model"
        ident = slug
        index = 2
        while (self.root / ident).exists():
            suffix = f"-{index}"
            ident = slug[:limit - len(suffix)] + suffix
            index += 1
        return ident

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", str(text).strip().lower()).strip("-_") or "model"

    @staticmethod
    def _check_id(model_id: Any) -> str:
        ident = str(model_id or "").strip()
        if not ID_RE.match(ident):
            raise BadRequest("模型 id 需为 1–32 位小写字母/数字/下划线/连字符，且以字母或数字开头")
        return ident

    def _read_file(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write(self, ident: str, data: dict) -> None:
        directory = self.root / ident
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "model.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise BadRequest(f"写入模型元数据失败: {exc}") from exc

    def _backup(self, ident: str, *, move: bool = False) -> None:
        source = self.root / ident
        if not source.exists():
            return
        self.trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.trash / f"{ident}-{stamp}"
        index = 2
        while target.exists():
            target = self.trash / f"{ident}-{stamp}-{index}"
            index += 1
        try:
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copytree(source, target)
        except OSError as exc:
            raise BadRequest(f"备份模型失败: {exc}") from exc

    @staticmethod
    def _public(ident: str, data: dict) -> dict:
        return {
            "id": ident,
            "name": str(data.get("name") or ident),
            "description": str(data.get("description") or ""),
            "createdAt": data.get("createdAt"),
            "updatedAt": data.get("updatedAt"),
            "source": data.get("source") if isinstance(data.get("source"), dict) else {},
            "obsDim": data.get("obsDim"),
            "numActions": data.get("numActions"),
            "hidden": data.get("hidden"),
            "episodes": data.get("episodes"),
            "steps": data.get("steps"),
            "meanReward": data.get("meanReward"),
            "sha256": data.get("sha256"),
            "note": str(data.get("note") or ""),
            "builtin": ident == DEFAULT_ID or bool(data.get("builtin")),
            "format": str(data.get("format") or FORMAT),
        }
