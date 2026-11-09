""".gmod 模型包的热扫描（G-623）。

一个 ``.gmod`` 就是 zip 包：

    rush-v3.gmod
    ├─ meta.json     # id / name / obsDim / actDim（必填），color / version / description / createdAt（可选）
    └─ policy.npz    # 与 training/gladiator_train/policy.py 保存的权重同构

设计要点：

- **放文件即出现**：``scan()`` 每次遍历 ``models/*.gmod``（``GET /api/models`` 每次调用），
  用 ``(mtime_ns, size)`` 做缓存，文件换了才重新解析；文件删除后自动消失。
- **单条失败不打断列表**：坏包（缺 meta / 缺 npz / 维度不符 / 损坏 zip）也会出现在列表里，
  标 ``valid=false`` + 中文 ``error``，同时汇总进 ``invalid``。
- **不删用户文件**：删除交给文件系统；训练选用时把包内 ``policy.npz`` 解到
  ``dashboard/var/gmod-cache/<policy 内容 sha256>.npz``，按内容复用。
- 不依赖 numpy：npz 是 zip，只解析每个 ``.npy`` 的 header 拿 shape。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from errors import BadRequest, Conflict
from models import ID_RE, REQUIRED_ARRAYS, _npy_header, _sha256, spaces_dimensions

META_NAME = "meta.json"
POLICY_NAME = "policy.npz"
FORMAT = "gladiator.model/v1"
COLOR_RE = re.compile(r"^[&§][0-9a-fA-F]$")
ALLOWED_MEMBERS = {META_NAME, POLICY_NAME}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(text).strip().lower()).strip("-_")


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _public(item: dict) -> dict:
    """去掉扫描器内部字段，避免把大对象/绝对路径细节直接塞进 JSON。"""
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_training(raw: dict) -> dict:
    """把包内 ``training`` 收敛成稳定结构（缺字段补默认，runs 只留最近 20 条）。"""
    runs = raw.get("runs")
    return {
        "format": str(raw.get("format") or "gladiator.model-training/v1"),
        "totalEpisodes": _as_int(raw.get("totalEpisodes")),
        "totalSteps": _as_int(raw.get("totalSteps")),
        "lastMeanReward": raw.get("lastMeanReward"),
        "lastTrainedAt": raw.get("lastTrainedAt"),
        "writeCount": _as_int(raw.get("writeCount")),
        "runs": list(runs)[-20:] if isinstance(runs, list) else [],
    }


class GmodScanner:
    """扫描一个目录里的 ``*.gmod``；``cache_dir`` 用于解出 policy.npz。"""

    def __init__(self, directory: Path | str, cache_dir: Path | str,
                 *, expected: tuple[int, int] | None = None) -> None:
        self.directory = Path(directory)
        self.cache_dir = Path(cache_dir)
        self.expected_obs, self.expected_actions = expected or spaces_dimensions()
        #: 文件名 → ((mtime_ns, size), 条目)。只缓存解析结果，不缓存文件内容。
        self._cache: dict[str, tuple[tuple[int, int], dict]] = {}
        self._by_id: dict[str, dict] = {}
        self._last_ids: set[str] = set()
        self.last_scan: dict = {"scanned": 0, "invalid": []}

    # --- 目录 -----------------------------------------------------------
    def ensure(self) -> None:
        """目录不存在时创建，并放一个 .gitkeep 让空目录能进版本库。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        keep = self.directory / ".gitkeep"
        if not keep.is_file():
            try:
                keep.write_text("", encoding="utf-8")
            except OSError:
                pass

    # --- 扫描 -----------------------------------------------------------
    def scan(self) -> dict:
        files: list[Path] = []
        if self.directory.is_dir():
            try:
                files = sorted(self.directory.glob("*.gmod"))
            except OSError:
                files = []
        items: list[dict] = []
        invalid: list[dict] = []
        present: set[str] = set()
        for path in files:
            present.add(path.name)
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (int(stat.st_mtime_ns), int(stat.st_size))
            cached = self._cache.get(path.name)
            if cached is not None and cached[0] == key:
                item = dict(cached[1])
            else:
                item = self._inspect(path, key)
                self._cache[path.name] = (key, dict(item))
            items.append(_public(item))
            if not item.get("valid"):
                invalid.append({"file": path.name, "error": str(item.get("error") or "未知错误")})
        # 文件被删掉后，缓存也要一起消失。
        for name in list(self._cache):
            if name not in present:
                del self._cache[name]
        self._by_id = {item["id"]: item for item in items}
        self.last_scan = {"scanned": len(files), "invalid": invalid}
        return {"items": items, "invalid": invalid, "scanned": len(files)}

    def get(self, ident: str) -> dict | None:
        """按 id 取最新扫描到的条目（不存在返回 None）；会先刷新一次目录。"""
        self.scan()
        item = self._by_id.get(ident)
        return _public(dict(item)) if item is not None else None

    def reload(self) -> dict:
        """强制重扫（清解析缓存），返回本次相对上次 reload 新增的条目数。"""
        self._cache.clear()
        result = self.scan()
        ids = {item["id"] for item in result["items"]}
        added = len(ids - self._last_ids)
        self._last_ids = ids
        return {
            "scanned": result["scanned"],
            "added": added,
            "invalid": result["invalid"],
            "items": result["items"],
        }

    # --- 单包解析 -------------------------------------------------------
    def _base(self, path: Path, key: tuple[int, int]) -> dict:
        ident = _slug(path.stem)
        return {
            "id": ident,
            #: 文件本身的 id（stem）。列表层的 ``id`` 会加 ``gmod:`` 前缀避免与策略模型冲突。
            "fileId": ident,
            "name": ident,
            "kind": "gmod",
            "builtin": False,
            "valid": False,
            "error": None,
            "warning": None,
            "source": {"type": "gmod", "file": path.name, "dir": self.directory.name},
            "sizeBytes": key[1],
            "sha256": None,
            "color": None,
            "obsDim": None,
            "numActions": None,
            "hidden": None,
            "episodes": 0,
            "steps": 0,
            "meanReward": None,
            "description": "",
            "version": None,
            "createdAt": None,
            "updatedAt": _iso(key[0] / 1_000_000_000),
            "hasPolicy": False,
            "format": FORMAT,
            #: 包里 meta.json 携带的训练累计（G-626）；旧包没有则为 None。
            "training": None,
            "rewardPreset": None,
        }

    def _invalid(self, item: dict, reason: str) -> dict:
        item["valid"] = False
        item["error"] = reason
        return item

    def _inspect(self, path: Path, key: tuple[int, int]) -> dict:
        item = self._base(path, key)
        if not ID_RE.match(item["id"]):
            return self._invalid(
                item,
                f"文件名 {path.name!r} 不能作为模型 id（需 1–32 位小写字母/数字/下划线/连字符，且以字母或数字开头）",
            )
        try:
            item["sha256"] = _sha256(path)
        except OSError as exc:
            return self._invalid(item, f"无法读取 .gmod 文件: {exc}")

        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                extra = sorted(name for name in names if name not in ALLOWED_MEMBERS
                               and not name.startswith("__MACOSX"))
                if META_NAME not in names:
                    return self._invalid(item, f".gmod 缺少 {META_NAME}")
                if POLICY_NAME not in names:
                    return self._invalid(item, f".gmod 缺少 {POLICY_NAME}")
                try:
                    meta = json.loads(archive.read(META_NAME).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    return self._invalid(item, f"{META_NAME} 不是合法 JSON: {exc}")
                policy_blob = archive.read(POLICY_NAME)
        except zipfile.BadZipFile as exc:
            return self._invalid(item, f"不是合法的 .gmod（zip）包: {exc}")
        except (OSError, KeyError, RuntimeError) as exc:
            return self._invalid(item, f"读取 .gmod 失败: {exc}")

        if not isinstance(meta, dict):
            return self._invalid(item, f"{META_NAME} 需要 JSON 对象")
        error = self._check_meta(item, meta)
        if error:
            return self._invalid(item, error)
        error = self._check_policy(item, policy_blob)
        if error:
            return self._invalid(item, error)
        if extra:
            item["warning"] = ".gmod 含额外文件（已忽略）: " + ", ".join(extra[:4])
        meta_id = str(meta.get("id"))
        if meta_id != item["id"]:
            note = f"meta.json 的 id={meta_id!r} 与文件名不一致，已按文件名 {item['id']} 作为 id"
            item["warning"] = (item["warning"] + "；" + note) if item["warning"] else note
        item["valid"] = True
        item["error"] = None
        item["hasPolicy"] = True
        return item

    def _check_meta(self, item: dict, meta: dict) -> str | None:
        ident = str(meta.get("id") or "").strip()
        if not ID_RE.match(ident):
            return "meta.json 的 id 需为 1–32 位小写字母/数字/下划线/连字符，且以字母或数字开头"
        name = str(meta.get("name") or "").strip()
        if not name:
            return "meta.json 的 name 不能为空"
        if len(name) > 40:
            return "meta.json 的 name 最多 40 个字符"
        obs = meta.get("obsDim")
        act = meta.get("actDim")
        if not isinstance(obs, int) or isinstance(obs, bool) or not isinstance(act, int) or isinstance(act, bool):
            return "meta.json 的 obsDim / actDim 必须是整数"
        if obs != self.expected_obs or act != self.expected_actions:
            return (f"meta.json 维度不符：obsDim={obs}（应为 {self.expected_obs}）、"
                    f"actDim={act}（应为 {self.expected_actions}）")
        color = meta.get("color")
        if color is not None and color != "":
            text = str(color).strip()
            if not COLOR_RE.match(text):
                return "meta.json 的 color 只允许 &0–&f / §0–§f 单色码"
            item["color"] = "§" + text[-1].lower()
        item["id"] = _slug(item["id"])  # 与文件名保持一致
        item["name"] = name
        item["obsDim"] = obs
        item["numActions"] = act
        item["description"] = str(meta.get("description") or meta.get("notes") or "")[:200]
        item["version"] = str(meta.get("version") or "")[:40] or None
        item["createdAt"] = str(meta.get("createdAt") or "")[:40] or None
        fmt = meta.get("format")
        if fmt and str(fmt) != FORMAT:
            item["warning"] = f"meta.json 的 format={fmt!r} 与 {FORMAT} 不一致（按 {FORMAT} 解析）"
        # G-626：导出的包带训练累计；解析后让列表/详情显示局数与均值，缺省保持「无文件」语义。
        training = meta.get("training")
        if isinstance(training, dict):
            summary = _normalize_training(training)
            item["training"] = summary
            item["episodes"] = summary["totalEpisodes"]
            item["steps"] = summary["totalSteps"]
            item["meanReward"] = summary["lastMeanReward"]
        preset = meta.get("rewardPreset")
        if isinstance(preset, (dict, str)) and preset:
            item["rewardPreset"] = preset
        return None

    def _check_policy(self, item: dict, blob: bytes) -> str | None:
        shapes: dict[str, tuple[int, ...]] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as npz:
                names = set(npz.namelist())
                missing = [key for key in REQUIRED_ARRAYS if f"{key}.npy" not in names]
                if missing:
                    return f"policy.npz 缺少参数数组: {', '.join(missing)}（需要 W1/b1/W2/b2）"
                if "_meta.npy" not in names:
                    return "policy.npz 缺少 _meta.npy，训练侧 Policy.load() 无法加载"
                with npz.open("_meta.npy") as handle:
                    _npy_header(handle.read())
                for key in REQUIRED_ARRAYS:
                    with npz.open(f"{key}.npy") as handle:
                        shapes[key], _descr = _npy_header(handle.read())
        except zipfile.BadZipFile:
            return "policy.npz 不是合法的 npz（zip）归档"
        except (OSError, KeyError, RuntimeError, BadRequest) as exc:
            return f"policy.npz 解析失败: {exc}"

        w1, b1, w2, b2 = (shapes[key] for key in REQUIRED_ARRAYS)
        if len(w1) != 2 or len(w2) != 2 or len(b1) != 1 or len(b2) != 1:
            return "policy.npz 参数形状非法：W1/W2 应为二维、b1/b2 应为一维"
        hidden, obs = w1
        act, hidden2 = w2
        if hidden2 != hidden or b1[0] != hidden or b2[0] != act:
            return "policy.npz 参数形状不自洽（hidden / 动作数不一致）"
        if obs != item["obsDim"] or act != item["numActions"]:
            return (f"policy.npz 形状与 meta.json 不一致：npz obsDim={obs}/actDim={act}，"
                    f"meta obsDim={item['obsDim']}/actDim={item['numActions']}")
        item["hidden"] = int(hidden)
        return None

    # --- 导出 -----------------------------------------------------------
    def write_package(self, ident: str, meta: dict, policy_file: Path | str,
                      *, overwrite: bool = False) -> dict:
        """把 ``meta`` + ``policy.npz`` 打成 ``<dir>/<id>.gmod``（原子替换）。

        包内只允许这两个成员；覆盖前写临时文件，``os.replace`` 保证不会留下半个包。
        """
        target = self.directory / f"{ident}.gmod"
        if target.exists() and not overwrite:
            raise Conflict(f"目标文件已存在：{target.name}；如需覆盖请传 overwrite=true")
        self.ensure()
        tmp = target.with_name(target.name + ".tmp")
        policy = Path(policy_file)
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(META_NAME,
                                 json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
                archive.write(policy, POLICY_NAME)
            os.replace(tmp, target)
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise BadRequest(f"写出 .gmod 失败: {exc}") from exc
        # 立刻失效缓存，保证下一次 GET /api/models 看到新包。
        self._cache.pop(target.name, None)
        stat = target.stat()
        return {
            "path": str(target),
            "relativePath": f"{self.directory.name}/{target.name}",
            "id": ident,
            "sha256": _sha256(target),
            "sizeBytes": int(stat.st_size),
        }

    # --- 训练用缓存 -----------------------------------------------------
    def cache_policy(self, ident: str) -> Path:
        """把包内 policy.npz 解到 ``gmod-cache/<policy sha256>.npz`` 并返回绝对路径。"""
        item = self._by_id.get(ident)
        if item is None:
            raise BadRequest(f"文件模型不存在: {ident}（请刷新模型列表）")
        if not item.get("valid"):
            raise BadRequest(f"文件模型 {ident} 不可用：{item.get('error')}")
        path = self.directory / str(item["source"]["file"])
        try:
            with zipfile.ZipFile(path) as archive:
                blob = archive.read(POLICY_NAME)
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise BadRequest(f"读取 {item['source']['file']} 内的 policy.npz 失败: {exc}") from exc
        digest = hashlib.sha256(blob).hexdigest()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BadRequest(f"创建 gmod-cache 失败: {exc}") from exc
        target = self.cache_dir / f"{digest}.npz"
        if not target.is_file():
            tmp = target.with_name(target.name + ".tmp")
            try:
                tmp.write_bytes(blob)
                os.replace(tmp, target)
            except OSError as exc:
                raise BadRequest(f"写入 gmod-cache 失败: {exc}") from exc
        return target.resolve()

    def failure_message(self, ident: str) -> str:
        item = self._by_id.get(ident) or {}
        return f".gmod 由文件系统管理，请删除 {self.directory.name}/{item.get('source', {}).get('file', ident + '.gmod')}"
