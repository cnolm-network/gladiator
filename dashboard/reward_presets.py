"""Reward 预设（``gladiator.reward-preset/v1``）的校验与本地存储。

权威 schema 在 ``training/gladiator_env/reward_presets.py``；本模块**只读引用**它的常量
（``SCHEMA`` / ``BASE_FIELDS`` / ``DEFAULT_BASE`` / ``CONDITION_KEYS`` / ``UNITS``），
不再手抄一份，避免训练侧与 Dashboard 侧规则漂移。源码不在同一台机器时退回内置副本。

目录布局（不随 git 提交）::

    dashboard/var/reward-presets/
        default.json
        trash/
            my-preset-20261108-204500.json      # 删除
            my-preset-overwrite-20261108-204800.json  # 覆盖保存前的旧版本

校验规则与训练侧 :func:`gladiator_env.reward_presets.normalize_preset` 等价：
未知基础权重键、未知条件键、非法 unit、重复 term id、非有限数字全部拒绝。
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from config import ROOT
from errors import BadRequest, NotFound

# --- 只读引用训练侧 schema ------------------------------------------------

_FALLBACK_BASE = (
    "kill",
    "death",
    "damageDealt",
    "damageTaken",
    "combo",
    "comboCap",
    "threatWindow",
    "pressure",
    "whiff",
    "still",
    "waste",
    "closing",
    "attackAttempt",
    "counter",
    "inactive",
)

_FALLBACK_DEFAULT = {
    "kill": 20.0,
    "death": -20.0,
    "damageDealt": 2.5,
    "damageTaken": -1.2,
    "combo": 0.6,
    "comboCap": 3.0,
    "threatWindow": 0.3,
    "pressure": 0.05,
    "whiff": -0.1,
    "still": -0.1,
    "waste": -0.08,
    "closing": 0.02,
    "attackAttempt": 0.05,
    "counter": 0.15,
    "inactive": 0.0,
}

_FALLBACK_CONDITIONS = (
    "distanceMin",
    "distanceMax",
    "actionIn",
    "stillSecondsMin",
    "healthBelow",
    "opponentHealthBelow",
    "afterHitSeconds",
    "comboMin",
    "attacked",
)

SCHEMA_FILE = Path(ROOT) / "training" / "gladiator_env" / "reward_presets.py"


def _load_schema_source() -> ModuleType | None:
    """按文件路径加载训练侧 schema（不触发 ``gladiator_env`` 包的重量级导入）。"""
    if not SCHEMA_FILE.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_gladiator_reward_preset_ref", SCHEMA_FILE)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # dataclass() 会在装饰期回查 sys.modules[cls.__module__]，未注册会直接抛
        # AttributeError：必须先登记再 exec。
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            return None
        return module
    except Exception:  # noqa: BLE001 - schema 文件异常时退回内置副本，不影响 Dashboard 启动
        return None


_REF = _load_schema_source()

SCHEMA: str = getattr(_REF, "SCHEMA", "gladiator.reward-preset/v1")
BASE_ORDER: tuple[str, ...] = tuple(getattr(_REF, "BASE_FIELDS", None) or _FALLBACK_BASE)
DEFAULT_BASE: dict[str, float] = dict(getattr(_REF, "DEFAULT_BASE", None) or _FALLBACK_DEFAULT)
CONDITION_KEYS: tuple[str, ...] = tuple(sorted(getattr(_REF, "CONDITION_KEYS", None) or _FALLBACK_CONDITIONS))
UNITS: tuple[str, ...] = tuple(sorted(getattr(_REF, "UNITS", None) or ("second", "step")))

#: 条件键的中文说明，供前端直接渲染（未列出的键不显示说明）。
CONDITION_LABELS: dict[str, str] = {
    "distanceMin": "距离 ≥",
    "distanceMax": "距离 ≤",
    "actionIn": "动作属于",
    "stillSecondsMin": "静止 ≥ 秒",
    "healthBelow": "自身血量 <",
    "opponentHealthBelow": "对手血量 <",
    "afterHitSeconds": "命中后 ≤ 秒",
    "comboMin": "连击 ≥",
    "attacked": "被攻击",
}

BASE_LABELS: dict[str, str] = {
    "kill": "击杀",
    "death": "阵亡",
    "damageDealt": "造成伤害",
    "damageTaken": "受到伤害",
    "combo": "连击",
    "comboCap": "连击上限",
    "threatWindow": "威胁间距窗口",
    "pressure": "贴身压迫",
    "whiff": "空挥",
    "still": "站桩",
    "waste": "无效动作",
    "closing": "拉近距离",
    "attackAttempt": "出手尝试",
    "counter": "反击",
    "inactive": "消极",
}

MAX_TERMS = 50
NAME_MAX = 80
DESCRIPTION_MAX = 500
#: 与 scripts 的 id 规则同形但长度不同（32 位上限）。
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
DEFAULT_ID = "default"


def default_preset(name: str = "均衡（默认）") -> dict[str, Any]:
    """内置默认预设的副本（与训练侧 ``default_preset()`` 同内容）。"""
    return {
        "format": SCHEMA,
        "name": name,
        "description": "默认权重：击杀/伤害为主，威胁间距与压迫位小幅引导，站桩扣分。",
        "base": dict(DEFAULT_BASE),
        "terms": [],
    }


def schema_document() -> dict:
    """暴露给前端的 schema 元数据：编辑 UI 按它渲染，不在前端硬编码键名。"""
    return {
        "format": SCHEMA,
        "baseKeys": [
            {"key": key, "label": BASE_LABELS.get(key, key), "default": float(DEFAULT_BASE.get(key, 0.0))}
            for key in BASE_ORDER
        ],
        "units": list(UNITS),
        "conditionKeys": [
            {"key": key, "label": CONDITION_LABELS.get(key, key)} for key in CONDITION_KEYS
        ],
        "limits": {"maxTerms": MAX_TERMS, "nameMax": NAME_MAX, "descriptionMax": DESCRIPTION_MAX},
        "defaultId": DEFAULT_ID,
    }


# --- 校验 ---------------------------------------------------------------


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(f"{label} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise BadRequest(f"{label} 必须是有限数字")
    return number


def _action_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BadRequest(f"{label} 必须是非空数组")
    items: list[str] = []
    for element in value:
        if isinstance(element, (dict, list)):
            raise BadRequest(f"{label} 的元素必须是字符串")
        text = str(element).strip()
        if not text:
            raise BadRequest(f"{label} 不能包含空字符串")
        items.append(text)
    return items


def normalize_preset(raw: Any) -> dict[str, Any]:
    """校验并补全预设；非法时抛 :class:`BadRequest`（中文原因）。"""
    if not isinstance(raw, dict):
        raise BadRequest("预设必须是 JSON 对象")

    fmt = raw.get("format") or SCHEMA
    if fmt != SCHEMA:
        raise BadRequest(f"不支持的预设格式：{fmt}（需要 {SCHEMA}）")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise BadRequest("预设需要 name")
    if len(name) > NAME_MAX:
        raise BadRequest(f"预设 name 过长（最多 {NAME_MAX} 字）")

    base_raw = raw.get("base") or {}
    if not isinstance(base_raw, dict):
        raise BadRequest("base 必须是对象")
    base = dict(DEFAULT_BASE)
    for key, value in base_raw.items():
        if key not in BASE_ORDER:
            raise BadRequest(f"未知的基础权重：{key}（可选: {', '.join(BASE_ORDER)}）")
        base[key] = _number(value, f"base.{key}")

    terms_raw = raw.get("terms") or []
    if not isinstance(terms_raw, list):
        raise BadRequest("terms 必须是数组")
    if len(terms_raw) > MAX_TERMS:
        raise BadRequest(f"terms 最多 {MAX_TERMS} 条，当前 {len(terms_raw)}")

    terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(terms_raw):
        if not isinstance(item, dict):
            raise BadRequest(f"terms[{index}] 必须是对象")
        term_id = str(item.get("id") or "").strip()
        if not term_id:
            raise BadRequest(f"terms[{index}] 需要 id")
        if term_id in seen:
            raise BadRequest(f"terms 里 id 重复：{term_id}")
        seen.add(term_id)

        unit = str(item.get("unit") or "second").strip()
        if unit not in UNITS:
            raise BadRequest(f"terms[{index}].unit 只能是 {' / '.join(UNITS)}，收到 {unit!r}")

        when = item.get("when") or {}
        if not isinstance(when, dict):
            raise BadRequest(f"terms[{index}].when 必须是对象")
        clean_when: dict[str, Any] = {}
        for key, value in when.items():
            if key not in CONDITION_KEYS:
                raise BadRequest(
                    f"terms[{index}].when 不支持的条件：{key}（可选: {', '.join(CONDITION_KEYS)}）"
                )
            label = f"terms[{index}].when.{key}"
            if key == "actionIn":
                clean_when[key] = _action_list(value, label)
            elif key == "attacked":
                if not isinstance(value, bool):
                    raise BadRequest(f"{label} 必须是布尔")
                clean_when[key] = value
            else:
                clean_when[key] = _number(value, label)

        terms.append({
            "id": term_id,
            "weight": _number(item.get("weight"), f"terms[{index}].weight"),
            "unit": unit,
            "when": clean_when,
        })

    return {
        "format": SCHEMA,
        "name": name,
        "description": str(raw.get("description") or "").strip()[:DESCRIPTION_MAX],
        "base": base,
        "terms": terms,
    }


# --- 存储 ---------------------------------------------------------------


class RewardPresetStore:
    """``dashboard/var/reward-presets`` 下的预设集合；删除/覆盖都保留可恢复副本。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.trash = self.root / "trash"

    # --- 查询 -----------------------------------------------------------
    def schema(self) -> dict:
        """编辑 UI 需要的 schema 元数据（基础权重键、条件键、单位、上限）。"""
        return schema_document()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not any(path.suffix == ".json" for path in self.root.glob("*.json")):
            self.save(DEFAULT_ID, default_preset(), allow_overwrite=False)

    def list(self) -> list[dict]:
        items: list[dict] = []
        if not self.root.is_dir():
            return items
        for path in self.root.glob("*.json"):
            # 文件名就是 id：非法 id 直接跳过（GET 它会得到中文 400，而不是崩溃）。
            if not ID_RE.match(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                preset = normalize_preset(data)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            items.append(self._summary(path.stem, preset, path))
        items.sort(key=lambda item: (item["id"] != DEFAULT_ID, item["id"]))
        return items

    def get(self, preset_id: Any) -> dict:
        ident = self._check_id(preset_id)
        path = self._path(ident)
        if not path.is_file():
            raise NotFound(f"reward 预设不存在: {ident}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BadRequest(f"reward 预设文件损坏: {ident}") from exc
        return normalize_preset(data)

    def detail(self, preset_id: Any) -> dict:
        ident = self._check_id(preset_id)
        preset = self.get(ident)
        return {"id": ident, "path": str(self._path(ident)), "preset": preset}

    def path_for(self, preset_id: Any) -> str:
        """``--reward-preset`` 需要的绝对路径。"""
        ident = self._check_id(preset_id)
        if not self._path(ident).is_file():
            raise NotFound(f"reward 预设不存在: {ident}")
        return str(self._path(ident))

    # --- 变更 -----------------------------------------------------------
    def save(self, preset_id: Any, raw: Any, *, allow_overwrite: bool = True) -> dict:
        ident = self._check_id(preset_id)
        preset = normalize_preset(raw)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(ident)
        if path.is_file():
            if not allow_overwrite:
                return preset
            self._backup(path, f"{ident}-overwrite")
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(preset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise BadRequest(f"写入 reward 预设失败: {exc}") from exc
        return preset

    def delete(self, preset_id: Any) -> str:
        ident = self._check_id(preset_id)
        path = self._path(ident)
        if not path.is_file():
            raise NotFound(f"reward 预设不存在: {ident}")
        self._backup(path, f"{ident}-deleted", move=True)
        return ident

    # --- 内部 -----------------------------------------------------------
    def _backup(self, path: Path, stem: str, *, move: bool = False) -> None:
        """覆盖/删除前的可恢复副本：``trash/<stem>-<时间戳>.json``。"""
        self.trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.trash / f"{stem}-{stamp}.json"
        index = 2
        while target.exists():
            target = self.trash / f"{stem}-{stamp}-{index}.json"
            index += 1
        try:
            if move:
                os.replace(path, target)
            else:
                shutil.copy2(path, target)
        except OSError as exc:
            raise BadRequest(f"备份 reward 预设失败: {exc}") from exc

    @staticmethod
    def _check_id(preset_id: Any) -> str:
        ident = str(preset_id or "").strip()
        if not ID_RE.match(ident):
            raise BadRequest(
                "reward 预设 id 需为 1–32 位小写字母/数字/下划线/连字符，且以字母或数字开头；"
                f"收到 {ident!r}"
            )
        return ident

    def _path(self, ident: str) -> Path:
        return self.root / f"{ident}.json"

    def _summary(self, ident: str, preset: dict, path: Path) -> dict:
        base = preset.get("base") or {}
        try:
            modified = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
        except OSError:
            modified = None
        return {
            "id": ident,
            "name": preset.get("name") or ident,
            "description": preset.get("description") or "",
            "termsCount": len(preset.get("terms") or []),
            "baseSummary": {
                "kill": base.get("kill"),
                "death": base.get("death"),
                "damageDealt": base.get("damageDealt"),
                "damageTaken": base.get("damageTaken"),
            },
            "builtin": ident == DEFAULT_ID,
            "updatedAt": modified,
        }
