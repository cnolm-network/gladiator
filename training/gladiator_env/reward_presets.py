"""奖励预设：把「基础权重 + 条件式附加项」打包成可分享的 JSON。

设计目标
--------
1. 基础权重（击杀/伤害/连击/威胁间距……）仍然是一组扁平的数字，便于直接调。
2. 需要「多条件」的塑形（例如：只有在 2.8~4.2 格、且刚命中 1.5 秒内才给间距奖励；
   只有连续 3 秒没动才扣站桩分）用 ``terms`` 表达，每条 term 自带一组条件，
   条件之间是 **AND** 关系，不同 term 之间自然叠加。

所以「多个条件判定」不需要写表达式引擎：一个 term = 一个条件对象 + 一个权重 + 一个计量单位。
需要「或」的时候写成两条 term 即可（它们各自独立结算）。

schema::

    {
      "format": "gladiator.reward-preset/v1",
      "name": "均衡",
      "description": "默认权重",
      "base": {"kill": 20.0, "death": -20.0, "damageDealt": 2.5, ...},
      "terms": [
        {"id": "threat", "weight": 0.3, "unit": "second",
         "when": {"distanceMin": 2.8, "distanceMax": 4.2}},
        {"id": "punishBlock", "weight": -0.2, "unit": "step",
         "when": {"actionIn": ["block"], "distanceMin": 4.2}}
      ]
    }
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA = "gladiator.reward-preset/v1"

#: 基础权重：预设里的键 -> ShapingWeights 的字段名。
BASE_FIELDS: dict[str, str] = {
    "kill": "kill",
    "death": "death",
    "damageDealt": "damage_dealt",
    "damageTaken": "damage_taken",
    "combo": "combo",
    "comboCap": "combo_cap",
    "threatWindow": "threat_window",
    "pressure": "pressure",
    "whiff": "whiff",
    "still": "still",
    "waste": "waste",
    "closing": "closing",
    "attackAttempt": "attack_attempt",
    "counter": "counter",
    "inactive": "inactive",
}

DEFAULT_BASE: dict[str, float] = {
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

#: term 支持的条件（全部 AND）。
CONDITION_KEYS = {
    "distanceMin",
    "distanceMax",
    "actionIn",
    "stillSecondsMin",
    "healthBelow",
    "opponentHealthBelow",
    "afterHitSeconds",
    "comboMin",
    "attacked",
}

UNITS = {"second", "step"}


class PresetError(ValueError):
    """预设不合法。"""


@dataclass(frozen=True)
class Term:
    """一条条件式奖励项。"""

    id: str
    weight: float
    unit: str
    when: dict[str, Any]

    def matches(self, ctx: dict[str, Any]) -> bool:
        for key, expected in self.when.items():
            if key == "distanceMin":
                if ctx.get("distance") is None or float(ctx["distance"]) < float(expected):
                    return False
            elif key == "distanceMax":
                if ctx.get("distance") is None or float(ctx["distance"]) > float(expected):
                    return False
            elif key == "actionIn":
                if str(ctx.get("action") or "") not in {str(v) for v in expected}:
                    return False
            elif key == "stillSecondsMin":
                if float(ctx.get("stillSeconds") or 0.0) < float(expected):
                    return False
            elif key == "healthBelow":
                if ctx.get("health") is None or float(ctx["health"]) >= float(expected):
                    return False
            elif key == "opponentHealthBelow":
                if ctx.get("opponentHealth") is None or float(ctx["opponentHealth"]) >= float(expected):
                    return False
            elif key == "afterHitSeconds":
                since = ctx.get("sinceHit")
                if since is None or float(since) > float(expected):
                    return False
            elif key == "comboMin":
                if float(ctx.get("combo") or 0.0) < float(expected):
                    return False
            elif key == "attacked":
                if bool(ctx.get("attacked")) is not bool(expected):
                    return False
        return True

    def evaluate(self, ctx: dict[str, Any], dt: float) -> float:
        if not self.matches(ctx):
            return 0.0
        if self.unit == "second":
            return self.weight * max(0.0, float(dt))
        return self.weight


def default_preset(name: str = "均衡（默认）") -> dict[str, Any]:
    """返回内置默认预设的副本。"""
    return {
        "format": SCHEMA,
        "name": name,
        "description": "默认权重：击杀/伤害为主，威胁间距与压迫位小幅引导，站桩扣分。",
        "base": dict(DEFAULT_BASE),
        "terms": [],
    }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresetError(f"{label} 必须是数字")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise PresetError(f"{label} 必须是有限数字")
    return number


def normalize_preset(raw: Any) -> dict[str, Any]:
    """校验并补全一个预设，非法时抛 :class:`PresetError`。"""
    if not isinstance(raw, dict):
        raise PresetError("预设必须是 JSON 对象")
    fmt = raw.get("format") or SCHEMA
    if fmt != SCHEMA:
        raise PresetError(f"不支持的预设格式：{fmt}（需要 {SCHEMA}）")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise PresetError("预设需要 name")
    if len(name) > 80:
        raise PresetError("预设 name 过长（最多 80 字）")

    base_raw = raw.get("base") or {}
    if not isinstance(base_raw, dict):
        raise PresetError("base 必须是对象")
    base = dict(DEFAULT_BASE)
    for key, value in base_raw.items():
        if key not in BASE_FIELDS:
            raise PresetError(f"未知的基础权重：{key}")
        base[key] = _number(value, f"base.{key}")

    terms_raw = raw.get("terms") or []
    if not isinstance(terms_raw, list):
        raise PresetError("terms 必须是数组")
    if len(terms_raw) > 50:
        raise PresetError("terms 最多 50 条")
    terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(terms_raw):
        if not isinstance(item, dict):
            raise PresetError(f"terms[{index}] 必须是对象")
        term_id = str(item.get("id") or "").strip()
        if not term_id:
            raise PresetError(f"terms[{index}] 需要 id")
        if term_id in seen:
            raise PresetError(f"terms 里 id 重复：{term_id}")
        seen.add(term_id)
        unit = str(item.get("unit") or "second")
        if unit not in UNITS:
            raise PresetError(f"terms[{index}].unit 只能是 {sorted(UNITS)}")
        when = item.get("when") or {}
        if not isinstance(when, dict):
            raise PresetError(f"terms[{index}].when 必须是对象")
        for key, value in when.items():
            if key not in CONDITION_KEYS:
                raise PresetError(f"terms[{index}].when 不支持的条件：{key}")
            if key == "actionIn":
                if not isinstance(value, list) or not value:
                    raise PresetError(f"terms[{index}].when.actionIn 必须是非空数组")
            elif key == "attacked":
                if not isinstance(value, bool):
                    raise PresetError(f"terms[{index}].when.attacked 必须是布尔")
            else:
                _number(value, f"terms[{index}].when.{key}")
        terms.append(
            {
                "id": term_id,
                "weight": _number(item.get("weight"), f"terms[{index}].weight"),
                "unit": unit,
                "when": dict(when),
            }
        )

    return {
        "format": SCHEMA,
        "name": name,
        "description": str(raw.get("description") or "").strip()[:500],
        "base": base,
        "terms": terms,
    }


def build_terms(raw_terms: Iterable[dict[str, Any]]) -> list[Term]:
    """把已校验的 term 字典转成可结算对象。"""
    return [Term(t["id"], float(t["weight"]), str(t["unit"]), dict(t["when"])) for t in raw_terms]


def load_preset(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return normalize_preset(json.load(handle))


def slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip().lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:48] or "preset"


class PresetStore:
    """一个目录下的 reward 预设集合；删除是软删除（移到 trash 子目录）。"""

    def __init__(self, directory: str, now=None) -> None:
        self.directory = directory
        self.trash = os.path.join(directory, "trash")
        self._now = now or time.time

    def _path(self, slug: str) -> str:
        if not slug or "/" in slug or "\\" in slug or ".." in slug:
            raise PresetError("非法的预设 id")
        return os.path.join(self.directory, slug + ".json")

    def ensure(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        # 首次运行写入一份内置默认预设，方便前端直接可用。
        if not any(name.endswith(".json") for name in os.listdir(self.directory)):
            self.save("default", default_preset())

    def list(self) -> list[dict[str, Any]]:
        os.makedirs(self.directory, exist_ok=True)
        items: list[dict[str, Any]] = []
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            try:
                preset = load_preset(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            items.append({"id": name[:-5], "mtime": os.path.getmtime(path), "preset": preset})
        return items

    def get(self, slug: str) -> dict[str, Any]:
        return load_preset(self._path(slug))

    def save(self, slug: str, raw: Any) -> dict[str, Any]:
        preset = normalize_preset(raw)
        os.makedirs(self.directory, exist_ok=True)
        path = self._path(slug)
        # 覆盖前先留一份备份，误保存可恢复。
        if os.path.exists(path):
            os.makedirs(self.trash, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
            os.replace(path, os.path.join(self.trash, f"{slug}-{stamp}.json"))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(preset, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
        return preset

    def delete(self, slug: str) -> bool:
        path = self._path(slug)
        if not os.path.exists(path):
            return False
        os.makedirs(self.trash, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
        os.replace(path, os.path.join(self.trash, f"{slug}-deleted-{stamp}.json"))
        return True


def preset_for_env(preset: dict[str, Any] | None) -> tuple[dict[str, float], list[Term]]:
    """把预设拆成 ``(ShapingWeights 关键字参数, terms)``。"""
    normalized = normalize_preset(preset) if preset is not None else default_preset()
    base = copy.deepcopy(normalized["base"])
    kwargs = {BASE_FIELDS[key]: float(value) for key, value in base.items() if key in BASE_FIELDS}
    return kwargs, build_terms(normalized["terms"])
