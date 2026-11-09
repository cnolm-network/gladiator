"""脚本 / 模型注册表：内置 6 个代码驱动脚本 + 用户自定义项（含显示颜色）。

schema（``dashboard/var/scripts.json``）::

    {
      "format": "gladiator.scripts/v1",
      "updatedAt": "2026-11-08T21:20:00",
      "items": [
        {"id": "rusher", "name": "Rusher", "color": "§c", "description": "高压近身"},
        {"id": "mine-hold", "name": "Hold Center", "color": "§b", "description": "我的脚本"}
      ]
    }

- 内置项（``BUILTIN_SCRIPTS``）永远存在，可以改 ``color`` / ``name`` / ``description``，**不可删除**；
- 自定义项可增删改；
- 写入前把旧文件备份到 ``dashboard/var/scripts-trash/scripts-<时间戳>.json``；
- 颜色只接受单个 Minecraft 颜色码（``&0``–``&f`` 或 ``§0``–``§f``），统一存成 ``§x``。

颜色用途：Dashboard 显示 + 训练前通过 RCON ``caller tagcolor <role> <color>`` 同步到
服务端标签（见 ``docs/scripting.md`` §7）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from errors import BadRequest, NotFound

FORMAT = "gladiator.scripts/v1"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
COLOR_RE = re.compile(r"^[&§][0-9a-fA-F]$")
NAME_MAX = 40
DESCRIPTION_MAX = 200

#: 内置脚本：id 必须与训练侧 ``OPPONENT_MODES`` 一致。
BUILTIN_SCRIPTS: tuple[dict[str, str], ...] = (
    {"id": "random", "name": "Random", "color": "§7", "description": "随机动作"},
    {"id": "wander", "name": "Wander", "color": "§a", "description": "随机游走"},
    {"id": "idle", "name": "Idle", "color": "§8", "description": "站立不动，只转身"},
    {"id": "scripted", "name": "Scripted", "color": "§e", "description": "固定套路（接近—攻击—后退）"},
    {"id": "rusher", "name": "Rusher", "color": "§c", "description": "高压近身，3.4 格内高频攻击"},
    {"id": "aimbot", "name": "Aimbot", "color": "§d", "description": "精确对准，3.9 格内持续输出"},
)

BUILTIN_IDS: tuple[str, ...] = tuple(item["id"] for item in BUILTIN_SCRIPTS)

#: 训练侧额外会打标签的角色（被训练方），不在脚本列表里但颜色可以配置。
EXTRA_ROLES: tuple[str, ...] = ("trained",)


def normalize_color(value: Any) -> str:
    text = str(value or "").strip()
    if not COLOR_RE.match(text):
        raise BadRequest(f"color 只允许单个颜色码（&0–&f 或 §0–§f），收到 {text!r}")
    return "§" + text[-1].lower()


def normalize_item(raw: Any, *, builtin: bool = False) -> dict:
    if not isinstance(raw, dict):
        raise BadRequest("脚本项必须是 JSON 对象")
    # 不自动小写：大写 id 直接拒绝，避免“存进去的 id 和输入的不一样”。
    ident = str(raw.get("id") or "").strip()
    if not ID_RE.match(ident):
        raise BadRequest("脚本 id 需为 1–24 位小写字母/数字/下划线/连字符，且以字母或数字开头")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise BadRequest("脚本 name 不能为空")
    if len(name) > NAME_MAX:
        raise BadRequest(f"脚本 name 最多 {NAME_MAX} 个字符")
    description = str(raw.get("description") or "").strip()
    if len(description) > DESCRIPTION_MAX:
        raise BadRequest(f"脚本 description 最多 {DESCRIPTION_MAX} 个字符")
    return {
        "id": ident,
        "name": name,
        "color": normalize_color(raw.get("color")),
        "description": description,
    }


def default_document() -> dict:
    return {
        "format": FORMAT,
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": [dict(item) for item in BUILTIN_SCRIPTS],
    }


class ScriptStore:
    """``dashboard/var/scripts.json`` 的读写；内置项与自定义项合并后对外输出。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.trash = self.path.parent / "scripts-trash"

    # --- 查询 -----------------------------------------------------------
    def ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self._write(default_document())

    def _load_overrides(self) -> list[dict]:
        """读盘（容错）：只返回合法项；文件损坏时退回空表（内置项仍会出现在输出里）。"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            return []
        items: list[dict] = []
        for raw in data["items"]:
            try:
                items.append(normalize_item(raw))
            except BadRequest:
                continue
        return items

    def items(self) -> list[dict]:
        """内置（含覆盖）+ 自定义，``builtin`` 标记随项返回。"""
        overrides = {item["id"]: item for item in self._load_overrides()}
        merged: list[dict] = []
        for base in BUILTIN_SCRIPTS:
            override = overrides.get(base["id"], {})
            merged.append({
                "id": base["id"],
                "name": override.get("name") or base["name"],
                "color": override.get("color") or base["color"],
                "description": override.get("description") or base["description"],
                "builtin": True,
            })
        for ident, item in overrides.items():
            if ident in BUILTIN_IDS:
                continue
            merged.append({**item, "builtin": False})
        # 保持内置项声明顺序（与训练侧 OPPONENT_MODES 一致），自定义项按 id 排在其后。
        merged.sort(key=lambda item: (0 if item["builtin"] else 1,))
        return merged

    def color_for(self, ident: str) -> str | None:
        target = str(ident or "").strip().lower()
        for item in self.items():
            if item["id"] == target:
                return item["color"]
        return None

    # --- 变更 -----------------------------------------------------------
    def save(self, raw: Any) -> dict:
        item = normalize_item(raw)
        overrides = {entry["id"]: entry for entry in self._load_overrides()}
        overrides[item["id"]] = item
        self._write_document(overrides)
        return {**item, "builtin": item["id"] in BUILTIN_IDS}

    def delete(self, ident: Any) -> str:
        target = str(ident or "").strip().lower()
        if not ID_RE.match(target):
            raise BadRequest("脚本 id 不合法")
        if target in BUILTIN_IDS:
            raise BadRequest("内置脚本不能删除（可以改颜色或名称）")
        overrides = {entry["id"]: entry for entry in self._load_overrides()}
        if target not in overrides:
            raise NotFound(f"脚本不存在: {target}")
        overrides.pop(target)
        self._write_document(overrides)
        return target

    # --- 内部 -----------------------------------------------------------
    def _write_document(self, overrides: dict[str, dict]) -> None:
        data = {
            "format": FORMAT,
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "items": sorted(overrides.values(), key=lambda item: item["id"]),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self._backup()
        self._write(data)

    def _backup(self) -> None:
        self.trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.trash / f"scripts-{stamp}.json"
        index = 2
        while target.exists():
            target = self.trash / f"scripts-{stamp}-{index}.json"
            index += 1
        try:
            shutil.copy2(self.path, target)
        except OSError as exc:
            raise BadRequest(f"备份脚本表失败: {exc}") from exc

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise BadRequest(f"写入脚本表失败: {exc}") from exc
