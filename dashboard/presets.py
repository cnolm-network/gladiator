"""训练预设：以单个 JSON 文件持久化在 ``dashboard/var/presets/<id>.json``。

文件格式（``gladiator.training-preset/v1``）::

    {
      "format": "gladiator.training-preset/v1",
      "name": "Preset-1",
      "createdAt": "2026-11-08T19:30:00",
      "updatedAt": "2026-11-08T19:30:00",
      "plan": [{"opponentMode": "scripted", "episodes": 10}],
      "planText": "scripted:10",
      "fields": {"epochs": 1, "maxSteps": 200, "seed": 7, "evalEpochs": 1}
    }

预设只保存**计划**与少量训练参数，不保存账号/端口/路径，避免换机器后失效。

两处可恢复副本（都在 ``dashboard/var/presets-trash/``，不进 git）：

- 删除：``<id>-<时间戳>.json``
- 同名覆盖保存：``<id>-overwrite-<时间戳>.json``（旧版本，写入新内容之前复制）
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from errors import BadRequest, NotFound
from validation import (
    format_plan,
    parse_bool,
    parse_center,
    parse_float,
    parse_int,
    parse_plan,
)

FORMAT = "gladiator.training-preset/v1"
NAME_MAX = 40
ID_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")
SLUG_RE = re.compile(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+")

# 只允许这些训练参数随预设保存；端口/用户名/路径与本机环境绑定，不落盘。
#
# 命名口径（与训练响应统一）：
# - ``epochs``      训练回合数（每个回合完整跑一遍计划）
# - ``evalEpochs``  评估回合数（旧字段名 ``evalEpisodes`` 只作读入兼容，一律归一为它）
#
# 训练响应里的 ``evalRounds`` = Σ计划轮次（每个评估回合的对局数），
# ``evalEpisodes`` = 评估总对局数 = evalEpochs × evalRounds，二者不再混用同一个名字。
FIELD_KEYS = (
    "epochs",
    "maxSteps",
    "seed",
    "evalEpochs",
    "center",
    "spacing",
    "prepSeconds",
    "autoSpawn",
    "autoCleanup",
)

#: 旧版预设文件里的字段名 → 新字段名（只读兼容，写回时不再出现旧名）。
LEGACY_FIELD_KEYS = {"evalEpisodes": "evalEpochs"}


def _field_value(name: str, value: Any) -> Any:
    """单个训练参数的类型/范围校验；错误信息直接指出字段与边界。"""
    label = f"预设字段 {name}"
    if name == "epochs":
        return parse_int(value, label, minimum=1, maximum=200)
    if name == "maxSteps":
        return parse_int(value, label, minimum=1, maximum=5000)
    if name == "seed":
        return parse_int(value, label, minimum=0, maximum=2**31 - 1)
    if name == "evalEpochs":
        return parse_int(value, label, minimum=0, maximum=50)
    if name == "center":
        center = parse_center(value, label)
        return ",".join(f"{part:g}" for part in center)
    if name == "spacing":
        return parse_float(value, label, minimum=1.0, maximum=20.0)
    if name == "prepSeconds":
        return parse_float(value, label, minimum=0.0, maximum=30.0)
    return parse_bool(value, label)


def _public_fields(raw: Any) -> dict:
    """读盘时归一化：旧名映射为新名，未知键丢弃，顺序固定。"""
    if not isinstance(raw, dict):
        return {}
    fields: dict[str, Any] = {}
    for key, value in raw.items():
        name = LEGACY_FIELD_KEYS.get(key, key)
        if name in FIELD_KEYS and name not in fields and value is not None:
            fields[name] = value
    return {key: fields[key] for key in FIELD_KEYS if key in fields}


class PresetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- 查询 -----------------------------------------------------------
    def list(self) -> list[dict]:
        items: list[dict] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            plan = data.get("plan")
            if not isinstance(plan, list) or not plan:
                continue
            items.append(self._public(path.stem, data))
        items.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return items

    def load(self, preset_id: Any) -> dict:
        ident = self._check_id(preset_id)
        path = self.root / f"{ident}.json"
        if not path.is_file():
            raise NotFound(f"预设不存在: {ident}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BadRequest(f"预设文件损坏: {ident}") from exc
        if not isinstance(data, dict):
            raise BadRequest(f"预设文件格式非法: {ident}")
        return self._public(ident, data)

    # --- 变更 -----------------------------------------------------------
    def save(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise BadRequest("请求体需要 JSON 对象")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise BadRequest("预设名称不能为空")
        if len(name) > NAME_MAX:
            raise BadRequest(f"预设名称最多 {NAME_MAX} 个字符")

        plan_items = parse_plan(
            payload.get("plan"),
            fallback_mode=str(payload.get("opponentMode") or "scripted"),
            fallback_episodes=4,
        )
        fields = self._fields(payload.get("fields"))
        preset_id = self._id_for(name)
        path = self.root / f"{preset_id}.json"

        created = time.strftime("%Y-%m-%dT%H:%M:%S")
        if path.is_file():
            # 覆盖保存前先备份旧文件：误保存也能从 presets-trash/ 找回上一版。
            self._backup(path, f"{preset_id}-overwrite")
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and old.get("createdAt"):
                    created = str(old["createdAt"])
            except (OSError, json.JSONDecodeError):
                pass

        data = {
            "format": FORMAT,
            "name": name,
            "createdAt": created,
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "plan": [{"opponentMode": mode, "episodes": count} for mode, count in plan_items],
            "planText": format_plan(plan_items),
            "fields": fields,
        }
        self._write(path, data)
        return self._public(preset_id, data)

    def delete(self, preset_id: Any) -> str:
        ident = self._check_id(preset_id)
        path = self.root / f"{ident}.json"
        if not path.is_file():
            raise NotFound(f"预设不存在: {ident}")
        # 软删除：移到 var/presets-trash/，误删可直接从目录恢复（unlink 不进回收站）。
        self._backup(path, ident, move=True)
        return ident

    # --- 内部 -----------------------------------------------------------
    def _backup(self, path: Path, stem: str, *, move: bool = False) -> None:
        """在 ``var/presets-trash/`` 留一份可恢复副本。

        ``move=True`` 用于删除（源文件消失），否则用于覆盖保存（源文件保留）。
        备份失败直接报错：宁可不让这次写入发生，也不要静默丢掉上一版。
        """
        trash = self.root.parent / "presets-trash"
        trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = trash / f"{stem}-{stamp}.json"
        index = 2
        while target.exists():
            target = trash / f"{stem}-{stamp}-{index}.json"
            index += 1
        try:
            if move:
                os.replace(path, target)
            else:
                shutil.copy2(path, target)
        except OSError as exc:
            raise BadRequest(f"备份预设失败: {exc}") from exc

    @staticmethod
    def _check_id(preset_id: Any) -> str:
        ident = str(preset_id or "").strip()
        if not ID_RE.match(ident):
            raise BadRequest("预设 id 非法")
        return ident

    @staticmethod
    def _id_for(name: str) -> str:
        slug = SLUG_RE.sub("_", name).strip("_")[:48] or "preset"
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
        return f"{slug}-{digest}"

    @staticmethod
    def _fields(raw: Any) -> dict:
        """按白名单校验并归一化训练参数；类型/范围非法时抛中文 400。

        未知键直接忽略（向前兼容旧前端），但已知键的值必须合法——这样预设不会成为
        绕过 ``/api/training/start`` 校验的后门。
        """
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise BadRequest("fields 需要 JSON 对象")
        fields: dict[str, Any] = {}
        for key, value in raw.items():
            name = LEGACY_FIELD_KEYS.get(key, key)
            if name not in FIELD_KEYS or value is None or name in fields:
                continue
            fields[name] = _field_value(name, value)
        return {key: fields[key] for key in FIELD_KEYS if key in fields}

    @staticmethod
    def _public(preset_id: str, data: dict) -> dict:
        raw_plan = data.get("plan") or []
        plan: list[dict] = []
        total = 0
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            try:
                episodes = int(item.get("episodes") or 0)
            except (TypeError, ValueError):
                episodes = 0
            mode = str(item.get("opponentMode") or "")
            plan.append({"opponentMode": mode, "episodes": episodes})
            total += max(0, episodes)
        plan_text = data.get("planText")
        if not isinstance(plan_text, str) or not plan_text:
            plan_text = ",".join(f"{item['opponentMode']}:{item['episodes']}" for item in plan)
        return {
            "id": preset_id,
            "name": str(data.get("name") or preset_id),
            "format": str(data.get("format") or FORMAT),
            "createdAt": data.get("createdAt"),
            "updatedAt": data.get("updatedAt"),
            "plan": plan,
            "totalEpisodes": total,
            "planText": plan_text,
            # 旧文件里的 evalEpisodes 在这里统一成 evalEpochs，避免同名两义。
            "fields": _public_fields(data.get("fields")),
        }

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise BadRequest(f"写入预设失败: {exc}") from exc
