"""API 入参校验（全部给出中文可读错误）。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from errors import BadRequest

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,16}$")
STAGE_MODES = ("random", "wander", "idle", "scripted", "rusher", "aimbot")


def pick(payload: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    raise BadRequest(f"{field} 需要布尔值")


def parse_int(value: Any, field: str, *, default: int | None = None, minimum: int | None = None,
              maximum: int | None = None) -> int:
    if value is None or value == "":
        if default is None:
            raise BadRequest(f"{field} 不能为空")
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{field} 需要整数，收到 {value!r}") from exc
    if minimum is not None and number < minimum:
        raise BadRequest(f"{field} 不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise BadRequest(f"{field} 不能大于 {maximum}")
    return number


def parse_float(value: Any, field: str, *, default: float | None = None,
                minimum: float | None = None, maximum: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise BadRequest(f"{field} 不能为空")
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{field} 需要数字，收到 {value!r}") from exc
    if number != number:  # NaN
        raise BadRequest(f"{field} 非法")
    if minimum is not None and number < minimum:
        raise BadRequest(f"{field} 不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise BadRequest(f"{field} 不能大于 {maximum}")
    return number


def validate_username(value: Any, field: str = "用户名") -> str:
    if not isinstance(value, str):
        raise BadRequest(f"{field} 需要字符串")
    name = value.strip()
    if not USERNAME_RE.match(name):
        raise BadRequest(f"{field} 需为 1-16 位字母/数字/下划线，收到 {name!r}")
    return name


def split_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        items = [item for item in re.split(r"[\s,]+", value.strip()) if item]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise BadRequest(f"{field} 需要数组或逗号分隔字符串")
    return items


def parse_port(value: Any, field: str, reserved: Iterable[int] = ()) -> int:
    port = parse_int(value, field, minimum=1024, maximum=65535)
    if port in {int(item) for item in reserved}:
        raise BadRequest(f"{field} 不能使用保留端口 {port}")
    return port


def parse_ports(value: Any, field: str, reserved: Iterable[int] = ()) -> list[int]:
    raw_items = split_list(value, field) if not isinstance(value, (list, tuple)) else value
    if not raw_items:
        raise BadRequest(f"{field} 不能为空")
    ports = [parse_port(item, f"{field}[{index}]", reserved) for index, item in enumerate(raw_items)]
    if len(set(ports)) != len(ports):
        raise BadRequest(f"{field} 存在重复端口")
    return ports


def parse_center(value: Any, field: str = "场地中心") -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value or "").split(",")
    if len(parts) != 3:
        raise BadRequest(f"{field} 需要 x,y,z 三个数字")
    try:
        center = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{field} 需要 x,y,z 三个数字") from exc
    return center  # type: ignore[return-value]


MAX_PLAN_ITEMS = 50
MAX_TOTAL_EPISODES = 5000


def parse_plan(value: Any, *, fallback_mode: str = "scripted",
               fallback_episodes: int = 4) -> list[tuple[str, int]]:
    """把训练计划规范化为 ``[(opponentMode, episodes), ...]``。

    接受三种形态，便于旧前端与脚本调用平滑迁移：
    - 缺省 / 空串：单段兜底 ``fallback_mode × fallback_episodes``；
    - 字符串 ``"rusher:2,idle:1"``：等价于旧的 ``stages`` 字段；
    - 数组 ``[{"opponentMode": "rusher", "episodes": 2}]`` 或 ``[["rusher", 2]]``。

    计划是**唯一**的回合来源：总回合 = 各项之和，不再有独立的“回合数”输入。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return [(fallback_mode, fallback_episodes)]
    if isinstance(value, str):
        stages = parse_stages(value, fallback_episodes)
        return stages or [(fallback_mode, fallback_episodes)]
    if not isinstance(value, (list, tuple)):
        raise BadRequest("plan 需要数组（[{opponentMode, episodes}]）或 '模式:回合数,...' 字符串")
    if not value:
        raise BadRequest("plan 不能为空；至少保留一项")
    if len(value) > MAX_PLAN_ITEMS:
        raise BadRequest(f"plan 最多 {MAX_PLAN_ITEMS} 项，当前 {len(value)}")
    items: list[tuple[str, int]] = []
    for index, raw in enumerate(value):
        if isinstance(raw, dict):
            mode = str(raw.get("opponentMode") or raw.get("mode") or "").strip().lower()
            episodes = parse_int(raw.get("episodes"), f"plan[{index}].episodes",
                                 default=fallback_episodes, minimum=1, maximum=500)
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            mode = str(raw[0] or "").strip().lower()
            episodes = parse_int(raw[1], f"plan[{index}].episodes", minimum=1, maximum=500)
        else:
            raise BadRequest(f"plan[{index}] 需要 {{\"opponentMode\", \"episodes\"}} 对象")
        if mode not in STAGE_MODES:
            raise BadRequest(f"plan[{index}] 未知对手模式 {mode!r}，可选: {', '.join(STAGE_MODES)}")
        items.append((mode, int(episodes)))
    total = sum(count for _mode, count in items)
    if total > MAX_TOTAL_EPISODES:
        raise BadRequest(f"计划总回合 {total} 超过上限 {MAX_TOTAL_EPISODES}")
    return items


def format_plan(items: Iterable[tuple[str, int]]) -> str:
    """序列化为训练脚本认识的 ``mode:episodes,...`` 字符串。"""
    return ",".join(f"{mode}:{count}" for mode, count in items)


def parse_stages(value: Any, fallback_episodes: int) -> list[tuple[str, int]]:
    text = str(value or "").strip()
    if not text:
        return []
    stages: list[tuple[str, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise BadRequest(f"阶段格式应为 模式:回合数，收到 {item!r}")
        mode, count = item.split(":", 1)
        mode = mode.strip().lower()
        if mode not in STAGE_MODES:
            raise BadRequest(f"未知对手模式 {mode!r}，可选: {', '.join(STAGE_MODES)}")
        try:
            episodes = int(count) if count.strip() else int(fallback_episodes)
        except ValueError as exc:
            raise BadRequest(f"阶段回合数需要整数: {item!r}") from exc
        if episodes <= 0:
            raise BadRequest(f"阶段回合数必须大于 0: {item!r}")
        stages.append((mode, episodes))
    return stages
