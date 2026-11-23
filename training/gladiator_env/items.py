"""物品能力表（动作空间 v2 的「物品无关」基础设施）。

设计要点（对应 ``docs/identity-and-action-space.md`` §2.3）：

- 动作空间只保留 ``slot`` / ``use`` 两个入口，**物品身份不进动作**；「这个物品怎么用」
  由本表回答，因此新增物品只需往 ``items.json`` 加一条，不必改训练代码、不必重训；
- 匹配支持通配与正则：``*_sword``、``*_potion``、``re:^.*_sword$``，先精确后通配；
- **未知物品一律落到 ``misc`` + ``use.mode=none``，不报错**（开放词汇下这是常态，
  例如别人丢在地上的模组物品）；
- 表本身的书写错误（分类拼错、mode 不认识）在加载时就报错——静默吞掉只会让
  mask 悄悄算错，比崩溃更难查。

``training/gladiator_env/items.json`` 同时被 ``bot-runtime`` 读取（``--items-json``），
两端的语义必须一致：训练侧用它算 action mask，运行时用它决定按住多少 tick。

用法::

    from gladiator_env.items import ItemTable

    table = ItemTable.load()                 # 默认读同目录 items.json
    table.category("diamond_sword")          # "sword"
    table.use_spec("golden_apple")           # {"mode": "hold", "ticks": 32}
    table.reach("diamond_sword")             # 3.0（1.8 原版水平）
    table.reach("golden_apple")              # None（不参与攻击距离）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterator

#: 允许的物品分类（观测 v2 的 one-hot 顺序也按这个元组，勿随意改序）。
CATEGORIES: tuple[str, ...] = ("sword", "tool", "food", "block", "bow", "potion", "misc")

#: 允许的使用方式：
#: - ``none``       不能右键（例如纯材料）
#: - ``click``      一次右键（鱼竿甩出、药水、放方块）
#: - ``hold``       按住 N tick 后松开（吃食物）
#: - ``hold_block`` 按住保持（剑格挡），松手即撤
#: - ``charge``     按住蓄力 N tick 后松开（拉弓）
USE_MODES: tuple[str, ...] = ("none", "click", "hold", "hold_block", "charge")

#: 一个 tick 的毫秒数（1.8 固定 20 TPS）；运行时按它把 ticks 换算成等待时间。
TICK_MS = 50

#: 未匹配任何条目时的能力：``misc`` + 不能用右键。
MISC_CAPABILITY: dict[str, Any] = {
    "category": "misc",
    "use": {"mode": "none", "ticks": 0},
    "cooldownTicks": 0,
    "reach": None,
    "stackGoal": 1,
    "projectile": False,
}

#: 默认表路径（与模块同目录，仓库内固定）。
DEFAULT_ITEMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "items.json")


def canonical_name(name: Any) -> str:
    """把物品名归一成「无命名空间的小写名」。

    ``minecraft:diamond_sword`` / ``Diamond_Sword`` / `` diamond_sword `` 都归一成
    ``diamond_sword``：协议里的观测用的是裸名，而 items.json 用带命名空间的名字，
    两边必须能对上。
    """
    text = str(name or "").strip().lower()
    if text.startswith("minecraft:"):
        text = text[len("minecraft:"):]
    return text


def _compile_pattern(pattern: str) -> "re.Pattern[str] | None":
    """把表里的键编译成匹配器；``None`` 表示「精确匹配」（走字典快查）。"""
    text = str(pattern or "").strip()
    if text.startswith("re:"):
        return re.compile(text[3:])
    text = canonical_name(text)
    if not text:
        return None
    if "*" in text or "?" in text:
        # 通配 → 全串锚定的正则；注意先 escape 再把 \* / \? 还原成通配符。
        return re.compile("^" + re.escape(text).replace("\\*", ".*").replace("\\?", ".") + "$")
    return None


def _coerce_use(raw: Any, label: str) -> dict[str, Any]:
    """校验并归一 ``use`` 字段。"""
    if raw is None:
        return {"mode": "none", "ticks": 0}
    if not isinstance(raw, dict):
        raise ValueError(f"items.json 的 {label}.use 必须是对象，收到 {type(raw).__name__}")
    mode = str(raw.get("mode") or "none").strip().lower()
    if mode not in USE_MODES:
        raise ValueError(
            f"items.json 的 {label}.use.mode 不认识：{mode!r}（允许 {', '.join(USE_MODES)}）")
    try:
        ticks = int(raw.get("ticks") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"items.json 的 {label}.use.ticks 必须是整数") from exc
    if ticks < 0:
        raise ValueError(f"items.json 的 {label}.use.ticks 不能为负：{ticks}")
    return {"mode": mode, "ticks": ticks}


def _coerce_entry(label: str, raw: Any) -> dict[str, Any]:
    """把一条原始配置补全成完整能力字典（缺省值见 ``MISC_CAPABILITY``）。"""
    if not isinstance(raw, dict):
        raise ValueError(f"items.json 的 {label} 必须是对象，收到 {type(raw).__name__}")
    category = str(raw.get("category") or "misc").strip().lower()
    if category not in CATEGORIES:
        raise ValueError(
            f"items.json 的 {label}.category 不认识：{category!r}（允许 {', '.join(CATEGORIES)}）")
    reach_raw = raw.get("reach")
    reach: float | None
    if reach_raw is None:
        reach = None
    else:
        try:
            reach = float(reach_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"items.json 的 {label}.reach 必须是数字") from exc
        if reach <= 0:
            raise ValueError(f"items.json 的 {label}.reach 必须为正数：{reach}")
    try:
        cooldown = max(0, int(raw.get("cooldownTicks") or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"items.json 的 {label}.cooldownTicks 必须是整数") from exc
    try:
        stack_goal = max(1, int(raw.get("stackGoal") or 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"items.json 的 {label}.stackGoal 必须是整数") from exc
    return {
        "category": category,
        "use": _coerce_use(raw.get("use"), label),
        "cooldownTicks": cooldown,
        "reach": reach,
        "stackGoal": stack_goal,
        "projectile": bool(raw.get("projectile", False)),
    }


class ItemTable:
    """物品能力表：``name -> capability`` 的只读查询对象。

    匹配顺序：精确名 → 按文件顺序的通配/正则。因此 ``*_sword`` 可以放在
    ``diamond_sword`` 之后作为兜底，而不会反过来盖掉精确项。
    """

    def __init__(
        self,
        entries: "dict[str, dict[str, Any]] | None" = None,
        *,
        path: str | None = None,
    ) -> None:
        self.path = path
        self._exact: dict[str, dict[str, Any]] = {}
        self._exact_labels: list[str] = []
        self._patterns: list[tuple[str, "re.Pattern[str]", dict[str, Any]]] = []
        self._patterns_source: list[str] = []
        for label, raw in (entries or {}).items():
            self._add(label, raw)

    # --- 构造 -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "ItemTable":
        """读 ``items.json``（默认同目录那份）。

        ``path`` 不存在时抛 ``FileNotFoundError`` 并带上绝对路径——运行时的
        ``--items-json`` 写错时，这行报错就是唯一的线索。
        """
        target = os.fspath(path) if path is not None else DEFAULT_ITEMS_PATH
        absolute = os.path.abspath(target)
        if not os.path.isfile(absolute):
            raise FileNotFoundError(f"找不到物品能力表 items.json：{absolute}")
        with open(absolute, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"items.json 不是合法 JSON（{absolute}）：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"items.json 顶层必须是对象（{absolute}）")
        entries = data.get("items")
        if not isinstance(entries, dict):
            # 兼容 §2.3 文档里的扁平写法：顶层直接是 name -> 配置。
            entries = {
                key: value for key, value in data.items()
                if not str(key).startswith("_") and str(key) != "version"
            }
        return cls(entries, path=absolute)

    def _add(self, label: str, raw: Any) -> None:
        capability = _coerce_entry(label, raw)
        matcher = _compile_pattern(label)
        if matcher is None:
            key = canonical_name(label)
            if not key:
                raise ValueError(f"items.json 里有空键名：{label!r}")
            self._exact[key] = capability
            self._exact_labels.append(str(label))
        else:
            self._patterns.append((str(label), matcher, capability))
            self._patterns_source.append(str(label))

    # --- 查询 -----------------------------------------------------------

    def _lookup(self, name: Any) -> "tuple[dict[str, Any], str | None]":
        key = canonical_name(name)
        if key and key in self._exact:
            return self._exact[key], key
        for label, matcher, capability in self._patterns:
            if key and matcher.match(key):
                return capability, label
        return MISC_CAPABILITY, None

    def capability(self, name: Any) -> dict[str, Any]:
        """返回该物品的能力字典（拷贝，调用方可随意改）。

        未知物品返回 ``misc`` 默认能力，``matched`` 为 ``None``——**不抛异常**，
        这条是「开放词汇」的前提：背包里出现没见过的东西不能把训练打断。
        """
        capability, matched = self._lookup(name)
        result = dict(capability)
        result["use"] = dict(capability.get("use") or {"mode": "none", "ticks": 0})
        result["name"] = canonical_name(name)
        result["matched"] = matched
        return result

    def use_spec(self, name: Any) -> dict[str, Any]:
        """返回 ``{"mode": ..., "ticks": ...}``（``use`` 的独立入口，mask 用得多）。"""
        return dict(self.capability(name)["use"])

    def reach(self, name: Any) -> float | None:
        """返回该物品的攻击/使用距离；表里没写就是 ``None``（由 BotProfile.reach 决定）。"""
        value = self.capability(name).get("reach")
        return None if value is None else float(value)

    def category(self, name: Any) -> str:
        """返回分类（``sword`` / ``food`` / … / ``misc``）。"""
        return str(self.capability(name)["category"])

    def is_usable(self, name: Any) -> bool:
        """该物品能否右键（``use`` 头的 1/2 档是否合法）。"""
        return self.use_spec(name)["mode"] != "none"

    # --- 诊断 -----------------------------------------------------------

    def patterns(self) -> list[str]:
        """表里声明的全部键（按文件顺序），诊断/日志用。"""
        return list(self._exact_labels) + list(self._patterns_source)

    def __len__(self) -> int:
        return len(self._exact) + len(self._patterns)

    def __iter__(self) -> Iterator[str]:
        return iter(self.patterns())

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        source = os.path.basename(self.path) if self.path else "memory"
        return f"<ItemTable {source} 精确 {len(self._exact)} 条 / 通配 {len(self._patterns)} 条>"


_DEFAULT_TABLE: ItemTable | None = None


def default_table() -> ItemTable:
    """进程级共享的默认表（惰性加载一次；``items.json`` 改动后重启生效）。"""
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = ItemTable.load()
    return _DEFAULT_TABLE
