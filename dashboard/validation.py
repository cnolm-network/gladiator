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


# ---------------------------------------------------------------------------
# 训练「实验参数（高级）」
#
# 这些字段原先只能从命令行传给 gladiator_train.train；Dashboard 把它们提升为
# 训练计划的正式字段。全部可选、全部有默认值；缺省时其余参数（users / ports /
# seed / epochs / plan 等）生成的命令行与改造前完全一致。
#
# 有两处默认值刻意与 train.py 的 CLI 缺省不同（网页默认更保守，表单里可改回）：
#   - ``bcEpisodes`` 默认 0（CLI 4）：网页默认不做行为克隆；
#   - ``entropyCoef`` 默认 0.01（CLI 0.02）。
# ---------------------------------------------------------------------------

#: 更新算法（train.py ``--algo``）。
TRAINING_ALGOS = ("ppo", "reinforce")
#: 动作空间（train.py ``--action-space``）：v1 = 11 个宏动作单头，v2 = 9 头因子化。
ACTION_SPACES = ("v1", "v2")

#: 实验参数缺省值：与 ``/api/training/options`` 的 ``defaults`` 同源，前端直接用。
EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "algo": "ppo",
    "actionSpace": "v1",
    "initPolicy": "",
    "bcEpisodes": 0,
    "evalBeforeTrain": False,
    "seed": 7,
    "victory": 50.0,
    "lr": 3e-4,
    "targetEntropy": 0.3,
    "entropyCoef": 0.01,
    "resetPolicyHead": False,
    "users": ["Bot_T1", "Bot_T2"],
    "ports": [8770, 8771],
}

#: 单场地训练支持 2..16 个 Bot（学习方 + 至少一个对手）。
MIN_TRAIN_BOTS = 2
MAX_TRAIN_BOTS = 16


def parse_choice(value: Any, field: str, label: str, choices: tuple[str, ...],
                 default: str) -> str:
    """枚举字段：去空白 + 小写归一化后比对，非法值报 400 并列出合法取值。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    text = str(value).strip().lower()
    if text not in choices:
        raise BadRequest(f"{label}（{field}）只支持 {' | '.join(choices)}，收到 {value!r}")
    return text


def _bounded_int(value: Any, field: str, label: str, minimum: int, maximum: int,
                 default: int) -> int:
    """整数 + 区间；越界时把字段名与合法范围一起写进错误信息。"""
    number = parse_int(value, field, default=default)
    if not minimum <= number <= maximum:
        raise BadRequest(f"{label}（{field}）需要在 {minimum} ~ {maximum} 之间，收到 {number}")
    return number


def _bounded_float(value: Any, field: str, label: str, minimum: float, maximum: float,
                   default: float) -> float:
    """浮点 + 区间；越界时把字段名与合法范围一起写进错误信息。"""
    number = parse_float(value, field, default=default)
    if not minimum <= number <= maximum:
        raise BadRequest(
            f"{label}（{field}）需要在 {minimum:g} ~ {maximum:g} 之间，收到 {number:g}"
        )
    return number


def parse_experiment_ports(value: Any, field: str = "端口（ports）",
                           reserved: Iterable[int] = ()) -> list[int]:
    """训练端口：1..65535、互不重复、至少 2 个、不能占用保留端口。"""
    raw_items = split_list(value, field)
    if not raw_items:
        raise BadRequest(f"{field} 不能为空：每个 Bot 需要一个通道 B 端口")
    reserved_set = {int(item) for item in reserved}
    ports: list[int] = []
    for index, item in enumerate(raw_items):
        port = parse_int(item, f"{field}[{index}]", minimum=1, maximum=65535)
        if port in reserved_set:
            occupied = "、".join(str(item) for item in sorted(reserved_set))
            raise BadRequest(
                f"{field}[{index}] 不能使用保留端口 {port}（Dashboard / 服务器已占用: {occupied}）"
            )
        ports.append(port)
    duplicates = sorted({port for port in ports if ports.count(port) > 1})
    if duplicates:
        raise BadRequest(
            f"{field} 不能重复：{'、'.join(str(port) for port in duplicates)}"
            "（每个 Bot 必须独占一个端口）"
        )
    if len(ports) < MIN_TRAIN_BOTS:
        raise BadRequest(
            f"{field} 至少需要 {MIN_TRAIN_BOTS} 个（每个 Bot 一个），当前 {len(ports)} 个"
        )
    return ports


def parse_experiment(payload: dict, *, defaults: dict | None = None,
                     reserved_ports: Iterable[int] = ()) -> dict:
    """解析训练计划的「实验参数（高级）」，返回规范化后的完整字段字典。

    所有字段可选：缺省取 ``EXPERIMENT_DEFAULTS``（可被 ``defaults`` 覆盖，便于
    后续在 options 里调整默认值）。交叉规则在这里统一处理，避免遗漏：

    - ``users`` / ``ports`` 一对一，且至少 2 个；
    - ``ports`` 1..65535、互不重复、不占用保留端口；
    - ``actionSpace=v2`` 时 ``bcEpisodes`` 必须为 0（BC 老师只会 v1 宏动作）。

    ``initPolicy`` 只做「字符串 + 空串 = 不用」这一层校验，模型 id / 路径的
    解析交给 ``TrainingRunner._resolve_init_policy``（那里才知道 var/models）。
    """
    base = dict(EXPERIMENT_DEFAULTS)
    if defaults:
        base.update({key: value for key, value in defaults.items() if key in base})

    users = [
        validate_username(item, f"用户名[{index}]")
        for index, item in enumerate(split_list(pick(payload, "users", default=base["users"]),
                                                "用户（users）"))
    ]
    if not users:
        raise BadRequest("用户（users）不能为空：至少需要 2 个 Bot（学习方 + 对手）")
    if not MIN_TRAIN_BOTS <= len(users) <= MAX_TRAIN_BOTS:
        raise BadRequest(
            f"用户（users）需要 {MIN_TRAIN_BOTS}~{MAX_TRAIN_BOTS} 个 Bot，当前 {len(users)} 个"
            "（顺序对应端口，第 1 个是被训练方）"
        )

    ports = parse_experiment_ports(pick(payload, "ports", default=base["ports"]),
                                   "端口（ports）", reserved_ports)
    if len(ports) != len(users):
        raise BadRequest(
            f"端口（ports）数量必须与用户（users）一致：users {len(users)} 个、ports {len(ports)} 个"
        )

    algo = parse_choice(pick(payload, "algo", default=base["algo"]), "algo", "算法",
                        TRAINING_ALGOS, base["algo"])
    action_space = parse_choice(pick(payload, "actionSpace", default=base["actionSpace"]),
                                "actionSpace", "动作空间", ACTION_SPACES, base["actionSpace"])

    raw_init = pick(payload, "initPolicy", default=base["initPolicy"])
    if raw_init is None:
        raw_init = ""
    if not isinstance(raw_init, str):
        raise BadRequest("初始模型（initPolicy）需要字符串：模型 id 或 policy.npz 路径")
    init_policy = raw_init.strip()

    bc_episodes = _bounded_int(pick(payload, "bcEpisodes", default=base["bcEpisodes"]),
                               "bcEpisodes", "BC 局数", 0, 100, base["bcEpisodes"])
    if action_space == "v2" and bc_episodes > 0:
        raise BadRequest(
            "动作空间（actionSpace）v2 不支持 BC 预训练：BC 老师输出的是 v1 宏动作，"
            "v2 没有对应标签；请把 BC 局数设为 0，或改用 v1"
        )

    return {
        "algo": algo,
        "actionSpace": action_space,
        "initPolicy": init_policy,
        "bcEpisodes": bc_episodes,
        "evalBeforeTrain": parse_bool(
            pick(payload, "evalBeforeTrain", default=base["evalBeforeTrain"]),
            "训练前评估（evalBeforeTrain）",
        ),
        "seed": _bounded_int(pick(payload, "seed", default=base["seed"]),
                             "seed", "随机种子", 0, 2**31 - 1, base["seed"]),
        "victory": _bounded_float(pick(payload, "victory", default=base["victory"]),
                                  "victory", "胜利权重", 0.0, 1000.0, base["victory"]),
        "lr": _bounded_float(pick(payload, "lr", default=base["lr"]),
                             "lr", "学习率", 1e-6, 1e-1, base["lr"]),
        "targetEntropy": _bounded_float(
            pick(payload, "targetEntropy", default=base["targetEntropy"]),
            "targetEntropy", "目标熵", 0.0, 5.0, base["targetEntropy"],
        ),
        "entropyCoef": _bounded_float(
            pick(payload, "entropyCoef", default=base["entropyCoef"]),
            "entropyCoef", "熵系数", 0.0, 10.0, base["entropyCoef"],
        ),
        "resetPolicyHead": parse_bool(
            pick(payload, "resetPolicyHead", default=base["resetPolicyHead"]),
            "重置策略头（resetPolicyHead）",
        ),
        "users": users,
        "ports": ports,
    }


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


# ---------------------------------------------------------------------------
# 软约束（运行时参数，不写进模型）
# ---------------------------------------------------------------------------

#: 布尔键别名（全部小写、去掉 ``-``/``_``）。与 bot-runtime/src/soft-constraints.js 同表。
SOFT_BOOL_ALIASES = {
    "rod": "allowRod",
    "allowrod": "allowRod",
    "userod": "allowRod",
    "gapple": "allowGapple",
    "apple": "allowGapple",
    "goldenapple": "allowGapple",
    "allowgapple": "allowGapple",
    "block": "allowBlock",
    "swordblock": "allowBlock",
    "allowblock": "allowBlock",
    "jump": "allowJumpCrit",
    "jumpcrit": "allowJumpCrit",
    "crit": "allowJumpCrit",
    "allowjumpcrit": "allowJumpCrit",
}
#: 数值键：别名 → (字段名, 最小, 最大)
SOFT_NUMBER_ALIASES = {
    "cps": ("maxCps", 1, 40),
    "maxcps": ("maxCps", 1, 40),
    "reachoffset": ("minReachOffset", 0, 3),
    "minreachoffset": ("minReachOffset", 0, 3),
}
SOFT_KEY_LABELS = {
    "allowRod": "rod",
    "allowGapple": "gapple",
    "allowBlock": "block",
    "allowJumpCrit": "jump",
    "maxCps": "cps",
    "minReachOffset": "reachOffset",
}
SOFT_TRUE_WORDS = {"1", "true", "yes", "on", "enable", "enabled"}
SOFT_FALSE_WORDS = {"0", "false", "no", "off", "disable", "disabled"}
MAX_SOFT_SPEC = 200


def _soft_token(text: Any) -> str:
    return re.sub(r"[-_\s]+", "", str(text or "").lower())


def parse_soft_spec(value: Any, field: str = "软约束") -> str:
    """校验 ``--soft`` 串并原样返回（供运行时透传）。

    规则与 ``bot-runtime/src/soft-constraints.js::parseSoft`` 一致：逗号分隔，
    ``name`` / ``noName`` / ``name=on|off`` 三种布尔写法，``cps=12``、``reachOffset=0.5``
    两种数值写法；未知键与非法值都报 400（不静默忽略，否则用户会以为生效了）。
    """
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise BadRequest(f"{field} 需要字符串（如 cps=12,noRod）")
    text = value.strip()
    if not text:
        return ""
    if len(text) > MAX_SOFT_SPEC:
        raise BadRequest(f"{field} 太长（上限 {MAX_SOFT_SPEC} 字符）")
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        name, sep, raw_value = token.partition("=")
        name = name.strip()
        raw_value = raw_value.strip() if sep else None
        normalized = _soft_token(name)
        if not normalized:
            raise BadRequest(f"{field} 里有空键（{token}）")

        negated = False
        key = SOFT_BOOL_ALIASES.get(normalized)
        if key is None and normalized.startswith("no"):
            base = normalized[2:]
            if base in SOFT_BOOL_ALIASES:
                key = SOFT_BOOL_ALIASES[base]
                negated = True
            elif base in SOFT_NUMBER_ALIASES:
                label = SOFT_KEY_LABELS[SOFT_NUMBER_ALIASES[base][0]]
                raise BadRequest(f"{name} 没有意义（{label} 是数值键，请用 {label}=值）")
        if key is None and normalized in SOFT_NUMBER_ALIASES:
            label, minimum, maximum = SOFT_NUMBER_ALIASES[normalized]
            if negated:
                raise BadRequest(f"{name} 没有意义（数值键请用 {SOFT_KEY_LABELS[label]}=值）")
            if raw_value is None:
                raise BadRequest(f"{SOFT_KEY_LABELS[label]} 需要值（例如 {SOFT_KEY_LABELS[label]}=12）")
            try:
                number = float(raw_value)
            except ValueError as exc:
                raise BadRequest(f"{SOFT_KEY_LABELS[label]} 的值 {raw_value} 不是数字") from exc
            if not minimum <= number <= maximum:
                raise BadRequest(f"{SOFT_KEY_LABELS[label]} 需要在 {minimum}..{maximum} 之间"
                                 f"（收到 {raw_value}）")
            continue
        if key is None:
            raise BadRequest(
                f"未知软约束键: {name}（可用: rod, gapple, block, jump, cps, reachOffset；"
                "取反写 noRod）")
        label = SOFT_KEY_LABELS[key]
        if negated and raw_value is not None:
            raise BadRequest(f"{name} 不能同时带值（写成 {label}=off）")
        if negated:
            continue          # noRod 合法
        if raw_value is None:
            continue          # 裸写 rod = 打开
        word = _soft_token(raw_value)
        if word in SOFT_TRUE_WORDS or word in SOFT_FALSE_WORDS:
            continue
        raise BadRequest(f"{label} 的值 \"{raw_value}\" 不是布尔（on/off/true/false/1/0）")
    return text


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
