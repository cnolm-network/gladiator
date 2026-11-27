"""Gymnasium 训练环境（G-401）。

「普通世界 + 无 Mode 空手混战」的默认形态：

- 场地准备（传送/回满/清背包/主手空手）由调用方职责完成；本地训练场通过 RCON 驱动
  服务端控制台执行 `tp/clear/effect`，Bot 的生成与训练会话用 `/gladiator spawn|train`。
- 观测与动作全部走 Layer 2 的通道 B（`:8766+`），本层不接触 mineflayer、不调用 Bukkit API。
- `reset()` = 摆位 + 回满 + 等到「Alive 且血量满」；`step()` = 每个 Bot 一条复合动作，
  等待一个控制周期后收集新观测、结算奖励、判断回合终止。

环境不判定对局规则：回合终止只看「血量 ≤ 0 / 状态 Dead|Eliminated」，赢家由调用方 Mode 决定。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .link import LinkClient, LinkError
from .reward import RewardContext, RewardFn, default_reward, is_out, self_health
from .rcon import RconClient, RconError
from . import reward_presets
from .shaping import ShapingReward, ShapingWeights
from .spaces import (
    ACTION_NAMES,
    FRAMES_NORMALIZER,
    MULTI_DISCRETE_V2,
    MOVE_MS,
    NUM_ACTIONS,
    OBS_DIM,
    GroundEstimator,
    action_mask,
    attack_cooldown_remaining,
    decode_action,
    decode_intent,
    encode_observation,
    episode_fraction,
    nearest_player,
    self_yaw_of,
)
from .items import default_table

#: G-711：跨帧帧数计数器的饱和值（= spaces 里的分母）。超过它之后不再区分「多久以前」。
CROSS_FRAME_CAP = int(FRAMES_NORMALIZER)

# --- reach 口径（G-708）：默认原版水平，超限必须显式作弊 -------------------
#: 合法（非作弊）攻击距离上限：1.8 原版玩家约 3 格。
LEGIT_REACH_MAX = 3.0
#: 配置下限 / 硬上限（格）。
MIN_REACH = 1.0
MAX_REACH = 6.0


def check_reach(value: float, *, cheat: bool = False) -> float:
    """校验 reach 并返回 float；超过 ``LEGIT_REACH_MAX`` 时必须 ``cheat=True``。

    与 Java ``BotProfile`` / 运行时 ``normalizeProfile`` 同一条口径：
    非法值直接报错（不要静默夹住，否则「3.99 被悄悄改成 3.0」没人会发现），
    超限值要求显式开关，调用方负责打 ``[cheat] reach 超限=…（仅对照实验）``。
    """
    reach = float(value)
    if reach < MIN_REACH or reach > MAX_REACH:
        raise ValueError(f"reach 必须在 [{MIN_REACH}, {MAX_REACH}] 内，收到 {reach}")
    if reach > LEGIT_REACH_MAX and not cheat:
        raise ValueError(
            f"reach={reach} 超限：1.8 原版玩家约 {LEGIT_REACH_MAX} 格。"
            "要跑对照实验请显式开 cheat（`cheat_reach=True` / `--cheat-reach <值>`）")
    return reach


def cheat_reach_banner(reach: float) -> str:
    """作弊开关的日志行（格式固定，日志/报告里 grep 这一个串就能找出超限实验）。"""
    return f"[cheat] reach 超限={float(reach):g}（仅对照实验）"


#: 背包/护甲校验必需项（剑、金苹果、鱼竿）；护甲另外按 4 个槽位统计。
KIT_MUST_HAVE: tuple[tuple[str, int], ...] = (
    ("diamond_sword", 1),
    ("golden_apple", 5),
    ("fishing_rod", 1),
)
ARMOR_SLOTS = ("head", "torso", "legs", "feet")
#: 护甲槽位 → 物品名后缀：死亡掉落后要按**槽位**补发，``give`` 只会按名字给。
ARMOR_SLOT_SUFFIX = {
    "head": "_helmet",
    "torso": "_chestplate",
    "legs": "_leggings",
    "feet": "_boots",
}
#: 背包里找不到对应套件时的兜底护甲（与 ``ArenaConfig.kit`` 的默认值一致）。
DEFAULT_ARMOR_ITEMS = {
    "head": "diamond_helmet",
    "torso": "diamond_chestplate",
    "legs": "diamond_leggings",
    "feet": "diamond_boots",
}
#: 重生补装的确认窗口（tick）。``give`` 没有回执，重生后要盯几帧才算数；
#: 这段时间里该 Bot 不做进攻动作（背包可能还没同步）。
_RESTOCK_TICKS = 3
#: 补装确认期间要剥掉的动作类型（``use`` 要按参数细分，见 ``_KEEP_USE_ITEMS``）。
_COMBAT_ACTIONS = frozenset({"attack", "dig", "place"})
#: 补装期间**必须放行**的 use：切剑 / 穿甲 —— 它们本身就是补装动作，
#: 剥掉会造成「补装期间把自己的补装动作也剥了」的死循环（run 20261115-2239 的 FAILED_KIT 就是它）。
_KEEP_USE_ITEMS = frozenset({"armor", "sword"})
#: 补装最多重试几轮（每轮 ``_RESTOCK_TICKS`` 帧）；用完才放弃。
_RESTOCK_MAX_ROUNDS = 3
#: 单个物品 ``give`` 之后至少等几帧才复核（1.8 的 ``give`` 没有回执，背包同步要 1–2 tick）。
_GIVE_SETTLE_TICKS = 2
#: 单个物品整轮最多重发几次（用完就等下一轮，别再往 RCON 灌命令）。
_RESTOCK_MAX_GIVES = 3
#: 主动体检周期（步）：每 N 步查一次套装，缺件就触发补装。
_RESTOCK_CHECK_EVERY = 10
#: 这些运行时失败码说明「手上有动作但没东西」，直接当作补装触发源。
_RESTOCK_TRIGGER_CODES = frozenset({"ITEM_MISSING", "ITEM_NOT_FOUND", "NO_ITEM"})


def _coerce_opponent_policy(value: Any, *, obs_dim: int = OBS_DIM) -> tuple[Any | None, str]:
    """把 ``opponent_policy`` 归一成 ``(策略对象, 来源描述)``。

    接受 ``None`` / ``Policy`` 之类的对象（只要有 ``act``）/ ``.npz`` 路径（``str`` 或
    ``PathLike``，用 ``Policy.load`` 读，并把旧模型的 ``W1`` 右侧补零到当前 ``obs_dim``）。
    """
    if value is None:
        return None, "hand-written"
    if isinstance(value, (str, bytes)) or hasattr(value, "__fspath__"):
        # 延迟导入：``gladiator_env`` 不应该在 import 期就依赖训练侧模块。
        from gladiator_train.policy import Policy  # noqa: WPS433 - 见上

        path = os.fspath(value)
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        policy = Policy.load(str(path), obs_dim=int(obs_dim))
        return policy, str(path)
    if callable(getattr(value, "act", None)):
        return value, "memory"
    raise ValueError(
        f"opponent_policy 需要 Policy 对象或 .npz 路径，收到 {type(value).__name__}"
    )


def loadout_report(observation: dict | None) -> dict:
    """从一条 OBSERVATION 里读出“套装是否齐全”。

    这是修「装备偶尔被吞」的关键：1.8.8 的 ``clear``/``give`` 是控制台命令，
    没有回执保证，Bot 又可能在摆位阶段死亡掉物品，所以每局开始都要按观测校验一次。
    """
    self_view = (observation or {}).get("self") or {}
    counts: dict[str, int] = {}
    for item in self_view.get("inventory") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        counts[name] = counts.get(name, 0) + int(item.get("count") or 0)
    armor = self_view.get("armor") or {}
    worn = sum(1 for slot in ARMOR_SLOTS if armor.get(slot))
    # 护甲不进 ``missing``（那里只表示物品数量），单独给槽位列表：
    # 死亡后四件护甲是**掉落消失**的，只发 ``use{item:"armor"}`` 永远穿不上（背包里没有）。
    missing_armor = [slot for slot in ARMOR_SLOTS if not armor.get(slot)]
    held = str(((self_view.get("heldItem") or {}).get("name") or ""))
    missing = [
        (name, need - counts.get(name, 0))
        for name, need in KIT_MUST_HAVE
        if counts.get(name, 0) < need
    ]
    return {
        "counts": counts,
        "armor": worn,
        "missingArmor": missing_armor,
        "held": held,
        "missing": missing,
        "ok": not missing and worn == 4,
        "holding_sword": held.endswith("_sword"),
    }


@dataclass
class BotConfig:
    """单个 Bot 的连接信息 + 行为配置（G-507）。

    行为配置与 Java 侧 ``BotProfile`` 同义：CPS（含抖动）、反应延时（含抖动）、
    攻击距离、训练模型标签。默认值即“原版手感”：15±2 CPS / 220±40ms / **3.0 格**。

    G-708：``reach`` 默认 3.0（1.8 原版玩家约 3 格）。3.99 这类超限值必须显式
    ``cheat_reach=True``（对应 Java 的 ``cheat.reach``）——否则训练出来的距离感
    在真人面前是假的；打开后 ``spawn_args()`` 会带 ``cheat=on``，日志里也会打
    ``[cheat] reach 超限=…（仅对照实验）``。
    """

    username: str
    env_url: str = "ws://127.0.0.1:8766"
    bot_id: str | None = None
    cps: float = 15.0
    cps_jitter: float = 2.0
    reaction_ms: float = 220.0
    reaction_jitter_ms: float = 40.0
    reach: float = 3.0
    #: 显式 reach 作弊开关（默认关）：只有它为 True 时 ``reach`` 才能超过 3.0。
    cheat_reach: bool = False
    model: str = "default"
    #: 头顶标签角色（BetterNameTag）：trained = 被训练方，其它见 code 驱动的策略名。
    role: str = "trained"
    #: 战斗辅助（“外挂级”能力，默认关闭，课程后期逐个打开）。
    auto_rod: bool = False
    auto_block: bool = False
    wtap: bool = False
    aim_lock: bool = False

    def __post_init__(self) -> None:
        if self.bot_id is None:
            self.bot_id = f"glad-{self.username.lower()}"
        # G-708：reach 口径与 Java BotProfile 一致（非法值直接报错，不要静默夹住）。
        self.reach = check_reach(self.reach, cheat=self.cheat_reach)

    def spawn_args(self) -> str:
        """拼成 ``/gladiator spawn`` 的 key=value 参数串。"""
        args = (
            f"cps={self.cps:g} jitter={self.cps_jitter:g} "
            f"reaction={self.reaction_ms:g} rjitter={self.reaction_jitter_ms:g} "
            f"reach={self.reach:g} model={self.model} "
            f"wtap={'on' if self.wtap else 'off'} "
            f"autorod={'on' if self.auto_rod else 'off'} "
            f"autoblock={'on' if self.auto_block else 'off'} "
            f"aimlock={'on' if self.aim_lock else 'off'}"
            + (" cheat=on" if self.cheat_reach else "")
        )
        return args


@dataclass
class ArenaConfig:
    """训练场地参数：普通世界里的一个平坦点，Bot 沿 X 轴对称摆开。"""

    center: tuple[float, float, float] = (0.5, 4.0, 0.5)
    #: 双方初始间距（格）。6 格约等于两把剑的攻击距离，给准备期留出缓冲，
    #: 避免一开局就贴脸（旧值 3 格时对手几乎站在攻击范围内）。
    spacing: float = 6.0
    #: 开局准备期（秒）：发令前压住双方、再校验一次套装。0 表示不等待。
    prep_seconds: float = 3.0
    #: 起始分布随机化（P1.6）：(中心偏移, 间距抖动)，单位格。每局用 np_random 采样一次，
    #: 同一局内两只 Bot 共用同一组参数（只在 X 轴、Y 轴偏移中心，间距各自对称展开）。
    #: ``(0.0, 0.0)`` = 完全确定（旧行为）。
    jitter: tuple[float, float] = (2.0, 1.0)
    #: 每次 reset 是否清背包（空手训练必须为 True）。
    clear_inventory: bool = True
    #: 每次 reset 是否把血量回满（1.8 用 Instant Health IV）。
    heal: bool = True
    #: reset 后等待 Bot 到位的超时。
    settle_timeout: float = 10.0
    #: 死亡后等待自动重生的超时。
    respawn_timeout: float = 15.0
    #: 是否发放默认套件（训练切物品/用物品/左右键交互）。
    give_kit: bool = True
    #: 默认套件：全套钻石甲 + 钻石剑 + 5 金苹果 + 鱼竿。
    #:
    #: 顺序有意义：1.8 的 ``give`` 会按“第一个空槽位”放置，先给剑能保证它落在
    #: 快捷栏第 0 格；护甲会直接进护甲槽，放最后不影响快捷栏布局。
    kit: tuple[tuple[str, int], ...] = (
        ("diamond_sword", 1),
        ("golden_apple", 5),
        ("fishing_rod", 1),
        ("diamond_helmet", 1),
        ("diamond_chestplate", 1),
        ("diamond_leggings", 1),
        ("diamond_boots", 1),
    )


MIN_SPACING = 4.0

#: 对手池的每局随机化区间（P1.6）：模式 -> (attackProb 区间, approachThreshold 区间)。
#: 固定节奏的对手会让策略过拟合，所以每局重采一次，评估分布也更宽。
OPPONENT_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    # P1.7：新增「起手延迟」与「出手间隔」两维。
    # rusher 之前是 attackProb 0.96 + approachThreshold 4.0：几乎不给反应窗口，
    # 训练里根本没有「被先手压制」的样本 → 该对手 0/8。给它 0.5–2s 起手延迟后，
    # 学习方才有机会学到「开局先站住/先手」。
    "rusher": {"attackProb": (0.80, 0.98), "approachThreshold": (3.0, 4.0),
               "startDelayTicks": (10, 40), "attackCooldownTicks": (1, 3)},
    "scripted": {"attackProb": (0.70, 0.90), "approachThreshold": (2.4, 3.2),
                 "startDelayTicks": (20, 60), "attackCooldownTicks": (2, 5)},
    "aimbot": {"attackProb": (0.90, 1.00), "approachThreshold": (3.4, 4.2),
               "startDelayTicks": (0, 20), "attackCooldownTicks": (1, 2)},
}
#: 取整的区间键（用 np_random.integers 采样，结果是 int）。
INT_PROFILE_KEYS = frozenset({"startDelayTicks", "attackCooldownTicks"})
#: 所有模式共用的「对手 reach 折减」区间：对手不会用到满格 reach（给学习方留空间）。
OPPONENT_REACH_OFFSET = (0.5, 1.0)
#: 非池化模式（random/wander/idle）使用的固定折减，保持旧行为。
DEFAULT_REACH_OFFSET = 0.75
#: 对手有效 reach 的下限。
MIN_OPPONENT_REACH = 1.5


class GladiatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bots: list[BotConfig] | list[str],
        arena: ArenaConfig | None = None,
        *,
        rcon: RconClient | dict[str, Any] | None = None,
        control_hz: float | None = None,
        action_ticks: int = 2,
        max_steps: int = 600,
        reward_fn: RewardFn | None = None,
        single_agent: bool = False,
        links: list[Any] | None = None,
        link_factory: Callable[[BotConfig], Any] | None = None,
        connect: bool = True,
        auto_spawn: bool = False,
        spawn_timeout: float = 60.0,
        observation_timeout: float = 2.0,
        idle_threshold: float = 5.0,
        wait_action_results: bool = False,
        opponent_mode: str = "random",
        opponent_policy: Any | None = None,
        opponent_epsilon: float = 0.0,
        reward_mode: str = "shaped",
        shaping_weights: ShapingWeights | None = None,
        reward_preset: dict | None = None,
        action_space: str = "v1",
        logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.bots = [b if isinstance(b, BotConfig) else BotConfig(str(b)) for b in bots]
        if len(self.bots) < 1:
            raise ValueError("至少需要一个 Bot 才能建立训练环境")
        self.arena = arena or ArenaConfig()
        #: 本局对手（index != 0）的行为参数（reset 按 opponent_mode 采样）。
        self._opp_profile: dict[str, float] = {}
        #: 起手延迟剩余 tick（P1.7）：>0 时该对手只 idle，不移动也不出手。
        self._opp_hold_ticks: dict[str, int] = {}
        #: 下一次允许出手的 step 号（P1.7 出手间隔）；未到点的 attack 退化为靠近。
        self._opp_next_attack_tick: dict[str, int] = {}
        #: 本局实际使用的中心/间距（reset 按 jitter 采样；默认等于配置值）。
        self._arena_center: tuple[float, float, float] = tuple(self.arena.center)  # type: ignore[assignment]
        self._arena_spacing: float = float(self.arena.spacing)
        # 两个 Bot 站在同一点会互相卡住：死亡后在原地复活、对手又贴着打，补给永远发不下去。
        # 这里兜底一个最小间距，避免表单里留空/填 0 直接把整个训练变成“出生点互砍”。
        if self.arena.spacing < MIN_SPACING:
            self.arena = replace(self.arena, spacing=MIN_SPACING)
        self.action_ticks = max(1, int(action_ticks))
        self.max_steps = max(1, int(max_steps))
        # 动作空间版本（规格 §2.5）：v1 = 11 个宏动作（保留作冷启动基线/回归对照）；
        # v2 = 9 个正交意图头 + action mask（物品无关，加物品不用重训）。
        self.action_space_name = str(action_space or "v1").strip().lower()
        if self.action_space_name not in ("v1", "v2"):
            raise ValueError(f"未知动作空间: {action_space}（允许 v1 / v2）")
        self._action_v2 = self.action_space_name == "v2"
        #: v2 的物品能力表（算 action mask 用）；测试可替换成自定义 ItemTable。
        self.items = default_table()
        self.reward_fn: RewardFn = reward_fn or default_reward
        self.single_agent = bool(single_agent)
        self.auto_spawn = auto_spawn
        self.spawn_timeout = spawn_timeout
        self.observation_timeout = observation_timeout
        # G-405：空闲窗口 < idle_threshold 秒时训练循环只记录，不触发任何外部查询。
        self.idle_threshold = float(idle_threshold)
        # 默认不等 ACTION_RESULT：动作在运行时异步执行，等结果会把每个 Bot 串行拖慢。
        self.wait_action_results = bool(wait_action_results)
        self._last_active_at: list[float] = []
        # 对手行为（课程学习用）：random=随机动作，wander=只走不打，idle=站着，scripted=简单逼近+攻击。
        self.opponent_mode = str(opponent_mode)
        if self.opponent_mode not in ("random", "wander", "idle", "scripted", "rusher", "aimbot"):
            raise ValueError(f"未知 opponent_mode: {opponent_mode}")
        # 自博弈：对手由**策略**驱动（传 Policy 或 .npz 路径）。非 None 时忽略 opponent_mode
        # 的手写控制器（含起手延迟/出手间隔这些「手写补偿」），完全由策略决定动作。
        self.opponent_policy, self.opponent_policy_source = _coerce_opponent_policy(
            opponent_policy, obs_dim=OBS_DIM)
        if self.opponent_policy is not None and not self.single_agent:
            # 多智能体模式下动作表由调用方提供（环境不替对手决策），给 opponent_policy
            # 只会「看起来配了但完全不生效」——直接报错，别让它静默失效。
            raise ValueError("opponent_policy 只支持 single_agent=True"
                             "（多智能体时所有动作由调用方传入）")
        self.opponent_epsilon = float(opponent_epsilon or 0.0)
        if not 0.0 <= self.opponent_epsilon <= 1.0:
            raise ValueError(f"opponent_epsilon 必须在 [0, 1]，收到 {opponent_epsilon}")
        #: 对手策略是不是 **v2 意图策略**（G-707）。
        #: 为 True 时对手给的是 10 头数组，``step`` 会走 ``decode_intent`` 那条路径；
        #: 为 False 时仍是 v1 的 11 路宏动作编号（历史行为，保持不变）。
        self._opponent_v2 = bool(
            getattr(self.opponent_policy, "is_v2", False)
            or str(getattr(self.opponent_policy, "action_space", "v1")).strip().lower() == "v2")
        #: 对手（策略驱动）最近一次真下发 attack 的时刻，观测 self_since_attack 用。
        self._opponent_last_attack_at: dict[str, float] = {}
        #: 对手策略的运行统计（累计，不随 reset 清零）：决策数/随机探索数/动作失败数。
        self.opponent_policy_stats: dict[str, Any] = {
            "decisions": 0,
            "epsilonPicks": 0,
            "emptyObservations": 0,
            "invalidActions": 0,
            "errors": 0,
            "actionFailures": {},
            "lastAction": None,
            "lastGreedy": None,
        }
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in ("default", "shaped"):
            raise ValueError(f"未知 reward_mode: {reward_mode}")
        if shaping_weights is not None and reward_preset is not None:
            raise ValueError("shaping_weights 与 reward_preset 只能给一个")
        if reward_preset is not None:
            weights_kwargs, terms = reward_presets.preset_for_env(reward_preset)
            self.shaping = ShapingReward(ShapingWeights(**weights_kwargs), terms=terms)
            self.reward_preset_name = str((reward_preset or {}).get("name") or "")
        else:
            self.shaping = ShapingReward(shaping_weights)
            self.reward_preset_name = "builtin" if shaping_weights is None else "custom-weights"

        self._last_events: dict[str, list[dict]] = {}
        #: 最近一次 step 真正下发的协议动作（按用户名）：复盘/BC 诊断用。
        self._last_dispatch: dict[str, list[dict]] = {}
        self._last_attacked: dict[str, bool] = {}
        self._last_distance: dict[str, float | None] = {}
        #: 上次下发 attack 动作的时刻（P1：观测 self_since_attack 用，初始视为已冷却）。
        self._last_attack_at: dict[str, float] = {}
        #: G-711 跨帧观测：上一帧策略选中的十头动作（v1 / 无意图 = ``None``，观测填 0）。
        self._prev_action: dict[str, tuple[int, ...] | None] = {}
        #: G-711：距上次出手 / 上次受伤的帧数（0 = 本帧刚发生；饱和值见 ``_CROSS_FRAME_CAP``）。
        self._frames_since_attack: dict[str, int] = {}
        self._frames_since_hit: dict[str, int] = {}
        #: G-731：上一帧编码时「自身朝向读没读到」（观测 37/38 是否有效）。
        #: 缺 yaw 时那两维写 (0,0)，这里留痕，便于「模型学不会转向」时先看数据而不是先怀疑模型。
        self._yaw_available: dict[str, bool] = {}
        #: G-731：缺 yaw 的告警只打一次（每 200ms 一帧，绝不能每帧刷）。
        self._yaw_warned = False
        #: 重生后的「观察档」：这段时间内代码驱动的对手只观察不出手（旧坐标会被传送打飞）。
        self._respawn_grace: dict[str, int] = {}
        #: 对手「出手前先确认」：本 tick 判到 attack 但还没确认时记 1，下一 tick 复核后才真出手。
        self._opponent_hold: dict[str, int] = {}
        #: 待确认的攻击目标实体 id；换了目标要重新确认，避免对着旧目标补刀。
        self._opponent_target: dict[str, int] = {}
        #: 重生补装待确认次数（A2）：give 没有回执，重生后紧跟着的几步要复核是否真的到位。
        self._restock_pending: dict[str, int] = {}
        #: 每个重生周期只打一次「仍缺件」日志，避免刷屏。
        self._restock_logged: set[str] = set()
        #: 补装确认进行到第几 tick（日志用「第 N tick 已补齐」）。
        self._restock_tick: dict[str, int] = {}
        #: 已用完的补装轮数（到期仍缺件就再来一轮，最多 ``_RESTOCK_MAX_ROUNDS`` 轮）。
        self._restock_rounds: dict[str, int] = {}
        #: 逐物品补发记账：``{用户名: {物品名: {"attempts": n, "wait": k}}}``。
        #: ``wait`` 是「还要等几帧才复核」，避免 give 完立刻判失败（金苹果补装时序 bug）。
        self._restock_gives: dict[str, dict[str, dict[str, int]]] = {}
        self._log = logger or (lambda message: print(f"[env] {message}", flush=True))

        self._control_hz = float(control_hz or 0.0)
        self._steps = 0
        self._raw: list[dict] = []
        #: 逐 Bot 的「对手是否着地」推断器（``spaces.GroundEstimator``）。
        #: 协议里的 ``entities[].onGround`` 是 prismarine-entity 的构造默认值（恒 true），
        #: 不可信，所以只能从 vy + y 自行推；每个 Bot 一个实例，因为它吃的是各自那份观测。
        self._ground = [GroundEstimator() for _ in self.bots]

        self._link_factory = link_factory
        if links is not None:
            if len(links) != len(self.bots):
                raise ValueError(f"links 数量（{len(links)}）与 bots（{len(self.bots)}）不一致")
            self.links = list(links)
        else:
            factory = link_factory or (lambda bot: LinkClient(bot.env_url, bot.bot_id or f"glad-{bot.username.lower()}"))
            self.links = [factory(bot) for bot in self.bots]

        self._rcon: RconClient | None = None
        self._rcon_owned = False
        if isinstance(rcon, RconClient):
            self._rcon = rcon
        elif isinstance(rcon, dict):
            self._rcon = RconClient(**rcon)
            self._rcon_owned = True
        # 连接复用：Paper 的 RCON 对「短时间大量新建/断开」很敏感（日志里每局好几行
        # `Rcon connection from`），所以整局生命周期只连一次，reset 只在断线时重连。
        # 初次连接失败不在构造期抛：留到 reset()，保持「失败发生在 reset」的既有语义。
        if self._rcon_owned and self._rcon is not None and not self._rcon.is_connected():
            try:
                self._rcon.connect()
            except Exception as exc:  # noqa: BLE001 - reset() 会重试并抛出
                self._log(f"RCON 初次连接失败（reset 时重试）：{exc}")

        if self.single_agent:
            # 单智能体视角（Gymnasium 标准）：控制 bot_0，其余 Bot 作为对手由环境随机采样动作。
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = (
                spaces.MultiDiscrete(MULTI_DISCRETE_V2) if self._action_v2
                else spaces.Discrete(NUM_ACTIONS)
            )
        else:
            self.observation_space = spaces.Dict({
                self._key(i): spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
                for i in range(len(self.bots))
            })
            self.action_space = spaces.Dict({
                self._key(i): (
                    spaces.MultiDiscrete(MULTI_DISCRETE_V2) if self._action_v2
                    else spaces.Discrete(NUM_ACTIONS)
                )
                for i in range(len(self.bots))
            })

        self._connected = False
        if connect:
            self._ensure_connected()

    # --- Gymnasium API --------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.shaping.reset()
        options = options or {}
        # 对手策略的每局钩子：脚本老师用它清空跨帧记忆与按局统计（否则不同局的状态串味）。
        if self.opponent_policy is not None:
            hook = getattr(self.opponent_policy, "reset_episode", None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:  # noqa: BLE001 - 对手侧的统计不该打断训练
                    self._log(f"对手策略 reset_episode 失败（忽略）: {exc}")
        # P1.6：起始分布随机化。同一局两只 Bot 共用一组中心/间距，保证对称。
        self._sample_arena()
        self._sample_opponent_profile()
        # 起手延迟 / 出手间隔按「本局采样」重置，跨局不残留。
        self._opp_hold_ticks = {
            bot.username: int(self._opp_profile.get("startDelayTicks", 0) or 0)
            for bot in self.bots
        }
        self._opp_next_attack_tick = {bot.username: 0 for bot in self.bots}
        self._steps = 0
        # 新回合：着地推断的「上一帧 y / 连续帧计数」也不能跨局残留
        # （上一局临死前的 y 会让首帧误判，进而污染 ``target_on_ground``）。
        for estimator in self._ground:
            estimator.reset()
        # 新回合：动作确认状态与补装确认都不该跨回合残留。
        self._opponent_hold.clear()
        self._opponent_target.clear()
        self._restock_pending.clear()
        self._restock_logged.clear()
        self._restock_tick.clear()
        self._restock_rounds.clear()
        self._restock_gives.clear()
        self._last_attack_at.clear()
        self._opponent_last_attack_at.clear()
        # G-711：跨帧观测从「不知道」开始——上一帧动作没有（None → 观测全 0），
        # 两个帧计数置饱和（= 很久以前，等价于 since_attack 的缺省 5s）。
        self._prev_action = {bot.username: None for bot in self.bots}
        self._frames_since_attack = {bot.username: CROSS_FRAME_CAP for bot in self.bots}
        self._frames_since_hit = {bot.username: CROSS_FRAME_CAP for bot in self.bots}
        self._ensure_connected()

        if self._rcon is not None and not self._rcon.is_connected():
            # 复用长连接；只有真的断了（或初次没连上）才新建。
            self._rcon.connect()

        if self.auto_spawn and options.get("spawn", True):
            self._spawn_bots()

        self._wait_ready(timeout=self.arena.respawn_timeout)
        if self._rcon is not None and options.get("prepare_arena", True):
            self._prepare_arena()
            self._wait_in_place(timeout=self.arena.settle_timeout)
            self._prep_phase()

        self._raw = [self._latest_raw(i) for i in range(len(self.bots))]
        observation = self._encode_all()
        info = {"step": 0, "arena": self._arena_info(),
                "opponentProfile": self._opponent_profile_info()}
        if self._action_v2:
            # 第一帧就要给掩码：策略在 reset 之后立刻采样，缺 mask 等于开局先撞一次墙。
            info["actionMask"] = self._action_masks()
        if self.single_agent:
            return observation[self._key(0)], info
        return observation, info

    def step(self, action):
        self._steps += 1
        if self.single_agent:
            # 单智能体：只把 bot_0 的决策交给调用方；对手默认用 v1 宏动作（手写控制器）。
            # G-707：策略驱动的对手可能是 **v2 意图策略**（例如 teacher v4 要能表达
            # `sprintReset`），此时它给的是 10 头数组而不是动作编号，原样带给 decode_intent。
            action_map = {self._key(0): action}
            for index in range(1, len(self.bots)):
                action_map[self._key(index)] = self._coerce_action_value(
                    self._opponent_action(index))
        else:
            action_map = action

        dispatch: list[list[dict]] = []
        labels: list[str] = []
        #: G-711：本帧每个 Bot 的策略选择了哪 10 个头（v1 / 宏动作 → None）。
        picked_intents: dict[str, tuple[int, ...] | None] = {}
        if len(self._last_active_at) != len(self.bots):
            self._last_active_at = [time.monotonic() for _ in self.bots]
        opponents = self._opponent_names()
        for i, bot in enumerate(self.bots):
            value = action_map.get(self._key(i), 0)
            # 多智能体 v2：所有 Bot 都走意图；单智能体 v2：学习方走意图；
            # 单智能体 + v2 对手策略（G-707）：对手也走意图。
            use_intent = ((self._action_v2 and (not self.single_agent or i == 0))
                          or (i != 0 and self._opponent_v2))
            intent: dict | None = None
            if use_intent:
                intent = decode_intent(value)
                label = self._intent_label(intent)
                # 只转视角也算“活跃”：否则空闲判定会把「原地瞄准」误计成挂机。
                active = (label != "idle"
                          or intent["look"]["dyaw"] != 0.0
                          or intent["look"]["dpitch"] != 0.0)
                # G-711：记「策略这一帧选了什么」（而不是执行成什么）。跨帧特征要回答的是
                # 「我上一帧往哪走」，被补装/节流改成 idle 的执行结果会把节拍抹平。
                try:
                    picked_intents[bot.username] = tuple(
                        int(item) for item in np.asarray(value, dtype=np.int64).reshape(-1))
                except (TypeError, ValueError):
                    picked_intents[bot.username] = None
            else:
                index = int(value)
                label = ACTION_NAMES[index] if 0 <= index < len(ACTION_NAMES) else "idle"
                active = index != 0
                picked_intents[bot.username] = None
            labels.append(label)
            if active:  # 只有真正下发动作才算“活跃”
                self._last_active_at[i] = time.monotonic()
            raw = self._raw[i] if i < len(self._raw) else {}
            # 代码驱动的对手比学习方多留 0.75 格余量：它没有「奖励」纠偏，而观测到出手之间
            # 有 1 个 tick 的延迟（约 5 格/s × 0.2s ≈ 1.1 格），加上击退会让边界攻击吃 OUT_OF_REACH。
            # 策略驱动的对手不扣这个余量：它的观测里 reach 就是 bot.reach（见 _policy_opponent_action），
            # 编码与解码必须同一个 reach，否则 in_reach=1 的动作会被解码成「先靠近」。
            if i == 0 or self.opponent_policy is not None:
                reach = float(bot.reach)
            else:
                reach = self._opponent_reach(i, float(bot.reach))
            if intent is not None:
                # v2：一帧意图就是一条载荷（{type:"intent", intent:{...}}），
                # 选槽/右键/按住全部由运行时的能力表执行，环境不再拼宏动作。
                actions = [{"type": "intent", "intent": intent}]
            else:
                actions = decode_action(index, raw, bot.username, opponents, reach=reach)
            if i != 0 and self._respawn_grace.get(bot.username, 0) > 0:
                # 刚重生（可能被传送到出生点）：本 tick 只做观察，避免对旧坐标出手。
                self._respawn_grace[bot.username] -= 1
                self._clear_opponent_hold(bot.username)
                actions = [{"type": "idle", "params": {"durationMs": MOVE_MS}}]
            elif i != 0 and self.opponent_policy is None:
                # 对手「出手前先确认」：观测与出手之间隔着一个 tick（≈1.1 格位移）+ 击退，
                # 直接出拳会在边界吃 OUT_OF_REACH。这里把第一次判到的 attack 换成贴近，
                # 下一 tick 用新观测复核距离，仍在射程内才真的挥。
                # 策略驱动的对手不走这一步：它是手写控制器的补偿，不是协议要求。
                actions = self._confirm_opponent_attack(bot, actions)
            if self._restock_pending.get(bot.username, 0) > 0:
                # 重生补装确认中：背包可能还没同步，进攻动作直接不发
                # （否则运行时回 ITEM_MISSING，日志刷屏且这一 tick 白费）。
                if i != 0:
                    # 代码驱动的对手没有「策略动作」需要保护：整条动作换成穿甲/切剑。
                    # 只剥进攻动作时，`_send_use` 的补发会和本 tick 的对手动作撞车，
                    # 运行时常常只执行后者 → 护甲一直 0/4（真机 2026-11-19 01:0x 实测：
                    # 4 个 B 侧 Bot 全裸，A 侧全有甲，且每 tick 补发把 steps/s 打到 1.8）。
                    actions = self._equip_actions()
                elif use_intent:
                    # v2 没有「宏动作」可剥，只能把意图里的进攻位清掉（保留移动/视角/蹲跳）。
                    actions = self._strip_intent_actions(actions)
                else:
                    actions = self._strip_combat_actions(actions)
            if self._has_attack(actions):
                # 只要真的下发了 attack 就记时（含运行时把它改成追击的情况）。
                # 必须在下发时记，而不是算奖励时：观测在这一 tick 末尾生成，
                # 策略要能看到「我刚挥过一刀」（冷却/连击节奏靠这个）。
                self._last_attack_at[bot.username] = time.monotonic()
                if i != 0 and self.opponent_policy is not None:
                    # 对手自己的「距上次攻击」是它观测的一维，单独维护（与学习方对称）。
                    self._opponent_last_attack_at[bot.username] = time.monotonic()
            dispatch.append(actions)
            self._last_dispatch[bot.username] = list(actions)
            self._handle_action_results(i, self._safe_send(i, actions))

        time.sleep(self.action_ticks / self._control_hz)
        if not self.wait_action_results:
            self._drain_results()
        previous = self._raw
        self._raw = self._refresh_raw(previous)
        # A2：观测是 5Hz 采样，1.8 的「死→自动重生」只有 2 tick，采样常常整个跳过：
        # 回合不结束、对手裸奔、补装窗口每 tick 发 give 把 steps/s 打到 1.8。
        # STATE 帧（协议 §2.4）是事件推送、不会漏，这里用它把「刚死的」写回观测。
        deaths = self._apply_state_deaths()
        # A2：重生补装确认（必须在观测刷新之后、算奖励之前，让策略看到补齐后的状态）。
        self._confirm_restock()
        # 兜底体检：局内「死→活」可能整个被刷新跳过（is_out 看不到），这里只看现状。
        if _RESTOCK_CHECK_EVERY > 0 and self._steps % _RESTOCK_CHECK_EVERY == 0:
            self._periodic_loadout_check()
        # G-711：跨帧计数必须在**观测生成之前**推进，这样策略看到的是「本帧刚发生的事」。
        self._update_cross_frame(previous, dispatch, picked_intents)
        observation = self._encode_all()

        terminated = any(is_out(obs) for obs in self._raw)
        if not terminated:
            self._restock_after_respawn(previous)
        truncated = self._steps >= self.max_steps
        rewards = {}
        self._last_events = {}
        self._last_dispatch = {}
        self._last_attacked = {}
        self._last_distance = {}
        now = time.monotonic()
        for i, bot in enumerate(self.bots):
            opponent = self._opponent_index(i)
            distance = self._distance(i, opponent)
            previous_distance = self._last_distance.get(bot.username)
            self._last_distance[bot.username] = distance
            wants_attack = self._has_attack(dispatch[i])
            # 超出 reach 的 attack 根本挥不出去（运行时抛 OUT_OF_REACH），不能算空挥。
            in_reach = distance is None or distance <= float(bot.reach) + 0.2
            # 观测口径的「这一刀打得到」：水平距离 ≤ 自己的 reach（与 spaces.in_reach 同源）。
            in_reach_horizontal = self._in_reach_horizontal(i, opponent, float(bot.reach))
            self._last_attacked[bot.username] = wants_attack and in_reach
            action_name = labels[i] if i < len(labels) else "idle"
            if self.reward_mode == "shaped":
                value, events = self.shaping.compute(
                    bot.username,
                    previous=previous[i] if i < len(previous) else {},
                    current=self._raw[i],
                    opponent_previous=previous[opponent] if opponent is not None and opponent < len(previous) else None,
                    opponent_current=self._raw[opponent] if opponent is not None else None,
                    terminated=terminated,
                    attacked=self._last_attacked[bot.username],
                    in_reach=in_reach_horizontal,
                    action=action_name,
                    distance=distance,
                    previous_distance=previous_distance,
                    now=now,
                    dt=self.action_ticks / max(self._control_hz, 1.0),
                )
                self._last_events[bot.username] = events
                rewards[self._key(i)] = float(value)
                continue
            ctx = RewardContext(
                username=bot.username,
                index=i,
                previous=previous[i] if i < len(previous) else {},
                current=self._raw[i],
                step=self._steps,
                terminated=terminated,
                truncated=truncated,
                opponent_previous=previous[opponent] if opponent is not None and opponent < len(previous) else None,
                opponent_current=self._raw[opponent] if opponent is not None else None,
            )
            rewards[self._key(i)] = float(self.reward_fn(ctx))

        info = {
            "step": self._steps,
            "arena": self._arena_info(),
            "opponentProfile": self._opponent_profile_info(),
            "health": {bot.username: self_health(self._raw[i]) for i, bot in enumerate(self.bots)},
            "states": {bot.username: str(self._raw[i].get("state", "Alive")) for i, bot in enumerate(self.bots)},
            "winner": self._winner(),
        }
        if self._action_v2:
            # dict head -> list[bool]（单智能体）/ {bot_key: dict head -> list[bool]}（多智能体）。
            # 只加 info，不改 reward/termination，也不改观测维度。
            info["actionMask"] = self._action_masks()
        if self.single_agent:
            return observation[self._key(0)], rewards[self._key(0)], terminated, truncated, info
        return observation, rewards, terminated, truncated, info

    @property
    def raw_observations(self) -> list[dict]:
        """最近一次 step 的原始观测（复盘用）。"""
        return list(self._raw)

    @property
    def yaw_available(self) -> dict[str, bool]:
        """G-731 诊断：上一帧编码时各 Bot 的 ``self.yaw`` 是否读到了。

        ``False`` 说明观测 37/38 恒为 ``(0,0)``——此时「模型学不会转向」不是模型的问题，
        而是协议里没有朝向。训练报告与排障先看这个。
        """
        return dict(self._yaw_available)

    @property
    def opponent_mode_label(self) -> str:
        """报告/统计用的对手标签：策略驱动时是 ``policy:<快照名>``。

        ``opponent_mode`` 本身不动（它描述的是**手写**控制器的模式，策略驱动的对手
        根本不看它）；只是别让报告把自博弈的对局误记成 "scripted"。
        """
        if self.opponent_policy is not None:
            source = self.opponent_policy_source or "memory"
            name = os.path.basename(source) if source != "memory" else "memory"
            return f"policy:{name}"
        return self.opponent_mode

    def last_events(self) -> dict[str, list[dict]]:
        """最近一次 step 的塑形事件（hit/whiff/combo/kill/death）。"""
        return {name: list(events) for name, events in self._last_events.items()}

    def last_dispatch(self, username: str) -> list[dict]:
        """最近一次 step 真正下发给某个 Bot 的协议动作（复盘口径核对用）。

        注意与「策略选了哪个动作编号」不同：``decode_action`` 可能把越界的 attack
        改成「靠近」，所以复盘要按实际下发的动作判断有没有真挥刀。
        """
        return list(self._last_dispatch.get(username, []))

    # --- 动作空间 v2 辅助 -----------------------------------------------

    @staticmethod
    def _intent_label(intent: dict) -> str:
        """意图 → 一个短标签（奖励塑形/日志用）。

        ``action`` 这一维在塑形里只用来识别「block / use_rod 这类无效消耗」，
        v2 没有宏动作名，这里按「进攻 > 用物品 > 移动」的优先级给个近似标签。
        """
        if intent.get("attack"):
            return "attack"
        if intent.get("use"):
            return "use"
        return str(intent.get("move") or "idle")

    @staticmethod
    def _has_attack(actions: list[dict]) -> bool:
        """这组动作里有没有「真会挥刀」的意图（兼容 v1 的 attack 与 v2 的 intent）。"""
        for action in actions or []:
            if action.get("type") == "attack":
                return True
            if action.get("type") == "intent" and (action.get("intent") or {}).get("attack"):
                return True
        return False

    @classmethod
    def _strip_intent_actions(cls, actions: list[dict]) -> list[dict]:
        """补装确认期间清掉意图里的进攻位（``attack`` / ``use``），保留移动与视角。

        与 v1 的 :meth:`_strip_combat_actions` 同义：背包还没同步时按右键只会拿到
        ``ITEM_MISSING``，而移动/转向不会受伤。
        """
        stripped: list[dict] = []
        for action in actions or []:
            if action.get("type") != "intent":
                stripped.append(action)
                continue
            intent = dict(action.get("intent") or {})
            intent["attack"] = False
            intent["use"] = None
            stripped.append({"type": "intent", "intent": intent})
        return stripped or [{"type": "idle", "params": {"durationMs": MOVE_MS}}]

    def _action_masks(self) -> dict:
        """当前每个 Bot 的 v2 action mask（单智能体只返回学习方那一份）。"""
        masks = {
            self._key(i): action_mask(self._raw[i] if i < len(self._raw) else {}, self.items)
            for i in range(len(self.bots))
        }
        return masks[self._key(0)] if self.single_agent else masks

    def _in_reach_horizontal(self, index: int, opponent: int | None, reach: float) -> bool:
        """水平距离是否在 reach 内（缺数据时返回 False，不白送奖励）。"""
        if opponent is None or opponent >= len(self._raw) or index >= len(self._raw):
            return False
        a = (self._raw[index].get("self") or {}).get("pos") or {}
        b = (self._raw[opponent].get("self") or {}).get("pos") or {}
        try:
            dx = float(a["x"]) - float(b["x"])
            dz = float(a["z"]) - float(b["z"])
        except (KeyError, TypeError, ValueError):
            return False
        return (dx * dx + dz * dz) ** 0.5 <= reach

    def idle_snapshot(self) -> dict:
        """G-405：空闲状态快照（不进 ``info``，避免破坏 Gymnasium 的确定性检查）。

        返回 ``{"seconds": {username: float}, "idle": {username: bool}, "events": {username: [...]}}``。
        只有超过 ``idle_threshold`` 才置 ``idle=True``；本方法不会触发任何外部查询。
        """
        now = time.monotonic()
        seconds = {
            bot.username: round(now - self._last_active_at[i], 3)
            for i, bot in enumerate(self.bots)
        }
        events: dict[str, list[dict]] = {}
        for i, bot in enumerate(self.bots):
            getter = getattr(self.links[i], "idle_events", None)
            if callable(getter):
                pending = getter()
                if pending:
                    events[bot.username] = pending
        return {
            "seconds": seconds,
            "idle": {name: value >= self.idle_threshold for name, value in seconds.items()},
            "events": events,
        }

    def close(self) -> None:
        for link in self.links:
            try:
                link.stop()
            except Exception:  # noqa: BLE001 - 关闭阶段不再抛错
                pass
        # 只有自己创建的连接才由自己断开（调用方传进来的连接归调用方管）。
        if self._rcon is not None and self._rcon_owned:
            self._rcon.close()
        self._rcon = None
        self._connected = False

    # --- 观测/动作辅助 --------------------------------------------------

    def _distance(self, index: int, opponent: int | None) -> float | None:
        """两个 Bot 的欧氏距离（观测里的 self.pos）。"""
        if opponent is None or opponent >= len(self._raw):
            return None
        a = (self._raw[index].get("self") or {}).get("pos") or {}
        b = (self._raw[opponent].get("self") or {}).get("pos") or {}
        try:
            return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5
        except (KeyError, TypeError):
            return None

    # --- 对手出手确认（A1） ---------------------------------------------

    def _clear_opponent_hold(self, username: str) -> None:
        self._opponent_hold.pop(username, None)
        self._opponent_target.pop(username, None)

    def _confirm_opponent_attack(self, bot: BotConfig, actions: list[dict]) -> list[dict]:
        """把「本 tick 第一次判到的 attack」降级成贴近，下一 tick 复核后再出手。

        - 只对代码驱动的对手生效（学习方的动作完全不经过这里）；
        - 不引入随机性：连续两个 tick 都判到同一目标 attack 才真正出手；
        - 中间换目标或不再判 attack → 状态清空，重新开始确认。
        """
        username = bot.username
        attack = next((item for item in actions if item.get("type") == "attack"), None)
        if attack is None:
            # 本 tick 不打算出手（例如上一 tick 确认后距离变大、已退化为 move）：清掉待确认。
            self._clear_opponent_hold(username)
            return actions

        target_id = (attack.get("params") or {}).get("targetEntityId")
        held = int(self._opponent_hold.get(username) or 0)
        same_target = self._opponent_target.get(username) == target_id
        if held <= 0 or not same_target:
            self._opponent_hold[username] = 1
            self._opponent_target[username] = target_id
            return self._approach_only(actions)
        # 连续两次判到同一目标且仍在射程内：真正出手，并清掉确认状态。
        self._clear_opponent_hold(username)
        return actions

    @staticmethod
    def _approach_only(actions: list[dict]) -> list[dict]:
        """保留朝向、把 attack 换成疾跑靠近（贴近/保持距离）。"""
        look = next((item for item in actions if item.get("type") == "look"), None)
        move = {"type": "move", "params": {
            "direction": "forward", "durationMs": MOVE_MS, "sprint": True}}
        return ([look] if look is not None else []) + [move]

    def _profile(self) -> dict[str, float]:
        """本局对手参数；未采样时给各模式的默认值（构造后未 reset 的调用方兜底）。"""
        if self._opp_profile.get("attackProb") is not None and "attackProb" in self._opp_profile:
            return self._opp_profile
        mode = self.opponent_mode
        defaults = {
            "rusher": {"attackProb": 0.96, "approachThreshold": 3.4},
            "scripted": {"attackProb": 0.85, "approachThreshold": 2.8},
            "aimbot": {"attackProb": 1.0, "approachThreshold": 3.9},
        }
        base = {"mode": mode, "reachOffset": float(DEFAULT_REACH_OFFSET),
                "startDelayTicks": 0, "attackCooldownTicks": 0}
        base.update(defaults.get(mode, {"attackProb": 1.0, "approachThreshold": 2.8}))
        return base

    def _opponent_ready(self, index: int) -> bool:
        """对手是否已过「起手延迟」（P1.7）。

        未就绪时 ``_opponent_action`` 只回 idle——这样训练里才会出现
        「对手还在起手、我先动手」的窗口，而不是开局就被 0.96 概率连打。
        """
        username = self.bots[index].username if 0 <= index < len(self.bots) else ""
        return int(self._opp_hold_ticks.get(username, 0)) <= 0

    def _opponent_cooldown(self) -> int:
        return int(self._opp_profile.get("attackCooldownTicks", 0) or 0)

    def _opponent_action(self, index: int) -> int:
        """非受控 Bot 的动作：课程学习的不同难度。"""
        if self.opponent_policy is not None and index != 0:
            # 自博弈：对手也走策略（不看 opponent_mode，也没有起手延迟/出手间隔）。
            return self._policy_opponent_action(index)
        username = self.bots[index].username if 0 <= index < len(self.bots) else ""
        if not self._opponent_ready(index):
            # 起手延迟：只观察，不动也不打。
            self._opp_hold_ticks[username] = max(0, int(self._opp_hold_ticks.get(username, 0)) - 1)
            return 0
        mode = self.opponent_mode
        if mode == "idle":
            return 0
        if mode == "wander":
            return int(self.np_random.choice([1, 3, 4, 6]))  # approach/strafe/jump，不打人
        if mode in ("rusher", "aimbot", "scripted"):
            # 三种池化模式共用一套判定，参数来自本局采样的 profile（P1.6）：
            # 阈值/出手概率每局不同，策略不能只记住一种节奏。
            profile = self._profile()
            observation = self._raw[index] if index < len(self._raw) else {}
            target = nearest_player(observation, self.bots[index].username, self._opponent_names())
            if target is None:
                return 1 if mode != "scripted" else int(self.np_random.choice([1, 6]))
            distance = float(target.get("distance", 99))
            if distance > profile["approachThreshold"]:
                return 1
            if self.np_random.random() < profile["attackProb"]:
                # 出手间隔：没到点的攻击退化为靠近（保持"想打但打不了"的压迫感）。
                if self._steps < int(self._opp_next_attack_tick.get(username, 0)):
                    return 1
                self._opp_next_attack_tick[username] = self._steps + self._opponent_cooldown()
                return 2
            return 6
        return int(self.np_random.integers(NUM_ACTIONS))

    def _opponent_act(self, vector, *, greedy: bool):
        """按**对手自己的动作空间**取一帧动作（G-703）。

        老师（``TeacherV4``）和 v2 策略快照都返回 10 个头，但只有老师实现了 ``act()``：
        ``Policy.act()`` 是 v1 的接口，在 v2 上会主动抛错（防止调用方把 10 个头当动作编号用）。
        所以这里必须分叉——此前统一走 ``act()``，v2 **策略**对手会每帧抛异常、被下面的
        ``except`` 吞成「随机动作」，表面能跑、实则全程乱打（联赛排出来的序也就没意义了）。

        ``TeacherV4`` 有 ``is_v2=True`` 但没有 ``sample()``，仍走 ``act()``，行为不变。
        """
        policy = self.opponent_policy
        if getattr(policy, "is_v2", False) and hasattr(policy, "sample"):
            return policy.sample(vector, self.np_random, greedy=greedy)
        return policy.act(vector, self.np_random, greedy=greedy)

    def _policy_opponent_action(self, index: int) -> int:
        """策略驱动的对手动作（自博弈）。

        观测来源：``self._raw[index]`` —— 就是该 Bot 通道 B 上最近一帧 OBSERVATION
        （``_refresh_raw`` 用 ``link.wait_observation`` 取回、与学习方 ``_encode_all``
        用的是同一份数据），不需要另开协议。

        编码口径与学习方完全一致（同一个 ``encode_observation``）：把**对手自己**当
        ``username``、``opponent_health`` 取学习方血量、``reach`` 用对手自己的、
        ``since_attack`` 用 ``_opponent_last_attack_at``。返回动作编号（v1 策略）或
        10 头数组（v2 策略，见 ``_opponent_v2``），decode + 发送沿用 ``step()`` 里
        与学习方相同的那条路径。

        ``opponent_epsilon``：以该概率把动作**换成随机动作**（策略仍会被调用一次——
        见 :meth:`_opponent_act`，v2 走 ``sample()``、v1 走 ``act()``——统计里能对上步数，
        也便于观察策略到底想干什么）。
        """
        bot = self.bots[index]
        stats = self.opponent_policy_stats
        raw = self._raw[index] if index < len(self._raw) else {}
        if not raw:
            # 还没收到该 Bot 的观测帧：这一 tick 只能 idle（不去猜坐标）。
            stats["emptyObservations"] += 1
            return 0
        target = self._opponent_index(index)
        opponent_health = (self_health(self._raw[target])
                           if target is not None and target < len(self._raw) else None)
        last_attack = self._opponent_last_attack_at.get(bot.username)
        since_attack = 5.0 if last_attack is None else max(0.0, time.monotonic() - last_attack)
        fraction = episode_fraction(self._steps, self.max_steps)
        vector = encode_observation(
            raw,
            bot.username,
            step_fraction=fraction,
            opponent_health=opponent_health,
            opponents=self._opponent_names(),
            reach=float(bot.reach),
            since_attack=since_attack,
            # G-711：对手策略的观测与学习方同口径（同一份跨帧状态，按自己的用户名取）。
            prev_action=self._prev_action.get(bot.username),
            frames_since_attack=self._frames_since_attack.get(bot.username),
            frames_since_hit=self._frames_since_hit.get(bot.username),
            # 着地推断与学习方同源：同一 step 的 frame 幂等（先到者推进，后到者只读），
            # 所以这里既可能推进、也可能直接拿到 _encode_all 已经算好的结果。
            opponent_ground=self._ground[index].update(raw.get("entities") or [],
                                                       frame=self._steps),
        )
        explore = float(self.np_random.random()) < self.opponent_epsilon
        try:
            action = self._opponent_act(vector, greedy=not explore)
        except Exception as exc:  # noqa: BLE001 - 坏快照不该打断整局采样
            stats["errors"] += 1
            if stats["errors"] == 1:
                self._log(f"{bot.username}: 对手策略异常（后续只计数）: "
                          f"{type(exc).__name__}: {exc}")
            action = 0
        stats["decisions"] += 1
        if self._opponent_v2:
            # v2 对手（teacher v4 / 自博弈 v2 快照）：动作是 10 头数组，不做 v1 的编号校验。
            picked = np.asarray(action, dtype=np.int64).reshape(-1)
            if explore or picked.size != len(MULTI_DISCRETE_V2):
                # 探索时按各头均匀采样（与 env 的 MultiDiscrete 同分布）；
                # 头数对不上说明策略与当前动作空间不一致，退回随机而不是把越界值发出去。
                if not explore:
                    stats["invalidActions"] += 1
                picked = np.asarray(
                    [self.np_random.integers(0, size) for size in MULTI_DISCRETE_V2],
                    dtype=np.int64)
                if explore:
                    stats["epsilonPicks"] += 1
            stats["lastAction"] = [int(v) for v in picked]
            stats["lastGreedy"] = not explore
            return picked
        action = int(action)
        if explore:
            action = int(self.np_random.integers(NUM_ACTIONS))
            stats["epsilonPicks"] += 1
        if not 0 <= int(action) < NUM_ACTIONS:
            stats["invalidActions"] += 1
            action = 0
        stats["lastAction"] = int(action)
        stats["lastGreedy"] = not explore
        return int(action)

    @staticmethod
    def _coerce_action_value(value: Any) -> Any:
        """对手动作归一：v2 是头数组（保持数组），v1 是动作编号（转 ``int``）。

        ``step`` 里对 ``self._opponent_action(index)`` 的返回值统一走这里，
        这样「v1 对手 + v2 对手」可以共存于同一段派发代码，不必分叉。
        """
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False).reshape(-1)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.int64).reshape(-1)
        return int(value)

    @staticmethod
    def _key(index: int) -> str:
        return f"bot_{index}"

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        for link in self.links:
            if hasattr(link, "start"):
                link.start()
        deadline = time.monotonic() + self.spawn_timeout
        for link in self.links:
            if hasattr(link, "wait_welcome"):
                remaining = max(0.5, deadline - time.monotonic())
                link.wait_welcome(timeout=remaining)
        hz = self._control_hz
        if not hz:
            for link in self.links:
                value = getattr(link, "control_hz", None)
                hz = value() if callable(value) else value
                if hz:
                    break
        self._control_hz = float(hz or 10.0)
        self._connected = True
        self._log(f"通道 B 已连接：{len(self.links)} 个 Bot，controlHz={self._control_hz:g}")

    def _spawn_bots(self) -> None:
        if self._rcon is None:
            raise LinkError("auto_spawn 需要配置 rcon（调用方编排接口）")
        for bot in self.bots:
            self._command(f"gladiator spawn {bot.username} {bot.spawn_args()}")
        deadline = time.monotonic() + self.spawn_timeout
        while time.monotonic() < deadline:
            if all(self._has_spawned(i) for i in range(len(self.bots))):
                self._log("所有 Bot 已登录并绑定")
                return
            time.sleep(0.5)
        raise TimeoutError(f"等待 Bot 登录超时（{self.spawn_timeout:.0f}s）")

    def _has_spawned(self, index: int) -> bool:
        raw = self.links[index].latest() if hasattr(self.links[index], "latest") else None
        return bool(raw) and self_health(raw) > 0.0

    def _prepare_arena(self) -> None:
        assert self._rcon is not None
        # 整段摆位/发装备期间都给双方套抗性 V：否则先被武装起来的一方（或上一回合残留的
        # 攻击）能把还在等装备的对手砍死，出现“收不到装备两刀就死”的死亡循环。
        self._set_resistance(max(5.0, self.arena.prep_seconds + 15.0))
        # 上一回合可能刚好死在摆位前，先等重生再发装备（死亡状态下 give 会失败）。
        self._wait_ready(timeout=min(10.0, self.arena.respawn_timeout))
        for index, bot in enumerate(self.bots):
            x, y, z = self._spawn_point(index)
            self._command(f"gamemode 0 {bot.username}")
            if self.arena.clear_inventory:
                self._command(f"clear {bot.username}")
            if self.arena.heal:
                self._command(f"effect {bot.username} clear")
                self._command(f"effect {bot.username} 6 1 120 4 true")
            if self.arena.give_kit:
                for item, count in self.arena.kit:
                    self._command(f"give {bot.username} {item} {count}")
                # 套件发完显式让运行时穿护甲：1.8.8 下 RCON give 不一定触发背包 update 事件，
                # 只靠运行时监听会出现“有甲不穿”。
                self._send_use(index, {"item": "armor"})
            self._command(f"spawnpoint {bot.username} {int(round(x))} {int(round(y))} {int(round(z))}")
            self._command(f"tp {bot.username} {x:.1f} {y:.1f} {z:.1f}")
            self._tag_bot(index, bot)

        if self.arena.give_kit:
            for index, bot in enumerate(self.bots):
                self._ensure_loadout(index, bot)
        if self.arena.prep_seconds <= 0.0:
            # 没有准备期时，摆位结束就解除无敌，避免把抗性带进对局。
            self._set_resistance(0.0)

    def _prep_phase(self) -> None:
        """开局准备期：确认穿好护甲、双方站定之后才放行。

        真实问题：观战里能看到“还没穿护甲就开打”。``clear``/``give``/穿甲都是异步命令，
        只要有一条没生效（或准备期里被对手推挤/打死掉装备），就会带着残缺套装进入对局。
        这里在发令前压住双方并持续校验套装，避免开局就吃亏。
        """
        seconds = max(0.0, float(self.arena.prep_seconds))
        if seconds <= 0.0:
            return
        self._set_resistance(seconds + 2.0)
        deadline = time.monotonic() + seconds
        next_pulse = 0.0
        next_check = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            # idle 会清掉上一回合残留的移动/疾跑按键，并让 Bot 站着不动。
            if now >= next_pulse:
                next_pulse = now + 0.5
                for index in range(len(self.bots)):
                    self._safe_send(index, [{"type": "idle", "params": {"durationMs": 120}}])
            if self.arena.give_kit and now >= next_check:
                next_check = now + 1.0
                for index, bot in enumerate(self.bots):
                    report = loadout_report(self._latest_raw(index))
                    if report["ok"] and report["armor"] == 4 and not report["holding_sword"]:
                        # 装备齐了只是没拿剑：让运行时把剑挪进快捷栏并选中（不必重发套件）。
                        self._ensure_sword_in_hand(index, bot, report=report)
                        continue
                    if not report["ok"] or report["armor"] < 4 or not report["holding_sword"]:
                        self._log(
                            f"[prep] {bot.username} 装备缺失（护甲 {report['armor']}/4，"
                            f"手持 {report['held'] or '空手'}），重新补发"
                        )
                        self._ensure_loadout(index, bot)
            time.sleep(0.1)
        self._set_resistance(0.0)
        self._log(f"[prep] 准备期 {seconds:g}s 结束，开始对局")
        # 准备期里仍可能被推挤，最后再对齐一次站位。
        self._wait_in_place(timeout=min(5.0, self.arena.settle_timeout))

    def _set_resistance(self, seconds: float) -> None:
        """准备期给双方套抗性 V（100% 免伤），避免发装备/穿甲期间被对手砍死。

        ``seconds <= 0`` 时清掉该效果（1.8 用 ``effect <p> 11 0 0`` 移除）。
        """
        if self._rcon is None:
            return
        for bot in self.bots:
            try:
                if seconds <= 0.0:
                    self._command(f"effect {bot.username} 11 0 0")
                else:
                    self._command(f"effect {bot.username} 11 {int(seconds)} 4 true")
            except LinkError:
                pass

    def _send_use(self, index: int, params: dict) -> None:
        """向某个 Bot 发一条 ``use`` 动作（不等结果）。"""
        try:
            send = getattr(self.links[index], "send_actions_nowait", None)
            if send is None:
                return
            send([{"type": "use", "params": params}])
        except (LinkError, AttributeError, RuntimeError):
            pass

    def _select_sword(self, index: int) -> None:
        self._send_use(index, {"item": "sword", "select": True})

    @staticmethod
    def _equip_actions() -> list[dict]:
        """「先把装备穿好」的一条动作：穿甲 → 切剑 → 站住。

        用于补装窗口里**代码驱动的对手**：它们没有策略动作，直接把整条动作换成
        装备动作，避免 ``_send_use`` 另发的一条 ``use`` 与本 tick 的对手动作互相挤掉。
        """
        return [
            {"type": "use", "params": {"item": "armor"}},
            {"type": "use", "params": {"item": "sword", "select": True}},
            {"type": "idle", "params": {"durationMs": MOVE_MS}},
        ]

    def _ensure_loadout(self, index: int, bot: BotConfig, attempts: int = 3) -> None:
        """校验并补齐套装，最后把主手固定在剑上。

        两个真实问题都在这里收口：
        1. ``clear``/``give`` 没有回执，偶尔出现“装备被吞”（尤其 Bot 在摆位阶段死亡掉物品）；
        2. 手持物不确定——拿着金苹果/鱼竿的对手强度完全不同，会让训练曲线剧烈波动。
        """
        for attempt in range(attempts):
            self._select_sword(index)
            deadline = time.monotonic() + 2.0
            report = loadout_report(self._latest_raw(index))
            while time.monotonic() < deadline:
                report = loadout_report(self._latest_raw(index))
                if report["ok"] and report["holding_sword"]:
                    break
                time.sleep(0.2)
            if report["ok"] and report["holding_sword"]:
                self._log(
                    f"[kit] {bot.username} 套装就绪：手持剑，"
                    f"金苹果 {report['counts'].get('golden_apple', 0)}，护甲 {report['armor']}/4"
                )
                return
            if report["missing"] == [] and report["holding_sword"]:
                # 缺件列表为空说明物品都在背包里；护甲没穿上属于运行时 equip 的问题
                # （1.8 偶发 "Server didn't respond to transaction"），再 give 也没用。
                # 这里直接放行，避免每局开头刷 3 轮 give + 6 秒等待。
                self._log(
                    f"[kit] {bot.username} 物品齐全但护甲只穿上 {report['armor']}/4，"
                    "按现状开打（运行时 equip 超时，已记录）"
                )
                return
            self._log(
                f"[kit] {bot.username} 套装不完整（第 {attempt + 1} 次）："
                f"缺 {report['missing']}，护甲 {report['armor']}/4，手持 {report['held'] or '空手'}"
            )
            for name, need in report["missing"]:
                self._command(f"give {bot.username} {name} {need}")
            self._give_armor(index, bot, report)
        self._log(f"[kit] {bot.username} 校验未通过，继续训练（该局数据可能有偏差）")

    def _ensure_sword_in_hand(self, index: int, bot: BotConfig,
                              report: dict | None = None) -> bool:
        """校验主手是剑；不是就让运行时把剑挪到快捷栏并选中。

        A1：``[kit] ... 手持剑`` 之后仍报 ``ITEM_MISSING 快捷栏里没有 sword`` 的根因在运行时
        （只看快捷栏、不挪背包），运行时已修；这里再补一层环境侧校验，避免空手开局。
        返回当前是否已手持剑（观测尚未刷新时可能仍是 False）。
        """
        report = report or loadout_report(self._latest_raw(index))
        if report["holding_sword"]:
            return True
        self._log(f"[kit] {bot.username} 主手不是剑（{report['held'] or '空手'}），切回剑")
        self._select_sword(index)
        return False

    def _restock_after_respawn(self, previous: list[dict]) -> None:
        """回合中死亡重生后补发套件（A1）。

        1.8.8 死亡会掉落背包，重生后往往是空手；如果不补，剩下的时间等于白送对手一分。
        这里只在「上一帧已死、这一帧活」时触发。

        注意：观测 200ms 刷一次，**局内「死→活」很可能被跳过一个刷新点**，
        这条路径会静默失效——所以它只是一条「能看见就用」的快路径，
        真正兜底的是 :meth:`_request_restock` 的另外两个触发源（运行时失败码 / 周期体检）。
        """
        for index, bot in enumerate(self.bots):
            before = previous[index] if index < len(previous) else None
            if not before:
                continue
            current = self._raw[index] if index < len(self._raw) else {}
            was_dead = self_health(before) <= 0.0 or is_out(before)
            now_alive = self_health(current) > 0.0 and not is_out(current)
            if not (was_dead and now_alive):
                continue
            self._log(f"[kit] {bot.username} 重生（血量 {self_health(current):g}）")
            self._request_restock(index, reason="respawn")

    def _request_restock(self, index: int, reason: str = "") -> bool:
        """给某个 Bot 开一轮补装（三个触发源共用这一条路径）。

        触发源：① 观测到「死→活」（``respawn``）；② 运行时回 ``ITEM_MISSING`` 之类的失败码
        （``code=...``）；③ 每 ``_RESTOCK_CHECK_EVERY`` 步的主动体检（``periodic``）。

        返回 True 表示新开了一轮（打了「触发补装」日志），False 表示只是把已有窗口补足。
        """
        if not 0 <= index < len(self.bots):
            return False
        bot = self.bots[index]
        user = bot.username
        current = int(self._restock_pending.get(user, 0))
        if current > 0:
            # 已在补装窗口里：只把计数补足，不重复打日志（否则每 tick 一条）。
            self._restock_pending[user] = max(current, _RESTOCK_TICKS)
            self._respawn_grace[user] = max(
                self._respawn_grace.get(user, 0), _RESTOCK_TICKS)
            self._restock_rounds.setdefault(user, 0)
            return False

        self._restock_pending[user] = _RESTOCK_TICKS
        self._restock_tick[user] = 0
        self._restock_rounds[user] = 0
        self._restock_logged.discard(user)
        # 新的一轮补装：逐物品的重发预算重新记（上一轮用掉的 3 次不带到这一轮）。
        self._restock_gives.pop(user, None)
        # 观察档与补装窗口同步：补装没确认完之前，代码驱动对手只观察不出手。
        self._respawn_grace[user] = max(
            self._respawn_grace.get(user, 0), _RESTOCK_TICKS)

        raw = self._raw[index] if index < len(self._raw) else None
        report = loadout_report(raw)
        self._log(f"[restock] {user} 触发补装（原因：{reason or 'unknown'}）")
        self._give_missing(index, bot, report)
        return True

    def _give_missing(self, index: int, bot: BotConfig, report: dict) -> None:
        """按 ``loadout_report`` 的结果补件：缺什么 ``give`` 什么 + 重穿护甲 + 切剑。

        **逐物品记账**（golden_apple 补装时序 bug 的修复）：``give`` 是控制台命令，
        没有回执，1.8 的背包同步还要 1–2 tick。旧实现每 tick 把「仍缺」全量重发、
        并在同一 tick 之后就判失败，4 场地的日志里因此反复出现
        ``[kit] 补装失败（缺 [('golden_apple', 5)]，护甲 4/4）``——东西其实发下来了，
        只是还没同步到观测里。现在的规则：

        1. **逐物品**：每个缺失项单独记账（``self._restock_gives``）；
        2. ``give`` 之后至少等 ``_GIVE_SETTLE_TICKS`` 帧才复核该项；
        3. 复核时只重发**仍然缺**的项（``report["missing"]`` 每帧重算）；
        4. 单个物品整轮最多 ``_RESTOCK_MAX_GIVES`` 次，用完就交给下一轮，不再刷 RCON。

        ``give`` 依赖 RCON 与 ``give_kit``；两者都不具备时至少把剑切到手上
        （可能是「有剑但没选中」，运行时能自己挪进快捷栏）。
        """
        user = bot.username
        if self.arena.give_kit and self._rcon is not None:
            state = self._restock_gives.setdefault(user, {})
            for name, need in report["missing"]:
                entry = state.get(name)
                if entry is None:
                    entry = {"attempts": 0, "wait": 0}
                    state[name] = entry
                if entry["attempts"] >= _RESTOCK_MAX_GIVES:
                    # 这个物品已经重发够次数：留给下一轮（或直接放弃），别再灌命令。
                    continue
                if entry["wait"] > 0:
                    # 刚 give 过，背包还没同步：这一帧只观察，不重复下发。
                    continue
                try:
                    self._command(f"give {user} {name} {max(1, int(need))}")
                except LinkError:
                    continue  # RCON 没通不算一次尝试
                entry["attempts"] += 1
                entry["wait"] = _GIVE_SETTLE_TICKS
            self._give_armor(index, bot, report)
        self._select_sword(index)

    def _settle_gives(self, user: str) -> None:
        """把某个 Bot 每个物品的「等待帧数」减一（``give`` 之后的观察窗口）。

        只做记账：减到 0 之后，下一次 :meth:`_give_missing` 才会复核并（仍缺时）重发。
        """
        state = self._restock_gives.get(user)
        if not state:
            return
        for entry in state.values():
            if int(entry.get("wait", 0)) > 0:
                entry["wait"] = int(entry["wait"]) - 1

    def _give_armor(self, index: int, bot: BotConfig, report: dict) -> None:
        """按缺失**槽位**补发护甲，再请运行时穿上。

        护甲掉落是「整件消失」，只发 ``use{item:"armor"}`` 是穿不上的（背包里没有这件）。
        必须先把头盔/胸甲/护腿/靴子 ``give`` 回来，``use{armor}`` 才有东西可穿。
        """
        missing_armor = report.get("missingArmor") or []
        if not missing_armor or not (self.arena.give_kit and self._rcon is not None):
            return
        for slot in missing_armor:
            try:
                self._command(f"give {bot.username} {self._armor_item_for(slot)} 1")
            except LinkError:
                pass
        self._send_use(index, {"item": "armor"})

    def _armor_item_for(self, slot: str) -> str:
        """某个护甲槽位应该发什么物品：优先用 ``ArenaConfig.kit`` 里的同后缀件。"""
        suffix = ARMOR_SLOT_SUFFIX.get(slot)
        if suffix:
            for name, _count in self.arena.kit:
                if str(name).endswith(suffix):
                    return str(name)
        return DEFAULT_ARMOR_ITEMS[slot]

    def _confirm_restock(self) -> None:
        """复核补发的套件是否真的到位（A2）。

        1.8.8 的 ``give`` 没有回执，重生后 1–2 tick 内容易出现「有甲无剑」，运行时于是回
        ``ITEM_MISSING 背包里没有剑``。这里在窗口内持续复核：缺件按数量补发、护甲不足重新穿、
        顺手把主手切回剑。

        窗口（``_RESTOCK_TICKS`` 帧）用完仍缺件时**不再直接放弃**：重置窗口再来一轮，
        最多 ``_RESTOCK_MAX_ROUNDS`` 轮；期间进攻动作一直被剥掉（:meth:`_strip_combat_actions`）。
        全部失败才打一行 ``[kit] <user> 补装失败（已重试 N 轮，缺 ...）``。

        完成判据只看**数量 + 护甲**（``missing == []`` 且护甲 4/4），**不把 ``holding_sword``
        当硬条件**：切剑是策略自己的事，卡在这里只会让窗口 3 轮白跑（日志表现为
        ``补装失败（已重试 3 轮，缺 []）``）。数量齐但没持剑时打一行提示并立即结束窗口。
        """
        for index, bot in enumerate(self.bots):
            left = int(self._restock_pending.get(bot.username, 0))
            if left <= 0:
                continue
            tick = int(self._restock_tick.get(bot.username, 0)) + 1
            self._restock_tick[bot.username] = tick
            current = self._raw[index] if index < len(self._raw) else {}
            report = loadout_report(current)
            # 数量齐 + 护甲满即算补齐；持不持剑交给策略（见方法注释）。
            if report["ok"] and report["armor"] == 4:
                rounds = int(self._restock_rounds.get(bot.username, 0))
                if not report["holding_sword"]:
                    # 每轮最多一次（窗口在马上升级结束，天然只有一条）。
                    self._log(
                        f"[restock] {bot.username} 数量齐但不持剑，等策略自行切换"
                    )
                self._restock_pending.pop(bot.username, None)
                self._restock_logged.discard(bot.username)
                self._restock_tick.pop(bot.username, None)
                self._restock_rounds.pop(bot.username, None)
                self._restock_gives.pop(bot.username, None)
                self._respawn_grace.pop(bot.username, None)
                suffix = f"，第 {rounds + 1} 轮" if rounds else ""
                self._log(f"[restock] {bot.username} 已补齐（第 {tick} tick{suffix}）")
                continue
            # 逐物品复核：先把每个缺失项的「等待帧数」减一，再补发**仍然缺**的项。
            # 旧实现是「隔 tick 全量重发」：give 完当帧就判缺，金苹果因此反复报补装失败。
            # 现在每个物品 give 后至少等 ``_GIVE_SETTLE_TICKS`` 帧，天然限流，不需要再隔 tick。
            self._settle_gives(bot.username)
            self._give_missing(index, bot, report)
            if bot.username not in self._restock_logged:
                self._restock_logged.add(bot.username)
                self._log(
                    f"[kit] {bot.username} 补装中（缺 {report['missing']}，"
                    f"护甲 {report['armor']}/4，手持 {report['held'] or '空手'}）"
                )
            left -= 1
            if left > 0:
                self._restock_pending[bot.username] = left
                # 观察档跟随 pending：确认期间对手不出手。
                self._respawn_grace[bot.username] = max(
                    self._respawn_grace.get(bot.username, 0), left)
                continue

            # 本轮窗口用完仍缺件：还有余量就再来一轮（strip 继续生效），否则放弃。
            rounds = int(self._restock_rounds.get(bot.username, 0)) + 1
            self._restock_rounds[bot.username] = rounds
            if rounds < _RESTOCK_MAX_ROUNDS:
                self._restock_pending[bot.username] = _RESTOCK_TICKS
                self._restock_tick[bot.username] = 0
                # 新一轮重新允许打一次「补装中」日志（每轮最多一条）。
                self._restock_logged.discard(bot.username)
                self._respawn_grace[bot.username] = max(
                    self._respawn_grace.get(bot.username, 0), _RESTOCK_TICKS)
                self._log(
                    f"[kit] {bot.username} 补装第 {rounds} 轮未齐（缺 {report['missing']}），重试"
                )
            else:
                self._restock_pending.pop(bot.username, None)
                self._restock_tick.pop(bot.username, None)
                self._restock_rounds.pop(bot.username, None)
                self._restock_gives.pop(bot.username, None)
                self._restock_logged.discard(bot.username)
                self._respawn_grace.pop(bot.username, None)
                self._log(
                    f"[kit] {bot.username} 补装失败（已重试 {rounds} 轮，"
                    f"缺 {report['missing']}，护甲 {report['armor']}/4）"
                )

    def _periodic_loadout_check(self) -> None:
        """每 ``_RESTOCK_CHECK_EVERY`` 步主动体检一次套装（兜底）。

        局内死亡→复活可能整个被观测跳过（200ms 一次刷新），``is_out`` 根本没看到，
        ``_restock_after_respawn`` 就不会触发。这里不问「死没死」，只看「现在缺不缺」。
        """
        if not self.arena.give_kit:
            # 场地自己不供装（例如自定义补给），主动补装会把规则改乱。
            return
        for index, bot in enumerate(self.bots):
            if self._restock_pending.get(bot.username, 0) > 0:
                continue
            report = loadout_report(self._raw[index] if index < len(self._raw) else None)
            if report["ok"] and report["armor"] == 4 and report["holding_sword"]:
                continue
            if report["ok"] and report["armor"] == 4:
                if self._action_v2:
                    # v2 的 slot 头自己决定手持物：这里再强制切剑会和策略抢控制权
                    # （实测表现为「切鱼竿/金苹果 → 立刻被切回剑」每 tick 刷屏）。
                    continue
                # 只是没拿剑（例如刚吃完金苹果）：切剑即可，不必开补装窗口把战斗动作剥掉。
                self._ensure_sword_in_hand(index, bot, report=report)
                continue
            self._request_restock(index, reason="periodic")

    @staticmethod
    def _is_equip_use(action: dict) -> bool:
        """这条 ``use`` 是不是「装备性」的（切剑 / 穿甲）——补装期间必须放行。"""
        params = action.get("params") or {}
        if params.get("select") is True:
            return True
        return str(params.get("item") or "") in _KEEP_USE_ITEMS

    @classmethod
    def _strip_combat_actions(cls, actions: list[dict]) -> list[dict]:
        """补装确认期间只剥「进攻性」动作，保留装备动作与移动/观察。

        - 剥掉：``attack`` / ``dig`` / ``place``，以及进攻/消耗性的 ``use``
          （``gapple`` / ``rod`` / ``blocks`` / ``food`` 等）；
        - 保留：``look`` / ``move`` / ``jump`` / ``idle``，以及**装备性** ``use``
          （``select=true`` 或 ``item in {armor, sword}``）。

        保留装备性 use 是关键：``_give_missing`` / ``_select_sword`` 走的正是
        ``use{item:"sword",select:true}`` 与 ``use{item:"armor"}``。
        若把它们一起剥掉，补装期间就再也无法切剑 → ``holding_sword`` 永远为假 →
        窗口 3 轮必然失败（``缺 []`` 却 FAILED_KIT）。
        """
        kept = []
        for item in actions:
            action = item or {}
            kind = str(action.get("type") or "")
            if kind in _COMBAT_ACTIONS:
                continue
            if kind == "use" and not cls._is_equip_use(action):
                continue
            kept.append(action)
        if not kept:
            # 全被剥光时给个 idle，保持「每 tick 都有动作」的既有约定。
            return [{"type": "idle", "params": {"durationMs": MOVE_MS}}]
        return kept

    def _tag_bot(self, index: int, bot: BotConfig) -> None:
        """给头顶名称打角色标签（BetterNameTag）：训练方红色，代码驱动按策略着色。

        依赖 caller-sample 的 ``/caller tag``；没装该插件时只是记日志，不中断训练。
        """
        role = bot.role if index == 0 else self.opponent_mode
        try:
            self._command(f"caller tag {bot.username} {role}")
        except Exception as exc:  # noqa: BLE001 - 标签是可视化辅助，不应影响训练
            self._log(f"设置 {bot.username} 的头顶标签失败（{role}）：{exc}")

    def _spawn_point(self, index: int) -> tuple[float, float, float]:
        cx, cy, cz = self._arena_center
        offset = (index - (len(self.bots) - 1) / 2.0) * self._arena_spacing
        return cx + offset, cy, cz

    def _sample_arena(self) -> None:
        """按 ``ArenaConfig.jitter`` 采样本局的中心偏移与间距抖动。

        ``jitter=(0, 0)`` 时**逐位等于**配置值（完全确定，旧行为）；
        间距仍受 ``MIN_SPACING`` 夹取，避免抖出「贴脸开局」。
        """
        cx, cy, cz = (float(v) for v in self.arena.center)
        spacing = float(self.arena.spacing)
        try:
            jitter_center, jitter_spacing = (float(self.arena.jitter[0]),
                                             float(self.arena.jitter[1]))
        except (TypeError, IndexError):
            jitter_center, jitter_spacing = 0.0, 0.0
        jitter_center = abs(jitter_center)
        jitter_spacing = abs(jitter_spacing)
        if jitter_center > 0.0:
            # 只用 x/z 偏移：y 是地面高度，抖它会把 Bot 埋进方块里。
            cx += float(self.np_random.uniform(-jitter_center, jitter_center))
            cz += float(self.np_random.uniform(-jitter_center, jitter_center))
        if jitter_spacing > 0.0:
            spacing += float(self.np_random.uniform(-jitter_spacing, jitter_spacing))
        spacing = max(MIN_SPACING, spacing)
        self._arena_center = (cx, cy, cz)
        self._arena_spacing = float(spacing)

    def _sample_opponent_profile(self) -> None:
        """按当前 ``opponent_mode`` 采样本局的对手参数（P1.6 对手池）。

        - ``rusher``  : attackProb 0.80–0.98、approachThreshold 3.0–4.0
        - ``scripted``: attackProb 0.70–0.90、approachThreshold 2.4–3.2
        - ``aimbot``  : attackProb 0.90–1.00、approachThreshold 3.4–4.2
        - 其它模式（random/wander/idle）保持旧行为，只带固定 reach 折减。

        区间来自 ``OPPONENT_PROFILES``；同一局内固定，换局重采 → 训练/评估都不会只
        见一种节奏。
        """
        spec = OPPONENT_PROFILES.get(self.opponent_mode)
        if spec is None:
            self._opp_profile = {
                "mode": str(self.opponent_mode),
                "reachOffset": float(DEFAULT_REACH_OFFSET),
                # 非池化模式没有 profile：视为「无起手延迟、无出手间隔」，行为不变。
                "startDelayTicks": 0,
                "attackCooldownTicks": 0,
            }
            return
        profile: dict[str, float] = {"mode": str(self.opponent_mode)}
        for key, (low, high) in spec.items():
            if key in INT_PROFILE_KEYS:
                # 闭区间：integers 的上界是不含的，+1 才是 [low, high]。
                profile[key] = int(self.np_random.integers(int(low), int(high) + 1))
            else:
                profile[key] = float(self.np_random.uniform(low, high))
        profile["reachOffset"] = float(self.np_random.uniform(*OPPONENT_REACH_OFFSET))
        self._opp_profile = profile

    def _opponent_profile_info(self) -> dict:
        """给 info / 复盘用的快照（不在池里的模式只返回 mode + reachOffset）。"""
        return {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in self._opp_profile.items()
        }

    def _opponent_reach(self, index: int, reach: float) -> float:
        """对手的有效 reach：``reach - reachOffset``，并夹到 ``MIN_OPPONENT_REACH``。"""
        offset = float(self._opp_profile.get("reachOffset", DEFAULT_REACH_OFFSET))
        return max(MIN_OPPONENT_REACH, float(reach) - offset)

    def _arena_info(self) -> dict:
        return {
            "center": [round(v, 3) for v in self._arena_center],
            "spacing": round(float(self._arena_spacing), 3),
        }

    def _command(self, command: str) -> str:
        assert self._rcon is not None
        try:
            response = self._rcon.command(command)
        except RconError as exc:
            raise LinkError(f"RCON 命令失败（{command}）: {exc}") from exc
        return response

    def _wait_ready(self, timeout: float) -> None:
        """等待每个 Bot 都有观测、且活着（死亡后先等自动重生）。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self._is_alive(index) for index in range(len(self.bots))):
                return
            time.sleep(0.2)
        raise TimeoutError("等待 Bot 存活/就位超时（检查 spawn、死亡重生与通道 B 心跳）")

    def _wait_in_place(self, timeout: float) -> None:
        started = time.monotonic()
        deadline = started + timeout
        last_retry = 0.0
        while time.monotonic() < deadline:
            if all(self._is_alive(index) and self._at_spawn(index, self._latest_raw(index))
                   for index in range(len(self.bots))):
                return
            now = time.monotonic()
            # 场地挨得近时对手（rusher/aimbot）可能在摆位阶段就冲上来把 Bot 撞开，
            # 这里补一次 tp，避免整场训练因为“没站回原点”直接失败。
            if now - started > 3.0 and now - last_retry > 2.0:
                last_retry = now
                for index, bot in enumerate(self.bots):
                    if not self._at_spawn(index, self._latest_raw(index)):
                        x, y, z = self._spawn_point(index)
                        try:
                            self._command(f"tp {bot.username} {x:.1f} {y:.1f} {z:.1f}")
                        except LinkError:
                            pass
            time.sleep(0.2)
        raise TimeoutError("等待 Bot 传送到训练点超时（检查 arena.center 与 RCON/tp 权限）")

    def _is_alive(self, index: int) -> bool:
        raw = self.links[index].latest() if hasattr(self.links[index], "latest") else None
        if not raw:
            return False
        return self_health(raw) > 0.0 and not is_out(raw)

    def _at_spawn(self, index: int, raw: dict) -> bool:
        if self._rcon is None:
            return True
        payload = raw.get("self") or {}
        pos = payload.get("pos") or {}
        x, y, z = self._spawn_point(index)
        try:
            dx = float(pos.get("x", 0.0)) - x
            dz = float(pos.get("z", 0.0)) - z
        except (TypeError, ValueError):
            return False
        return math.hypot(dx, dz) <= 2.0

    def _latest_raw(self, index: int) -> dict:
        link = self.links[index]
        raw = link.latest() if hasattr(link, "latest") else None
        if raw is None:
            raw = link.wait_observation(timeout=self.observation_timeout)
        return raw or {}

    def _refresh_raw(self, previous: list[dict]) -> list[dict]:
        refreshed = []
        for index, link in enumerate(self.links):
            before = previous[index] if index < len(previous) else None
            raw = None
            wait = getattr(link, "wait_observation", None)
            if wait is not None:
                try:
                    raw = wait(timeout=self.observation_timeout, after=before)
                except TypeError:
                    raw = wait(timeout=self.observation_timeout)
                except LinkError:
                    raw = None
            if raw is None and hasattr(link, "latest"):
                raw = link.latest()
            refreshed.append(raw or (before or {}))
        return refreshed

    def _encode_all(self) -> dict[str, np.ndarray]:
        encoded = {}
        # 口径 = 本局已过时间 / 本局时长上限（训练里步长固定，等价于 steps / max_steps）。
        # 运行时按 elapsed / roundSeconds 算同一个量，见 spaces.episode_fraction。
        fraction = episode_fraction(self._steps, self.max_steps)
        for index, bot in enumerate(self.bots):
            opponent = self._opponent_index(index)
            opponent_health = self_health(self._raw[opponent]) if opponent is not None and opponent < len(self._raw) else None
            last_attack = self._last_attack_at.get(bot.username)
            since_attack = 5.0 if last_attack is None else (time.monotonic() - last_attack)
            raw = self._raw[index] if index < len(self._raw) else {}
            # G-702：自身 vy 直接取协议里 self.velocity.y（运行时由 mineflayer 的
            # bot.entity.velocity 补上）。缺字段按 0，等价于旧模型的「不知道」。
            self_velocity_y = ((raw.get("self") or {}).get("velocity") or {}).get("y")
            # G-702：出手冷却剩多少秒到 0.5s。与 index 21 用**同一个** since_attack，
            # 避免两个时钟（monotonic 与 tick 计数）在掉帧时给出互相矛盾的读数。
            cooldown_remaining = attack_cooldown_remaining(since_attack)
            # 着地推断：推进本观测者这一帧的推断状态（`frame` 保证同一 step 只推进一次，
            # 因为对手策略的观测也在同一 step 内编码）。
            opponent_ground = self._ground[index].update(raw.get("entities") or [],
                                                         frame=self._steps)
            # G-731：自身朝向（弧度）。协议里 self.yaw 由运行时从 mineflayer 的
            # bot.entity.yaw 补上；缺失时编码成 (0,0)（= 不知道），并在 _yaw_available 留痕。
            yaw_value = self_yaw_of(raw)
            self._yaw_available[bot.username] = yaw_value is not None
            if yaw_value is None and not self._yaw_warned:
                self._yaw_warned = True
                self._log("观测缺 self.yaw：target_bearing_sin/cos 将恒为 0（模型学不会转向）"
                          "——检查运行时的 OBSERVATION.self.yaw")
            encoded[self._key(index)] = encode_observation(
                self._raw[index],
                bot.username,
                step_fraction=fraction,
                opponent_health=opponent_health,
                opponents=self._opponent_names(),
                reach=float(bot.reach),
                since_attack=since_attack,
                self_velocity_y=self_velocity_y,
                attack_cooldown_remaining=cooldown_remaining,
                # G-711：跨帧信息（上一帧十头动作 + 距上次出手/受伤的帧数）。
                prev_action=self._prev_action.get(bot.username),
                frames_since_attack=self._frames_since_attack.get(bot.username),
                frames_since_hit=self._frames_since_hit.get(bot.username),
                opponent_ground=opponent_ground,
                # G-731：朝向误差（索引 37/38）。
                self_yaw=yaw_value,
            )
        return encoded

    def _update_cross_frame(
        self,
        previous: list[dict],
        dispatch: list[list[dict]],
        picked_intents: dict[str, tuple[int, ...] | None],
    ) -> None:
        """G-711：推进跨帧状态（``_prev_action`` / ``_frames_since_*``）。

        两条计数器的规则：

        - **出手**：本帧真的下发了 attack（``_has_attack``，与奖励/节流判定同一口径）
          就清零，否则 +1 到饱和；
        - **受伤**：本帧自身血量下降就清零，否则 +1 到饱和。用血量差而不是
          ``ENTITY_DAMAGE`` 事件，是因为观测只由 ``_raw`` 刷新，两者必须同一份数据
          （否则会出现「事件说挨打了、血量没掉」的相位差）。

        饱和值 50 帧 ≈ 20Hz 下的 2.5 秒：再久就没有分辨意义了，也让归一化落在 0..1。
        """
        cap = CROSS_FRAME_CAP
        for index, bot in enumerate(self.bots):
            name = bot.username
            attacked = index < len(dispatch) and self._has_attack(dispatch[index])
            current_attack = self._frames_since_attack.get(name, cap)
            self._frames_since_attack[name] = 0 if attacked else min(cap, current_attack + 1)

            before = self_health(previous[index]) if index < len(previous) else None
            after = self_health(self._raw[index]) if index < len(self._raw) else None
            hurt = before is not None and after is not None and (before - after) > 1e-3
            current_hit = self._frames_since_hit.get(name, cap)
            self._frames_since_hit[name] = 0 if hurt else min(cap, current_hit + 1)

            self._prev_action[name] = picked_intents.get(name)

    def _opponent_names(self) -> set[str]:
        """本环境内的参赛 Bot 用户名：观测/动作只把彼此当目标。

        这样观战者（spectator）或误入服务器的路人不会成为 Bot 的追击/攻击目标，
        也不会污染 ``nearest_player``。
        """
        return {bot.username.lower() for bot in self.bots}

    def _opponent_index(self, index: int) -> int | None:
        if len(self.bots) == 1:
            return None
        if len(self.bots) == 2:
            return 1 - index
        raw = self._raw[index] if index < len(self._raw) else {}
        target = nearest_player(raw, self.bots[index].username, self._opponent_names())
        if target is not None:
            name = str(target.get("name", "")).lower()
            for other, bot in enumerate(self.bots):
                if other != index and bot.username.lower() == name:
                    return other
        return None

    def _apply_state_deaths(self) -> set[str]:
        """把 STATE 帧里的「刚死」补回 ``self._raw``，返回本次补上的用户名集合。

        1.8 的死亡→自动重生只有 2 tick（≈100ms），而观测刷新是 5Hz，采样几乎必然
        跨过去。漏掉的后果不只是「回合不结束」：对手会带着 0 护甲反复重生、补装窗口
        每 tick 发 ``give``，把 4 场地的 steps/s 从 6+ 打到 1.8。
        """
        deaths: set[str] = set()
        for index, link in enumerate(self.links):
            getter = getattr(link, "state_events", None)
            if getter is None:
                continue
            try:
                events = getter()
            except Exception:  # noqa: BLE001 - 观测线程上的异常不应打断训练
                continue
            for event in events:
                if str(event.get("state")) != "Dead":
                    continue
                if str(event.get("previous")) == "Dead":
                    continue
                if index >= len(self.bots) or index >= len(self._raw):
                    continue
                raw = dict(self._raw[index] or {})
                if is_out(raw):
                    continue  # 采样已经看到了，不必改写
                # 观测结构是 {"self": {"health": ...}, "state": ...}；只改 state 不够，
                # 奖励/胜负判据看的是 self.health。
                inner = dict(raw.get("self") or {})
                inner["health"] = 0.0
                raw["self"] = inner
                raw["state"] = "Dead"
                self._raw[index] = raw
                username = self.bots[index].username
                deaths.add(username)
                # 本局已结束；下一局的补装窗口照旧（重生后第一 tick 就要穿甲）。
                self._restock_pending[username] = max(
                    self._restock_pending.get(username, 0), _RESTOCK_TICKS)
                self._respawn_grace[username] = max(
                    self._respawn_grace.get(username, 0), _RESTOCK_TICKS)
        return deaths

    def _winner(self) -> str | None:
        for i, bot in enumerate(self.bots):
            if not is_out(self._raw[i]):
                continue
            # 有 Bot 出局时，赢家是仍存活的另一个（1v1 语义）
            alive = [b.username for j, b in enumerate(self.bots) if j != i and not is_out(self._raw[j])]
            if len(alive) == 1:
                return alive[0]
        return None

    def _safe_send(self, index: int, actions: list[dict]) -> list[dict]:
        link = self.links[index]
        try:
            if not self.wait_action_results:
                nowait = getattr(link, "send_actions_nowait", None)
                if nowait is not None:
                    nowait(actions)
                    return []
            return link.send_actions(actions)
        except LinkError as exc:
            self._log(f"{self.bots[index].username}: 发送动作失败: {exc}")
            return []

    def _drain_results(self) -> None:
        """非阻塞模式下把上一 tick 的 ACTION_RESULT 取回来（日志 + 补装触发）。"""
        for index, link in enumerate(self.links):
            poll = getattr(link, "poll_results", None)
            if poll is None:
                continue
            try:
                results = poll()
            except Exception as exc:  # noqa: BLE001
                self._log(f"{self.bots[index].username}: 读取动作结果失败: {exc}")
                continue
            self._handle_action_results(index, results)

    def _handle_action_results(self, index: int, results: list[dict]) -> None:
        """统一处理 ACTION_RESULT：打日志 + 把「没东西可打」变成补装触发源。

        关键：局内死亡→复活可能整个被 200ms 的观测刷新跳过，
        ``_restock_after_respawn`` 看不到 ``is_out`` 就不会触发。运行时的
        ``ITEM_MISSING``（背包里没有剑/鱼竿/金苹果）是**必然会被发出来**的可见信号，
        用它当触发源，补装就不会再静默失效。
        """
        if not results:
            return
        user = self.bots[index].username if 0 <= index < len(self.bots) else "?"
        policy_opponent = index != 0 and self.opponent_policy is not None
        for result in results:
            if not result or result.get("ok") is not False:
                continue
            code = str(result.get("code") or "")
            self._log(f"{user}: 动作失败 code={code} {result.get('reason', '')}")
            if policy_opponent:
                # 对手自己的失误只计数：它不该影响学习方的 reward，也不该静默消失
                # （自博弈里 OUT_OF_REACH/ITEM_MISSING 的比例是判断快照是否可用的信号）。
                failures = self.opponent_policy_stats["actionFailures"]
                failures[code] = int(failures.get(code, 0)) + 1
            if code in _RESTOCK_TRIGGER_CODES:
                self._request_restock(index, reason=f"code={code}")
