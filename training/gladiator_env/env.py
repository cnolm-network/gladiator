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
    MOVE_MS,
    NUM_ACTIONS,
    OBS_DIM,
    decode_action,
    encode_observation,
    nearest_player,
)

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
#: 主动体检周期（步）：每 N 步查一次套装，缺件就触发补装。
_RESTOCK_CHECK_EVERY = 10
#: 这些运行时失败码说明「手上有动作但没东西」，直接当作补装触发源。
_RESTOCK_TRIGGER_CODES = frozenset({"ITEM_MISSING", "ITEM_NOT_FOUND", "NO_ITEM"})


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
    攻击距离、训练模型标签。默认值即“原版手感”：15±2 CPS / 220±40ms / 3.99 格。
    """

    username: str
    env_url: str = "ws://127.0.0.1:8766"
    bot_id: str | None = None
    cps: float = 15.0
    cps_jitter: float = 2.0
    reaction_ms: float = 220.0
    reaction_jitter_ms: float = 40.0
    reach: float = 3.99
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
        reward_mode: str = "shaped",
        shaping_weights: ShapingWeights | None = None,
        reward_preset: dict | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.bots = [b if isinstance(b, BotConfig) else BotConfig(str(b)) for b in bots]
        if len(self.bots) < 1:
            raise ValueError("至少需要一个 Bot 才能建立训练环境")
        self.arena = arena or ArenaConfig()
        # 两个 Bot 站在同一点会互相卡住：死亡后在原地复活、对手又贴着打，补给永远发不下去。
        # 这里兜底一个最小间距，避免表单里留空/填 0 直接把整个训练变成“出生点互砍”。
        if self.arena.spacing < MIN_SPACING:
            self.arena = replace(self.arena, spacing=MIN_SPACING)
        self.action_ticks = max(1, int(action_ticks))
        self.max_steps = max(1, int(max_steps))
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
        self._last_attacked: dict[str, bool] = {}
        self._last_distance: dict[str, float | None] = {}
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
        self._log = logger or (lambda message: print(f"[env] {message}", flush=True))

        self._control_hz = float(control_hz or 0.0)
        self._steps = 0
        self._raw: list[dict] = []

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

        if self.single_agent:
            # 单智能体视角（Gymnasium 标准）：控制 bot_0，其余 Bot 作为对手由环境随机采样动作。
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = spaces.Discrete(NUM_ACTIONS)
        else:
            self.observation_space = spaces.Dict({
                self._key(i): spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
                for i in range(len(self.bots))
            })
            self.action_space = spaces.Dict({
                self._key(i): spaces.Discrete(NUM_ACTIONS) for i in range(len(self.bots))
            })

        self._connected = False
        if connect:
            self._ensure_connected()

    # --- Gymnasium API --------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.shaping.reset()
        options = options or {}
        self._steps = 0
        # 新回合：动作确认状态与补装确认都不该跨回合残留。
        self._opponent_hold.clear()
        self._opponent_target.clear()
        self._restock_pending.clear()
        self._restock_logged.clear()
        self._restock_tick.clear()
        self._restock_rounds.clear()
        self._ensure_connected()

        if self._rcon_owned and self._rcon is not None:
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
        if self.single_agent:
            return observation[self._key(0)], {"step": 0}
        return observation, {"step": 0}

    def step(self, action):
        self._steps += 1
        if self.single_agent:
            action_map = {self._key(0): int(action)}
            for index in range(1, len(self.bots)):
                action_map[self._key(index)] = int(self._opponent_action(index))
        else:
            action_map = action

        dispatch: list[list[dict]] = []
        if len(self._last_active_at) != len(self.bots):
            self._last_active_at = [time.monotonic() for _ in self.bots]
        opponents = self._opponent_names()
        for i, bot in enumerate(self.bots):
            index = int(action_map.get(self._key(i), 0))
            if index != 0:  # 0 = idle，只有真正下发动作才算“活跃”
                self._last_active_at[i] = time.monotonic()
            raw = self._raw[i] if i < len(self._raw) else {}
            # 代码驱动的对手比学习方多留 0.75 格余量：它没有「奖励」纠偏，而观测到出手之间
            # 有 1 个 tick 的延迟（约 5 格/s × 0.2s ≈ 1.1 格），加上击退会让边界攻击吃 OUT_OF_REACH。
            reach = bot.reach if i == 0 else max(1.5, float(bot.reach) - 0.75)
            actions = decode_action(index, raw, bot.username, opponents, reach=reach)
            if i != 0 and self._respawn_grace.get(bot.username, 0) > 0:
                # 刚重生（可能被传送到出生点）：本 tick 只做观察，避免对旧坐标出手。
                self._respawn_grace[bot.username] -= 1
                self._clear_opponent_hold(bot.username)
                actions = [{"type": "idle", "params": {"durationMs": MOVE_MS}}]
            elif i != 0:
                # 对手「出手前先确认」：观测与出手之间隔着一个 tick（≈1.1 格位移）+ 击退，
                # 直接出拳会在边界吃 OUT_OF_REACH。这里把第一次判到的 attack 换成贴近，
                # 下一 tick 用新观测复核距离，仍在射程内才真的挥。
                actions = self._confirm_opponent_attack(bot, actions)
            if self._restock_pending.get(bot.username, 0) > 0:
                # 重生补装确认中：背包可能还没同步，进攻动作直接不发
                # （否则运行时回 ITEM_MISSING，日志刷屏且这一 tick 白费）。
                actions = self._strip_combat_actions(actions)
            dispatch.append(actions)
            self._handle_action_results(i, self._safe_send(i, actions))

        time.sleep(self.action_ticks / self._control_hz)
        if not self.wait_action_results:
            self._drain_results()
        previous = self._raw
        self._raw = self._refresh_raw(previous)
        # A2：重生补装确认（必须在观测刷新之后、算奖励之前，让策略看到补齐后的状态）。
        self._confirm_restock()
        # 兜底体检：局内「死→活」可能整个被刷新跳过（is_out 看不到），这里只看现状。
        if _RESTOCK_CHECK_EVERY > 0 and self._steps % _RESTOCK_CHECK_EVERY == 0:
            self._periodic_loadout_check()
        observation = self._encode_all()

        terminated = any(is_out(obs) for obs in self._raw)
        if not terminated:
            self._restock_after_respawn(previous)
        truncated = self._steps >= self.max_steps
        rewards = {}
        self._last_events = {}
        self._last_attacked = {}
        self._last_distance = {}
        now = time.monotonic()
        for i, bot in enumerate(self.bots):
            opponent = self._opponent_index(i)
            distance = self._distance(i, opponent)
            previous_distance = self._last_distance.get(bot.username)
            self._last_distance[bot.username] = distance
            wants_attack = any(action.get("type") == "attack" for action in dispatch[i])
            # 超出 reach 的 attack 根本挥不出去（运行时抛 OUT_OF_REACH），不能算空挥。
            in_reach = distance is None or distance <= float(bot.reach) + 0.2
            self._last_attacked[bot.username] = wants_attack and in_reach
            action_name = ACTION_NAMES[index] if 0 <= index < len(ACTION_NAMES) else "idle"
            if self.reward_mode == "shaped":
                value, events = self.shaping.compute(
                    bot.username,
                    previous=previous[i] if i < len(previous) else {},
                    current=self._raw[i],
                    opponent_previous=previous[opponent] if opponent is not None and opponent < len(previous) else None,
                    opponent_current=self._raw[opponent] if opponent is not None else None,
                    terminated=terminated,
                    attacked=self._last_attacked[bot.username],
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
            "health": {bot.username: self_health(self._raw[i]) for i, bot in enumerate(self.bots)},
            "states": {bot.username: str(self._raw[i].get("state", "Alive")) for i, bot in enumerate(self.bots)},
            "winner": self._winner(),
        }
        if self.single_agent:
            return observation[self._key(0)], rewards[self._key(0)], terminated, truncated, info
        return observation, rewards, terminated, truncated, info

    @property
    def raw_observations(self) -> list[dict]:
        """最近一次 step 的原始观测（复盘用）。"""
        return list(self._raw)

    def last_events(self) -> dict[str, list[dict]]:
        """最近一次 step 的塑形事件（hit/whiff/combo/kill/death）。"""
        return {name: list(events) for name, events in self._last_events.items()}

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

    def _opponent_action(self, index: int) -> int:
        """非受控 Bot 的动作：课程学习的不同难度。"""
        mode = self.opponent_mode
        if mode == "idle":
            return 0
        if mode == "wander":
            return int(self.np_random.choice([1, 3, 4, 6]))  # approach/strafe/jump，不打人
        if mode == "rusher":
            # 猛冲型：几乎不做拉扯，贴脸就砍（给学习方制造持续压力）。
            observation = self._raw[index] if index < len(self._raw) else {}
            target = nearest_player(observation, self.bots[index].username, self._opponent_names())
            if target is None:
                return 1
            distance = float(target.get("distance", 99))
            if distance > 3.4:
                return 1
            return 2 if self.np_random.random() < 0.96 else 6
        if mode == "aimbot":
            # 100% 瞄准型：每一步都重新转向并攻击，几乎不空挥。
            observation = self._raw[index] if index < len(self._raw) else {}
            target = nearest_player(observation, self.bots[index].username, self._opponent_names())
            if target is None:
                return 1
            distance = float(target.get("distance", 99))
            # 贴到攻击距离内就一直砍，不再在 3.5 格外停下。
            return 1 if distance > 3.9 else 2
        if mode == "scripted":
            # 简化脚本：远则靠近，近则攻击，偶尔跳。
            observation = self._raw[index] if index < len(self._raw) else {}
            target = nearest_player(observation, self.bots[index].username, self._opponent_names())
            if target is None:
                return int(self.np_random.choice([1, 6]))
            distance = float(target.get("distance", 99))
            if distance > 2.8:
                return 1
            return 2 if self.np_random.random() < 0.85 else 6
        return int(self.np_random.integers(NUM_ACTIONS))

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

        ``give`` 依赖 RCON 与 ``give_kit``；两者都不具备时至少把剑切到手上
        （可能是「有剑但没选中」，运行时能自己挪进快捷栏）。
        """
        if self.arena.give_kit and self._rcon is not None:
            for name, need in report["missing"]:
                try:
                    self._command(f"give {bot.username} {name} {max(1, int(need))}")
                except LinkError:
                    pass
            self._give_armor(index, bot, report)
        self._select_sword(index)

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
                self._respawn_grace.pop(bot.username, None)
                suffix = f"，第 {rounds + 1} 轮" if rounds else ""
                self._log(f"[restock] {bot.username} 已补齐（第 {tick} tick{suffix}）")
                continue
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
        cx, cy, cz = self.arena.center
        offset = (index - (len(self.bots) - 1) / 2.0) * self.arena.spacing
        return cx + offset, cy, cz

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
        fraction = self._steps / float(self.max_steps) if self.max_steps else 0.0
        for index, bot in enumerate(self.bots):
            opponent = self._opponent_index(index)
            opponent_health = self_health(self._raw[opponent]) if opponent is not None and opponent < len(self._raw) else None
            encoded[self._key(index)] = encode_observation(
                self._raw[index],
                bot.username,
                step_fraction=fraction,
                opponent_health=opponent_health,
                opponents=self._opponent_names(),
            )
        return encoded

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
        for result in results:
            if not result or result.get("ok") is not False:
                continue
            code = str(result.get("code") or "")
            self._log(f"{user}: 动作失败 code={code} {result.get('reason', '')}")
            if code in _RESTOCK_TRIGGER_CODES:
                self._request_restock(index, reason=f"code={code}")
