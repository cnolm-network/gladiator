# 训练环境 API（M4）

`training/gladiator_env` 是一个 Gymnasium 环境：它通过 **通道 B**（WebSocket）读取
Bot 的观测、下发动作，并通过 **RCON** 完成训练场准备（spawn、传送、清背包、回血）。

> 分层约束（见 `ARCHITECTURE.md` / `training/AGENTS.md`）：本层不直接操作 mineflayer、
> 不调用 Bukkit API、不做对局规则判定、不写训练算法。

---

## 1. 安装与自检

```bash
cd training
python -m venv .venv           # 仓库里已有 .venv，可跳过
.venv/Scripts/pip install -r requirements.txt

# 离线自检（不需要服务端，使用假通道 B）
.venv/Scripts/python.exe scripts/check_env.py

# 单元测试
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

`check_env.py` 跑的是 Gymnasium 官方 `check_env`，覆盖 `G-401` 验收。

---

## 2. 运行前提

| 组件 | 要求 |
|---|---|
| Layer 1 | Paper/Spigot 1.8.8 + Gladiator 0.1.0（`api 1.1.0`），`bridge.port` 默认 8765 |
| 控制面 | 服务端开启 RCON（`enable-rcon=true`、`rcon.port=25575`、`rcon.password=...`），执行者需 OP |
| Layer 2 | **每个 Bot 一个 bot-runtime 进程**，各自监听一个通道 B 端口（8766、8767……） |
| 世界 | 示例使用超平坦世界（`level-type=FLAT`，地面 y=3，站立 y=4）；其他世界请显式传 `arena.center` |

启动两个 Bot 运行时：

```bash
cd bot-runtime
node src/index.js --username Glad_Bot_1 --host 127.0.0.1 --mc-port 25565 --env-port 8766
node src/index.js --username Glad_Bot_2 --host 127.0.0.1 --mc-port 25565 --env-port 8767
```

运行时连上 8765 后收到 `WELCOME`，并在 8766/8767 上提供通道 B；Bot 本身要等服务端
`spawnBot` 后才真正登录（环境在 `reset()` 里通过 RCON `/gladiator spawn <username>` 触发）。

---

## 3. 快速开始

```python
import gymnasium as gym
import gladiator_env  # 注册 GladiatorBrawl-v0

env = gym.make("GladiatorBrawl-v0", bots=["Glad_Bot_1", "Glad_Bot_2"])
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

直接构造（便于传 RCON、单智能体模式等）：

```python
from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv

env = GladiatorEnv(
    [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
     BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
    arena=ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
    rcon={"host": "127.0.0.1", "port": 25575, "password": "devpass"},
    auto_spawn=True,       # reset 时自动 /gladiator spawn
    single_agent=True,     # 只控制 bot_0，其余 Bot 用随机对手策略
    max_steps=200,
)
```

一行跑完随机混战（`G-406`）：

```bash
.venv/Scripts/python.exe scripts/random_brawl.py --steps 200 --no-spawn
```

---

## 4. API 参考

### 4.1 `BotConfig`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `username` | `str` | 必填 | MC 用户名，需匹配 `[A-Za-z0-9_]{3,16}` |
| `env_url` | `str` | `ws://127.0.0.1:8766` | 该 Bot 的通道 B 地址 |
| `bot_id` | `str \| None` | `glad-<小写用户名>` | 必须与服务端 `BotRegistry.botIdFor` 一致 |
| `cps` / `cps_jitter` | `float` | `15.0` / `2.0` | 每秒点击数及其抖动，经 `spawn_args()` 下发给运行时 |
| `reaction_ms` / `reaction_jitter_ms` | `float` | `220.0` / `40.0` | 反应延时及其抖动（毫秒） |
| `reach` | `float` | `3.99` | 攻击距离（格），超出由运行时返回 `OUT_OF_REACH` |
| `model` | `str` | `"default"` | 训练模型标识 |

> 每个 Bot 的手感是**独立**的：`reset()` 时用 `/gladiator spawn <user> cps=.. reaction=.. reach=.. model=..`
> 逐个别名下发，运行时按 `SPAWN.profile` 覆盖自身默认值。

### 4.2 `ArenaConfig`

| 字段 | 默认 | 说明 |
|---|---|---|
| `center` | `(0.5, 4.0, 0.5)` | 训练场中心（超平坦世界地面 y=4）；其它地形请改 |
| `spacing` | `6.0` | Bot 沿 X 轴等距分布（3 格时对手几乎站在剑距内，一开局就贴脸） |
| `prep_seconds` | `3.0` | 开局准备期：发令前压住双方并复核套装；0 表示不等待 |
| `clear_inventory` | `True` | `reset()` 时 `/clear`（空手训练） |
| `heal` | `True` | `/effect <p> clear` + 瞬间治疗 |
| `settle_timeout` | `10.0` | 等待全员落地（水平距离 ≤2）的秒数 |
| `respawn_timeout` | `15.0` | 等待全员存活（health>0）的秒数 |

### 4.3 `GladiatorEnv`

```python
GladiatorEnv(
    bots, arena=None, *,
    rcon=None, control_hz=None, action_ticks=2, max_steps=600,
    reward_fn=None, single_agent=False,
    links=None, link_factory=None, connect=True,
    auto_spawn=False, spawn_timeout=60.0,
    observation_timeout=2.0, idle_threshold=5.0,
    wait_action_results=False, opponent_mode="random",
    reward_mode="default", shaping_weights=None, logger=None,
)
```

- `rcon`：`RconClient` 实例或 `{host, port, password}` 字典；`None` 表示只读观测、不摆位。
- `control_hz`：默认取通道 B `WELCOME.controlHz`（10）。
- `action_ticks`：每个 RL step 占用几个控制 tick（`sleep(action_ticks / control_hz)`）。
- `single_agent`：`True` 时是标准 Gym 接口（`Box(16,)` + `Discrete(7)`，浮点奖励）；
  `False` 时是 `Dict` 多智能体接口（每个 Bot 一份 obs/action/reward）。
- `links`：注入假链路（测试用，见 `gladiator_env.testing.FakeLink`）。
- `wait_action_results`：默认 `False`——只发动作、不等 `ACTION_RESULT`，靠观测判断效果。
- `opponent_mode`：非学习方 Bot 的行为，`random` / `wander` / `idle` / `scripted`（课程学习用）；
  课程脚本会逐阶段修改该属性。
- `reward_mode`：`default`（简单伤害差）或 `shaped`（`gladiator_env/shaping.py` 的参考权重表）。
- `shaping_weights`：`ShapingWeights` 实例，可覆盖塑形权重。
  打开后每个动作都会等运行时回报，4 Bot 时单步会从约 0.2s 退化到约 0.95s，仅在调试时打开。
- `idle_threshold`：多长时间没有动作算「闲置」（秒），用于 `env.idle_snapshot()`。

#### 空间定义

| 模式 | observation_space | action_space |
|---|---|---|
| `single_agent=True` | `Box(-5, 5, (16,), float32)` | `Discrete(7)` |
| `single_agent=False` | `Dict({"bot_i": Box(16,)})` | `Dict({"bot_i": Discrete(7)})` |

#### 目标选择与观战者

`target_*` 只从**本环境的参赛 Bot** 中挑最近的（`opponents` 过滤，按用户名匹配），
所以进服旁观的人、路过的玩家都不会被当成目标，也不会被 Bot 追打。
观战方法见 [`spectating.md`](spectating.md)。

#### 观测（19 维，`float32`，裁剪到 ±5）

| # | 字段 | 说明 |
|---|---|---|
| 0 | `self_health` | 自身血量 |
| 1 | `self_food` | 饱食度 |
| 2 | `self_saturation` | 饱和度 |
| 3 | `self_on_ground` | 是否着地 |
| 4–5 | `self_yaw_sin/cos` | 朝向（弧度编码，避免 ±π 跳变） |
| 6 | `self_pitch` | 俯仰角（度） |
| 7–9 | `target_dx/dy/dz` | 最近对手相对位置（格） |
| 10 | `target_distance` | 到最近对手的距离 |
| 11 | `target_health` | 对手血量（来自对手自己的观测，`entities` 不含 health） |
| 12 | `target_on_ground` | 对手是否着地 |
| 13 | `has_target` | 是否存在对手 |
| 14 | `step_fraction` | `step / max_steps` |
| 15 | `state_alive` | 是否存活 |
| 16 | `self_held_durability` | 手持物品剩余耐久比例（无耐久=1，`G-608`） |
| 17 | `self_armor_remaining` | 四件护甲平均剩余耐久比例（无甲=0，`G-608`） |
| 18 | `self_gapples` | 背包金苹果数 / 5（`G-608`，训练“什么时候吃”） |

缺失字段有默认值；`NaN`/`inf` 会被替换成 0，`G-402` 验收即此。

#### 动作（11 个离散动作，`G-403` / `G-607`）

| index | 名称 | 协议动作 |
|---|---|---|
| 0 | `idle` | 停机（不发送任何动作） |
| 1 | `approach` | `look` → `move(forward, 180ms)` |
| 2 | `attack` | 距离 ≤3 且有实体 id：`look` → `attack`；否则退化为 `move(forward)` |
| 3 | `strafe_left` | `look` → `move(left)` |
| 4 | `strafe_right` | `look` → `move(right)` |
| 5 | `retreat` | `look` → `move(back)` |
| 6 | `jump` | `jump` |

每个动作都带单调 `seq`；重发同一 `seq` 由服务端幂等缓存去重（`ACTION_RESULT.deduped=true`）。

#### 奖励（可插拔，`G-404`）

`reward_fn(ctx: RewardContext) -> float`，默认 `default_reward`：

- 造成伤害 `+0.1 × HP`，受到伤害 `-0.1 × HP`（伤害由对方观测血量差推算）；
- 击杀 `+1.0`、阵亡 `-1.0`、每步 `-0.001`。

默认奖励**不含**任何对局规则判定（是否允许复活、胜负裁定由调用方 / Mode 决定）。

#### `reset(seed=None, options=None)`

完成：连通道 B → 收 `WELCOME` → 读 `controlHz` → （可选）`/gladiator spawn` →
等服务端把 Bot 放进来并回满血 → `/gamemode 0`、`/clear`、`/effect clear`、
瞬间治疗、`/spawnpoint`、`/tp` 到训练点 → 等落地 → 取首帧观测。

返回 `(observation, {"step": 0})`。

#### `step(action)`

1. 把离散动作解码成协议动作序列；
2. 通过通道 B 下发（失败只记日志，不抛异常）；
3. `sleep(action_ticks / control_hz)`；
4. 取新观测，算 `terminated = 任一 Bot 出局`，`truncated = step >= max_steps`；
5. 返回 `(obs, reward, terminated, truncated, info)`。

`info` 字段：`step`、`health`、`states`、`winner`。

空闲状态通过 `env.idle_snapshot()` 读取（为保持 Gymnasium 的确定性检查，不放进 `info`）：

```python
{"seconds": {"Glad_Bot_1": 0.42}, "idle": {"Glad_Bot_1": False}, "events": {...}}
```

`seconds` 是距上一次真正下发动作的秒数；只有超过 `idle_threshold`（默认 5s）才在
`idle` 里标记为 `True`，且**不会**因此发起任何外部查询（`G-405`）。`events` 是通道 B
下发的 `IDLE` 通知（最多保留 32 条），上一次 `idle_snapshot()` 之后的新事件。

#### `close()`

停止所有通道 B 连接，关闭环境自建的 RCON 连接。

---


### 4.6 默认套件与物品动作（G-607）

### 4.6.1 开局准备期（G-615）

`reset()` 在摆位之后、`step()` 之前会走一段准备期（`ArenaConfig.prep_seconds`，默认 3 秒）：

1. 每 0.5 秒给双方发一次 `idle`，清掉上一回合残留的移动/疾跑按键，避免“还没穿甲就冲”；
2. 每 1 秒复核一次套装（护甲 4/4、手持剑、金苹果 5、鱼竿 1），缺什么补什么；
3. 准备期结束再对齐一次站位，然后才返回第一帧观测。

日志形如 `[kit] <user> 套装就绪：手持剑，金苹果 5，护甲 4/4` → `[prep] 准备期 3s 结束，开始对局`。
准备期不计入 `max_steps`，也不产生 reward。

`ArenaConfig.give_kit=True` 时，每次 reset 用 RCON 发一套装备（顺序固定为
剑 → 金苹果 → 鱼竿 → 四件护甲，保证剑落在快捷栏第一格）：

```text
diamond_helmet / chestplate / leggings / boots
diamond_sword · golden_apple × 5 · fishing_rod
```

**套件校验（G-611，消除训练波动）**：RCON `give` 没有回执，Bot 在摆位阶段死亡还会
掉物品，所以 `_prepare_arena` 发完套件后会执行 `_ensure_loadout`：

1. 发 `use {item:"sword", select:true}`（只选中不挥砍）把主手固定到剑；
2. 2 秒内轮询观测，用 `loadout_report()` 检查剑 / 5 苹果 / 鱼竿 / 4 件护甲；
3. 缺件按缺量补 `give`、护甲不足补 `use {item:"armor"}`，最多 3 次；
4. 成功打印 `[kit] <user> 套装就绪：手持剑，金苹果 5，护甲 4/4`，失败打印
   `校验未通过（该局数据可能有偏差）`。

这条校验解决两个曾导致 reward 剧烈波动的根因：装备被吞（缺剑/缺甲）和
**双方开局手持物不确定**（对手握剑 vs 握苹果强度差异巨大）。

运行时会在背包变化后自动穿上护甲（`items.js: equipArmor`），所以观测里的 `self.inventory`
能看到护甲之外的战斗物品；`entities[].heldItem` 额外暴露对手主手物品，训练方可以
判断“对面是不是握着剑”。动作空间从 7 扩到 **11**：

| 编号 | 名称 | 行为 |
|---|---|---|
| 0–6 | idle/approach/attack/strafe_left/strafe_right/retreat/jump | 同前 |
| 7 | `use_rod` | 2–6 格内切鱼竿右键击退，再切回剑 |
| 8 | `eat_gapple` | 血量 <19.5 时吃金苹果（按住 1.6s） |
| 9 | `block` | 举剑格挡 300ms |
| 10 | `switch_sword` | 从其它物品切回剑 |

### 4.7 战斗辅助（“外挂级”开关）

`BotConfig(wtap=/auto_rod=/auto_block=/aim_lock=)` 与 Java `BotProfile` 同义，
经 SPAWN 的 `assist` 字段下发；命令行对应 `train.py --wtap --auto-rod --auto-block --aim-lock`，
对手可用 `--opp-assist` 全开施压。

| 开关 | 效果 |
|---|---|
| `aim_lock` | 攻击前 3D 对准目标眼睛（避免背对空挥） |
| `auto_rod` | 2.5–4.5 格自动放鱼竿击退，1.5s 冷却 |
| `auto_block` | 空闲动作自动举剑格挡 |
| `wtap` | 命中后松开前进/疾跑 1 tick 再按回（sprint reset） |

### 4.8 角色标签、策略与 web 训练（G-608 / G-619）

标签的**颜色与文案可配置**：`caller-sample` 的 `config.yml` 里 `tags:` 段（`&` 颜色码），
默认给出 trained / rusher / aimbot / scripted / wander / random / idle 七种。
训练环境按「被训练方 = `BotConfig.role`、代码驱动方 = 对手策略名」自动调用
`/caller tag <player> <role>`；想给新脚本或新模型单独配色，在 `tags:` 里加一行再执行
`/caller reloadtags` 即可，不需要重新编译。


训练时每个 Bot 的 nametag 会被自动打标（`caller tag`，需要服务端装 BetterNameTag + LuckPerms）：

| 角色 | 前缀 | 说明 |
|---|---|---|
| 被训练方 | `§c§o[训练]` | `users[0]`，策略模型输出，正在学习 |
| rusher | `§6§o[Rush]` | 只冲脸、贴身猛砍的脚本 |
| aimbot | `§b§o[Aim]` | 贴到剑距内持续攻击的脚本 |
| scripted | `§a§o[Script]` | 远近结合的通用脚本 |
| wander / random / idle | 黄 / 粉 / 灰 | 弱对手，用于课程前期 |

`GladiatorEnv._prepare_arena` 在摆好位后自动调用 `caller tag <username> <role>`，失败只记日志，不影响训练。

对手更激进：`rusher` 在 3.4 格内 96% 攻击、`aimbot` 在 3.9 格内持续砍；`scripted` 2.8 格内 85% 攻击。奖励新增“压迫位”`pressure: 0.05/秒`（2.8–4.2 格内、未终局），与原有的攻击窗口间距奖励叠加，用来惩罚一味后退。

**Web 优先**：训练统一从 dashboard（`http://127.0.0.1:8080` → 训练页）启动，`autoSpawn` 打开时 dashboard 会自动拉起缺失的运行时；「运行时」页会用徽章标出每个 Bot 是**被训练方（红）**还是**代码驱动 · 策略名（蓝）**，鼠标悬停显示策略说明。一键「停止所有 Bot」「停止服务器」在同一页。

### 4.9 产物目录与 run 隔离（G-609）

每次训练写入独立目录，避免 `episode-001` 在多次运行之间互相覆盖：

```
training/artifacts/runs/20261107-2320-multi-Bot_C/
    report.json        # runDir / kind / startedAt / history（每局带 startedAt 与 durationSeconds）
    policy.npz         # train；multi 为 policy-pair1.npz …
    reviews/           # episode-001.md / .json；multi 为 pair-1/…
```

目录名 = `<YYYYMMDD-HHMM>-<kind>-<tag>`，同一分钟重复启动会加 `-2`。`train.py` / `multi_train.py`
都支持 `--run-dir` 指定目录；不指定则自动创建。dashboard 的「数据可视化」按运行时间倒序列出这些目录。

奖励权重（`ShapingWeights`）在参考文档基础上做了三处针对“畏首畏尾”的调整：
空挥 `-0.3 → -0.1`、剑距内出手 `+0.05`、远距离格挡/挥竿 `-0.08`，并新增主动贴近 `+0.02/step`
与挨打后 0.6s 内反手 `+0.15`；超出 `reach` 的 attack 不再计空挥（运行时本来就会拒绝）。

### 4.10 奖励预设（G-619）

奖励参数不再写死在代码里：一组权重 + 若干条件式附加项可以打包成 JSON 预设，训练时用
`--reward-preset <path>` 指定；`train.py` 与 `multi_train.py` 都支持。Web 端（Dashboard）
提供同一份 schema 的增删改与选择。

```json
{
  "format": "gladiator.reward-preset/v1",
  "name": "均衡（默认）",
  "description": "默认权重",
  "base": {
    "kill": 20.0, "death": -20.0,
    "damageDealt": 2.5, "damageTaken": -1.2,
    "combo": 0.6, "comboCap": 3.0,
    "threatWindow": 0.3, "pressure": 0.05,
    "whiff": -0.1, "still": -0.1,
    "waste": -0.08, "closing": 0.02,
    "attackAttempt": 0.05, "counter": 0.15
  },
  "terms": [
    {"id": "threat", "weight": 0.3, "unit": "second",
     "when": {"distanceMin": 2.8, "distanceMax": 4.2}},
    {"id": "punishBlock", "weight": -0.2, "unit": "step",
     "when": {"actionIn": ["block"], "distanceMin": 4.2}}
  ]
}
```

**`base` 键**：`kill` / `death` / `damageDealt` / `damageTaken` / `combo` / `comboCap` /
`threatWindow` / `pressure` / `whiff` / `still` / `waste` / `closing` / `attackAttempt` /
`counter` / `inactive`（兼容旧名，默认 0）。

- 伤害类按 HP 计：`damageDealt * 造成伤害`，`damageTaken * 承受伤害`。
- `still`：**连续 3 秒没有水平位移**才按秒扣分；走动本身不扣分（G-619 起）。
  旧的无交互惩罚 `inactive` 默认关闭，只有显式配置非 0 才生效。
- `combo` 按「一串连击封顶」结算，`comboCap` 是单串上限。

**`terms` 多条件**：一个 term = 一个条件对象 + 一个权重 + 一个计量单位。
条件之间是 **AND**；需要「或」就写两条 term（各自独立结算）。支持的条件：

| 条件 | 含义 |
|---|---|
| `distanceMin` / `distanceMax` | 与对手的水平距离区间（格） |
| `actionIn` | 本步动作名属于该数组（`attack` / `block` / `use_rod` …） |
| `stillSecondsMin` | 已连续静止秒数下限 |
| `healthBelow` / `opponentHealthBelow` | 自身 / 对手血量低于该值 |
| `afterHitSeconds` | 距上次命中不超过该秒数 |
| `comboMin` | 当前连击数下限 |
| `attacked` | 本步是否出手 |

`unit` 为 `second` 时按 `weight * dt` 累加（例如每秒 +0.3），为 `step` 时每步结算一次。

示例（临时文件）：

```bash
python -m gladiator_train.train --users Bot_A Bot_B --ports 8766 8767 \
  --reward-preset D:/cbot-gladiator/dashboard/var/reward-presets/default.json
```

预设校验规则（后端与 Web 一致）：未知键、未知条件、非法 `unit`、重复 `id`、非有限数字
都会被拒绝，错误信息为中文。Dashboard 的预设目录是 `dashboard/var/reward-presets/`，
删除是软删除（移到 `trash/`），同名覆盖前会备份。

## 5. 与控制面（RCON）的关系

`reset()` 用到的命令都属于**调用方职责**，不在 Gladiator 内部执行：

```text
/gladiator spawn <username>      # 触发 spawnBot
/gladiator train <username>      # 可选：pushPlace 训练会话
/gamemode 0 <player>
/clear <player>
/effect <player> clear
/effect <player> 6 1 120 4 true  # 瞬间治疗
/spawnpoint <player> <x> <y> <z>
/tp <player> <x> <y> <z>
```

这也是 `docs/api-contract.md` §2 的“传送 / 套件 / 地图 / 结算由调用方负责”的落地方式。

---

## 6. 已知限制

- `entities` 不含血量，对手血量只能从对手自身的观测里读；因此环境要求所有参战 Bot
  都由本环境管理（各自一条通道 B 链路）。
- 通道 B 的 `self.state` 目前由 Layer 1 状态机推送；`state` 只用于 `state_alive`
  与 `terminated` 判定，不参与奖励规则。
- 朝向公式 `YAW_FORMULA="neg"` 是按 1.8 的 yaw 约定推的，若发现 Bot 转向相反，
  在 `gladiator_env/spaces.py` 改成 `"pos"` 并重跑 `tests/test_spaces.py`。
- 世界必须是可站立地形；`ArenaConfig.center` 的 y 要落在地面上，否则 Bot 会摔死
  （日志出现 `fell from a high place`）。
