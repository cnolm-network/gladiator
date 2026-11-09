# 观测与动作 schema（通道 B，json-v1）

> 适用范围：Layer 2（`bot-runtime`）↔ Layer 3（`training/gladiator_env`）之间的 `OBSERVATION` / `ACTION` 字段级契约。
> 权威来源：`bot-runtime/src/observation.js`、`bot-runtime/src/link-server.js`、`bot-runtime/src/actions.js`、`training/gladiator_env/spaces.py`。
> 本文档与代码冲突时**以代码为准**，并在「§7 已知偏差」里登记差异，待协议文档对齐。
> 传输、信封、错误码、背压策略仍见 [`docs/protocol.md`](protocol.md) §1 / §3 / §5，本文不复制。

---

## 1. 定位与文档分工

| 文档 | 回答什么问题 | 不回答什么 |
|---|---|---|
| `docs/protocol.md` | 有哪些帧、信封长什么样、握手与错误码、限流策略 | 单个字段的单位、缺省值、逐维含义 |
| **本文档 `docs/observation.md`** | 每个帧的 payload 字段表、19 维向量逐维定义、11 个离散动作如何翻译成协议帧 | Python 侧 `reset()/step()` 怎么调用 |
| `docs/training-env-api.md` | Gymnasium 环境的构造参数、`BotConfig`/`ArenaConfig`、RCON 准备流程、奖励预设 | 观测字段语义、动作编号与协议帧的映射 |

一句话：**本文档讲「线上传什么」，`training-env-api.md` 讲「Python 里怎么用」。**

---

## 2. 通道 B 帧序

### 2.1 完整生命周期

| # | 方向 | 帧 | 触发条件 | 关键 payload | `replyTo` |
|---|---|---|---|---|---|
| 1 | L3 → L2 | `HELLO` | 客户端连上 `ws://127.0.0.1:8766` 后立即发送 | `{botId, v:1, protocol:"gladiator-env"}` | — |
| 2 | L2 → L3 | `WELCOME` | 收到合法 `HELLO` 后 | `{observationSpec, actionSpec, controlHz}` | `HELLO.seq` |
| 3 | L2 → L3 | `OBSERVATION` | `WELCOME` 之后按 `controlHz` 周期推送 | 见 §3 | `null` |
| 4 | L3 → L2 | `ACTION` | 每个控制周期发 1 组（复合动作 = 多帧） | `{type, params}` | — |
| 5 | L2 → L3 | `ACTION_RESULT` | 每个 `ACTION` 必回一帧 | `{ok, code?, reason?, deduped?}` | `ACTION.seq` |
| 6 | L3 → L2 | `RESET` | 回合开始 / 需要重置 | `{resetId, gameId, seed}` | — |
| 7 | L2 → L3 | `RESET_DONE` | 收到 `RESET` 后 | `{state, tick, timeoutMs}` | `RESET.seq` |
| 8 | L2 → L3 | `IDLE` | 空闲窗口开启（**当前运行时未发送**，见 §7.4） | `{sinceSeconds, windowSeconds}` | `null` |
| 9 | L2 → L3 | `REJECT` | 握手/帧非法 | `{code, message}` | 触发帧 `seq` 或 `null` |
| 10 | 双向 | `PING` / `PONG` | 心跳；训练客户端当前不发 | `{}` | 对端 `seq` |

### 2.2 握手与拒绝

| 场景 | 判定位置 | 回帧 | `code` |
|---|---|---|---|
| `payload.v` 存在但不等于 1 | `link-server.js` `onFrame` | `REJECT` | `VERSION_UNSUPPORTED` |
| `payload.protocol` 存在且不是 `"gladiator-env"` | 同上 | `REJECT` | `VERSION_UNSUPPORTED` |
| `payload.botId` 为空 | 同上 | `REJECT` | `INTERNAL` |
| `HELLO` 之前发其它帧 | 同上 | `REJECT` | `SESSION_UNKNOWN` |
| 未知 `type` | 同上 | `REJECT` | `ACTION_UNSUPPORTED` |
| 帧不是合法 JSON | `onConnection` | `REJECT` | `INTERNAL` |

`botId` 约定：`glad-<小写用户名>`（`BotConfig.__post_init__` 与 `BotRegistry.botIdFor` 同一公式）。通道 B 用它把连接绑到具体会话，不一致会导致 `BOT_NOT_READY`。

### 2.3 WELCOME 的 `observationSpec` / `actionSpec`

| 字段 | 类型 | 取值 | 语义 |
|---|---|---|---|
| `observationSpec.encoding` | string | `"json-v1"` | 观测编码版本标识；消费方按此选择解码器 |
| `observationSpec.fields` | string[9] | 见 §3.1 | 观测 payload 的**顶层键名清单**，顺序即代码中的声明顺序 |
| `actionSpec.encoding` | string | `"json-v1"` | 动作编码版本标识 |
| `actionSpec.types` | string[9] | `move, look, jump, attack, dig, place, use, talk, idle` | 运行时**能执行**的协议动作类型；不在表内的 `type` 会被 `ACTION_UNSUPPORTED` 拒绝 |
| `controlHz` | number | 1..20，默认 `10` | 观测推送频率（Hz）；`interval = max(50, 1000/controlHz)` ms |

**重要**：`actionSpec.types` 是 9 个**协议动作类型**（§5 的底层 verb），不是训练用的 11 个离散动作。11 个离散动作是 Layer 3 的抽象，由 `decode_action()` 翻译成这些 verb 的有序列表。

### 2.4 `seq` / `replyTo` 规则

| 规则 | 依据 |
|---|---|
| L2 → L3 每连接独立自增，从 1 开始；`WELCOME` 为第 1 帧，`OBSERVATION` 也占号 | `link-server.js` `send()` 中 `client.seq += 1` |
| L3 → L2 `ACTION.seq` 从 1 单调递增，跨重连不重置 | `link.py` `_next_seq()` |
| 重复 `seq` 不重复执行，回放缓存结果并附 `deduped:true` | `scheduler.js` `run()` |
| 同一 `seq` 并发到达时等待首个执行完再回放 | `scheduler.js` `pending` 分支 |
| 超过 `maxActionsPerSecond`（默认 10）直接丢弃并回 `ACTION_RATE_LIMITED` | 同上 |

---

## 3. OBSERVATION 全字段表

### 3.1 payload 顶层（`OBSERVATION_FIELDS`，9 个）

| 字段 | 类型 | 单位 / 取值 | 来源与说明 |
|---|---|---|---|
| `tick` | int | 单调递增计数 | 运行时进程级累加（`this.tick += 1`），**非**服务端 game tick；所有连接共享同一计数器 |
| `gameId` | string | 对局标识，未知为 `""` | 会话 `session.gameId()` |
| `state` | enum string | `Alive` / `Dead` / `Respawning` / `Eliminated` | 会话 `session.state()`，由插件侧状态机计算，运行时不推导 |
| `self` | object | 见 §3.2 | 自身状态 |
| `entities` | array | 见 §3.3，按 `distance` 升序 | 已排除自身实体 |
| `blockWindow` | object | 见 §3.4，固定 7×4×7 | 调色板 + 索引压平 |
| `restrictions` | array | 见 §3.5，可为 `[]` | 规则缓存快照（仅 deny 条目） |
| `chat` | array | 见 §3.6，最多 20 条 | 最近聊天，`slice(-20)` |
| `pendingNotices` | array | 见 §3.7，最多 10 条 | 最近通知，`slice(-10)` |

### 3.2 `self`

| 字段 | 类型 | 单位 / 取值 | 说明 |
|---|---|---|---|
| `pos.x` / `pos.y` / `pos.z` | number | 方块坐标（格），双精度 | 脚部位置；`y` 是脚底高度 |
| `yaw` | number | **弧度** | mineflayer 约定：`0` 面向 `+z`，正值逆时针（俯视）；缺失回落 `0` |
| `pitch` | number | **弧度** | 正值向下；缺失回落 `0` |
| `onGround` | bool | `true` / `false` | 是否着地（`Boolean(self.onGround)`） |
| `health` | number | 0..20，0.5 粒度 | `bot.health` 原样透传，运行时不做夹取；消费方须容忍缺失（训练侧缺省按 20 处理） |
| `food` | number | 0..20 | `bot.food`（饥饿值） |
| `saturation` | number | 0..5 | `bot.foodSaturation`（饱和度） |
| `heldItem` | object \| null | 见下 | 主手物品，空手为 `null` |
| `heldItem.name` | string | 物品 id，如 `iron_sword` | |
| `heldItem.count` | int | ≥1 | |
| `heldItem.slot` | int | 背包槽位（快捷栏 36..44） | |
| `heldItem.durability.used` | number | ≥0 | 已消耗耐久；无法取得时为 `0` |
| `heldItem.durability.max` | number \| null | >0 或 `null` | 无耐久概念（方块、食物）为 `null` |
| `heldItem.durability.remaining` | number | 0..1 | `(max-used)/max` 并夹取；`max===null` 时**固定 1** |
| `armor` | object | `{head, torso, legs, feet}` | 每项为 `{name, durability{...}}` 或 `null`；对应背包槽 5/6/7/8 |
| `armorRemaining` | number | 0..1 | 已穿护甲 `durability.remaining` 的**算术平均**；一件未穿为 `0` |
| `inventory` | array | `[{slot, name, count}]` | 所有非空槽位（含护甲槽 5..8），不区分热栏 |

### 3.3 `entities[]`

| 字段 | 类型 | 单位 / 取值 | 说明 |
|---|---|---|---|
| `id` | int | mineflayer 实体 id | 与 `ACTION.attack.params.targetEntityId` 同源，可直接回填 |
| `kind` | string | `player` / `mob` / 其它 / `unknown` | 玩家判定优先（`entity.type === 'player'`） |
| `name` | string | 玩家名或实体名 | 取 `username` → `displayName` → `name` → `String(id)` |
| `pos.x/y/z` | number | 方块坐标（格） | 实体脚部位置 |
| `distance` | number | 格 | 自身 `position.distanceTo(entity.position)`，**含 y 的 3D 距离** |
| `heldItem` | object \| null | `{name, count}` | 对手手持物；`air` 视为 `null`。**不含耐久** |

注意：`entities` **不携带血量**。19 维向量里的 `target_health` 由训练环境从「对手自己的 `OBSERVATION.self.health`」取（§4 第 11 维）。

### 3.4 `blockWindow`

| 字段 | 类型 | 取值 | 说明 |
|---|---|---|---|
| `origin.x` | int | `floor(self.x) - 3` | 窗口最小角 |
| `origin.y` | int | `floor(self.y) - 1` | 脚下 1 格起 |
| `origin.z` | int | `floor(self.z) - 3` | |
| `size` | int[3] | `[7, 4, 7]` | `[width, height, depth]`，硬编码，暂不可配 |
| `palette` | string[] | 去重方块名 | 无名方块记 `id:<type>`；读取失败记 `unknown` |
| `data` | int[] | 长度 196 | `palette` 下标；遍历顺序 `y` 外层 → `z` 中层 → `x` 内层（`x` 最快变化） |

索引公式：`data[ly * 49 + lz * 7 + lx]`，其中 `lx∈[0,7)`、`ly∈[0,4)`、`lz∈[0,7)`。
全局坐标：`(origin.x + lx, origin.y + ly, origin.z + lz)`。

### 3.5 `restrictions[]`

来自 `RestrictionCache.snapshot()`，**只保留 `allowPvp:false` 的 deny 条目**，已过 `expiresAt` 的在快照时被 `sweep()` 清理。

| 字段 | 类型 | 单位 / 取值 | 说明 |
|---|---|---|---|
| `regionId` | string | 区域名 | 无区域名时用矩形坐标串做 key |
| `box.minX/minY/minZ` | int | 方块坐标 | **扁平结构**（不是嵌套 `min/max`） |
| `box.maxX/maxY/maxZ` | int | 方块坐标 | 归一化后保证 `min ≤ max` |
| `allowPvp` | bool | 快照中为 `false` | 该矩形禁止 PVP |
| `expiresAt` | number | epoch 毫秒 | TTL 到期时间 |

### 3.6 `chat[]`

| 字段 | 类型 | 单位 / 取值 | 说明 |
|---|---|---|---|
| `at` | number | epoch 毫秒 | **字段名为 `at`**，不是 `ts` |
| `username` | string | 发言者 | |
| `message` | string | 正文 | |

### 3.7 `pendingNotices[]`

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | string | `RESTRICTED` / `CHAT` / `IDLE` / `EVENT` |
| `at` | number | epoch 毫秒 |
| `data` | object | 按 `kind` 变化；例：`{"name":"damage_taken","amount":4.0}` |

---

## 4. 训练用 19 维向量（`OBS_FIELDS` / `OBS_DIM = 19`）

编码后为 `numpy.ndarray`，`shape=(19,)`，`dtype=float32`，最后统一 `np.clip(-5.0, 5.0)`。

| idx | 名称 | 来源字段 | 归一化 / 裁剪 | 取值含义 | 缺失缺省 |
|---|---|---|---|---|---|
| 0 | `self_health` | `self.health` | `/20` | 1.0 = 满血，0 = 死亡 | 缺失按 `20` → **1.0** |
| 1 | `self_food` | `self.food` | `/20` | 1.0 = 不饿 | 缺失按 `20` → **1.0** |
| 2 | `self_saturation` | `self.saturation` | `/5` | 1.0 = 饱和满 | 缺失 → `0.0` |
| 3 | `self_on_ground` | `self.onGround` | 布尔 → `1.0/0.0` | 1 = 着地（可跳/无坠落） | 缺失 → `0.0` |
| 4 | `self_yaw_sin` | `self.yaw` | `sin(yaw)` | 水平朝向的正弦分量；用 sin/cos 避免 ±π 处跳变 | yaw 缺失按 `0` |
| 5 | `self_yaw_cos` | `self.yaw` | `cos(yaw)` | 水平朝向的余弦分量；与第 4 维合成单位向量 | 同上 |
| 6 | `self_pitch` | `self.pitch` | `clamp(pitch / (π/2), -1, 1)` | -1 = 仰望正上，+1 = 俯视正下 | 缺失按 `0` |
| 7 | `target_dx` | `entities` 目标的 `pos.x` − `self.pos.x` | `/16` | 对手相对 x 偏移，1.0 ≈ 16 格 | 无目标 → `0.0` |
| 8 | `target_dy` | 同上 y | `/8` | 垂直高差，1.0 ≈ 8 格 | 无目标 → `0.0` |
| 9 | `target_dz` | 同上 z | `/16` | 对手相对 z 偏移 | 无目标 → `0.0` |
| 10 | `target_distance` | 目标的 `distance`（缺失时用 `dx,dy,dz` 欧氏） | `/32` | 3D 距离，1.0 ≈ 32 格 | 无目标 → `0.0` |
| 11 | `target_health` | 环境注入的 `opponent_health`（对手自身观测的 `health`）；退化到 `target.health`（实体表无此字段，实际取缺省 20） | `clamp(·/20, 0, 1)` | 1.0 = 对手满血 | 无目标 → `0.0`；有目标但无血量 → `1.0` |
| 12 | `target_on_ground` | 目标的 `onGround` | 布尔 → `1.0/0.0` | 1 = 对手着地 | 无目标 → `0.0` |
| 13 | `has_target` | 是否找到合法目标 | 布尔 → `1.0/0.0` | 1 = 视野内存在有效对手 | `0.0` |
| 14 | `step_fraction` | 环境 `steps / max_steps` | `clamp(·, 0, 1)` | 回合进度，用于时间相关策略 | `0.0` |
| 15 | `state_alive` | `payload.state` | `state == "Alive"` → `1.0/0.0` | 1 = 可行动 | 缺失按 `"Alive"` → **1.0** |
| 16 | `self_held_durability` | `self.heldItem.durability.remaining` | `clamp(·, 0, 1)` | 1 = 满耐久/无耐久概念 | 缺失 → `1.0` |
| 17 | `self_armor_remaining` | `self.armorRemaining` | `clamp(·, 0, 1)` | 1 = 四件满耐久，0 = 未穿甲 | 缺失 → `0.0` |
| 18 | `self_gapples` | `self.inventory` 中 `name == "golden_apple"` 的 `count` 之和 | `/5` 再 `clamp(0,1)` | 1.0 = ≥5 个金苹果 | 缺失 → `0.0` |

### 4.1 目标选择规则（决定 7..13 维）

`nearest_player(observation, username, opponents)`：

1. 只接受 `kind == "player"` 的实体；
2. 排除 `name` 与自身 `username` 同名的实体（大小写不敏感）；
3. `opponents` 非空时只接受名单内玩家（小写比较）——训练环境传入参赛 Bot 用户名，避免观战者/路人成为目标；
4. 在剩余候选中取 `distance` 最小者。

### 4.2 裁剪与异常值

| 规则 | 值 |
|---|---|
| 单维裁剪 | `np.clip(-5.0, 5.0)`（`OBS_CLIP`），归一化后理论上不应超出，仅防 NaN/异常协议值 |
| `NaN` / `Inf` / 非数值 | `_f()` 统一回落到该维缺省值，不写入 NaN |
| dtype | `float32`（固定，模型输入不做二次转换） |

---

## 5. 11 个离散动作（`NUM_ACTIONS = 11`）

`ACTION_NAMES` 顺序即动作编号 `0..10`。复合动作 = **按顺序发送多帧**，每帧各自带 `seq`。

### 5.1 总表

| # | 动作 | 发送帧（有序） | 参数 | 前置条件 | 退化规则 |
|---|---|---|---|---|---|
| 0 | `idle` | `idle` | `durationMs = 180` | 无 | 无 |
| 1 | `approach` | `look`, `move` | `direction:"forward"`，`durationMs = 180` 或 `360`，`sprint` | 需有目标 | 无目标 → `idle(180)` |
| 2 | `attack` | 近距离：`look`, `attack`, `look`；够不到：`look`, `move` | `targetEntityId`（int） | 需有目标且 `distance ≤ reach` | 无目标 / 无 `id` / 超出 `reach` → 前进冲刺 |
| 3 | `strafe_left` | `look`, `move` | `direction:"left"`, `durationMs:180` | 需有目标 | 无目标 → `idle(180)` |
| 4 | `strafe_right` | `look`, `move` | `direction:"right"`, `durationMs:180` | 需有目标 | 无目标 → `idle(180)` |
| 5 | `retreat` | `look`, `move` | `direction:"back"`, `durationMs:180` | 需有目标 | 无目标 → `idle(180)` |
| 6 | `jump` | `jump` | `{}` | 无 | 无 |
| 7 | `use_rod` | 有效距离：`look`, `use{item:"rod"}`；距离不合适：`look`, `move` | `item:"rod"` 或 `direction:"forward", sprint:true` | 需有目标，且 `2.0 ≤ distance ≤ 6.0` | 无目标 → `idle(180)`；`distance > 6` 或 `< 2` → 前进冲刺 |
| 8 | `eat_gapple` | 血量未满：`use{item:"gapple", holdMs:1600}`；血量满：`idle` | `holdMs = 1600` | `self.health < 19.5` | `health ≥ 19.5` → `idle(180)` |
| 9 | `block` | `use{item:"sword", holdMs:300}` | `holdMs = 300` | 无 | 无剑 → 运行时抛 `ITEM_MISSING`（结果 `ok:false`） |
| 10 | `switch_sword` | `use{item:"sword", holdMs:50}` | `holdMs = 50` | 无 | 无剑 → `ITEM_MISSING` |

### 5.2 `look` 帧的构造

所有带目标的移动/攻击动作都以 `look` 开头，让 Bot 面向对手（不依赖模型学朝向）：

| 量 | 公式 | 说明 |
|---|---|---|
| `dx, dz` | `target.pos - self.pos` | 水平分量 |
| `dy` | `(target.y + 1.4) - (self.y + 1.62)` | 目标胸口 vs 自身眼高 |
| `yaw` | `atan2(-dx, -dz)` | `YAW_FORMULA = "neg"`；若真实服标定相反，改为 `atan2(dx, dz)` |
| `pitch` | `atan2(-dy, horizontal)`；`horizontal < 1e-6` 时取 `0` | mineflayer 正值向下 |

发送形如 `{"type":"look","params":{"yaw":<rad>,"pitch":<rad>}}`；运行时会把 `pitch` 夹到 `[-90°, 90°]` 并 `look(yaw,pitch,true)`。

### 5.3 逐动作细节

**`idle`（0）** — `MOVE_MS = 180`。运行时执行时会先清空 `forward/back/left/right/jump/sprint/sneak` 全部控制状态；若该 Bot 的 `profile.assist.autoBlock` 开启且 `durationMs ≥ 100`，改为举剑格挡同样时长（`idle` 的副作用，非独立动作）。

**`approach`（1）** — `sprint = distance > 2.5`（疾跑必须与 `forward` 同时按下才生效）；`durationMs = 180`（`distance ≤ 4.0`）否则 `360`，避免远距离时一步走不到。

**`attack`（2）** — 三帧序列 `look → attack → look`，第二帧 `targetEntityId` 来自 `entities[].id`，第三帧是击退后的重新对准，减少背对空挥。
退化条件三条任一成立即改为「前进」：目标不存在、`target.id` 为 `null`、`distance > reach`（默认 `3.99`，由 `BotProfile.reach` 决定）。退化的前进帧 `sprint = distance > 2.5`。
运行时侧还有两层额外约束（不属于 schema，但影响结果）：超出 `reach` 抛 `OUT_OF_REACH`；规则区域拒绝抛 `RULE_DENIED`；`profile.assist.autoRod` 开启时中距离（2.5 < d ≤ 4.5，冷却 1.5s）会先自动放鱼竿。

**`strafe_left` / `strafe_right`（3 / 4）** — 横向绕圈，`look` 每帧重算，因此绕行的同时始终面向对手。

**`retreat`（5）** — `direction:"back"`，用于脱离攻击距离；`look` 仍然面朝对手（后退不转身）。

**`jump`（6）** — 单帧，运行时按住 `jump` 50ms 后释放；不改变水平控制状态，可与上一帧的 `move` 叠加。

**`use_rod`（7）** — 鱼竿击退是**中距离**手段：`2.0 ≤ distance ≤ 6.0` 才真的右键放竿（运行时：切鱼竿 → 右键 → 50ms → 切回剑）；否则退化为前进冲刺，用来进入有效窗口。两个分支都先发 `look`。

**`eat_gapple`（8）** — 需要按住右键 1.6s（原版吃食物时长）。血量 `≥ 19.5` 时吃苹果是纯浪费，因此退化为 `idle`，让模型学会「什么时候吃」。运行时：切金苹果 → `activateItem` → 等 `holdMs` → `deactivateItem`。

**`block`（9）** — 举剑格挡的**显式**版本（区别于 `idle` 的自动格挡副作用）：切剑 → 右键按住 300ms → 松开。1.8.9 格挡可减免正面伤害。

**`switch_sword`（10)** — 从其它物品（金苹果/鱼竿/方块）切回剑：`use{item:"sword", holdMs:50}` 等价于「选中剑 + 极短右键」，用于训练切物品能力。与 `block` 的差别只在按住时长（50ms vs 300ms）。

### 5.4 与协议动作的对应关系

| 离散动作使用的协议 verb | 运行时实现 | 参数约束 |
|---|---|---|
| `move` | `setControlState(direction, true)` + 延迟释放（120ms 宽限） | `direction ∈ forward/back/left/right/sprint/sneak`；`durationMs` 夹到 `1..maxMoveMs(5000)`；`sprint` 仅在 `direction==="forward"` 时生效 |
| `look` | `bot.look(yaw, pitch, true)` | `yaw/pitch` 必须为有限数字；`pitch` 夹到 `[-90,90]` |
| `jump` | `setControlState('jump', true)` 50ms | — |
| `attack` | `bot.attack(entity)`，含 CPS 闸门与反应延时 | `targetEntityId` 必须为整数且实体存在 |
| `use` | `activateItem()` / 物品细化流程 | `item ∈ rod/gapple/sword/blocks/food/armor`；`select:true` 仅切换不使用 |
| `idle` | 清空全部控制状态后等待 | `durationMs` 夹到 `1..5000` |

---

## 6. 编码与兼容规则

### 6.1 `encoding: json-v1`

| 规则 | 内容 |
|---|---|
| 观测编码 | `observationSpec.encoding = "json-v1"`：UTF-8 JSON 文本帧，字段名语义化，数组扁平 |
| 动作编码 | `actionSpec.encoding = "json-v1"`：`{type, params}` 两段式，`params` 为扁平对象 |
| 二进制帧 | v1 不使用，为未来压缩编码保留 |
| 版本号 | 信封 `v = 1`，`HELLO.payload.protocol = "gladiator-env"`；不匹配即 `VERSION_UNSUPPORTED` |

### 6.2 字段只增不改

- 观测字段**只增不改**：新增字段必须追加在 `OBSERVATION_FIELDS` 末尾，消费方按**名字**读取，禁止按下标。
- 19 维向量是**训练侧**契约：新增观测字段**不自动**进向量，必须同时改 `OBS_FIELDS`、`OBS_DIM`、`encode_observation`，并递增模型输入版本；否则旧模型权重与新向量错位。
- 删除字段、改单位、改字段名属于破坏性变更：必须递增协议 `v` 并记录 ADR（`docs/protocol.md` §6 检查清单）。
- `actionSpec.types` 追加新 verb 是兼容变更；移除或改语义是破坏性变更。

### 6.3 缺失字段的默认值

| 位置 | 字段缺失时的行为 |
|---|---|
| `self.health` | 训练侧按 `20` 处理 → 第 0 维 = `1.0` |
| `self.food` | 按 `20` → 第 1 维 = `1.0` |
| `self.saturation` | `0.0` |
| `self.yaw` / `self.pitch` | `0.0`（运行时侧用逻辑或兜底为 0） |
| `self.onGround` | `false` → `0.0` |
| `self.heldItem` | `null` → 第 16 维 = `1.0`（视作满耐久） |
| `self.heldItem.durability.remaining` | `1.0` |
| `self.armorRemaining` | `0.0`（未穿甲语义） |
| `self.inventory` | 空数组 → 第 18 维 = `0.0` |
| `entities` | 空数组 / 缺失 → 第 7..13 维全 `0.0`，第 13 维 `has_target = 0` |
| `entities[].heldItem` | `null` |
| `entities[].distance` | 用 `dx,dy,dz` 欧氏距离补算 |
| `payload.state` | 训练侧按 `"Alive"` → 第 15 维 = `1.0` |
| `restrictions` / `chat` / `pendingNotices` | 缺省为 `[]`（会话取不到时分别回落 `[]`） |
| `gameId` | `""` |
| `tick` | `0` |

### 6.4 观测未就绪时的行为

| 阶段 | 行为 |
|---|---|
| 会话未登录（`session` 或 `session.bot` 为空） | **不发送** `OBSERVATION`，定时器空转（`startObservation` 直接 `return`） |
| 客户端消费不过来（`bufferedAmount > 1 MiB`） | 跳过本次推送，不排队（协议 §5） |
| 客户端等待超时（`observation_timeout`，默认 2.0s） | `_refresh_raw` 回落到 `latest()`，仍为空则沿用上一帧；`_latest_raw` 无缓存时调用 `wait_observation` 并可能抛 `LinkError` |
| 单帧为空对象 `{}` 编码 | 得到 `[1.0, 1.0, 0, 0, 0, 1.0, 0, 0,0,0,0,0,0,0, 0, 1.0, 1.0, 0, 0]`（第 5 维 `cos(0)=1`）：血量/饱食/`state_alive` 因缺省值为「满/存活」，**不是全零向量**；消费方应结合 `has_target`/`step_fraction` 判断 |
| `ACTION` 早于 Bot 登录 | 回 `ACTION_RESULT{ok:false, code:"BOT_NOT_READY"}` |
| 动作超出 `reach` / 物品不在包内 | 回 `{ok:false, code:"OUT_OF_REACH"}` 或 `{ok:false, code:"ITEM_MISSING"}` |

### 6.5 单帧大小

`blockWindow.data` 固定 196 个整数、`palette` 通常 < 10 项，单帧 OBSERVATION 体积在 **1–3 KB** 量级；`controlHz = 10` 时带宽约 20 KB/s/Bot。这是 7×4×7 窗口被选为默认值的原因：更大的窗口会让每帧体积线性膨胀，而 WebSocket 文本帧无法压缩。

---

## 7. 已知偏差（待协议文档对齐）

以下差异均为「代码与 `docs/protocol.md` §4 示例不一致」，本文档一律按代码记录：

| # | 项目 | `protocol.md` 示例 | 代码实际 | 影响 |
|---|---|---|---|---|
| 7.1 | `self.yaw` / `self.pitch` 单位 | 示例写 `"yaw":90.0`（暗示角度） | **弧度**（`math.sin(yaw)`、`/ (π/2)` 均按弧度） | 误按角度解析会让 `self_yaw_sin/cos` 完全错位 |
| 7.2 | `restrictions[].box` | 嵌套 `{min:{},max:{}}` | 扁平 `{minX,minY,minZ,maxX,maxY,maxZ}` | 解析 `box.min.x` 会得到 `undefined` |
| 7.3 | `chat[].ts` | `"ts"` | **`at`** | 按 `ts` 读会拿到空值 |
| 7.4 | `IDLE`（§4.5） | 定义为 L2 → L3 周期性下发 | 当前 `bot-runtime` **未发送**；`link.py` 仅收集 `_idle_events`，训练循环不消费 | 依赖 `IDLE` 做外部检索的 Layer 4 逻辑当前不会被触发 |
| 7.5 | `RESET_DONE.timeoutMs` | 示例 `180000` | 固定回 `5000` | 若上层按 180s 设超时会等不到，实际按 5s 处理 |
| 7.6 | `OBS_FIELDS` 注释 | — | `self_pitch` 注释写「pitch / 90」，代码为 `pitch / (π/2)`（等价于「弧度除以 π/2」） | 仅注释与实现表述不同，数值行为以代码为准 |
| 7.7 | `ACTION.use` 的 params | §4.3 动作表写 `itemSlot? / offhand?` | 实际接受 `item`（`rod`/`gapple`/`sword`/`blocks`/`food`/`armor`）、`holdMs`、`select`、`offhand`、`position` | 按 `itemSlot` 传参会被当成「无 item」分支，只执行 `activateItem(offhand)`，达不到预期效果 |

---

## 8. 代码依据索引

| 结论 | 出处 |
|---|---|
| 19 维字段名与顺序、`OBS_CLIP`、`MOVE_MS`、`NUM_ACTIONS`、`ACTION_NAMES` | `training/gladiator_env/spaces.py` L18–L64 |
| 19 维逐维归一化与缺省值 | `spaces.py` `encode_observation` L107–L165 |
| 目标选择（玩家/排除自身/名单过滤/最近） | `spaces.py` `nearest_player` L77–L104 |
| `look` 帧构造、`yaw_to_target`、`pitch_to_target` | `spaces.py` L175–L199 |
| 11 动作 → 协议帧与退化规则 | `spaces.py` `decode_action` L202–L269 |
| 观测顶层 9 字段清单 | `bot-runtime/src/observation.js` L10–L12 |
| `self` / `entities` / `blockWindow` / `armor` / `inventory` 字段与单位 | `observation.js` L26–L195 |
| 7×4×7 窗口与索引顺序、`origin` 计算 | `observation.js` `encodeBlockWindow` L98–L129 |
| `chat` 取末 20 条、`pendingNotices` 取末 10 条 | `observation.js` L192–L193 |
| `WELCOME` 三段 payload、`controlHz` 夹取 1..20、观测间隔 ≥50ms | `bot-runtime/src/link-server.js` L127–L131、L42、L187 |
| `ACTION_RESULT` 回填 `replyTo`、`BOT_NOT_READY` 分支 | `link-server.js` `onAction` L163–L181 |
| `RESET_DONE.timeoutMs = 5000` | `link-server.js` L151–L159 |
| `REJECT` 各错误码判定 | `link-server.js` `onFrame` L109–L161 |
| `use` 物品细化（rod/gapple/sword/blocks/food/armor）与 `holdMs` 夹取 | `bot-runtime/src/actions.js` `use` L342–L388 |
| `move` 方向枚举、`durationMs` 夹取、`sprint` 仅前进生效 | `actions.js` `move` L156–L184 |
| `idle` 清空控制状态 + 自动格挡副作用 | `actions.js` `idle` L405–L427 |
| `attack` 的 `OUT_OF_REACH` / `RULE_DENIED` / 自动鱼竿 | `actions.js` `attack` L235–L280 |
| 吃金苹果 1.6s、鱼竿切回剑、剑格挡按住 | `bot-runtime/src/items.js` L108–L142 |
| 逻辑物品名 → 正则模式 | `items.js` `ITEM_PATTERNS` L15–L20 |
| 重复 `seq` 回放、限流 `ACTION_RATE_LIMITED` | `bot-runtime/src/scheduler.js` `run` L37–L62 |
| `restrictions` 快照结构与扁平 `box` | `bot-runtime/src/restriction-cache.js` `snapshot` L99–L107 |
| L3 侧 `seq` 自增、`wait_observation` 以 `tick` 判新鲜 | `training/gladiator_env/link.py` L205–L280、L159–L176 |
| `botId = glad-<小写用户名>` | `training/gladiator_env/env.py` `BotConfig.__post_init__` |
