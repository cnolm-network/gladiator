# PROTOCOL — Layer 1 ↔ Layer 2 ↔ Layer 3

> 版本：协议 v1（2026-11-07 冻结）
> 变更规则：向后不兼容的改动必须递增 `v`，并在项目决策记录（ADR）中留档。

## 1. 传输与通用信封

- 传输：WebSocket 文本帧，UTF-8 JSON。二进制帧为未来的压缩编码保留，v1 不使用。
- 通道 A（Layer 1 ↔ Layer 2）：插件监听 `127.0.0.1:8765`，运行时作为客户端。
- 通道 B（Layer 2 ↔ Layer 3）：运行时监听 `127.0.0.1:8766`，Python 作为客户端。

统一信封：

```json
{
  "v": 1,
  "type": "ACTION",
  "seq": 1024,
  "ts": 1762500000000,
  "botId": "glad-bot-1",
  "replyTo": null,
  "payload": {}
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `v` | int | 协议主版本，不匹配则握手失败 |
| `type` | string | 见 §2 / §4 |
| `seq` | int | 单方向单调递增，从 1 开始；用于去重与乱序检测 |
| `ts` | number (ms) | 发送方毫秒时间戳，仅用于诊断，不参与排序。毫秒值超出 32 位整数范围，实现按 64 位整数处理 |
| `botId` | string | 会话标识；握手前置空 |
| `replyTo` | int\|null | 请求-响应配对时填对端 `seq` |
| `payload` | object | 见下文字段表 |

**禁止**：传输 NMS/Bukkit 对象、Java 序列化数据、文件路径、堆栈。只允许原语、字符串枚举、扁平数组。

## 2. 通道 A：Layer 1 ↔ Layer 2

### 2.1 HELLO（L2 → L1）

```json
{ "v":1, "type":"HELLO", "seq":1, "ts":0, "botId":"", "payload":{
  "token":"", "botId":"glad-bot-1", "username":"GladiatorBot1",
  "mcVersion":"1.8.8", "runtimeVersion":"0.1.0"
}}
```

### 2.2 WELCOME（L1 → L2）

```json
{ "v":1, "type":"WELCOME", "seq":1, "ts":0, "botId":"glad-bot-1", "payload":{
  "sessionId":"3f2a…", "playerUuid":"…", "serverVersion":"git-Spigot-…",
  "features":["guard.worldguard","state.machine","analyze"],
  "limits": { "maxActionsPerSecond":10, "observationHz":5, "idleAfterSeconds":15 }
}}
```

`features` 用于能力协商：运行时不得调用未声明的能力。WorldGuard 未安装时 `guard.worldguard` 缺席，运行时不得发送 `QUERY_PVP`。

`playerUuid`：已绑定真实玩家时为其 UUID；玩家实体尚未生成（或运行时先于玩家连上链路）时为空串 `""`。
运行时以此判断是否需要等待 `BOT_BOUND` 或继续指数退避重试（`BOT_NOT_READY`）。

### 2.3 REJECT（L1 → L2）

```json
{ "type":"REJECT", "payload": { "code":"AUTH_FAILED", "message":"token mismatch" } }
```

### 2.4 STATE（L1 → L2）

```json
{ "type":"STATE", "payload": {
  "state":"Alive", "previous":"Respawning", "gameId":"arena-1",
  "reason":"respawn", "at":1762500000123
}}
```

`state ∈ {Alive, Dead, Respawning, Eliminated}`。**状态由插件计算**，运行时不推导、不覆盖。

### 2.5 NOTICE（L1 → L2）

```json
{ "type":"NOTICE", "payload": {
  "kind":"RESTRICTED",
  "at":1762500000123,
  "data": {
    "action":"attack", "reason":"PVP_DENIED",
    "location":{"world":"world","x":12.5,"y":64.0,"z":-7.5},
    "regionIds":["spawn-safe"],
    "box":{"min":{"x":0,"y":60,"z":-20},"max":{"x":30,"y":80,"z":0}},
    "expiresAt":1762500060123
  }
}}
```

**`source=observed` 语义（A3，2026-11-13 追加）**：Gladiator 不查 WorldGuard、不查任何插件 API，
只在「别的插件取消了 Bot 的攻击/交互」时记一条 deny（`EntityDamageByEntityEvent` /
`PlayerInteractEvent` / `BlockPlaceEvent` / `BlockBreakEvent` 的 `isCancelled()`）。
对应通知的 `data` 形如：

```json
{ "action":"attack", "reason":"damage-cancelled", "source":"observed",
  "regionIds":[], "location":"world,-2160.5,4.0,-430.5", "expiresAt":1762500060123 }
```

它与 `source=worldguard` 的区别：
* 没有区域信息（`regionIds` 为空、`box` 由 `guard.observed-radius` 推出），因此**不能**当作领地边界使用；
* 有有效期（`guard.observed-ttl-seconds`，默认 60s），到期自动失效；
* 只对**被拒的那只 Bot** 生效（缓存按 botId 隔离）。

Layer 2 建议把 `source=observed` 的条目当作短 TTL 的软限制写入 `RestrictionCache`：
到期即丢，不要持久化。

`kind` 取值：

| kind | data 关键字段 | 语义 |
|---|---|---|
| `RESTRICTED` | action, reason, location, regionIds, box?, expiresAt, source? | 规则拒绝，写入 `RestrictionCache`。`source` 省略时按 `worldguard` 处理；`source="observed"` 表示 Gladiator 从实时事件里观测到的拒绝（A3），此时 `regionIds` 恒为空、`box` 由观测点 ± 半径推出 |
| `CHAT` | username, message | 结构化聊天摘要（与 `bot.on('chat')` 双通道） |
| `IDLE` | sinceSeconds | 空闲窗口开启，允许 Layer 4 发起外部检索 |
| `EVENT` | name, detail | 伤害/击杀/方块等对局事件的通知面 |

### 2.6 QUERY_PVP（L2 → L1） / QUERY_RESULT（L1 → L2）

```json
{ "type":"QUERY_PVP", "payload": { "location":{"world":"world","x":1,"y":64,"z":1} } }
```
```json
{ "type":"QUERY_RESULT", "replyTo":17, "payload": {
  "allowed": false, "regionIds":["spawn-safe"], "source":"flag",
  "box":{ "min":{...}, "max":{...} }
}}
```

`source ∈ {flag, global, degraded, unknown, observed}`。`degraded` = WorldGuard 未安装；
`observed` = 命中观测式 deny 缓存（A3），此时 `regionIds` 为空、`box` 为观测点 ± 半径。

### 2.7 ACTION_RESULT（L1 → L2）

服务端观察到动作被取消/无效时回写（不是动作执行结果，mineflayer 自己知道成败）：

```json
{ "type":"ACTION_RESULT", "replyTo":1024, "payload": {
  "ok": false, "code":"RULE_DENIED", "reason":"PVP_DENIED"
}}
```

### 2.8 ANALYZE（双向）

请求（管理命令或 L3）：
```json
{ "type":"ANALYZE_REQUEST", "seq":88, "payload": { "gameId":"arena-1", "windowSeconds":60 } }
```
响应：
```json
{ "type":"ANALYZE_RESULT", "replyTo":88, "payload":{
  "gameId":"arena-1",
  "perBot":[{
    "botId":"glad-bot-1","aliveSeconds":92,"damageDealt":34.0,"damageTaken":21.5,
    "blocksPlaced":6,"blocksBroken":18,"pathMeters":143.2,"ruleDenials":4,
    "actionsAccepted":811,"actionsDropped":12
  }]
}}
```

### 2.9 PING / PONG

双向，`payload: {}`；超过 `heartbeat-timeout-seconds`（默认 30s）无 PING 即判死并回收会话。

### 2.10 SPAWN / BOT_BOUND / SPAWN_FAILED（spawnBot 编排，G-107）

`GladiatorAPI.spawnBot(BotSpec)` 在主线程立即返回 `BotRequest`（`PENDING`），真正的登录由运行时完成：

1. 运行时先建立链路并发送 HELLO；
2. 插件下发 `SPAWN`（若 HELLO 早于 `spawnBot`，则在 `spawnBot` 调用时补发）；
3. 运行时用 mineflayer 以真实协议登录（不伪造服务端连接）；
4. 服务端 `PlayerJoinEvent` 绑定 `BotHandle` 后回 `BOT_BOUND`，并在主线程派发 `GladiatorBotReadyEvent`；
5. 登录失败回 `SPAWN_FAILED`，插件派发 `GladiatorBotFailedEvent`；超过 `bridge.spawn-timeout-seconds`（默认 30s）未完成同样派发 Failed，`reason` 以 `TIMEOUT:` 开头。

SPAWN（L1 → L2）：
```json
{ "type":"SPAWN", "payload": {
  "requestId":"req-1", "botId":"glad-bot-1", "username":"Glad_Bot_1", "group":"",
  "server":{ "host":"127.0.0.1", "port":25565 }, "metadata":{},
  "profile":{ "cps":15.0, "cpsJitter":2.0, "reactionMs":220.0, "reactionJitterMs":40.0,
              "reach":3.99, "model":"default" }
}}
```
`server` 是插件视角的 MC 地址（`Bukkit.getIp()`/`getPort()`），运行时可用它覆盖自身启动参数。`metadata` 原样透传 `BotSpec` 的自定义键。
`profile` 是该 Bot 的手感配置（`BotProfile`，G-507）：`cps` 每秒点击数、`cpsJitter` 点击间隔抖动、`reactionMs` 反应延时、`reactionJitterMs` 延时抖动、`reach` 攻击距离（格，原版 3.99）、`model` 训练模型标识（默认 `default`，不参与协议语义）。运行时应以 `profile` 覆盖自身 CLI/env 默认值；字段缺失时逐项回落默认值。

BOT_BOUND（L1 → L2）：
```json
{ "type":"BOT_BOUND", "payload": {
  "sessionId":"3f2a…", "playerUuid":"…", "username":"Glad_Bot_1",
  "state":"Alive", "ready":true
}}
```

SPAWN_FAILED（L2 → L1）：
```json
{ "type":"SPAWN_FAILED", "payload": {
  "requestId":"req-1", "botId":"glad-bot-1",
  "code":"ONLINE_MODE", "message":"服务端开启了 online-mode，但 Bot 以离线身份登录…"
}}
```

`botId` 两端统一为 `glad-<小写用户名>`（`BotRegistry.botIdFor`），运行时必须使用同一公式，否则对不上请求。

## 3. 错误码

| code | 方向 | 含义 | 运行时行为 |
|---|---|---|---|
| `AUTH_FAILED` | L1→L2 | token 不匹配 | 停止重连，等待人工修复 |
| `VERSION_UNSUPPORTED` | 双向 | `v` 不匹配 | 停止，输出双方版本 |
| `SESSION_UNKNOWN` | L1→L2 | sessionId 已回收 | 重新 HELLO |
| `BOT_NOT_READY` | L1→L2 | 玩家实体尚未生成 | 指数退避重试（≤5 次） |
| `ACTION_UNSUPPORTED` | 双向 | 未声明的动作 | 丢弃，上报 L3 |
| `ACTION_RATE_LIMITED` | L2 内部 | 超过速率上限 | 丢弃并计数，不排队 |
| `RULE_DENIED` | L1→L2 | 规则拒绝 | 写缓存 + 不重试同一动作 |
| `INTERNAL` | 双向 | 未预期异常 | 记录日志，会话可继续 |

`SPAWN_FAILED.payload.code` 取值（登录期专用，随 Failed 事件交给调用方）：

| code | 含义 | 典型修复 |
|---|---|---|
| `VERSION_MISMATCH` | mineflayer 协议版本与服务端不符 | 1.8.9 服务端用 `version:"1.8.8"` |
| `ONLINE_MODE` | 服务端 online-mode 与 Bot 的 auth 方式不匹配 | 服务端 `online-mode=false` 或改用正版账号 |
| `NOT_WHITELISTED` | Bot 用户名不在白名单 | 加入 `whitelist.json` |
| `SERVER_UNREACHABLE` | MC 端口不可达 | 检查地址/端口与服务端状态 |
| `LOGIN_FAILED` | 其余登录错误 | 看 `message` 原始信息 |

## 4. 通道 B：Layer 2 ↔ Layer 3

> 字段级 schema（每个 payload 字段的类型、单位、缺省值、19 维观测向量逐维定义、11 个离散动作与协议帧的映射）见 [`docs/observation.md`](observation.md)。本节只定义帧与信封。

### 4.1 HELLO / WELCOME

```json
{ "type":"HELLO", "payload": { "botId":"glad-bot-1", "v":1, "protocol":"gladiator-env" } }
```
```json
{ "type":"WELCOME", "payload": {
  "observationSpec":{ "encoding":"json-v1", "fields":[...] },
  "actionSpec":{ "encoding":"json-v1", "types":["move","look","jump","attack","dig","place","use","talk","idle"] },
  "controlHz":10
}}
```

### 4.2 OBSERVATION（L2 → L3）

```json
{ "type":"OBSERVATION", "payload":{
  "tick": 4211, "gameId":"arena-1", "state":"Alive",
  "self": {
    "pos":{"x":12.5,"y":64.0,"z":-7.5}, "yaw":90.0, "pitch":0.0,
    "onGround":true, "health":20.0, "food":18, "saturation":5.0,
    "heldItem":{"name":"iron_sword","count":1,"slot":0},
    "inventory":[{"name":"golden_apple","count":3,"slot":1}]
  },
  "entities":[{"id":91,"kind":"player","name":"GladiatorBot2","pos":{...},"distance":3.2}],
  "blockWindow": { "origin":{"x":9,"y":61,"z":-11}, "size":[7,4,7], "palette":["air","stone"], "data":[0,0,1,...] },
  "restrictions":[{"regionId":"spawn-safe","box":{...},"allowPvp":false,"expiresAt":1762500060123}],
  "chat":[{"ts":1762500000000,"username":"admin","message":"round start"}],
  "pendingNotices":[{"kind":"EVENT","data":{"name":"damage_taken","amount":4.0}}]
}
```

`blockWindow` 用「调色板 + 索引」压平，避免逐方块对象；`size` 默认 `7×4×7`，可在配置中调整。

### 4.3 ACTION（L3 → L2）

```json
{ "type":"ACTION", "seq":1024, "payload":{
  "type":"place",
  "params":{ "block":"cobblestone","against":{"x":12,"y":63,"z":-8},"face":"up" }
}}
```

动作表（v1）：

| type | params | 对应 mineflayer |
|---|---|---|
| `move` | direction ∈ forward/back/left/right, durationMs | `setControlState` |
| `look` | yaw, pitch | `look(yaw,pitch,force)` |
| `jump` | — | `setControlState('jump',true)` 一 tick |
| `attack` | targetEntityId | `attack(entity)` |
| `dig` | position | `dig(block)` |
| `place` | block, against, face | `placeBlock(refBlock, face)` |
| `use` | itemSlot? / offhand? | `activateItem()` |
| `talk` | message | `chat(message)` |
| `idle` | durationMs | 停止所有控制状态 |

- `seq` 单调递增；重复 `seq` 直接丢弃并回 `replyTo` 缓存结果（幂等）。
- 动作执行失败返回同信封 `ACTION_RESULT`（通道 B），`ok=false` 必须带 `code`。

### 4.4 RESET / RESET_DONE

```json
{ "type":"RESET", "payload":{ "resetId":7, "gameId":"arena-1", "seed":null } }
```
```json
{ "type":"RESET_DONE", "replyTo":7, "payload":{ "state":"Alive","tick":0, "timeoutMs":180000 } }
```

`RESET` 与 Layer 1 的 `pushGame` / `pushPlace` 边界对应：插件负责实际的对局/训练场重置，运行时只负责报告 Bot 已就绪。

### 4.5 IDLE（L2 → L3）

```json
{ "type":"IDLE", "payload":{ "sinceSeconds":18, "windowSeconds":30 } }
```

Layer 3 收到后回调策略层；外部检索耗时超过 `windowSeconds` 应放弃本次并等下一个窗口。

## 5. 背压与限流

| 位置 | 策略 |
|---|---|
| L3 → L2 动作 | 超过 `maxActionsPerSecond` 丢弃，回 `ACTION_RATE_LIMITED` |
| L2 → L3 观测 | 队列深度 > 8 时丢弃旧帧，保留最新 |
| L2 → L1 通知 | 单连接队列深度 > 256 时合并同类 `EVENT` |
| L1 事件 → 观测 | 按 `observationHz` 采样，不逐事件转发 |

## 6. 兼容性检查清单

改动协议时必须同步：
1. 本文档 `v` 号与 §2/§4 字段表。
2. `bridge/dto/` 与 `bot-runtime/src/link/` 的字段定义。
3. `training/gladiator_env/` 的观测/动作映射。
4. 项目决策记录（ADR）中新增一条。
