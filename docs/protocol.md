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

`kind` 取值：

| kind | data 关键字段 | 语义 |
|---|---|---|
| `RESTRICTED` | action, reason, location, regionIds, box?, expiresAt | 规则拒绝，写入 `RestrictionCache` |
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

`source ∈ {flag, global, degraded, unknown}`。`degraded` = WorldGuard 未安装。

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

## 4. 通道 B：Layer 2 ↔ Layer 3

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
