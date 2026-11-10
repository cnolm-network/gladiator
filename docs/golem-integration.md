# golem 接入指南（`spawn.mode=golem`）

> 适用版本：Gladiator `0.1.0`（api `1.1.0`）+ Golem `0.1.0`，服务端 Paper/Spigot 1.8.8（协议 47）。
> 对外 API 未变化：仍是 `spawnBot(BotSpec) → BotRequest` + `GladiatorBotReadyEvent`，
> 变的只是「Bot 的身体从哪来」。权威接口契约见 [`api-contract.md`](api-contract.md) §B1，
> 本文只回答「怎么装、怎么配、怎么验、有什么坑」。

---

## 1. 定位与边界

**Golem 是一个独立插件**（`top.cnolm.golem.api`），负责在服务端进程内生成 NMS 身体
（真实 `EntityPlayer` + 无 socket 的连接），对外暴露 `GolemService` 与 `BotBody` 两个 API 面。
它可独立使用（只装 Golem + `/golem` 命令），也可被 Gladiator 通过 Bukkit `ServicesManager` 装配。

Gladiator 侧的边界如下：

| 项 | 约束 |
|---|---|
| 依赖形态 | `compileOnly files('libs/golem-api-0.1.0.jar')`，只依赖 **api 子包**；运行期由服务器上的 Golem 提供实现 |
| `plugin.yml` | `softdepend: [WorldGuard, Golem]` — **绝不能写成 `depend`**，否则没装 Golem 的服务器会直接跳过 Gladiator 的加载 |
| 代码边界 | 只 import `top.cnolm.golem.api.*`；`top.cnolm.golem.nms.*` / `.plugin.*` 的 import 会让 `verifyStandalone` 门禁失败 |
| 装配隔离 | `GladiatorMain` 不直接引用 golem 类，只经 `GolemBootstrap` 调用；golem-api 不在运行期时由 `catch (Throwable)` 降级，不崩服 |
| 独立性 | Gladiator 仍是独立插件：不 import / depend 竞技场插件，状态机自有，传送/套件/结算仍归调用方 |

调用链固定为：**竞技场插件 → golem（身体）→ gladiator（状态机 / 规则 / API）**，
详见 `docs/decisions.md` 的 ADR-018 及其修订记录。

> Gladiator 本体仍**不写 NMS**。「禁止在服务端 JVM 内伪造 PlayerConnection / CraftPlayer」这条约束
> 依然约束 gladiator 源码；NMS 身体由 golem 项目承担（ADR-018 修订 5）。

---

## 2. 两种 Bot 来源对比

`spawn.mode` 只有两个取值：`runtime`（默认）与 `golem`。

| 维度 | `runtime`（mineflayer） | `golem`（NMS 身体） |
|---|---|---|
| 登录方式 | 真实协议登录，经通道 A（WS `127.0.0.1:8765`）握手 → SPAWN → `PlayerJoinEvent` | 服务端进程内直接生成 `EntityPlayer`，**不经过网络、不产生登录包** |
| 进程要求 | 每个 Bot 一个 Node 进程（`bot-runtime/src/index.js`） | **不需要 bot-runtime**，无需任何外部进程 |
| UUID | 服务端按离线用户名生成 | 由 golem 身体决定，`Bukkit` 侧照常可读 |
| 事件完整性 | 完整：加入/退出/移动/伤害/死亡/重生/交互全由服务端原生派发 | 伤害/死亡/重生原生；`PlayerMoveEvent` 由 golem 补发；交互类事件缺失（见 §9） |
| 观测 / 动作（通道 B） | 有：`OBSERVATION` / `ACTION` / `RESET` / `IDLE`，10Hz | **无**：没有 WS 会话，`notice()` 落本地队列，`STATE` 帧不下发 |
| 可训练性 | 可（`GladiatorEnv` + `training/`） | 不可（没有观测/动作通道；动作级控制要走 golem 的 `BotBody` 原语） |
| 适用场景 | 训练、压测、需要完整协议行为验证 | 对局 / 陪练 / 演出：启动快、无外部进程、无重名与登录失败问题 |
| 票据语义 | `spawnBot` 返回 `PENDING`，随后 `PlayerJoinEvent` 绑定 → 主线程 `GladiatorBotReadyEvent` | 通常**同一 tick** 返回 `READY`；个别实现晚一 tick，仍走 `PENDING` + `Ready` |
| 失败模式 | `TIMEOUT:`（30s 等不到登录）、`SPAWN_FAILED`（`VERSION_MISMATCH` / `ONLINE_MODE` / `NOT_WHITELISTED` / `SERVER_UNREACHABLE`） | `GOLEM_TIMEOUT:`（等 `bukkitPlayer()` 超时）、`GOLEM_SPAWN_ERROR:`（golem 侧抛错）；未装 Golem 时直接抛中文「spawnBot 不可用：…」 |
| 回收 | 会话断开 / Bot 退出 → `GladiatorBotRemovedEvent` | golem 侧 `despawn` **不发 Bukkit 事件** → 2 秒一次对账，`reason=golem-body-gone` |

**两条链路是并存的**：训练仍走 mineflayer + 通道 B（观测/动作）；对局/陪练可走 golem 身体。
同一个服务器进程一次只能选一种 `spawn.mode`（启动期配置，`/gladiator reload` 不切换来源）。

---

## 3. 配置

### 3.1 `config.yml`

```yaml
spawn:
  # runtime = 外部 mineflayer 运行时（默认，走通道 A 真实登录）
  # golem   = 进程内 NMS 身体（需要 Golem 插件；不登录、不依赖 bot-runtime）
  # golem 缺失时插件仍会启用，但 spawnBot 会返回"未检测到 Golem"的中文错误。
  mode: golem
  golem:
    # 身体生成世界；留空取第一个世界；世界名不存在时打 WARNING 并回落第一个世界
    world: ""
    # 生成坐标 x,y,z；留空取该世界的出生点；格式非法直接抛错（中文原因）
    location: ""
```

| 键 | 默认 | 说明 |
|---|---|---|
| `spawn.mode` | `runtime` | `golem` 以外的任何值都会打 WARNING 并回落到 `runtime` |
| `spawn.golem.world` | `""` | 空 = `Bukkit.getWorlds().get(0)`；名字不存在 = WARNING + 回落 |
| `spawn.golem.location` | `""` | 空 = `world.getSpawnLocation()`；必须是 `x,y,z`（三个 double） |
| `bridge.spawn-timeout-seconds` | `30` | **golem 模式复用此值**作为「等 `bukkitPlayer()` 就绪」的上限 |

> `spawn.mode` 与 `spawn.golem.*` 都是**启动期配置**：改完必须重启服务端；
> `/gladiator reload` 只重载 `training.*` 等运行期键，不会切换 spawn 来源。

### 3.2 `plugin.yml`

```yaml
# Golem 是可选依赖：仅 spawn.mode=golem 时需要。绝不能写成 depend，
# 否则没装 Golem 的服务器会直接跳过 Gladiator 的加载。
softdepend: [WorldGuard, Golem]
```

### 3.3 部署位置

```
server/plugins/
├── gladiator-0.1.0.jar     # Gladiator（本仓库构建）
└── golem-0.1.0.jar         # Golem（独立项目构建）
```

`golem` 模式下 `server/plugins/` 里**不需要** bot-runtime 的任何产物，也不需要 Node 进程与 8770+ 端口。

---

## 4. 装配时序

### 4.1 正常路径（装了 Golem 且 `spawn.mode=golem`）

```text
onEnable()
  ├─ saveDefaultConfig()  /  detectWorldGuard()
  ├─ service = new GladiatorServiceImpl(this)
  │    └─ setGuardBridge(WorldGuardBridge | NoopGuardBridge)
  ├─ spawnTimeout = config.bridge.spawn-timeout-seconds
  ├─ wireSpawner(spawnTimeout)                    ← 按 spawn.mode 分派
  │    └─ GolemBootstrap.install(plugin, service, spawnTimeout)
  │         ├─ plugin.getServer().getServicesManager().load(GolemService.class)
  │         │     └─ 拿不到 / 不是 GolemService → return null（走 4.2）
  │         ├─ golem.available() == false → WARNING + return null（走 4.2）
  │         ├─ new GolemBodySpawner(plugin, service, golem, spawnTimeout)
  │         ├─ PluginManager.registerEvents(spawner, plugin)   ← PlayerQuitEvent 回收
  │         ├─ spawner.start()                                  ← 40 tick 对账任务
  │         └─ 日志：spawn mode=golem · Golem <version> | 身体由 golem 在进程内生成，无需 bot-runtime 登录
  ├─ service.attachSpawner(golemSpawner)          ← API 的唯一 spawn 出口
  ├─ spawnShutdown = () -> GolemBootstrap.shutdown(attached)
  ├─ registerEvents(PlayerStateListener) / registerPvpListener()
  ├─ register ServicesManager(GladiatorAPI) + GladiatorProvider + UserStateRegistry
  ├─ setExecutor(GladiatorCommand)
  ├─ startLink()                                  ← 通道 A 照常起（见 §5.4）
  └─ service.markReady()
       └─ 日志：Gladiator 0.1.0 enabled | api=1.1.0 | ready=true | spawn=golem | link=...
```

### 4.2 降级路径（`spawn.mode=golem` 但没装 Golem）

```text
wireSpawner()
  └─ try { GolemBootstrap.install(...) }
       catch (Throwable t)                       ← NoClassDefFoundError: top/cnolm/golem/api/GolemService
         └─ WARNING「Golem 装配失败（spawn.mode=golem）」
  └─ golemSpawner == null
       ├─ WARNING「spawn.mode=golem 但未检测到 Golem 插件，spawnBot 将失败
       │            （装好 Golem 后重启，或把 spawn.mode 改回 runtime）」
       ├─ spawner = new UnavailableSpawner("spawn.mode=golem 但未检测到可用的 Golem 插件")
       └─ service.attachSpawner(unavailable)     ← 插件照常启用
```

`UnavailableSpawner` 的行为是**可解释的失败**，而非伪造成功：

- 插件正常 enable，`/gladiator status` 显示 `spawn=golem`；
- `/gladiator list` 正常（空注册表，返回 `bots=0 pending=0` 而不是 NPE）；
- `/gladiator spawn <user>` 抛 `IllegalStateException`，命令层转成
  `[Gladiator] spawn 失败: spawnBot 不可用：spawn.mode=golem 但未检测到可用的 Golem 插件`；
- 不返回任何假票据、不产生 `GladiatorBotReadyEvent`。

### 4.3 关闭

```text
onDisable()
  ├─ spectatorService.shutdown()        ← 先还原在线观战者
  ├─ spawnService.shutdown()            ← runtime 模式
  ├─ spawnShutdown.run()                ← golem 模式：取消对账任务、清空注册表、
  │                                        静默丢弃 pending（不派发 Ready/Failed）
  ├─ LinkServer 停止 / ServicesManager 注销 / GladiatorProvider.uninstall()
```

---

## 5. `spawnBot` 在 golem 模式下的行为

### 5.1 返回票据

```java
BotRequest request = GladiatorAPI.spawnBot(BotSpec.of("Goblin_A", "default"));
// 通常：request.getState() == READY，request.getBot() != null（同一 tick）
// 少数实现：PENDING，随后 GladiatorBotReadyEvent
```

- 同一 tick 能取到 `BotBody#bukkitPlayer()` 就**直接返回 `READY`**，不派发 `Pending` 后的等待；
- 取不到就按 tick 重试（`runTaskTimer`，周期 1 tick），日志打印
  `spawnBot req-N bot=glad-xxx user=xxx (golem body 已创建，等待 Player 视图)`；
- 超过 `bridge.spawn-timeout-seconds` → `GladiatorBotFailedEvent(reason="GOLEM_TIMEOUT: 等待 Golem 身体就绪超过 30s")`，
  并 `golem.despawn(body.id(), DespawnReason.REQUESTED)` 清掉残留身体；
- `golem.spawn` 抛错 → `GOLEM_SPAWN_ERROR: <message>`。

### 5.2 事件在主线程

`GladiatorBotReadyEvent` / `GladiatorBotFailedEvent` / `GladiatorBotRemovedEvent`
全部由主线程派发，调用方可以**直接在事件里**做 `teleport`、发套件、`pushGame`，不需要再切线程
（与 runtime 模式一致）。

`spawn` 本身要求主线程，否则抛
`spawnBot 必须在服务端主线程调用（当前: <thread>）`。

### 5.3 `BotHandle` 绑定

```java
BotHandle handle = new BotHandle(pending.botId, player.getName());  // botId = glad-<小写用户名>
handle.bind(player);        // ← Player 来自 BotBody#bukkitPlayer()
registry.register(handle);
service.registerBot(handle);
```

`bind(Player)` 之后：

- `handle.asPlayer()` 返回该 `EntityPlayer` 的 Bukkit 视图，调用方可以正常 `teleport`、读写背包、发套件；
- `handle.getUniqueId()` 是身体的 UUID；
- **`BotHandle.isReady()` 放宽为 `player.isOnline() || player.isValid()`**——golem 身体不进 `PlayerList`，
  1.8.8 的 `Player#isOnline()` 实现是 `CraftServer.getPlayer(uuid) != null`，只用 `isOnline()` 会恒为 false，
  导致 `pushPlace` 直接拒绝、`train` 不可用。此缺陷已在 B1 修复。

### 5.4 `playerUuid` 与 WELCOME

golem 模式**没有通道 A 会话**，因此不产生 `WELCOME` 帧。但通道 A 服务端仍照常启动，
`BotSpawner#playerUuid(username)` 的实现仍有效：

```java
// GolemBodySpawner#playerUuid
BotHandle handle = registry.findByUsername(username);
if (handle != null) return handle.getUniqueId().toString();
Player player = Bukkit.getServer().getPlayerExact(username);   // 兜底
return player == null ? null : player.getUniqueId().toString();
```

含义：如果同一台服务器上仍有 runtime 客户端连入通道 A（例如 `spawn.mode=golem` 下另起了一个
用于调试的 runtime），`WELCOME.playerUuid` 也能解析到已绑定的身体 UUID，而不是空串。
纯 golem 部署下这段逻辑不会被触发。

---

## 6. 事件与状态机

### 6.1 原生触发的事件

golem 身体是服务端内真实实体，1.8.9 的原生路径可直接命中：

| 行为 | 原生落点 | Gladiator 的用法 |
|---|---|---|
| 受伤 | `EntityLiving#d(DamageSource, float)`（`damage(...)`） | `SessionCounters` 记录伤害；`PlayerStateListener` 只读 |
| 死亡 | `EntityPlayer#die(DamageSource)` | `PlayerDeathEvent` → `transition(DEAD, "player-death")` |
| 重生 | `EntityPlayer#die` 后的重生流程 | 状态机 `DEAD → RESPAWNING → ALIVE`；**但见 §9 的 `RESPAWN` 限制** |
| 移动 | golem **补发** `PlayerMoveEvent` | 观测/统计、`SpectatorService` 跟随 |
| 退出 | 若走原版登出路径则发 `PlayerQuitEvent` | `GolemBodySpawner#onPlayerQuit`（MONITOR）回收注册 |

Gladiator 侧所有监听器都是 **MONITOR 只读**，从不 `setCancelled`——状态机是自有写入点，
竞技场插件取消死亡/重生事件时由调用方自行处理（A9 场景）。

### 6.2 `golem-body-gone` 清理路径

golem 的 `despawn`（例如执行 `/golem remove Goblin_A`）**不会**派发任何 Bukkit 事件，
因此 Gladiator 用周期对账兜底：

```text
每 40 tick（2s）：
  for handle in registry.all():
      body = golem.get(BotId.of(handle.getUsername()))
      if body == null || body.bukkitPlayer() == null:
          ── removeHandle(handle, "golem-body-gone")
               ├─ registry.unregister(handle)
               ├─ service.unregisterBot(handle, reason)   ← 会话/BotHandle 回收
               ├─ handle.unbind()                          ← Player 引用释放
               ├─ 日志：Bot 退出: Goblin_A reason=golem-body-gone
               └─ 未结束的训练/对局会话一并 endSession
```

实测：`/golem remove Goblin_A` 后 **4 秒内** `/gladiator list` 的 `bots` 归零，日志出现
`Bot 退出: Goblin_A reason=golem-body-gone`（对账周期 2s，最坏 2 个周期的可见延迟）。

刻意**不**在此回收的情况：身体死亡但仍在 golem 注册表里（`DeathPolicy.KEEP`）。
这种状态交给状态机走 `DEAD` / `RESPAWNING`，否则重生后会变成「身体还在、注册已丢」。

---

## 7. 构建依赖（`golem-api` 本地构建）

`golem-api-0.1.0.jar` **不入库**（`.gitignore` 忽略 `libs/*.jar`，只保留 `libs/.gitkeep`），
也**不打进 shadowJar**（`compileOnly`）。因此从公开仓库克隆后**必须先做这一步**，否则 `compileJava` 会报
「找不到符号：top.cnolm.golem.api.*」。

```powershell
# 1) 在 golem 项目里构建 api jar（需要本地已存在 1.8.8 服务端 jar 供 NMS 编译）
cd D:\cbot-golem
.\gradlew :golem-api:jar -PgolemNmsJar=D:/ServerUltrapvp/spigot-1.8.8.jar

# 2) 复制到 gladiator 的 libs/
Copy-Item golem-api\build\libs\golem-api-0.1.0.jar D:\cbot-gladiator\libs\ -Force

# 3) 回到 gladiator 正常构建（构建期即可用，运行期是否生效取决于服务器有没有装 Golem）
cd D:\cbot-gladiator
.\gradlew.bat build --offline
```

`build.gradle` 中的三行声明（不要改成 `implementation` / `shadow`）：

```gradle
compileOnly files('libs/golem-api-0.1.0.jar')       // 主代码
testCompileOnly files('libs/golem-api-0.1.0.jar')   // 单测
testRuntimeOnly files('libs/golem-api-0.1.0.jar')   // 单测运行期
```

注意事项：

- **不要把 gladiator 的构建指向 `D:/cbot-golem` 的源码目录**，也不要让 golem 依赖 gladiator；
  两者只通过 Bukkit `ServicesManager` + 这个 jar 的 API 面交互。
- 升级 golem 的 API 时，同步改 `build.gradle` 里的文件名与 `libs/` 下的 jar。
- `verifyStandalone` 门禁只放行 `top.cnolm.golem.api` 包与 `softdepend: [Golem]`；
  任何 `top.cnolm.golem.nms.*` / `.plugin.*` 的 import 都会让 `check` 失败。
- 完整环境说明见 [`ENVIRONMENT.md`](ENVIRONMENT.md) §11（含 `--offline` 用法与镜像注意事项）。
- **不要修改 `D:\cbot-golem`**：它是独立仓库，对本仓库而言是只读参考（与 `D:\ServerUltrapvp` 同级约束）。

---

## 8. 验收清单

前置：

- Paper/Spigot 1.8.8，`server.properties`：`online-mode=false`、`server-port=25566`；
- RCON：`enable-rcon=true`、`rcon.port=25575`、`rcon.password=devpass`（仅本地联调，验完还原）；
- `server/plugins/` 放 `gladiator-0.1.0.jar` 与 `golem-0.1.0.jar`；
- `config.yml` 的 `spawn.mode` 按步骤切换后**重启服务端**。

| # | 场景 | 命令 / 操作 | 期望输出 |
|---|---|---|---|
| 1 | runtime 回归 | `spawn.mode: runtime`，起 runtime 后 `/gladiator spawn Goblin_R` | `[Gladiator] spawn 已受理 request=req-1 botId=glad-goblin_r state=PENDING mode=runtime（等待运行时登录 + PlayerJoin）`；随后控制台 `Bot ready: glad-goblin_r user=Goblin_R uuid=...` |
| 2 | runtime 状态 | `/gladiator status` | 首行含 `ready=true`、`spawn=runtime`；第三行含 `bots=1 ... pending=0 spawn=runtime` |
| 3 | golem 启动 | `spawn.mode: golem`，重启服务端 | 控制台 `spawn mode=golem · Golem 0.1.0 | 身体由 golem 在进程内生成，无需 bot-runtime 登录`；enable 行含 `spawn=golem` |
| 4 | golem spawn | `/gladiator spawn Goblin_A` | `[Gladiator] spawn 已受理 request=req-1 botId=glad-goblin_a state=READY mode=golem（golem 进程内身体，无需登录）`；控制台 `Bot ready: glad-goblin_a user=Goblin_A uuid=... source=golem` |
| 5 | golem 列表 | `/gladiator list` | `bots=1 pending=0`；下一行 `- glad-goblin_a name=Goblin_A uuid=... ready=true state=ALIVE`（`state` 打印的是 `UserState` 枚举名） |
| 6 | 身体存在 | `/golem info Goblin_A` | 显示该身体 id / 世界 / 坐标，与 `spawn.golem.*` 配置一致 |
| 7 | 重复 spawn | 再执行 `/gladiator spawn Goblin_A` | `[Gladiator] spawn 失败: Bot 已就绪，不能重复 spawn: Goblin_A` |
| 8 | 训练会话 | `/gladiator train Goblin_A` | `[Gladiator] train 已开始 bot=glad-goblin_a session=train-1`；`/gladiator status` 的 `training=1` |
| 9 | 回收 | `/golem remove Goblin_A`，等 ≤4s，再 `/gladiator list` | `bots=0 pending=0`；控制台 `Bot 退出: Goblin_A reason=golem-body-gone`；`training=0` |
| 10 | 再次 spawn | `/gladiator spawn Goblin_A` | 同 #4 成功（注册表已清干净，无残留） |
| 11 | 未装 golem 降级 | 移除 `golem-0.1.0.jar`，保持 `spawn.mode: golem`，重启 | 控制台 `spawn.mode=golem 但未检测到 Golem 插件，spawnBot 将失败（装好 Golem 后重启，或把 spawn.mode 改回 runtime）`；插件仍 enable |
| 12 | 降级下 spawn | `/gladiator spawn Goblin_B` | `[Gladiator] spawn 失败: spawnBot 不可用：spawn.mode=golem 但未检测到可用的 Golem 插件` |
| 13 | 降级下 list | `/gladiator list` | `bots=0 pending=0`，无异常堆栈 |
| 14 | 状态机（golem） | 对 `Goblin_A` 造成致死伤害（`/kill` 或攻击） | `ALIVE → DEAD → RESPAWNING → ALIVE` 状态变更事件；对局/训练会话按调用方逻辑处理 |

单测与门禁（无需起服）：

```powershell
.\gradlew.bat test --offline          # B1 后 Java 48 项全绿（含 golem/占位/委托 3 个测试类）
.\gradlew.bat verifyStandalone        # 零绑定门禁：只放行 top.cnolm.golem.api + softdepend Golem
```

---

## 9. 已知限制

1. **没有通道 B，不能训练**。`OBSERVATION` / `ACTION` / `RESET` / `IDLE` 全部不存在；
   `notice(bot, info)` 不下发帧，落到 Gladiator 本地队列（`MAX_PENDING_NOTICES` 上限内）；
   `STATE` 帧不下发，状态变化只走 `GladiatorStateChangedEvent`。
   需要动作级控制的调用方（训练循环）必须继续用 runtime + mineflayer。
2. **`PlayerInteractEvent` 的方块面不准确**：golem 不发送客户端 `PlayerActionPacket`，
   方块面/朝向信息无法还原，依赖 `getBlockFace()` 做放置朝向判定的逻辑在 golem 模式下不可信（对应 A1/A2 未关）。
3. **`PlayerArmSwingEvent` 不触发**：手臂摆动是纯客户端表现，服务端无人补发；依赖挥臂做统计/反作弊的监听器拿不到事件。
4. **`PlayerItemHeldEvent` 不触发**：快捷栏切换不产生该事件；手持物只能通过 `getItemInHand()` 主动读，
   不能靠事件驱动。
5. **`RESPAWN` 不发 `PlayerRespawnEvent`**：重生走 golem 自己的流程，
   依赖 `PlayerRespawnEvent` 做「重生后传送/发套件」的调用方要改为监听 `GladiatorStateChangedEvent`（回到 `ALIVE`）或在 `Ready` 后自行安排。
6. **`BLOCKS` 重叠即拒**：同一位置已有方块/实体占用时，身体生成会被拒绝，不会自动找空地。
   多 Bot 同点生成必须由调用方（或 `spawn.golem.location` + 间距）自行错开。
7. **出生无敌 60 tick（3 秒）**：身体刚生成时带出生保护，期间的伤害不生效。
   做「落地即开打」的计时逻辑要把这 3 秒算进去，否则会误判为「攻击无效」。
8. **`heldItem` 用 Material 名**：与 bot-runtime 的观测口径一致（例如 `DIAMOND_SWORD`、`GOLDEN_APPLE`），
   不是显示名、不带耐久后缀；跨两条链路比对数据时按 Material 名对齐。
9. **一次只能选一种来源**：`spawn.mode` 是启动期配置，`/gladiator reload` 不切换；
   同一进程内不能同时用 runtime 与 golem 发起 Bot（通道 A 服务端仍在跑，但 `spawnBot` 只有一个出口）。
10. **golem 侧 `despawn` 无事件**：回收依赖 2 秒对账，最坏有 2 个周期的可见延迟；
   对「立刻必须归零」的验收场景，等待 ≤4 秒再断言。

---

## 10. 相关文档

| 文档 | 关系 |
|---|---|
| [`api-contract.md`](api-contract.md) §B1 | 对外 API 在 golem 模式下的四条语义（权威） |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) §11 | `golem-api` 本地构建、`build.gradle` 声明、门禁白名单 |
| [`operations.md`](operations.md) | 端口、启动顺序、排障表 |
| [`manual-testing.md`](manual-testing.md) | 进服验收清单（含 golem 快速验证一条） |
| `docs/decisions.md` ADR-018 | 为什么 NMS 身体放在独立项目 golem |
