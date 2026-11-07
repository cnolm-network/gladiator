# Gladiator 对外 API 契约（集成唯一入口）

> 2026-11-07 定稿方向。**Gladiator 是独立的 Spigot 插件，不依赖任何竞技场插件。**
> 竞技场插件（当前为 Ultrapvp，未来可为任意插件）是**调用方**：它调用本 API 发起 Bot，
> 并自行完成传送、给予套件、地图、匹配、计分等"壳子"工作。
> Gladiator 只负责 Bot 的对战与训练过程（生命周期、动作、状态、观测、分析）。

---

## 1. 集成方向（不要搞反）

```
┌──────────────────────────────────────────────────┐
│ 竞技场插件（Ultrapvp / 未来的任意插件）              │
│  · 编译期依赖 gladiator-api（本插件 jar 或独立 api jar）│
│  · 调用 GladiatorAPI.spawnBot() / pushGame()        │
│  · 自己负责：传送 / 套件 / 地图 / 规则 / 结算 / 计分  │
└───────────────────────┬──────────────────────────┘
                        │ Java API 调用（同 JVM）
┌───────────────────────▼──────────────────────────┐
│ Gladiator（本仓库）                                │
│  · 不 import 任何竞技场插件的类                     │
│  · 只用标准 Bukkit 事件感知对局                     │
│  · 对外只暴露 top.cnolm.gladiator.api              │
└───────────────────────┬──────────────────────────┘
                        │ localhost WS
┌───────────────────────▼──────────────────────────┐
│ Layer 2 Node / Layer 3 Python / Layer 4 策略        │
└──────────────────────────────────────────────────┘
```

**换竞技场插件时：只改调用方的 API 调用代码，不改 Gladiator 源码。**
因此 Gladiator 源码中禁止出现 `top.cnolm.ultrapvp.*`、`Mode` 枚举的具体值、区域名、地图名等。

---

## 2. 职责切分（契约）

| 能力 | 竞技场插件（调用方） | Gladiator |
|---|---|---|
| 建图 / 选图 / 重置地图 | ✅ | ❌ |
| **传送 Bot 到对局位置** | ✅ | ❌ |
| **给予套件 / 装备 / 药水** | ✅ | ❌ |
| 匹配、排队、队伍、计分、结算 | ✅ | ❌ |
| 对局规则（何时淘汰、是否可重生） | ✅ 通过 `Mode` 接口声明 | 只读查询 |
| **发起 Bot 登录** | 调 `spawnBot()` | ✅ 真实 mineflayer 连接 |
| **把 Bot/玩家加入对局会话** | 调 `pushGame()` | ✅ 建立会话与状态机 |
| **Bot 移动 / 攻击 / 放置 / 拆除 / 使用物品 / 跳跃** | ❌ | ✅ |
| 读聊天栏、感知规则拒绝（WG PVP deny） | ❌ | ✅ |
| 状态 `[Alive, Dead, Respawning, Eliminated]` | 只读 + `eliminate()` | ✅ 唯一写入点 |
| 观测、episode、训练数据 | ❌ | ✅ |
| 空闲期互联网检索 | ❌ | ✅ 由 Layer 4 完成 |
| **环境准备 / init（权限组、反作弊豁免、区域放行、白名单）** | ✅ **调用方自己的 init 阶段** | ❌ 完全不感知，见 §8 |

**关键**：Gladiator 不碰 `Player` 的传送、背包、GameMode；调用方在 `BotReadyEvent` 之后自己做。

---

## 3. API 包（`top.cnolm.gladiator.api`）

> 以下为对外契约签名。**G-002 已于 2026-11-07 落地为 `src/main/java/top/cnolm/gladiator/api/` 下的 Java 代码**
> （接口、枚举、不可变值类型）与 G-007 的 5 个事件；实现时只做追加，不得变更已有签名。
> **G-008 已提供实现类** `core/GladiatorServiceImpl` 并在 `onEnable()` 注册到 `ServicesManager`，
> 因此插件 enable 后 `GladiatorProvider.get()` 返回非 null；未 enable 时仍返回 null，调用方必须判空。
> `spawnBot()` 目前显式抛 `UnsupportedOperationException`（真实登录属 G-107），其余方法已可用。

### 3.1 值类型

```java
public enum UserState { ALIVE, DEAD, RESPAWNING, ELIMINATED }

public enum InformationKind {
    RULE_DENIED, CHAT, MATCH_EVENT, IDLE, CUSTOM
}

public final class Information {
    private final InformationKind kind;
    private final long timestamp;
    private final Map<String, Object> payload;   // 只允许原语/字符串/枚举
    public static Information of(InformationKind kind, Map<String, Object> payload);
    public InformationKind getKind();
    public long getTimestamp();
    public Map<String, Object> getPayload();
}
```

### 3.2 参与者

```java
public interface User {
    UUID getUniqueId();
    String getName();
    UserState getState();
    boolean isBot();
    /** 真人返回 Bukkit Player；Bot 返回其真实登录得到的 Player；离线返回 null */
    Player asPlayer();
}

public interface Bot extends User {
    String getBotId();
    boolean isReady();
}

public final class Users {
    /** 把 Bukkit Player 包装成 User，供 pushGame 混合传参 */
    public static User of(Player player);
}
```

> Bot 是**真实协议登录**得到的 `CraftPlayer`，所以 `asPlayer()` 非 null，调用方可以直接
> `player.teleport(...)` / `player.getInventory().setArmorContents(...)`，无需任何 Gladiator 私有 API。

### 3.3 Mode（由调用方实现，Gladiator 不实现规则）

```java
public interface Mode {
    /** 稳定标识，例如 "nodebuff" / "sumo"，仅用于日志与观测，不参与规则判定 */
    String getId();
    /** 规则查询：该 User 死亡后是否允许重生（Gladiator 只读） */
    boolean allowsRespawn(User user);
    /** 规则查询：会话超时时间（毫秒），0 表示不限时 */
    long getTimeLimitMs();
}
```

`Mode` 不持有状态、不写 `UserState`。判定"不再允许重生"时，调用方通过
`GladiatorAPI.eliminate(User)` 打标记；状态迁移仍由 Gladiator 执行。

### 3.4 主 API

```java
public interface GladiatorAPI {
    /** 语义化版本，调用方可据此做兼容判断 */
    String getApiVersion();

    /** 插件是否已就绪（服务端 enable 完成、WS 已监听） */
    boolean isReady();

    /** 发起一个 Bot：立即返回票据，真实登录异步完成，完成后发 BotReadyEvent */
    BotRequest spawnBot(BotSpec spec);

    /** 把一组 Bot/真人塞进一个对局会话；调用方须已完成传送与套件 */
    GameSession pushGame(Mode mode, User... users);

    /** 把某个 Bot 放入无限制训练场（无对局规则、无终止 episode） */
    TrainingSession pushPlace(Bot bot);

    /** 向 Bot 推送结构化信息（规则拒绝、聊天摘要、对局事件、空闲信号） */
    void notice(Bot bot, Information information);

    /** 分析当前/指定会话的训练数据，返回不可变快照 */
    Analysis analyze(AnalysisQuery query);

    /** 淘汰标记：Mode/调用方唯一可写的状态入口 */
    void eliminate(User user);

    UserState stateOf(User user);
}
```

### 3.5 辅助类型

```java
public final class BotSpec {
    private final String username;      // 合法 MC 用户名，建议统一前缀
    private final String group;         // 不透明元数据：调用方自行解释（权限组名等），Gladiator 不应用权限
    private final Map<String,String> metadata;  // 调用方自定义，不参与规则
}

public final class BotRequest {
    private final String requestId;
    private final String botId;
    /** 登录完成后由 BotReadyEvent 携带；未就绪时返回 null */
    public Bot getBot();
    public State getState();            // PENDING / READY / FAILED
    public String getFailureReason();
}

public final class GameSession {
    private final String sessionId;
    private final Mode mode;
    private final List<User> users;
    private final long startedAt;
}

public final class TrainingSession { private final String sessionId; private final Bot bot; }

public final class AnalysisQuery  { /* sessionId / windowSeconds / 可选过滤 */ }
public final class Analysis      { /* 存活时长、伤害、方块操作、路径里程、规则拒绝次数 */ }
```

### 3.6 获取 API（两种等价方式）

```java
// 方式 A：静态门面（推荐，简单）
GladiatorAPI api = GladiatorProvider.get();

// 方式 B：Bukkit ServicesManager（避免硬编译依赖时可反射/服务查找）
RegisteredServiceProvider<GladiatorAPI> p =
    Bukkit.getServicesManager().getRegistration(GladiatorAPI.class);
GladiatorAPI api = p == null ? null : p.getProvider();
```

Gladiator 在 `onEnable()` 中向 `ServicesManager` 注册实现；`onDisable()` 注销。
`GladiatorProvider.get()` 在未 enable 时返回 `null`，调用方必须判空。

---

## 4. 事件（Gladiator 发出，调用方监听）

全部是标准 `org.bukkit.event.Event` 子类，不带任何竞技场插件类型。

| 事件 | 触发时机 | 携带 | 调用方典型动作 |
|---|---|---|---|
| `GladiatorBotReadyEvent` | Bot 真实登录、spawn 完成、`BotHandle` 注册后 | `getBot()` / `getPlayer()` | **传送到对局位置、给予套件**、设置权限组 |
| `GladiatorBotFailedEvent` | 登录失败 / 超时 / 被踢 | `requestId` / `reason` | 重试或回退 |
| `GladiatorStateChangedEvent` | `UserState` 迁移 | `user` / `from` / `to` | 结算、播报、决定是否 `eliminate()` |
| `GladiatorSessionEndedEvent` | 对局会话终止 | `sessionId` / `reason` | 释放地图、写战绩 |
| `GladiatorBotRemovedEvent` | Bot 退出 / 被回收 | `bot` | 清理调用方侧状态 |

**顺序保证**：`GladiatorBotReadyEvent` 一定在 `pushGame` 可用之前发出；调用方应在该事件里完成传送与套件，然后才调 `pushGame`。

---

## 5. Gladiator 内部如何感知对局（插件无关的关键）

只监听**标准 Bukkit 事件**，不监听任何插件自定义事件：

| 标准事件 | 用途 |
|---|---|
| `PlayerDeathEvent` | `ALIVE → DEAD` |
| `PlayerRespawnEvent` | `DEAD → RESPAWNING → ALIVE`（受 `Mode.allowsRespawn` 约束） |
| `EntityDamageByEntityEvent` / `EntityDamageEvent` | 伤害指标、PVP deny 观测 |
| `BlockPlaceEvent` / `BlockBreakEvent` | 方块操作指标（**只观察，不取消**） |
| `PlayerQuitEvent` | 会话回收 |
| `PlayerChatTabCompleteEvent` / `AsyncPlayerChatEvent` | 聊天观测 |
| WorldGuard `DisallowedPVPEvent`（可选桥接） | 规则拒绝 → `RestrictionCache` → 停止攻击并离开 |

`ELIMINATED` 只由 `eliminate(User)` 触发。因此**换成任何竞技场插件，只要它用标准 Bukkit 事件驱动死亡/重生，Gladiator 无需改动**。

---

## 6. 线程与调用约定

- `GladiatorAPI` 的所有方法**必须在服务端主线程调用**；从异步线程调用应抛 `IllegalStateException`。
- `spawnBot()` 立即返回，不阻塞；登录完成通过 `GladiatorBotReadyEvent`（主线程）通知。
- `analyze()` 返回不可变快照；数据采样异步，不阻塞主线程。
- `notice()` 只投递到 WS 通道，不做 Bukkit 操作。
- 所有 API 类型只含 Java 8 原语、字符串、枚举、集合；**不得暴露 mineflayer/session/NMS/Bukkit 内部对象**。

---

## 7. 兼容性与版本策略

- `getApiVersion()` 遵循语义化版本；**只增不改**：新增方法用 default 实现，删除方法需大版本。
- 调用方可以只依赖一个独立的 `gladiator-api.jar`（后续从主 jar 拆出），避免拉入整个插件。
- 竞技场插件侧的最小改动示例（伪代码，仅示意）：

```java
// 竞技场插件：发起 bot
BotRequest req = GladiatorProvider.get().spawnBot(
        BotSpec.of("Glad_Bot_1", "gladiator-bot"));

@EventHandler
public void onBotReady(GladiatorBotReadyEvent e) {
    // 壳子工作全在调用方：传送 + 套件
    e.getPlayer().teleport(arenaSpawn);
    Kit.apply(e.getPlayer(), mode);
    // 就绪后才纳入对局
    GladiatorProvider.get().pushGame(mode, Users.of(human), e.getBot());
}
```

换成另一套竞技场插件时，只需把上面 3 行换成新插件的传送/套件调用，**Gladiator 源码零改动**。

---

## 8. 调用方 init 阶段（环境准备，含反作弊）

> **约定**：并非所有服务器都使用同一种反作弊插件。
> 反作弊的"完善"（检测、豁免、权限配置）**属于竞技场插件自己的 init 阶段**，不是 Gladiator 的职责，
> 也不允许写进 Gladiator 的代码、配置或默认值。

### 8.1 职责归属

| 事项 | 谁做 | 说明 |
|---|---|---|
| 探测本服装了哪些反作弊插件 | **调用方 init** | 例如 `Bukkit.getPluginManager().getPlugin("Intave") != null`；没有就跳过 |
| 为 Bot 账号/组配置豁免 | **调用方 init** | 权限节点由该反作弊自己的文档定义；不同反作弊不同 |
| 创建权限组 / 设置 Bot 账号前缀 | **调用方 init** | 与 Gladiator 无关 |
| WorldGuard 区域放行、白名单、计分板权限 | **调用方 init** | 调用方最清楚自己的地图与规则 |
| 检查 Gladiator 是否就绪、API 版本是否兼容 | **调用方 init** | `GladiatorProvider.get().isReady()` / `getApiVersion()` |
| 上述任何一项 | **Gladiator** | ❌ 不做、不检测、不设置、不硬编码 |

Gladiator 的保证只有两条：`onEnable()` 后 `isReady() == true`；`spawnBot()` 之前调用方可以查询这两项。
`BotSpec.group` 是**不透明元数据**，Gladiator 只做透传/记录，不会据此设置任何权限。

### 8.2 调用方 init 参考形态（伪代码，不是 Gladiator 代码）

```java
// 竞技场插件 onEnable() / 自定义 init 阶段
public void initArenaEnvironment() {
    GladiatorAPI api = GladiatorProvider.get();
    if (api == null || !api.isReady()) throw new IllegalStateException("Gladiator not ready");
    checkApiCompatibility(api.getApiVersion());

    // 环境相关，全部由调用方按本服实际情况决定
    if (Bukkit.getPluginManager().getPlugin("Intave") != null) {
        // 仅在使用该反作弊的服务器执行；节点名以该反作弊官方文档为准
        // 例：LuckPerms 给 Bot 组授予 intave.bypass
        grantBotGroupPermission("gladiator-bot", "intave.bypass");
    }
    // 其他反作弊 / 无反作弊：各自处理或直接跳过
}
```

> `intave.bypass` 只是**一个具体环境下的示例节点**（官方 White / Full tolerance）。
> 它不是 Gladiator 的常量、配置项或依赖；没有 Intave 的服务器不应出现任何相关代码路径。

### 8.3 对 Gladiator 源码的硬约束

1. 不出现 `Intave` / `intave.bypass` / 任何反作弊插件名或权限节点（文档中的示例说明除外）。
2. 不在 `config.yml` 中提供反作弊开关或豁免配置。
3. 不在 `spawnBot` / `BotReadyEvent` 中修改权限、组、白名单。
4. 相应验收项：`G-009` 的零依赖门禁扩展为"零环境绑定检查"（源码 grep 反作弊名/权限节点）。
5. `A7` 不再是 Gladiator 的验收项；改为由调用方在各自环境自行验证豁免是否生效。

---

## 9. 与具体竞技场插件的关系

某个具体竞技场插件内部的状态枚举、匹配器队列、地图保护监听器等实现细节，
**仅供理解"调用方可能长什么样"**，不是 Gladiator 的依赖或约束。
特别是：Gladiator 不需要任何插件的切服方法、子服概念、`Mode` 枚举的具体值——
这些都是调用方自己的事，换插件时 Gladiator 源码零改动。

---

## 10. G-002 实现状态（2026-11-07）

**已落地的文件**（`src/main/java/top/cnolm/gladiator/api/`）：

| 类别 | 文件 |
|---|---|
| 枚举 | `UserState`、`InformationKind` |
| 接口 | `User`、`Bot`、`Mode`、`GladiatorAPI` |
| 不可变值类型 | `Information`、`BotSpec`、`BotRequest`、`GameSession`、`TrainingSession`、`AnalysisQuery`、`Analysis`(+`BotMetrics`) |
| 门面/辅助 | `Users`、`GladiatorProvider`、`UserStateRegistry` |
| 事件（G-007） | `event/GladiatorBotReadyEvent`、`GladiatorBotFailedEvent`、`GladiatorStateChangedEvent`、`GladiatorSessionEndedEvent`、`GladiatorBotRemovedEvent` |

**相对本契约的追加项（纯增量，不破坏签名）**：
- `UserStateRegistry`：`Users.of(Player).getState()` 的查询入口，由插件本体在 onEnable 安装 `Lookup`；
  api 包不 import core。未安装时在线玩家默认 `ALIVE`。
- `GladiatorProvider.install/uninstall`：仅供插件本体使用；`get()` 优先读静态缓存，
  其次走 Bukkit `ServicesManager`。
- 各值类型提供 `of(...)` 静态工厂；`BotRequest` 用 `pending/ready/failed` 产生新实例（不可变）。
- `Information.empty(kind)`、`AnalysisQuery.all()` 为便捷构造。

### G-008 追加：服务注册与最小实现（2026-11-07 14:35）

| 项 | 落地位置 | 状态 |
|---|---|---|
| `GladiatorAPI` 实现类 | `core/GladiatorServiceImpl.java` | ✅ 实现 `getApiVersion` / `isReady` / `pushGame` / `pushPlace` / `notice` / `analyze` / `eliminate` / `stateOf` |
| 指标累加器 | `core/SessionCounters.java` | ✅ 供 M2/M3 事件写入，`analyze()` 产出不可变 `BotMetrics` |
| `ServicesManager` 注册 / 注销 | `GladiatorMain.onEnable/onDisable` | ✅ `ServicePriority.Normal`；`GladiatorProvider.install/uninstall` 同步 |
| `UserStateRegistry.Lookup` | `GladiatorServiceImpl` | ✅ 与 `stateOf()` 共用一份状态，无双状态源 |
| 状态迁移 | `GladiatorServiceImpl.transition()` | ✅ 唯一写入点；变化时才发 `GladiatorStateChangedEvent` |
| `/gladiator status` | `command/GladiatorCommand.java` | ✅ 显示 API 版本、ready、bots/games/training 计数 |

**实现约定（不改变契约签名）**：
- 所有 `GladiatorAPI` 方法要求服务端主线程，异步调用抛 `IllegalStateException`；
  `getApiVersion()` 返回 `1.0.0`，`isReady()` = 已注册 ∧ 插件仍 enable。
- `pushGame` / `pushPlace` 不允许同一 `User` 重复加入活动会话（抛 `IllegalStateException`）。
- `notice()` 在 WS 建立前写入每 Bot 上限 256 条的积压队列，超限丢最旧并计入 `actionsDropped`；
  `drainNotices(botId)` 供 G-101 链路取走。
- `endSession(sessionId, reason)` / `registerBot` / `unregisterBot` 是 M1 内部钩子，
  不属于 `GladiatorAPI` 接口，调用方不得依赖。

**仍未实现**（M1）：`spawnBot()` 真实登录与 `GladiatorBotReadyEvent` 派发（G-107）、
`BotHandle` / WS 通道 / 心跳（G-101~G-106）。`spawnBot()` 现阶段直接抛
`UnsupportedOperationException`，不返回假的 PENDING 票据、不伪造 Player。

**验证**：IntelliJ MCP `execute_run_configuration("cbot-gladiator [build]")` → BUILD SUCCESSFUL；
`build_project(rebuild=true)` → `isSuccess:true, problems:[]`；`get_file_problems` 仅风格类 WARNING；
Java 8 toolchain 编译；产物 bytecode 52。
