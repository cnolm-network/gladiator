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

**方块与 PVP 放行（G-208）**：WorldGuard / 领地插件 / 反作弊常会拦截 Bot 的放置、破坏与攻击。这些拦截属于**调用方环境准备**：
请在调用方 `init`（或 Ready 事件）里让 Bot 通过区域/权限/白名单的放行检查，例如把 Bot 加入区域成员、
给予对应权限节点或调整服务端规则。**Gladiator 不注册任何方块/PVP 放行或取消监听器**，也不会替你修改外部插件的配置。
如果 Bot 的动作被外部插件拒绝，Gladiator 只能通过 STAT 观测/`ACTION_RESULT` 体现，具体放行必须由调用方完成。

---

## 3. API 包（`top.cnolm.gladiator.api`）

> 以下为对外契约签名。**G-002 已于 2026-11-07 落地为 `src/main/java/top/cnolm/gladiator/api/` 下的 Java 代码**
> （接口、枚举、不可变值类型）与 G-007 的 5 个事件；实现时只做追加，不得变更已有签名。
> **G-008 已提供实现类** `core/GladiatorServiceImpl` 并在 `onEnable()` 注册到 `ServicesManager`，
> 因此插件 enable 后 `GladiatorProvider.get()` 返回非 null；未 enable 时仍返回 null，调用方必须判空。
> `spawnBot()` 已在 G-107 接线：立即返回 `BotRequest(PENDING)`，真实登录由外部 mineflayer
> 运行时完成，成功后在主线程派发 `GladiatorBotReadyEvent`，失败派发 `GladiatorBotFailedEvent`；
> 详见 §10 与 `docs/protocol.md` §2.10。

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

    /** api 1.1.0：坐标处的 PVP 是否允许（等价于 queryPvp(...).isAllowed()） */
    boolean isPvpAllowed(Location location, User user);

    /** api 1.1.0：带来源与区域信息的判定；永不返回 null */
    PvpVerdict queryPvp(Location location, User user);
}
```

**规则查询语义（G-204）**：`queryPvp` 只读区域配置，**绝不修改区域、绝不取消任何事件**。
`PvpVerdict.source` 取值 `FLAG`（区域里显式配置了 PVP flag）/ `GLOBAL`（无显式 flag，按全局）/ `DEGRADED`（未装 WorldGuard，一律 allow）/ `UNKNOWN`（坐标或世界不可用）。
`regionIds` 为命中区域 id（可能多个），`box` 为这些区域的并集包围盒（可能为 null）。未装 WorldGuard 时首次查询会打印一次降级说明（G-205）。

```java
public final class PvpVerdict {
    public boolean isAllowed();
    public List<String> getRegionIds();
    public Source getSource();   // FLAG / GLOBAL / DEGRADED / UNKNOWN
    public Box getBox();         // minX/minY/minZ/maxX/maxY/maxZ
}
```

### 3.5 辅助类型

> **单 Bot 手感（G-507）**：`BotProfile` 只描述运行时的手感与模型标识，不参与规则判定，也不改变 Gladiator 的状态机。
> 约束：`cps ∈ (0,100]`、`cpsJitter ∈ [0,cps)`、`reactionMs ≥ 0`、`reach ∈ (0,6]`；非法值在构造时抛异常，运行时会逐字段回落默认值。
> `/gladiator spawn <username> cps=17 jitter=3 reaction=180 rjitter=30 reach=4.2 model=aggressive` 是等价的管理命令写法。


```java
public final class BotSpec {
    private final String username;      // 合法 MC 用户名，建议统一前缀
    private final String group;         // 不透明元数据：调用方自行解释（权限组名等），Gladiator 不应用权限
    private final Map<String,String> metadata;  // 调用方自定义，不参与规则
    private final BotProfile profile;   // G-507：该 Bot 的手感配置，默认 BotProfile.defaults()
}

public final class BotProfile {
    public static final double DEFAULT_CPS = 15.0;              // 每秒点击数
    public static final double DEFAULT_CPS_JITTER = 2.0;        // 点击间隔抖动
    public static final double DEFAULT_REACTION_MS = 220.0;     // 反应延时
    public static final double DEFAULT_REACTION_JITTER_MS = 40.0;
    public static final double DEFAULT_REACH = 3.99;            // 原版攻击距离（格）
    public static final String DEFAULT_MODEL = "default";       // 训练模型标识

    public static BotProfile defaults();
    public static BotProfile of(double cps, double cpsJitter,
                                double reactionMs, double reactionJitterMs,
                                double reach, String model);
    public static Builder builder();
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
4. 相应验收项：`G-009` / `G-108` 的零依赖门禁扩展为"零环境绑定检查"，已落地为 Gradle 任务
   `verifyStandalone`（`gradlew check` 会先跑它，见 `build.gradle`）。
   门禁**故意不用黑名单罗列第三方插件名**（那等于把名字写进本仓库），而是用白名单：
   - `src/main/java` 的 `import` 只允许 `java` / `javax` / `org.bukkit` / `org.spigotmc` /
     `com.sk89q`（WorldGuard API）/ `com.google`（Gson）/ `top.cnolm.gladiator`；
   - `plugin.yml` 的 `depend` / `softdepend` 只允许 `WorldGuard`；
   - `plugin.yml` 的 `main` 必须位于 `top.cnolm.gladiator` 包内。
   任何对竞技场插件、反作弊插件或其它服务端插件的新依赖都会让 `check` 失败。
5. `A7` 不再是 Gladiator 的验收项；改为由调用方在各自环境自行验证豁免是否生效。
6. 门禁只能证明"没有**静态**绑定"；运行期是否真的不需要某个插件，要靠
   `plugin.yml` 依赖白名单 + 真实无插件服务端启动验证（`G-003` 的服务器矩阵）。

### 8.4 调用方统计隔离（G-505）

Gladiator 不提供全局战绩/榜单，但 Bot 同样是 `User`，很容易被调用方已有的统计逻辑一把捞进去。
调用方在写统计、排行榜、赛季积分前**必须**判 `user.isBot()` 并跳过，否则训练数据会污染真人榜单。

示例见 `src/main/java/top/cnolm/gladiator/example/StatsBoard.java`（`recordWin(User)` 对 Bot 直接返回
`false`），单测 `StatsBoardTest` 覆盖"Bot 不计入 / null 不计入 / 真人正常排名"。


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

### G-104 / G-105 / G-107 追加：BotHandle、会话登记与 spawnBot（2026-11-07）

| 项 | 落地位置 | 状态 |
|---|---|---|
| `Bot` 代理 | `core/BotHandle.java` | ✅ `implements Bot`，Player 操作全部委托真实 `CraftPlayer`；未绑定时 UUID 退化为离线 UUID（与 `OfflinePlayer:` 算法一致），`asPlayer()` 为 null |
| Bot 三向索引 | `core/BotRegistry.java` | ✅ botId / UUID / username；`botIdFor(username) = "glad-" + lowercase`，两端统一 |
| 会话登记 | `core/GameRegistry.java` | ✅ 只登记 sessionId/参与者/开始时间；不传送、不发套件、不判胜负（G-105） |
| spawn 编排 | `core/BotSpawnService.java` | ✅ pending → HELLO 下发 SPAWN → PlayerJoin 绑定 → 主线程 `GladiatorBotReadyEvent`；超时/SPAWN_FAILED/链路断开 → `GladiatorBotFailedEvent` |
| `spawnBot()` | `GladiatorServiceImpl` | ✅ 主线程校验后委托 spawn 服务并立即返回 `BotRequest`；未接线时抛 `IllegalStateException` |
| WELCOME.playerUuid | `bridge/LinkServer` + `LinkHandler.playerUuid` | ✅ 已绑定时填真实 UUID，未生成时为空串（G-103 收尾） |
| 事件时序 | `BotSpawnService.bind()` | ✅ Ready 在玩家 join 的 MONITOR 阶段、主线程派发；此时 `bot.asPlayer().isOnline()` 为 true |

**行为约定**：
- `pushPlace(Bot)` 要求 `bot.isReady()`，否则抛 `IllegalStateException`（必须等 Ready 事件）。
- 同一用户名重复 `spawnBot` 时：已就绪或正在发起均抛 `IllegalStateException`。
- 链路断开不会回收已绑定的 BotHandle（mineflayer 进程可能仍在）；进程真死时 MC 断线，
  `PlayerQuitEvent` 负责 `unregisterBot` 与 `BotHandle.unbind()`。
- 超时时间由 `bridge.spawn-timeout-seconds`（默认 30）控制，失败 reason 以 `TIMEOUT:` 开头。
- 运行时断线重连由 `bot-runtime/src/connection.js` 指数退避（1s→2s→…→30s，抖动 ±20%），
  重连后必须重新 HELLO。

**仍未实现（M1 剩余）**：动作执行/观测面（G-108 之后的 M2）、`/gladiator list|game|analyze`（M3）、
WorldGuard 规则桥实际判定（M2）。`spawnBot()` 不再抛 `UnsupportedOperationException`。

**验证**：`.\gradlew.bat test --offline` → 20 tests PASSED（LinkServer 9 + BotRegistry 6 + GameRegistry 4 + 1 汇总）；
`node --test`（bot-runtime）→ 12 tests PASSED；`.\gradlew.bat clean build --offline` → BUILD SUCCESSFUL；
Java 8 toolchain 编译；产物 bytecode 52。

### M2 / M3 追加：规则感知、状态机与训练编排（2026-11-07，api 1.1.0）

| 项 | 落地位置 | 状态 |
|---|---|---|
| PVP 判定值类型 | `api/PvpVerdict.java` | ✅ `allowed` / `regionIds` / `source{FLAG,GLOBAL,DEGRADED,UNKNOWN}` / `box`（不可变） |
| 主动规则查询 | `GladiatorAPI#isPvpAllowed` / `#queryPvp` | ✅ api 1.1.0 纯增量；未装 WG 时 `DEGRADED` 且日志明示一次 |
| WorldGuard 桥接 | `core/guard/WorldGuardBridge.java` | ✅ 只读；`getApplicableRegions(Location)` 因 `worldedit-core` 不在编译类路径而走反射，区域 id / PVP flag / 包围盒均反射读取 |
| 无 WG 降级 | `core/guard/NoopGuardBridge.java` | ✅ `describe()=noop`，`isAvailable()=false` |
| 被动规则拒绝 | `core/guard/WorldGuardPvpListener.java` | ✅ MONITOR 优先级，只观察；**不调用 `setCancelled`/`setCancelled(false)`**（ADR-015、AGENTS.md） |
| 状态机 | `core/UserStateMachine.java` + `core/PlayerStateListener.java` | ✅ UUID→UserState 唯一存储；`PlayerDeathEvent`→DEAD、`PlayerRespawnEvent`→RESPAWNING→ALIVE、`eliminate()`/`Mode.allowsRespawn=false`→ELIMINATED |
| 状态事件 | `GladiatorServiceImpl.transition(user,to,reason)` | ✅ 唯一写入点；先下发 SPAWN 通道 STATE 帧，再派发 `GladiatorStateChangedEvent`（from==to 不发） |
| 指标快照 | `core/AnalyzeService.java` | ✅ 组合存活时长与 `SessionCounters`；计数器自带锁，可在任意线程采样（当前 `analyze()` 主线程同步调用） |
| 管理命令 | `command/GladiatorCommand.java` | ✅ `list` / `game` / `analyze [session] [window]` / `reload` / `spawn <name>` / `train <name>` / `pvp <x> <y> <z>` / `spectate [bot]` / `unspectate` / `status` |
| 参考实现 | `example/DeathmatchMode.java`、`example/MockArenaCaller.java` | ✅ 只 import Bukkit + `api` 包；演示 Ready 里传送后 `pushGame`，不含任何竞技场插件类型 |

**行为约定（M2/M3）**：
- 方块放置/破坏、伤害、聊天等事件只读：Gladiator **不注册任何放行/取消逻辑**；服务器保护（WorldGuard、领地插件等）
  的放行由调用方在自己的 `init` 里完成（G-208）。请把「放行 Bot 的方块与 PVP」视为调用方环境准备的一部分。
- 伤害统计会忽略 `<= 0`、非有限值或 `> 1000` 的伤害（虚空/`/kill` 之类哨兵值会让指标失真）。
- `analyze()` 的 `aliveSeconds`：有活动会话取会话开始时间，否则退化为 Bot Ready 时间；`windowSeconds` 会裁剪上限。
- 取消伤害/方块事件的插件不受影响：Gladiator 全部监听器使用 MONITOR 且从不取消。

**仍未实现（M4 起）**：Gymnasium 环境（`GladiatorEnv.reset/step/close`）、RESET 指令的 Layer 3 侧闭环、
服务器矩阵与观测 schema 文档收尾（G-003 / G-004）。

**验证**：`.\gradlew.bat test --offline` → 33 PASSED（LinkServer 9 / BotRegistry 7 / GameRegistry 4 /
PvpVerdict 5 / UserStateMachine 5 / AnalyzeService 3）；`node --test`（bot-runtime）→ 45 PASSED；
`.\gradlew.bat clean build --offline` → BUILD SUCCESSFUL；真实 Paper 1.8.8 + mineflayer + 通道 B 端到端联调通过。

### B1 追加：`spawnBot` 的两种来源（runtime / golem，2026-11-11）

对外 API **未变**（还是 `spawnBot(BotSpec) → BotRequest` + `GladiatorBotReadyEvent`），
变化只在服务端内部：Bot 由谁创建，由 `config.yml` 的 `spawn.mode` 决定。

| 模式 | Bot 来自 | 登录 | 依赖 |
|---|---|---|---|
| `runtime`（默认） | 外部 mineflayer 运行时经通道 A（WS `127.0.0.1:8765`）真实登录 | 是 | `bot-runtime/` |
| `golem` | Golem 插件在进程内生成的 NMS 身体（`GolemService.spawn`） | 否 | 服务器装 Golem（`softdepend`） |

调用方只需要知道四条语义：

1. **票据与事件不变**：golem 模式通常在同一 tick 内返回 `READY`；若身体晚一 tick 就绪，
   仍是 `PENDING` + 后续 `GladiatorBotReadyEvent`。超时派发 `GladiatorBotFailedEvent`
   （reason 前缀 `GOLEM_TIMEOUT:`），`spawn-timeout-seconds` 复用 `bridge.*` 的配置值。
2. **没有通道 B**：golem 模式没有 WebSocket 会话，
   - `notice(bot, info)` 不下发帧，落到 Gladiator 本地队列（`MAX_PENDING_NOTICES` 上限内），
   - `STATE` 帧不存在，状态变化只走 `GladiatorStateChangedEvent`。
   需要动作级控制的调用方（训练循环）必须走 golem 的 `BotBody` 原语，而不是 `notice`。
3. **`User.asPlayer()` 可用**：golem 身体不是登录玩家，`Player#isOnline()` 在 1.8.8 恒为 false
   （实现是 `CraftServer.getPlayer(uuid) != null`），因此 `BotHandle` 的可用判定放宽为
   `isOnline() || isValid()`。调用方可以正常 `teleport` / 读写背包。
4. **回收语义**：golem 侧 `despawn` 不发 Bukkit 事件，Gladiator 用 2 秒一次的对账任务
   （`GolemBodySpawner#reconcile`）清理注册表，并派发 `GladiatorBotRemovedEvent(reason="golem-body-gone")`；
   未结束的训练/对局会话一并 `endSession`。身体死亡但被 `DeathPolicy.KEEP` 保留时不回收，
   交给状态机走 `DEAD` / `RESPAWNING`。

调用方**不要**根据 `spawn.mode` 分支写代码：两种模式下 `Bot` / `User` 语义一致，
差异（登录、通道 B、动作原语）都属于服务端配置与训练侧实现细节。
