# 服务端内假玩家（Server-side Fake Player）调研报告

> 调研日期：2026-11-10
> 目的：为「把训练好的 `.gmod` 策略（19 维观测 → 11 个离散动作的 MLP）变成服务端内陪练 Bot」这一目标，
> 摸清业界在 **Minecraft 1.8.9 / Spigot** 上实现「服务端内玩家实体 + 插件驱动其移动与攻击」的既有做法，
> 并给出「Bot 协议/实体核心」独立项目的接口与形态建议。
>
> 方法：**第一手证据优先**。除公开仓库源码外，本文所有 1.8.9 侧的方法签名与调用链，均以本机
> `D:\ServerUltrapvp\spigot-1.8.8.jar`（v1_8_R3，协议 47）的字节码反编译核对（`javap -p -c`）为准，
> 不使用记忆中的 API，也不采信未标注出处的二手描述。凡未能直接核对的，正文明确标「未验证」。

---

## 1. 结论摘要

| # | 结论 | 依据强度 |
|---|---|---|
| 1 | 业界的标准做法是**「真实体 + 假连接」**：自建 `EntityPlayer` 子类，塞一个 `PlayerConnection` 子类进去并把 `sendPacket` 置空，网络层用一个无 `Channel` 的 `NetworkManager`。Citizens 是这一路线的成熟实现。 | 高（源码） |
| 2 | **连接不能省**。`EntityPlayer` 有 20+ 个方法直接解引用 `playerConnection`（发消息、药水效果、开背包、进战斗……），完全不建连接会在日常 tick 里 NPE。省掉的不是连接对象，而是连接背后的真实 socket。 | 高（字节码） |
| 3 | **注册位置决定语义**：只进 `World.players` → 实体可见、可被攻击，但**不**出现在 `Bukkit.getOnlinePlayers()`（Citizens 默认行为）；同时进 `PlayerList.players` → 变成"真在线玩家"，`getOnlinePlayers()`、`/list`、在线人数、统计插件全部会计入。 | 高（字节码 + 源码） |
| 4 | **1.8.9 没有 1.9 的攻击冷却**。`EntityHuman.attack(Entity)` 里不存在任何冷却字段参与伤害计算；1.8 的 PvP 是纯 CPS 驱动。任务书里「attackStrengthTicker（1.8.9 攻击冷却）」这一前提需要修正。 | 高（字节码） |
| 5 | **`PlayerMoveEvent` 不会自动触发**。该事件只在 `PlayerConnection.a(PacketPlayInFlying)`（入站移动包处理）里构造；服务端直接改坐标不走那条路径。需要自建桥接或显式 `callEvent`。 | 高（字节码） |
| 6 | 「Ultibot」在 1.8.9 语境下**未找到**对应项目；最可能指的是 UltiTools 的假人模块 `UltiKits/UltiBot`（已归档，仅 Paper 1.20.1–1.21.4）。1.8.9 可用性最强的公开参考是 `MonkeyMoon104/UltimateBOT`（含 1.8 NMS 桥）。 | 中（文档 + 仓库） |
| 7 | 建议 Bot 核心做成**独立 Gradle 多模块项目**，对外只暴露 `BotBody` 接口（spawn/despawn/tick/attack/move/look/use/snapshot），Gladiator 以 `compileOnly` 依赖接口模块，NMS 实现模块用 `ServiceLoader`/反射在运行时装配。 | 建议 |

一句话结论：**技术可行，成本主要在「1.8.9 NMS 的空连接与注册策略」和「动作语义与训练侧对齐」两处，
不在实体创建本身**。实体创建只有十几行代码，难的是让它「像玩家」而不是「像僵尸」。

---

## 2. Citizens（Citizens2）如何实现 NPC / 假玩家

### 2.1 版本谱系与本文引用的分支

Citizens2 仓库（`https://github.com/CitizensDev/Citizens2`）只保留了 7 个分支，其中与 1.8 相关的是 **`1.8`** 分支
（最后一次提交 `7739e5d`）。必须强调一点：**该分支的 NMS 包是 `net.minecraft.server.v1_8_R1`，即 MC 1.8.0**，
不是 1.8.8/1.8.9 的 `v1_8_R3`。两者 API 形状高度相似但字段名不同，直接照抄会在 1.8.9 上翻车。

`master` 分支已演进到 `v26_1_R1` 等新版本模块；本次检索 master 的完整树时数据被截断，
**未能确认 master 是否仍保留 `v1_8_R3` 模块**（标「未验证」）。Citizens 官方 wiki 声称「兼容 1.8.8 起的
所有主要版本」，若要在 1.8.9 上对照，应取 2.0.x 时期的发行构建，类名与结构与本節描述基本一致。

### 2.2 实体是怎么造的：`EntityHumanNPC extends EntityPlayer`

关键类（1.8 分支实际路径）：

```
src/main/java/net/citizensnpcs/npc/entity/EntityHumanNPC.java
src/main/java/net/citizensnpcs/npc/network/EmptyNetHandler.java
src/main/java/net/citizensnpcs/npc/network/EmptyNetworkManager.java
src/main/java/net/citizensnpcs/npc/network/EmptySocket.java
src/main/java/net/citizensnpcs/npc/network/EmptyChannel.java
src/main/java/net/citizensnpcs/util/NMS.java
```

`EntityHumanNPC` 的类声明与构造：

```java
public class EntityHumanNPC extends EntityPlayer implements NPCHolder {
    public EntityHumanNPC(MinecraftServer minecraftServer, WorldServer world,
                          GameProfile gameProfile, PlayerInteractManager playerInteractManager,
                          NPC npc) {
        super(minecraftServer, world, gameProfile, playerInteractManager);
        this.npc = (CitizensNPC) npc;
        if (npc != null) {
            playerInteractManager.setGameMode(EnumGamemode.SURVIVAL);
            initialise(minecraftServer);
        }
    }
```

也就是说：**它就是一个真正的 `EntityPlayer` 子类**，不是 ArmorStand、不是自定义 `Entity`、也不是只发数据包。
这带来三个直接好处——原版玩家碰撞箱与 hitbox、原版装备/背包/经验/状态效果、原版伤害与击退流程；
代价是必须满足 `EntityPlayer` 的全部前置条件（连接、GameProfile、PlayerInteractManager、world 注册）。

`initialise()` 是整篇调研里最关键的一段代码（下文 4.2 会逐条对照 1.8.9 字节码验证）：

```java
private void initialise(MinecraftServer minecraftServer) {
    Socket socket = new EmptySocket();
    NetworkManager conn = null;
    try {
        conn = new EmptyNetworkManager(EnumProtocolDirection.CLIENTBOUND);
        playerConnection = new EmptyNetHandler(minecraftServer, conn, this);
        conn.a(playerConnection);
        socket.close();
    } catch (IOException e) { /* swallow */ }
    AttributeInstance range = this.getAttributeInstance(GenericAttributes.b);
    if (range == null) range = this.getAttributeMap().b(GenericAttributes.b);
    range.setValue(Setting.DEFAULT_PATHFINDING_RANGE.asDouble());
    controllerJump = new PlayerControllerJump(this);
    controllerLook = new PlayerControllerLook(this);
    controllerMove = new PlayerControllerMove(this);
    navigation = new PlayerNavigation(this, world);
    NMS.setStepHeight(this, 1);   // 默认 0 会导致上不了台阶
}
```

`EmptySocket` 在这里其实**没有被真正使用**（`new` 完立刻 `close()`），属于历史遗留写法；
真正起作用的是 `EmptyNetworkManager` + `EmptyNetHandler` 这一对。

### 2.3 假连接三件套

`EmptyNetHandler` 全文只有一行有效逻辑——把发包变成空操作：

```java
public class EmptyNetHandler extends PlayerConnection {
    public EmptyNetHandler(MinecraftServer minecraftServer, NetworkManager networkManager, EntityPlayer entityPlayer) {
        super(minecraftServer, networkManager, entityPlayer);
    }
    @Override public void sendPacket(Packet packet) { }
}
```

`EmptyNetworkManager` 则必须再往下一层补一个「假 channel」：

```java
public class EmptyNetworkManager extends NetworkManager {
    public EmptyNetworkManager(EnumProtocolDirection flag) throws IOException {
        super(flag);
        NMS.initNetworkManager(this);
    }
}
```

`NMS.initNetworkManager` 用反射往 `NetworkManager` 里塞两个字段（1.8 分支写的是 `"i"` 与 `"j"`）：

```java
public static void initNetworkManager(NetworkManager network) {
    if (NETWORK_CHANNEL == null || NETWORK_ADDRESS == null) return;
    try {
        NETWORK_CHANNEL.set(network, new EmptyChannel(null));
        NETWORK_ADDRESS.set(network, new SocketAddress() { /* serialVersionUID */ });
    } catch (...) { ... }
}
```

为什么必须补这两个字段？因为 `NetworkManager` 的若干方法**不判空**就直接用 `channel`
（1.8.9 字节码里 `NetworkManager.a()` 即 flush 方法直接 `getfield channel` → `Channel.flush()`），
而 `PlayerConnection` 构造时会调用 `networkManager.a(this)`（挂 listener），后者同样会碰 `channel`。
1.8.9 有个好消息：`v1_8_R3` 的字段名已被 Spigot 部分还原，直接就是

```java
public io.netty.channel.Channel channel;
public java.net.SocketAddress l;
```

——**都是 public**，所以 1.8.9 上不需要反射，直接 `network.channel = new EmptyChannel(null)` 即可。
这是 1.8.0 → 1.8.9 迁移中少数「变简单」的地方。

### 2.4 注册：进 `World.players`，不进 `PlayerList.players`

Citizens 从不把 NPC 塞进 `PlayerList.players`。它做的是逐 tick 维护 `World.players`，
再用**手发数据包**维持 tab 列表外观。证据在 `EntityHumanNPC.updatePackets(boolean)`：

```java
boolean removeFromPlayerList = npc.data().get("removefromplayerlist",
        Setting.REMOVE_PLAYERS_FROM_PLAYER_LIST.asBoolean());
NMS.addOrRemoveFromPlayerList(getBukkitEntity(), removeFromPlayerList);
...
NMS.sendToOnline(getListPacket(getBukkitEntity(), true));   // ADD_PLAYER / REMOVE_PLAYER
NMS.sendPacketsNearby(getBukkitEntity(), current, packets); // 装备 + 头部朝向
```

配套的 `NMS.addOrRemoveFromPlayerList` 只动 `World.players`：

```java
public static void addOrRemoveFromPlayerList(org.bukkit.entity.Entity entity, boolean remove) {
    Entity handle = getHandle(entity);
    if (handle.world == null) return;
    if (remove) handle.world.players.remove(handle);
    else if (!handle.world.players.contains(handle)) handle.world.players.add(handle);
}
```

而 `Setting.REMOVE_PLAYERS_FROM_PLAYER_LIST("npc.player.remove-from-list", **true**)` —— **默认就是从
`World.players` 里摘掉**，只在发包层面「看起来在」。

这个设计的后果，必须理解清楚：

| 观察点 | Citizens 默认 NPC | 原因 |
|---|---|---|
| `Bukkit.getOnlinePlayers()` | **不包含** | 1.8.9 `CraftServer` 的 `playerView` 是 `Lists.transform(playerList.players, …)` 的不可变视图（字节码已核对），NPC 不在 `PlayerList.players` 里 |
| `Bukkit.getPlayerExact(name)` | **不返回** | 同上，走 `PlayerList.getPlayer(name)` |
| `World.getPlayers()` | **不包含**（默认配置） | `remove-from-list: true` 时被移出 `World.players` |
| 客户端能看到 | **能** | `PacketPlayOutPlayerInfo` + `PacketPlayOutNamedEntitySpawn`（追踪器或手发） |
| tab 列表 | 可控 | 靠 `EnumPlayerInfoAction.ADD_PLAYER / REMOVE_PLAYER` 两个包 |
| 伤害事件 | **正常** | 走标准 `EntityPlayer.damageEntity` → CraftBukkit 抛 `EntityDamageEvent` / `EntityDamageByEntityEvent` |
| 计分板 / 队伍 | 正常，且**会**被算进去 | 计分板按实体/名字记账，与是否 `getOnlinePlayers` 无关 |
| 聊天 | 只能"发"，不能"收" | 收聊天走 `PlayerConnection.a(PacketPlayInChat)`，假连接永远收不到 |
| Bukkit 实体对象 | `CitizensNPC` 的 `PlayerNPC extends CraftPlayer` | `EntityHumanNPC.getBukkitEntity()` 被重写，返回自定义 `CraftPlayer` 子类 |

最后一行很值得借鉴：Citizens 让 NPC 的 `CraftPlayer` 子类 `PlayerNPC` **重写了 metadata 相关方法**并持有 `npc` 引用，
使得 `player.hasMetadata(...)`、`getMetadata` 这类由 `CraftServer` 统一托管的接口不会因为「不在 players 列表里」而失效。
这是一个「插件兼容性」层面的坑位，自己写核心时同样会遇到。

### 2.5 Trait / 事件 / 调度：哪些值得借鉴

Citizens 的架构可以拆成四层，其中三层的思路值得直接搬到 Bot 核心：

1. **Trait（能力插件化）**。`NPC` 是一个数据壳，行为都挂在 `Trait` 上（`Equipment`、`LookClose`、`Inventory`、
   可选的 `Sentinel` 战斗 Trait）。Trait 有 `onSpawn/onDespawn/run()` 生命周期，`run()` 由统一的
   `NPC.tick()` 调度。**映射到本项目**：`BotBody` 应当是壳，`Locomotion`（移动）、`Combat`（攻击/格挡/暴击）、
   `ItemUse`（右键/吃苹果/甩竿）、`Perception`（观测采样）各自是可替换的模块——训练初期只做 `Locomotion + Combat`，
   后续接动作空间 v2 时不需要动核心。

2. **控制器组合模拟"玩家输入"**。Citizens 显式为假玩家补了四个控制器：
   `PlayerControllerMove`（目标点趋近）、`PlayerControllerLook`（看向实体）、`PlayerControllerJump`（跳跃）、
   `PlayerNavigation`（A* 寻路）。`EntityHumanNPC.s_()`（即 `tick()`）里的驱动片段是：

   ```java
   if (navigating) { NMS.updateNavigation(navigation); moveOnCurrentHeading(); }
   else if (motX != 0 || motZ != 0 || motY != 0) { g(0, 0); }
   ...
   aX *= 0.98F; aY *= 0.98F; aZ *= 0.9F;   // moveForward / moveStrafe 衰减
   g(aX, aY);                               // 真正的移动逻辑
   NMS.setHeadYaw(this, yaw);
   ```

   注意 `g(f, f1)` 是 `EntityLiving.g(float moveForward, float strafe)`——**这就是"模拟 WSWD 按住"的正确入口**，
   和真实客户端按住 W 走的是同一条原版代码路径（含摩擦、加速、台阶、水中减速）。
   比起「每 tick 改 `locX`」，这条路径能自然产出正确的 `motX/motZ`，从而让击退、摔落、`fallDistance` 暴击判定都成立。

3. **`NMS.attack` 的写法值得抄结构**。Citizens 给非玩家生物 NPC 实现的攻击是（1.8 分支 `NMS.attack`）：

   ```java
   float f = (float) handle.getAttributeInstance(GenericAttributes.e).getValue(); // ATTACK_DAMAGE
   int i = 0;
   if (target instanceof EntityLiving) {
       f += EnchantmentManager.a(handle.bz(), ((EntityLiving) target).getMonsterType());
       i += EnchantmentManager.a(handle);       // 击退附魔等级
   }
   boolean flag = target.damageEntity(DamageSource.mobAttack(handle), f);
   if (!flag) return;
   if (i > 0) { target.g(-sin(yaw)*i*0.5, 0.1, cos(yaw)*i*0.5); handle.motX *= 0.6; handle.motZ *= 0.6; }
   ```

   它是手写复刻原版伤害公式，而不是调 `EntityLiving.attack`。**这也是不推荐的做法**：假玩家既然是
   `EntityPlayer`，就直接调 `EntityHuman.attack(entity)`，让原版把附魔、暴击、冲刺击退、火焰附加全部算对（见 4.4）。

4. **发包节流**。`updatePackets` 用 `world.getWorld().getFullTime() % Setting.PACKET_UPDATE_DELAY` 控制
   装备/朝向包频率（默认 30 tick 一次），并把 6 个包攒成数组一次遍历发送。
   对 N 个假人的场景，这是必要的：**装备与头部朝向的广播频率与逻辑 tick 解耦**。

### 2.6 已知限制

- **不发包 = 不可见**。Citizens 的 NPC 可见性完全建立在服务端发包上，客户端没有任何"实体存在"的权威来源。
  因此「反作弊看不看得见」和「服务器认为存不存在」是两套体系——这正是假玩家路线的核心矛盾（见第 5 节）。
- **1.8 分支太老**。`v1_8_R1` 与 `v1_8_R3` 的 `NetworkManager` 字段名、`EntityPlayer` 构造参数顺序需要逐个核对，
  不能照抄。
- **不做战斗**。Citizens 本体只做存在感与交互（对话/商店/巡逻），战斗要额外的 `Sentinel`。也就是说：
  「假玩家 + 战斗 AI」这件事，Citizens 只解决了前一半。
- **反作弊兼容性**：Citizens 官方从不承诺与反作弊兼容；NPC 没有上行包、没有合法 rotation 变化，
  多数客户端侧反作弊根本看不到它（因为它不是"客户端"，没有客户端进程），而**服务端侧**反作弊看到的是
  "一个从不发移动包却一直在动的玩家"——这是最容易被标记的形态。

---

## 3. 「Ultibot」到底指什么

### 3.1 结论：1.8.9 语境下未找到

本次检索覆盖 GitHub、SpigotMC、UltiKits 官方文档与中文社区资料，**未找到任何名为 "Ultibot"（或 UltiBot）
且支持 1.8.9 的练习机器人项目**。同时复查了本仓库 `docs/integration-ultrapvp.md` 与 Ultrapvp 只读源码笔记，
**Ultrapvp 生态内也没有叫 Ultibot 的组件**——Ultrapvp 侧的相关设施是子服切换与 playerdata（H2），不是假人。

### 3.2 最可能的指向：`UltiKits/UltiBot`

| 项 | 事实 | 出处 |
|---|---|---|
| 定位 | UltiTools-API 的**假人（Fake Player）模块** | `https://github.com/UltiKits/UltiBot` |
| 架构 | Maven 多模块：`ultibot-api`（纯接口 `BotPlayer`/`NMSBridge`/`ActionType`）、`ultibot-core`（服务/命令/配置/事件）、`ultibot-v1_21_R1`（NMS 实现）、`ultibot-dist`（shaded） | 同上 README |
| 实体方式 | 基于 NMS 的**真实 `ServerPlayer`**，「完整加入服务器生命周期」 | 同上 |
| 支持版本 | **仅 Paper 1.20.1–1.21.4**，明确不支持 Spigot | `https://doc.ultikits.com/module/ultibot` |
| 维护状态 | **已停止维护，仓库已归档** | 同上（页首 WARNING） |
| 可借鉴 | ①「API 模块 / NMS 实现模块 / 分发模块」的三段式切分；②`NMSBridge` 抽象把版本差异收敛到一个接口；③动作枚举化（JUMP/SNEAK/SPRINT/USE/ATTACK/MINE/LOOK_AT_NEAREST…）；④宏录制回放 | 同上 |

它的「动作枚举 + 间隔 tick 循环执行」模型，和本项目的 11 个离散动作在语义层级上几乎同构——
**但它是"脚本驱动"，本项目是"策略驱动"**，所以只能借鉴外壳，不能借鉴决策。

### 3.3 三个更贴近 1.8.9 的替代参考

1. **`MonkeyMoon104/UltimateBOT`（最推荐）**
   - SpigotMC 资源页 `ultimatebot.138287` 明确列出支持 `1.7 / 1.8 / 1.9 … 1.21 / 26.x`，并声明
     「Bots are implemented as version-specific NMS fake players, not simple ArmorStand placeholders」。
   - 结构：多模块 + 每版本一个 NMS bridge（含 legacy `v1_7_R4`、`1.8.x`），`dist` 出单一 shaded jar。
   - 战斗侧有独立 AI 栈（移动/瞄准/攻击/治疗/图腾/珍珠），难度档位直接改「攻击精度、CPS、reach、反应时间」——
     与本项目 `BotProfile`（`cps`/`reactionMs`/`reach`）的字段设计几乎是同一套思路，可作为"难度参数该有哪些"的参照。
   - 注意：资源页标称开源（`Source Code: github.com/MonkeyMoon104/UltimateBOT`），但**1.8 桥的具体实现未逐行核对**
     （标「未验证」）；且其设计目标是"给真人练枪的对手"，不是"给训练环境提供可控观测源"。

2. **Citizens + `Sentinel`（战斗 Trait）**
   - `https://github.com/mcmonkeyprojects/Sentinel`：给 Citizens NPC 加战斗行为（目标选择、射线检测、走位、逃跑）。
   - 价值在**战斗状态机的外壳**（何时追、何时退、何时不吃伤害），以及它如何把"射线命中"翻译成伤害；
     它的伤害不是走 `EntityHuman.attack`，而是直接对 `LivingEntity` 施加伤害——**这一点本项目不应模仿**（见 4.4）。

3. **纯数据包假人（Packet-only Fake Player）**
   - 代表实现：`https://gist.github.com/DreamTexX/7220ec0253fb9f54bac720bc7ad9885b`，
     以及 SpigotMC 讨论帖 `threads/error-while-despawning-entities.81573`（`NPC extends EntityPlayer` +
     `PacketPlayOutPlayerInfo` + `PacketPlayOutNamedEntitySpawn`）。
   - 特点：`EntityPlayer` 只作为**数据载体**存在，从不进任何 world/player 列表，客户端可见性完全靠手发包。
   - 与本项目的关系：**这是"最省服务端开销、最不安全"的一档**。它证明了一件事——
     "造实体"与"注册实体"是两件可分离的事，第 4.3 节的三档注册策略就来自这里。

### 3.4 小结：可借鉴 / 不可借鉴

| 可借鉴 | 不可借鉴 |
|---|---|
| 三段式模块切分（api / nms-impl / dist） | 脚本化宏动作回放（本项目由策略驱动） |
| 动作枚举 + 固定 tick 节拍 | 直接 `setHealth`/直接施加伤害绕过原版流程 |
| 难度档位 → `cps`/`reach`/反应时间参数化 | 用 ArmorStand 伪装玩家（碰撞箱、装备、伤害全不对） |
| NMS 版本桥收敛到单一接口 | 依赖 `ServerPlayer` 完整登录流程（拉进 `PlayerList` 的副作用太大） |

---

## 4. 1.8.9 服务端模拟玩家的技术要点

> 本节所有签名均以 `javap -p -c` 对 `spigot-1.8.8.jar`（`v1_8_R3`）核对。1.8.8 与 1.8.9 同为 `v1_8_R3`、
> 同为协议 47，包名与签名一致，差异只在 Spigot/Paper 的补丁级别（见 5.3）。

### 4.1 造实体

构造签名（已核对）：

```java
public EntityPlayer(MinecraftServer server, WorldServer world,
                    com.mojang.authlib.GameProfile profile,
                    PlayerInteractManager playerInteractManager);
```

必需的四件套：

| 参数 | 获取方式 | 注意 |
|---|---|---|
| `MinecraftServer` | `((CraftServer) Bukkit.getServer()).getServer()` | 不要用 `MinecraftServer.getServer()` 的静态缓存做长期持有，重载插件后可能失效 |
| `WorldServer` | `((CraftWorld) location.getWorld()).getHandle()` | 跨世界移动必须重建实体或用 `PlayerList.changeDimension`，不能只改 `world` 字段 |
| `GameProfile` | `new GameProfile(uuid, name)` | **UUID 必须自造**（`UUID.nameUUIDFromBytes(("OfflinePlayer:"+name).getBytes(UTF_8))` 或随机）；皮肤要走 `profile.getProperties().put("textures", new Property("textures", value, signature))` |
| `PlayerInteractManager` | `new PlayerInteractManager(world)` | 构造后必须 `setGameMode(WorldSettings.EnumGamemode.SURVIVAL)`；否则默认 `NOT_SET`，破坏/放置行为异常 |

其他要点：

- **`Entity()` 构造里自增的 `entityCount` 决定 `getId()`**，追踪器与所有 `PacketPlayOut*` 都用这个 id；
  不要手动改 `id`，否则 `EntityTracker` 的 `IntHashMap` 会撞。
- **`setLocation(x, y, z, yaw, pitch)` 只是写字段**，不触发任何事件、不做碰撞检查。
- 生成后要 `world.addEntity(entity)`（或 `WorldServer.addEntity`），否则实体不进 tick 循环、不进追踪器。
- `EntityHuman` 构造会创建 `PlayerInventory`、`PlayerAbilities`、`DataWatcher`（含皮肤标志位 10）。
  1.8.9 里 `datawatcher.watch(10, (byte) 127)` 这一步不能漏，否则客户端渲染的皮肤层全隐藏（Citizens 就显式做了）。
- 血量：`EntityLiving.setHealth(float)` 是 `final`，`getMaxHealth()` 来自
  `GenericAttributes.maxHealth` 属性；**不要**为了"给 Bot 加血"去改 maxHealth 而不改 health，会立刻死。

### 4.2 连接：能不能不给连接？不能

**结论：必须给一个 `PlayerConnection` 实例，但可以让它背后没有 socket。**

字节码证据：对 `EntityPlayer` 全类扫描「引用 `Field playerConnection` 的方法」，命中 20+ 个，包括
`s()`（tick 内部分支）、`a(MobEffect)` / `a(MobEffect, boolean)` / `b(MobEffect)`（药水效果同步）、
`b(IChatBaseComponent)`（`sendMessage`）、`closeInventory()`、`enterCombat()` / `exitCombat()`、
`a(Container, int, ItemStack)`（背包同步）、`mount(Entity)`、`enderTeleportTo`、`broadcastCarriedItem()`、
`c(int)`（经验）、`a(boolean, boolean, boolean)`（能力同步）等。

也就是说：**只要不给连接，第一次喝药水/开背包/挨打进战斗状态就会 NPE**，而且是在主线程 tick 里 NPE，
影响面是整个服务器。这不是"能不能跑起来"的问题，是"跑多久炸"的问题。

正确做法（1.8.9 版的 Citizens 三件套）：

```java
// 1) 无 Channel 的网络管理器（1.8.9 字段是 public，直接赋值，无需反射）
NetworkManager net = new NetworkManager(EnumProtocolDirection.CLIENTBOUND);
net.channel = new EmptyChannel(null);          // 一个所有方法都空实现的 Channel 子类
net.l = new java.net.SocketAddress() {};       // 占位地址，getSocketAddress() 不返回 null

// 2) 空发包的 PlayerConnection（构造里会把 entityPlayer.playerConnection 回填成 this）
PlayerConnection conn = new PlayerConnection(server, net, entity);  // 或自定义子类，重写 sendPacket 为空

// 3) 让 NetworkManager 记录 listener（构造 PlayerConnection 时已自动完成 net.a(conn)）
```

`PlayerConnection` 构造的字节码顺序已核对：
`this.minecraftServer = server` → `this.networkManager = net` → `net.a(this)` → `this.player = entity` →
`this.player.playerConnection = this` → `this.server = MinecraftServer.server`。
所以**不要自己给 `entity.playerConnection` 赋值**，让构造器回填，顺序不会错。

哪些地方会因"无连接"或"空 channel"出问题（按风险排序）：

| 位置 | 症状 | 处理 |
|---|---|---|
| `PlayerConnection.sendPacket(Packet)` | 空实现即安全；不重写则 `NetworkManager` 的 `channel` 为空 → NPE | 必须重写成空方法 |
| `NetworkManager.a()`（flush） | 直接 `channel.flush()`；`channel == null` 时 NPE | 给 `EmptyChannel`（flush 返回 this） |
| `EntityTracker` / `EntityTrackerEntry.updatePlayer` | 对本实体而言无害；它发包给的是**真实玩家**的连接 | 无需处理 |
| `PlayerList.tick()` | 遍历 `players` 时读 `playerConnection`（keepalive/ping）；只对已进 `PlayerList` 的实体生效 | 见 4.3 的注册策略 |
| `CraftPlayer.kickPlayer` / `PlayerConnection.disconnect` | 走 `networkManager.close(component)` → `channel.close()` | `EmptyChannel` 的 `close()` 返回 `DefaultChannelPromise` 即可，或禁止对假人 kick |
| `EntityPlayer.die()` | 会走死亡流程并可能触碰连接 | 用 `EmptyNetHandler` 兜住；并考虑拦截死亡事件自行处理（重生/复位） |

**不存在"零连接也能 tick"的官方路径**。省掉的是 socket 与 Netty pipeline，省不掉 `PlayerConnection` 这个对象。

### 4.3 注册策略：三档，副作用完全不同

| 档 | 做法 | 实体存在 | 客户端可见 | `getOnlinePlayers()` | `PlayerJoinEvent` | 主要副作用 |
|---|---|---|---|---|---|---|
| **A. 只进 World** | `world.addEntity(e)` + 维护 `World.players` | 是 | 是（追踪器自动发 `NamedEntitySpawn`） | 否 | 否 | `World.getPlayers()` 会包含（要看 `remove-from-list` 配置）；实体参与区块 tick、碰撞、怪物索敌 |
| **B. 进 World + PlayerList** | 追加 `PlayerList.players.add(e)`（或走 `PlayerList.a(EntityPlayer, WorldServer)`） | 是 | 是 | **是** | **是**（`a(...)` 内 `new PlayerJoinEvent` 已核对） | 在线人数 +1、`/list` 可见、统计/经济/权限/计分板全部当真人、`PlayerList.tick` 会碰 keepalive；`a(...)` 会 `world.addEntity`（但不 `spawnIn`，需自行补） |
| **C. 纯发包** | 存 `EntityPlayer` 不注册，手发 `PlayerInfo`+`NamedEntitySpawn`+`EntityTeleport` | 否（不 tick） | 是 | 否 | 否 | 服务端零开销但**完全不动**：没有 tick、没有重力、没有伤害，等于自己写一套物理与战斗 |

字节码依据：

- `CraftServer` 的 `playerView` 字段在构造时 `Lists.transform(PlayerList.players, …)` 后
  `Collections.unmodifiableList(...)`；`getOnlinePlayers()` 直接返回该视图 → **A 档永远不出现在在线列表**，
  B 档立刻出现。
- `PlayerList.a(EntityPlayer, WorldServer)` 内部（字节码顺序）：`players.add` → `playersByName.put` →
  `new PlayerJoinEvent(player, joinMessage)` → `callEvent` → `world.addEntity(player)`；
  **注意它不调 `spawnIn`**（`spawnIn` 在登录路径 `PlayerList.a(NetworkManager, EntityPlayer)` 里），
  所以用它"入列"必须在调用前自己 `entity.spawnIn(world)`，否则部分状态（视角/维度相关）不完整。
- `EntityTracker.track(Entity)` 首行就是 `if (entity instanceof EntityPlayer)` 分支，走
  `addEntity(entity, 512, 2)`（视距 512、更新间隔 2）；`EntityTrackerEntry` 对 `EntityHuman` 构造的是
  `PacketPlayOutNamedEntitySpawn`（字节码已核对）→ **只要实体进了追踪器，可见性不需要手写包**。

**推荐：A 档为主 + 受控地"看起来像玩家"。** 理由：

1. 本项目 Bot 的对手是真人（陪练），需要被攻击/被击退/被计分，A 档全部满足；
2. 训练场景不允许 Bot 影响在线人数、排行榜、经济——B 档的副作用无法全部关掉；
3. C 档要自己实现物理与伤害，等于把原版重写一遍，且与 19 维观测的语义（血量/护甲/距离）脱钩。

唯一需要额外补的是「tab 列表与皮肤」：A 档下 `PacketPlayOutPlayerInfo` 不会自动发，
需要在 spawn 时对附近玩家手发一次 `ADD_PLAYER`（Citizens 的 `getListPacket` 就是这个用途），
despawn 时补 `REMOVE_PLAYER`。

### 4.4 战斗模拟

**（1）入口：`EntityHuman.attack(Entity)`，不要自己算伤害。**

1.8.9 的 `EntityPlayer.attack(Entity)` 只是把 SPECTATOR 分流到 `setSpectatorTarget`，其余
`invokespecial EntityHuman.attack`。真正的逻辑在 `EntityHuman.attack`，反编译得到的调用序列（节选）：

```
GenericAttributes.ATTACK_DAMAGE → AttributeInstance.getValue()      // 基础攻击力（含武器）
EnchantmentManager.a(ItemStack, EnumMonsterType)                    // 附魔额外伤害
EnchantmentManager.a(EntityLiving)                                  // 击退附魔等级
isSprinting() ...                                                   // 冲刺击退分支
fallDistance > 0 && !onGround && !k_() && !V() && !BLINDNESS ...    // 暴击判定
DamageSource.playerAttack(this) → Entity.damageEntity(src, f)       // 唯一伤害入口
motX/motY/motZ 修正（击退，方向由 yaw 决定）
```

要点：

- **暴击 = `fallDistance > 0 && !onGround && !isInWater() && !isPassenger() && !hasEffect(BLINDNESS)
  && vehicle == null`**。想模拟"跳砍暴击"，只需让实体真的跳起来并保持 `fallDistance > 0` 时调 `attack`——
  这也是为什么移动必须走原版 `g()`/`move()`，而不是瞬移。
- **击退走 `motX/motZ` 增量**，方向由实体 `yaw` 决定。所以 `look` 必须先于 `attack` 生效（本项目 11 个
  动作本来就都以 `look` 开头，见 `docs/observation.md` §5.2），否则击退方向会错。
- **1.8.9 没有攻击冷却**。`EntityHuman.attack` 的字节码里没有任何冷却字段参与伤害计算；1.9 才引入
  `attackStrengthTicker` 作为伤害缩放因子。任务书前提需修正为：1.8 的"手感"由 **CPS 上限** 与
  **点击节奏** 决定，`BotProfile.cps` 是正确的建模位置。
  （`EntityHuman` 里确实有一个每 tick 递减的整数计时器 `bm`，但它只在本类 `m()` 里自减，
  没有出现在伤害公式中；其确切语义标「未验证」，不应作为冷却建模依据。）
- **`EntityDamageByEntityEvent` 会自动触发**：CraftBukkit 在 `EntityLiving.damageEntity` → CraftBukkit 桥接处
  构造并派发 `EntityDamageEvent` / `EntityDamageByEntityEvent`；`DamageSource.playerAttack(this)` 保证
  `damager` 是 `Player`（即 `getBukkitEntity()`，A 档下是自定义 `CraftPlayer` 子类）。
  这意味着：**任何监听伤害的插件（含 WorldGuard 的 PVP 拦截、竞技场的伤害统计）都会正常收到事件**，
  这与"伪造实体"路线相比是巨大优势。

**（2）reach 必须自己判。** 原版 `attack` 不做距离校验——服务端信任客户端"我点到你了"。所以：

```java
double reach = 3.99;                       // 与 BotProfile.reach 一致
if (eyeLocation.distanceSquared(target.getEyeLocation()) > reach * reach) return; // 自己拦
```

不做这一步的后果是：AI 会在 8 格外"砍空气却掉血"，而这恰恰是服务端反作弊最容易抓的形态。

**（3）不要用 `setHealth` 造伤害。** 直接改血量会跳过护甲、抗性、吸收、无敌帧、击退、事件、
死亡结算，导致训练环境与真实对局分布不一致——这是本项目最忌讳的"训练/部署不一致"。

### 4.5 移动模拟

三种做法，按"像玩家"程度排序：

| 做法 | 机制 | 会不会触发 `PlayerMoveEvent` | 评价 |
|---|---|---|---|
| **改输入量**：自己维护「前进/横移/跳跃」三个输入量，每 tick 调 `EntityLiving.g(float moveForward, float strafe)`（1.8.9 已核对存在该重载） | 与真实客户端按住 W/A/S/D 完全同路径 | 否 | **推荐**。摩擦、加速、台阶、水中/爬梯、`fallDistance` 全部正确 |
| **用控制器**：`PlayerControllerMove.a(x, y, z, speed)`（Citizens 的 `setMoveDestination`） | 目标点趋近，产出 `aX/aY` | 否 | 适合"走到某点"，不适合 RL 的逐 tick 动作 |
| **直接改坐标**：`setPosition(x, y, z)` 或 `move(dx, dy, dz)` | 瞬移/强制位移 | 否 | 仅用于复位、传送、卡住脱困；用于逐 tick 移动会破坏击退/摔落/碰撞 |

**关键事实（字节码已核对）**：`PlayerMoveEvent` 的构造点唯一——在
`PlayerConnection.a(PacketPlayInFlying)`（入站移动包处理）里，`new PlayerMoveEvent(player, from, to)`。
服务端主动改坐标**不走**那条路径。因此 A/B 档假人都不会让 `PlayerMoveEvent` 自动触发。

> 字段名提醒：Citizens `1.8` 分支（`v1_8_R1`）里三个输入量叫 `aX`（前进）/`aY`（横移）/`aW`（跳跃），
> 并在每 tick 做 `aX *= 0.98F; aY *= 0.98F` 的衰减。`v1_8_R3` 的混淆字段名**本次未逐一确认**（标「未验证」）；
> 实现时应以「`EntityLiving.g(float, float)` 的入参」为准，并通过 `EntityHuman.m()`（tick）里读取的字段反查，
> 不要照抄 `v1_8_R1` 的名字。

影响：
- **领地/保护插件**（WorldGuard region flags、传送门、载具）多数依赖 `PlayerMoveEvent` → 假人"穿墙无感"，
  这既是便利（不会被拦）也是风险（无法用同一套规则约束）。
- **反作弊**的服务端侧模块常以"移动包频率 vs 位移"做校验 → 假人根本没有移动包，多数会被判为
  "静止却位移"或干脆不进检测样本。

缓解（按需选择，不要全上）：
1. 让 `BotBody` 的移动实现**自己**在同一 tick 内 `Bukkit.getPluginManager().callEvent(new PlayerMoveEvent(...))`，
   把语义补齐（注意：伪造事件会被某些插件当成真实移动，需评估）；
2. 或者只对"关注区域"（竞技场 region）做一次 `PlayerMoveEvent` 派发，用于触发保护逻辑；
3. 或者接受"假人不触发移动事件"，在 Bot 核心的文档里显式声明这条边界，由调用方（Gladiator / 竞技场插件）
   自行决定是否补事件。

### 4.6 使用物品 / 切换手持

| 需求 | 1.8.9 正确写法 | 说明 |
|---|---|---|
| 切手持 | `entity.inventory.itemInHandIndex = slot`（0–8） | 改完要 `entity.broadcastCarriedItem()`（内部就是发包，`EmptyNetHandler` 会吞掉）；不广播则其他客户端看到的还是旧物品 |
| 取手持 | `entity.inventory.getItemInHand()` / `entity.bz()` | `bz()` 是 `EntityLiving` 的 getHeldItem |
| 吃金苹果 / 右键物品 | `entity.playerInteractManager.useItem(entity, world, stack)` → 内部走 `ItemStack.useItemRightClick` | **必须传 `EntityHuman` 而不是 `Player`**；`PlayerInteractManager.useItem` 会处理 `interactResult` 与 `firedInteract` 标志 |
| 按住右键（吃苹果 1.6s） | `EntityHuman` 的 `activeItem` + `itemInUseCount` 字段对（1.8 的"持续使用"状态机），配 `PlayerInteractManager.useItem` 启动 | 字段名在 1.8.9 为混淆名，**本文未逐字节确认**（「未验证」）；`isBlocking()` 读的也是同一对字段 |
| 举剑格挡 | `entity.isBlocking()` 读的是 `activeItem` 与盾/剑状态 | 与吃苹果共用 `activeItem`，不能同时 |
| 丢物品 | `entity.drop(stack, false)` | 会走 `EntityItem` 生成流程与事件 |
| 攻击挥臂动画 | 无连接时客户端看不到挥臂 → 需对附近玩家手发 `PacketPlayOutAnimation(entity, 0)` | **反作弊/观感的关键**：真实客户端每次攻击都发 swing 包 |

**结论**：物品使用的"业务入口"在 `PlayerInteractManager.useItem` / `ItemStack.useItemRightClick`，
与是否有连接无关；但**可见性**（挥臂、举剑、吃苹果的手部动画）只能靠手发包补，
`EmptyNetHandler` 会把所有自动发包吞掉——这是"空连接"方案必须配套 `BotBody.broadcast()` 的原因。

---

## 5. 风险清单与缓解

### 5.1 反作弊（不点名具体产品，统称"常见反作弊"）

| 风险 | 触发机理 | 缓解 |
|---|---|---|
| 无上行移动包 | 服务端侧模块以"移动包 vs 位移"做校验，假人位移但零包 | ① 接受（多数服务端侧模块只对"有连接的玩家"采样）；② 或把假人排除在检测之外——**由调用方 init 处理，Gladiator 不感知**（ADR-015） |
| 瞬移 | 直接 `setPosition` 造成单 tick 位移 > 阈值 | 逐 tick 走 `g()`；仅在复位时瞬移并标记"传送" |
| reach 异常 | 手写 `attack` 不判距离 | 显式 `reachSquared` 校验（3.99） |
| 恒定 CPS / 零抖动 | 固定 tick 攻击，间隔完全一致 | `BotProfile.cps` + `cpsJitter` 采样（已有字段） |
| 视角零抖动 | yaw/pitch 精确锁头 | `reactionMs` + 抖动；`look` 加小角度噪声 |
| 无挥臂包 | 客户端看不到攻击动作 | 手发 `PacketPlayOutAnimation(id, 0)` |
| 长时间零输入 | 只在攻击 tick 有动作 | 空闲时给微小 `aX/aY` 抖动或 `idle` 动作 |

**最重要的认知**：假玩家在"客户端侧反作弊"眼里根本不存在（没有客户端进程），
所以真正要担心的只有**服务端侧**模块。而服务端侧模块通常只处理有 `PlayerConnection` 的玩家——
这恰好是 A 档（不进 `PlayerList`）的隐性优势：**它不在"可被检测的玩家集合"里**。
反过来的代价是：它也可能不在"应当被保护的玩家集合"里（如 PVP 区判定、伤害豁免）。

### 5.2 与其他插件的冲突面

| 冲突面 | 表现 | 缓解 |
|---|---|---|
| 统计/排行榜 | 击杀/死亡被计入，K/D 被刷 | 独立 UUID 命名空间（`AI_` 前缀 + 自造 UUID）；统计插件侧配置忽略；或 B 档下用 `vanilla-statistics` 开关（UltimateBot 就有这对配置） |
| 经济 | 掉落物/击杀奖励被 Bot 吃掉 | Bot 不参与经济：`PlayerInteractManager` 设为 SURVIVAL 但 inventory 由 Bot 核心独占；不触发商店 |
| 计分板/队伍 | 假人被算进队伍、显示在侧栏 | 显式不进任何 `Team`；或进专用 Team 并在计分板渲染层过滤 |
| 在线人数上限 | B 档会占 `max-players` | 用 A 档；若必须 B 档，注意 `PlayerList` 上限校验 |
| 领地/保护 | A 档不触发 `PlayerMoveEvent`，可穿 region | Bot 核心暴露 `isBot(uuid)` 供调用方判断；或补发移动事件 |
| 计分板/队伍 API 遍历 | `Bukkit.getOnlinePlayers()` 在 B 档下包含假人 → 所有"给所有玩家发消息/发资源包/发标题"的插件都会命中 | A 档天然规避 |
| 睡眠/时间跳过 | `World.players` 里的假人会阻止 `everyoneSleeping` | 默认不进 `World.players`（同 Citizens `remove-from-list: true`） |
| 实体数量上限/TPS | 见 5.4 | 硬上限 + 拒绝策略 |

### 5.3 版本兼容：1.8.8 vs 1.8.9

| 项 | 结论 |
|---|---|
| 包名 | **完全相同**：都是 `net.minecraft.server.v1_8_R3` / `org.bukkit.craftbukkit.v1_8_R3` |
| 协议 | 都是 47（`minecraft-data('1.8.9')` 亦为 47） |
| NMS 签名 | `EntityPlayer` 构造、`PlayerConnection` 构造、`NetworkManager` 字段一致（本文所有 javap 结论在 1.8.8 与 1.8.9 通用） |
| 真实差异 | 服务端补丁级别：Spigot 1.8.8（`spigot-1.8.8.jar`）vs Spigot/Paper 1.8.9 的 build 号；Paper 额外有 `PaperSpigot` 配置项与部分 `EntityTracker` 优化 |
| 1.8.0 vs 1.8.8/9 | **差异显著**：Citizens `1.8` 分支是 `v1_8_R1`，`NetworkManager` 字段是混淆名 `i`/`j`（要反射），`EntityPlayer` 构造参数顺序需重核 |
| 1.9+ | `EntityLivingBase` 在 1.8 **不存在**（1.8 的基类是 `EntityLiving`，`EntityHuman extends EntityLiving`）。任务书里提到的 `EntityLivingBase.attackEntityFrom` 是 1.9+ 的名字；1.8.9 对应 `EntityLiving.damageEntity(DamageSource, float)` 与 `EntityHuman.attack(Entity)` |

**建议**：Bot 核心的 NMS 模块只针对 `v1_8_R3` 写一份，用 `ServerVersion` 校验（`Bukkit.getBukkitVersion()`）
不匹配就禁用并给出明确日志，不做 1.8.0/1.9 的多版本桥——那是另一个量级的工程。

### 5.4 性能估算

单个假人每 tick 的成本构成（按原版实体 tick 路径）：

| 项 | 估算 | 说明 |
|---|---|---|
| `EntityPlayer.tick()` 基类链路 | 与一只僵尸同量级 | 包含 AI 属性、药水、食物、着火、空气 |
| 碰撞与方块查询（`move`） | 与移动速度相关，占比最大 | AABB 扫描 + `getBlocks` |
| `PlayerConnection`（空） | ≈ 0 | 所有发包被吞 |
| `EntityTracker` 发包 | 与**附近真实玩家数**成正比 | 每个观察者一组 `NamedEntitySpawn`/`EntityTeleport`/装备包 |
| 观测采样（19 维） | 每控制周期一次（10 Hz），可忽略 | 读实体字段，无分配最好复用 buffer |

粗略量级（**估算，必须实测**）：单假人 **0.02–0.05 ms/tick**；20 个假人约 **0.4–1.0 ms/tick**
（即 50 ms 预算的 1–2%）；100 个假人约 **2–5 ms/tick**，开始影响 TPS。社区经验（Citizens 文档与
多个假人插件 README 的表述）也为"几十个无感、上百个明显"。

真正的瓶颈不是实体 tick，而是：
1. **寻路**（A* 每 tick 或每 N tick 重算，开销可达实体 tick 的 3–10 倍）→ 本项目 11 个动作里没有"寻路到点"，
   只有 `move` 四方向 + `jump`，可完全避开寻路；
2. **区块加载**（假人所在区块必须加载，`KEEP_CHUNKS_LOADED`）；
3. **发包放大**（N 个假人 × M 个观察者 × 每 tick 包数）。

**容量建议**：单服务端假人上限默认 8、硬上限 16（可由调用方配置），并在 `BotBody.spawn` 时拒绝超限，
返回明确错误码而不是静默降级。

---

## 6. 架构建议

### 6.1 `BotBody` 接口（Bot 核心的唯一门面）

```java
public interface BotBody {
    // 生命周期
    BotId spawn(BotSpawnSpec spec);      // spec: name, uuid, location, gameMode, profile(cps/reach/reactionMs)
    void  despawn(DespawnReason reason);
    boolean isAlive();

    // 驱动（由核心自己的 tick 调度器调用，不暴露给业务方）
    void  tick();                        // 内部：输入衰减 → move → 状态同步 → 观测快照

    // 动作原语（与 11 个离散动作一一对应，语义与 bot-runtime 完全一致）
    void  move(float forward, float strafe);   // -1..1，对应 aX/aY
    void  jump(boolean jumping);
    void  look(float yaw, float pitch);
    void  attack(BotId target);                // 内部做 reach 校验 + EntityHuman.attack
    void  use(UseSpec spec);                   // item: gapple/rod/sword/blocks/food, holdTicks
    void  selectSlot(int slot);
    void  sneak(boolean), sprint(boolean);

    // 观测（服务端侧直接读实体，不需要 blockWindow 压缩）
    ObservationSnapshot snapshot();      // 19 维语义 + 原始字段

    // 只读属性
    Location location(); float health(); float yaw(); float pitch(); boolean onGround();
}
```

设计约束（这几条比接口本身重要）：

1. **`BotBody` 不暴露任何 `net.minecraft.server.*` 类型**，也不暴露 `Player`（否则调用方会 `instanceof` 分支，
   零依赖红线破功）。`BotId` 用自有值类型。
2. **`tick()` 不对外**。业务方只能"设意图"，不能"推进世界"，避免调用方在错误的线程/时机驱动实体。
3. **观测与动作的语义必须由核心保证**（19 维字段名/顺序/归一化、11 个动作的退化规则），
   而不是让 Gladiator 各自实现一遍——否则训练/部署两端必然漂移。
4. **所有世界写入必须切主线程**。核心内部用 `Bukkit.getScheduler().runTask` 包一层，
   外部即便是异步线程（如 Dashboard 的 HTTP 线程）也不会炸 `ConcurrentModificationException`。

### 6.2 项目形态：独立 Gradle 多模块，`compileOnly` 依赖

推荐结构：

```
bot-core/                     (独立仓库，独立 Gradle 项目)
├─ bot-core-api/              // 纯接口 + 值类型，零 Bukkit 依赖（或仅 compileOnly spigot-api）
├─ bot-core-nms-v1_8_R3/      // 唯一实现，compileOnly spigot-1.8.8
├─ bot-core-plugin/           // 可选：薄插件壳（/bot 命令、配置、ServiceLoader 装配）
└─ bot-core-test/             // 集成测试：起本地 1.8.8 服务端跑 spawn/attack/despawn
```

Gladiator 侧：

```gradle
dependencies {
    compileOnly project(':bot-core-api')   // 或 maven 坐标
    // 不依赖 bot-core-nms-v1_8_R3，运行时由服务端插件提供
}
```

装配方式二选一：

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **软依赖 + 服务发现** | `plugin.yml: softdepend: [BotCore]`；`Bukkit.getServicesManager().load(BotProvider.class)` | 类型安全、无反射 | 需要 BotCore 插件在位 |
| **反射加载** | 按类名 `Class.forName("...BotBodies")` + 方法反射 | 完全不依赖 | 类型不安全、易碎 |

**推荐第一种**，但要注意 `softdepend` 的语义：BotCore 缺失时 Gladiator 必须能正常启动并给出
"服务端假人不可用，回退到真实客户端模式"的明确提示——而不是抛异常。

**为什么不建议做成 Gladiator 的子模块**：

1. Bot 核心会被别的插件复用（换竞技场插件只改调用方，这条约束同样适用于 Bot 核心）；
2. NMS 模块需要独立的编译目标与独立的集成测试服务器，混在主项目里会让 `build` 变慢；
3. 一旦 NMS 升级（1.8.9 → 更高版本），独立项目可以单独发版，Gladiator 不动。

### 6.3 与 Citizens 共存

| 事项 | 做法 |
|---|---|
| 类冲突 | 不 `extend` 任何 Citizens 类，不复用其 `NPC`/`Trait`；两套 `EntityPlayer` 子类可以同时存在（`EntityTypes` 注册按类名，互不影响） |
| 包冲突 | 不要 `import net.citizensnpcs.*`；`plugin.yml` 不 depend/softdepend Citizens |
| 名单冲突 | bot 名字强制前缀（`AI_`），并在 spawn 前查 `Bukkit.getPlayerExact` / `hasPlayedBefore`（见 `docs/identity-and-action-space.md` §1.3） |
| 发包冲突 | 两边都可能对同一实体发 `PlayerInfo`；不同实体 id 不冲突，但**同一玩家**收到两组 `ADD_PLAYER` 相同名字会串——前缀即可规避 |
| 调度冲突 | 各自用自己的 `BukkitRunnable`；不要共用 Citizens 的 `NPC.tick()` |
| 观测冲突 | Citizens NPC 会出现在 `world.getEntities()` 里；Bot 的观测（`entities` 数组）必须按"是否 Bot/是否参赛者"过滤，不能"最近实体即对手" |

### 6.4 与 `bot-runtime`（真实 mineflayer 客户端）的分工

| 维度 | `bot-runtime`（训练） | Bot 核心（部署/陪练） |
|---|---|---|
| 登录 | 真实 TCP 登录，离线 UUID | 无连接，自造 UUID |
| 观测来源 | `OBSERVATION` 帧（json-v1，含 `blockWindow` 196 格） | 服务端直读实体/方块，**语义相同但无需压缩** |
| 动作入口 | `ACTION` 帧 → `link-server.js` | `BotBody` 方法调用 |
| 时钟 | `controlHz = 10`，`MOVE_MS = 180` | 同：10 Hz 决策 + 内部 20 tick 执行 |
| reach | `profile.reach = 3.99` | 同 |
| 吃苹果 | `activateItem` + 1600 ms + `deactivateItem` | `use{gapple, holdTicks:32}` |
| 数值一致性 | 血量/护甲/距离/物品 | **必须逐字段对齐**，见 `docs/observation.md` §4 |

**必须守住的语义契约**（否则策略迁移即失效）：

1. **19 维向量逐维同义同序同归一化**。服务端拿到的是"真值"，训练侧拿到的是"帧值"，
   但 `encode_observation` 的裁剪与归一化必须共用同一套定义（建议把 `OBS_FIELDS`/`OBS_CLIP` 做成
   跨语言共享的 JSON，Java 侧读同一份）。
2. **11 个动作的退化规则一致**：`eat_gapple` 在 `health ≥ 19.5` 时退化为 `idle(180ms)`；
   `attack` 超 reach 时在训练侧是 `OUT_OF_REACH`，在服务端侧必须表现为"空挥但不掉血"（同样的观测后果）。
3. **动作节拍一致**：每个动作占用的 tick 数（`MOVE_MS=180` ≈ 3.6 tick）必须一致，
   否则同一策略在两端的"每秒决策数"不同，价值估计会漂。
4. **观测量纲一致**：`target_health` 在训练侧来自对手自己的 `OBSERVATION.self.health`，
   服务端侧直接读 `EntityLiving.getHealth()`——两者都是 0–20 的浮点，不能一边取整。

### 6.5 分阶段落地建议

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | `BotBody` 接口 + A 档 spawn/despawn + `move/look/jump` | 假人能被真人看到、能被推动、能走上台阶、不 NPE 跑 30 分钟 |
| P1 | `attack`（reach + `EntityHuman.attack`）+ swing 包 | 真人被打掉血、被击退；`EntityDamageByEntityEvent` 正常；跳砍出暴击 |
| P2 | `use`（金苹果/鱼竿）+ `selectSlot` + 装备同步 | 吃苹果回血、鱼竿拉人、手持物品在客户端正确显示 |
| P3 | `snapshot()` 19 维 + 与 `training/gladiator_env` 对齐 | 同一场景下两端 19 维向量逐维误差 < 1e-3（除时间相关项） |
| P4 | 与 Gladiator 集成（`GladiatorBotReadyEvent` 语义兼容）+ 反作弊豁免交接 | 换竞技场插件只改调用方；反作弊豁免由调用方 init 完成 |

---

## 7. 未验证清单（后续必须实测）

1. `EntityHuman.activeItem` / `itemInUseCount` 在 1.8.9 的确切字段名与"按住右键 1.6s"的正确驱动方式
   （本文只确认了 `PlayerInteractManager.useItem` 入口，连续使用的字段未逐字节核对）。
2. `EmptyChannel` 的最小实现集合：`flush/close/writeAndFlush/isOpen/...` 是否全部需要重写，
   还是只重写 `flush`/`close` 即可（取决于 `NetworkManager` 在 1.8.9 里还调用了哪些 channel 方法）。
3. `EntityHuman.bm` 的真实语义（每 tick 自减的整数计时器，是否与挥臂/格挡节流相关）。
4. 服务端侧反作弊对"无移动包玩家"的实际判定（需要在实际部署环境跑一次对照实验，
   且**该实验由调用方完成**，Gladiator 不引入任何反作弊依赖）。
5. 单假人 tick 成本的真实值（本文 0.02–0.05 ms 为估算）；20/50/100 假人的 TPS 曲线。
6. `PlayerMoveEvent` 手动派发对 WorldGuard / 领地类插件的实际影响（是否会造成重复扣费/重复传送）。
7. Citizens 当前 `master` 是否仍带 `v1_8_R3` 模块（本次 tree 检索被截断）。

---

## 8. 参考来源

| # | 来源 | 用途 |
|---|---|---|
| 1 | `https://github.com/CitizensDev/Citizens2`（分支 `1.8`，commit `7739e5d`） | NPC 实体、假连接、注册策略、控制器 |
| 2 | 同上：`src/main/java/net/citizensnpcs/npc/entity/EntityHumanNPC.java` | `extends EntityPlayer`、`initialise()`、`updatePackets()` |
| 3 | 同上：`.../npc/network/EmptyNetHandler.java`、`EmptyNetworkManager.java`、`EmptySocket.java`、`EmptyChannel.java` | 假连接三件套 |
| 4 | 同上：`src/main/java/net/citizensnpcs/util/NMS.java` | `initNetworkManager` / `addOrRemoveFromPlayerList` / `attack` / `sendPacketsNearby` / `setHeadYaw` |
| 5 | 同上：`src/main/java/net/citizensnpcs/Settings.java` | `npc.player.remove-from-list` 默认 `true` |
| 6 | 本机 `D:\ServerUltrapvp\spigot-1.8.8.jar`（`javap -p -c`，`v1_8_R3`） | 全部 1.8.9 侧签名与调用链：`EntityPlayer`/`EntityHuman`/`EntityLiving`/`PlayerConnection`/`NetworkManager`/`PlayerList`/`EntityTracker(Entry)`/`CraftServer` |
| 7 | `https://github.com/UltiKits/UltiBot` + `https://doc.ultikits.com/module/ultibot` | UltiBot 定位、模块划分、版本约束、归档状态 |
| 8 | `https://www.spigotmc.org/resources/ultimatebot.138287/` + `https://github.com/MonkeyMoon104/UltimateBOT` | 1.8 系 NMS 假人陪练的可用参考 |
| 9 | `https://github.com/mcmonkeyprojects/Sentinel` | Citizens 生态的战斗 Trait（状态机外壳） |
| 10 | `https://gist.github.com/DreamTexX/7220ec0253fb9f54bac720bc7ad9885b`、`https://www.spigotmc.org/threads/error-while-despawning-entities.81573/` | 纯发包假人（`PacketPlayOutPlayerInfo` + `NamedEntitySpawn`）的社区实现 |
| 11 | 本仓库 `docs/observation.md`、`docs/identity-and-action-space.md`、`docs/protocol.md` | 19 维/11 动作语义、Bot 命名与 UUID 策略、`BotProfile` 字段 |

> 说明：本报告未修改任何源码、配置或运行时产物，也未触碰 `dashboard/var` 与 `training/artifacts`。
