# 验收：A6（Ready→传送/套件→pushGame 顺序）、A8（数据隔离）、A9（取消死亡/重生的状态机迁移）

> 全部数据由 caller-sample 里的三个自动探针（`/caller a6`、`/caller a8`、`/caller a9 ...`）在真实
> Paper 1.8.8 上跑出，未手填数字。执行时间：2026-11-13 00:56:33 – 00:57:33 (+08:00)。

## 结论

| 开放假设 | 结论 | 关键证据 |
|---|---|---|
| **A6** Ready → 调用方传送/套件 → `pushGame` 顺序稳定 | **PASS** | 每只 Bot 的 Ready 序号（同 tick 内可比）都早于 pushGame；两只都 Ready 后才 pushGame；`sessionId=game-1` 非空、两只 `stateOf=ALIVE` |
| **A8** 两场对局的数据隔离有效 | **PASS** | A/B 两场 session 不同（game-2 / game-3）；两板参与者各 2 人且不相交；A 场胜场只在 A 板、B 场只在 B 板，跨场查询均为 0 |
| **A9** 竞技场取消死亡/重生时状态机仍迁移 | **部分（不能关闭）** | 取消致死伤害时状态机**不**误迁移（保持 ALIVE，与真实世界一致）；但 golem 后端**无法**产生 `PlayerRespawnEvent`，`DEAD→RESPAWNING→ALIVE` 两段走不通（normal / manual-respawn 均 FAIL） |

## 环境

| 项 | 值 |
|---|---|
| 服务端 | Paper 1.8.8（`git-PaperSpigot-445`） |
| Java | OpenJDK 1.8.0_492（Temurin，`jdk-8.0.492.9-hotspot`） |
| 端口 | MC `25566`；RCON `127.0.0.1:25575`（password `devpass`） |
| golem | `golem-0.1.0.jar`（进程内 Bot 身体） |
| gladiator | `gladiator-0.1.0.jar`（本轮构建），`api=1.1.0` |
| caller-sample | `caller-sample.jar`（本轮构建，含 a6/a8/a9 探针） |
| **`spawn.mode`** | **`golem`**（临时改动，跑完已还原为 `runtime`；原因见「环境说明」） |
| 世界/出生点 | `world`，`-2173, 4, -438` |
| Bot 命名 | 探针按 `时间戳%100000` 生成，如 `a6a93899` / `a8a2011` / `a9can10113`（保证每次唯一） |

### 环境说明（为什么用 golem 模式）

`spawn.mode` 默认是 `runtime`：Bot 由外部 mineflayer 运行时经通道 A 登录（每只 Bot 一个 node 进程）。
本次验收环境里没有运行 `bot-runtime` 进程，`spawnBot` 只会停在「等待 runtime HELLO」，
240 tick 后超时（第一次 A6 实测即如此：`session=null`、两只 Bot 无 Ready）。

因此把 `server/plugins/Gladiator/config.yml` 的 `spawn.mode` 临时改为 `golem`（进程内生成身体，
无需外部运行时）。**这个后端差异直接决定了 A9 的结果**（见「未通过项原因」），跑完已还原。

## 复现步骤

```powershell
# 1) 构建（根工程出 gladiator jar，子工程出 caller-sample jar）
cd D:\cbot-gladiator
.\gradlew build :caller-sample:jar

# 2) 部署
Copy-Item build\libs\gladiator-0.1.0.jar      server\plugins\ -Force
Copy-Item caller-sample\build\libs\caller-sample.jar server\plugins\ -Force

# 3)（可选）把 spawn.mode 改为 golem：server\plugins\Gladiator\config.yml → spawn.mode: golem

# 4) 起服后经 RCON 驱动（node %TEMP%\rcon.mjs "<command>"，命令不带前导 /）
caller a6
caller a8
caller a9 normal
caller a9 cancel-death
caller a9 manual-respawn
```

## A6 · 顺序探针（PASS）

```
[00:56:35 INFO]: [CallerSample] [A6] 顺序探针：PASS
[00:56:35 INFO]: [CallerSample] session=game-1  bots=a6a93899,a6b93899
[00:56:35 INFO]: [CallerSample] —— 时间线 ——
[00:56:35 INFO]: [CallerSample] t+0   #01 spawnBot 请求 a6a93899
[00:56:35 INFO]: [CallerSample] t+0   #02 GladiatorBotReadyEvent a6a93899 botId=glad-a6a93899
[00:56:35 INFO]: [CallerSample] t+0   #03 调用方 teleport a6a93899 → world -2173.00,4.00,-438.00
[00:56:35 INFO]: [CallerSample] t+0   #04 调用方发套件 a6a93899（+1 diamond_sword）
[00:56:35 INFO]: [CallerSample] t+0   #05 spawnBot 请求 a6b93899
[00:56:35 INFO]: [CallerSample] t+0   #06 GladiatorBotReadyEvent a6b93899 botId=glad-a6b93899
[00:56:35 INFO]: [CallerSample] t+0   #07 调用方 teleport a6b93899 → world -2173.00,4.00,-438.00
[00:56:35 INFO]: [CallerSample] t+0   #08 调用方发套件 a6b93899（+1 diamond_sword）
[00:56:35 INFO]: [CallerSample] t+0   #09 pushGame(DuelMode, [a6a93899, a6b93899]) 调用
[00:56:35 INFO]: [CallerSample] t+0   #10 pushGame 返回 sessionId=game-1 users=2
[00:56:35 INFO]: [CallerSample] t+0   #11 stateOf(a6a93899)=ALIVE
[00:56:35 INFO]: [CallerSample] t+0   #12 stateOf(a6b93899)=ALIVE
[00:56:35 INFO]: [CallerSample] 状态迁移：（无）pushGame 前两者已是 ALIVE，没有 ALIVE→ALIVE 的迁移事件
```

断言与读法：

1. **每只 Bot 的 Ready 早于 pushGame**：`#02 < #09`、`#06 < #09`（用同 tick 内单调递增的序号比较，
   而不是 tick 值——golem 模式下 spawn/Ready/pushGame 全落在同一 tick）。
2. **两只都 Ready 之后才 pushGame**：`readyBots.size()==2` 是 pushGame 的唯一触发条件（#09 之前两只都到齐）。
3. **sessionId 非空 + 状态 ALIVE**：`game-1`、两只 `stateOf=ALIVE`。
4. `GladiatorStateChangedEvent` 在本场景**没有出现**：两只 Bot 在 pushGame 前就是 ALIVE，
   pushGame 不产生"变化"，所以没有 `ALIVE→ALIVE` 事件。这是实现现状（事件只在状态**变化**时发），
   不是探针漏采——A9 里能正常采到 `ALIVE→DEAD`。

## A8 · 隔离探针（PASS）

```
[00:56:43 INFO]: [CallerSample] [A8] 隔离探针：PASS
[00:56:43 INFO]: [CallerSample] sessionA=game-2 sessionB=game-3
[00:56:43 INFO]: [CallerSample] A场 参与者=2 胜场=3d71ec46:1, e8296a20:0
[00:56:43 INFO]: [CallerSample] B场 参与者=2 胜场=40f5a1db:1, 358c27f2:0
[00:56:43 INFO]: [CallerSample] —— 时间线 ——
[00:56:43 INFO]: [CallerSample] t+0   #01 spawnBot 请求 a8a2011
[00:56:43 INFO]: [CallerSample] t+0   #02 Ready a8a2011
[00:56:43 INFO]: [CallerSample] t+0   #03 spawnBot 请求 a8b2011
[00:56:43 INFO]: [CallerSample] t+0   #04 Ready a8b2011
[00:56:43 INFO]: [CallerSample] t+0   #05 spawnBot 请求 a8c2011
[00:56:43 INFO]: [CallerSample] t+0   #06 Ready a8c2011
[00:56:43 INFO]: [CallerSample] t+0   #07 spawnBot 请求 a8d2011
[00:56:43 INFO]: [CallerSample] t+0   #08 Ready a8d2011
[00:56:43 INFO]: [CallerSample] t+0   #09 pushGame A场 (a8a2011,a8b2011)
[00:56:43 INFO]: [CallerSample] t+0   #10 A场 session=game-2
[00:56:43 INFO]: [CallerSample] t+0   #11 pushGame B场 (a8c2011,a8d2011)
[00:56:43 INFO]: [CallerSample] t+0   #12 B场 session=game-3
[00:56:43 INFO]: [CallerSample] t+0   #13 A场胜者 a8a2011 记入 A 板；B场胜者 a8c2011 记入 B 板
[00:56:43 INFO]: [CallerSample] t+0   #14 断言 A 板 winsOf(A胜者) = 1（期望 1）
[00:56:43 INFO]: [CallerSample] t+0   #15 断言 A 板 winsOf(B胜者) = 0（期望 0）
[00:56:43 INFO]: [CallerSample] t+0   #16 断言 B 板 winsOf(B胜者) = 1（期望 1）
[00:56:43 INFO]: [CallerSample] t+0   #17 断言 B 板 winsOf(A胜者) = 0（期望 0）
[00:56:43 INFO]: [CallerSample] t+0   #18 断言 A 板 size = 2（期望 2）
[00:56:43 INFO]: [CallerSample] t+0   #19 断言 B 板 size = 2（期望 2）
```

补充断言（无独立输出行）：两场参与者集合交集为空、两个 sessionId 不同、胜负双方 `stateOf=ALIVE`。

为什么不用 `example.StatsBoard`：它按设计**过滤 Bot**（`recordWin` 对 Bot 返回 false），
用它统计 Bot 对局永远是 0 胜，证明不了隔离。探针用等价的 `UUID → 胜场` Map，
只测"A 场的账不会记到 B 场"。

## A9 · 状态机探针（部分通过）

### A9-a `normal` —— FAIL

```
[00:57:19 INFO]: [CallerSample] [A9/NORMAL] 状态机探针：FAIL
[00:57:19 INFO]: [CallerSample] 期望：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
[00:57:19 INFO]: [CallerSample] 实际：ALIVE→DEAD
[00:57:19 INFO]: [CallerSample] 玩家：isDead=true hp=0.0 finalState=DEAD
[00:57:19 INFO]: [CallerSample] —— 时间线 ——
[00:57:19 INFO]: [CallerSample] t+0   #01 Ready a9nor28209 player=a9nor28209
[00:57:19 INFO]: [CallerSample] t+0   #02 teleport → world -2173.00,4.00,-438.00
[00:57:19 INFO]: [CallerSample] t+0   #03 spawnBot a9nor28209 场景=NORMAL
[00:57:19 INFO]: [CallerSample] t+0   #04 pushGame(DuelMode, a9nor28209)
[00:57:19 INFO]: [CallerSample] t+0   #05 session=game-5 stateOf=ALIVE
[00:57:19 INFO]: [CallerSample] t+80  #06 尝试击杀：a9nor28209.damage(100) hp=20.0
[00:57:19 INFO]: [CallerSample] t+80  #07 StateChanged ALIVE→DEAD
[00:57:19 INFO]: [CallerSample] t+80  #08 PlayerDeathEvent 实际触发（未被取消，NORMAL）
[00:57:19 INFO]: [CallerSample] t+80  #09 damage 返回：isDead=true hp=0.0 stateOf=DEAD
[00:57:19 INFO]: [CallerSample] t+90  #10 调用 spigot().respawn()
[00:57:19 INFO]: [CallerSample] t+90  #11 respawn 返回：isDead=true hp=0.0 stateOf=DEAD
```

死亡段正常（`ALIVE→DEAD`、`PlayerDeathEvent` 一次）；但 `spigot().respawn()` **什么也没做**：
`isDead` 仍为 true、没有 `PlayerRespawnEvent`、状态停在 DEAD。

### A9-b `cancel-death` —— PASS（但假设不成立）

```
[00:56:59 INFO]: [CallerSample] [A9/CANCEL_DEATH] 状态机探针：PASS
[00:56:59 INFO]: [CallerSample] 期望：（假设）取消死亡后状态机仍迁移到 DEAD
[00:56:59 INFO]: [CallerSample] 实际：（无状态迁移）
[00:56:59 INFO]: [CallerSample] 玩家：isDead=false hp=20.0 finalState=ALIVE
[00:56:59 INFO]: [CallerSample] 事件被取消、未触发死亡：玩家 isDead=false hp=20.0，未出现 PlayerDeathEvent 迁移
[00:56:59 INFO]: [CallerSample] 结论：假设「取消死亡后状态机仍迁移」在本条件下不成立；状态机跟随真实世界（保持 ALIVE）
[00:56:59 INFO]: [CallerSample] —— 时间线 ——
[00:56:59 INFO]: [CallerSample] t+0   #01 Ready a9can10113 player=a9can10113
[00:56:59 INFO]: [CallerSample] t+0   #02 teleport → world -2173.00,4.00,-438.00
[00:56:59 INFO]: [CallerSample] t+0   #03 spawnBot a9can10113 场景=CANCEL_DEATH
[00:56:59 INFO]: [CallerSample] t+0   #04 Ready a9atk10113（攻击者）
[00:56:59 INFO]: [CallerSample] t+0   #05 spawnBot a9atk10113（伤害来源）
[00:56:59 INFO]: [CallerSample] t+0   #06 pushGame(DuelMode, a9can10113)
[00:56:59 INFO]: [CallerSample] t+0   #07 session=game-4 stateOf=ALIVE
[00:56:59 INFO]: [CallerSample] t+80  #08 golem attack a9atk10113 → a9can10113（第 1 击）
[00:56:59 INFO]: [CallerSample] t+80  #09 EntityDamageEvent(LOWEST) setCancelled(true) cause=ENTITY_ATTACK damage=8.0
[00:56:59 INFO]: [CallerSample] t+80  #10 hp=20.0 isDead=false cancelled=true stateOf=ALIVE
[00:56:59 INFO]: [CallerSample] t+106 #11 golem attack a9atk10113 → a9can10113（第 2 击）
[00:56:59 INFO]: [CallerSample] t+106 #12 EntityDamageEvent(LOWEST) setCancelled(true) cause=ENTITY_ATTACK damage=8.0
[00:56:59 INFO]: [CallerSample] t+106 #13 hp=20.0 isDead=false cancelled=true stateOf=ALIVE
[00:56:59 INFO]: [CallerSample] t+130 #14 golem attack a9atk10113 → a9can10113（第 3 击）
[00:56:59 INFO]: [CallerSample] t+130 #15 EntityDamageEvent(LOWEST) setCancelled(true) cause=ENTITY_ATTACK damage=8.0
[00:56:59 INFO]: [CallerSample] t+130 #16 hp=20.0 isDead=false cancelled=true stateOf=ALIVE
```

三击各 8 点（合计 24 > 20 血，本来足以致死）全部在 LOWEST 被取消；
玩家没死，状态机也**没有**迁移到 DEAD。也就是说：**状态机不会凭空迁移**，它忠实跟随真实世界。
`TASKS.md` 里 A9 的表述（"取消死亡/重生时状态机仍迁移"）在 1.8.8 + golem 后端下**不成立**，
反倒是更安全的行为；这里按事实记为"假设不成立"，不按通过处理。

### A9-c `manual-respawn` —— FAIL

```
[00:57:33 INFO]: [CallerSample] [A9/MANUAL_RESPAWN] 状态机探针：FAIL
[00:57:33 INFO]: [CallerSample] 期望：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
[00:57:33 INFO]: [CallerSample] 实际：ALIVE→DEAD
[00:57:33 INFO]: [CallerSample] 玩家：isDead=true hp=0.0 finalState=DEAD
[00:57:33 INFO]: [CallerSample] t+80  #06 尝试击杀：a9man42311.damage(100) hp=20.0
[00:57:33 INFO]: [CallerSample] t+80  #07 StateChanged ALIVE→DEAD
[00:57:33 INFO]: [CallerSample] t+80  #08 PlayerDeathEvent 实际触发（未被取消，MANUAL_RESPAWN）
[00:57:33 INFO]: [CallerSample] t+80  #09 damage 返回：isDead=true hp=0.0 stateOf=DEAD
[00:57:33 INFO]: [CallerSample] t+90  #10 先 pushPlace(bot)，再 respawn
[00:57:33 INFO]: [CallerSample] t+90  #11 调用 spigot().respawn()
[00:57:33 INFO]: [CallerSample] t+90  #12 respawn 返回：isDead=true hp=0.0 stateOf=DEAD
[00:57:33 INFO]: [CallerSample] ! pushPlace 抛异常: java.lang.IllegalStateException: Bot 尚未就绪（未收到 GladiatorBotReadyEvent）: glad-a9man42311
[00:57:33 INFO]: [CallerSample] ! 击杀后 140 tick 仍未走完状态迁移
[00:57:33 INFO]: [CallerSample] ! 状态序列不满足 [ALIVE→DEAD, DEAD→RESPAWNING, RESPAWNING→ALIVE]
[00:57:33 INFO]: [CallerSample] ! 结束后 stateOf=DEAD（期望 ALIVE）
```

## 未通过项的具体原因

### 1. `spigot().respawn()` 对 golem Bot 是空操作（A9-a / A9-c 的直接原因）

- `CraftPlayer.Spigot#respawn()` 走的是原版重生链路（客户端 `PERFORM_RESPAWN` → `PlayerList`），
  而 golem Bot **不在 `PlayerList` 里**（它没有真实网络会话，这是 golem 的刻意取舍）。
  所以重生请求没有接收方，`PlayerRespawnEvent` 从未派发。
- 实测表现：调用后 `isDead=true`、`hp=0.0`、没有 `PlayerRespawnEvent`、状态停在 `DEAD`。
- 这条路在 `spawn.mode=runtime` 下会不一样（mineflayer Bot 是真实登录玩家，在 `PlayerList` 里）——
  但本次环境没有运行 `bot-runtime`，无法验证；**待办**：用 runtime 模式复跑 A9-a/A9-c。

### 2. 死亡后 `BotHandle.isReady()` 变 false，`pushPlace` 直接抛异常（A9-c 的第二个原因）

`BotHandle.isReady()` = `ready && player != null && isLive(player)`；Bot 死亡后 `isLive` 为 false，
于是 `pushPlace(bot)` 抛 `IllegalStateException: Bot 尚未就绪`。也就是说"A9-c 用 pushPlace 把尸体
拉回训练场"这条路径在 API 层就是被禁止的（不只是环境问题）。

### 3. 1.8.8 的 `PlayerDeathEvent` 不能取消（A9-b 的对象需要换）

`org.bukkit.event.entity.PlayerDeathEvent` 在 1.8.8 只是 `EntityDeathEvent` 的子类，
**没有实现 `Cancellable`**（1.9+ 才有）——`javap` 已核对。所以"注册 LOWEST 监听器
`setCancelled(true)`"在 1.8.8 **编译不过**，任务书里的写法是 1.9+ 的语义。

本次的等价替代：取消**致死伤害**。探针为此用第二个 Bot 挥剑（钻石剑 8 点/击）产生
`EntityDamageEvent(ENTITY_ATTACK)`，再在 LOWEST 取消——这是 1.8.8 里唯一"能取消死亡"的真实路径。

### 4. `Player#damage(double)` 不派发 `EntityDamageEvent`（探针踩到的坑）

`CraftLivingEntity.damage(double)` 调用的是 `Entity.damageEntity(DamageSource, float)`（低层入口），
而不是带 CraftBukkit 事件补丁的 `EntityLiving.d(DamageSource, float)`。实测：`damage(100)` 直接把
20 血打到 0 并触发 `PlayerDeathEvent`，但探针的 `EntityDamageEvent` 监听器**一次都没收到**。
要构造"可取消的伤害"必须走攻击路径（`EntityHuman.attack` → `EntityLiving.d`），即 A9-b 的做法。

## 已知限制

1. **A9 只在 golem 后端验证过**；`spawn.mode=runtime`（默认）下 mineflayer Bot 在 `PlayerList` 里，
   `spigot().respawn()` 是否可用**未验证**，A9-a/A9-c 不能据此关闭。
2. A9 的"重生"完全依赖调用方显式触发（`spigot().respawn()`）；golem 后端没有"客户端点重生"这回事，
   也没有任何 gladiator API 能驱动 `DEAD→RESPAWNING`。这是后端能力缺口，不是状态机逻辑缺陷。
3. A6 断言只覆盖"Ready 早于 pushGame、两者都在 pushGame 前 Ready、session 非空、状态 ALIVE"；
   **不**覆盖"传送/发套件本身成功"（传送只记录到时间线，未校验落地坐标；套件只记录发了几件）。
4. A8 用自建 `MatchBoard`，不覆盖 `example.StatsBoard` 的"过滤 Bot"约定（那是另一个已有单测的对象）。
5. 三个探针都会在世界上留下 Bot 实体（`keep` 语义），每次运行生成的新 Bot 名不会复用；
   连续跑多次会在 golem 注册表里累积尸体，需要 `/golem remove <name>` 或重启清场。
6. A6/A8 的 Bot 名带时间戳，跨次运行不可复现同一名字；`docs/caller-sample.md` 里的手工步骤仍可用固定名。

## 对 TASKS.md 的影响

- **A6、A8 可以勾选为完成**（证据见本文）。
- **A9 不能关闭**：假设表述需修正为「取消致死伤害时状态机不误迁移（已验证）；重生段需要能派发
  `PlayerRespawnEvent` 的后端（golem 后端不具备）」，并在 runtime 模式复跑 A9-a/A9-c 后再定。
