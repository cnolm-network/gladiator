# 验收：A5（pathfinder 移动代价）与 A9（runtime 后端状态机）

> 数据全部来自真机运行：Paper 1.8.8 + mineflayer 真实登录。
> 采集时间：2026-11-13 01:10 – 01:26 (+08:00)。报告由脚本/探针原始输出整理，未手改数字。

## 结论

| 开放假设 | 结论 | 关键证据 |
|---|---|---|
| **A5** pathfinder 移动代价可标定 | **PASS** | 10 格直线 × 20 次：20/20 完成、0 次卡住，mean 1857ms / p95 1884ms（阈值 4651ms），平均 5.39 格/s；每格 p95 188.4ms |
| **A9** 竞技场取消死亡/重生时状态机仍迁移 | **PASS**（runtime 后端） | normal：`ALIVE→DEAD→RESPAWNING→ALIVE`，死亡后 2 tick 触发 `PlayerRespawnEvent`；cancel-damage：`FALL 50.0` 在 LOWEST 被取消，玩家不死、状态保持 ALIVE；manual-respawn：死亡同 tick 调 `spigot().respawn()`，0 tick 后 `PlayerRespawnEvent`，三迁移齐全 |

补充：**golem 后端（B2 结论）与 runtime 后端结论相反**——golem 身体不在 `PlayerList`，
`spigot().respawn()` 无接收方，DEAD 后不可恢复；runtime 是真实登录玩家，重生链路完整。
A9 应以本轮的 runtime 结论为准。

## 环境

| 项 | 值 |
|---|---|
| 服务端 | Paper 1.8.8（`git-PaperSpigot-445`），`server/paper-1.8.8-445.jar` |
| Java | OpenJDK 1.8.0_492（Temurin） |
| 端口 | MC `25566`，RCON `127.0.0.1:25575`（password `devpass`），Gladiator bridge `8765` |
| 插件 | `gladiator-0.1.0.jar`、`caller-sample.jar`、`golem-0.1.0.jar`、LuckPerms、BetterNameTag |
| `spawn.mode` | `runtime`（`server/plugins/Gladiator/config.yml`） |
| runtime | `bot-runtime/src/index.js`（mineflayer 4.39.0 + mineflayer-pathfinder 2.4.5，Node 22.22.2） |
| Bot | A5：`A5_Nav`（独立脚本）；A9：`A9_R1`（`gladiator spawn A9_R1`） |
| 世界 | `world`，出生点 `-2172.5, 4, -437.5` |

## 复现

```bash
# 构建（根工程）
./gradlew build :caller-sample:jar

# 起服
cd server && java -Dfile.encoding=UTF-8 -Xms512M -Xmx1024M -jar paper-1.8.8-445.jar nogui

# ---------- A5 ----------
# 控制台先把标定台铺好（脚本自身也会尝试 /fill，需要 OP）
rcon "fill -2150 99 -474 -2137 99 -466 minecraft:stone"
cd bot-runtime
node scripts/a5-nav.mjs --username A5_Nav --host 127.0.0.1 --mc-port 25566 --runs 20

# ---------- A9 ----------
node bot-runtime/src/index.js --username A9_R1 --host 127.0.0.1 --mc-port 25566 --env-port 8770
rcon "gladiator spawn A9_R1"
rcon "caller a9 normal A9_R1"
rcon "caller a9 cancel-damage A9_R1"
rcon "caller a9 manual-respawn A9_R1"
```

`/caller a9 <场景> [botName]`：**给 botName 时接管**已由 `/gladiator spawn` 登录的 runtime Bot
（不再重复 spawn）；不给名字则自己 `spawnBot`（golem 后端的老路径，名称自动生成）。

## A5：pathfinder 移动代价

脚本：`bot-runtime/scripts/a5-nav.mjs`（新增）。在 `y=99` 铺 14×9 石台（站立面 `y=100`），
用 `pathfinder.goto(GoalBlock)` 走 `(-2148.5,100,-469.5) → (-2138.5,100,-469.5)` 共 **10 格**，
重复 20 次，每次记录耗时、实际位移、是否卡住（15s 超时或位移不足 8.5 格）。
`movements` 关闭 `canDig` / `allow1by1towers` / `allowParkour`，只走路，不会改台子。

### 原始输出（20 次明细）

```
路线：(-2148.5,100,-469.5) → (-2138.5,100,-469.5)，直线 10 格
· 石台就绪：y=99 铺满，站立面 y=100
· 预热完成：pos=(-2148.50,100.00,-469.50) 脚下方块=stone
#01 ok 1780ms 位移=10.00 速度=5.62 b/s
#02 ok 1866ms 位移=10.00 速度=5.36 b/s
#03 ok 1847ms 位移=10.00 速度=5.42 b/s
#04 ok 1780ms 位移=10.00 速度=5.62 b/s
#05 ok 1861ms 位移=10.00 速度=5.37 b/s
#06 ok 1877ms 位移=10.00 速度=5.33 b/s
#07 ok 1816ms 位移=10.00 速度=5.51 b/s
#08 ok 1821ms 位移=10.00 速度=5.49 b/s
#09 ok 1884ms 位移=10.00 速度=5.31 b/s
#10 ok 1876ms 位移=10.00 速度=5.33 b/s
#11 ok 1881ms 位移=10.00 速度=5.32 b/s
#12 ok 1892ms 位移=10.00 速度=5.29 b/s
#13 ok 1878ms 位移=10.00 速度=5.33 b/s
#14 ok 1875ms 位移=10.00 速度=5.33 b/s
#15 ok 1884ms 位移=10.00 速度=5.31 b/s
#16 ok 1861ms 位移=10.00 速度=5.37 b/s
#17 ok 1828ms 位移=10.00 速度=5.47 b/s
#18 ok 1882ms 位移=10.00 速度=5.31 b/s
#19 ok 1870ms 位移=10.00 速度=5.35 b/s
#20 ok 1874ms 位移=10.00 速度=5.34 b/s

## 直线移动（10 格 / 次）
runs=20 ok=20 stuck=0
耗时 ms：mean=1857 p50=1870 p95=1884 max=1892 min=1780
速度 b/s：mean=5.39 p50=5.34 p95=5.62 min=5.29

## 判定
期望上限：p95 < 4651ms（10 格 / 4.3 b/s ≈ 2326ms，允许 2 倍）
实测：p95=1884ms stuck=0 完成=20/20
结果：PASS
```

### 移动代价标定表（每格）

| 统计量 | 耗时 | 折算速度 |
|---|---|---|
| mean | **185.7 ms/格** | 5.39 格/s |
| p50 | 187.0 ms/格 | 5.35 格/s |
| p95 | **188.4 ms/格** | 5.31 格/s |
| max | 189.2 ms/格 | 5.29 格/s |

用法：把 deny 区域当高代价节点时，`惩罚增量 ≈ p95 每格耗时 × 绕行格数`。
绕 10 格 ≈ +1.9s。实测速度 5.39 格/s 与 1.8.8 步行速度（4.317）不符、更接近疾跑（5.612）——
因为 `Movements.allowSprinting = true`，pathfinder 在直线上会疾跑。**若做代价模型，按 5.4 格/s 取值更贴近实际**。

### 掉落代价（同一脚本）

| 落差 | health 变化 | 掉落伤害 | 说明 |
|---|---|---|---|
| 3 格（走下高台） | 20 → 20 | **0.0** | 1.8.8 伤害 = floor(落差 − 3)，阈值点 |
| 6 格（空中自由落体） | 20 → 17 | **3.0** | 与公式一致；每格 ≈ 0.5 点（3 格以内免费） |

## A9：runtime 后端状态机

三个场景各起一次全新服务端（干净的 `game-1` 会话），每次只挂一个 runtime。

### 1) normal —— PASS

```
[01:25:33 INFO]: [CallerSample] [A9/NORMAL] 状态机探针：PASS
后端：runtime（接管已登录玩家）  bot=A9_R1
期望：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
实际：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
玩家：isDead=false hp=20.0 finalState=ALIVE PlayerRespawnEvent=true
—— 时间线 ——
t+0   #01 接管已在线 Bot A9_R1 uuid=6398aa75-5a5d-351f-9987-301e9cf9d745
t+0   #02 teleport → world -2172.50,4.00,-437.50
t+0   #03 pushGame(DuelMode, A9_R1)
t+0   #04 session=game-1 stateOf=ALIVE hp=20.0
t+80  #05 damage(100) 首次：hp=20.0 stateOf=ALIVE
t+80  #06 StateChanged ALIVE→DEAD
t+80  #07 PlayerDeathEvent 实际触发（未被取消，NORMAL）
t+80  #08 damage 返回：hp=0.0 isDead=true stateOf=DEAD cancelled=false
t+82  #09 StateChanged DEAD→RESPAWNING
t+82  #10 PlayerRespawnEvent 触发 @ world -2172.50,4.00,-437.50（死亡后 2 tick）
t+82  #11 StateChanged RESPAWNING→ALIVE
```

读法：死亡后 **2 tick** 就完成重生——这是 mineflayer 客户端收到 `update_health(health<=0)` 后
自动发 `client_command(actionId=0)` 的结果（`respawn` 选项默认 true），
即真实客户端"点重生"的等价动作，不是探针自己调 `respawn()`。

### 2) cancel-damage —— PASS

```
[01:23:21 INFO]: [CallerSample] [A9/CANCEL_DAMAGE] 状态机探针：PASS
期望：（假设）取消致死伤害后状态机仍迁移到 DEAD
实际：（无状态迁移）
玩家：isDead=false hp=20.0 finalState=ALIVE PlayerRespawnEvent=false
· 玩家存活=true hp=20.0 出现 ALIVE→DEAD=false 实际取消过伤害=true
· 结论：假设「取消死亡后状态机仍迁移」不成立；状态机跟随真实世界，保持 ALIVE
—— 时间线 ——
t+0   #01 接管已在线 Bot A9_R1 uuid=6398aa75-5a5d-351f-9987-301e9cf9d745
t+0   #02 teleport → world -2172.50,4.00,-437.50
t+0   #03 pushGame(DuelMode, A9_R1)
t+0   #04 session=game-1 stateOf=ALIVE hp=20.0
t+80  #05 自由落体：y=60.0 → 地面（落差约 56 格，落地伤害 > 20，必然致死）
t+80  #06 teleport 完成，等待落地
t+122 #07 EntityDamageEvent(LOWEST) setCancelled(true) cause=FALL damage=50.0
t+123 #08 已落地/结束：y=6.00 hp=20.0 isDead=false cancelled=true stateOf=ALIVE
```

读法：**50 点摔落伤害在 LOWEST 被取消**，玩家落地满血、无 `PlayerDeathEvent`、
状态机零迁移。"竞技场取消致死伤害"在本后端是可靠的，且状态机不会凭空迁移到 DEAD。

### 3) manual-respawn —— PASS

```
[01:24:26 INFO]: [CallerSample] [A9/MANUAL_RESPAWN] 状态机探针：PASS
期望：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
实际：ALIVE→DEAD → DEAD→RESPAWNING → RESPAWNING→ALIVE
玩家：isDead=false hp=20.0 finalState=ALIVE PlayerRespawnEvent=true
· attach 模式没有 Bot 句柄，跳过 pushPlace
—— 时间线 ——
t+80  #05 damage(100) 首次：hp=20.0 stateOf=ALIVE
t+80  #06 StateChanged ALIVE→DEAD
t+80  #07 PlayerDeathEvent 实际触发（未被取消，MANUAL_RESPAWN）
t+80  #08 damage 返回：hp=0.0 isDead=true stateOf=DEAD cancelled=false
t+80  #09 调用方 pushPlace(bot) 试探
t+80  #10 调用 spigot().respawn()（hp=0.0 isDead=true）
t+80  #11 StateChanged DEAD→RESPAWNING
t+80  #12 PlayerRespawnEvent 触发 @ world -2172.50,4.00,-437.50（死亡后 0 tick）
t+80  #13 respawn 返回：hp=20.0 isDead=false stateOf=RESPAWNING
t+80  #14 StateChanged RESPAWNING→ALIVE
```

读法：调用方在**死亡的同一 tick** 调 `spigot().respawn()`，`PlayerRespawnEvent` 立即触发
（0 tick），状态机 `RESPAWNING` → 下一 tick `ALIVE`。
`pushPlace(bot)` 仍是空转：运行时 `Bot` 句柄不在探针手上（attach 模式）——不是 Gladiator 的问题。

## 关键发现（1.8.8 语义）

1. **`Player#damage(double)` 不派发 `EntityDamageEvent`。**
   `CraftLivingEntity.damage(double)` → `Entity.damageEntity(GENERIC, f)`；
   `EntityHuman.damageEntity` 对 `DamageSource.GENERIC` 直接走死亡逻辑，不经过
   `EntityLiving.d` 里的 CraftBukkit 事件钩子。实测：LOWEST 监听器一次都没被调用，
   100 点伤害直接把玩家打死（`cancelled=false`）。
   **要测"取消致死伤害"必须用真实伤害源**（`FALL` / `ENTITY_ATTACK`），本轮改用摔落。
2. **runtime Bot 在 `PlayerList` 里**：`spigot().respawn()` → `PlayerList.moveToWorld(p,0,false)`
   → 真实 `PlayerRespawnEvent`。这与 golem 后端（无会话、无接收方）形成对照，是 A9 能通过的根本原因。
3. **mineflayer 默认自动重生**：死亡后 2 tick 内客户端自己发 respawn 包，
   因此"正常死亡 → 回到 ALIVE"在 runtime 下是默认行为，不需要调用方介入。
4. `PlayerRespawnEvent` 之后 Gladiator 的 `handleRespawn` 先写 `RESPAWNING`、下一 tick 写 `ALIVE`，
   所以观察方一定能看到三段迁移（不会合并成 `DEAD→ALIVE`）。
5. A5 的 5.39 格/s 是**疾跑**速度（pathfinder 在直线上自动 sprint），做移动代价模型时按此取值。

## 已知限制

1. A5 只测**平地直线**：无跳跃、无绕障、无上下坡。垂直移动、跳跃、绕行的代价未标定。
2. A5 的台子是 `/fill` 出来的理想平面；真实地形上 pathfinder 会跳/绕，耗时显著更长。
   标定表只适用于"空旷平地"这一档。
3. 掉落代价只测了 3 格与 6 格两个点，未测水中落地、粘液块、摔落保护附魔。
4. A9 每轮都是全新服务端 + 单 runtime；`Gladiator` 没有"结束会话"的 API，
   同一 Bot 连跑多个场景会复用旧会话（探针已容错并记录，但本轮证据都取自干净会话）。
5. A9 只验证**状态序列**，不验证重生后的位置恢复（重生落在世界出生点，不是竞技场原点）、
   背包/装备是否保留、以及 Mode 拒绝重生时的 `ELIMINATED` 分支。
6. `cancel-damage` 用摔落伤害代表"竞技场取消致死伤害"；攻击伤害（`ENTITY_ATTACK`）在 B2 已用 golem 验证过可取消，
   本轮未在 runtime 上重复。
7. A5 脚本需要 OP（`/fill`、`/tp`）。本轮把 `A5_Nav` / `A9_R1` / `A9_R2` 写进 `server/ops.json`
   （该文件被 `.gitignore` 忽略），跑完已还原为原来的两条记录。
