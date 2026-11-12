# 验收：A2（物品使用时序）与 A4（高频攻击）

> 本文件的数据由 golem 侧的自动探针 `/golem a2`、`/golem a4` 在真机跑出，未手填数字。
> 生成时间：2026-11-13 00:26 (+08:00)；数据时间窗：00:24:10 – 00:25:44。

## 结论

| 开放假设 | 结论 | 关键证据 |
|---|---|---|
| **A1** 方块放置朝向（另见 `docs/golem-integration.md` §7） | **PASS** | `/golem a1 Probe` → 6 面各 `10/10`，合计 60/60；比较的是 `PlayerInteractEvent#getBlockFace()`（早于 `BlockPlaceEvent`），明细 `golem-plugin/build/a1-report.txt` |
| **A2** 物品使用时序（金苹果 / 鱼竿 / 剑格挡） | **PASS** | 9/9 行全绿：金苹果 31 tick 触发 `PlayerItemConsumeEvent`（期望 32±3）、鱼竿 1 tick 生成钩并在 3 tick 内回收、剑格挡 +1 tick 起效并持续 40 tick |
| **A4** 高频攻击（10Hz × 60s） | **PASS** | TPS 均值 19.98（阈值 ≥19.5）、WARN/ERROR 0、`attack` p95 0.301ms（阈值 <5ms）、600 次调用 0 异常 |

原始报告：`golem-plugin/build/a2-report.txt`、`golem-plugin/build/a4-report.txt`（在 golem 仓库，不在本仓库）。

## 环境

| 项 | 值 |
|---|---|
| 服务端 | Paper 1.8.8（`git-PaperSpigot-445`），`server/paper-1.8.8-445.jar` |
| Java | OpenJDK 1.8.0_492（Temurin，`jdk-8.0.492.9-hotspot`） |
| 机器 | Windows 开发机，服务端与 RCON 客户端同机 |
| 端口 | MC `25566`；RCON `127.0.0.1:25575`（password `devpass`） |
| golem | `golem-0.1.0.jar`（本轮构建，含 `/golem a2`、`/golem a4`） |
| 其它插件 | gladiator 0.1.0、LuckPerms、BetterNameTag、CallerSample |
| 世界 | `world`，默认出生点 `-2173, 4, -438` 附近 |
| Bot | `Probe`（被测，策略 `keep`）、`Dummy`（靶，策略 `respawn`） |

## 复现步骤

```powershell
# 1) 构建 golem 并部署
cd D:\cbot-golem
.\gradlew build -PgolemNmsJar=D:/ServerUltrapvp/spigot-1.8.8.jar
Copy-Item .\golem-plugin\build\libs\golem-0.1.0.jar D:\cbot-gladiator\server\plugins\ -Force

# 2) 起服，把两份报告直接写到 golem 的构建目录
cd D:\cbot-gladiator\server
& "C:\Program Files\Eclipse Adoptium\jdk-8.0.492.9-hotspot\bin\java.exe" -Dfile.encoding=UTF-8 -Dgolem.a2.report=D:/cbot-golem/golem-plugin/build/a2-report.txt -Dgolem.a4.report=D:/cbot-golem/golem-plugin/build/a4-report.txt -Xms512M -Xmx1024M -jar paper-1.8.8-445.jar nogui

# 3) 通过 RCON 驱动（node %TEMP%\rcon.mjs "<command>"，命令不带前导 /）
golem spawn Probe keep
golem spawn Dummy respawn
golem a2 Probe                 # 金苹果/鱼竿/剑 各 3 次，约 18 秒
golem a4 Probe Dummy 10 60     # 10Hz × 60s
```

## A2 结果（`/golem a2 Probe`）

控制台（00:24:12 → 00:24:30）：

```
[00:24:12 INFO]: [Golem] [A2] 就绪：Probe @world -2173.00,4.00,-438.00
[00:24:30 INFO]: [Golem] [A2] Probe 结果：PASS (9/9)
[00:24:30 INFO]: [Golem] item    expectation                        expected measured   result
[00:24:30 INFO]: [Golem] gapple  use→PlayerItemConsumeEvent               32       31   PASS
[00:24:30 INFO]: [Golem] gapple  use→PlayerItemConsumeEvent               32       31   PASS
[00:24:30 INFO]: [Golem] gapple  use→PlayerItemConsumeEvent               32       31   PASS
[00:24:30 INFO]: [Golem] rod     抛竿生成鱼漂 + 收竿回收                    1        1   PASS
[00:24:30 INFO]: [Golem] rod     抛竿生成鱼漂 + 收竿回收                    1        1   PASS
[00:24:30 INFO]: [Golem] rod     抛竿生成鱼漂 + 收竿回收                    1        1   PASS
[00:24:30 INFO]: [Golem] sword   isBlocking 起效 / 持续 40 tick            40       40   PASS
[00:24:30 INFO]: [Golem] sword   isBlocking 起效 / 持续 40 tick            40       40   PASS
[00:24:30 INFO]: [Golem] sword   isBlocking 起效 / 持续 40 tick            40       40   PASS
[00:24:30 INFO]: [Golem] 总体：PASS (9/9)
```

明细（`a2-report.txt`）：

```
#01 gapple expected=32t measured=31t PASS | 血量 10.0→12.0 饥饿 20→20 消耗 tick=31（±3）
#02 gapple expected=32t measured=31t PASS | 血量 10.0→14.0 饥饿 20→20 消耗 tick=31（±3）
#03 gapple expected=32t measured=31t PASS | 血量 10.0→14.0 饥饿 20→20 消耗 tick=31（±3）
#04 rod    expected=1t  measured=1t  PASS | 实体数 2→3→2，FISHING_HOOK 0→1→0（峰值），收竿 3 tick
#05 rod    expected=1t  measured=1t  PASS | 实体数 2→3→2，FISHING_HOOK 0→1→0（峰值），收竿 3 tick
#06 rod    expected=1t  measured=1t  PASS | 实体数 2→3→2，FISHING_HOOK 0→1→0（峰值），收竿 3 tick
#07 sword  expected=40t measured=40t PASS | 首次 blocking=true 在 +1 tick，持续 40 tick（±2）
#08 sword  expected=40t measured=40t PASS | 首次 blocking=true 在 +1 tick，持续 40 tick（±2）
#09 sword  expected=40t measured=40t PASS | 首次 blocking=true 在 +1 tick，持续 40 tick（±2）
```

读法与判定：

- **金苹果**：`use` 后 **31 tick** 触发 `PlayerItemConsumeEvent`（1.8 食物使用时长为 32 tick；
  探针在 `use` 当 tick 记起点、消耗在下一 tick 的 `EntityHuman.t_()` 完成，故稳定早 1 tick）。
  血量 10 → 12/14（Regeneration 生效），饥饿保持 20（金苹果 `alwaysEdible`，满饥饿也能吃）。
  **时序结论：消耗完成节奏与客户端一致（32±1 tick），不是"立刻消耗"，也不是"永不消耗"。**
- **鱼竿**：`use` 后 **1 tick** 世界实体数 +1 且类型为 `FISHING_HOOK`；再 `use` 一次 3 tick 内回收
  （实体数回到 2）。**结论：抛竿/收竿都走原版 `EntityFishingHook` 生命周期。**
- **剑格挡**：`use(sword, 40)` 后 **+1 tick** `isBlocking()` 为 true，**持续 40 tick** 回到 false。
  **结论：格挡窗口与 `holdTicks` 严格一致。**

阈值：金苹果 32±3 tick；鱼竿 ≤2 tick；剑格挡起效 ≤1 tick、持续 40±2 tick。

## A4 结果（`/golem a4 Probe Dummy 10 60`）

控制台（00:24:44 → 00:25:44）：

```
[00:24:44 INFO]: [Golem] [A4] 开始：Probe → Dummy，10.0Hz，60s
[00:25:44 INFO]: [Golem] [A4] Probe → Dummy 10.0Hz/60s：PASS
[00:25:44 INFO]: [Golem] TPS 均值 19.98 （窗口最差 20.00） 最差 tick 120.5ms 抖动 70.5ms
[00:25:44 INFO]: [Golem] 攻击 600 （命中 600 / 未命中 0 ） 伤害事件 30 WARN/ERROR 0
[00:25:44 INFO]: [Golem] 耗时 p50 0.135ms p95 0.301ms max 3.709ms mean 0.171ms
```

报告（`a4-report.txt`）：

```
bot=Probe target=Dummy hz=10.0 seconds=60 ticks=1200
result=PASS

## 判定阈值
min_average_tps=19.5 max_p95_ms=5.0 errors_must_be=0

## TPS（每 5 秒窗口）
window#01 20.00
window#02 20.00
window#03 20.00
window#04 20.00
window#05 20.00
window#06 20.00
window#07 20.00
window#08 20.00
window#09 20.00
window#10 20.00
window#11 20.00
average=19.98 worst_window=20.00 worst_tick=120.5ms max_jitter=70.5ms

## 攻击
calls=600 hits=600 misses=0 damage_events=30 errors=0
latency p50=0.135ms p95=0.301ms max=3.709ms mean=0.171ms

## 结论
PASS
```

判定项逐条对照：

| 指标 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| TPS 均值 | ≥ 19.5 | 19.98 | PASS |
| WARN/ERROR 计数 | = 0 | 0 | PASS |
| `attack` p95 | < 5 ms | 0.301 ms | PASS |
| 调用异常 | = 0 | 0 | PASS |
| 调用次数 | 600（10Hz × 60s 理论值） | 600 | PASS |

说明：

- **频率**：1200 tick 内 600 次 `attack`，即每 2 tick 一次，与 10Hz 目标一致（误差 0）。
- **命中 600 / 伤害事件 30**：`attack()` 的返回值只表示"在 reach 内并执行了攻击"。靶子 `Dummy`
  20 血、每击 8 点，死亡后 40 tick 复活、复活后 60 tick 无敌，所以 60 秒里真正造成伤害的只有 30 次，
  **其余 570 次打在尸体/无敌帧上**。这不影响吞吐与耗时结论，但**不能**把 600 当真实战斗命中率。
- **耗时**：单次 `attack`（含 `EntityHuman.attack` → 伤害事件 → 装备属性扫描）p50 0.135ms、
  p95 0.301ms、max 3.709ms，远低于 5ms 阈值。
- **TPS**：11 个 5 秒窗口全部 20.00，整体均值 19.98（尾部不足一窗的样本参与整体均值、
  但不参与窗口统计）。期间人工核对服务端日志窗口：无 `Can't keep up`、无 WARN/ERROR。
- **一次 120.5ms 的单 tick 抖动**（max_jitter 70.5ms）、出现在尾部窗口外的样本里，
  未把均值拉低（19.98 仍 > 19.5）；来源未定位，**不排除**报告写盘或 JVM GC，见「已知限制」。

## 已知限制

1. **TPS 指标是"插件自测的相邻 tick 间隔"，不是 Paper 的 `/tps`**。空服 + 两个 Bot 的场景下
   两者等价，但**不能**替代有玩家、有区块加载的压力测试；重负载需重跑本探针。
2. 单次 120.5ms 的 tick 抖动未定位（1200 tick 中 1 次）。它没有让任何 5 秒窗口 TPS 掉下 20，
   但若验收要求"零抖动"，本轮数据不满足。
3. `attack()` 的"命中"只是 reach 内执行了攻击；真实伤害次数看 `damage_events`（本轮 30 次，
   受靶子死亡/复活无敌影响）。
4. A2 金苹果时序稳定比期望早 1 tick（31 vs 32），是探针起点相位造成的固定偏差，容差 ±3 已覆盖；
   **不要**把它当成精确 32。
5. A2 的鱼竿只验证"钩生成/回收"，不验证钓鱼判定（咬钩、钓上物品）。
6. 两个探针都会临时改 Bot 状态：A2 把血量设 10、发放金苹果/鱼竿/铁剑，跑完把血量/饥饿复位，
   但**背包里的物品会留着**；A4 会把靶子打死（`respawn` 策略下 40 tick 后自动复活）。
7. WARN/ERROR 计数是**探针自己的 try/catch**（任务体与 `attack` 调用），不覆盖其它插件在别处抛的
   异常；本轮另人工核对了服务端日志窗口，确认无 WARN/ERROR 行。
8. `PlayerArmSwingEvent` 在 1.8.8 不存在（等价物是 `PlayerAnimationEvent(ARM_SWING)`）；
   与本轮 A2/A4 无关，但会影响按 1.9+ 事件名写监听器的插件。

## 对 TASKS.md 的影响

`TASKS.md`「开放假设」中的 A2、A4 可据本文勾选为已完成。本次未修改 `TASKS.md`（按约定只新增
`docs/` 下的文件）。
