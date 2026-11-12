# 调用方 init 检查清单（A7）

> 本清单是 **Gladiator 对外契约的一部分**：Gladiator 只负责 Bot 生命周期、状态机、动作与观测；
> 环境准备、规则放行、套件、传送、匹配、结算全部由调用方在自己插件里完成。
> Gladiator 源码、配置、默认值里不出现任何反作弊/领地插件的名字或权限节点（ADR-015、G-208）。

调用方在自己环境里按以下条目逐项自检，全部满足即可认为 A7 通过（Gladiator 侧不读取、不检测这些项）。

## 1. 进程与端口

| 项 | 要求 | 检查方式 |
|---|---|---|
| 服务端 | Spigot/Paper 1.8.8（`git-PaperSpigot-445` 已验证） | `/version` |
| Gladiator | `plugins/gladiator-*.jar`，`ready=true` | `/gladiator status` |
| Bot 来源 | `spawn.mode: runtime`（mineflayer 登录）或 `golem`（进程内身体） | `/gladiator status` 里的 `spawn=` |
| 回环链路 | `127.0.0.1:8765`（通道 A）、`127.0.0.1:8770+`（通道 B，每个运行时一个） | `Get-NetTCPConnection -State Listen` |
| 控制通道 | RCON 仅在自测/运维时开启，生产建议关闭 | `server.properties` 的 `enable-rcon` |

## 2. 服务器规则（由调用方设置）

| 项 | 要求 | 说明 |
|---|---|---|
| `online-mode` | 与 Bot 登录方式一致：离线 Bot → `false`；正版 Bot → `true` | 否则登录失败码 `ONLINE_MODE` |
| `spawn-protection` | 建议 `0`（否则 Bot 无法放置方块） | 只影响 `place` |
| `pvp` | 训练/对局世界需要 `true` | 否则攻击无效 |
| 游戏模式 | Bot 生成后为 `SURVIVAL`；观战者用 `SPECTATOR` | Gladiator 不改调用方世界的默认值 |
| 世界高度/地面 | 出生点必须有实心地面；超平坦世界建议 `y=4` | 否则 Bot 落地摔死（训练环境已内置 `spawn.golem.location` 与准备期免伤） |

## 3. 反作弊与环境（调用方负责，Gladiator 不涉及）

- 反作弊插件（如有）需为 Bot 账号放行：`bypass`/白名单/豁免权限**在调用方插件或配置里给**。
- 领地/保护插件（WorldGuard 等）的区域、旗帜、成员**由调用方配置**；
  Gladiator 不注册任何「放行/取消」监听器，只在**观察到被拒**时记一次 deny 并推 `RULE_DENIED(source=observed)`。
- 若需要精确的区域 ID（而不是观测点 ± 半径），由调用方通过 `queryPvp` 的 `PvpVerdict` 或自己的事件归因补充。

## 4. 套件与出生

- 套件、清背包、回血、出生点（`spawnpoint`）、传送（`tp`）由调用方在 `GladiatorBotReadyEvent` 里做；
- 顺序：收到 Ready → 传送/发套件 → 再 `pushGame`/`pushPlace`（提前 push 会因 `bot.isReady()` 或状态判定失败）；
- 训练环境提供了 `ArenaConfig.kit` 的参考实现（钻石套 + 剑 + 5 金苹果 + 鱼竿 + 套件核对），可照抄逻辑到自己插件。

## 5. 事件与统计

- 只监听、不取消：`GladiatorStateChangedEvent`、`GladiatorSessionEndedEvent`、`GladiatorBotReadyEvent`、`GladiatorBotFailedEvent`、`GladiatorBotRemovedEvent`。
- 对局统计、排行榜、经济等**全部在调用方**；Gladiator 只提供 `analyze()` 的只读快照（伤害/方块/路径/规则拒绝/动作计数）。
- 需要跨对局保存战绩时用调用方自己的存储；Gladiator 的 `TrainingSession`/`GameSession` 会随 `endSession` 清理。

## 6. 零绑定门禁（Gladiator 侧已自动验证）

- `./gradlew check` 会跑 `verifyStandalone`：禁止 `top.cnolm.<arena-plugin>` 之类的 import、禁止 `plugin.yml` 里 depend/softdepend 竞技场插件。
- 例外白名单只有 `top.cnolm.golem.api`（BotBody 接口）与 Bukkit/WorldGuard/WorldEdit 这些公开编译依赖。
- 换竞技场插件只改调用方，不改 Gladiator 源码——这是 G-003/G-009 的验收口径。

## 7. 自检记录

| 日期 | 环境 | 结果 | 备注 |
|---|---|---|---|
| 2026-11-13 | `caller-sample` + Paper 1.8.8（runtime / golem 两种后端） | PASS | A1/A2/A4/A5/A6/A8/A9 真机；A3 观测式 deny |
