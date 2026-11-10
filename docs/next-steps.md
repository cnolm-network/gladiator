# 下一步建议（Stage N+1）

> 写于 2026-11-09，2026-11-10 追加「身份 / 动作空间」两项结构性议题。按「先证明正确、再提升强度、最后扩边界」排序；
> 每条都标了为什么现在做、验收怎么算过。可选做（P2）的部分不影响主线。

## 0. 当前基线（先读这个）

| 维度 | 现状 |
|---|---|
| 功能 | spawnBot → 真实登录 → PlayerJoin 绑定 → `GladiatorBotReadyEvent`；状态机 / 规则桥 / 观战 / 训练 / 持久化模型 / `.gmod` 全通 |
| 训练 | REINFORCE + 19 维观测 + 11 动作；Battle reward 预设调参后仍偏保守（对 scripted 每局 ~2 命中） |
| 模型 | `default`（18 局 / 1495 步）、`release`（+8 局 / 814 步）、`flash`（6 局 / 474 步）三只内置 `.gmod` |
| 测试 | Java 41 / node 68 / training 129 / dashboard 285，全绿 |
| 身份 | 全部 Bot 走**离线登录**（`auth: 'offline'` + `online-mode=false`），用户名与真人共用命名空间，`BotSpec` 只校验格式不校验占用 |
| 动作 | 11 个**宏动作**（`eat_gapple` / `use_rod` / `block` / `switch_sword`），新物品必须加动作 + 重训 |
| 未关闭 | Appendix A 的 **A1–A9**（1.8.8 真机行为实测）；`G-003` 的 `bridge/dto` 抽包（可选） |

结论：**工程闭环已经成立，短板在「身份可信度」「策略强度」和「真机行为覆盖」**。下一步不要急着加新功能。

## P0-A · 身份：离线登录与重名（1 天）

详细方案见 **`docs/identity-and-action-space.md` 第一部分**（四条路线对比 + 落地清单）。要点：

1. **命名空间**：`bot.name-prefix`（默认 `AI_`），`BotSpec` 校验前缀，命令与 Dashboard 表单统一走同一校验。
2. **占用检查前置**：`BotSpawnService.spawn()` 在下发 SPAWN 前检查
   `getPlayerExact` / `getOfflinePlayer(name).hasPlayedBefore()` / `usercache.json`，
   命中直接派发 `GladiatorBotFailedEvent`，reason 前缀 `NAME_TAKEN: `，新增同名 `SPAWN_FAILED` code。
3. **登录期兜底**：`describeLoginFailure` 补 `already logged in` / `name is already taken`；
   离线 Bot 撞 `online-mode=true` 时，提示同时给出「正版账号池」与「代理转发」两条出路。
4. **不做自动改名重试**：改名会让 usercache、统计、观战链接全部错位；失败就交给调用方决定。

长期若需要「真人与 Bot 同服」：走代理转发（Velocity/BungeeCord，UUID 由代理分配）或正版账号池
（`auth: 'microsoft'`，每个 Bot 一个账号）。两者都不需要改 `api` 或 `BotHandle` 的三向索引。

## P0-B · 动作空间：从「语义动作」改成「能力 + 槽位」（3–5 天，可与 P1 并行）

详细方案见 **`docs/identity-and-action-space.md` 第二部分**。要点：

- 动作拆成正交意图：`move / look(dyaw,dpitch) / attack / use(单击|按住) / jump / sneak / sprint / slot(0-8) / drop`，
  **物品身份只进观测**（9 槽 × category one-hot + count + durability + cooldown），新增物品零改代码；
- 物品语义放 `training/gladiator_env/items.json` 能力表（category / use.mode / ticks / cooldown / reach），
  支持正则匹配，未知物品落 `misc`；
- 由能力表 + 背包状态算 **action mask**（没物品不 `use`、超 reach 不 `attack`、空槽不 `slot`）；
- `--action-space v1|v2` 并存：v1 保留作冷启动基线与回归对照，模型元数据记录 `actionSpace`，载入不匹配直接报错；
- **一票判据**：往 `items.json` 加一条药水、训练场 `give` 药水，不改任何 Python/JS 代码就能让 Bot 学会用它。

## P0 · 用真机把 A1–A9 关掉（1–2 天）

`docs/feasibility.md` 与 `TASKS.md` 附录 A 里的九项都还是「待实测」。它们决定的是**协议语义是否真的成立**，
一旦某条错了，后面的训练数据全部要重跑。


| # | 验收动作 | 通过标准 |
|---|---|---|
| A1 | 对墙/对地放方块、六个朝向各 3 次 | 方块落在预期坐标，`place` 的 `against+face` 与客户端一致 |
| A2 | 连续吃金苹果、挥鱼竿、格挡 | `use` 时序无错位；金苹果 1.6s 内不回血视为失败 |
| A3 | 在 WG 保护区边缘 PVP | `DisallowedPVPEvent` 触发一次、Bot 收到一次 `RULE_DENIED`，**事件未被取消** |
| A4 | 10 CPS 连点 + 移动 | 服务端不踢人、`ACTION_RESULT` 全部 `ok` |
| A5 | 跨区域寻路 40 格 | pathfinder 不卡死；deny 区域确实绕行 |
| A6 | 竞技场插件取消 `PlayerDeathEvent` | Gladiator 状态机**仍**从死亡事件迁移，不读插件状态 |
| A7 | 服务端重启后运行时自动重连 | ≤30s 内重新 HELLO，`BotHandle` 仍可 pushGame |
| A8 | 训练中 `kill` 掉一个 runtime | ≤30s 心跳回收 session，不踢玩家实体 |
| A9 | 双 Bot 同账号/同名 | 后连接者替换旧会话，不产生两个 `BotHandle` |

做法：直接复用 `docs/manual-testing.md` 的清单 + `training/scripts/chaos.py` 的故障注入；
每项通过后在 `TASKS.md` 附录 A 勾掉并写一条变更日志。**这一批不需要写新代码**，只补脚本与记录。

## P1 · 把策略从「会打」推到「能赢」（主线，1–2 周）

现在 reward 与胜率脱节（mean reward 高但命中少），三个改动按顺序做：

### 1. 观测/动作升级为「战斗状态机」

- 动作部分已单列到 **P0-B**（`docs/identity-and-action-space.md` 第二部分）：v2 是正交意图 + 能力表 + action mask，
  v1 的 11 个宏动作保留为冷启动基线。本节的剩余工作是把 v2 的动作/观测接进 PPO 主线。
- 观测（19 → ~28 维，叠加 P0-B 的 9 槽背包段）：加 `self_velocity`、`opponent_velocity`、`self_sprint/on_ground`、
  `自上次命中/受击的 tick 数`、`连击数`、`是否在攻击后 1.5s 窗口内`。参考 `D:\downloads\bot-training.md` 的状态向量表。

### 2. 训练目标从「单局 reward」改成「对固定对手池的胜率」

- 对手池：`scripted`（70% 进攻）、`rusher`、`aimbot`、`wander` 各占一档，每 200 局轮换一次，记录**分对手胜率**。
- 早停与选模型用「对池胜率」而不是 reward；reward 只用来观察塑形是否健康。
- 引入 GAE + 多回合 rollout（PPO），或先把 REINFORCE 的 baseline 从「批次均值」换成「价值网络」。
  纯 REINFORCE 在 11 动作 + 长回合上方差太大，这是当前「训练很久没起色」的主因之一。

### 3. 奖励按参考文档重排，别再叠「万金油」项

- 伤害:承伤 ≈ 2.1:1；连击封顶、单步塑形总和 ≤ 击杀奖励 1/3；间距奖励只放在攻击后窗口（现在 `pressure` 是常开的，容易养出「贴脸不砍」）。
- 空挥惩罚不要超过 `-0.1`，且**超出 reach 的出手不算空挥**（已修，保持）。
- 每加一项塑形，跑 200 局看「命中率 / 空挥率 / 平均距离 / 受伤比」四项，涨了才留。

验收：对 `scripted` 胜率 > 55%，对 `aimbot` > 40%，命中率 > 45%，平均每局出手 > 15 次。

## P2 · 值得做但不阻塞主线

- **模型颜色真正下发**：`.gmod` 的 `color` → `dashboard/var/scripts.json` → `caller tagcolor`，现在只到「元数据展示」。
- **`bridge/dto` 抽包**（关闭 G-003）：协议已稳定三个月，抽 `bridge/dto/*` 让调用方也能复用帧定义；纯重构，风险低。
- **Linux/macOS 支持**：`launcher` 与 `dashboard` 的进程管理是 Windows 实现；抽 `ProcessBackend` 接口即可，别改业务层。
- **CI**：GitHub Actions 跑 `gradlew test` + `node --test` + 两套 Python 测试；Dashboard 测试要清空 `models/` 才是确定性结果（README 已注明）。
- **发布节奏**：每完成一个阶段跑一次 `scripts/publish.ps1`，别攒大版本；公开仓库不要带 `dashboard/var`、`training/artifacts`。

## 不建议现在做

- 上 CNN / 视觉输入：状态是结构化的，CNN 只会更慢更难调。
- 自博弈快照池：单机 4 场地跑到 45–55% 胜率之前，自博弈只会放大噪声。
- 把训练搬进插件 JVM：Layer 3 现在能独立重启、独立升级，塞进 Bukkit 会毁掉这个边界（也是 ADR 的初衷）。

## 一句话

**先把身份（命名空间 + 占用检查）和 A1–A9 用真机钉死，再把动作空间换成 v2、把「胜率」变成唯一优化目标，
最后才谈更花哨的模型。**
