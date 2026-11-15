# 下一步建议（Stage N+1）

> 写于 2026-11-09，2026-11-10 追加「身份 / 动作空间」两项结构性议题。按「先证明正确、再提升强度、最后扩边界」排序；
> 每条都标了为什么现在做、验收怎么算过。可选做（P2）的部分不影响主线。

## 2026-11-15 更新 · 训练重启后的下一阶段主线

**当前状态（本轮实测）**
- 训练链路已重启并跑通：`spawn.mode: runtime`、Dashboard 8080 托管 Paper 25566 + 2 个运行时 8770/8771，
  运行 `20261115-2025-train-Bot_T1`：`scripted×6 → rusher×6 → aimbot×4`（16 局）、`maxSteps=200`、
  Battle reward 预设、持久化到 `default`、autoSpawn + autoCleanup。
- 双后端可用：`runtime`（mineflayer 真登录，训练用）与 `golem`（进程内身体，`/gladiator spawn` 直接可用）。
- 验收全绿：A1–A9（A3 走观测式 deny、A7 走调用方清单）；Java 65 / golem 73 / node 68 测试通过。

**本轮暴露的问题（P0 优先，直接决定训练是否可信）**
1. `ITEM_MISSING`：`[kit] 套装就绪` 之后仍出现「快捷栏里没有 sword / rod」。疑似 `equipArmor`/`use item` 改选中槽、
   死亡清背包、或 `give` 落到背包而非快捷栏；需要把「每步前置条件」收进环境（出手前确认手持剑）。
2. 评估口径：`POST /api/training/start` 返回的 `evalEpisodes` 被算成 16（等于总回合数），需确认是「评估回合数」还是「评估总对局数」，
   并让 UI/报告只出现一个含义。
3. `OUT_OF_REACH`（距离 14.42 > 3.99）出现在脚本对手身上：说明「超出射程仍发 attack」的路径没被动作层拦住，
   策略会把这类空动作当噪声学进去。

---

### P0 · 让训练「可信」（1–2 天）

| # | 事项 | 验收 |
|---|---|---|
| P0-1 | 训练前置条件硬化：每步 `attack` 前确认手持剑（缺则自动 `slot` 切剑并回 `ACTION_RESULT` 说明）；`_ensure_loadout` 覆盖「回合中途死亡重生」 | 连续 50 局 `ITEM_MISSING` 计数 = 0；复盘里 `attack` 命中率不再是 0 |
| P0-2 | 评估口径统一：一个 `evalRounds`（每阶段）概念，报告/Dashboard/CLI 同名字段 | 启动响应与报告里 `evalEpisodes == evalRounds × 阶段数`，且 UI 只显示「评估回合 × 阶段数」 |
| P0-3 | 超射程动作拦截：`decode_action` 用 `reach` 判定，超出时改为 `approach`（而不是发一个必失败的 attack） | `OUT_OF_REACH` 计数在 10 局内降为 0 |

### P1 · 让策略「变强」（1–2 周）

| # | 事项 | 说明 / 验收 |
|---|---|---|
| P1-1 | 指标先行：固定对手池（`scripted/rusher/aimbot` 各 N 局）的**胜率**作为唯一主指标 | 每轮训练产出「分对手胜率表」，替代 mean reward 作决策依据 |
| P1-2 | 算法升级：REINFORCE → PPO + GAE（rollout 覆盖 2–3 回合），8–32 并行 | 同预算下对 `rusher` 胜率 ≥ 50%（当前 `default` 约 0%） |
| P1-3 | 课程与自博弈：阶段 0（静止假人）→ 3（多风格池）→ 4（历史快照） | 每阶段有进入/退出阈值（命中率 >70% / 击退命中 >40% / 胜率 >55%） |
| P1-4 | 动作空间 v2：正交意图 + `items.json` 能力表 + action mask（见 `docs/identity-and-action-space.md` 第二部分） | 一票判据：加一条 `minecraft:potion` 能力项，不改 Python/JS 即可学会使用 |

### P2 · 让系统「可交付」（1–2 周）

| # | 事项 | 说明 |
|---|---|---|
| P2-1 | 训练-部署一致：给 `bot-runtime` 加 **attached 模式**（只开通道 B、不登录 MC），让训练环境能驱动 golem 身体 | 目前训练必须 runtime 登录，golem 只能手动 `/golem` 验证；打通后「训练用什么身体，部署就用什么身体」 |
| P2-2 | 身份命名空间：`AI_` 前缀 + 占用预检 + `NAME_TAKEN`（`docs/identity-and-action-space.md` 路线 A） | 离线 Bot 不与真人重名；上线前必做 |
| P2-3 | 模型分发：模型颜色/名称由 Gladiator 广播给 `caller-sample` 之类调用方，`.gmod` 热更新 | 换模型不用重启服 |
| P2-4 | CI：GitHub Actions 跑 Java/golem/node/training/dashboard 五套测试 + `verifyStandalone` | 每次 push 自动跑，PR 必过 |

### 本周不做（避免分心）
- 不追 1.12+；不做正版账号池（C1）；不做 GUI 编辑器；不引入第三方 RL 框架（保持零依赖基线）。

---

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

## P0-C · 生产化：让陪练 Bot 合法进正版服（详见 `docs/production-deployment.md`）

现状：只有离线登录（`auth:'offline'`），只能在自有 Paper 里跑；训练侧没问题，**交付侧不成立**。
关键结论：NMS 假玩家不登录、出不了自家服，解决不了「进正版生产服」；唯一合法路径是
**真实客户端 + 正版账号**（mineflayer `auth:'microsoft'` 或插件内 Java 协议客户端）。

1. **C1 · 正版登录打通（1–2 天，最高优先）**：`--auth microsoft` + `profilesFolder` 账号池
   （`config/accounts.json` 不入库）；失败码补 `INVALID_SESSION` / `RATE_LIMITED` / `AUTH_REQUIRED`；
   验收 = `online-mode=true` 服上真实登录、真实 UUID、白名单放行、重连复用 session。
2. **C2 · `.gmod` 进插件（2–3 天）**：Java 侧解 `npz`（zip 内 `.npy`）+ MLP 前向（1995 float），
   与 Python `Policy.forward` 交叉验证（1e-6）；插件扫描 `models/*.gmod`，`/gladiator models` 管理。
3. **C3 · 单 jar 形态（2–4 周，按需）**：若生产服不能跑 Node，用 MCProtocolLib 或自写 1.8.9 协议子集
   把客户端搬进 JVM（仍是真实登录），动作/观测语义与 `bot-runtime` 保持一致。
4. **C+ · NMS 假玩家（可选，需 ADR-018 批准）**：仅自有服、默认关；代价是自实现攻击冷却/击退/reach/暴击。

决策请求：①是否批准 ADR-018（放开 NMS 的唯一例外）；②生产服形态是 A（可跑 Node）/ B（只给 jar）/ C（只自家服）。

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

**先让 Bot 能合法登录正版服（P0-C1）、把身份与 A1–A9 用真机钉死，再换动作空间 v2、
把「胜率」变成唯一优化目标，最后才谈更花哨的模型。**
