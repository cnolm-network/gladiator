# 生产化部署：让陪练 Bot 合法进入正版服务器

> 状态：方案文档（2026-11-10）。对应 ADR-018（提议中）。
> 本文回答一个问题：**训练完之后，怎么把 `.gmod` 变成一个能进 `online-mode=true` 生产服的陪练 Bot。**

## 0. 结论先行

1. **NMS 假玩家不解决「合法登录」**。它不登录——它直接在你的服务端 JVM 里造一个实体，
   没有网络会话、没有 Mojang 认证。因此它只能在**你自己拥有、且装了本插件**的服务端里跑；
   想在别人的正版服上当陪练，它一步也走不出去。
2. **进正版服的唯一合法路径是「真实客户端 + 正版账号」**。mineflayer 的 `auth: 'microsoft'`
   就是干这个的：走微软 OAuth，拿真实 UUID 与 session，和玩家开原版客户端没有任何区别。
3. 所以架构调整不是「改用 NMS」，而是**把「真实客户端」从"开发期工具"提升为"交付物的一部分"**，
   并解决它的两个工程问题：**谁来跑**（Node 进程 vs 插件内 Java 客户端）、**账号从哪来**（账号池）。

## 1. 三种交付形态

| | A · Node 边车（现状延伸） | B · 插件内 Java 协议客户端 | C · NMS 假玩家 |
|---|---|---|---|
| 登录方式 | 真实微软账号（合法） | 真实微软账号（合法） | **不登录**（服务端内部造实体） |
| 能进他人正版服 | ✅ | ✅ | ❌（必须对方装插件） |
| 部署形态 | 服务端宿主跑 Node + 插件 | **单 jar**，零外部进程 | 单 jar |
| 反作弊视角 | 正常玩家 | 正常玩家 | 无客户端，可能被判异常 |
| 战斗保真 | 原版客户端物理/冷却 | 需自写 1.8.9 客户端物理 | 需自写攻击冷却/击退/reach/暴击 |
| 工程量 | 已有（训练那套直接复用） | 中（协议子集 + 客户端物理） | 中（但保真度风险最高） |
| 与现有约束 | 兼容 | 兼容 | **冲突**（ADR/AGENTS 禁止 NMS 伪造 PlayerConnection） |

**关键判断**：C 的「省事」是假象。假玩家没有客户端，攻击冷却、击退、连击、暴击、reach 全部要你在
服务端重新实现一遍——这恰恰是 PVP 训练里最难的物理；而且它永远出不了自家服。
只有在「完全自有、且愿意接受保真度折损」的场景下，C 才值得作为**可选模块**存在。

## 2. 推荐路线（三阶段，每阶段可独立交付）

### P0-C1 · 正版登录打通（1–2 天，最高优先）

- `bot-runtime` 增加 `--auth microsoft`（`mineflayer.createBot({ auth: 'microsoft', profilesFolder })`），
  首次登录走设备码/浏览器授权，之后复用 `profilesFolder` 里的缓存，避免反复触发限流。
- 账号池：`bot-runtime/config/accounts.json`（**不入库**，字段 `username / profilesFolder / enabled`），
  运行时按用户名取对应缓存目录；一个账号同时只能有一个会话。
- 配置化：`mc.auth: offline|microsoft`、`mc.profilesFolder`，默认 `offline`（开发环境不变）。
- 验收：`online-mode=true` 的测试服上，用真实账号登录成功、`playerUuid` 为真实 UUID、
  能被 `/whitelist` 正常放行、断线重连后 session 复用不报 `invalid session`。
- 失败码映射补：`INVALID_SESSION`（缓存失效，需重新授权）、`RATE_LIMITED`（微软限流，退避）、
  `AUTH_REQUIRED`（无缓存且无法交互式授权）。

### P0-C2 · `.gmod` 进插件（2–3 天，可与 C1 并行）

目的：让「模型」成为插件的一部分，而不是只活在 `dashboard/var/models/`。

- 插件新增 `cnolm.gladiator.policy` 包：`.gmod`（zip：`meta.json` + `policy.npz`）解析 + 纯前向推理。
  - `npz` 就是 zip 里的 `.npy`：magic `\x93NUMPY` + 版本 + header 长度 + Python dict 文本
    （`descr '<f4' / shape (64,19) / fortran_order False`），Java 解析约 100 行，无第三方依赖。
  - 前向：`h = tanh(W1·x + b1)`，`logits = W2·h + b2`，`argmax`。**19×64 + 64 + 64×11 + 11 = 1995 个 float（约 8 KB）**，
    单次推理微秒级，主线程调用无压力。
  - 与 Python 侧交叉验证：同一 `policy.npz` + 同一观测向量，Java 与 `Policy.forward` 输出概率差 < 1e-6。
- 插件启动时扫描 `plugins/Gladiator/models/*.gmod`，注册到 `/gladiator models`
  （id / name / obsDim / actDim / sha256 / 来源训练量），选定后写入 `config.yml` 的 `bot.model`。
- 此时「推理」在插件内，「观测/动作执行」仍可走通道 B 给外部客户端——**为 C3 铺路，但不依赖 C3**。

### P0-C3 · 单 jar 形态（2–4 周，按需）

若生产服不允许跑 Node 进程（面板服常见），把客户端搬进 JVM：

- 方案 ①：**MCProtocolLib**（Java，支持 1.8.x）作为客户端库，shade 进插件，真实登录 + 真实物理由库承担；
- 方案 ②：自写 1.8.9 协议子集（login / play / position / entity / block / chat / use / attack），
  工作量最大但完全可控。
- 客户端逻辑直接平移 `bot-runtime` 的 `actions.js` / `observation.js` / `scheduler.js` 语义，
  保持「同一套动作与观测定义」，训练产物不需要重新适配。
- 验收：单 jar 启动 → 用正版账号登录 → 与真人玩家对打一局 → 插件内推理驱动，无外部进程。

### 可选 · C+（NMS 假玩家，需 ADR-018 批准）

- 仅在 `mode: server-agent` 且显式开启时启用，**默认关闭**，且明确「只允许自有服」。
- 需要补齐：攻击冷却（1.8.9 `attackStrengthTicker`）、击退向量、reach 判定、暴击条件、
  以及与反作弊的兼容性说明。这些是「自己实现一遍原版战斗」，不是配置项。

## 3. 账号从哪来（这是比代码更硬的约束）

- **一个账号 = 一个并发 Bot**。多 Bot 训练/陪练需要多个正版账号（或代理转发，见下）。
- 微软 OAuth 有设备码频率限制；缓存 `profilesFolder` 后日常启动不再触发授权。
- 合规：用自己购买的正版账号运行自动化客户端本身不违反 Mojang EULA（不是作弊客户端），
  但**服务器规则**可能禁止 Bot——陪练 Bot 应部署在自有服或已获授权的服务器上。
- 若不想买账号：加一层 **Velocity/BungeeCord 代理 + 代理侧认证插件**（`online-mode=false` 仅对代理内网开放），
  UUID 由代理下发。这属于部署形态变化，不影响插件代码。

## 4. 对训练侧的影响

- **训练不需要改**：训练继续用离线账号 + 自有 Paper，成本最低、迭代最快。
- **交付需要改**：在 `docs/next-steps.md` 的发布物里，把「陪练包」定义为
  `gladiator.jar + models/*.gmod + 账号配置（不入库）`，并把 `auth` 作为部署文档的一等公民。
- 模型与动作空间的解耦（P0-B）在这里变得更值钱：生产服上的陪练会用各种物品，
  宏动作空间撑不住，v2（能力表 + slot/use）是生产化的前置条件。

## 5. 需要你拍板的两件事

1. **是否批准 ADR-018**（放开「NMS 伪造 PlayerConnection」的唯一例外：仅自有服、可选模块、默认关）。
   不批准也不影响 P0-C1/C2/C3，只是没有 C+ 这条路。
2. **生产服的形态**：A（能跑 Node）/ B（只给 jar）/ C（只在自己服内跑、接受保真度折损）。
   这决定 C3 与 C+ 的优先级；C1/C2 三条路都要做，先做它们不会浪费。
