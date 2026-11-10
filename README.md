# Gladiator

**面向 Minecraft 1.8.8（协议 47）的 AI 对局 / 训练框架**：一个独立的 Spigot 插件对外提供 `GladiatorAPI`，配合 Node.js Bot 运行时、Python Gymnasium 训练环境和本地 Web Dashboard，把「真实协议登录的 Bot → 对局 → 训练 → 持久化模型」串成一条可复现的流水线。

- 插件 `0.1.0`，对外 API `1.1.0`，主类 `top.cnolm.gladiator.GladiatorMain`
- **不是** NMS 伪玩家，也**不依赖**任何竞技场插件：Bot 由 mineflayer 真实登录，服务端拿到的是货真价实的 `CraftPlayer`，Bukkit 事件、WorldGuard 规则、其它保护插件全部自然生效
- 集成方向：竞技场插件 → 调用 `GladiatorAPI` → Gladiator。换竞技场插件只改调用方，Gladiator 源码零改动

---

## 能力清单

| 能力 | 说明 | 参考 |
|---|---|---|
| Bot 生命周期 | `spawnBot(BotSpec)` → `BotRequest`；`BotHandle` 三向索引（botId / UUID / username），`asPlayer()` 返回真实 `CraftPlayer` | `docs/api-contract.md` §3.4 |
| 标准事件 | `GladiatorBotReadyEvent` / `GladiatorBotFailedEvent` / `GladiatorBotRemovedEvent` / `GladiatorStateChangedEvent` / `GladiatorSessionEndedEvent`，均为标准 Bukkit 事件 | `api/event/` |
| 状态机 | 自有 `UserState`（ALIVE / DEAD / RESPAWNING / ELIMINATED），由 `PlayerDeathEvent` / `PlayerRespawnEvent` / `eliminate()` 驱动，唯一写入点 `transition()` | `core/UserStateMachine` |
| WorldGuard 只读规则桥 | `isPvpAllowed` / `queryPvp` 返回 `PvpVerdict{allowed, regionIds, source, box}`；未装 WorldGuard 时 `DEGRADED`；所有监听器为 MONITOR 且**从不取消事件** | `core/guard/*` |
| 通道 A（控制面） | 插件监听 `127.0.0.1:8765`，运行时作为 WS 客户端；HELLO / WELCOME / SPAWN / STATE / NOTICE / QUERY_PVP / PING | `docs/protocol.md` §2 |
| 通道 B（训练面） | 运行时监听 `127.0.0.1:8770+`，Python 连入；OBSERVATION / ACTION / RESET / IDLE，控制频率 10Hz | `docs/protocol.md` §4 |
| 观测与动作 | **19 维观测**（血量/饱食/朝向/相对位置/对手状态/耐久/苹果数…）× **11 个离散动作**（`idle` `approach` `attack` `strafe_left/right` `retreat` `jump` `use_rod` `eat_gapple` `block` `switch_sword`） | `training/gladiator_env/spaces.py` |
| 默认套件 | `give_kit=True` 时每局用 RCON 发钻石套 + 钻石剑 + 5 金苹果 + 鱼竿，并做装备校验（4/4 护甲、手持剑）；3 秒准备期不计步数 | `docs/training-env-api.md` §4.6 |
| 持久化训练 | 训练**正常完成**后把最终权重原子写回所选模型（累加局数/步数/均值），失败或停止不写回；带模型写锁与 `models-trash/` 备份 | `docs/models.md` §9 |
| Reward 预设 | `gladiator.reward-preset/v1`，Web 可编辑基础权重与条件项；训练与多场地都支持 `--reward-preset` | `docs/training-plan-ui.md` §6 |
| `.gmod` 模型包 | zip{`meta.json`,`policy.npz`}；放进 `models/` 刷新即出现、坏包不打断列表；Web 可导出（含训练统计与 `rewardPreset`） | `docs/models.md` §8 / §12 |
| 多场地融合 | `multi_train --merge mean/weighted/none`，融合出唯一 `policy.npz` 后再持久化写回 | `docs/models.md` §10 |
| 只读观战 | 玩家用自己的客户端旁观训练，切 SPECTATOR、退出还原；旁观者不参与对局，也不污染 `nearest_player` 决策 | `docs/spectating.md` |
| Dashboard + 启动器 | 零第三方依赖的本地 Web 控制台：起停 Paper/runtime、跑训练、人机对战、实时终端、产物与模型可视化、模型导出 | `dashboard/README.md` |

---

## 架构

```
调用方（竞技场插件）
  └─ spawnBot / pushGame / eliminate         ← Java API + 标准 Bukkit 事件（同 JVM，主线程）
       │
Layer 1  Gladiator 插件（Java 8 / Spigot 1.8.8）
  ├─ 对外 GladiatorAPI + 5 个标准事件
  ├─ 状态机 / WorldGuard 只读规则桥 / spawn 编排 / 指标快照
  └─ 通道 A 服务端 127.0.0.1:8765
       │ WS :8765（通道 A）                        │ RCON :25575（训练控制面）
Layer 2  bot-runtime（Node ≥22 + mineflayer）      Layer 3  training（Python ≥3.10 + Gymnasium）
  ├─ 通道 A 客户端  ◀── 通道 B :8770+ ──▶          ├─ GladiatorEnv / 19 维观测 / 11 动作 / 奖励
  ├─ 动作执行 / 寻路 / 自动穿甲 / 观测上报          └─ REINFORCE 训练循环 / 报告 / 模型产物
  └─ MC 协议 :25566 ──▶ 游戏服务端
                                                     │ HTTP :8080
                                            dashboard（Web 控制台 + 启动器）
```

| 端口 | 归属 | 用途 |
|---|---|---|
| `25566` | 服务端 | Minecraft 游戏端口（Paper/Spigot 1.8.8，`online-mode=false`，可在 `server.properties` 改） |
| `8765` | Gladiator 插件 | 通道 A：控制面（spawn / 状态 / 通知 / 规则查询），只绑 `127.0.0.1` |
| `8770+` | bot-runtime | 通道 B：训练面（观测 / 动作 / 重置），每个 Bot 一个端口；Dashboard 默认从 8770 起，`multi_train --ports-base` 默认 8766 |
| `25575` | 服务端 | RCON：训练环境摆位、发套件、回血、查询（仅本地联调） |
| `8080` | dashboard | Web 控制台，只绑回环地址 |

> 所有 WS/HTTP 绑定强制回环，非本机来源在握手阶段拒绝。传送、套件、地图、匹配、结算属于调用方职责，Gladiator 不碰。

---

## 快速开始

### 1. 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| JDK（编译/运行插件） | **8** | 产物 bytecode 52；Paper 1.8.8 不能用 17/21 启动 |
| JDK（运行 Gradle） | **17** | 由 `gradle/gradle-daemon-jvm.properties` 固定，优先于 `JAVA_HOME`，无需手动切换 |
| Gradle | **8.10.2** | 仓库自带 wrapper |
| Node.js | **≥ 22** | `bot-runtime/package.json` 的 `engines` 要求（mineflayer 4.39） |
| Python | **≥ 3.10**（本机验证 3.13） | 训练环境与 Dashboard；Dashboard 零第三方依赖 |
| WorldGuard | 6.2（可选） | 缺失时规则查询降级为 `DEGRADED` |

### 2. 构建插件

```powershell
.\gradlew.bat build --offline
# 产物：build\libs\gladiator-0.1.0.jar → 放进服务端 plugins\
# build 会同时跑 Java 单元测试
```

示例调用方（可选，用于验证「换插件只改调用方」）：

```powershell
.\gradlew.bat :caller-sample:jar --offline
Copy-Item caller-sample\build\libs\caller-sample.jar server\plugins\ -Force
```

### 3. 启动控制台

```powershell
.\launcher\GladiatorLauncher.exe
```

启动器通过 WMI 把 `pythonw dashboard/app.py` 挂到 `WmiPrvSE.exe` 下，关终端 / IDE 不会带走进程；浏览器自动打开 <http://127.0.0.1:8080/>，之后启停 Paper、起停 runtime、跑训练、人机对战都在网页里完成。

```powershell
.\launcher\GladiatorLauncher.exe --no-browser   # 只启动，不开浏览器
.\launcher\GladiatorLauncher.exe --status       # 查看是否在跑
.\launcher\GladiatorLauncher.exe --stop         # 停止（含托管的 Paper / runtime / 训练）
```

> 手工方式：`cd server; & "$env:JAVA_HOME\bin\java.exe" -Dfile.encoding=UTF-8 -jar paper-1.8.8-445.jar nogui`，再为每个 Bot 起一个 `node bot-runtime/src/index.js --username ... --env-port 8770`。手工进程不归 Dashboard 托管（页面显示「外部进程（只读）」）。

### 4. 进服与常用命令

游戏内（需要 `gladiator.admin`，别名 `/gl`）：

| 命令 | 作用 |
|---|---|
| `/gladiator status` | 版本、api、ready、WorldGuard、bots / games / training / sessions、堆内存 |
| `/gladiator list` | Bot 列表（botId / 名字 / UUID / ready / state） |
| `/gladiator spawn <user> [cps=… reaction=… reach=…]` | 拉起一个 Bot |
| `/gladiator train <user>` | 把已就绪 Bot 纳入训练会话 |
| `/gladiator game` | 当前对局 / 训练会话与参与者 |
| `/gladiator analyze [session] [window]` | 存活时长、伤害、方块、规则拒绝、动作统计 |
| `/gladiator pvp <x> <y> <z>` | 查询该坐标的 PVP 判定 |
| `/gladiator spectate [user]` / `unspectate` | 只读观战 / 退出观战 |
| `/gladiator reload` | 重载 `config.yml`（启动期参数需重启） |

调用方示例插件另有 `/caller arena|duel|tagcolor|…`，见 `docs/caller-sample.md`。

### 5. 跑一次训练

**Web（推荐）**：Dashboard → 训练页 → 选计划（对手模式 + 回合）→ 选模型 →「持久化训练」默认开启 → 开始。

**CLI（等价）**：

```powershell
cd training
.venv\Scripts\python.exe -m gladiator_train.train `
  --users Bot_T1 Bot_T2 --ports 8770 8771 `
  --stages scripted:5,rusher:5 --epochs 4 --auto-spawn
```

多场地（N 组并行 + 权重融合）：

```powershell
python -m gladiator_train.multi_train --pairs 2 --users-prefix Bot_P --ports-base 8770 `
  --stages idle:2,rusher:2 --merge mean
```

从已有权重续训 / 指定 reward 预设：

```powershell
python -m gladiator_train.train --users Bot_T1 Bot_T2 --ports 8770 8771 `
  --init-policy dashboard\var\models\default\policy.npz `
  --reward-preset dashboard\var\reward-presets\default.json --auto-spawn
```

首次准备依赖：

```powershell
cd training
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..\bot-runtime
npm install
```

---

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/main/java/top/cnolm/gladiator/` | 插件源码：`api/`（对外契约 + 5 个事件，唯一可被调用方 import 的包）、`core/`（实现、状态机、规则桥、spawn 编排）、`bridge/`（通道 A）、`command/`、`example/`（示例 Mode 与调用方） |
| `src/main/resources/` | `plugin.yml` / `config.yml`（`bridge.port`、`bridge.token`、`bridge.spawn-timeout-seconds`、`spectate.enabled`） |
| `src/test/java/` | Java 单元测试（JUnit 4）：链路层、注册表、状态机、规则判定、统计 |
| `bot-runtime/` | Layer 2：mineflayer 运行时（通道 A 客户端 + 通道 B 服务端、动作/观测、寻路、自动穿甲） |
| `training/` | Layer 3：`gladiator_env/`（Gymnasium 环境、观测/动作/奖励、RCON 训练场）+ `gladiator_train/`（策略、训练循环、CLI） |
| `dashboard/` | 本地 Web 控制台（Python 标准库 + 原生 JS）：服务/运行时/训练/对战/终端/产物/模型/reward 预设/脚本注册表 |
| `caller-sample/` | 独立示例调用方插件，验证「换插件只改调用方」 |
| `launcher/` | `GladiatorLauncher.exe`（C# + WMI）与 `build.ps1` |
| `server/` | 备用 Paper 1.8.8 与本地服配置（产物与世界不进版本库） |
| `models/` | `.gmod` 模型包热扫描目录（放文件即出现） |
| `docs/` | 契约、协议、训练、运维、决策记录（见下方索引） |
| `scripts/` | `publish.ps1`（白名单导出到公开仓库）等工具脚本 |

---

## 测试

| 套件 | 命令 | 当前通过数 |
|---|---|---|
| Java（Gradle/JUnit） | `.\gradlew.bat test --offline` | **41**（LinkServer 9 / BotRegistry 7 / GameRegistry 4 / BotProfile 5 / PvpVerdict 5 / UserStateMachine 5 / AnalyzeService 3 / StatsBoard 3） |
| bot-runtime（Node） | `cd bot-runtime; node --test` | **68** |
| training（Python） | `cd training; .venv\Scripts\python.exe -m unittest discover -s tests` | **129** |
| dashboard（Python） | `cd dashboard; ..\training\.venv\Scripts\python.exe -m unittest discover -s tests` | **285** |

- 数字为 2026-11-09 实测（Java 用 `--rerun-tasks` 强制重跑）。测试数随开发推进变化，以命令输出为准。
- Dashboard 测试全部使用临时目录；若仓库根 `models/` 里已有 `.gmod`，`test_viz` 会读取真实扫描目录导致个别断言随环境变化，先清空该目录可得到稳定结果。
- 前端无构建步骤：改 `dashboard/static/` 刷新即生效，`node --check dashboard/static/app.js` 可做语法检查。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/api-contract.md` | **集成唯一入口**：API 签名、事件、线程约定、调用方 init 职责、版本策略 |
| `docs/protocol.md` | Layer 1↔2↔3 协议 v1：信封、通道 A/B 帧、错误码、背压 |
| `docs/observation.md` | 通道 B 观测 / 动作的字段级 schema（19 维向量、11 个动作与协议帧映射） |
| `docs/training-env-api.md` | 训练环境 API：`GladiatorEnv.reset/step`、观测、动作映射、奖励、默认套件、RCON 训练场 |
| `docs/training-plan-ui.md` | 训练计划 / 预设 / reward 预设的 UI 与接口口径（含多场地边界） |
| `docs/scripting.md` | 脚本与颜色现状、`.gmod` 装载、阶段规划 |
| `docs/models.md` | 模型注册表、`.gmod` 格式与热扫描、持久化训练、多场地融合、可视化、导出与可移植性 |
| `docs/spectating.md` | 只读观战：命令、权限、行为，以及为什么不会干扰训练 |
| `docs/operations.md` | 端口与启动顺序、Java 探测、打开文件夹、日志/锁/trash、排障表 |
| `docs/manual-testing.md` | 真机联调步骤与期望日志 |
| `docs/caller-sample.md` | 示例调用方验收（`/caller arena`、`/caller duel`） |
| `docs/training.md` | 训练细节：课程、续训、奖励、报告字段 |
| `docs/ENVIRONMENT.md` | 本机构建环境（JDK/Gradle/Node/Python 版本与坑） |
| `docs/decisions.md` | ADR 决策记录（只追加） |
| `docs/feasibility.md` | 可行性验证与风险清单（含 A1–A9） |
| `docs/integration-ultrapvp.md` | 参考调用方的只读勘察结论（不是依赖） |
| `docs/identity-and-action-space.md` | 身份（离线登录/重名/正版/代理）与动作空间 v2（能力表 + 槽位/使用分离）方案 |
| `ARCHITECTURE.md` | 四层划分、进程拓扑、关键数据流、状态机 |
| `HANDOVER.md` / `TASKS.md` / `PROJECT_PLAN.md` | 交接记录、任务台账、验收计划（**内部文档，不发布**） |

---

## 发布到公开仓库

只发布产品文件，内部 Agent / 流程文档一律不出库：

```powershell
.\scripts\publish.ps1            # 白名单导出到独立目录后提交并推送
.\scripts\publish.ps1 -NoPush    # 只导出、不推送
```

- **白名单**：`README.md`、构建文件与 wrapper、`src/main/java`、`src/main/resources`、`src/test/java`、`docs/api-contract.md`、`docs/protocol.md`、`bot-runtime`（`package*.json` / `config.example.json` / `README.md` / `scripts`）、`training`（`requirements.txt` / `README.md`）
- **硬排除**：`AGENTS.md`（含子目录）、`HANDOVER.md`、`TASKS.md`、`PROJECT_PLAN.md`、`ARCHITECTURE.md`、`handover-skill/`、`.workbuddy/`、`.idea/`、`server/`、`docs/{ENVIRONMENT,feasibility,decisions,integration-ultrapvp}.md`
- **令牌**：仓库地址只从 Windows 用户级环境变量 `GIT_REPO` 读取，通过 `GIT_CONFIG_*` 注入 git 子进程；不写入 URL、`.git/config`、日志或任何文件
- **禁止 `--force`**：推送前做白名单校验、Agent 文件扫描与明文密钥扫描，异常直接中止
- 导出目录与主仓库分离：主仓库不配置 remote，避免误推内部文件

---

## 已知限制

1. **Bot 是外部进程**：bot-runtime 独立于服务端 JVM，训练/对局期间必须保证进程与通道 B 端口存活；Dashboard 只托管它自己拉起的实例。
2. **Bot 目前一律离线登录**：`auth: 'offline'` + 服务端 `online-mode=false`，因此 Bot 与真人共用用户名空间，`BotSpec` 只校验格式、不校验占用（与在线玩家/历史玩家重名会被顶号或被踢）。命名空间、占用前置检查与正版/代理两条长期路线的方案见 `docs/identity-and-action-space.md`。
3. **动作空间是「宏动作」**：11 个离散动作把「选槽 + 右键 + 按住」写死，新增物品要加动作并重训；物品无关的 v2 动作空间（能力表 + `slot`/`use` 正交 + action mask）同见 `docs/identity-and-action-space.md`。
4. **不做 NMS 伪玩家**：不伪造 `PlayerConnection` / `CraftPlayer`，每个 Bot 都是真实协议登录；也因此没有「离线假人」这类捷径。
5. **反作弊与环境准备归调用方**：权限组、区域放行、白名单、反作弊豁免都在调用方 `init` 阶段完成；Gladiator 源码、配置与默认值中不出现任何反作弊插件名或权限节点。
6. **A1–A9 待真机实测**：方块放置朝向、物品使用时序、WG 事件时机、高频攻击、pathfinder 代价、竞技场插件取消死亡/重生事件时状态机是否仍迁移等，仍需在真实服上逐项验收（见 `docs/feasibility.md`、`TASKS.md` §A）。
7. **训练是单智能体 + 脚本对手**：当前学到的是「打脚本/随机对手」的策略；自博弈与真实人类对手需要扩展 Trainer。
8. **模型颜色尚未下发**：`.gmod` 的 `color` 目前仅作元数据展示，Bot 名字前缀颜色仍由 `dashboard/var/scripts.json` + 调用方 `tagcolor` 决定。
9. **Windows 优先**：Dashboard 的进程管理、打开文件夹、启动器为 Windows 实现；其它平台可只跑插件 + CLI 训练。

---

## 许可证与贡献者

- 维护：**CnolmLab**
- 仓库当前**未附带 LICENSE 文件**，即未声明开源许可证；如需引用、分发或商用，请先联系维护者确认授权。
- 问题与改动建议请通过 Issue / Pull Request 提交；涉及协议或对外 API 的改动必须同步 `docs/protocol.md`、`docs/api-contract.md` 并追加 ADR（`docs/decisions.md` 只追加不回改）。
