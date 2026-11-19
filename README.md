# Gladiator

面向 Minecraft 1.8.8 的 **AI 对局 / 训练框架**：一个独立 Spigot 插件对外提供 `GladiatorAPI`，配合 Node.js Bot 运行时、Python Gymnasium 训练环境与本地 Web 控制台，把「真实协议登录的 Bot → 对局 → 训练 → 模型持久化」串成一条可复现的流水线。

- 插件 `0.1.0`，对外 API `1.1.0`，主类 `top.cnolm.gladiator.GladiatorMain`
- Bot 由 mineflayer 真实登录，服务端拿到真实 `CraftPlayer`：Bukkit 事件、WorldGuard 规则、其它保护插件自然生效（**不是** NMS 伪玩家）
- 集成方向固定：竞技场插件 → `GladiatorAPI` → Gladiator。换竞技场插件只改调用方，Gladiator 源码零改动

## 特性

| 能力 | 说明 |
|---|---|
| Bot 生命周期 | `spawnBot(BotSpec)` 返回 `BotRequest`；`BotHandle` 按 botId / UUID / username 三向索引，`asPlayer()` 给出真实玩家 |
| 标准事件 | `GladiatorBotReady/Failed/Removed`、`GladiatorStateChanged`、`GladiatorSessionEnded`，全部是普通 Bukkit 事件 |
| 状态机 | `UserState{ALIVE, DEAD, RESPAWNING, ELIMINATED}`，由死亡 / 重生事件驱动，唯一写入点 `transition()` |
| 规则桥 | `isPvpAllowed` / `queryPvp` 返回 `PvpVerdict{allowed, regionIds, source, box}`；监听器一律 MONITOR、从不取消事件 |
| 通道 A（控制面） | 插件监听 `127.0.0.1:8765`，运行时为 WS 客户端：HELLO / SPAWN / STATE / NOTICE / QUERY_PVP / PING |
| 通道 B（训练面） | 运行时监听 `127.0.0.1:8770+`，训练侧连入：OBSERVATION / ACTION / RESET / STATE，10Hz |
| 观测与动作 | **23 维观测**（血量 / 饱食 / 朝向 / 相对位置 / 对手速度 / 距离 / 受击窗口…）× **11 个宏动作** |
| 默认套件 | RCON 发钻石套 + 钻石剑 + 5 金苹果 + 鱼竿，含装备校验与反应式补装；3 秒准备期不计步 |
| 模型管理 | 持久化训练原子写回（失败不写回）、`.gmod` 热扫描、奖励预设、多场地融合 |
| 观战与监控 | 只读观战（旁观者不进入决策）、Dashboard 起停服务 / 训练 / 人机对战 / 实时终端 / 模型可视化 |

## 两种 Bot 来源

`spawn.mode` 决定 Bot 身体从哪来，对外 API 与事件完全一致：

| 模式 | 身体 | 需要 bot-runtime | 可训练 | 适用 |
|---|---|---|---|---|
| `runtime`（默认） | mineflayer 真实登录 | 是（每 Bot 一个进程） | 是 | 训练、压测、协议验证 |
| `golem` | 独立 Golem 插件在服务端进程内生成的 NMS 身体 | 否 | 否 | 陪练、演出、快速对局 |

golem 为 `softdepend`，未安装时 Gladiator 正常启用、`spawnBot` 返回中文错误；详见 [`docs/golem-integration.md`](docs/golem-integration.md)。

## 架构

```
竞技场插件 ──(Java API + Bukkit 事件, 同 JVM)──▶ Layer 1  Gladiator 插件
                                                  通道 A :8765  ◀── WS ── Layer 2 bot-runtime（mineflayer）
                                                  通道 B :8770+ ◀── WS ── Layer 3 training（Gymnasium）
                                                                              └── RCON :25575 控制训练场
Layer 3 ──HTTP :8080──▶ dashboard（Web 控制台 + 启动器）
```

| 端口 | 归属 | 用途 |
|---|---|---|
| `25566` | 服务端 | 游戏端口（Paper/Spigot 1.8.8，`online-mode=false`） |
| `8765` | 插件 | 通道 A 控制面 |
| `8770+` | bot-runtime | 通道 B 训练面（每 Bot 一个端口） |
| `25575` | 服务端 | RCON：摆位、发套件、回血、状态查询 |
| `8080` | dashboard | Web 控制台 |

所有 WS/HTTP 只绑回环，非本机来源在握手阶段拒绝。传送、套件、地图、匹配、结算归调用方。

## 快速开始

**环境**：JDK 8（编译 / 运行插件）、JDK 17（Gradle daemon，由 `gradle/gradle-daemon-jvm.properties` 固定）、Node ≥ 22、Python ≥ 3.10（本机验证 3.13）。WorldGuard 可选，缺失时规则查询降级为 `DEGRADED`。

```powershell
# 构建插件（含 Java 单元测试）
.\gradlew.bat build --offline
# 产物 build\libs\gladiator-0.1.0.jar → 放进服务端 plugins\

# 依赖
cd training; python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt; cd ..
cd bot-runtime; npm install; cd ..

# 启动控制台（自动打开 http://127.0.0.1:8080/）
.\launcher\GladiatorLauncher.exe
```

启动器把 `dashboard/app.py` 挂到 WMI 宿主进程下，关终端不会带走；服务 / 运行时 / 训练 / 人机对战都在网页里操作。`--status` 查看状态、`--stop` 全部停止、`--no-browser` 只启动。

**游戏内命令**（`gladiator.admin`，别名 `/gl`）：

| 命令 | 作用 |
|---|---|
| `/gladiator status` / `list` / `game` | 版本与计数、Bot 列表、当前对局 |
| `/gladiator spawn <user> [cps=… reaction=… reach=…]` | 拉起 Bot |
| `/gladiator train <user>` | 把就绪 Bot 纳入训练会话 |
| `/gladiator pvp <x> <y> <z>` | 查询该坐标 PVP 判定 |
| `/gladiator spectate [user]` / `unspectate` | 只读观战 / 退出 |
| `/gladiator analyze [session] [window]` | 存活、伤害、规则拒绝、动作统计 |
| `/gladiator respawn <player>` | 强制重生（人机对战的死亡恢复用） |
| `/gladiator reload` | 重载 `config.yml`（启动期参数需重启） |

**跑训练**（Web 训练页等价）：

```powershell
cd training
.venv\Scripts\python.exe -m gladiator_train.train --users Bot_T1 Bot_T2 --ports 8770 8771 `
  --stages scripted:5,rusher:5 --epochs 4 --auto-spawn
```

多场地并行与镜像评估见 `docs/training.md`（`multi_train` / `vec_train` / `eval_suite` / `mirror_eval`）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/main/java/.../api/` | 对外契约与事件，调用方唯一可 import 的包 |
| `src/main/java/.../core/` | 实现：状态机、规则桥、spawn 编排、指标、通道 A |
| `src/main/java/.../command/` | 管理命令 |
| `bot-runtime/` | Layer 2：mineflayer 运行时（通道 A 客户端 + 通道 B 服务端、动作 / 观测） |
| `training/` | Layer 3：`gladiator_env/`（环境、RCON 训练场）+ `gladiator_train/`（策略、训练、评估） |
| `dashboard/` | 本地 Web 控制台（Python 标准库 + 原生 JS） |
| `caller-sample/` | 独立示例调用方，验证「换插件只改调用方」 |
| `launcher/` | `GladiatorLauncher.exe`（C# + WMI） |
| `server/` | 本地 Paper 1.8.8 与服配置（产物与世界不入库） |
| `docs/` | 契约、协议、训练、运维、ADR |

## 测试

| 套件 | 命令 | 通过数 |
|---|---|---|
| Java | `.\gradlew.bat test --offline` | 41 |
| bot-runtime | `cd bot-runtime; node --test` | 122 |
| training | `cd training; .venv\Scripts\python.exe -m unittest discover -s tests` | 424 |
| dashboard | `cd dashboard; ..\training\.venv\Scripts\python.exe -m unittest discover -s tests` | 402 |

数字随开发推进变化，以命令输出为准；Node 测试要在 `bot-runtime/` 目录内跑，Dashboard 测试使用临时目录。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/api-contract.md`](docs/api-contract.md) | **集成唯一入口**：签名、事件、线程约定、调用方职责、版本策略 |
| [`docs/protocol.md`](docs/protocol.md) | Layer 1↔2↔3 协议 v1：信封、帧、错误码 |
| [`docs/training-env-api.md`](docs/training-env-api.md) | 环境 API：`reset` / `step`、观测、动作、奖励、套件、RCON |
| [`docs/training.md`](docs/training.md) | 训练细节：课程、续训、并行、评估口径、报告字段 |
| [`docs/models.md`](docs/models.md) | 模型注册表、`.gmod`、持久化、融合、可视化 |
| [`docs/operations.md`](docs/operations.md) | 启动顺序、日志 / 锁 / trash、排障 |
| [`docs/golem-integration.md`](docs/golem-integration.md) | golem 接入与限制 |
| [`docs/spectating.md`](docs/spectating.md) · [`docs/manual-testing.md`](docs/manual-testing.md) | 观战、真机联调步骤 |

## 已知限制

1. Bot 是外部进程：训练 / 对局期间必须保证运行时与通道 B 端口存活，Dashboard 只托管自己拉起的实例。
2. Bot 目前一律离线登录（`online-mode=false`），与真人共用用户名空间，命名冲突由调用方规避。
3. 动作空间是宏动作：11 个离散动作把「选槽 + 右键 + 按住」写死，新增物品需要加动作并重训。
4. Gladiator 不伪造 `PlayerConnection` / `CraftPlayer`；需要进程内身体时用 `spawn.mode=golem`。
5. 反作弊、权限组、区域放行等环境准备由调用方 `init` 完成，Gladiator 不探测也不配置它们。
6. Windows 优先：进程管理、打开文件夹、启动器为 Windows 实现；其它平台可只跑插件 + CLI。

## 许可

维护：**CnolmLab**。仓库未附带 LICENSE 文件，即未声明开源许可证；引用、分发或商用前请先联系维护者确认授权。
