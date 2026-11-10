# 运维手册（本地 / 小型训练场）

面向"自己起一套环境跑 Bot 训练"的运维说明：启动顺序、端口、配置项与排障表。
架构与协议细节见 [`ARCHITECTURE.md`](../ARCHITECTURE.md) 与 [`protocol.md`](protocol.md)。

## 1. 拓扑与端口

```text
        ┌──────────────────────────── Paper 1.8.8 (Java 8) ───────────────────────────┐
Minecraft│ Gladiator 插件                                                              │
客户端 ──▶│  25565 游戏端口                                                             │
(观战)    │  8765  通道 A：插件监听，bot-runtime 作为客户端（HELLO/SPAWN/BOT_BOUND…）    │
        │  25575 RCON（仅本地联调；生产可关）                                          │
        └────────────────────────────────────────────────────────────────────────────┘
              ▲ 通道 A                      ▲ 通道 B（每个 Bot 一个端口）
              │                             │
     ┌────────┴─────────┐          ┌────────┴──────────┐
     │ bot-runtime × N  │          │ training（Layer 3）│
     │ mineflayer 客户端 │◀────────▶│ GladiatorEnv       │
     └──────────────────┘  8766+    └───────────────────┘
```

| 端口 | 归属 | 用途 | 默认 |
|---|---|---|---|
| 25565 | Paper | 游戏端口 | `server.properties: server-port` |
| 8765 | Gladiator | 通道 A（控制面：spawn/状态/通知/规则查询） | `config.yml: bridge.port` |
| 8766, 8767, … | bot-runtime | 通道 B（观测/动作），每个 Bot 一个 | `--env-port` / `GLADIATOR_ENV_PORT` |
| 25575 | Paper | RCON，供训练环境摆位/回血/查询 | `server.properties` |

> **`spawn.mode=golem` 时不需要 bot-runtime**：Bot 身体由 Golem 插件在服务端进程内生成，
> 没有通道 A 会话、也没有 8770+ 的通道 B 端口（因此不可训练）。此时只需 `server/plugins/` 有
> `gladiator-0.1.0.jar` + `golem-0.1.0.jar`，通道 A（8765）仍照常监听。
> 配置与验收见 [`golem-integration.md`](golem-integration.md)。

## 2. 启动顺序

### 2.1 推荐：启动器 + 浏览器（进程与终端解耦）

```powershell
.\launcher\GladiatorLauncher.exe
```

- 启动器通过 WMI 拉起 `pythonw dashboard/app.py`，父进程是 `WmiPrvSE.exe`，**不挂在当前终端/IDE/Harness 下面**；关终端、重启 IDE 都不会带走 Dashboard 和它托管的 Paper / runtime / 训练进程。
- 浏览器打开 <http://127.0.0.1:8080/>，之后所有启停都在网页里：启动服务器、起停 bot-runtime、跑训练、人机对战、实时终端、产物可视化。
- `--status` 查看是否在跑，`--stop` 停止全部（含托管进程），`--port=` 换端口；日志在 `dashboard/var/out.log`，详见 `launcher/README.md`。
- 训练预设存放在 `dashboard/var/presets/*.json`（计划 + 训练参数的快照，同名覆盖）；该目录可安全备份/清空，不含用户名、端口或路径。
- 网页上的「停止服务器 / 停止所有 Bot」是正常关闭路径：先停训练 → 停 runtime → 停服务端（插件会还原在线观战者）。

### 2.2 手工方式（排查用）

```powershell
# 1) 服务端（Java 8；-Dfile.encoding=UTF-8 避免 config.yml 编码告警）
#    本机 java 路径用 JAVA_HOME 或自己机器上的位置，仓库不硬编码。
cd server
& "$env:JAVA_HOME\bin\java.exe" `
    -Dfile.encoding=UTF-8 -Xms512M -Xmx1024M -jar paper-1.8.8-445.jar nogui

# 2) 确认插件就绪
#    [Gladiator] Gladiator 0.1.0 enabled | api=1.1.0 | ready=true | link=127.0.0.1:8765

# 3) 每个 Bot 一个 runtime（端口递增）
cd bot-runtime
node src/index.js --username Glad_Bot_1 --host 127.0.0.1 --mc-port 25566 --env-port 8770
node src/index.js --username Glad_Bot_2 --host 127.0.0.1 --mc-port 25566 --env-port 8771

# 4) 拉起 Bot（或在训练环境里 auto_spawn=True）
#    控制台执行：gladiator spawn Glad_Bot_1

# 5) 训练 / 压测
cd training
.venv\Scripts\python.exe scripts\random_brawl.py --steps 200 --center 0.5,4,0.5
```

> 注意：手工方式起的东西不归 Dashboard 管；Dashboard 会把它们识别为“外部进程（只读）”，停止按钮不会动它们。

## 3. 关键配置

| 位置 | 键 | 说明 |
|---|---|---|
| `config.yml` | `bridge.host` / `bridge.port` | 通道 A 监听地址，**只允许回环地址** |
| `config.yml` | `bridge.token` | 空 = 不校验；非空时 HELLO 必须带同名 token |
| `config.yml` | `bridge.heartbeat-timeout-seconds` | 心跳超时（默认 30s），超时会话被回收 |
| `config.yml` | `bridge.spawn-timeout-seconds` | spawn 后等待登录的上限（默认 30s，超时 `TIMEOUT:`） |
| `config.yml` | `spectate.enabled` | 是否允许 `/gladiator spectate`（默认 true） |
| `config.yml` | `training.max-actions-per-second` | 动作节流（默认 10/s/Bot） |
| `config.yml` | `spawn.mode` | `runtime`（默认，mineflayer 登录）或 `golem`（Golem 进程内身体）；启动期参数，需重启 |
| `config.yml` | `spawn.golem.world` / `spawn.golem.location` | golem 身体的生成世界（空=第一个世界）与坐标 `x,y,z`（空=世界出生点） |
| 环境/参数 | `GLADIATOR_TEST_DROP_MS` | **仅测试**：runtime 启动 N ms 后掐一次链路，用于故障注入 |

## 3.1 Dashboard 的 Java 探测与配置（G-621）

Dashboard 启动 Paper 时**必须是 Java 8**（Paper 1.8.8 在 Java 17/21 上无法运行）。
**仓库不硬编码任何本机路径**，并且**不只看文件存在**：每个候选都会执行一次
`java -version` 实测 major（`dashboard/config.py` 的 `java_major_version()`），
只有 `major == 8` 才会被采用。探测顺序（`discover_java()`）：

1. `dashboard/config.json` 的 `javaPath`
2. 环境变量 `GLADIATOR_JAVA`（兼容旧名 `DASHBOARD_JAVA`）
3. `JAVA_HOME\bin\java.exe`
4. `PATH` 里的 `java`
5. 常见安装根下的 `jdk*8*` / `jdk-8*` / `zulu8*` 目录：
   `C:\Program Files\{Eclipse Adoptium,Java,Zulu,Microsoft}`（含 `Program Files (x86)`）

**不是 8 就跳过并继续找下一个**（例如 `PATH` 里是 JDK 17，会继续去安装根找 8），
绝不退化使用 17。如果所有候选都不是 8：

- `config.java` 为 `null`，`config.javaVersion` 是检测到的版本，`config.javaHint` 形如
  「检测到 Java 17（<path>），Paper 1.8.8 需要 Java 8，请配置 javaPath…」；
- 网页「启动服务器」**直接拒绝启动**并返回中文错误（不会尝试 spawn）。

版本探测结果按 `(路径, mtime, size)` 缓存，替换 JDK 后自动失效（`clear_java_version_cache()` 手动清）。

推荐显式配置（`dashboard/config.json`，该文件不入库）：

```json
{ "javaPath": "C:\\Program Files\\Eclipse Adoptium\\jdk8u402-b06\\bin\\java.exe" }
```

也可以临时用环境变量（PowerShell）：

```powershell
$env:GLADIATOR_JAVA = 'D:\jdk8\bin\java.exe'
.\launcher\GladiatorLauncher.exe --no-browser
```

> 注意：机器上同时装了 JDK 17/21 时，`PATH` 里的 `java` 通常就是 17 —— 它会被**拒绝**
> （不会被当成可用 java），Dashboard 会继续找 Java 8；找不到就 `java=null` + `javaHint`
> 说明原因。用 `GET /api/health` 或 `GET /api/config` 的
> `config.java` / `config.javaVersion` / `config.javaHint` 确认当前用的是哪一个。

## 3.2 「打开文件夹」与只读验证实例（G-622 / G-623）

`POST /api/open-folder`，body `{ "kind": "...", "id": "<runId>" }`，返回
`{ "ok": true, "path": "<绝对路径>" }`。Windows 用 `os.startfile` 打开；
非 Windows 返回 501「打开文件夹仅支持 Windows」；
`DASHBOARD_ALLOW_PROCESS_CONTROL=0` 的只读实例返回 403。

| kind | 目录 |
|---|---|
| `scripts` | `dashboard/var`（注册表 `scripts.json` 所在层） |
| `presets` | `dashboard/var/presets` |
| `reward-presets` | `dashboard/var/reward-presets` |
| `models` | `dashboard/var/models`（策略产物注册表） |
| `gmod` | 仓库根 `models/`（`.gmod` 文件模型，模型页「打开文件夹」按钮用这个 kind） |
| `run` | `training/artifacts/runs/<id>`（id 必须是已存在的 run，否则 404） |
| `run-reviews` | `<run>/reviews` |
| `artifacts` | `training/artifacts` |
| `logs` | `dashboard/var/logs` |

路径先 `resolve()` 再检查是否落在 `dashboard/var` / `training/artifacts` /
`models/` 根内，防 `../` 穿越；未知 kind 与非法 runId 一律 400。前端在脚本管理、
预设列表、reward 预设、模型列表、run 详情与对局详情都放了按钮，失败会 toast 中文原因。

### 3.2.1 `models/` 目录（`.gmod` 文件模型，G-623）

- 默认位置：**仓库根 `models/`**；Dashboard 启动时自动创建并放一个 `.gitkeep`（空目录可入库）。
- 覆盖：`dashboard/config.json` 的 `"gmodDir": "D:/somewhere/models"` 或环境变量
  `DASHBOARD_GMOD_DIR`（相对路径按仓库根解析）；`GET /api/config` 的 `config.gmodDir` 可核对。
- 往目录里放 `xxx.gmod`（zip：`meta.json` + `policy.npz`），刷新模型页即出现；
  删除文件后自动消失。校验失败的包会显示「不可用 + 中文原因」且训练下拉禁选，
  `POST /api/models/reload` 返回 `{scanned, added, invalid}`。
- **不要**用 `DELETE /api/models?id=<gmod>` 删文件：接口返回 409「`.gmod` 由文件系统管理，
  请删除 models/xxx.gmod」。改名/改 meta 也直接编辑文件后刷新。
- 训练选用时把包内 npz 解到 `dashboard/var/gmod-cache/<sha256>.npz`（按内容复用，可安全清空）。
  完整格式与校验规则见 `docs/models.md` §8。
- 用临时只读实例验证时，若需要扫描**真实的** `models/`，不要设 `DASHBOARD_GMOD_DIR`；
  若想完全隔离，就把它指向临时目录。测试一律注入临时 `gmod_dir`。

### 3.2.2 持久化训练的锁文件与备份（G-624）

选中的策略模型在训练结束会被写回，相关文件都在 `dashboard/var/`：

| 路径 | 用途 |
|---|---|
| `models/<id>/.lock` | 写锁，JSON `{pid, runId, at, modelId}`；同一模型只允许一个持久化训练持有 |
| `models/<id>/policy.npz.tmp` | 原子替换用的临时文件，正常情况写完立即消失 |
| `models/<id>/training.json` | 累加的 `totalEpisodes` / `totalSteps` + 最近 20 次写回记录 |
| `models-trash/<id>-<时间戳>.npz` | **覆盖前**备份的旧权重（可人工恢复） |
| `models-trash/<id>-<时间戳>/` | 导入/改名/软删除时备份的整个模型目录（G-622 起） |
| `gmod-cache/<sha256>.npz` | `.gmod` 包内解出的只读权重副本，按内容复用 |

- 启动持久化训练时若 `.lock` 被**活进程**持有 → 接口 409「模型 X 正在被 run Y 的持久化训练占用」；
  等它结束、换一个模型，或关闭「持久化训练」开关即可。
- 若锁里的 pid 已不存在（Dashboard 被强杀留下的孤儿锁）→ 自动回收，并在进度卡
  notes 里写「发现并回收了模型 X 的过期锁」。进程存活检查用 `OpenProcess`（Windows），
  **不会**用 `os.kill(pid, 0)` 去 terminate 进程。
- 训练进入任何终态都会释放锁；`models-trash/` 与 `gmod-cache/` 不自动清理，
  可手动删除（后者会按内容重新生成）。清理前建议确认没有训练在跑。
- 写回失败（磁盘满/权限/npz 损坏）只体现在 `job.persist.error` 与前端 toast，
  **不会**把已经正常完成的训练标记成 failed。详见 `docs/models.md` §9。

**有训练在跑时做接口验证**（不改 8080、不碰 Paper / 训练）：另开一个终端，
先把 `DASHBOARD_VAR` / `DASHBOARD_ARTIFACTS` 指到临时目录，设
`DASHBOARD_SCAN_EXTERNAL=0` 与 `DASHBOARD_ALLOW_PROCESS_CONTROL=0`，再用
`training\.venv\Scripts\python.exe dashboard\app.py --port 8081` 起临时实例。
该实例只允许调 GET 接口与 `/api/open-folder`（只读模式下返回 403，符合预期），
**不要**在 8081 上启动训练或开服；验证完必须终止 8081。它使用独立的 var/artifacts，
不会与正在跑的 run 争用 `dashboard/var/**`。

> 只读实例不会启动/停止任何子进程，因此也无法真正打开资源管理器；
> `open-folder` 的 200 分支由 `dashboard/tests/test_models.py` 用注入的
> `folder_opener` 在临时端口上覆盖。

## 4. 排障表

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `WELCOME` 里 `playerUuid=""` | 运行时先于 `spawnBot` 连线，玩家实体还没生成 | 正常；等 `BOT_BOUND` 或收到 `SPAWN` 后再 spawn |
| `spawn 失败：TIMEOUT: 等待 Bot 登录超过 30s` | runtime 未连上 / 用户名不匹配 / 服务端白名单或 online-mode | 看 runtime 日志是否有 `WELCOME`；核对用户名；检查 `server.properties` 的 `online-mode`/`white-list` |
| `SPAWN_FAILED` 里 `VERSION_MISMATCH` | 客户端版本与服务端不一致 | runtime 用 `--version 1.8.8`（1.8.9 服务端也用该协议版本） |
| `ONLINE_MODE` / `NOT_WHITELISTED` | 正版校验或白名单拒绝 | 改 `online-mode=false` 或把 Bot 名加入白名单 |
| `spawn 失败：Bot 已就绪，不能重复 spawn` | 同一用户名已 Ready 或正在 spawn | 先 `/gladiator list`；需要重建时先让 Bot 退出 |
| `spawn 失败: spawnBot 不可用：spawn.mode=golem 但未检测到可用的 Golem 插件` | `config.yml` 配了 `spawn.mode=golem`，但 `server/plugins/` 里没有 `golem-0.1.0.jar`（或 Golem 未成功装配） | 放入 golem jar 后重启，或把 `spawn.mode` 改回 `runtime`；插件本身仍正常启用 |
| golem Bot 消失后 `bots` 未立刻归零 | golem 侧 `despawn` 不发 Bukkit 事件，靠 2s 周期对账回收 | 等 ≤4s 再看 `bots`；日志会打印 `Bot 退出: <name> reason=golem-body-gone` |
| `/gladiator spectate` 后仍是生存模式 | `server/plugins` 里是旧 jar（只跑了 `compileJava`） | `gradlew build` → 重启服务端 → 重新 `spawn` |
| 观战被 Bot 追打 | 训练环境未启用 `opponents` 过滤（旧版本） | 升级 `training/gladiator_env`；目标只取参赛 Bot |
| 动作全部 `ACTION_RATE_LIMITED` | 超过 `training.max-actions-per-second` | 降频或调大配置；检查是否重复发同一 seq（应走 `deduped`） |
| Bot 一 spawn 就摔死 | `ArenaConfig.center` 的 y 与世界地面不符（FLAT 地面 y=4） | `--center 0.5,4,0.5` 或改 `ArenaConfig` |
| `analyze` 伤害数值异常巨大 | 虚空/哨兵伤害（历史 bug） | 已修复：超过 1000 的伤害被忽略 |
| 训练 `step()` 很慢（~1s） | `wait_action_results=True` 等每个 `ACTION_RESULT` | 保持默认 `False`（只发不等） |
| `sessions` 比 runtime 多 | 上次进程异常退出留下的会话 | 30s 心跳超时后自动回收；也可重启服务端清空 |
| `sessions` 比 runtime 少 | runtime 还在退避重连 | 等 1s→2s→4s…30s；看 runtime 的 `reconnecting` 日志 |
| RCON 认证失败 | 密码错误或 RCON 未开 | 对齐 `rcon.password`；生产环境建议关闭 RCON |
| 「打开文件夹」按钮返回 403 / 501 | 只读模式（`DASHBOARD_ALLOW_PROCESS_CONTROL=0`）或非 Windows | 属预期；需要真正打开时去掉只读模式并使用 Windows。路径穿越/未知 kind 会返回 400 |
| 放进 `models/` 的 `.gmod` 没出现 / 显示「不可用」 | 没刷新、`meta.json` 维度不符、`policy.npz` 损坏，或 `gmodDir` 指向别的目录 | 点模型页「刷新」（`POST /api/models/reload`），看返回的 `invalid` 中文原因；用 `GET /api/config` 的 `gmodDir` 核对目录 |
| 删除 `.gmod` 模型返回 409 | 文件模型由文件系统管理，Dashboard 不删用户文件 | 按提示手动删除 `models/xxx.gmod`，再刷新 |
| 启动训练返回 409「模型 X 正在被 run Y 占用」 | 另一个持久化训练持有 `models/X/.lock` | 等它结束、换模型或关闭「持久化训练」；若 pid 已不存在会自动回收孤儿锁 |
| 进度卡提示「模型写回失败」但训练是 finished | 磁盘/权限/npz 校验失败，写回被独立捕获 | 看具体中文原因与 `models-trash/` 备份；修复后重跑一次即可（训练本身没失败） |
| HTTP 返回 `501 Unsupported method ('{}GET')` | keep-alive 连接上某个 POST 的 body 没被读完，残留字节被当成下一行请求行 | 已修（G-621）：`handle_one_request` 每请求重置 body 缓存、支持 chunked；重启 Dashboard 生效 |
| 插件日志出现 mojibake | 源码/配置被非 UTF-8 工具改写 | 统一用 UTF-8（无 BOM）读写；不要用 `Get-Content`/`Set-Content` 往返改源码 |

## 5. 已知边界

- 通道 A 只绑定回环地址：不支持跨机部署（设计如此）。
- 每个 Bot 需要独立 runtime 进程与独立通道 B 端口；4 Bot 实测 TPS ≈ 20.0、≈4.5 step/s。
- RCON 仅用于本地训练场摆位；生产接入应由调用方自行完成环境准备（ADR-015）。
