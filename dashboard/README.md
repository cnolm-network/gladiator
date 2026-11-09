# Gladiator Dashboard

Minecraft 1.8.9 Bot 训练框架的本地 Web 控制台。用浏览器管理 Paper 服务端、bot-runtime、训练任务与人机对战，并可视化 `training/artifacts` 中的训练产物。

- 技术栈：Python 3 标准库（`http.server` + `ThreadingHTTPServer`）+ 单页 HTML/CSS/原生 JS
- 依赖：**零第三方依赖**（`requirements.txt` 为空；不访问任何 CDN，可完全离线使用）
- 监听：仅回环地址，默认 `http://127.0.0.1:8080/`

## 推荐启动方式：启动器（进程与终端解耦）

```powershell
.\launcher\GladiatorLauncher.exe
```

启动器用 WMI 把 Dashboard 挂到 `WmiPrvSE.exe` 下并以 `pythonw` 运行，关掉终端 / IDE / Harness 都不会带走它；要停用 `--stop` 或网页上的按钮。详见 `launcher/README.md`。

## 快速启动

```powershell
# 在仓库根目录
.\training\.venv\Scripts\python.exe .\dashboard\app.py
```

或使用脚本（可选指定端口）：

```powershell
.\dashboard\run.ps1
.\dashboard\run.ps1 -Port 8090
```

然后浏览器打开 <http://127.0.0.1:8080/>。控制台按 `Ctrl+C` 退出；退出时会尝试优雅停止**由本 Dashboard 托管**的 Paper / bot-runtime / 训练进程。

> 服务器、训练、bot-runtime 的启动都依赖 Windows 本机路径；换机器请参考下方环境变量。

## 功能

### 1. 服务管理

- 查看 Paper 进程是否运行、是否由 Dashboard 托管、PID、运行时长；
- 实时探测 25566（游戏）、8765（通道 A）、25575（RCON）端口；
- 通过 RCON 执行 `gladiator status` 与 `tps`，解析 bots / games / training / sessions / worldguard / heap；
- 启动 / 停止 Paper（`java -Dfile.encoding=UTF-8 -Xms… -Xmx… -jar paper-1.8.8-445.jar nogui`，工作目录 `server/`）；
- 控制台输入任意服务端命令；
- 展示服务端日志尾部，并提供完整日志路径。

启动前会检查 jar、Java 8 可执行文件、25566 是否空闲。停止时先发送 `stop`，超时才强制结束。

> **接入已有服务**：25566 上已有**不是 Dashboard 启动**的 Paper 时，页面会显示“外部进程（只读）”、PID（通过 netstat 发现）与 `gladiator status`，但停止/控制台按钮不可用。Dashboard 不会终止外部进程。

### 2. 运行时管理

- 列出由 Dashboard 启动的 bot-runtime（用户名、通道 B 端口、PID、存活、端口监听、日志尾部）；
- 启动：`node src/index.js --username <name> --host 127.0.0.1 --mc-port 25566 --env-port <port>`（工作目录 `bot-runtime/`）；
- 单个停止 / 全部停止；
- 端口与用户名在服务端再次校验；保留 25566 / 8765 / 25575 / Dashboard 端口。

> 通道 B 端口默认从 **8770** 起（范围由 `DASHBOARD_RUNTIME_PORT_MIN/MAX` 控制，默认 8770–8789）。在该范围内监听、但不是 Dashboard 启动的 bot-runtime 会以“外部运行中”只读展示（显示 PID），停止按钮禁用，不会误杀外部进程。

> bot-runtime 启动后先连接通道 A 的 `ws://127.0.0.1:8765/gladiator`，收到插件的 `SPAWN` 消息后才会真正登录游戏。因此「进程存活」不等于「Bot 已进服」。

### 3. 训练计划

- 表单化启动 `python -m gladiator_train.train`（单场地）或 `python -m gladiator_train.multi_train`（多场地并行）；
- 参数：用户名列表、通道 B 端口列表、**训练计划**（有序 `[{对手模式, 轮次}]`）、**训练回合**（每回合完整跑一遍计划，默认 1）、**评估回合**（训练后跑几遍，贪心不学习，默认 1）、max-steps、seed、场地中心、间距（默认 6.0）、开局准备期 `--prep-seconds`（默认 3.0）、auto-spawn、**autoCleanup**；multi 模式额外支持 pairs / 用户名前缀 / 起始端口；
- 三层语义：回合（整遍计划）→ 计划（有序段）→ 轮次（一局对局）；训练总对局 = 训练回合 × Σ(轮次)，由 Dashboard 计算并传给训练脚本（`--epochs` / `--eval-epochs`）；
- 计划与预设：计划列表可拖拽排序、行内改轮次、删除；`保存为预设…` 把计划与训练参数写入 `dashboard/var/presets/<id>.json`（同名覆盖），可加载/删除；预设**不保存**用户名、端口与路径，换机后不会指向错误环境；契约见 `docs/training-plan-ui.md`；
- 每次训练由 Dashboard 创建独立目录 `training/artifacts/runs/<YYYYMMDD-HHMM>-<kind>-<tag>/`，只向训练脚本传 `--run-dir`，不再覆盖旧的扁平报告；
- 启动/停止是**异步**的：`POST /api/training/start` 先返回 `status=starting` 的 job，运行时的并发拉起、端口校验和训练进程 spawn 在后台完成（统一 7 秒 deadline）；`POST /api/training/stop` 先返回 `status=stopping`，进程终止与 autoCleanup 在后台收尾，前端继续轮询状态；
- 后台运行，实时解析 stdout：
  - 阶段切换（`[train] === 阶段 … ===`）
  - 每回合 episode / steps / reward / winner / **开始时间 / 耗时**
  - 多场地 `[pairN] 完成` / `失败`
- 展示进度条、当前计划项（第 X/N 段）、总回合、步数、最近 reward、平均 reward、reward 曲线、对局查询表与日志尾部；
- 训练状态接口会透出 `runDir`、`startedAt`、`finishedAt`、`durationSeconds`，以及每局的 `startedAt` / `durationSeconds`；
- 结束后从该 run 的 `report.json` 加载报告摘要并刷新数据页；
- **autoCleanup（默认开启）**：只停止本次训练自动拉起的 bot-runtime；训练开始前已经存在的运行时不记录、不停止。关闭后保留这些运行时，并在日志/接口里记一行说明；
- 停止按钮结束进程（先 terminate，再 kill），停止后同样执行本次运行时的清理；
- 每个 run 的 `run.log` **只包含本次训练**：Dashboard 的 `training.log` 是复用追加文件，落盘时会按最后一次 `启动 training` 标记切片并去掉头部，首行就是本次命令行；

启动前校验：至少 2 个 Bot、用户名格式、端口范围/保留端口/重复端口、对应通道 B 端口已有 bot-runtime 监听（缺失且开启 autoSpawn 时先自动拉起）、当前没有其他训练任务。

### 4. 人机对战

- 一键启动 N 个 bot-runtime（1–8 个），等待通道 A 会话出现后，用 RCON 执行 `gladiator spawn <username>`，再轮询 `gladiator list` 等待 `ready=true`；
- 展示进服地址 `127.0.0.1:25566`（Minecraft 1.8.9，`online-mode=false`，离线用户名）；
- 给出并支持复制观战命令：`/gladiator spectate <bot>`、`/gladiator unspectate`；
- 展示每个 Bot 的端口、runtime id、ready、PID 与对战启动日志；
- 「结束对战」会停止本场拉起的 bot-runtime。

需要 Paper 已监听 25566 且 RCON 可达。

### 5. 数据可视化

只读读取 `training/artifacts/`，不会修改训练产物：

- **按 run 分组**：扫描 `training/artifacts/runs/*`，按目录名倒序展示每次运行的时间、kind、plan、episodes、steps、状态（finished / running / interrupted）与平均 reward；点击某次运行查看该 run 的 reward 曲线与逐局复盘；
- **多选 / 全选 / 时间区间合并**：runs 表每行可勾选，工具栏支持全选、按开始/结束时间选中，以及「合并选中」把多次运行的历史按时间合并；合并后卡片、曲线、复盘和多场地 pair 都展示聚合结果，图表上的 `R1/R2` 对应来源列表；
- **选中 run 驱动整页**：顶部列表切换 run 后，回合数、平均 reward、命中率、reward 曲线、多场地区块与复盘列表都重新按该 run 的 `report.json` 计算；命中率由 `history[].review.hitRate/hits/whiffs` 聚合，不是只读旧扁平报告；
- **兼容旧数据**：`train-report.json`、`multi-report.json`、`reviews/` 会作为「旧版扁平数据（legacy）」虚拟 run 继续显示，旧页面数据不会消失；legacy 只聚合扁平 `reviews/`，不会把新 run 的复盘重复算进去；
- **多场地**：选中 multi run 时展示每个 pair 的 mean reward、命中/落空、回合、步数与开始时间/耗时；点击 pair 行或「复盘」按钮可过滤该场地的复盘；
- **计数口径统一**：进度卡、曲线与卡片都以 `report.json` 的 `history` 为准；multi 的总回合是各 pair 之和，进度卡显示「已完成 X / 计划 Y（N 场地 × M）」，曲线标题标「合并 N 个场地」；若某 run 的 `reviews` 份数与 `history` 条数不一致，数据页会显示口径提示并以 history 为准；
- **训练计划是唯一回合来源**：进度卡显示当前计划项「第 X/N 段 · 本段 a/b 局」，回合卡是**总回合**（= 各段之和），下方按顺序列出完整计划（`模式×轮次`）与合计数。训练表单不再提供「阶段计划」文本框、独立「对手模式」下拉或独立「回合数」输入，避免三处数值互相矛盾；
- **压力测试**：有 `stress-report.json` 时展示 TPS 均值/最低/最高与步频；没有报告时从选中 run 派生步频（`totalSteps / durationSeconds`）、回合数与平均 reward，并把 `/api/server` 的实时 TPS 标为「TPS（实时）」；
- **图表交互**：reward 曲线的每个数据点都有透明命中带，鼠标悬停立即显示来源 run、回合、reward、步数、耗时、开始时间与胜负；点击数据点会切到该 run 并滚动到运行详情；
- `reviews/**/*.json|.md`：逐局复盘列表，显示每局开始时间与耗时，点击后用安全 Markdown 渲染器展示 `.md`。

不同 run 的 `episode-001` 不再互相覆盖；列表和详情里始终带 run 名称。没有数据时显示占位说明，不报错。

### 6. 实时终端（`/terminal`）

- 基于 SSE（`text/event-stream`）的**只读**日志流，不执行任何命令；
- 来源：
  - `server`：`server/console.out.log`，先回最后 200 行再跟随新行；
  - `training`：当前/最近 run 的 `run.log`；文件尚未生成时退回 Dashboard 捕获的 stdout / 内存日志；
  - `runtime`：按用户名 / runtime id / 端口选择托管 bot-runtime，读其 stdout；外部进程或无可读日志时退化为状态 + 最近事件，并在页面注明「暂无输出流」；
- 页面支持来源/目标切换、自动滚动、暂停/继续、清屏（只清浏览器缓冲）、自动重连；
- 服务端只按白名单枚举路径（`server/console.out.log`、`runs/<id>/run.log` 等），客户端不能传任意文件路径。

## 配置

所有配置都可通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHBOARD_HOST` | `127.0.0.1` | 只允许回环地址 |
| `DASHBOARD_PORT` | `8080` | 监听端口 |
| `DASHBOARD_ARTIFACTS` | `training/artifacts` | 训练产物目录（只读） |
| `DASHBOARD_VAR` | `dashboard/var` | Dashboard 日志、进程审计 |
| `DASHBOARD_SERVER_DIR` | `server` | Paper 工作目录 |
| `DASHBOARD_BOT_RUNTIME_DIR` | `bot-runtime` | bot-runtime 目录 |
| `DASHBOARD_TRAINING_DIR` | `training` | 训练代码目录 |
| `DASHBOARD_PYTHON` | `training/.venv/Scripts/python.exe` | 训练用解释器 |
| `DASHBOARD_JAVA` | 本机 JDK 8 `java.exe` | 启动 Paper 用 |
| `DASHBOARD_NODE` | PATH 中的 `node` | bot-runtime 用 |
| `DASHBOARD_ALLOW_PROCESS_CONTROL` | `1` | 设为 `0` 进入只读模式（禁止一切启动/停止） |
| `DASHBOARD_RCON_HOST/PORT/PASSWORD` | `127.0.0.1` / `25575` / `devpass` | RCON |
| `DASHBOARD_MC_HOST/PORT` | `127.0.0.1` / `25566` | 游戏地址 |
| `DASHBOARD_BRIDGE_PORT` | `8765` | 通道 A 端口 |
| `DASHBOARD_RUNTIME_PORT_MIN/MAX` | `8770` / `8789` | 通道 B 端口范围（含外部进程发现） |
| `DASHBOARD_SCAN_EXTERNAL` | `1` | 设为 `0` 关闭外部 bot-runtime 发现 |
| `DASHBOARD_SERVER_JAR` | `paper-1.8.8-445.jar` | 服务端 jar 文件名 |
| `DASHBOARD_SERVER_MEMORY_MB` | `1024` | `-Xmx` |

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 版本、PID、运行时长、只读状态、脱敏配置 |
| GET | `/api/config` | 脱敏配置 |
| GET | `/api/overview` | 总览聚合 |
| GET | `/api/server`、`/api/server/status` | 服务端状态/端口/RCON/TPS/日志 |
| POST | `/api/server/start`、`/api/server/stop`、`/api/server/console` | 服务端控制；console 请求体 `{ "line": "say hi" }` |
| GET | `/api/runtimes` | 运行时列表 + 默认用户名/端口 |
| POST | `/api/runtimes/start`、`/api/runtimes/stop`、`/api/runtimes/stop-all` | 运行时控制；start 请求体 `{ "username": "...", "envPort": 8770 }` |
| GET | `/api/training`、`/api/training/options` | 当前任务/选项 |
| POST | `/api/training/start`、`/api/training/stop` | 训练控制 |
| GET | `/api/match` | 对战状态、Bot 列表、观战命令 |
| POST | `/api/match/start`、`/api/match/stop` | 对战控制；start 请求体 `{ "count": 2, "usersPrefix": "Glad_Bot", "portsBase": 8770 }` |
| GET | `/api/artifacts?run=<runId>` | 选中 run 的 train/multi/reviews/stress 聚合；不传时默认最近 run |
| GET | `/api/artifacts/merge?runs=<id1,id2,...>` | 合并多个 run 的历史、复盘、pair 与压力派生指标 |
| GET | `/api/artifacts/runs` | run 列表（含 legacy 虚拟 run） |
| GET | `/api/artifacts/run?id=<runId>` | 单次 run 的摘要、history、复盘与报告详情 |
| GET | `/api/artifacts/review?path=...` | 读取 `reviews/` 或 `runs/**/reviews/` 下的 `.md` |
| GET | `/terminal` | 实时终端页面（单页应用路由） |
| GET | `/api/terminal/stream?source=<来源>&name=<目标>` | SSE 只读日志流；来源为 `server` / `training` / `runtime` |

错误统一返回 `{ "ok": false, "error": "可读的中文错误" }`，并使用合适的状态码（400/403/404/409/413/500）。

## 安全边界

- 只监听 `127.0.0.1` / `localhost` / `::1`，配置其他地址会直接拒绝启动；
- 静态文件与复盘读取都做了目录穿越校验；请求体上限 1 MB；
- CSP：`default-src 'self'`，不加载外部脚本/样式/字体；
- 只操作**自己启动**的子进程：启动时记录 PID 到 `dashboard/var/managed.json`，外部已存在的 Java/Node 进程只展示、不接管、不终止；
- 所有启动/停止/训练/对战接口都受 `DASHBOARD_ALLOW_PROCESS_CONTROL` 控制；
- 不修改 `src/`、`bot-runtime/src/`、`training/gladiator_env/`、`training/gladiator_train/`，只通过命令行调用它们。

## 测试

```powershell
cd .\dashboard
..\training\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 40 项。测试使用临时目录和假产物，不启动 Java/Node；覆盖静态页、只读 API 结构、选中 run 与 legacy 隔离、多 run 合并（含未知/空列表拒绝）、multi pair/复盘、stress 派生、训练开始/停止立即返回与后台收尾、autoCleanup 只停本次拉起的运行时、训练 argv（含 `--run-dir`、`--spacing 6.0`、`--prep-seconds 3.0`）、multi 计数口径与 report 校准、run.log 只截取本次启动、SSE 返回 `event: meta`、只读模式 403、非法 JSON、目录穿越、复盘 Markdown、状态解析等。

## 已知限制

1. **不接管外部进程**：先于 Dashboard 启动的 Paper/bot-runtime 只会被识别为「外部」，不能从页面停止；Dashboard 重启后也不会重新接管之前启动的进程（`managed.json` 仅作审计）。请先停止旧进程再从 Dashboard 启动。
2. **Java 路径是 Windows 默认值**：其他机器需要设置 `DASHBOARD_JAVA`；Node 同样可用 `DASHBOARD_NODE` 覆盖。
3. **训练进度依赖 stdout 文案**：解析 `[train] episode …`、`[multi] …` 等固定格式；训练器输出格式变更时进度会退化（日志仍可见，报告仍会加载）。
4. **只展示聚合压力指标**：`stress-report.json` 只有 TPS/步频聚合值，没有逐秒时间序列，因此不能画 TPS 抖动曲线。
5. **人机对战是编排，不是对局规则**：Dashboard 只负责拉起 Bot、spawn、展示观战命令；胜负、套件、地图由服务端插件/调用方决定。
6. **无身份认证**：仅限本机使用；不要将端口暴露到公网或反代到非回环地址。
7. **端口占用会明确报错**：服务端/运行时/训练端口被占用时返回 409 并给出提示，不会自动改端口。
8. **外部训练进度不可见**：Dashboard 只能通过 `gladiator status`、进程与产物目录“看到”外部训练；外部训练器的 stdout 不在 Dashboard 手里，终端页只能显示已有 `run.log` 或提示「暂无输出流」，进度/日志无法实时展示。
9. **运行时 stdout 可能不可用**：Windows 下外部 bot-runtime 的 stdout 无法被 Dashboard 读取，终端页会退化为状态 + 最近事件；托管进程的 stdout 则正常。历史 run 若没有 `run.log`，也只会显示已捕获的内存日志或占位。
10. **旧 run.log 无法回溯拆分**：`run.log` 按“本次启动标记”切片的行为从本版开始生效；此前生成的 run 目录可能仍含旧版整体追加的内容，只能重新跑一次获得干净日志。
10. **autoCleanup 按 id 清理**：只清理本次训练通过 autoSpawn 拉起且仍存活的 runtime；窗口内被手动停掉或改名的进程不会被误杀。关闭开关后这些 runtime 会保留，需要手动停止。
11. 退出 Dashboard 时若强制杀进程（任务管理器），被托管的 Java/Node 可能残留；正常 `Ctrl+C` 会尽量优雅清理。

## 目录

```
dashboard/
  app.py              HTTP 路由 / 静态服务 / 进程组装
  config.py           配置（环境变量）
  processes.py        子进程托管、日志尾部、PID 审计
  server_manager.py   Paper + RCON 状态
  runtime_manager.py  bot-runtime 管理
  training_runner.py  train / multi_train 任务
  match_runner.py     人机对战编排
  terminal.py         SSE 只读日志流
  artifacts.py        训练产物只读读取（run / legacy）
  rcon_client.py      标准库 RCON 客户端
  ports.py            端口探测
  validation.py       入参校验
  static/             单页前端
  tests/test_app.py   冒烟测试
  run.ps1             启动脚本
```
