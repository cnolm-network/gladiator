# 手动测试指南（Manual Testing）

本文面向“人工进服验收”：服务端 + Bot 运行时 + 训练脚本都在本机跑，用 Minecraft 1.8.9 客户端
进入服务器观察 Bot、观战、执行管理命令，并把结果记录在文末模板里。

> 适用范围：Paper 1.8.8 + Gladiator 0.1.0 + `bot-runtime`（每个 Bot 一个 Node 进程）+
> `training/`（Gymnasium 训练环境）。协议细节见 `docs/protocol.md`，观战细节见 `docs/spectating.md`。

## 1. 前置检查

| 项 | 要求 | 检查方式 |
|---|---|---|
| JDK | Java 8（`C:\Program Files\Eclipse Adoptium\jdk-8.0.492.9-hotspot`） | `java -version` 显示 1.8 |
| 服务端 | `server/paper-1.8.8-445.jar`，`online-mode=false` | `server/server.properties` |
| 插件 | `server/plugins/gladiator-0.1.0.jar`（构建产物） | `gradlew build --offline` 后复制 |
| RCON（训练控制面） | `enable-rcon=true`、`rcon.port=25575`、`rcon.password=devpass` | 联调时临时加，结束要还原 |
| 端口 | 25565 / 8765 / 8766 / 8767 / 25575 未被占用 | `Get-NetTCPConnection -State Listen` |
| Node | v22+（实测 v24.18.0），`bot-runtime/node_modules` 已安装 | `node --version` |

## 2. 启动顺序

```powershell
# 1) 构建并部署插件（改过 Java 就重新构建，别只 compileJava）
cd D:\cbot-gladiator
.\gradlew.bat build --offline
Copy-Item build\libs\gladiator-0.1.0.jar server\plugins\gladiator-0.1.0.jar -Force

# 2) 启动服务端（工作目录 server/，日志重定向便于排查）
Start-Process -FilePath 'C:\Program Files\Eclipse Adoptium\jdk-8.0.492.9-hotspot\bin\java.exe' `
  -ArgumentList '-Dfile.encoding=UTF-8','-Xms512M','-Xmx1024M','-jar','paper-1.8.8-445.jar','nogui' `
  -WorkingDirectory 'D:\cbot-gladiator\server' `
  -RedirectStandardOutput 'D:\cbot-gladiator\server\console.out.log' `
  -RedirectStandardError  'D:\cbot-gladiator\server\console.err.log'

# 3) 每个 Bot 一个运行时（端口 8766/8767 ...）
cd D:\cbot-gladiator\bot-runtime
Start-Process node -ArgumentList 'src/index.js','--username','Bot_T1','--host','127.0.0.1','--mc-port','25565','--env-port','8766'
Start-Process node -ArgumentList 'src/index.js','--username','Bot_T2','--host','127.0.0.1','--mc-port','25565','--env-port','8767'

# 4) 拉 Bot 进服（也可以由训练脚本 --auto-spawn 代劳）
node $env:TEMP\rcon.mjs "gladiator spawn Bot_T1 cps=15 jitter=2 reaction=220 rjitter=40 reach=3.99 model=default"
node $env:TEMP\rcon.mjs "gladiator spawn Bot_T2 cps=15 jitter=2 reaction=220 rjitter=40 reach=3.99 model=default"
```

**成功标志**（`server/console.out.log`）：

```
[Gladiator] Gladiator 0.1.0 enabled | api=1.1.0 | ready=true | link=127.0.0.1:8765 ...
[Gladiator] LinkServer listening on 127.0.0.1:8765 (token=disabled, heartbeat=30s)
[Gladiator] Bot ready: glad-bot_t1 user=Bot_T1 uuid=...
```

运行时侧日志：

```
[link] WELCOME session=... playerUuid=(pending)
[env] 通道 B listening on ws://127.0.0.1:8766 (controlHz=10)
[gladiator] SPAWN ... profile=cps 15±2, reaction 220±40ms, reach 3.99, model default
```

## 3. 客户端进服

1. Minecraft **1.8.9**，多人游戏 → 添加服务器 → 地址 `127.0.0.1`（或 `localhost`）→ 进入。
2. 首次进入会落在世界出生点；Bot 在训练场（默认 `center=0.5,4,0.5`，超平坦世界）。
3. 若提示版本不匹配：服务端是 1.8.8 协议，1.8.9 客户端可直接进；不要用 1.9+。

进服后先确认基础可用：

```
/gladiator status      # 期望：ready=true、bots=N、sessions=N、heap used=...
/gladiator list        # 期望：列出 botId / 名字 / UUID / ready / state
```

## 4. 观战（只读）

| 命令 | 作用 | 期望 |
|---|---|---|
| `/gladiator spectate` | 观战第一个可用 Bot | 切旁观者模式，传送到 Bot 上方 3 格，视角朝向 Bot |
| `/gladiator spectate Bot_T1` | 观战指定 Bot | 同上，目标为 Bot_T1 |
| `/gladiator unspectate` | 退出观战 | 还原游戏模式并回到进入前的位置 |

验收要点：

- 进入后 F3 看 `gameMode=spectator`，能自由飞行、无法攻击/放置方块；
- 退出后回到原位置、原游戏模式（生存/创造）；
- 断线重连后不会卡在旁观者模式（插件在 `PlayerQuitEvent` 里会还原模式）；
- 观战者**不会**被训练环境当成目标（`opponents` 过滤），Bot 不会追着观战者打。

## 5. 管理命令清单

| 命令 | 说明 |
|---|---|
| `/gladiator status` | 版本、api、ready、WorldGuard、bots、games、training、sessions、堆内存 |
| `/gladiator list` | Bot 列表（botId、名字、UUID、ready、state） |
| `/gladiator spawn <user> [key=value ...]` | 拉起 Bot；可带 `cps=/jitter=/reaction=/rjitter=/reach=/model=` |
| `/gladiator train <user>` | 把已就绪 Bot 放进训练会话（`pushPlace`） |
| `/gladiator game` | 当前对局/训练会话与参与者 |
| `/gladiator analyze [sessionId] [windowSec]` | 分析快照：存活时长、伤害、方块、路径、规则拒绝、动作统计 |
| `/gladiator pvp <x> <y> <z>` | 查询该坐标的 PVP 判定（需要玩家执行） |
| `/gladiator reload` | 重载 `config.yml`（启动期参数需重启） |
| `/gladiator spectate [bot]` / `unspectate` | 见上 |

## 6. 训练与复盘

```powershell
cd D:\cbot-gladiator\training
$env:PYTHONIOENCODING = "utf-8"

# 课程训练：静止 → 走动 → 随机 → 会还手；每局自动写复盘
.venv\Scripts\python.exe -m gladiator_train.train `
  --users Bot_T1 Bot_T2 --ports 8766 8767 `
  --stages idle:1,wander:1,random:1,scripted:2 --max-steps 200 --auto-spawn
```

观察点：

- 控制台每回合打印 `episode X/Y steps=... reward=... winner=...`；
- `training/artifacts/reviews/episode-XXX.md`：伤害交换 / 命中率 / 连击峰值 / 动作分布 / 奖励分解；
- `training/artifacts/train-report.json`：`config`（课程计划、双方手感）、`history`、`evaluation`；
- 训练中可随时用客户端 `/gladiator spectate Bot_T1` 看实时表现，退出观战不会打断训练。

## 7. 验收清单

| # | 检查项 | 命令/操作 | 期望 | 结果 |
|---|---|---|---|---|
| 1 | 插件加载 | 看启动日志 | `ready=true`、`api=1.1.0` | ☐ |
| 2 | 运行时连接 | `gladiator status` | `sessions=2` | ☐ |
| 3 | Bot 就绪 | `gladiator list` | 两个 Bot `ready=true`、state=`Alive` | ☐ |
| 4 | 观战进入 | `/gladiator spectate Bot_T1` | 旁观者 + 传送到 Bot 附近 | ☐ |
| 5 | 观战退出 | `/gladiator unspectate` | 模式/位置还原 | ☐ |
| 6 | 观战隔离 | 观战时看 Bot 行为 | Bot 不追观战者 | ☐ |
| 7 | 手感配置 | `/gladiator spawn X cps=20 reaction=100 reach=4.5` | 日志回显对应 profile | ☐ |
| 8 | 重复 spawn | 再 spawn 同名 | 返回“Bot 已就绪，不能重复 spawn” | ☐ |
| 9 | 训练跑通 | 训练脚本 | 回合正常结束、有 review 文件 | ☐ |
| 10 | 状态机 | 训练中让 Bot 死亡（`/kill`） | `Alive → Dead → Respawning → Alive` | ☐ |
| 11 | 分析快照 | `/gladiator analyze` | aliveSeconds、伤害、动作统计非零 | ☐ |
| 12 | 掉线回收 | 杀掉一个 runtime | `sessions`/`bots` 下降，另一个 Bot 不受影响 | ☐ |
| 13 | 疾跑 | 观战 `Bot_T1` 远距离接近时 | 有明显加速（1.3 倍左右），近身后恢复步行 | ☐ |
| 14 | 多场地 | `multi_train.py --pairs 2` | 两组 Bot 在 X 轴相距 60 格各自对打，互不干扰 | ☐ |
| 15 | 对手风格 | `--opponent-mode rusher` / `aimbot` | 对手会主动贴脸/几乎每步命中 | ☐ |
| 16 | 计划与回合（G-620） | 计划设为 `scripted:2` + `rusher:3`，训练回合 2、评估回合 1 | 计划卡显示「共 2 段 · 每回合 5 局 · 训练合计 10 局」；进度卡出现「回合 1/2 · 第 1/2 段」；`report.json` 的 `history` 10 条、`evaluation` 5 条；页面上不存在「阶段计划」「对手模式」控件 | ☐ |
| 17 | 预设（G-621） | 「保存为预设…」→ 刷新页面 → 加载该预设 | 计划与 max-steps/seed/间距等参数恢复；`dashboard/var/presets/*.json` 出现且不含用户名/端口；删除后卡片消失 | ☐ |

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `spawnBot 尚未接线` | 运行时没连上（HELLO 前） | 先启 runtime，再看 `sessions` |
| Bot spawn 后立刻摔死 | 竞技场 y 坐标高于地面（超平坦地面 y=3，站立 y=4） | `--center 0.5,4,0.5` |
| `TIMEOUT: 等待 Bot 登录超过 30s` | 运行时未处理 SPAWN 或 MC 登录失败 | 看 runtime 日志里的 `SPAWN_FAILED` code |
| 客户端卡在旁观者 | 退出时异常 | 重进服务器，或 `/gamemode 0` |
| 训练脚本 RCON 认证失败 | 密码/端口不对，或 `server.properties` 被还原 | 检查 `rcon.password=devpass`、`rcon.port=25575` |
| 训练很慢（<1 step/s） | `wait_action_results` 打开了 | 保持默认 `False` |
| 改了代码但行为没变 | 只 `compileJava`，`server/plugins` 里还是旧 jar | `gradlew build` + 复制 jar + 重启 |

**收尾**：停 Paper 与所有 node 运行时；把 `server/server.properties` 的 rcon 行还原（`git diff` 应为空）。

## 9. 记录模板

```
日期/时间：
服务端：Paper 1.8.8-445 + gladiator-0.1.0.jar（api 1.1.0）
客户端：Minecraft 1.8.9，用户名：
训练：stages=...，max-steps=...，结果（episodes/steps/mean reward）：
观战：进入耗时、位置/朝向是否符合预期、退出是否还原：
异常/日志片段：
结论：
```
