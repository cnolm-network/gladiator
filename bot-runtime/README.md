# bot-runtime — Layer 2

mineflayer 运行时。职责：真实协议登录、物理运动、方块交互、寻路、聊天监听、规则缓存。

## 当前状态

M2 / M3 已落地：通道 A（控制：SPAWN / STATE / NOTICE / QUERY_PVP）与通道 B（观测/动作：:8766）
并行工作；动作分发、节流、规则缓存、寻路、观测编码均已有实现与单测，并在真实 Paper 1.8.8 上端到端验证。

```
bot-runtime/
├─ package.json
├─ config.example.json     配置样例（复制为 config.json 后本地修改，不入库）
├─ scripts/probe.js        能力探针（离线 + 真连两种模式）
├─ scripts/link-smoke.js   通道 A 冒烟：HELLO → WELCOME → PING → PONG
├─ scripts/env-probe.mjs   通道 B 冒烟（最小 Layer 3）：OBSERVATION / ACTION / 幂等 / 限流
├─ src/
│  ├─ index.js             入口：双链路 → 等 SPAWN → mineflayer 登录 → 通道 B 服务端
│  ├─ connection.js        Backoff（1s→30s，±20% 抖动）+ LinkClient（重连/心跳/HELLO）
│  ├─ minecraft.js         launchBot + describeLoginFailure（版本/online-mode/白名单）
│  ├─ actions.js           动作分发（move/look/jump/attack/dig/place/use/talk/idle）
│  ├─ scheduler.js         ≤10 actions/s 滑动窗口 + seq 幂等（含 in-flight 去重）
│  ├─ restriction-cache.js 区域矩形 + TTL 的规则缓存（RESTRICTED / QUERY 共用）
│  ├─ navigation.js        mineflayer-pathfinder 集成；deny 区域不发路径请求
│  ├─ observation.js       7×4×7 blockWindow、self/entities/restrictions 编码
│  └─ link-server.js       通道 B WebSocket 服务端（只绑回环）
└─ test/                   node:test：链路/动作/节流/缓存/观测/退避/失败映射
```

## 使用

```bash
npm install
npm test                              # node --test（无参，自动发现 test/）

# 正式入口：先连 Layer 1，等插件下发 SPAWN 后登录 MC
BOT_USERNAME=Glad_Bot_1 GLADIATOR_BRIDGE_TOKEN=... node src/index.js

# 探针 / 冒烟（不需要完整编排）
node scripts/probe.js
node scripts/probe.js --host 127.0.0.1 --port 25565 --user ProbeBot
node scripts/link-smoke.js --port 8765 --token "与 config.yml bridge.token 一致"
node scripts/env-probe.mjs            # 通道 B：观测 + talk + 幂等 + 限流
```

`src/index.js` 支持环境变量：`MC_HOST`（默认 127.0.0.1）、`MC_PORT`（25565）、`MC_VERSION`（1.8.8）、
`BOT_USERNAME`、`GLADIATOR_BRIDGE_HOST`、`GLADIATOR_BRIDGE_PORT`、`GLADIATOR_BRIDGE_TOKEN`、
`GLADIATOR_ENV_PORT`（8766）、`OBSERVATION_HZ`（10）以及对应的 `--kebab-case` 命令行参数。
`botId` 统一为 `glad-<小写用户名>`，必须与服务端 `BotRegistry.botIdFor` 一致。

登录失败会上报 `SPAWN_FAILED` 并给出可读原因：`VERSION_MISMATCH` / `ONLINE_MODE` /
`NOT_WHITELISTED` / `SERVER_UNREACHABLE` / `LOGIN_FAILED`；链路断开按 1s→2s→…→30s 退避重连，
重连后重新 HELLO，`AUTH_FAILED` / `VERSION_UNSUPPORTED` 不重连。

动作语义见 `docs/protocol.md` §4：Layer 3 通过通道 B 发 `{seq,type,params}`，失败回
`ACTION_RESULT{ok:false,code}`（`RULE_DENIED` / `ACTION_RATE_LIMITED` / `BOT_NOT_READY` /
`ACTION_UNSUPPORTED` …）；同一 seq 重放会直接回放缓存结果并带 `deduped:true`。
规则拒绝（`NOTICE{RESTRICTED}`）会写入 `RestrictionCache`，寻路与 dig/place 都先查该缓存。

真连模式会在 30 秒内输出 `login / spawn / position / health / game.version`，用于确认协议层握手与服务端 `online-mode=false` 配置正确。

## 版本要点

- `mineflayer 4.39.0`：`oldestSupportedVersion=1.8.8`，`testedVersions` 含 `1.8.8`；1.8.9 与 1.8.8 共用协议 47 可用但未列入测试矩阵。**真实服务端是 spigot-1.8.8，配置统一用 `1.8.8`**。
- 1.8.8 的动作细节（放置朝向、物品使用时序）**尚未在真实服务端实测**，M1 逐项验证。
- 集成门禁：Gladiator 是独立插件，不依赖任何竞技场插件；调用方通过 `GladiatorAPI` 发起 Bot，
  并在 `GladiatorBotReadyEvent` 后自行完成传送与套件。见 `docs/api-contract.md`。
- 环境准备（权限组、反作弊豁免等）由**调用方 init** 完成；Gladiator 与 bot-runtime 不绑定任何具体反作弊。见 `docs/api-contract.md` §8。
- 攻击能力由 `entities` 插件提供，mineflayer 4.x 无独立 `attack` 模块。

## 禁止

- 不做奖励计算、不做对局规则判定、不硬编码 WorldGuard 区域名（区域信息只能来自 Layer 1）。
