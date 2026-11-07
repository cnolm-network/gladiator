# bot-runtime — Layer 2

mineflayer 运行时。职责：真实协议登录、物理运动、方块交互、寻路、聊天监听、规则缓存。

## 当前状态

M1 起点：`scripts/link-smoke.js` 已可验证 Layer 1 的 WebSocket 链路；
`src/` 正式模块（connection / actions / scheduler / restriction-cache / navigation / observation / link）仍待建立。

```
bot-runtime/
├─ package.json
├─ config.example.json     配置样例（复制为 config.json 后本地修改，不入库）
├─ scripts/probe.js        能力探针（离线 + 真连两种模式）
├─ scripts/link-smoke.js   通道 A 冒烟：HELLO → WELCOME → PING → PONG
└─ src/                    待建立：connection / actions / scheduler / restriction-cache / navigation / observation / link
```

## 使用

```bash
npm install
node scripts/probe.js
node scripts/probe.js --host 127.0.0.1 --port 25565 --user ProbeBot

# 链路冒烟：需插件已启用并监听 127.0.0.1:8765
node scripts/link-smoke.js --port 8765 --token "与 config.yml bridge.token 一致"
```

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
