# 只读观战（Spectator）

用你自己的 Minecraft 客户端进服，旁观 Bot 训练，全程只读、不参与对局。

## 前置条件

1. 服务端已加载 Gladiator，且 `config.yml` 里 `spectate.enabled: true`（默认）。
2. 至少有一个 Bot 已经 Ready（`/gladiator list` 里 `ready=true`），否则没有可观的目标。
3. 客户端版本与服务端一致（1.8.8 / 1.8.9）。

## 命令

| 命令 | 说明 |
|---|---|
| `/gladiator spectate` | 观战训练会话里的 Bot；没有训练会话时取第一个已就绪的 Bot |
| `/gladiator spectate <username\|botId>` | 观战指定 Bot，例如 `spectate Glad_Bot_1` 或 `spectate glad-glad_bot_1` |
| `/gladiator unspectate` | 退出观战，还原进入前的游戏模式与位置 |

别名 `/gl` 同样可用。

## 权限

| 权限 | 默认 | 说明 |
|---|---|---|
| `gladiator.admin` | op | 命令入口权限（与其它管理子命令一致） |
| `gladiator.spectate` | op | 只读观战权限；可单独授予普通玩家而不给管理权限 |

> 只想让普通玩家观战时，把 `gladiator.spectate` 设为 `true` 即可；但 `/gladiator` 命令本身受
> `gladiator.admin` 约束，需要同时放开或在权限插件里做子命令级控制。

## 行为

1. 进入观战时切换到 **旁观者模式（SPECTATOR）**，传送到目标 Bot 上方 3 格并朝向 Bot。
2. 退出观战时恢复进入前的游戏模式与坐标。
3. 玩家在观战状态下断线：立即恢复游戏模式（不传送），避免下次登录卡在旁观者模式。
4. 插件卸载/服务端关闭：对所有在线观战者做同样的还原。

## 为什么不会干扰训练

- 旁观者模式下无法攻击、放置/破坏方块、拾取物品，也不会被怪物当作目标。
- 训练环境（`training/gladiator_env`）只把**参赛 Bot 用户名**当作目标（`opponents` 过滤），
  观战者即使站在 Bot 面前，也不会被 Bot 追着打、不会污染 `nearest_player` 决策。

## 已知限制

- 传送到 Bot 上方 3 格：如果头顶刚好是低矮空间，客户端会贴在天花板下方，用旁观者飞行挪开即可。
- 多个观战者传送到同一 Bot 会重叠，属正常现象（旁观者之间没有碰撞）。
- 观战不写任何对局状态，也不触发 `pushGame`/`pushPlace`；它只是"看"。
