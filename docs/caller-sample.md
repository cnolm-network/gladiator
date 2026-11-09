# 示例调用方（caller-sample）— G-506 验收

Gladiator 的定位是**独立插件 + 对外 API 提供方**。判断"独立"是否成立，不能只看编译期：
必须让**另一个插件**在同一个服务端上通过公开 API 跑通一次对局。

`caller-sample/` 就是这个"另一个插件"。它：

- 在 `plugin.yml` 里 `depend: [Gladiator]`；
- 只 import `top.cnolm.gladiator.api.*` 与 `example.*` 里的两个示例调用方；
- **不修改 Gladiator 的任何源码、配置或 `plugin.yml`**。

## 两个调用方

| 命令 | 调用方类 | 包 | Mode | 特点 |
|---|---|---|---|---|
| `/caller arena <user> [x y z]` | `example.MockArenaCaller` | `top.cnolm.gladiator.example` | `DeathmatchMode` | 死斗，不重生；Ready 后传送到指定点并登记会话 |
| `/caller duel <a> <b>  [x y z]` | `example.arena.DuelCaller` | `top.cnolm.gladiator.example.arena` | `DuelMode` | 决斗，允许重生 + 5 分钟时限；两个 Bot 都就绪后才 `pushGame` |
| `/caller reset` | — | — | — | 注销当前调用方的事件监听 |
| `/caller info` | — | — | — | 打印 API 版本与就绪状态 |

两个调用方分属不同包、使用不同 `Mode` 实现——这正是"换插件只改调用方"的验证点。

## 构建

```powershell
.\gradlew.bat :caller-sample:jar --offline
Copy-Item caller-sample\build\libs\caller-sample.jar server\plugins\ -Force
```

`caller-sample` 是独立子工程，产物**不进** Gladiator 的 shadowJar；它只是示例与验收载体。

## 验收步骤（真实服务端）

前置：`server/plugins/` 里同时有 `gladiator-0.1.0.jar` 与 `caller-sample.jar`；启动 1~2 个
`bot-runtime` 进程（每个 Bot 一个，通道 B 端口 8766 起）。

```text
[20:13:45 INFO]: [CallerSample] Loading CallerSample v0.1.0
[20:13:47 INFO]: [CallerSample] Enabling CallerSample v0.1.0
[20:13:47 INFO]: [CallerSample] 接入 Gladiator api=1.1.0 ready=true
```

### A. `MockArenaCaller` + `DeathmatchMode`

```text
> caller arena Bot_D1 0.5 4 0.5
[Gladiator] Bot ready: glad-bot_d1 user=Bot_D1 uuid=...
> gladiator game
[Gladiator] games=1 training=0
  [game] game-1 mode=deathmatch users=Bot_D1(bot),Bot_D1
```

### B. `DuelCaller` + `DuelMode`

```text
> caller duel Bot_E1 Bot_E2 4 4 4
[CallerSample] [duel] 已受理 req-2 / req-3
[CallerSample] [duel] Bot 就绪 Bot_E1（1/2）
[CallerSample] [duel] Bot 就绪 Bot_E2（2/2）
[Gladiator] pushGame game-2 mode=duel users=2
[CallerSample] [duel] 对局开始 session=game-2
> gladiator game
  [game] game-2 mode=duel users=Bot_E1(bot),Bot_E2(bot)
```

### C. 会话随玩家下线回收

停掉 runtime 后：

```text
> gladiator list
[Gladiator] bots=0 pending=0
> gladiator game
[Gladiator] games=0 training=0
```

## 验收结论

- 两个不同包、不同 `Mode` 的调用方在**同一个 Gladiator 实例**上各自跑通；
- 期间 Gladiator 源码零改动（只新增 `caller-sample/` 与 `settings.gradle` 的 `include`）；
- 会话生命周期（Ready → pushGame → 玩家下线回收）由 Gladiator 自身完成，调用方只提供传送与 `Mode`。
