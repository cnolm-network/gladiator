# Gladiator

面向 Minecraft 1.8.8（协议 47）的**独立 Spigot 插件**：对外提供 `GladiatorAPI`，让竞技场插件可以发起真实协议登录的 AI Bot 并完成对战 / 训练。配套一个 Node.js Bot 运行时与一个 Gymnasium 训练环境。

- 主类：`top.cnolm.gladiator.GladiatorMain`
- 架构：四层解耦（Java 编排 / Node 运行时 / Python 训练 / 策略后端）
- 定位：**不依赖任何竞技场插件**。由竞技场插件调用 `GladiatorAPI.spawnBot()` 发起 Bot，并在 `GladiatorBotReadyEvent` 后自行完成传送、给予套件等对局"壳子"工作。换竞技场插件只改调用方，不改 Gladiator 源码。
- 反作弊与环境准备（权限组、区域放行、白名单）属于**调用方 init 阶段**；Gladiator 不检测、不配置、不硬编码任何具体反作弊插件。
- 现状：M0 完成（对外 API 契约 + 最小可用实现 + 可构建工程）；Bot 真实登录与 WebSocket 通道属于 M1，尚未实现。

## 目录

```
├─ src/main/java/top/cnolm/gladiator/
│  ├─ api/                    ★ 对外契约（唯一可被调用方 import 的包）
│  │  └─ event/              5 个标准 Bukkit 事件
│  ├─ core/                  API 实现、状态镜像、指标累加器
│  └─ command/               管理命令
├─ src/main/resources/        plugin.yml / config.yml
├─ gradle/
│  ├─ wrapper/                Gradle 8.10.2 wrapper
│  └─ gradle-daemon-jvm.properties   daemon 固定 JDK 17
├─ bot-runtime/              Layer 2：mineflayer 运行时（Node ≥22）
├─ training/                 Layer 3：Gymnasium 环境（Python ≥3.10）
└─ docs/
   ├─ api-contract.md         ★ 对外 API 契约（集成唯一入口）
   └─ protocol.md             Layer 1↔2↔3 协议 v1
```

## 快速开始

### 1. 构建

```bash
./gradlew build
# 产物：build/libs/gladiator-0.1.0.jar
```

Gradle daemon 的 JVM 由 `gradle/gradle-daemon-jvm.properties` 固定为 **JDK 17**：Gradle 8.10.2 最高只支持到 JDK 23，用 Java 25 会报 `Unsupported class file major version 69`。编译目标由 `build.gradle` 的 **Java 8 toolchain** 决定，与 daemon 版本解耦，无需手动设置 `JAVA_HOME`。

把产物放进服务端 `plugins/` 即可；`WorldGuard` 是可选 softdepend，缺失时规则查询降级。

### 2. 调用 API

```java
GladiatorAPI api = GladiatorProvider.get();          // 或 Bukkit ServicesManager
if (api == null || !api.isReady()) return;

api.spawnBot(BotSpec.of("Glad_Bot_1", "gladiator-bot"));

@EventHandler
public void onBotReady(GladiatorBotReadyEvent e) {
    // 壳子工作全在调用方：传送 + 套件
    e.getPlayer().teleport(arenaSpawn);
    Kit.apply(e.getPlayer(), mode);
    api.pushGame(mode, Users.of(human), e.getBot());
}
```

完整签名、事件、线程约定与调用方 init 职责见 [`docs/api-contract.md`](docs/api-contract.md)。

### 3. Bot 运行时与训练环境

```bash
cd bot-runtime && npm install && node scripts/probe.js
cd training && python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| 服务端 | Spigot / Paper 1.8.8 | 协议 47；`online-mode=false` 便于本地开发 |
| JDK（编译） | 8 | 产物 bytecode 52 |
| JDK（Gradle 运行） | 17 | 由 daemon JVM criteria 固定 |
| Gradle | 8.10.2 | wrapper 自带 |
| Node.js | ≥22 | mineflayer 4.39.0 |
| Python | ≥3.10 | gymnasium / numpy / websockets |
| WorldGuard | 6.2（可选） | 规则查询桥接 |

## 关键约束

- 不在服务端 JVM 内伪造 `PlayerConnection` / `CraftPlayer`；Bot 必须以真实协议登录。
- `Mode` 只读状态 + `eliminate(User)` 标记；Gladiator 不实现对局规则、不做传送/套件/选图/匹配/结算。
- 源码不 import / depend / softdepend 任何竞技场插件；只监听标准 Bukkit 事件。
- 训练算法不进 Java 插件，Java 侧不发起任何外部 HTTP 请求。
- 全链路只监听 `127.0.0.1`；`online-mode=false` 仅限本机开发。
- API 只增不改（语义化版本）；协议变更需同步 `docs/protocol.md`。

## 文档

- [`docs/api-contract.md`](docs/api-contract.md) — 对外 API 契约（权威）
- [`docs/protocol.md`](docs/protocol.md) — Layer 1↔2↔3 消息协议 v1
