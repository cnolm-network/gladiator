# 身份与动作空间：两个结构性问题的方案

> 写于 2026-11-10。记录两个「不是 bug、但会限制上限」的问题，以及可执行的改造路径。
> 结论先放前面：**身份问题用「命名空间 + 专用 Bot 通道」解，动作问题用「能力表 + 槽位/使用分离」解**，
> 两者都不需要推翻现有架构，也不需要重训历史模型（v1 与 v2 可并存）。

---

## 一、Bot 只能离线登录、不能与真实玩家重名

### 1.1 现状与根因

| 事实 | 位置 | 后果 |
|---|---|---|
| `auth = 'offline'` | `bot-runtime/src/minecraft.js` | 不向 Mojang 认证，UUID 由服务端按 `OfflinePlayer:<name>` 生成 |
| `online-mode=false` | `server/server.properties` | 服务端接受离线登录；真实玩家也拿离线 UUID |
| `BotSpec.NAME = [A-Za-z0-9_]{3,16}` | `api/BotSpec.java` | 只校验格式，**不校验是否已被占用** |
| 服务端按名字判重 | Spigot 登录流程 | 与在线玩家同名 → 顶号/被踢；与历史 usercache 同名 → 继承旧数据 |
| `ONLINE_MODE` 错误码 | `minecraft.js::describeLoginFailure` | 只是"失败后提示"，没有预检 |

本质上：**离线模式下「用户名 = 身份」，而用户名是全局共享的稀缺资源**。只要 Bot 和真人共用命名空间，
就一定会有重名与冒名问题；这不是 Gladiator 能靠重试解决的。

### 1.2 四条路线

| 路线 | 做法 | UUID 来源 | 重名风险 | 成本 | 适用 |
|---|---|---|---|---|---|
| **A. 命名空间 + 专用通道**（短期推荐） | Bot 名强制前缀（如 `AI_`），并在 spawn 前做占用检查；Bot 与真人分属不同后端（或不同子服） | 离线 UUID | 前缀内可控，前缀外为 0 | 极低 | 单机训练/内网对局 |
| **B. 正版账号池** | `auth: 'microsoft'`（device-code 首次登录，之后刷新令牌），每个 Bot 一个微软账号；服务端 `online-mode=true` | Mojang 真实 UUID | 结构上不可能重名 | 中（账号、2FA、限流） | 对外服、需要真实 UUID |
| **C. 代理转发**（中期推荐） | Velocity/BungeeCord 在前：真人走正版认证，Bot 走离线名；后端 `online-mode=false` + 转发密钥 | 代理分配（`uuid-forwarding`） | 由代理命名空间管理 | 中（多一跳、要运维代理） | 真人与 Bot 同服、又不想买账号 |
| D. 服务端自建认证 | 插件伪造登录/改 UUID | 自造 | 高（NMS/反作弊冲突） | 高 | 不建议 |

**推荐组合**：现在用 A 把风险关掉；等需要「真人与 Bot 同服」时切 C；只有必须走纯正版服时才上 B。
D 违反本仓库「不伪造 PlayerConnection」的硬约束，直接排除。

### 1.3 A 路线落地清单（1 天，不改协议）

1. **命名空间**
   - 新增配置 `bot.name-prefix`（默认 `AI_`，可空）。`BotSpec` 校验：非空前缀时必须匹配
     `^<prefix>[A-Za-z0-9_]{1,13}$`，总长仍 ≤16；`BotRegistry.botIdFor()` 保持 `glad-<小写名>` 不变。
   - `GladiatorCommand spawn` 与 Dashboard 训练表单都走同一校验，报错文案给出"去掉前缀/换名前缀"的建议。
2. **占用检查（服务端侧，spawn 之前）**
   - `BotSpawnService.spawn()` 增加 `nameTaken(username)`：
     `Bukkit.getPlayerExact(name) != null` → 占用；
     `Bukkit.getOfflinePlayer(name).hasPlayedBefore()` → 占用（历史玩家）；
     可选：`usercache.json` 存在同名 → 占用（覆盖"从没上线但被其他插件预注册"的情况）。
   - 命中后直接派发 `GladiatorBotFailedEvent`，`reason` 前缀 `NAME_TAKEN: `，**不下发 SPAWN**（不浪费登录）。
   - 新增 `SPAWN_FAILED` code：`NAME_TAKEN`（协议 §3 追加，向后兼容）。
3. **登录期兜底**
   - `minecraft.js` 的失败映射补 `you are already logged in|logged in from another location|name is already taken` → `NAME_TAKEN`；
   - 若服务端 `online-mode=true` 而 Bot 用离线身份，映射到现有 `ONLINE_MODE`，提示里同时给出
     「改用 B 路线（正版账号）」与「后端改 offline-mode + 代理转发（C 路线）」两条出路。
4. **运行时侧不猜**
   - 不做"自动改名重试"：改名会让服务端 usercache、统计、观战链接全部错位；失败就上报，由调用方决定。
5. **验收**
   - 真人 `lvsm_` 在线时 spawn `lvsm_` → 立即 `NAME_TAKEN`，控制台无登录尝试；
   - spawn 一个曾经上线过、当前离线的名字 → 同样 `NAME_TAKEN`；
   - 前缀不合规 → 命令层直接拒绝，不发 SPAWN。

### 1.4 B/C 路线要点（等需要时再做）

- **B（正版）**：`minecraft.js` 增 `--auth microsoft --profiles-folder <dir>`；每个账号首次 device-code 授权，
  令牌落盘在独立 profiles 目录（**不进仓库**）；`BotProfile` 增 `account` 字段（只存账号别名，不存密码）；
  启动时用 `GET /api/bots/accounts` 看令牌健康度。注意微软账号的并发登录限制与"同一账号不能多开"。
- **C（代理）**：后端 `online-mode=false` + `bungeecord: true` + 转发密钥；代理侧给 Bot 分配
  `uuid-forwarding` 模式；`BotHandle` 以 `Player.getUniqueId()`（代理给的）为准，不再依赖离线 UUID；
  `playerUuid` 在 WELCOME 里自然变成代理 UUID，训练环境无需改动。
- 无论 B/C，`BotHandle` 的三向索引（botId/UUID/username）已就绪，不需要改 API。

---

## 二、动作空间是「语义动作」，新物品就要重新适配

### 2.1 根因

现在的 11 个动作是**宏动作**：`eat_gapple`、`use_rod`、`block`、`switch_sword` 各自把
「选槽 + 右键 + 按住 N tick + 切回剑」写死在 `spaces.py::decode_action` 里。于是：

- 加一个「喝药水 / 搭方块 / 用弓」就要加一个动作 + 改 `NUM_ACTIONS` + 重训（旧模型作废）；
- 策略学到的是「按第 9 个动作键」而不是「用掉手上的消耗品」，跨物品无法迁移；
- 动作数随物品线性膨胀，而物品表是开放的。

### 2.2 目标动作空间 v2（物品无关 + 可掩码）

把动作拆成**正交的意图**，物品身份只进观测、不进动作：

| 头 | 取值 | 说明 |
|---|---|---|
| `move` | 前/后/左/右 各 -1/0/1（或 5 档） | 与现在的 `move` 一致，保留按住语义 |
| `look` | dyaw ∈ {-30,-10,-3,0,3,10,30}°，dpitch ∈ {-15,-5,0,5,15}° | 替代"内部算朝向"，让策略自己学瞄准 |
| `attack` | 0/1 | 左键；是否命中由 reach/冷却决定 |
| `use` | 0/1/2（不按 / 单击 / 按住至松开） | 右键的**唯一**入口 |
| `jump` / `sneak` / `sprint` | 0/1 | 与 move 正交 |
| `slot` | 0..8（+ offhand） | 热键栏选槽，**唯一的物品切换入口** |
| `drop`（可选） | 0/1 | 扔掉手上物品（清背包/避免误吃） |

这样「吃金苹果」= `slot=苹果槽, use=按住 1.6s`；「甩鱼竿」= `slot=鱼竿槽, use=单击`；
「搭方块」= `slot=方块槽, use=右键 + 需要 place 上下文`。**新增物品零改代码**。

### 2.3 物品能力表（新增文件，不改训练代码）

`training/gladiator_env/items.json`（也供 bot-runtime 读）：

```json
{
  "minecraft:golden_apple": {"category": "food", "use": {"mode": "hold", "ticks": 32},
                             "cooldownTicks": 0, "stackGoal": 5},
  "minecraft:fishing_rod":  {"category": "tool", "use": {"mode": "click", "ticks": 2},
                             "cooldownTicks": 10, "projectile": true},
  "minecraft:diamond_sword":{"category": "sword", "use": {"mode": "hold_block", "ticks": 5},
                             "reach": 3.99},
  "minecraft:bow":          {"category": "bow", "use": {"mode": "charge", "ticks": 20},
                             "projectile": true}
}
```

- 匹配支持正则（`*_sword`、`*_potion`），未知物品落到 `category: "misc"`，不报错；
- 动作合法性由该表 + 背包状态算 **action mask**：没选中可 `use` 的物品 → `use` 被屏蔽；
  超 reach → `attack` 屏蔽；空槽 → `slot` 屏蔽。策略不会把容量浪费在无效动作上。

### 2.4 观测扩展

- 热键栏 9 槽 × 每槽 `{category one-hot(8), count/16, durability, cooldown}` ≈ 9×12 = 108 维；
  当前只给"手持物品 + 苹果数"，新物品完全不可见。
- 手持/对手手持的 `category` one-hot（不是 item id），保证开放词汇；
  若确实需要区分具体物品，用 **feature hashing**（16 桶）而不是 one-hot。
- 保留现有 19 维作为 `v1` 观测，新环境用 `--observation v2` 切换。

### 2.5 与现有宏动作的关系

- `v1`（11 宏动作）保留：它是 stage 0 的冷启动基线，也是回归对照；
- `BotProfile.assist`（W-tap / 自动鱼竿 / 自动格挡 / 3D 对准）继续留在**执行层**，
  不进入动作空间——"机制"不该让策略重新学，策略只学"什么时候用"；
- 训练环境新增 `--action-space v1|v2`；模型元数据记录 `actionSpace`，
  `.gmod` 载入时维度/动作空间不匹配 → 明确报错（已有 obsDim/actDim 校验，扩一个字段即可）。

### 2.6 课程与验收

| 阶段 | 动作空间 | 对手 | 过线条件 |
|---|---|---|---|
| 0 | v1 | 静止假人 | 命中率 > 70%（现状基线） |
| 1 | v2（只开 slot/use/attack，movement 固定） | 静止假人 + 随机掉落物品 | 正确选择并使用新物品 > 90% |
| 2 | v2 全量 + mask | scripted / rusher | 对 scripted 胜率 > 50%，空挥率 < 35% |
| 3 | v2 全量 | 对手池（rusher/aimbot/保守） | 各 > 45% |

**关键验收**：往 `items.json` 加一条 `minecraft:potion` 能力项 + 训练场 `give` 药水，
**不改任何 Python/JS 代码**就能让 Bot 学会用它——这是 v2 是否成立的一票判据。

---

## 三、排期建议

| 优先级 | 事项 | 工作量 | 阻塞关系 |
|---|---|---|---|
| P0 | 身份 A 路线（前缀 + 占用检查 + `NAME_TAKEN`） | 1 天 | 无；不做则训练数据可能被"冒名真人"污染 |
| P0 | A1–A9 真机验收（见 `docs/next-steps.md`） | 1–2 天 | 无 |
| P1 | 动作空间 v2 + 能力表 + mask | 3–5 天 | 依赖观测 v2；建议先做 stage 1 验证 |
| P1 | 观测 v2（9 槽 + category one-hot） | 2 天 | 与 v2 同步 |
| P2 | 正版账号池 / 代理转发（B/C） | 按需 | 只有在"真人与 Bot 同服"时才需要 |
| P2 | feature hashing 物品嵌入 | 1 天 | v2 之后，用于开放词汇 |

先做「身份 A + A1–A9」，再动动作空间：身份问题会让所有训练数据不可信，动作空间只影响上限。
