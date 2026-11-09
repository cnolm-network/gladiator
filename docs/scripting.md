# 脚本与模型规范（v1 草案）

> 状态：**规范先行，能力分批接入**。本文区分「已实现」与「规划中」，未实现的项不得在 UI 上表现为可用。
> 关联：`docs/training-plan-ui.md`（计划与预设）、`docs/training-env-api.md`（训练侧 API）。

## 1. 现状（2026-11-08）

| 能力 | 状态 | 说明 |
|---|---|---|
| 内置对手脚本 | ✅ 已实现 | `random` / `wander` / `idle` / `scripted` / `rusher` / `aimbot`，由 `training/gladiator_env` 的启发式实现 |
| 计划列表（脚本 × 轮次） | ✅ 已实现 | Dashboard 训练表单，`plan: [{opponentMode, episodes}]` |
| 预设（计划 + 参数） | ✅ 已实现 | `dashboard/var/presets/<id>.json`，格式见 `docs/training-plan-ui.md` §3 |
| 自定义脚本文件（可移植） | ⛔ 规划中 | 需要一个 `decide(observation) -> action` 的加载器与沙箱边界 |
| 模型打包 `.gmod` | ✅ 已实现（G-623） | `dashboard/gmod.py` 热扫描 `models/*.gmod`；`modelSupport=true`、训练下拉可选文件模型；详见 `docs/models.md` §8 |

> `modelSupport` 自 G-622 起恒为 `true`（模型注册表可用）；未实现的「自定义脚本文件」
> 与「计划项里的 script/model 字段」不得在 UI 上表现为可选择，避免“看似可选、实际无效”。

## 2. 内置脚本（当前唯一可用的「代码驱动类型」）

六种模式全部在 Layer 2/3 侧执行，Gladiator 插件不参与：

| id | 行为要点 | 建议显示色 |
|---|---|---|
| `idle` | 站立不动，只转身 | `§8` 深灰 |
| `wander` | 随机游走 | `§a` 绿 |
| `random` | 随机动作 | `§7` 灰 |
| `scripted` | 固定套路（接近—攻击—后退） | `§e` 黄 |
| `rusher` | 高压近身，3.4 格内高频攻击 | `§c` 红 |
| `aimbot` | 精确对准，3.9 格内持续输出 | `§d` 淡紫 |

颜色只用于显示：`§` 码由前端映射为 CSS 颜色，不写入训练数据。

## 3. 自定义脚本契约（规划中）

目标：脚本**分离、可移植、可编译**，不硬编码在 Dashboard 里。

```python
# training/gladiator_env/scripts/example_hold.py
from typing import Any, Mapping

SCRIPT_META = {
    "id": "hold_center",
    "name": "Hold Center",
    "color": "§b",
    "version": "1.0.0",
    "description": "站在中心，仅在对手进入 3 格内时攻击",
}

def decide(observation: Mapping[str, Any], ctx: Any) -> str:
    """返回动作空间中的一个动作 id；不得产生副作用、不得联网。"""
    ...
```

约束：

1. 一个脚本一个文件，文件名即 `id` 的默认来源；`SCRIPT_META` 必填 `id` / `name`。
2. `decide` 必须是**纯函数**：只读 `observation` 与 `ctx`（只读视图），返回动作字符串。
3. 不得 import 训练算法、不得读写文件、不得发起网络请求；违规脚本由加载器拒绝。
4. 加载失败（语法错误/签名不符）时，UI 显示该脚本为「不可用」并给出原因，而不是静默回退。
5. 训练侧通过 `--opponent-script <path>`（规划）加载；计划项用 `script` 字段引用脚本 id。

## 4. 模型包 `.gmod`（已实现，G-623）

模型是策略权重 + 元数据的可移植包，本质是 zip。**放进仓库根 `models/` 目录，Web 刷新
就自动出现**（热扫描）；删掉文件即消失。Dashboard 不删除、不改写用户的包。

```
rush-v3.gmod
├─ meta.json          # 必填：id / name / obsDim / actDim
└─ policy.npz         # W1/b1/W2/b2/_meta，与 training/gladiator_train/policy.py 保存的权重同构
```

`meta.json` 示例：

```json
{
  "format": "gladiator.model/v1",
  "id": "rush-v3",
  "name": "Rush v3",
  "color": "§6",
  "version": "0.3.0",
  "obsDim": 19,
  "actDim": 11,
  "createdAt": "2026-11-08T19:00:00",
  "description": "以 rusher 为对手训练 120 局"
}
```

约束：

1. `obsDim` / `actDim` 必须等于 `training/gladiator_env/spaces.py` 的 `OBS_DIM`(19) /
   `NUM_ACTIONS`(11)，且 `policy.npz` 的形状要与 meta 一致；校验失败时该条目标
   `valid=false` + 中文 `error`，**不打断列表**，训练下拉里禁用。
2. 包内不得包含可执行代码（只允许 `meta.json` 与 `.npz`；额外文件只给 `warning`）。
3. 模型只作为**初始化权重**由训练表单显式选择（`--init-policy`），不自动替换正在训练的策略。
4. 文件 id 以文件名为准；列表/接口里用 `gmod:<fileId>` 引用，同名的策略模型与 `.gmod` 可并存
   （裸 id 指向策略模型）。删除/改名由文件系统负责（`DELETE /api/models?id=gmod:<fileId>` 返回 409）。
5. `color` 目前只作元数据展示，**尚未**下发到游戏，见 §7.2 阶段 3。

完整契约（目录配置、缓存、校验失败示例、`/api/models/reload`、gmod-cache）见
**`docs/models.md` §8**。

## 5. 与计划列表的关系

计划项（`docs/training-plan-ui.md` §2）当前只使用 `opponentMode`。自定义脚本与模型接入后：

```json
{ "opponentMode": "scripted", "episodes": 10, "script": "hold_center", "model": "rush-v3" }
```

- `opponentMode` 保留为“代码驱动类型”的基类，`script`/`model` 为可选的更具体实现。
- 字段只增不改：旧客户端不传这两个字段时行为不变。
- **当前状态（G-623）**：`script` / 计划项里的 `model` 字段**仍未实现**；模型是
  “这一次训练整体用哪份初始权重”，由训练表单的「训练模型」下拉选择（含 `.gmod`），
  不与单条计划项绑定。见 `docs/models.md` §5。

## 6. 迁移与验收

- 接入自定义脚本时必须新增 ADR（沙箱边界与危险操作清单），并补 `dashboard/tests` 与 `training` 侧测试。
- `.gmod` 接入已落地（G-623）：维度/形状校验、坏包中文报错且不打断列表、`modelSupport=true`、训练下拉按来源分组并禁用 invalid 项；详见 §4 与 `docs/models.md` §8。

## 7. 脚本 / 模型的颜色（color）现状与接口设计

需求原文：脚本/模型可分离、用户可自由编辑、**每个脚本/模型给一个 color，Minecraft 侧自动用这个 color 当名字前缀**。

### 7.1 核查结论（2026-11-08，G-621 阶段 1 + G-622 阶段 2 落地后的现状）

| 环节 | 状态 | 证据 |
|---|---|---|
| Web 端「脚本列表显示色」 | ✅ 已实现 | `dashboard/training_runner.py` 的 `options()["scripts"][].color`（`§` 码）+ `app.js` 的 `mcColor()` 把 `§x` 映射为 CSS 颜色；计划行、脚本下拉也用同一颜色 |
| Web 端「用户自定义颜色的入口」 | ✅ 已实现（阶段 1） | `dashboard/scripts.py` + `dashboard/var/scripts.json`（schema `gladiator.scripts/v1`）；内置 6 项可改色/改名，自定义项可增删；`GET/POST /api/scripts`、`DELETE /api/scripts?id=` |
| 训练侧 / 运行时读取 color | ⚠️ 部分 | `training/`、`bot-runtime/` 仍不读 color；颜色由 **Dashboard 在训练前** 用 RCON `caller tagcolor <role> <color>` 同步到调用方插件 |
| Minecraft 侧「用 color 当名字前缀」 | ✅ 已实现（经调用方插件） | `caller-sample` 新增 `/caller tagcolor <role> <color|reset>`：写回自己的 `config.yml tags` 并热重载；训练环境照旧 `/caller tag <user> <role>`，所以前缀颜色与 Web 一致 |
| 脚本/模型与计划项分离 | ⚠️ 部分 | 计划项当前只有 `opponentMode`（等价于内置脚本）；模型改为「一次训练选一个初始化权重」，不与单条计划项绑定 |
| 模型注册表 / 训练可切换模型 | ✅ 已实现（阶段 2，G-622） | `dashboard/models.py` + `dashboard/var/models/`（schema `gladiator.model/v1`）、`GET/POST/DELETE /api/models`，训练侧 `--init-policy`；详见 `docs/models.md` |
| `.gmod` 装载（放文件即出现） | ✅ 已实现（阶段 3，G-623） | `dashboard/gmod.py` 扫描仓库根 `models/*.gmod`；`GET /api/models` 每次热扫描、`POST /api/models/reload` 强制重扫；详见 `docs/models.md` §8 |

**如实说明**：阶段 1 已闭环——Web 可改颜色、颜色会经调用方插件进游戏（前提：装了
`caller-sample` 或同类支持 `tagcolor` 的调用方，且 RCON 可用）；阶段 2 的模型注册表、
训练切换，以及阶段 3 的 `.gmod` 装载都已闭环。仍未实现的是「Gladiator 插件自己染名」
与「模型自带颜色下发到游戏」，见 §7.2 阶段 3。

### 7.2 接口设计（阶段 1/2 已落地、阶段 3 部分落地；未实现部分 UI 不得显示为可用）

**阶段 1（已实现，G-621）**：脚本注册表 + 可编辑 color + 训练前同步到游戏。

存储 `dashboard/var/scripts.json`：

```json
{
  "format": "gladiator.scripts/v1",
  "updatedAt": "2026-11-08T21:20:00",
  "items": [
    {"id": "rusher", "name": "Rusher", "color": "§b", "description": "高压近身"},
    {"id": "mine-hold", "name": "Hold Center", "color": "§5", "description": "自定义脚本"}
  ]
}
```

- 校验：`id` `^[a-z0-9][a-z0-9_-]{0,23}$`（大写直接拒绝，不静默改写）；`name` ≤40；`description` ≤200；
  `color` 只接受单个颜色码 `&0–&f` / `§0–§f`，统一存成 `§x`。
- 内置 6 项（`random/wander/idle/scripted/rusher/aimbot`）恒存在，可改颜色/名称/说明，**不可删除**；
  自定义项可增删改。每次写入前备份到 `dashboard/var/scripts-trash/scripts-<时间戳>.json`。
- API：`GET /api/scripts`（含 `builtin` 标记与 16 色调色板）、`POST /api/scripts`（新增/更新）、
  `DELETE /api/scripts?id=`（内置 400、不存在 404）。
- `GET /api/training/options` 的 `scripts[]` 直接来自注册表（内置默认 ← 用户覆盖的合成结果），
  前端脚本列表、计划行、下拉颜色都随之变化。
- **进游戏的颜色来源与优先级**：
  1. `dashboard/var/scripts.json`（Web 配置，最高）
  2. `caller-sample` 的 `config.yml` → `tags:`（服务端默认值）
  3. 插件内置 `ROLE_TAGS` 常量（最后兜底）

  同步时机：Dashboard 在训练进程 spawn 之前，对本次用到的每个 role（`trained` + plan 里的每个
  `opponentMode`）执行一次 `caller tagcolor <role> <color>`；随后训练环境照旧用
  `caller tag <user> <role>` 给 Bot 打标签，颜色即与 Web 一致。RCON 不可用时只跳过同步
  （训练照常启动，原因写进 job.notes）。

> 注意：`tagcolor` 会改写服务端的 `plugins/CallerSample/config.yml`（`tags` 段），这是**调用方**
> 的配置，不是 Gladiator 的；Gladiator 源码仍然零绑定第三方插件。

**阶段 2（已实现，G-622）：模型注册表 + 训练可切换模型**。

原先草案里「颜色真正变成 Minecraft 名字前缀」的部分，已由阶段 1 的
`/caller tagcolor` 落地：**没有**改 `GladiatorAPI`、**没有**新增通道 A 协议消息
（那需要递增 `docs/protocol.md` 的 `v` 并追加 ADR）。当时的 `displayColor` /
`scriptStyles` / bot-runtime 消息草案因此**未采用**，此处不再保留。

- 注册表：`dashboard/var/models/<id>/` 下 `model.json`（schema `gladiator.model/v1`）
  + `policy.npz` 副本；内置 `default` 表示「全新初始化」（无策略文件、不可删、可改名）；
  覆盖/删除前备份到 `dashboard/var/models-trash/`。完整契约见 **`docs/models.md`**。
- API：`GET /api/models`（列表 + default）、`GET /api/models?id=`（详情 + 最近 5 次使用它的 run）、
  `POST /api/models`（从 run / 绝对路径导入）、`POST /api/models/update`、`DELETE /api/models?id=`。
- 训练侧：`train.py` 与 `multi_train.py` 新增 `--init-policy`（另有 `--model-id` / `--model-name`
  只写 report/config）。只把 `W1/b1/W2/b2` 装进本次策略，**不接管 optimizer / updates**；
  文件缺失、npz 损坏或 `obs_dim / n_actions / hidden` 不匹配时，在创建任何环境之前
  打印中文原因并 `exit 2`。
- Dashboard：训练表单「训练模型」下拉来自注册表（`default` + 已注册模型 + 「从 run 导入…」入口），
  启动前校验维度并传绝对路径 `--init-policy`；`run.json`、run 详情、单局详情都回显模型 id/名称。
- **颜色与模型的关系**：模型项本身没有 `color` 字段。Bot 名字前缀颜色仍由
  `scripts.json` 的 role 颜色决定（阶段 1 的三级优先级链），与选用哪个初始化权重无关。

**阶段 3（`.gmod` 装载 ✅ 已实现 G-623；模型自带颜色 ⛔ 未实现，需 ADR）**：

- ✅ **`.gmod` 装载**：`dashboard/gmod.py` 扫描仓库根 `models/*.gmod`（可配置 `gmodDir`），
  校验 `meta.json` 与 `policy.npz` 的维度/形状/完整性，坏包标 `valid=false` + 中文原因、
  不打断列表；每次 `GET /api/models` 热扫描（mtime+size 缓存），`POST /api/models/reload`
  强制重扫；训练选用时解到 `var/gmod-cache/<sha>.npz` 再传 `--init-policy`。
  前端按「策略产物 / 文件模型」分组，invalid 置灰禁选。详见 `docs/models.md` §8。
- ⛔ **模型自带颜色**：`meta.json` 的 `color` 只是元数据（列表/详情展示），**没有**下发到
  Bot 名字前缀；游戏内前缀仍由阶段 1 的 `scripts.json` → `/caller tagcolor` 决定。
  要让模型颜色生效，需要新增「按模型设置显示名/队伍前缀」的调用方契约（或通道 A 消息，
  那要递增 `docs/protocol.md` 的 `v` 并追加 ADR），并明确与 role 颜色的优先级。
  在此之前 UI 明确标注「仅元数据展示」，不得显示为已生效。

**不建议的做法**（明确排除）：Gladiator 插件直接改 `CraftPlayer` 的 `displayName`/`team`
或伪造 `PlayerConnection`——违反项目红线（AGENTS.md「不可接受的做法」）。
