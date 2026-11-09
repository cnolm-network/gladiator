# 训练计划（Dashboard 训练表单）设计与契约

> 版本：v2（2026-11-08）
> 变更：v2 增加 reward 预设（§6）、明确 evalRounds/evalEpisodes 口径、多场地无评估阶段、
> 训练预设 fields 校验与覆盖备份、预设卡片文案与实际一致（§4.5）。
> 适用范围：`dashboard/static/*` 前端与 `dashboard/*.py` 后端。
> 变更规则：字段只增不改；破坏性变更需同步 `TASKS.md` 与 `HANDOVER.md`。

## 1. 问题与目标

旧表单同时提供「阶段计划」（`tr-stages` 文本框，`idle:2,rusher:2`）与「对手模式」（`tr-opponent` 下拉），
两者语义重叠且都能编辑；更严重的是启用「阶段计划」后，表单里的「回合数」不再生效
（总回合由计划求和），用户无法从界面判断哪个值才是权威。

**目标**：删除这两个控件，改为**唯一的有序计划列表**。列表每项是一个
`{对手模式, 轮次}`，支持新增、行内修改、上下移动 / 拖拽排序、删除。

**三层语义**（用户 2026-11-08 确认）：

```
回合（epoch）= 把计划完整执行一遍
  └ 计划（plan）= 有序的若干段
      └ 轮次（episode）= 一局对局，直到一方出局或步数耗尽
```

- 训练总对局 = `训练回合 × Σ(轮次)`；每个训练轮次结束更新一次参数。
- 评估总对局 = `评估回合 × Σ(轮次)`；贪心、不学习、不更新参数。
- 表单只有一个可编辑的「训练回合」和「评估回合」，**没有**独立的“总回合数”输入。

## 2. 数据模型

计划项（JSON）：

```json
{ "opponentMode": "rusher", "episodes": 10 }
```

- `opponentMode`：必填，取值 `random | wander | idle | scripted | rusher | aimbot`
  （这些就是当前训练侧支持的全部「代码驱动类型」，等价于内置脚本）。
- `episodes`：必填，整数 1..500。
- 计划最多 50 项，总回合上限 5000。

训练请求（`POST /api/training/start`）：

```json
{
  "kind": "train",
  "users": ["Bot_T1", "Bot_T2"],
  "ports": [8770, 8771],
  "plan": [{"opponentMode": "scripted", "episodes": 10},
           {"opponentMode": "rusher", "episodes": 10}],
  "epochs": 2,
  "evalEpochs": 1,
  "rewardPreset": "default",
  "maxSteps": 200, "seed": 7,
  "autoSpawn": true, "autoCleanup": true,
  "center": "0.5,4,0.5", "spacing": 6, "prepSeconds": 3
}
```

- `epochs`：训练回合数（1–200），默认 1；每回合完整按 `plan` 执行一遍。
- `evalEpochs`：评估回合数（0–50），默认 1；同样按 `plan` 走，但贪心不更新。
  **仅单场地 `train` 支持**：`kind=multi` 时该字段必须缺省或为 0，否则返回 400
  「多场地训练不支持评估回合」（`multi_train` 没有评估阶段，静默忽略会让用户误以为生效）。
- `rewardPreset`：reward 预设 id（见 §6），缺省则不传 `--reward-preset`，训练侧用内置默认权重。
- 训练总对局 = `epochs × Σ(plan.episodes)`，由后端计算并写入 `params.totalEpisodes`。

**响应里的三个口径（不再混用同名）**：

| 字段 | 含义 |
|---|---|
| `params.evalEpochs` | 评估回合数（与请求同名） |
| `params.evalRounds` | 每个评估回合的对局数 = `Σ(plan.episodes)` |
| `params.evalEpisodes` | 评估总对局数 = `evalEpochs × evalRounds` |
| `params.evalSupported` | `false` 表示该任务类型没有评估阶段（多场地） |
| `params.rewardPreset` | `{id, name, path, applied}`；`applied=false` 表示本次任务没有把预设传进命令行 |

**兼容**：仍接受旧的 `stages` 字符串与 `opponentMode` + `episodes`；三者同时存在时优先级为
`plan` > `stages` > `opponentMode + episodes`。旧字段 `evalEpisodes` 仍被接受，等价于 `evalEpochs`。
新前端只发送 `plan` + `epochs` + `evalEpochs`。

`GET /api/training/options` 新增：

| 字段 | 说明 |
|---|---|
| `scripts` | 内置脚本列表 `[{"id","name","builtin","color"}]`，`color` 为 MC 颜色码（如 `§e`） |
| `models` | 已装载模型；当前为空数组 |
| `modelSupport` | `false` 表示模型装载能力尚未接入训练侧 |
| `defaults.plan` | 默认计划 `[{"opponentMode":"scripted","episodes":4}]` |
| `rewardPresets` | reward 预设摘要列表（见 §6.2），与 `GET /api/reward-presets` 同构 |
| `rewardPresetSchema` | 编辑 UI 用的 schema 元数据：`baseKeys` / `conditionKeys` / `units` / `limits` |
| `defaults.rewardPreset` | 默认 reward 预设 id（`default`） |

## 3. 预设（Preset）

预设是**计划 + 少量训练参数**的 JSON 快照，落盘在 `dashboard/var/presets/<id>.json`，
不包含用户名、端口、路径（与机器绑定，换机后应失效）。

```json
{
  "format": "gladiator.training-preset/v1",
  "name": "Preset-1",
  "createdAt": "2026-11-08T19:30:00",
  "updatedAt": "2026-11-08T19:30:00",
  "plan": [{"opponentMode": "scripted", "episodes": 10}],
  "planText": "scripted:10",
  "fields": {"epochs": 1, "maxSteps": 200, "seed": 7, "evalEpochs": 1,
             "center": "0.5,4,0.5", "spacing": 6, "prepSeconds": 3,
             "autoSpawn": true, "autoCleanup": true}
}
```

`fields` 白名单与范围（服务端强校验，非法值返回 400 + 中文原因）：

| 字段 | 范围 | 说明 |
|---|---|---|
| `epochs` | 1–200 | 训练回合数 |
| `evalEpochs` | 0–50 | **评估回合数**（旧名 `evalEpisodes` 读入时自动映射，写回一律用新名） |
| `maxSteps` | 1–5000 | 每回合最大步数 |
| `seed` | 0–2^31-1 | 随机种子 |
| `center` | `x,y,z` | 归一化为 `1,4.5,2` 这种紧凑写法 |
| `spacing` | 1–20 | 场地间距 |
| `prepSeconds` | 0–30 | 开局准备期 |
| `autoSpawn` / `autoCleanup` | 布尔 | 预设里存布尔值 |

`plan` 的限制同样在写入前校验：每项 `episodes` 1–500、项数 ≤50、总回合 ≤5000。
未知 `fields` 键被忽略（向前兼容），已知键的非法值直接拒绝——预设不能成为绕过
`/api/training/start` 校验的后门。

**可恢复副本（软删除）**：两处都在 `dashboard/var/presets-trash/`，不进 git、不自动清理：

| 动作 | 产物 |
|---|---|
| `POST /api/presets/delete` | `<id>-<YYYYmmdd-HHMMSS>.json`（原文件被移走） |
| 同名 `POST /api/presets/save`（覆盖） | `<id>-overwrite-<YYYYmmdd-HHMMSS>.json`（写入新内容**之前**复制旧版本） |

恢复方式：把文件移回 `dashboard/var/presets/` 并去掉后缀里的时间戳即可（id 必须符合
`[A-Za-z0-9_-]{1,64}`）。备份失败时整次写入都会失败，不会出现“新内容写好了但旧版本没了”。

接口：

| 方法 | 路径 | 请求 | 响应 |
|---|---|---|---|
| GET | `/api/presets` | — | `{ok, presets:[{id,name,plan,totalEpisodes,planText,fields,createdAt,updatedAt}]}` |
| POST | `/api/presets/save` | `{name, plan, fields?}` | `{ok, preset}`；同名覆盖 |
| POST | `/api/presets/load` | `{id}` | `{ok, preset}` |
| POST | `/api/presets/delete` | `{id}` | `{ok, deleted}` |

## 4. UI 行为

布局（参考 `training-goals-list-preview.png`）：

1. **左栏：脚本 / 模型**
   - 列出 `options.scripts`，每项用其 `color` 渲染（`§` 码映射为 CSS 颜色，不显示原始码）。
   - 内置项正常字重；非内置（未来自定义脚本）斜体。
   - 模型下拉来自 `options.models`（含 `.gmod` 文件模型，按「策略产物 / 文件模型」分组）；
     `valid=false` 的项禁用并显示中文校验原因。
2. **主列表（计划）**
   - 每行：拖拽手柄、脚本名（着色）、轮次输入、删除按钮。
   - 拖拽排序（HTML5 draggable），移动后立即重算总回合。
   - 轮次为空或 ≤0 时行标红且禁止开始训练。
3. **添加一条**
   - 训练内容下拉（`options.scripts`）+ 对手模式与训练内容是同一概念，只保留一个下拉，避免重复。
   - 轮次输入 + 「添加到计划」。
4. **右侧：训练计划**
   - 显示 `planText`（`mode:episodes,...`，可复制）。
   - 「清空计划」、 「从字符串加载」（解析失败提示且不覆盖原计划）。
   - 只读展示「共 N 段 / 总回合 M」。
5. **预设**
   - 「保存为预设…」弹窗输入名称 → `POST /api/presets/save`。
   - 预设列表卡片显示名称与「N 段 · 总回合 M」（当前不显示模型/脚本数：模型能力未接入，
     脚本数恒等于 6 个内置项，写了反而是噪声）。
   - 点击「加载」回填计划与 `fields`，「删除」按钮 → `POST /api/presets/delete`。
6. **回合**
   - 「训练回合（每回合跑一遍计划）」可编辑，1–200，默认 1；「评估回合（训练后，贪心不学习）」可编辑，0–50，默认 1。
   - `kind=multi` 时**隐藏**评估回合字段并显示说明，提交时也不发送 `evalEpochs`
     （后端仍会拒绝 >0 的请求，双保险）。
   - 计划卡只读显示 `共 N 段 · 每回合 M 局 · 训练合计 K 局`，其中 `M = Σ(轮次)`、`K = 训练回合 × M`。
   - 进度卡显示 `回合 X/Y · 第 i/n 段 · 本段 a/b 局`；回合切换时阶段计数重置。
7. **Reward 预设**
   - 「Reward 预设」下拉（默认 `default`）+「编辑预设…」按钮，详见 §6。
   - 进度卡显示本次 run 使用的预设名；`applied=false` 时追加「本次未应用」。
8. **训练模型 + 持久化训练（G-624）**
   - 「训练模型」下拉是**这次训练的权重起点**，与计划项无关（一条计划不绑定一个模型）。
     `default` 是可持久化默认模型：首次写回后生成 `policy.npz`，之后选它即接着上次继续练。
   - 旁边「持久化训练（训练结束写回该模型）」开关 **默认开启**。开启时：
     - 启动前对目标模型加写锁（冲突返回 409）；
     - 只有训练**正常完成**（`finished`）才把 `run/policy.npz` 写回所选模型并累加局数/步数；
     - `failed` / `stopped` / 启动前失败一律不写回，产物仅留在本次 run 里。
   - 关闭开关 = **只读起点**：仍可加载该模型的权重开训，但结束不写回，产物只在 `runs/<id>/`。
     这是「从零重开」或「拿现有模型试一组参数不想污染它」的方式。
   - 选中 `.gmod` 文件模型时开关自动禁用并提示「文件模型只读，可先导入为策略模型」；
     旁边出现「导入并继续训练」按钮，调 `POST /api/models {name, gmodId}` 生成可写回的策略模型，
     导入成功后自动选中新 id 并恢复开关。
   - `kind=multi` 时表单多出「权重融合（多场地→唯一权重）」下拉：`mean`（等权，默认）/
     `weighted`（按步数加权）/ `none`（不融合）。开关**可用**（默认开）但依赖融合：
     `mean|weighted` 时允许持久化，写回的是融合后的 `run/policy.npz`
     （`job.persist` 带 `merged:true` / `mergePairCount`）；
     `merge=none` 时后端强制 `persist=false` 并给中文 `persistWarning`，开关同时禁用。
     运行期融合失败（无可用场地权重）会自动降级为不写回，并在进度卡显示原因，训练本身仍算完成。
     详见 `docs/models.md` §10。
   - 进度卡 / run 详情显示 `persist.ok` 与目标模型；写回失败会 toast 中文原因，但**不会**把
     job 标记为 failed（训练本身已完成）。
9. **模型可视化入口（G-625）**
   - 「模型」页每行末尾「可视化」按钮打开可视化抽屉；表格另有「复制链接」生成直达 URL
     `/?model=<id>#model-viz`，可直接截图/分享。
   - 抽屉内 9 张卡片（基本信息、权重直方图与范数、备份差异、两条 reward 曲线、效果指标、
     对手分组、动作分布、累计统计表）；缺数据的卡片显示「暂无数据」，不报错也不空白。
     字段与数据来源见 `docs/models.md` §11。

## 5. 边界

- 模型装载（`.gmod`）与自定义脚本文件尚未接入训练侧；前端不得伪造可选项，
  只能显示「待接入」，脚本编写规范见 `docs/scripting.md`（后续批次）。
- 预设只做本地持久化，不做跨机同步。
- **多场地任务没有评估阶段**（`multi_train` 无 `--eval-epochs`）：UI 隐藏字段、后端拒绝 >0。
- **多场地任务支持 reward 预设**（G-620 修复）：`multi_train.py` 已注册 `--reward-preset`，
  Dashboard 与单场地一样传绝对路径，`params.rewardPreset.applied = true`。

## 6. Reward 预设（`gladiator.reward-preset/v1`）

权威 schema 在 `training/gladiator_env/reward_presets.py`；Dashboard 侧
`dashboard/reward_presets.py` **只读引用**它的常量（`SCHEMA` / `BASE_FIELDS` /
`DEFAULT_BASE` / `CONDITION_KEYS` / `UNITS`），并有单测比对两边的常量集合，
避免 Web 与训练各写一份规则后漂移。

### 6.1 权重语义（与用户约定一致）

- **走位零惩罚**：`inactive` 默认 `0.0`（消极不扣分），且只有显式设为非 0 才生效。
- **长时间不动才扣分**：`still` 为负值（默认 `-0.1`），仅当连续 3 秒水平位移
  `< 0.05` 格时按秒扣分（`STILL_AFTER_SECONDS=3.0`、`STILL_EPSILON=0.05`，见
  `training/gladiator_env/shaping.py`）。走动本身不产生任何负奖励。
- 其他扣分项（`waste`）只针对「远距离格挡/挥鱼竿」这类拖延动作，不针对移动。

### 6.2 结构

```json
{
  "format": "gladiator.reward-preset/v1",
  "name": "均衡（默认）",
  "description": "默认权重：击杀/伤害为主，威胁间距与压迫位小幅引导，站桩扣分。",
  "base": {"kill": 20.0, "death": -20.0, "damageDealt": 2.5, "still": -0.1, "inactive": 0.0},
  "terms": [
    {"id": "threat", "weight": 0.3, "unit": "second",
     "when": {"distanceMin": 2.8, "distanceMax": 4.2}},
    {"id": "punishBlock", "weight": -0.2, "unit": "step",
     "when": {"actionIn": ["block"], "distanceMin": 4.2}}
  ]
}
```

- `base`：15 个扁平权重，键固定为 `kill` / `death` / `damageDealt` / `damageTaken` /
  `combo` / `comboCap` / `threatWindow` / `pressure` / `whiff` / `still` / `waste` /
  `closing` / `attackAttempt` / `counter` / `inactive`；未提供的键补默认值，**未知键直接拒绝**。
- `terms`：条件式附加项，每项 `{id, weight, unit, when}`；最多 50 条，`id` 不可重复。
- `unit`：`second`（按秒结算，`weight × dt`）或 `step`（每个决策步结算一次 `weight`）。

### 6.3 多条件判定方案

不做表达式引擎，用「一条 term = 一个条件对象 + 权重 + 单位」：

- **AND**：同一个 `when` 里写多个条件；支持
  `distanceMin` / `distanceMax` / `actionIn` / `stillSecondsMin` / `healthBelow` /
  `opponentHealthBelow` / `afterHitSeconds` / `comboMin` / `attacked`。
- **OR**：写成两条 term（各自独立结算，效果叠加）。
- **叠加**：所有 term 的结算结果相加，再与 `base` 的各项相加构成该步奖励。

例：只在 2.8–4.2 格、且 1.5 秒内刚命中过时才给间距奖励 →

```json
{"id": "threatAfterHit", "weight": 0.3, "unit": "second",
 "when": {"distanceMin": 2.8, "distanceMax": 4.2, "afterHitSeconds": 1.5}}
```

### 6.4 存储

| 路径 | 用途 |
|---|---|
| `dashboard/var/reward-presets/<id>.json` | 预设本体（不进 git，`.gitignore` 已含 `var/`） |
| `dashboard/var/reward-presets/trash/<id>-deleted-<时间戳>.json` | 删除的软删除副本 |
| `dashboard/var/reward-presets/trash/<id>-overwrite-<时间戳>.json` | 覆盖保存前的旧版本 |

首次启动（`RewardPresetStore.ensure()`）写入内置 `default`，内容等于训练侧
`default_preset()`。id 规则与脚本同形但长度不同：`^[a-z0-9][a-z0-9_-]{0,31}$`
（1–32 位小写字母/数字/下划线/连字符，须以字母或数字开头；大写不再静默改写，直接 400）。
目录里文件名不合规的旧文件在列表中跳过，直接 `GET` 该 id 会得到中文 400 而不是崩溃。

### 6.5 接口

| 方法 | 路径 | 请求 | 响应 |
|---|---|---|---|
| GET | `/api/reward-presets` | — | `{ok, presets:[{id,name,description,termsCount,baseSummary,builtin,updatedAt}], schema}` |
| GET | `/api/reward-presets/<id>` | — | `{ok, id, path, preset}`（404 表示不存在） |
| POST | `/api/reward-presets` | `{id, preset}`（也接受预设字段与 `id` 平铺） | `{ok, id, preset}`；覆盖前自动备份 |
| DELETE | `/api/reward-presets/<id>` | — | `{ok, deleted}`；软删除到 `trash/` |

`schema` 字段：`{format, baseKeys:[{key,label,default}], units, conditionKeys:[{key,label}], limits}`，
前端按它渲染编辑表单，不在 JS 里硬编码键名。

校验失败一律 400，`error` 为中文可读原因（例如 `未知的基础权重：walking`、
`terms[0].unit 只能是 second / step`、`terms 里 id 重复：threat`、`base.kill 必须是有限数字`）。

### 6.6 UI

- 训练表单「Reward 预设」下拉 +「编辑预设…」按钮（`<dialog>`，不引入第三方库）。
- 基础权重全部可编辑；terms 可增删，每条可编辑 `id` / `weight` / `unit`，条件可增删。
- 「复制默认预设」把 `default` 的内容装入表单并给一个新 id，改完保存即新建。
- 内置 `default` 允许覆盖保存但禁用删除按钮（避免唯一默认项被清空）。
- 启动响应与数据页 `run.rewardPreset` 回显本次 run 使用的预设。

### 6.7 训练接线

- `kind=train`：命令行追加 `--reward-preset <绝对路径>`，路径来自
  `RewardPresetStore.path_for(id)`；`params.rewardPreset.applied = true`。
- `kind=multi`：同样追加 `--reward-preset <绝对路径>`（`multi_train.py` 已注册该参数），
  `params.rewardPreset.applied = true`，并写入 `run.json`。
- `run.json` 额外记录 `plan` / `planText` / `epochs` / `evalEpochs` / `evalRounds` /
  `evalEpisodes` / `rewardPreset`，`/api/artifacts/run?id=<id>` 与 `/api/artifacts` 的
  `rewardPreset` 字段即读自这里（训练脚本本身只写 `report.json`）。
