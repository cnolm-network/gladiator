# 模型注册表（G-622 阶段 2 + G-623 .gmod）

模型注册表把每次训练产出的 `policy.npz` 变成可复用、可命名的「初始权重」，
让下一次训练能从已有模型继续，而不是每次都从 `default` 随机初始化。

- 后端实现：`dashboard/models.py`（`ModelStore`）+ `dashboard/gmod.py`（`GmodScanner`）
- 存储目录：`dashboard/var/models/<id>/`（策略产物）；仓库根 `models/*.gmod`（文件模型）
- 训练侧消费：`training/gladiator_train/{train,multi_train}.py` 的 `--init-policy`
- 前端：训练表单「训练模型」下拉 + 导航「模型」页

> 两条来源：① **策略产物**——从 run 导入，持久化到 `dashboard/var/models/`；
> ② **`.gmod` 文件模型**——往 `models/` 里放一个包，刷新就自动出现（见 §8）。
> `.gmod` 装载已在 G-623 实现；**模型自带颜色仍未下发到游戏**：
> `meta.json` 的 `color` 目前只是元数据，Bot 名字前缀颜色仍由 `scripts.json` 决定。

## 1. 目录与 schema

```
dashboard/var/
├── models/
│   ├── default/
│   │   └── model.json                 # 内置项，无 policy.npz
│   └── 20261108-2200-train-bot_a/
│       ├── model.json
│       └── policy.npz                 # 从 run 复制的副本
└── models-trash/                      # 覆盖 / 删除前的备份（按目录整体复制或移动）
    └── <id>-<YYYYmmdd-HHMMSS>[-2]/...
```

`model.json` 的 schema 标识为 `gladiator.model/v1`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `format` | str | 固定 `gladiator.model/v1` |
| `id` | str | `^[a-z0-9][a-z0-9_-]{0,31}$`（1–32 位，小写开头） |
| `name` | str | 显示名，1–60 字符 |
| `description` | str | ≤300 字符 |
| `createdAt` / `updatedAt` | str | 本地时间 `YYYY-MM-DDTHH:MM:SS` |
| `source` | object | `{runId, runName, pair}`；直接给 `policyPath` 导入时 `runId=null` |
| `obsDim` / `numActions` / `hidden` | int | 从 npz 形状实测；`obsDim`/`numActions` 必须等于 `training/gladiator_env/spaces.py` 的 `OBS_DIM`(19) / `NUM_ACTIONS`(11) |
| `episodes` / `steps` / `meanReward` | int / int / float\|null | 来源 run 的 `report.json` 统计（无 summary 时从 history 现算） |
| `sha256` | str\|null | `policy.npz` 的字节哈希 |
| `note` | str | ≤300 字符，用户备注 |
| `builtin` | bool | 只有 `default` 为 true |

API 返回时额外附上 `hasPolicy`、`sizeBytes`、`policyPath`（仅详情）与
`recentRuns`（仅详情），这些是派生字段，不落盘。

## 2. 内置 `default` 语义

- `default` **表示「全新初始化」**：不加载任何权重，训练侧按 `--seed` 随机初始化。
- 它没有 `policy.npz`，`policy_path("default") is None`；选它时 Dashboard **不传**
  `--init-policy`。
- **不可删除**（`DELETE /api/models?id=default` → 400），**不可改 id**；
  可以改 `name` / `description` / `note`（`POST /api/models/update`）。
- 列表里恒排第一；即使 `models/` 被清空，`list()` 也会补回一个内存里的 default。

## 3. API

| 方法与路径 | 说明 |
|---|---|
| `GET /api/models` | `{ok, models:[...]}`，含内置 default 与派生 `hasPolicy`/`sizeBytes` |
| `GET /api/models?id=<id>` | `{ok, model:{...}, models:[...]}`；`model` 额外含 `policyPath` 与 `recentRuns`（最近 5 个 `run.json` 里 `modelId` 匹配的 run） |
| `POST /api/models` | 导入：`{name, description?, sourceRunId?, pair?, policyPath?, note?, id?}` → `{ok, model, models}` |
| `POST /api/models/update` | 改名/说明/备注：`{id, name?, description?, note?}` → `{ok, model, models}` |
| `DELETE /api/models?id=<id>` | 软删除（整个目录移到 `models-trash/`）→ `{ok, deleted, models}`；内置 400、不存在 404 |
| `POST /api/open-folder` | `{kind:"models"}` 打开 `dashboard/var/models`（Windows 专用；路径经 realpath 校验；只读模式 403） |

`GET /api/training/options` 的 `models[]` 与 `modelSupport=true` 也来自同一注册表，
训练表单下拉直接消费它。

### 导入来源（二选一，不能同时给）

1. **`sourceRunId`**：在 `<artifacts>/runs/<runId>/` 里按序找
   `policy-pair<N>.npz`（指定 `pair` 时）→ `policy.npz` → 其余 `policy-pair*.npz`。
   同时从 `report.json` 的 `summary`（缺失则按 `history` 现算）取 episodes/steps/meanReward。
2. **`policyPath`**：绝对路径，`realpath` 后必须落在 `training/artifacts` 内，
   后缀必须是 `.npz`。越界返回 400（防读取任意本机文件）。

未显式给 `id` 时，用来源 runId 或策略文件所在目录名做 slug（小写、非字母数字转 `-`、
截到 32 位）；重名自动追加 `-2`、`-3`…。大写 id 直接 400，不静默改写。

## 4. npz 校验规则

`inspect_policy()` **不依赖 numpy**：`.npz` 是 zip，逐个解析 `.npy` 的 header
（`magic` + 版本 + `shape` / `descr`），因此 Dashboard 不需要装 numpy。

必须满足：

1. 归档里存在 `W1.npy` / `b1.npy` / `W2.npy` / `b2.npy`，缺任意一个 → 400 中文说明。
2. 存在 `_meta.npy`：训练侧 `Policy.load()` 靠它还原 `obs_dim/n_actions/hidden`，
   缺了会在训练启动时才崩溃，所以导入时先挡掉。
3. 形状自洽：`W1=(hidden, obsDim)`、`W2=(numActions, hidden)`、
   `b1=(hidden,)`、`b2=(numActions,)`，且 W1/W2 的 hidden 一致。
4. `obsDim == OBS_DIM(19)`、`numActions == NUM_ACTIONS(11)`，与
   `training/gladiator_env/spaces.py` 不一致 → 400 中文说明。

通过后计算 `sha256` 与文件字节数，把 npz **复制**到 `models/<id>/policy.npz`
（原 run 的产物保持不动）。任何覆盖或删除前，旧目录都会先备份到 `models-trash/`。

训练侧还有第二道校验（`training/gladiator_train/policy.py` 的
`validate_init_policy()`）：文件缺失、npz 损坏、缺 `_meta` 或
`obs_dim / n_actions / hidden` 与 `--hidden` 不符时，在**创建任何环境之前**打印中文原因
并 `exit 2`，不产生目录、线程或 Bot 连接。

## 5. 与 run 的关系

```
run A（训练产物 report.json + policy.npz）
      │ POST /api/models {sourceRunId: run A}
      ▼
model m（policy.npz 副本 + source.runId=run A + 统计快照）
      │ 训练表单选 m → --init-policy <abs path> --model-id m --model-name …
      ▼
run B：run.json / report.config 记录 {modelId, modelName, initPolicy}
      │ GET /api/artifacts/run?id=run B → run.model
      │ GET /api/training/episodes?run=run B&episode=1 → episode.model
      ▼
模型详情 recentRuns：扫所有 run.json 的 modelId，列出最近 5 个使用过 m 的 run
```

- 一条计划项（plan）**不绑定模型**；模型是「这一次训练整体用哪份初始权重」。
- `--init-policy` 只把 `W1/b1/W2/b2` 装进本次 `Policy`，**不接管 optimizer 状态、
  `updates` 计数**；`lr` / `beta` / `seed` 一律用本次命令行的值。
- `report.json` 的 `config.initPolicy` / `modelId` / `modelName` 由训练脚本写入；
  `run.json` 的同名三字段由 Dashboard 的 `_record_run_meta()` 补齐（训练脚本本身不写 run.json）。
- 单局详情的 `model` 字段来自 `run.json`，前端对局弹窗与 run 详情都显示「名称（id）」。
- 删除模型**不会**删除任何 run；run 里记录的 `modelId` 只是历史引用。

## 6. 前端入口

- 导航「模型」页：列表（名称 / 来源 run / obs×actions / 回合数 / 均 reward / 大小 / 更新时间）、
  详情 + 最近使用、从 run 导入、编辑、删除、打开文件夹。
- 训练表单「训练模型」下拉：`default` + 注册表所有模型 + 「＋ 从 run 导入…」（跳到模型页并打开导入卡片）。
  选中非 default 时，进度卡会显示所用模型。
- 数据页 run 详情：「训练模型」一行；单局详情弹窗：「模型」一行。
- 「打开文件夹」按钮：`scripts` / `presets` / `reward-presets` / `models` / `gmod` / `run` /
  `run-reviews` / `artifacts` / `logs` 九种 kind（`gmod` 指向仓库根 `models/`），见 `docs/operations.md`。

## 7. 已知边界

- `pair` 只在按 `sourceRunId` 导入时生效；多场地训练的每个场地都用同一份 `--init-policy`
  初始化（各自 seed 不同）。
- `policy.npz` 是**副本**：导入后修改/删除原 run 的产物不影响已注册模型。
- 备份目录 `models-trash/` 不自动清理，长期导入/删除会累积，需人工归档。
- Dashboard 的校验只看形状与 `_meta` 是否存在，不解析 `_meta` 内容；
  真正的可加载性由训练侧 `Policy.load()` 兜底（失败即 exit 2）。

## 8. `.gmod` 格式与热扫描（G-623）

**放文件即出现**：把 `xxx.gmod` 放进仓库根 `models/`，刷新模型页就出现在列表里；
删掉文件，下次刷新自动消失。Dashboard **不会删除/改写用户放进去的包**。

### 8.1 目录与配置

- 默认目录：仓库根 `models/`，不存在时自动创建并放一个 `.gitkeep`（空目录也能进版本库）。
- 覆盖方式（优先级从高到低）：环境变量 `DASHBOARD_GMOD_DIR` →
  `dashboard/config.json` 的 `gmodDir` → 默认 `models/`。相对路径按仓库根解析。
  `GET /api/config` 的 `config.gmodDir` 可核对当前值。
- 训练时解出的 npz 缓存在 `dashboard/var/gmod-cache/`（可随时清空，会按内容重建）。

### 8.2 包结构

一个 `.gmod` 就是 zip，**只允许** `meta.json` 与 `policy.npz` 两个成员：

```
rush-v3.gmod
├─ meta.json
└─ policy.npz        # 与 training/gladiator_train/policy.py 保存的权重同构
```

`meta.json` 示例：

```json
{
  "format": "gladiator.model/v1",
  "id": "rush-v3",
  "name": "Rush v3",
  "obsDim": 19,
  "actDim": 11,
  "color": "§6",
  "version": "0.3.0",
  "createdAt": "2026-11-08T19:00:00",
  "description": "以 rusher 为对手训练 120 局"
}
```

| 字段 | 必填 | 规则 |
|---|---|---|
| `id` | ✅ | `^[a-z0-9][a-z0-9_-]{0,31}$`（1–32 位）；**文件 id 以文件名为准**，列表/接口里用 `gmod:<fileId>` 引用（见 §12.4） |
| `name` | ✅ | 1–40 字符 |
| `obsDim` | ✅ | 必须等于 `training/gladiator_env/spaces.py` 的 `OBS_DIM`（19） |
| `actDim` | ✅ | 必须等于 `NUM_ACTIONS`（11） |
| `color` | ⬜ | `&0–&f` / `§0–§f` 单色码，归一为 `§x`；**目前仅展示，不下发到游戏** |
| `version` / `createdAt` / `description` | ⬜ | 字符串，展示用 |
| `format` | ⬜ | 建议 `gladiator.model/v1`；不一致只给 `warning` |

`policy.npz` 必须含 `W1.npy / b1.npy / W2.npy / b2.npy / _meta.npy`，且
`W1=(hidden, obsDim)`、`W2=(actDim, hidden)`、`b1=(hidden,)`、`b2=(actDim,)`；
形状要与 `meta.json` 一致。`_meta.npy` 是训练侧 `Policy.load()` 必需的，缺了会在
**导入/扫描阶段**就被挡掉，而不是等训练启动才崩。

### 8.3 校验失败：不打断列表

坏包不会抛异常，而是照样出现在 `/api/models` 里，只标 `valid=false` + 中文 `error`：

```json
{
  "id": "broken", "kind": "gmod", "valid": false,
  "error": "不是合法的 .gmod（zip）包: File is not a zip file",
  "source": {"type": "gmod", "file": "broken.gmod", "dir": "models"},
  "sizeBytes": 9, "sha256": null
}
```

其它常见错误：`.gmod 缺少 meta.json`、`.gmod 缺少 policy.npz`、
`meta.json 维度不符：obsDim=16（应为 19）…`、`policy.npz 形状与 meta.json 不一致：…`、
`policy.npz 缺少 _meta.npy…`、`meta.json 的 color 只允许 &0–&f / §0–§f 单色码`。
`POST /api/models/reload` 会把所有坏包汇总进 `invalid: [{file, error}]`。

前端：模型表「状态」列显示红色「不可用」+ 错误原因、整行灰显；训练模型下拉里该
`<option disabled>` 并显示原因。后端也会在 `_build_train()` 阶段拒绝 `valid=false` 的模型。

### 8.4 热扫描与缓存

- `GET /api/models` **每次**都扫描 `models/*.gmod`；解析结果按 `(mtime_ns, size)` 缓存，
  文件内容没变就不重新解析（不依赖 numpy，只解析 npy header）。
- 文件被删除 → 缓存项与列表项一起消失；文件被替换（mtime/大小变化）→ 重新校验。
- `POST /api/models/reload` → `{ok, scanned, added, invalid, models, gmodDir}`：
  强制清空解析缓存重扫，`added` 是相对**上次 reload** 新增的条目数。前端「刷新」按钮调它。
- **文件 id vs 列表 id**：文件 id（`fileId`）恒为文件名的小写 slug；`GET /api/models`
  里的条目 id 是 `gmod:<fileId>`。两者都要区分，否则无法与同名策略模型并存。
- **同名冲突**：与已注册策略模型同名时**两行都保留**——策略模型行 id 为 `<id>`、
  `.gmod` 行 id 为 `gmod:<id>`，后者额外带 `conflict:true` 与
  `warning:"与可持久化模型同名，已并列为只读副本"`。裸 id 始终指向可持久化模型，
  绝不因为存在同名 `.gmod` 而变成只读（G-626 回归修复）。详见 §12.4。
- `meta.json` 的 `id` 与文件名不一致时给
  `warning: "meta.json 的 id=… 与文件名不一致，已按文件名 … 作为 id"`。

### 8.5 删除与训练

- `DELETE /api/models?id=gmod:<fileId>` → **409**「`.gmod 由文件系统管理，请删除 models/xxx.gmod`」；不带前缀的裸 id 若没有同名策略模型也会按兼容路径回同样 409。
  `POST /api/models/update` 对 gmod 同样 409。要删/改名请直接操作文件，然后刷新。
  策略产物仍走原来的软删除（`models-trash/`）。
- 训练表单选中 gmod 时，Dashboard 把包内 `policy.npz` 解到
  `dashboard/var/gmod-cache/<policy 内容 sha256>.npz`（按内容复用），
  把这个绝对路径作为 `--init-policy` 传给训练脚本；`report.config` 与 `run.json`
  记录 `initPolicy` / `modelId` / `modelName`。原始 `.gmod` 保持只读。
- `GET /api/training/options` 的 `models[]` 含 gmod 条目与 `kind` / `valid`，
  前端按「策略产物 / 文件模型（.gmod）」分组渲染 `<optgroup>`。

### 8.6 颜色现状（如实说明）

`meta.json` 的 `color` 会被读出来并显示在模型详情里，但 **没有** 进游戏：
Bot 名字前缀颜色仍由 `dashboard/var/scripts.json` 的 role 颜色经
`/caller tagcolor` 同步（见 `docs/scripting.md` §7 阶段 1）。
「模型自带颜色」仍属阶段 3 未实现部分，需要新的协议/调用方接口与 ADR。

## 9. 持久化训练（G-624）

**问题**：G-622/G-623 的 `--init-policy` 只是把权重装进一次训练，训练产物只落在
`runs/<id>/policy.npz`；注册表里的 `var/models/<id>/policy.npz` 从不更新，
`default` 永远随机初始化 —— 等于「读一次权重、练完丢掉」。

**现在**：选中的策略模型就是本次训练要改动的对象。表单「持久化训练」开关默认开，
训练**正常完成**后由 Dashboard 把最终权重写回该模型，并累加训练统计。

### 9.1 写回时机与原子性

| 训练终态 | 是否写回 |
|---|---|
| `finished`（exit 0） | ✅ 写回 |
| `failed`（非零退出 / 启动失败） | ❌ 不写回 |
| `stopped`（手动停止 / 启动前被停止） | ❌ 不写回 |
| 写回过程报错（磁盘、权限、npz 损坏） | ❌ 不写回，但 **job 仍为 finished** |

流程（`dashboard/models.py` 的 `ModelStore.write_back()`）：

1. 读 `runs/<id>/policy.npz`，用 `inspect_policy()` 校验形状与维度（含 `_meta.npy`）。
2. 若目标 `var/models/<id>/policy.npz` 已存在，先复制到
   `dashboard/var/models-trash/<id>-<YYYYmmdd-HHMMSS>[-2].npz`（可人工恢复）。
3. 写 `policy.npz.tmp` → `os.replace(tmp, policy.npz)` **原子替换**；临时文件在
   `finally` 里清理，任何时刻目标文件都是完整的一份。
4. 更新 `training.json` 与 `model.json`（见下）。
5. 任何异常都只返回 `{ok:false, error:"模型写回失败: …"}`，不抛出、不改 job 状态；
   结果写进 `job.persist` 与 `run.json`，前端 toast 中文原因。

训练侧只增加 `--persist-to <model_id>`（写进 `report.config.persistTo` 作为审计），
**写回逻辑完全在 Dashboard**（它才知道备份、锁、元数据）。`train.py` 的
`--save` 默认就是 `<run-dir>/policy.npz`，保证产物一定在 run 目录。

### 9.2 元数据与累加规则

`var/models/<id>/training.json`（schema `gladiator.model-training/v1`）：

```json
{
  "format": "gladiator.model-training/v1",
  "modelId": "default",
  "lastRunId": "20261109-0010-train-Bot_A",
  "lastTrainedAt": "2026-11-09T00:10:00",
  "totalEpisodes": 12,
  "totalSteps": 2400,
  "lastMeanReward": 3.5,
  "writeCount": 2,
  "runs": [
    {"runId": "…", "at": "…", "episodes": 4, "steps": 800, "meanReward": 2.1},
    {"runId": "…", "at": "…", "episodes": 8, "steps": 1600, "meanReward": 3.5}
  ]
}
```

- `totalEpisodes` / `totalSteps` 是**累加**（`旧值 + 本次`），不是覆盖；`runs` 只保留最近 20 条。
- 本次 episode/step/reward 来自 `report.json` 校准后的内存 job；报告缺失时退回内存进度。
- 同步把 `model.json` 的 `episodes` / `steps` / `meanReward` / `sha256` / `obsDim` /
  `numActions` / `hidden` / `updatedAt` 更新，模型列表与详情页据此显示
  「最近训练：时间 · 累计 N 局 / M 步 · 最近均值 reward」，详情页列出最近 20 次写回。

### 9.3 `default` 变成真正的可持久化模型

- 首次持久化训练写回时会创建 `var/models/default/policy.npz`（原先没有文件）。
- 之后选 `default` 开训，`_model_from_payload()` 会发现该文件并自动传
  `--init-policy <abs path>`，**接着上次继续练**。
- `default` 仍不可删除、不可改 id；名称/说明可改。文案已从「全新初始化」改为「持久化训练」。
- **如何从零重开**（任选其一）：
  1. 关闭「持久化训练」开关跑一次（只读起点，不写回）；
  2. 把 `default` 当前的 `policy.npz` 导入成一个新模型，之后用新模型练；
  3. 手动删除 `var/models/default/policy.npz` + `training.json`（建议先备份到 `models-trash/`）。

### 9.4 `.gmod` 只读与 multi 策略

- `.gmod` 是用户放进去的文件包，**Dashboard 不改写它**。`write_back()` 对 gmod id 直接返回
  `{ok:false, error:".gmod 是只读文件包，请先「导入为策略模型」…"}`；
  `_persist_options()` 也会强制 `persist=false` 并给 `persistWarning`。
- 一键路径：表单「导入并继续训练」→ `POST /api/models {name, gmodId}` →
  把包内 npz 复制成 `var/models/<新id>/policy.npz` 的**策略模型**，
  前端自动选中新 id、开关恢复可用，之后就能正常写回。

**multi 选择「禁用持久化」**（而不是「合并最后一个完成的场地权重」）：

- 每个 pair 产出独立 `policy-pairN.npz`，写回哪一份没有唯一语义，选「最后一个完成的」
  会让结果依赖线程调度、不可复现；
- N 个场地同时练、同时写回同一模型，会破坏「同一模型只允许一个持久化训练持锁」的独占性；
- 数据口径也难对齐（一次 job 的总步数是所有 pair 之和，但写回的只是一份权重）。

因此 `kind=multi` 时后端强制 `persist=false`、`persistSupported=false` 并给出中文
`persistWarning`，前端禁用开关并提示改用单场地训练（或先导入某个场地的 npz 再继续）。
训练侧 `multi_train.py` 仍接受 `--persist-to`（只写进 report.config），Dashboard 不会传。

### 9.5 写锁与并发

`dashboard/model_locks.py`：锁文件 `var/models/<id>/.lock`，内容
`{"pid": <Dashboard 进程 pid>, "runId": "...", "at": "...", "modelId": "..."}`。

- 启动持久化训练时 `acquire()`：同 runId + 同 pid 可重入；若锁被**活进程**持有 →
  `Conflict(409)`「模型 X 正在被 run Y 的持久化训练占用…」，请等它结束 / 换模型 / 关开关。
- 锁存在但 pid 已不存在 → **回收**（删锁）并把「发现并回收了过期锁」写进 `job.notes`。
  进程存活检查用 Windows `OpenProcess` / POSIX `/proc`，**不用** `os.kill(pid,0)`
  （Windows 上会真的 TerminateProcess）。
- 训练进入任何终态（含启动前失败）都 `release()`，且只释放 runId 匹配的锁，避免误删别人的。
- 失败/停止时同样释放锁，并把 `job.persist` 标为 `{ok:false, skipped:true, error:…}`。

### 9.6 用户可见表现

| 场景 | 前端 |
|---|---|
| 写回成功 | 进度卡「持久化训练：已写回模型 X（权重已更新，下次选它继续练）」；run 详情「持久化写回：已写回模型 X」；模型页累计数字刷新 |
| 写回失败（磁盘/权限） | 进度卡红字「写回失败：…」+ toast「模型写回失败：…（训练本身已完成）」；**job 仍是 finished** |
| 失败/停止/启动前失败 | 「未写回模型 X：训练状态为 failed…」；不弹错误 toast |
| 锁冲突 | 启动接口 409，中文说明占用 run 与 pid |
| gmod / multi | 开关禁用 + 灰色说明；gmod 额外显示「导入并继续训练」 |

`job.persist` 结构：`{enabled, modelId, ok: true|false|null, skipped?, error?, path?, training?}`；
`run.json` 同步写同一对象，数据页可回显。

## 10. 多场地模型融合（G-625）

多场地训练（`multi_train`）每个场地各自产出一份 `policy-pairN.npz`，直接持久化「写回哪一份」
没有唯一语义。G-625 起先在训练末尾把各场地权重**融合**成一份 `run/policy.npz`，再交给
Dashboard 写回所选模型。

### 10.1 三种策略

`multi_train.py --merge {mean,weighted,none}`，Dashboard 默认 `mean`：

| 方法 | 语义 | 何时用 |
|---|---|---|
| `mean` | 对 `W1/b1/W2/b2` 逐元素**等权**平均 | 各场地强度接近时的默认选择 |
| `weighted` | 按各场地**完成步数**加权（`steps<=0` 退化为等权，避免除零） | 场地难度/时长差异大，想让练得多的场地占比更高 |
| `none` | 不融合，不写 `run/policy.npz` | 只想分别看每个场地的产物，不需要统一模型 |
| `swa` | **未实现**（预留：随机权重平均） | 目前不要使用 |

融合结果固定写到 `<run>/policy.npz`；原始 `policy-pairN.npz` 保留不动，便于对比。

### 10.2 `report.merge` 字段

```json
{
  "method": "mean",
  "pairs": [
    {"pair": 1, "steps": 1200, "sha256": "…"},
    {"pair": 2, "steps": 980, "sha256": "…"}
  ],
  "mergedPolicyPath": ".../runs/<runId>/policy.npz",
  "mergedSha256": "…",
  "available": true,
  "skipped": false
}
```

- `available=true` 才表示融合成功；失败时另有 `error`（中文）且 `skipped=true`，
  不写 `policy.npz`。
- 某个 pair 训练失败 **不会** 拖垮整个 run：它被跳过（不进入 `pairs`），其余场地继续融合；
  可用场地 < 1 时 `available=false`、`error="没有可用的场地权重…"`。
- 各场地权重形状/维度不一致（例如 `hidden` 或 `obs_dim` 不同）会直接抛错终止融合，
  因为无法逐元素相加。

### 10.3 与持久化训练的配合

- `merge=mean|weighted`：Dashboard 的「持久化训练」开关**可用**（默认开），
  写回的是 `run/policy.npz`（即融合结果）；`job.persist` 额外带
  `merged:true`、`mergePairCount`（参与融合的场地数）、`mergeMethod`。
- `merge=none`：开关**禁用**，后端强制 `persist=false` 并给 `persistWarning`
  「多场地训练需要开启融合才能写回模型」；不会阻塞训练。
- 运行期融合失败：`job.persist = {ok:false, skipped:true, degraded:true, error:…}`，
  `persistWarning` 同步写进 `run.json`；job 仍是 `finished`，只是不写回模型。
- 累计口径：写回时 `totalEpisodes / totalSteps` 取**所有场地之和**（来自 report 校准后的
  `history`），不是单个 pair 的数字。

### 10.4 监控建议

1. 先看 `report.merge.available`：`false` 说明没产出统一权重，此时不要指望模型被更新。
2. 对比各场地 `steps` 与 `sha256`：某个场地步数异常低，`mean` 会被它拉平，
   可改用 `weighted` 或排查该场地失败原因（`report.pairs[].error`）。
3. 融合后若效果反而下降，通常是场地目标冲突；先用 `merge=none` 分别评估，
   再决定是否只保留强相关场地。

## 11. 模型可视化（G-625）

独立页面，纯原生 JS + SVG，零 CDN、零新依赖；不改训练侧、不新增埋点，全部读现有产物。

### 11.1 直达 URL

```
http://127.0.0.1:8080/?model=<modelId>#model-viz
```

- `?model=<id>` 指定模型；`#model-viz` 触发打开可视化抽屉（`?model=<id>&viz=1` 等价）。
- 模型页每行「可视化」按钮也走同一入口；页面右上「复制链接」生成上述直达 URL，
  可直接用于截图/分享。
- 未知 id 返回 404，前端在抽屉里显示中文错误而不是空白。

### 11.2 页面内容（9 张卡片）

| 卡片 | 内容 | 数据来源 |
|---|---|---|
| 基本信息 | id / name / kind / obs×actions×hidden / 参数量 / sizeBytes / sha256 前 16 / 创建与更新时间 / 来源 / 颜色 | `model.json`、`policy.npz` |
| 权重直方图与范数 | `W1/b1/W2/b2` 合并分布的 24 桶直方图 + 各层 `‖W‖₂` | `policy.npz` |
| 与最近备份的权重差异 | L2 相对变化 `‖w−w₀‖/‖w₀‖`、余弦相似度、备份路径 | `models-trash/*.npz` |
| 累计 reward 曲线（按 run） | 每个 run 一条折线（x=局号，y=reward） | `report.history` / `training.runs` |
| 累计 reward 曲线（累计和） | 跨 run 的累计 reward 折线 | 同上聚合 |
| 效果指标 | 胜率、命中率、空挥率、伤害/承伤比、平均对局时长 | `report.history[].review` |
| 对手分组对比 | 按 `opponentMode` 分组的场次/reward/胜率/承伤表 | 同上 |
| 动作分布 | 11 个动作的横向条形与占比 | `reviews/*.json` 的 `actionCounts` |
| 累计统计表 | 每个 run 的局数/步数/均 reward/胜率/命中率/伤害比/时长 + 合计行 | `training.json` + report |

### 11.3 API 字段结构

`GET /api/models/viz?id=<modelId>`（一次请求返回全部数据）：

```json
{
  "ok": true,
  "id": "default",
  "model": {"id","name","description","builtin","kind","obsDim","numActions","hidden",
             "episodes","steps","meanReward","sha256","updatedAt","source","training"},
  "weights": {"available": true, "obsDim": 19, "numActions": 11, "hidden": 64,
              "sizeBytes": 0, "sha256": "…", "norms": {"W1": 11.3, "b1": 0.08, "W2": 3.26, "b2": 0.03},
              "totalNorm": 11.4, "count": 1995,
              "histogram": {"available": true, "binEdges": [...25], "counts": [...24]}},
  "backupDiff": {"available": true, "backupPath": "…", "backupSha256": "…",
                 "l2RelativeChange": 0.04, "cosineSimilarity": 1.0},
  "runs": {
    "available": true, "runCount": 1, "episodeCount": 6,
    "byRun": {"labels": ["20261109-0107-train-Bot_D1"],
              "series": [{"runId": "...", "points": [{"i": 1, "reward": -26.0}]}]},
    "cumulative": {"labels": ["1", "2"], "points": [{"i": 1, "reward": -26.0, "cumulative": -26.0}]},
    "metrics": {"available": true, "episodeCount": 6, "winRate": 0.0, "hitRate": 0.33,
                "whiffRate": 0.67, "damageRatio": 0.21, "avgDuration": 16.06},
    "opponents": [{"opponentMode": "rusher", "nEpisodes": 3, "winRate": 0.0, "hitRate": 0.33,
                   "damageRatio": 0.2, "avgReward": -29.1, "avgDuration": 16.2}],
    "actions": [{"action": "idle", "count": 73, "fraction": 0.154}],
    "table": {"columns": [...], "rows": [...], "totals": {"episodes": 6, "steps": 474, "meanReward": -27.99}}
  }
}
```

### 11.4 缺数据表现

- 没有 `policy.npz`（例如刚创建的 `default`）→ `weights.available=false` + `note`，
  页面显示「暂无数据：该模型没有 policy.npz…」，**不是** 500。
- 没有备份 → `backupDiff.available=false` + `note`。
- 没有训练记录 → `runs.available=false` + `note`；`metrics` 里数值为 `null`。
- 某个对手模式没有样本 → 不进入 `opponents`；`actions` 恒返回 11 项（无样本为 0）。
- 未知模型 id → HTTP 404（中文「模型不存在」）。

> 可视化页是只读的：不触发训练、不改 `var/`、不写 `models-trash/`。

## 12. 导出与可移植性（G-626）

`POST /api/models/export` 把注册表里的策略模型（`dashboard/var/models/<id>/policy.npz`
+ `model.json` 的统计）打包成 `models/<id>.gmod`，之后就能像普通 `.gmod` 一样被扫描、选中训练。

```http
POST /api/models/export
{"id": "default", "overwrite": false}
```

也支持 query：`POST /api/models/export?id=default&overwrite=1`。

| 情况 | 响应 |
|---|---|
| 成功 | `200 {ok, path, relativePath, id, sha256, sizeBytes}`（`sha256` 是**包文件**哈希） |
| 模型不存在 | `404 {"error":"模型不存在: …"}` |
| 目标 `.gmod` 已存在且 `overwrite=false` | `409 {"error":"目标文件已存在：<id>.gmod；如需覆盖请传 overwrite=true"}` |
| 模型没有 `policy.npz`（从未训练写回 / 未导入） | `400 {"error":"模型 X 没有 policy.npz…"}` |
| 缺 id | `400` |

写出过程：先建 `<id>.gmod.tmp`，写完 `os.replace` 到目标；包内**只允许**
`meta.json` + `policy.npz` 两个成员。

### 12.1 meta.json 字段

```json
{
  "format": "gladiator.model/v1",
  "id": "rush-v3",
  "name": "Rush v3",
  "description": "以 rusher 为对手训练 120 局",
  "color": "§6",
  "obsDim": 19,
  "actDim": 11,
  "hidden": 64,
  "version": 1,
  "createdAt": "2026-11-08T19:00:00",
  "exportedAt": "2026-11-09T22:30:00",
  "sha256": "<policy.npz 内容哈希>",
  "training": {
    "format": "gladiator.model-training/v1",
    "totalEpisodes": 12,
    "totalSteps": 2400,
    "lastMeanReward": 3.25,
    "lastTrainedAt": "2026-11-09T01:07:30",
    "writeCount": 3,
    "runs": [{"runId": "...", "at": "...", "episodes": 12, "steps": 2400, "meanReward": 3.25}]
  },
  "rewardPreset": "rush-v1"
}
```

- `sha256` 是**权重内容**哈希（可用 `sha256sum` 对包内 `policy.npz` 复核）；
  响应里的 `sha256` 是**整个 .gmod 文件**的哈希。
- `training.runs` 只保留最近 20 条；`color` 没有时写 `null`；`rewardPreset`
  仅当模型记录里有时出现（字符串 id 或对象）。
- 扫描端读到 `training` 后，列表的「累计局数/步数/最近均值」与详情页
  `training` 段都来自包里这份快照；没有 `training` 的旧包保持原有的「无文件」语义。

### 12.2 包含 / 不包含

**包含**：最终权重 `W1/b1/W2/b2` + `_meta`（obs_dim / n_actions / hidden / updates）、
维度与 hidden、`training` 累计统计（局数、步数、最近均值、最近 20 次 run）、
可选 `color` 与 `rewardPreset` id。

**不包含**（有意为之）：

| 未包含 | 原因 / 影响 |
|---|---|
| Adam 动量 `m` / `v`、`updates` 之外的优化器状态 | 续训会**重新累积**动量，前几百步的学习率行为与源机器不同；这是可移植包的常见取舍 |
| 复盘明细（`reviews/*.json`、每局动作分布） | 体积大且与 run 绑定；留在 `training/artifacts/runs/<runId>/` |
| `run/` 目录、`run.json`、`report.json` | 属运行记录，不随模型走；跨机器只保留累计数字 |
| dashboard 的锁文件 / 缓存 | 本机运行时状态 |
| 训练会话的随机种子、超参 | 目标机器按自己的表单参数训练 |

### 12.3 跨机器使用步骤

1. 在源机器：模型页点「导出 .gmod」（或 `curl -X POST /api/models/export -d '{"id":"<id>"}'`），
   得到 `models/<id>.gmod`。
2. 把该文件复制到目标机器的仓库根 `models/` 目录（或目标机器 `dashboard/config.json`
   里 `gmodDir` 指向的目录）。
3. 刷新 Dashboard 模型页（或点「刷新」按钮）→ 出现文件模型行（id 为 `gmod:<id>`，与源模型的
   `<id>` 并存），状态「正常」，
   「来源」列显示 `文件 · models/<id>.gmod`，并带导出时的训练累计。
4. 训练表单「训练模型」下拉的「文件模型（.gmod）」分组里选中它即可开训；
   文件模型只读，想要继续持久化写回就先点「导入并继续训练」转成策略模型。
5. 校验可选：`sha256sum models/<id>.gmod` 与导出响应里的 `sha256` 对比；
   包内 `meta.json` 的 `sha256` 应与解包出的 `policy.npz` 哈希一致。

### 12.4 同 id 冲突规则（列表 / 训练 / 导出）

同一个 id 既可以属于可持久化策略模型，也可以属于 `models/<id>.gmod`。两者**并存**：

| 引用方式 | 指向 | 训练 | 持久化 | 列表字段 |
|---|---|---|---|---|
| `<id>`（裸 id，如 `default`） | 策略模型（`var/models/<id>/`） | 用它作为起点 | 允许写回 | `kind=policy`、`hasPolicy=true` |
| `gmod:<id>`（前缀 id，如 `gmod:default`） | 文件模型（`models/<id>.gmod`） | 解包到 `gmod-cache` 作为起点 | **自动降级为 false** + `persistWarning` | `kind=gmod`、`conflict=true`、`warning=与可持久化模型同名，已并列为只读副本` |

要点：

- 裸 id **永远优先策略模型**；只有注册表里没有该 id 时，裸 id 才回落到 `.gmod`
  （兼容旧链接/手填 id）。
- 训练下拉会同时列出两行（策略产物分组 + 文件模型分组）；选中 `gmod:<id>` 时开关禁用并提示
  「文件模型只读」，不会 500，也不会把训练标记为失败。
- 导出完成后**不会**把 `default` 变成只读：列表里仍能看到 `default`（可持久化）+ `gmod:default`
  （只读副本）。想继续在副本上训练，点「导入并继续训练」生成新的策略模型。
- 删除/改名：`gmod:<id>` 一律 409（文件系统管理）；裸 id 删的是策略模型。
- 可视化：`GET /api/models/viz?id=default` 看策略模型，`?id=gmod:default` 看只读副本
  （权重直方图可算，训练曲线因包内没有 run 明细而显示「暂无数据」）。
