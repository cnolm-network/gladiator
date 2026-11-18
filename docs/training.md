# 策略训练（training/gladiator_train）

环境只暴露观测/动作/奖励（见 [training-env-api.md](training-env-api.md)）；**算法**放在
`training/gladiator_train/`，与 `gladiator_env` 分开，符合 `training/AGENTS.md`
"不将具体 RL 算法写入环境代码"。

## 1. 算法

| 项 | 选择 | 理由 |
|---|---|---|
| 策略 | 单隐藏层 tanh MLP + softmax（16 → 64 → 7） | 观测只有 16 维、动作 7 个；小网络训练快、可解释 |
| 目标 | REINFORCE：`J = mean(A·logπ(a\|x)) + β·H(π)` | 无需 critic，先跑通闭环 |
| 基线 | 折扣回报逐 episode 标准化（`A = (R-μ)/σ`） | 降低方差，省掉价值网络 |
| 优化 | Adam（手写），默认 `lr=3e-3`，`β=0.01` 熵正则 | 无 torch 依赖，numpy 足够 |
| 反向传播 | 手写（见 `policy.gradients`） | 网络只有两层，手写比引框架更轻 |

梯度推导（softmax 交叉熵 + 熵）：

```
z = W2·tanh(W1·x + b1) + b2,  p = softmax(z)
∂J/∂z = (onehot(a) - p)·A - β·p·(log p + H)
```

`tests/test_policy.py::test_gradient_matches_finite_difference` 用数值梯度逐参数校验了这条公式。

## 2. 训练循环

三层结构：**回合（epoch）→ 计划（plan）→ 轮次（episode）**。

- **回合**：把计划完整执行一遍；`--epochs N` 表示跑 N 遍。
- **计划**：有序阶段列表 `mode:轮次,...`，例如 `--stages rusher:5,scripted:5`。
- **轮次**：一局对局，直到一方出局或 `max_steps` 耗尽；**每个训练轮次结束更新一次参数**。

因此训练总对局数 = `epochs × Σ(轮次)`，参数更新次数与之相同。

每个轮次：

1. `env.reset()`：RCON 摆位、回血、清背包（调用方职责）；
2. 逐步 `policy.act(obs)` → `env.step(action)`，记录 `(obs, action, reward)`；
3. episode 结束（一方出局或到 `max_steps`）后计算折扣回报、标准化优势、更新策略。

对手是**随机策略**（`single_agent=True` 时环境用 `np_random.integers` 采样），
所以学的是"如何打赢乱打一通的对手"，适合冒烟与基线，不适合作为最终强度。

## 3. 运行

```powershell
# 前置：Paper + Gladiator 已启动、RCON 已开、每个 Bot 一个 bot-runtime
cd training
.venv\Scripts\python.exe -m gladiator_train.train `
    --users Bot_T1 Bot_T2 --ports 8766 8767 `
    --episodes 12 --max-steps 120 --auto-spawn
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 1 | **训练回合数**：每回合完整按计划执行一遍 |
| `--stages` | 空 | **计划**：`mode:轮次,...`，按顺序执行；为空时退化为 `--opponent-mode × --episodes` |
| `--episodes` | 10 | 无 `--stages` 时的轮次；有 `--stages` 时是每阶段的默认轮次 |
| `--max-steps` | 150 | 每个**轮次**的步数上限（截断即 truncated） |
| `--action-ticks` | 2 | 每个动作持续的控制周期数（10Hz → 0.2s） |
| `--lr` / `--hidden` / `--beta` / `--gamma` | 3e-3 / 64 / 0.01 / 0.99 | 策略超参 |
| `--auto-spawn` | 关 | 由脚本调用 `/gladiator spawn` |
| `--init-model` | 空 | 从已有 `policy.npz` 初始化（续训）；留空则随机初始化 |
| `--eval-epochs` | 1 | **评估回合数**：训练后完整按计划走几遍（贪心、不学习） |
| `--eval-episodes` | — | 旧参数（已隐藏），等价于 `--eval-epochs` |
| `--save` / `--report` | `artifacts/policy.npz` / `artifacts/train-report.json` | 产物 |

## 4. 产物

- `artifacts/policy.npz`：`W1/b1/W2/b2` + `_meta`（obs_dim/n_actions/hidden/updates）。
  用 `Policy.load(path)` 读回，`policy.act(obs, rng, greedy=True)` 即可推理。
- `artifacts/train-report.json`：`config` / `history`（每回合 steps、totalReward、winner、
  opponentMode、win、durationSeconds、policyUpdates）/ `summary` / `evaluation` /
  `evaluationByOpponent` / `evaluationOverall` / `policyPath`。

`training/artifacts/` 已加入 `.gitignore`，训练产物不入库。

### 4.1 分对手胜率（P1-1）

`history` 的每条记录新增两个字段：

| 字段 | 含义 |
|---|---|
| `opponentMode` | 本局对手策略（取自 `env.opponent_mode`，即训练计划当前段） |
| `win` | 学习方是否赢下本局（`winner == bots[0].username`）；超时截断为 `false` |

`report.summary` 是训练局的分桶聚合：

```json
{
  "overall":    {"episodes": 16, "wins": 3, "winRate": 0.1875, "meanReward": -39.68},
  "byOpponent": {
    "scripted": {"episodes": 6, "wins": 2, "winRate": 0.3333, "meanReward": -20.1},
    "rusher":   {"episodes": 6, "wins": 1, "winRate": 0.1667, "meanReward": -41.2},
    "aimbot":   {"episodes": 4, "wins": 0, "winRate": 0.0,    "meanReward": -68.3}
  }
}
```

评估局单独给 `report.evaluationByOpponent` / `report.evaluationOverall`
（`evaluation` 本身仍是数组——Dashboard 与旧脚本按数组消费，分桶放在同级字段，不破坏既有读取方）。

Dashboard「运行详情」据此渲染「分对手胜率」小表：对手 / 局数 / 胜 / 胜率 / 均 reward。
旧 run 没有 `summary.byOpponent` 时，Dashboard 会用 `history` 里的 `opponentMode` + `winner`
现算；连 `opponentMode` 都没有的更老 run 返回空表并显示「暂无」。

## 3.5 手感配置（G-507）

每个 Bot 可以单独配置手感，默认值贴近原版：

| 配置 | 默认 | 含义 |
|---|---|---|
| `cps` | 15.0 | 每秒点击数上限 |
| `cpsJitter` | 2.0 | 点击间隔抖动（±） |
| `reactionMs` | 220.0 | 收到动作到真正执行的延时 |
| `reactionJitterMs` | 40.0 | 延时抖动 |
| `reach` | 3.99 | 攻击距离（格），超出返回 `OUT_OF_REACH` |
| `model` | `default` | 训练模型标识，训练脚本据此切换策略 |

- Java 侧 `BotProfile` 通过 `BotSpec.profile` 随 `SPAWN` 下发；运行时 `bot-profile.js` 负责校验与回落，
  `ClickGate` 控制连点节奏，`ReactionDelay` 控制反应延时，`actions.js` 的 `attack` 会先做距离判定。
- 训练侧 `BotConfig` 同名参数会在 `spawn_args()` 里拼成 `/gladiator spawn <user> cps=.. reaction=..`，
  所以训练脚本里每个 Bot 的配置是真正下发给运行时的，而不是只写在报告里。

## 3.6 奖励塑形与复盘（G-604）

`reward_mode="shaped"` 时用 `gladiator_env/shaping.py` 的权重（对齐 `bot-training.md`）：

| 项 | 权重 |
|---|---|
| 击杀 / 死亡 | +20 / -20 |
| 造成伤害 / 承受伤害 | +2.5 / -1.2（每 HP） |
| 连击 | +0.6 × combo，按串封顶 +3.0 |
| 命中后 1.5s 内保持 2.8–4.2 格 | +0.3 / 秒 |
| 空挥 | -0.3 |
| 超过 3s 无攻击交互 | -0.1 / 秒 |

每回合结束由 `Trainer` 调 `write_review(dir, review)` 产出 `episode-XXX.md` + `.json`，包含：
结果、伤害交换（伤害比/命中率/连击峰值/平均距离）、动作分布、奖励分解、关键事件。
默认写到 `training/artifacts/reviews/`，可用 `--review-dir` 改。

## 3.7 课程与对手池（G-605）

`train.py --stages idle:1,wander:1,scripted:2` 会按阶段切换 `env.opponent_mode`：

- `idle`：静止假人，只验证伤害/空挥信号；
- `wander`：随机走动、不反击；
- `random`：随机出招；
- `scripted`：远则靠近、近则 70% 攻击 / 30% 跳跃，会还手；
- `rusher`：猛冲型，距离 >1.5 就疾跑贴脸，贴身后 90% 攻击——给学习方持续压力；
- `aimbot`：100% 瞄准型，每步重新转向并在射程内必攻击，几乎不空挥——用来测“打不打得过完美对手”。

多风格对手是训练有效性的前提：只会打木桩的策略在第一次遇到会还手的对手时奖励会从 +50 掉到个位数。

### 3.8 多场地并行（G-606）

`--pairs N` 时 `multi_train.py` 会把 N 组「学习方 + 对手」放在同一条 X 轴上、中心间隔 `--arena-gap`（默认 60 格），
每组一个 `GladiatorEnv` + `Trainer` 线程，各自保存 `policy-pairN.npz` 与 `reviews/pair-N/`：

```powershell
python -m gladiator_train.multi_train --pairs 2 --episodes 4 --max-steps 200 `
  --auto-spawn --ports-base 8768 --arena-gap 60
```

注意端口与用户名要一一对应（每个 Bot 一个 `bot-runtime`），否则 HELLO 的 `botId` 会和 SPAWN 对不上、等待超时。

每条 history 记录 `opponentMode` 与 `epoch`，报告 `config.plan` 保留课程计划，便于对比各阶段表现。

### 3.9 续训（复用已有策略）

默认每次训练都会新建随机初始化的策略，历史 `policy.npz` **不会**被自动加载。若要从已有结果继续：

```powershell
python -m gladiator_train.train --users Bot_T1 Bot_T2 --ports 8770 8771 `
  --stages scripted:5,rusher:5 --epochs 4 `
  --init-model artifacts/runs/<run-id>/policy.npz --auto-spawn
```

- 加载前校验 `obs_dim` / `n_actions` / `hidden`，与当前环境不一致时**直接报错退出**，不会静默随机重置；
- 成功时日志打印 `[train] 已加载初始模型 ...（累计更新 N 次）`；
- 产物写入新的 run 目录，旧策略文件不被覆盖。

### 3.10 单策略多场地（VecEnv，`vec_train.py`）

`multi_train.py` 是「并行跑 N 个独立实验」：每个场地一个 `Policy`（各自随机初值），
最后按 mean 逐元素合并权重。**不同初值的 tanh 网络隐藏单元可以任意置换**，
逐元素平均会把隐藏层打碎，所以那种合并只对「同一策略的多个副本」有意义。

`vec_train.py` 是真正的并行采样：**一个策略**，M 个场地同时跑 episode，样本汇总成一个大 batch
做一次 PPO 更新（梯度聚合，不做权重平均）。

```powershell
cd training
python scripts/vec_train.py --arenas 4 --episodes 8 --workers 4 `
  --init-policy ../dashboard/var/models/dagger-1118b/policy.npz
```

- 场地 i 自动使用 `<prefix><i>A/B>` 与 `ports-base + 2i / +2i+1`，中心沿 X 轴按 `--arena-gap` 排开；
- 每个场地需要**两个已连上 8765 的 `bot-runtime`**（用户名/端口一一对应）；脚本只负责
  RCON `spawn`/摆位，不负责拉起 node 进程（与 `train.py` 一致，运行时由 Dashboard 或手工启动）；
- `--episodes` 是**每个场地的局数**（= 批数），总对局 = episodes × arenas；
- 只支持 `--algo ppo`（`reinforce` 会在 `VecTrainer` 处直接 `SystemExit`，不静默降级）；
- 每批日志：`[vec] batch i/N episodes steps 耗时 steps/s win=k/M updates=U`；
- 产物：`artifacts/runs/<时间>-vec-<tag>/` 下的 `policy.npz`、`meta.json`
  （场地/端口/config/summary）、`report.json`（逐批 history + summary）。

### 3.11 权重合并（`merge_policies.py`）

把多支训练出来的权重合成一支：**对齐 → 合并 → 门禁 → 蒸馏兜底**。
与 `multi_train.py --merge mean` 的区别只有一条，但很致命：那个是逐元素平均，
**不同随机初值的隐藏单元不在同一坐标系**，平均会把隐藏层打碎。

```powershell
cd training
# 三支 rebasin-ties 合并（不带校准集会用确定性伪样本，并在 stderr 警告）
python scripts/merge_policies.py `
  --inputs runs/a/policy.npz runs/b/policy.npz runs/c/policy.npz `
  --method rebasin-ties --trim 0.2 --calibration artifacts/bc/obs.npz `
  --out artifacts/merges/m1/merged.npz

# 带门禁：合并模型 vs 基线，同 seed 配对；不通过非零退出
python scripts/merge_policies.py --inputs a.npz b.npz --calibration artifacts/bc/obs.npz `
  --gate-baseline ../dashboard/var/models/dagger-1118b/policy.npz `
  --gate-episodes 8 --no-spawn --users Bot_V50A Bot_V50B

# 门禁不过时用集成蒸馏兜底（写出 merged.distill.npz）
... --distill --distill-episodes 6
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--method` | `rebasin-ties` | `mean` / `ties` / `rebasin-ties` / `rebasin-fisher` |
| `--trim` | `0.2` | TIES 裁剪比例（`0` = 退化为加权平均） |
| `--weights` | 等权 | 逗号分隔，如 `1,1,2` |
| `--calibration` | 伪样本 | `.npz`（含 `obs` 数组）或 `.jsonl`（每行 `{"obs": [...]}`） |
| `--gate-baseline` | 无 | 给出才跑门禁（复用 `eval_suite` 的 `run_suite` + `mcnemar_exact`） |
| `--allow-regression` | 关 | 门禁不通过也不非零退出 |
| `--distill` / `--distill-episodes` | 关 / `6` | 集成当老师蒸馏学生的兜底路径 |

- 输出 `merged.npz` + `summary.json`（method、输入 sha256、`permCount`、每支与合并的
  L2 差、门禁结论、蒸馏 KL）；
- 退出码：正常 `0`；门禁不通过 `1`（`--allow-regression` 可放行）；参数/文件错误 `2`；
- `rebasin-*` 必须给校准观测（或接受伪样本）；`rebasin-fisher` 的 Fisher 用各模型在
  校准样本上的贪心动作估计（无标签时的近似）。

### 3.12 自博弈训练（`selfplay_train.py`）

#### 为什么手写对手有天花板

`scripted` / `rusher` / `aimbot` 是**固定的状态机**：逼近阈值、出手概率、出手间隔都是常数。
当 BC 一致率到 1.0、对这三种对手全胜之后，梯度里已经没有新信息了——对手不会因为你变强
而变强，训练就开始在同一个水平线上打转。自博弈把**上一轮的自己**放上场：对手随 learner
一起进步，难度自动抬高；而且它用的是 learner 自己的动作分布，天然对齐（不会出现「人类
不会用的走位」这种分布外输入）。

#### 一轮做什么

```
round k: 存快照 round<k>.npz ──> 用 round<k-1>.npz 当对手训练 ──> 固定 seed 评估
```

- **对手落后一轮**：第 k 轮（k ≥ 1）的对手是 `round{k-1}.npz`，第 0 轮是 `--init-policy`
  （没给就用刚存下的 `round0.npz`，也就是当前策略本身）。落后一轮比「完全同步」稳：
  同步自博弈容易出现「你改我也改」的相互追逐；
- **快照不是 checkpoint**：它只用来当对手，learner 只有一份（`final.npz`）；
- **评估用固定 seed 集**（`eval_suite` 的 `run_suite`，对手仍是手写模式），所以
  `evalWinRate` 跨轮可比；训练胜率（`trainWinRate`，对手是快照）不可跨轮比较。

```powershell
cd training
python scripts/selfplay_train.py --rounds 3 --episodes-per-round 8 --arenas 4 `
  --init-policy ../dashboard/var/models/dagger-1118b/policy.npz --no-spawn
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rounds` | `3` | 轮数 |
| `--episodes-per-round` | `8` | 每轮每个场地的局数（批数），总对局 = 轮 × 批 × 场地 |
| `--arenas` | `4` | 场地数（**建议 ≤ 4**，见下） |
| `--init-policy` | 无 | 初始策略，同时是第 0 轮的对手 |
| `--snapshot-dir` | `artifacts/selfplay/<时间>` | 快照目录 |
| `--eval-episodes` | `4` | 每轮每个评估对手的局数（×3 对手 = 12 局） |
| `--eval-opponents` | `scripted,rusher,aimbot` | 评估用的手写对手 |
| `--opponent-epsilon` | `0.02` | 对手以该概率随机动作（避免快照过于确定） |
| `--victory` | `50` | 胜负常数（与 `train.py` 同口径） |

产物：`artifacts/runs/<时间>-selfplay-<tag>/`（`policy.npz` = final、`meta.json` 含每轮快照
sha256 与对手来源、`report.json` 含 `rounds[]` 与逐轮 `evalSummary`）+ 快照目录里的
`round0..k.npz`。

#### 需要哪些 bot-runtime

**8 个普通运行时（4 场地 × 2），都不带 `--brain`**：

| 场地 | 用户名 | 通道 B 端口 |
|---|---|---|
| 0 | `Bot_V0A` / `Bot_V0B` | 8770 / 8771 |
| 1 | `Bot_V1A` / `Bot_V1B` | 8772 / 8773 |
| 2 | `Bot_V2A` / `Bot_V2B` | 8774 / 8775 |
| 3 | `Bot_V3A` / `Bot_V3B` | 8776 / 8777 |

对手的动作由 **env 侧**（`GladiatorEnv(opponent_policy=...)`）算，再用与学习方相同的
通道 B 下发 —— 所以 `Bot_V*A`/`Bot_V*B` 一律是「执行动作的运行时」，第 2 个 Bot **不需要**
自己的 `--brain` 进程，也不需要人机对战那套 RCON 编排。

#### 算力预算：建议 4 场地以内

自博弈的对手是**真的在服务端跑**的第二个 Bot：它同样占实体 tick、同样吃插件主线程的动作
处理。4 场地 = 8 个在线 Bot，已经接近 1.8.8 单线程 TPS 的安全区（验收线 ≥ 19.5）；
超过 4 场地时收益主要体现在采样量上，但 TPS 掉下来会让每步的观测抖动变大、动作延迟变长，
反而伤害训练。要更多吞吐优先用 `vec_train.py`（同策略多场地）而不是继续加自博弈场地。

注意：

- 快照是「上一轮的自己」，如果某轮训崩了（策略退化），下一轮的对手也跟着退化；
  建议每轮看 `evalWinRate`，连续两轮下降就回滚；
- 对手策略与 learner 共用一个进程内的 numpy 前向（env 主线程），4 场地 ≈ 每步多 4 次
  64 单元前向，相对 I/O 可忽略。

## 5. 已知限制

- 自博弈（`selfplay_train.py`）的对手仍是「自己的快照」，不是人类或外部强策略；
  要接真人数据继续用 BC/DAgger（`import_human_match.py`）。
- 奖励有 `default_reward` 与 `shaped` 两种；塑形权重是参考值，未做逐项消融。
- 回合结束依赖 `state != Alive` 或 `health == 0`；服务端重启后需重新 `reset()`。
- 观测里的 `yaw` 标准化方向（`YAW_FORMULA="neg"`）尚未逐项标定，若发现转身方向相反，
  在 `gladiator_env/spaces.py` 改成 `"pos"` 并重测。
