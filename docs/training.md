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
  durationSeconds、policyUpdates）/ `evaluation` / `policyPath`。

`training/artifacts/` 已加入 `.gitignore`，训练产物不入库。

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

## 5. 已知限制

- 对手目前是脚本/随机，学到的是"打脚本"策略；接真实对手或 self-play 需要扩展 Trainer（当前只支持单智能体）。
- 奖励有 `default_reward` 与 `shaped` 两种；塑形权重是参考值，未做逐项消融。
- 回合结束依赖 `state != Alive` 或 `health == 0`；服务端重启后需重新 `reset()`。
- 观测里的 `yaw` 标准化方向（`YAW_FORMULA="neg"`）尚未逐项标定，若发现转身方向相反，
  在 `gladiator_env/spaces.py` 改成 `"pos"` 并重测。
