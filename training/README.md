# training — Layer 3

Gymnasium 环境封装。职责：观测编码、动作空间、奖励接口、episode 生命周期。

## 当前状态

M4 已落地（G-401 ~ G-406）：`gladiator_env/` 提供 `GladiatorEnv`，通过通道 B 观测/下发动作，
通过 RCON 准备训练场；`check_env` 通过；随机策略空手混战可跑满 200 step。

M5 起步（G-501）：`scripts/stress.py` 支持 N 个 Bot 并发压测并采样 TPS。4 Bot 实测
≈4.4 step/s、TPS 均值 20.0（最低 19.98）、无断连。压测期间可用 `/gladiator spectate` 进服旁观，
见 [`docs/spectating.md`](../docs/spectating.md)。

```text
gladiator_env/
  env.py               # GladiatorEnv（Gymnasium.Env）
  spaces.py            # 观测编码 / 动作解码（16 维 obs，7 个动作）
  reward.py            # 可插拔奖励（默认奖励不含规则判定）
  link.py              # 通道 B 同步客户端（后台线程 + asyncio）
  rcon.py              # 训练场控制面（Source RCON）
  testing.py           # 离线测试用的假链路
gladiator_train/       # 策略后端（不属于环境层）
  policy.py            # 单隐藏层 softmax 策略 + 手写反向传播 + Adam
  trainer.py           # REINFORCE 训练循环
  train.py             # CLI 入口：python -m gladiator_train.train
tests/                 # 43 个 unittest 用例
scripts/
  check_env.py         # Gymnasium check_env 自检（--live 连真服）
  random_brawl.py      # 随机策略混战，验收 G-406
  stress.py            # 多 Bot 并发压测 + TPS/堆采样，验收 G-501/G-502
  chaos.py             # 故障注入：杀 runtime / 链路闪断，验收 G-503
artifacts/             # 训练产物（policy.npz / train-report.json，已 gitignore）
```

## 使用

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 单元测试
.venv/Scripts/python.exe -m unittest discover -s tests -v

# 离线契约自检
.venv/Scripts/python.exe scripts/check_env.py
```

真实训练场（Paper + 每个 Bot 一条 bot-runtime 链路）：

```bash
# 1) 服务端开启 RCON（enable-rcon/rcon.port/rcon.password），启动 Paper
# 2) 启动两个运行时（各自一个通道 B 端口）
cd bot-runtime
node src/index.js --username Glad_Bot_1 --env-port 8766
node src/index.js --username Glad_Bot_2 --env-port 8767

# 3) 跑 200 步随机混战
cd training
.venv/Scripts/python.exe scripts/random_brawl.py --steps 200 --center 0.5,4,0.5

# 4) 4 Bot 压测（默认 30 分钟，--minutes 可缩短）
.venv/Scripts/python.exe scripts/stress.py --minutes 3 --report stress-report.json

# 5) 故障注入（需要 runtime 的 PID，顺序与 --users 一致）
.venv/Scripts/python.exe scripts/chaos.py --pids 1111,2222,3333,4444
```

运维（启动顺序 / 端口 / 排障）见 [`docs/operations.md`](../docs/operations.md)。

训练期间想自己进服旁观：`/gladiator spectate Glad_Bot_1`，退出用 `/gladiator unspectate`
（权限 `gladiator.admin` + `gladiator.spectate`，详见 [`docs/spectating.md`](../docs/spectating.md)）。

API 细节见 [`docs/training-env-api.md`](../docs/training-env-api.md)。

## 边界

- 只通过 Layer 2 的 WebSocket 协议操纵 Bot，不直接接触 mineflayer。
- 不实现具体 RL 算法（算法属策略后端）。
- 奖励函数可插拔，但不得包含对局规则判定（规则属 Layer 1 / Mode）。
- 传送 / 清背包 / 回血 / 地图准备属于调用方职责，环境只通过 RCON 调标准命令。

## 训练（课程 + 复盘）

```powershell
# 2 个 Bot 已在线上时：按课程训练（静止→走动→会还手），每局自动写复盘
.venv\Scripts\python.exe -m gladiator_train.train `
  --users Bot_T1 Bot_T2 --ports 8766 8767 `
  --stages idle:1,wander:1,scripted:2 --max-steps 120 --no-spawn `
  --cps 15 --cps-jitter 2 --reaction-ms 220 --reach 3.99 --model default
```

- 手感配置按 Bot 下发（`cps` / `cpsJitter` / `reactionMs` / `reach` / `model`），默认 15±2 CPS、220±40ms、3.99 格。
- `--reward-mode shaped`（默认已启用塑形）使用 `bot-training.md` 的权重表；`--stages` 走课程。
- 复盘写到 `training/artifacts/reviews/episode-XXX.md`（伤害交换、动作分布、奖励分解、关键事件），
  报告写到 `training/artifacts/train-report.json`。
