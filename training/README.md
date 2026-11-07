# training — Layer 3

Gymnasium 环境封装。职责：观测编码、动作空间、奖励接口、episode 生命周期。

## 当前状态

M0：依赖可安装（gymnasium 1.4.0 / numpy 2.5.4 / websockets 17.4）；`gladiator_env/` 在 M4 建立（G-401 起）。

## 使用

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -c "import gymnasium; print(gymnasium.__version__)"
```

## 边界

- 只通过 Layer 2 的 WebSocket 协议操纵 Bot，不直接接触 mineflayer。
- 不实现具体 RL 算法（算法属策略后端）。
- 奖励函数可插拔，但不得包含对局规则判定（规则属 Layer 1 / Mode）。
