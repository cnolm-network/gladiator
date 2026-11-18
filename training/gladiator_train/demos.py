"""行为克隆热启动（P1.5）。

为什么需要它：PPO 从零学「贴身之后要挥剑」很慢——实测 24 训练 + 24 评估 0 胜，
策略塌成「approach 120 步 / attack 0 步」：因为 ``attack`` 只有命中才给正奖励，
「不挥剑」反而没有空挥惩罚，是稳定的局部最优。

做法：先用一个**只看观测向量**的脚本老师跑几局，把 (obs, 动作, 折扣回报) 存下来，
再用交叉熵把策略头拉到老师的行为上（只更新 W1/b1/W2/b2，价值头 Wv/bv 不动——
价值头属于 critic，必须由 PPO 自己学，否则会污染 GAE 的基线）。

老师刻意不读 env 内部状态：它只吃 23 维观测，所以离线、真机、单测三处行为完全一致。
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from gladiator_env.spaces import ACTION_NAMES, OBS_DIM, OBS_FIELDS

# --- 观测下标：全部从 OBS_FIELDS 查，不写死数字 ---------------------------------


def field_index(name: str) -> int:
    try:
        return OBS_FIELDS.index(name)
    except ValueError as exc:  # pragma: no cover - 只在改观测定义时触发
        raise AssertionError(f"OBS_FIELDS 里没有 {name}，观测定义变了但 demos.py 没跟上") from exc


IDX_SELF_HEALTH = field_index("self_health")
IDX_HAS_TARGET = field_index("has_target")
IDX_TARGET_DISTANCE = field_index("target_distance")
IDX_IN_REACH = field_index("in_reach")

ACTION_IDLE = ACTION_NAMES.index("idle")
ACTION_APPROACH = ACTION_NAMES.index("approach")
ACTION_ATTACK = ACTION_NAMES.index("attack")
ACTION_EAT = ACTION_NAMES.index("eat_gapple")

#: 观测里的归一化尺度（与 spaces.encode_observation 一致）。
HEALTH_SCALE = 20.0
DISTANCE_SCALE = 32.0

#: 老师认为「该继续靠近」的距离阈值（格）。
TEACHER_APPROACH_DISTANCE = 4.0
#: 老师认为「该回血」的血量阈值（点）。
TEACHER_HEAL_HEALTH = 8.0

#: 编译期自检：字段顺序变了就立刻炸，而不是静默错位。
assert OBS_DIM == len(OBS_FIELDS), f"OBS_DIM={OBS_DIM} 与 OBS_FIELDS={len(OBS_FIELDS)} 不一致"
assert IDX_IN_REACH == OBS_DIM - 1, "in_reach 应是最后一个观测字段"
assert 0 <= IDX_SELF_HEALTH < OBS_DIM and 0 <= IDX_HAS_TARGET < OBS_DIM
assert 0 <= IDX_TARGET_DISTANCE < OBS_DIM


def _get(obs: np.ndarray, index: int) -> float:
    try:
        return float(np.asarray(obs, dtype=np.float64).reshape(-1)[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def heuristic_action(obs: np.ndarray) -> int:
    """脚本老师：只读观测向量，返回动作编号。

    规则（顺序即优先级）：
    1. 射程内（``in_reach==1``）且看得到目标 → ``attack``；
    2. 看不到目标 → ``idle``；
    3. 距离 > 4 格 → ``approach``（先贴上去）；
    4. 自己血量 < 8 → ``eat_gapple``；
    5. 其余 → ``approach``。
    """
    has_target = _get(obs, IDX_HAS_TARGET) >= 0.5
    if not has_target:
        return ACTION_IDLE
    if _get(obs, IDX_IN_REACH) >= 0.5:
        return ACTION_ATTACK
    if _get(obs, IDX_TARGET_DISTANCE) * DISTANCE_SCALE > TEACHER_APPROACH_DISTANCE:
        return ACTION_APPROACH
    if _get(obs, IDX_SELF_HEALTH) * HEALTH_SCALE < TEACHER_HEAL_HEALTH:
        return ACTION_EAT
    return ACTION_APPROACH


def collect_demos(
    env,
    episodes: int = 4,
    max_steps: int = 120,
    teacher: Callable[[np.ndarray], int] = heuristic_action,
    gamma: float = 0.99,
) -> list[tuple[np.ndarray, int, float]]:
    """用老师跑 ``episodes`` 局，返回 ``[(obs, action, discounted_return)]``。

    回报按 **该局后续折扣回报** 计算（末步之后为 0）；单步环境不做 bootstrap。
    只支持单智能体环境（``single_agent=True``），否则动作契约不同，直接报错而不是瞎猜。
    """
    samples: list[tuple[np.ndarray, int, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, _info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_demos 只支持 single_agent=True 的环境（obs 应为向量）")
        steps: list[tuple[np.ndarray, int, float]] = []
        done = False
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            action = int(teacher(vector))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            steps.append((vector, action, float(reward)))
            obs = next_obs
            if terminated or truncated:
                done = True
                break
        # 反向累加折扣回报；局末之后视为 0（terminated/truncated 都按终止处理）。
        running = 0.0
        returns = [0.0] * len(steps)
        for index in range(len(steps) - 1, -1, -1):
            running = steps[index][2] + gamma * running
            returns[index] = running
        for (vector, action, _reward), value in zip(steps, returns):
            samples.append((vector, action, float(value)))
        if not done:
            # 截断的对局同样可用（回报就是到截断为止），这里不做额外处理。
            pass
    return samples


def collect_dagger_samples(
    env,
    policy,
    episodes: int = 4,
    max_steps: int = 120,
    teacher: Callable[[np.ndarray], int] = heuristic_action,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, int, float]]:
    """DAgger 的一轮采样：用**当前策略**跑，每步让老师给动作标签（P2.3）。

    与 :func:`collect_demos` 的区别是「谁在跑」：BC 用老师跑（数据分布 = 老师的状态分布），
    策略自己跑久了会走到老师从没去过的状态（covariate shift），DAgger 每轮都用
    当前策略的状态分布重新打标签，于是「策略越强、标签越贴它的实际处境」。

    返回结构与 :func:`collect_demos` 一致（``(obs, action, 0.0)``，第三个字段不用），
    这样可以直接喂给 :func:`behavior_clone`。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    samples: list[tuple[np.ndarray, int, float]] = []
    for _episode in range(int(max(0, episodes))):
        obs, _info = env.reset()
        if isinstance(obs, dict):
            raise ValueError("collect_dagger_samples 只支持 single_agent=True 的环境（obs 应为向量）")
        for _step in range(int(max(0, max_steps))):
            vector = np.asarray(obs, dtype=np.float64).reshape(-1)
            label = int(teacher(vector))          # 老师对**这个状态**的标签
            action = int(policy.act(vector, rng))  # 策略自己走，才能采到自己的状态分布
            samples.append((vector, label, 0.0))
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    return samples


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(params["W1"] @ x + params["b1"])
    p = _softmax(params["W2"] @ h + params["b2"])
    return h, p


def behavior_clone(
    policy,
    samples: Sequence[tuple[np.ndarray, int, float]],
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 64,
    entropy_coef: float = 0.0,
    seed: int = 0,
) -> dict:
    """交叉熵监督训练策略头（W1/b1/W2/b2），**不动价值头 Wv/bv**。

    返回 ``{"loss": [...每轮平均交叉熵...], "accuracy": 最终 teacher 动作命中率,
    "samples": 样本数, "epochs": 实际轮数}``。
    ``entropy_coef>0`` 时在损失里减去 ``β·H(π)``（鼓励保留探索）。
    """
    data = [(np.asarray(x, dtype=np.float64).reshape(-1), int(a)) for x, a, _ret in samples]
    result = {"loss": [], "accuracy": 0.0, "samples": len(data), "epochs": 0}
    if not data:
        return result

    rng = np.random.default_rng(seed)
    names = ("W1", "b1", "W2", "b2")
    m = {k: np.zeros_like(policy.params[k]) for k in names}
    v = {k: np.zeros_like(policy.params[k]) for k in names}
    step = 0
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    n_actions = policy.n_actions

    for _epoch in range(int(max(0, epochs))):
        order = rng.permutation(len(data))
        epoch_loss = 0.0
        for start in range(0, len(data), max(1, int(batch))):
            chunk = [data[i] for i in order[start:start + max(1, int(batch))]]
            grads = {k: np.zeros_like(policy.params[k]) for k in names}
            for x, action in chunk:
                h, p = _forward(policy.params, x)
                onehot = np.zeros(n_actions)
                onehot[action] = 1.0
                epoch_loss += float(-np.log(p[action] + 1e-12))
                dlogits = p - onehot
                if entropy_coef:
                    entropy = float(-np.sum(p * np.log(p + 1e-12)))
                    dlogits = dlogits - entropy_coef * p * (np.log(p + 1e-12) + entropy)
                grads["W2"] += np.outer(dlogits, h)
                grads["b2"] += dlogits
                dz = (policy.params["W2"].T @ dlogits) * (1.0 - h * h)
                grads["W1"] += np.outer(dz, x)
                grads["b1"] += dz
            n = float(len(chunk))
            step += 1
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            for key in names:
                g = grads[key] / n  # 最小化方向（交叉熵）
                m[key] = b1 * m[key] + (1.0 - b1) * g
                v[key] = b2 * v[key] + (1.0 - b2) * (g * g)
                policy.params[key] -= lr * (m[key] / bc1) / (np.sqrt(v[key] / bc2) + eps)
        result["loss"].append(round(epoch_loss / max(1, len(data)), 6))
        result["epochs"] += 1

    correct = 0
    for x, action in data:
        _h, p = _forward(policy.params, x)
        correct += int(int(np.argmax(p)) == action)
    result["accuracy"] = round(correct / max(1, len(data)), 4)
    policy.updates += int(result["epochs"])  # 与 PPO 的更新计数分开记，但保持单调
    return result


def teacher_agreement(policy, samples: Iterable[tuple[np.ndarray, int, float]]) -> float:
    """老师动作与策略 argmax 的一致率（BC 前后对比用）。"""
    total = 0
    correct = 0
    for x, action, _ret in samples:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        _h, p = _forward(policy.params, x)
        correct += int(int(np.argmax(p)) == int(action))
        total += 1
    return round(correct / total, 4) if total else 0.0
