"""训练入口：真实服务端上的 REINFORCE + 课程学习 + 每局复盘。

用法（2 个 Bot、通道 B 端口 8766/8767）：

    python -m gladiator_train.train --users Bot_T1 Bot_T2 --ports 8766 8767 \
        --episodes 8 --max-steps 200 --auto-spawn

课程（对手从静止到会反击，逐阶段放开）：

    python -m gladiator_train.train --stages idle:3,wander:3,random:3,scripted:3

每个 Bot 的手感配置（G-507）默认 15±2 CPS / 220±40ms / **3.0 格**（G-708：3.99 是外挂级超限值，
要跑对照实验用 `--cheat-reach <值>` 显式打开）：

    ... --cps 17 --cps-jitter 3 --reaction-ms 180 --reach 4.2 --model aggressive

前置：Paper 已启动并加载 Gladiator；每个 Bot 一个 `bot-runtime` 进程；
RCON 已开启（控制面：摆位/回血/清背包）。详见 docs/training-env-api.md。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from typing import Any, List

from .policy import Policy, adopt_weights, validate_init_policy
from .runs import new_run_dir, run_paths
from .trainer import Trainer, save_report

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STAGES: tuple[tuple[str, int], ...] = ()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gladiator 策略训练（REINFORCE + 课程）")
    p.add_argument("--users", nargs="+", default=["Bot_T1", "Bot_T2"], help="Bot 用户名，顺序对应 --ports")
    p.add_argument("--ports", nargs="+", type=int, default=[8766, 8767], help="每个 Bot 的通道 B 端口")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--mc-port", type=int, default=25565, help="仅供日志/占位（Bot 由运行时拉起）")
    p.add_argument("--env-host", default="127.0.0.1", help="通道 B 主机")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", default="devpass")
    p.add_argument("--center", default="0.5,4,0.5", help="竞技场中心 x,y,z")
    p.add_argument("--spacing", type=float, default=6.0, help="Bot 间距（格）")
    p.add_argument("--arena-jitter", default="2,1",
                   help="起始分布抖动 \"中心偏移,间距抖动\"（格）；\"0,0\" = 完全确定（旧行为）")
    p.add_argument("--prep-seconds", type=float, default=3.0, help="开局准备期（秒），期间双方不动")
    p.add_argument("--episodes", type=int, default=10, help="无 --stages 时的轮次；有 --stages 时是每阶段的轮次")
    p.add_argument("--epochs", type=int, default=1,
                   help="训练回合数：每回合完整按计划执行一遍（每轮次更新一次参数）")
    p.add_argument("--init-policy", default="",
                   help="初始策略 npz（模型注册表导出）；优先于 --init-model，只加载权重不加载优化器状态")
    p.add_argument("--model-id", default="", help="模型注册表 id（仅写入 report/config）")
    p.add_argument("--model-name", default="", help="模型显示名（仅写入 report/config）")
    p.add_argument("--persist-to", default="",
                   help="持久化目标模型 id（仅写入 report/config；写回由 Dashboard 负责）")
    p.add_argument("--init-model", default="",
                   help="从已有 policy.npz 初始化权重（续训）；留空则随机初始化")
    p.add_argument("--stages", default="", help="课程：idle:3,wander:3,random:3,scripted:3（模式:回合数）")
    p.add_argument("--opponent-mode", default="scripted", choices=["random", "wander", "idle", "scripted", "rusher", "aimbot"],
                   help="无 --stages 时对手的行为")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--action-ticks", type=int, default=2, help="每个动作持续的控制周期数")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="PPO 学习率（每个 update 后线性衰减到初始的 10%%）")
    p.add_argument("--hidden", type=int, default=64)
    # --- P2.3：策略饱和治理 ---
    p.add_argument("--logit-temp", type=float, default=None,
                   help="softmax 温度（>1 更平、熵更大）；缺省沿用模型里的值（旧模型 1.0）")
    p.add_argument("--reset-policy-head", action="store_true",
                   help="训练前用 Xavier 重设策略头 W2/b2（保留 W1/b1 与价值头）")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="权重衰减（只作用于 W1/W2，bias 不衰减）；0 = 关闭")
    p.add_argument("--beta", type=float, default=0.01, help="熵正则系数")
    # --- P1：算法（PPO + GAE） ---
    p.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"],
                   help="更新算法：ppo（默认，clip+GAE）或 reinforce（旧行为）")
    p.add_argument("--ppo-epochs", type=int, default=4,
                   help="PPO 每回合复用同一批数据的更新轮数（默认 4）")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 重要性采样裁剪 ε")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE λ")
    p.add_argument("--value-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--entropy-coef", type=float, default=0.02,
                   help="熵正则系数（PPO 用，默认 0.02；reinforce 用 --beta）")
    # P1.6：评估口径。训练用采样，评估默认也采样（否则量到的是另一个策略）；
    # --eval-greedy 反选回 argmax。
    p.add_argument("--eval-sampling", dest="eval_sampling", action="store_true", default=True,
                   help="评估时按策略采样（默认；与训练行为一致）")
    p.add_argument("--eval-greedy", dest="eval_sampling", action="store_false",
                   help="评估时用贪心 argmax（反选 --eval-sampling）")
    p.add_argument("--eval-seed", type=int, default=7,
                   help="评估固定种子（写入 report.config.evalSeed，跨轮可比）")
    # --- 动作空间（v1 单头 / v2 九头 factorized） ---
    p.add_argument("--action-space", default="v1", choices=["v1", "v2"],
                   help="v1 = 11 个宏动作的单头 softmax（默认）；v2 = 10 头 MultiDiscrete"
                        "（BC 需配 --bc-teacher v4）")
    # --- P1.5：行为克隆热启动 ---
    p.add_argument("--bc-episodes", type=int, default=4,
                   help="BC 老师的对局数（0 表示不采集）")
    p.add_argument("--bc-epochs", type=int, default=30,
                   help="BC 监督训练轮数；0 = 完全跳过（旧行为）")
    p.add_argument("--bc-max-steps", type=int, default=120,
                   help="BC 每局最多步数")
    # --- 老师 v2 + BC 软目标（P2.3 后续：别再让 BC 把策略压死） ---
    p.add_argument("--bc-teacher", default="v1", choices=["v1", "v2", "v3", "v4"],
                   help="BC/DAgger 的老师版本：v1 = 静态四规则（默认）；"
                        "v2 = v1 + 间距/鱼竿/金苹果/暴击跳/w-tap/低血格挡；"
                        "v3 = 冷却/暴击窗口感知（0.2s 步长，最好的规则老师）；"
                        "v4 = 跑在 v2 十头动作空间上的 1.8 招式组合（**v2 动作空间下唯一可用**）")
    p.add_argument("--bc-temperature", type=float, default=1.0,
                   help="BC 软目标温度 T：q = softmax(log p_teacher / T)。"
                        "T>1 更平（熵更大、PPO 推得动），T<1 更尖；1.0 = 旧行为")
    p.add_argument("--bc-label-smoothing", type=float, default=0.0,
                   help="BC 标签平滑 eps：q 与均匀分布按 eps 混合（0 = 关闭）")
    # --- P2.3 块 C：DAgger 式迭代模仿 ---
    p.add_argument("--dagger-rounds", type=int, default=0,
                   help="DAgger 轮数（0 = 关闭）；每轮用当前策略跑局、老师打标签、再做 BC + 评估")
    p.add_argument("--dagger-episodes", type=int, default=4,
                   help="DAgger 每轮用当前策略跑的对局数")
    p.add_argument("--dagger-epochs", type=int, default=2,
                   help="DAgger 每轮 BC 的监督训练轮数")
    p.add_argument("--dagger-max-steps", type=int, default=None,
                   help="DAgger 每局最多步数（默认同 --bc-max-steps）")
    p.add_argument("--dagger-target-agreement", type=float, default=0.9,
                   help="老师一致率达到该值就提前结束 DAgger")
    # --- P2：return 归一化（PPO 标准；关掉用于对照实验） ---
    # --- P3：PPO 稳定性 ---
    p.add_argument("--entropy-floor", type=float, default=0.15,
                   help="熵下限：低于该值时自动加倍 entropy_coef（0 = 关闭下限保护）")
    p.add_argument("--entropy-coef-max", type=float, default=2.0,
                   help="自适应熵系数上限（P2.3 从 0.5 抬到 2.0：实测 0.5 顶死仍拉不动饱和策略）")
    p.add_argument("--target-entropy", type=float, default=0.30,
                   help="目标熵（0 = 关闭目标熵控制器，退回 P2.1 的翻倍逻辑）")
    p.add_argument("--value-clip", type=float, default=0.2,
                   help="价值头裁剪幅度（以行为策略的 V 为锚；0 = 关闭）")
    p.add_argument("--minibatch", type=int, default=256,
                   help="每批更新的样本数（0 = 整批一次更新，旧行为）")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                   help="全局梯度 L2 裁剪阈值（0 = 不裁剪）")
    p.add_argument("--no-lr-decay", dest="lr_decay", action="store_false", default=True,
                   help="关闭 lr 线性衰减")
    p.add_argument("--eval-before-train", action="store_true",
                   help="训练开始前先跑一遍评估（采样 + 贪心），写入 evaluationBefore / evaluationGreedyBefore")
    p.add_argument("--victory", type=float, default=None,
                   help="覆盖胜负常数（默认取 ShapingWeights.victory=50；优先级高于预设）")
    p.add_argument("--normalize-returns", dest="normalize_returns", action="store_true",
                   default=True, help="PPO 目标/优势按 running mean-std 归一化（默认开）")
    p.add_argument("--no-normalize-returns", dest="normalize_returns", action="store_false",
                   help="关闭 return 归一化")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--auto-spawn", action="store_true", help="由训练脚本调用 /gladiator spawn")
    p.add_argument("--no-spawn", action="store_true", help="跳过 spawn，直接使用已在线 Bot")
    p.add_argument("--eval-epochs", type=int, default=None,
                   help="评估回合数：训练后完整按计划走几遍（贪心、不学习）；默认 1")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help=argparse.SUPPRESS)  # 旧参数，等价于 --eval-epochs
    p.add_argument("--run-dir", default=None,
                   help="本次运行的产物目录；默认 artifacts/runs/<时间>-train-<用户名>")
    p.add_argument("--save", default=None, help="策略文件；默认 <run-dir>/policy.npz")
    p.add_argument("--report", default=None, help="报告；默认 <run-dir>/report.json")
    p.add_argument("--review-dir", default=None, help="每局复盘目录；默认 <run-dir>/reviews")
    # --- 每个 Bot 的手感配置（G-507） ---
    p.add_argument("--cps", type=float, default=15.0, help="学习 Bot 的每秒点击数")
    p.add_argument("--cps-jitter", type=float, default=2.0)
    p.add_argument("--reaction-ms", type=float, default=220.0)
    p.add_argument("--reaction-jitter-ms", type=float, default=40.0)
    p.add_argument("--reach", type=float, default=3.0,
                   help="学习 Bot 攻击距离（默认 3.0 = 原版水平；>3 需 --cheat-reach）")
    p.add_argument("--cheat-reach", type=float, default=None,
                   help="作弊开关：显式指定超限攻击距离（同时对双方生效，仅对照实验）")
    p.add_argument("--model", default="default")
    # --- 战斗辅助（“外挂级”能力，默认关闭，按课程逐步打开） ---
    p.add_argument("--wtap", action="store_true", help="命中后 sprint reset（W-tap 控距）")
    p.add_argument("--auto-rod", action="store_true", help="中距离自动放鱼竿击退")
    p.add_argument("--auto-block", action="store_true", help="空闲时自动举剑格挡")
    p.add_argument("--aim-lock", action="store_true", help="攻击前 3D 对准目标")
    # --- 对手手感（默认稍慢，给学习方留出空间） ---
    p.add_argument("--opp-cps", type=float, default=12.0)
    p.add_argument("--opp-cps-jitter", type=float, default=2.0)
    p.add_argument("--opp-reaction-ms", type=float, default=260.0)
    p.add_argument("--opp-reaction-jitter-ms", type=float, default=50.0)
    p.add_argument("--opp-reach", type=float, default=3.0,
                   help="对手攻击距离（默认 3.0；>3 需 --cheat-reach）")
    p.add_argument("--opp-model", default="default")
    p.add_argument("--opp-assist", action="store_true", help="对手开启全部战斗辅助（压力测试）")
    p.add_argument("--reward-preset", default=None,
                   help="reward 预设 JSON 路径（见 training/gladiator_env/reward_presets.py）")
    return p


def parse_stages(raw: str, fallback_episodes: int) -> list[tuple[str, int]]:
    """``idle:3,wander:3`` → [("idle", 3), ("wander", 3)]；空串返回空列表。"""
    if not raw.strip():
        return []
    stages: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"阶段格式应为 模式:回合数，收到 {item}")
        mode, count = item.split(":", 1)
        mode = mode.strip()
        if mode not in ("random", "wander", "idle", "scripted", "rusher", "aimbot"):
            raise ValueError(f"未知对手模式: {mode}")
        stages.append((mode, int(count) if count.strip() else fallback_episodes))
    return stages


def _bucket(records: list[dict], username: str) -> dict:
    """一组的胜率/均 reward（P1-1）。"""
    episodes = len(records)
    wins = sum(1 for item in records
               if bool(item.get("win")) or (item.get("winner") is not None
                                            and item.get("winner") == username))
    rewards = [float(item.get("totalReward") or 0.0) for item in records]
    return {
        "episodes": episodes,
        "wins": wins,
        "winRate": round(wins / episodes, 4) if episodes else 0.0,
        "meanReward": round(sum(rewards) / episodes, 4) if episodes else 0.0,
    }


def summarize_records(records: list[dict], username: str) -> dict:
    """``{"overall": {...}, "byOpponent": {mode: {...}}}``。"""
    groups: dict[str, list[dict]] = {}
    for item in records:
        key = str(item.get("opponentMode") or "unknown")
        groups.setdefault(key, []).append(item)
    return {
        "overall": _bucket(records, username),
        "byOpponent": {key: _bucket(items, username) for key, items in groups.items()},
    }


def _fmt_eval(evaluation: dict | None) -> str:
    """把 DAgger 的单轮评估压成一行日志：``采样 8/12 · 贪心 10/12``。"""
    if not evaluation:
        return "未评估"
    parts = []
    for label, key in (("采样", "sampling"), ("贪心", "greedy")):
        bucket = evaluation.get(key)
        if isinstance(bucket, dict) and bucket.get("episodes"):
            parts.append(f"{label} {bucket['wins']}/{bucket['episodes']}")
    return " · ".join(parts) if parts else "未评估"


def _default_victory() -> float:
    from gladiator_env.shaping import ShapingWeights

    return float(ShapingWeights().victory)


def _default_shaping_cap() -> float:
    from gladiator_env.shaping import ShapingWeights

    return float(ShapingWeights().shaping_cap)


def _mean_ppo_metrics(records: list[dict]) -> dict:
    """history 里 PPO 指标的平均值（reinforce 路径没有这些字段 → 全 None）。"""
    keys = ("policyLoss", "valueLoss", "entropy", "approxKL", "clipFraction",
            "explainedVariance")
    out: dict = {}
    for key in keys:
        values = [float(item[key]) for item in records
                  if isinstance(item.get(key), (int, float))]
        out[key] = round(sum(values) / len(values), 6) if values else None
    return out


def sliding_windows(records: list[dict], username: str, size: int = 10) -> list[dict]:
    """每 ``size`` 局一个窗口的胜率/均 reward（``summary.windows``）。"""
    if size <= 0:
        return []
    out: list[dict] = []
    for start in range(0, len(records), size):
        chunk = records[start:start + size]
        bucket = _bucket(chunk, username)
        out.append({
            "startEpisode": start + 1,
            "episodes": bucket["episodes"],
            "wins": bucket["wins"],
            "winRate": bucket["winRate"],
            "meanReward": bucket["meanReward"],
        })
    return out


def _agreement_scalar(value: Any) -> float | None:
    """把一致率压成一个数字（v1 是 float，v2 是 dict → 取逐头平均）。

    report 里 ``agreementBefore/After`` 必须始终是「一个可比数字」：dashboard 与
    ``--dagger-target-agreement`` 都按单值读，v2 的十头明细另存
    ``headAgreementBefore/After``（见 :func:`_agreement_detail`）。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("meanHeadAgreement")
        return None if raw is None else float(raw)
    return float(value)


def _agreement_detail(value: Any, key: str):
    """v2 的明细字段（``exactMatch`` / ``perHead``）；v1 返回 ``None``。"""
    if isinstance(value, dict):
        return value.get(key)
    return None


def _fmt_agreement(value: Any) -> str:
    """打印用：v1 一个数，v2 加上全体完全一致率。"""
    scalar = _agreement_scalar(value)
    if scalar is None:
        return "n/a"
    if isinstance(value, dict):
        exact = value.get("exactMatch")
        if exact is None:
            return f"{scalar:.3f}"
        return f"{scalar:.3f}（逐头均）/ 全一致 {float(exact):.3f}"
    return f"{scalar:.3f}"


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 参数组合校验放在建目录之前：拒绝时不该留下空 run 目录。
    # G-710：v2 动作空间下 BC/DAgger 只允许 teacher v4——v1/v2/v3 给的是「一个动作编号」，
    # 没有 10 个头的标签，强行喂给 v2 只会把 10 个头压成 1 个（静默错标签）。
    if args.action_space == "v2" and (args.bc_episodes > 0 or args.dagger_rounds > 0):
        if str(args.bc_teacher or "").strip().lower() != "v4":
            print(
                "[train] v2 下的 BC 只支持 teacher v4："
                f"收到 --bc-teacher {args.bc_teacher}。"
                "老师 v1/v2/v3 输出的是 v1 宏动作编号，没有 v2 的 10 头标签。"
                "请加 --bc-teacher v4，或把 --bc-episodes / --dagger-rounds 设为 0。",
                file=sys.stderr)
            return 2
    run_dir = args.run_dir or new_run_dir("train", "-".join(args.users[:2]))
    paths = run_paths(run_dir, "train")
    args.save = args.save or paths["policy"]
    args.report = args.report or paths["report"]
    args.review_dir = args.review_dir or paths["reviews"]
    print(f"[train] 运行目录 {run_dir}")

    center = tuple(float(v) for v in args.center.split(","))
    if len(center) != 3:
        print("--center 需要 x,y,z", file=sys.stderr)
        return 2
    if len(args.users) != len(args.ports):
        print("--users 与 --ports 数量必须一致", file=sys.stderr)
        return 2
    try:
        stages = parse_stages(args.stages, args.episodes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # 延迟导入：允许在没有 gymnasium 的环境里只跑 `python -m gladiator_train.policy` 之类
    try:
        from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv
        from gladiator_env.env import check_reach, cheat_reach_banner
        from gladiator_env.env import OPPONENT_PROFILES, OPPONENT_REACH_OFFSET
        from gladiator_env.reward_presets import load_preset
        from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可读提示
        print(f"无法导入 gladiator_env：{exc}（请在 training/ 下运行）", file=sys.stderr)
        return 2

    # 初始策略在创建任何环境之前校验：无效直接 exit 2，不产生目录/进程。
    init_path = args.init_policy or args.init_model
    loaded, init_error = validate_init_policy(
        init_path, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden,
        action_space=args.action_space)
    if init_error:
        print(f"[train] {init_error}", file=sys.stderr)
        return 2

    # G-708：作弊开关一旦给了值，双方都按它来（对照实验的语义是「同一把超限尺子」）。
    cheat_reach = args.cheat_reach is not None
    reach_value = check_reach(args.cheat_reach if cheat_reach else args.reach, cheat=cheat_reach)
    opp_reach_value = check_reach(args.cheat_reach if cheat_reach else args.opp_reach,
                                  cheat=cheat_reach)
    if cheat_reach:
        print(cheat_reach_banner(reach_value), file=sys.stderr)

    bots = [BotConfig(
        username=user,
        env_url=f"ws://{args.env_host}:{port}",
        cps=args.cps,
        cps_jitter=args.cps_jitter,
        reaction_ms=args.reaction_ms,
        reaction_jitter_ms=args.reaction_jitter_ms,
        reach=reach_value,
        cheat_reach=cheat_reach,
        model=args.model,
        wtap=args.wtap,
        auto_rod=args.auto_rod,
        auto_block=args.auto_block,
        aim_lock=args.aim_lock,
    ) for user, port in zip(args.users, args.ports)]

    # 对手的独立配置：同一个训练场里两个 Bot 手感不同，才谈得上“对齐不同模型”。
    for bot in bots[1:]:
        bot.cps = args.opp_cps
        bot.cps_jitter = args.opp_cps_jitter
        bot.reaction_ms = args.opp_reaction_ms
        bot.reaction_jitter_ms = args.opp_reaction_jitter_ms
        bot.reach = opp_reach_value
        bot.cheat_reach = cheat_reach
        bot.model = args.opp_model
        if args.opp_assist:
            bot.wtap = True
            bot.auto_rod = True
            bot.auto_block = True
            bot.aim_lock = True

    try:
        jitter_parts = [float(v) for v in str(args.arena_jitter).split(",") if v.strip()]
        if len(jitter_parts) != 2:
            raise ValueError
    except ValueError:
        print("--arena-jitter 需要两个数字，例如 2,1 或 0,0", file=sys.stderr)
        return 2
    arena = ArenaConfig(center=center, spacing=args.spacing, prep_seconds=args.prep_seconds,
                        jitter=(jitter_parts[0], jitter_parts[1]))
    env = GladiatorEnv(
        bots,
        arena,
        rcon={"host": args.host, "port": args.rcon_port, "password": args.rcon_password},
        action_ticks=args.action_ticks,
        max_steps=args.max_steps,
        single_agent=True,
        auto_spawn=args.auto_spawn and not args.no_spawn,
        opponent_mode=stages[0][0] if stages else args.opponent_mode,
        reward_preset=load_preset(args.reward_preset) if args.reward_preset else None,
        action_space=args.action_space,
    )

    # P3：--victory 覆盖（优先级高于 reward-preset 里的 base.victory）。
    if args.victory is not None:
        env.shaping.weights.victory = float(args.victory)
        print(f"[train] 胜负常数 victory={args.victory}（覆盖预设默认）")

    obs_dim = int(env.observation_space.shape[0])
    if args.action_space == "v2":
        # MultiDiscrete 没有 .n：动作维度 = 各头大小之和（37），策略按头重建。
        n_actions = int(sum(int(v) for v in env.action_space.nvec))
        print(f"[train] 动作空间 v2：{len(env.action_space.nvec)} 个头，"
              f"nvec={list(int(v) for v in env.action_space.nvec)}")
    else:
        n_actions = int(env.action_space.n)
    policy = Policy(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=args.hidden,
        seed=args.seed,
        lr=args.lr,
        beta=args.beta,
        temperature=(args.logit_temp if args.logit_temp else 1.0),
        weight_decay=args.weight_decay,
        action_space=args.action_space,
    )
    if loaded is not None:
        # 只接管权重（不接管 optimizer/updates 状态），lr/beta 仍用本次参数。
        adopt_weights(policy, loaded)
        label = args.model_name or args.model_id or init_path
        print(f"[train] 已加载初始策略 {label}（obs={obs_dim} actions={n_actions} hidden={args.hidden}）")
        # P2.3：温度随模型走，但 --logit-temp 显式给出时以命令行优先。
        if args.logit_temp is not None:
            policy.temperature = float(args.logit_temp)
    if args.logit_temp is not None:
        print(f"[train] softmax 温度 temperature={policy.temperature:g}")
    # P2.3：重设策略头（保留隐层与价值头）——放在 BC / 训练之前。
    if args.reset_policy_head:
        policy.reset_policy_head(seed=args.seed)
        print("[p23] 策略头已重设（W2/b2 Xavier，W1/b1 与价值头保留）")
    # --- P1.5：行为克隆热启动（必须在 init-policy 载入之后、正式训练之前） ---
    bc_info = {"episodes": 0, "epochs": 0, "samples": 0, "accuracy": None,
               "agreementBefore": None, "agreementAfter": None, "loss": None,
               "teacher": args.bc_teacher, "temperature": float(args.bc_temperature),
               "labelSmoothing": float(args.bc_label_smoothing),
               "teacherEntropy": None, "perHeadAccuracy": None, "heads": None,
               "totalWeight": None, "agreementBeforeExact": None,
               "agreementAfterExact": None, "headAgreementBefore": None,
               "headAgreementAfter": None}
    if args.bc_epochs > 0 and args.bc_episodes > 0:
        from .demos import behavior_clone, collect_demos, teacher_agreement

        print(f"[bc] 采集老师对局：{args.bc_episodes} 局 × 最多 {args.bc_max_steps} 步"
              f"（teacher={args.bc_teacher} / space={args.action_space}）")
        samples = collect_demos(env, episodes=args.bc_episodes, max_steps=args.bc_max_steps,
                                teacher_version=args.bc_teacher)
        before = teacher_agreement(policy, samples)
        bc_result = behavior_clone(policy, samples, epochs=args.bc_epochs,
                                   temperature=args.bc_temperature,
                                   label_smoothing=args.bc_label_smoothing)
        # BC 后再量一次一致率：与 accuracy 同口径（argmax 命中老师动作），
        # 单独存一份是为了让「BC 前 / BC 后」在 report 里成对可比（P2.2）。
        after = teacher_agreement(policy, samples)
        bc_info = {
            "episodes": int(args.bc_episodes),
            "epochs": int(args.bc_epochs),
            "samples": len(samples),
            "accuracy": bc_result["accuracy"],
            # v1 是 float；v2 是「逐头平均」（单值），明细另存 headAgreement*。
            "agreementBefore": _agreement_scalar(before),
            "agreementAfter": _agreement_scalar(after),
            "agreementBeforeExact": _agreement_detail(before, "exactMatch"),
            "agreementAfterExact": _agreement_detail(after, "exactMatch"),
            "headAgreementBefore": _agreement_detail(before, "perHead"),
            "headAgreementAfter": _agreement_detail(after, "perHead"),
            "loss": bc_result["loss"][-1] if bc_result["loss"] else None,
            "teacher": args.bc_teacher,
            "temperature": float(bc_result["temperature"]),
            "labelSmoothing": float(bc_result["labelSmoothing"]),
            "teacherEntropy": bc_result["teacherEntropy"],
            # G-710：v2 才有（v1 为 None），逐头的 argmax 命中率
            "perHeadAccuracy": bc_result.get("perHeadAccuracy"),
            "heads": bc_result.get("heads") or policy.head_names(),
            "totalWeight": bc_result.get("totalWeight"),
        }
        print(
            f"[bc] teacher 一致率 {_fmt_agreement(before)} → {_fmt_agreement(after)}（BC 后）"
            f" / loss {bc_info['loss']} / 样本 {len(samples)} / {bc_result['epochs']} 轮"
            f" / T={bc_info['temperature']:g} eps={bc_info['labelSmoothing']:g}"
            f" / 老师熵 {bc_info['teacherEntropy']}"
        )
        if bc_info["perHeadAccuracy"]:
            head_names = bc_info["heads"] or []
            print("[bc] 逐头准确率 " + " ".join(
                f"{name}={float(value):.2f}"
                for name, value in zip(head_names, bc_info["perHeadAccuracy"])))
    elif args.bc_epochs <= 0:
        print("[bc] 跳过（--bc-epochs 0）")
    else:
        print("[bc] 跳过（--bc-episodes 0）：不采集老师对局，策略只由 PPO 学")

    trainer = Trainer(
        env, policy,
        gamma=args.gamma,
        seed=args.seed,
        review_dir=args.review_dir,
        algo=args.algo,
        epochs=args.ppo_epochs,
        clip=args.clip,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        normalize_returns=args.normalize_returns,
        lr_decay=args.lr_decay,
        entropy_floor=args.entropy_floor,
        entropy_coef_max=args.entropy_coef_max,
        target_entropy=args.target_entropy,
        minibatch=args.minibatch,
        max_grad_norm=args.max_grad_norm,
        value_clip=args.value_clip,
        action_space=args.action_space,
    )

    started = time.time()
    plan = stages or [(args.opponent_mode, args.episodes)]
    epochs = max(1, int(args.epochs))
    eval_epochs = args.eval_epochs if args.eval_epochs is not None else (
        args.eval_episodes if args.eval_episodes is not None else 1)
    eval_epochs = max(0, int(eval_epochs))
    per_epoch = sum(count for _mode, count in plan)
    # G-710：`--episodes 0`（纯蒸馏 / 只做训练前评估）时训练计划每回合 0 局，
    # 照 plan 跑评估会得到 0 局、胜率恒 0/0。此时每个阶段每回合跑 **1** 局，
    # 于是 `eval_epochs` 遍正好是 `eval_epochs` 局（按 eval_epochs 局塞进 count
    # 会变成 eval_epochs² 局——第一版就是这么写成 36 局的）。
    # per_epoch>0 时 eval_plan == plan，与旧行为完全一致。
    eval_plan = plan if per_epoch > 0 else [(mode, 1) for mode, _count in plan]
    eval_episodes = eval_epochs * sum(count for _mode, count in eval_plan)
    print(
        f"[train] bots={args.users} ports={args.ports} plan={plan} "
        f"max_steps={args.max_steps} obs={env.observation_space.shape} "
        f"actions={n_actions} space={args.action_space}"
    )
    if args.algo == "ppo":
        print(
            f"[train] 算法 PPO：epochs={max(1, args.ppo_epochs)} clip={args.clip} "
            f"gaeλ={args.gae_lambda} valueCoef={args.value_coef} entropyCoef={args.entropy_coef}"
        )
    else:
        print(f"[train] 算法 REINFORCE：beta={args.beta} gamma={args.gamma}")
    print(
        f"[train] 学习方 cps={bots[0].cps}±{bots[0].cps_jitter} "
        f"reaction={bots[0].reaction_ms}±{bots[0].reaction_jitter_ms}ms reach={bots[0].reach}"
    )
    print(
        f"[train] 训练回合={epochs} × 每回合 {per_epoch} 轮次 = {epochs * per_epoch} 局；"
        f"评估回合={eval_epochs}（共 {eval_episodes} 局，"
        f"{'采样' if args.eval_sampling else '贪心'}不学习）"
    )

    # 评估：同样按计划走，跑 eval_epochs 遍；不更新参数。
    # P1.6：出两组——按 eval_sampling 的「训练同口径」与强制 argmax 的「贪心口径」，
    # 两次都用同一个 eval_seed 重置采样器，保证跨轮/跨口径可比。
    def run_eval_pass(greedy: bool) -> list[dict]:
        trainer.rng = np.random.default_rng(args.eval_seed)
        records: list[dict] = []
        for epoch in range(1, eval_epochs + 1):
            for mode, count in eval_plan:
                env.opponent_mode = mode
                for record in trainer.evaluate(count, greedy=greedy):
                    record["opponentMode"] = mode
                    record["evalEpoch"] = epoch
                    records.append(record)
        return records

    # P2.2：按**整份计划**设定 lr 衰减的总更新数（跨阶段连续、不回弹）。
    if args.lr_decay:
        total_updates = trainer.plan_lr_schedule(epochs * per_epoch, args.max_steps)
        print(
            f"[train] lr 计划：{args.lr:g} → {args.lr * 0.1:g}（共 {total_updates} 次参数更新，"
            f"单调衰减至初始 10%）"
        )

    # P2.2 块 B：训练前评估。把「BC 热启动后的水平」与「PPO 训练后的水平」分开量，
    # 否则 10/12 胜到底来自 BC 还是 PPO 无从判断。
    eval_before: list[dict] | None = None
    eval_greedy_before: list[dict] | None = None

    def eval_pair() -> tuple[list[dict], list[dict]]:
        """按 eval_sampling 的口径跑两组评估（采样 + 贪心）。"""
        sampling = run_eval_pass(greedy=not args.eval_sampling)
        greedy = run_eval_pass(greedy=True) if args.eval_sampling else sampling
        return sampling, greedy

    # --- P2.3 块 C：DAgger 式迭代模仿（在 PPO 之前） ---
    dagger_report: list[dict] = []
    if args.dagger_rounds > 0:
        from .demos import behavior_clone, collect_dagger_samples, teacher_agreement

        dagger_steps = args.dagger_max_steps if args.dagger_max_steps else args.bc_max_steps
        # 第 0 轮 = 现有的 BC 阶段（可能被 --bc-episodes 0 跳过），作为对照基线。
        base_eval = {"sampling": None, "greedy": None}
        if eval_epochs > 0:
            sampling, greedy = eval_pair()
            base_eval = {"sampling": _bucket(sampling, bots[0].username),
                         "greedy": _bucket(greedy, bots[0].username)}
        dagger_report.append({
            "round": 0,
            "samples": bc_info["samples"],
            "agreement": bc_info.get("agreementAfter"),
            "loss": bc_info.get("loss"),
            "evaluation": base_eval,
        })
        print(
            f"[dagger] 第 0 轮（BC 基线）：samples={bc_info['samples']} "
            f"agreement={bc_info.get('agreementAfter')} 评估={_fmt_eval(base_eval)}"
        )
        for round_no in range(1, max(0, int(args.dagger_rounds)) + 1):
            samples = collect_dagger_samples(
                env, policy,
                episodes=args.dagger_episodes,
                max_steps=dagger_steps,
                rng=np.random.default_rng(args.seed + round_no),
                teacher_version=args.bc_teacher,
            )
            before = teacher_agreement(policy, samples)
            bc_result = behavior_clone(policy, samples, epochs=args.dagger_epochs,
                                       temperature=args.bc_temperature,
                                       label_smoothing=args.bc_label_smoothing)
            after = teacher_agreement(policy, samples)
            # v1 是 float、v2 是 dict：report 里统一存「单值」，明细另存。
            after_scalar = _agreement_scalar(after)
            entry = {
                "round": round_no,
                "samples": len(samples),
                "agreement": after_scalar,
                "agreementBefore": _agreement_scalar(before),
                "agreementExact": _agreement_detail(after, "exactMatch"),
                "headAgreement": _agreement_detail(after, "perHead"),
                "perHeadAccuracy": bc_result.get("perHeadAccuracy"),
                "loss": bc_result["loss"][-1] if bc_result["loss"] else None,
                "evaluation": {"sampling": None, "greedy": None},
            }
            if eval_epochs > 0:
                sampling, greedy = eval_pair()
                entry["evaluation"] = {
                    "sampling": _bucket(sampling, bots[0].username),
                    "greedy": _bucket(greedy, bots[0].username),
                }
            dagger_report.append(entry)
            print(
                f"[dagger] 第 {round_no} 轮：samples={len(samples)} "
                f"agreement {_fmt_agreement(before)} → {_fmt_agreement(after)} "
                f"/ loss {entry['loss']} 评估={_fmt_eval(entry['evaluation'])}"
            )
            if after_scalar is not None and after_scalar >= float(args.dagger_target_agreement):
                print(f"[dagger] 一致率 {after_scalar:.3f} ≥ {args.dagger_target_agreement}，提前结束")
                break

    if args.eval_before_train and eval_epochs > 0:
        print("[train] 训练前评估（模仿阶段结束 / PPO 前）…")
        eval_before, eval_greedy_before = eval_pair()
        before_overall = summarize_records(eval_before, bots[0].username)["overall"]
        print(
            f"[train] 训练前评估：{before_overall['wins']}/{before_overall['episodes']} 胜"
            f"（{before_overall['winRate']:.0%}）均 reward {before_overall['meanReward']:.2f}"
        )

    # 回合（epoch）→ 计划（plan）→ 轮次（count）：每一轮次是一局对局，结束即更新一次参数。
    history = []
    for epoch in range(1, epochs + 1):
        print(f"[train] === 回合 {epoch}/{epochs} ===")
        for mode, count in plan:
            env.opponent_mode = mode
            print(f"[train] --- 阶段 {mode}：{count} 轮次 ---")
            for record in trainer.train(count):
                record["opponentMode"] = mode
                record["epoch"] = epoch
                record["episode"] = len(history) + 1
                history.append(record)

    eval_results = run_eval_pass(greedy=not args.eval_sampling)
    if args.eval_sampling:
        # 采样口径之外再补一组贪心：用来判断「是不确定性拖累，还是策略本身不行」。
        eval_greedy = run_eval_pass(greedy=True)
    else:
        eval_greedy = eval_results
    env.close()

    # 产物目录可能不存在（artifacts/ 已被 .gitignore 忽略）
    for path in (args.save, args.report, args.review_dir):
        parent = os.path.dirname(os.path.abspath(path)) if os.path.splitext(path)[1] else path
        if parent:
            os.makedirs(parent, exist_ok=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    policy.save(args.save, extra={"episodes": len(history), "seed": args.seed})
    train_summary = summarize_records(history, bots[0].username)
    eval_summary = summarize_records(eval_results, bots[0].username)
    eval_greedy_summary = summarize_records(eval_greedy, bots[0].username)
    report = {
        "runDir": run_dir,
        "kind": "train",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "startedAtEpoch": round(started, 3),
        "durationSeconds": round(time.time() - started, 1),
        "config": {
            "users": args.users,
            "ports": args.ports,
            "plan": [{"opponentMode": mode, "episodes": count} for mode, count in plan],
            "epochs": epochs,
            # B1 口径：evalRounds = 每阶段评估回合数（Dashboard 以 --eval-epochs 传这个值），
            # evalEpisodes = 实际评估对局数 = evalRounds × Σ(阶段轮次)（派生，只读）；
            # evalEpochs 保留给旧读者。
            "evalRounds": eval_epochs,
            "evalEpisodes": eval_episodes,
            "evalEpochs": eval_epochs,
            "initModel": args.init_model or None,
            "initPolicy": init_path or None,
            "modelId": args.model_id or None,
            "modelName": args.model_name or None,
            "persistTo": args.persist_to or None,
            "maxSteps": args.max_steps,
            "algo": args.algo,
            # v1 = 单头 11 宏动作；v2 = 9 头 MultiDiscrete（factorized）
            "actionSpace": args.action_space,
            "ppoEpochs": max(1, int(args.ppo_epochs)),
            "clip": args.clip,
            "gaeLambda": args.gae_lambda,
            "valueCoef": args.value_coef,
            "normalizeReturns": bool(args.normalize_returns),
            # P2：胜负常数与 dense 封顶（供复盘/对照）
            "victory": (float(args.victory) if args.victory is not None
                        else _default_victory()),
            "lrDecay": bool(args.lr_decay),
            "entropyFloor": args.entropy_floor,
            "entropyCoefMax": args.entropy_coef_max,
            # P2.2：目标熵控制器与价值裁剪（0 = 关闭，退回 P2.1 行为）
            "targetEntropy": args.target_entropy,
            "valueClip": args.value_clip,
            # P2.3：策略饱和治理与 DAgger
            "logitTemp": float(policy.temperature),
            "resetPolicyHead": bool(args.reset_policy_head),
            "weightDecay": float(args.weight_decay),
            "daggerRounds": int(args.dagger_rounds),
            "daggerEpisodes": int(args.dagger_episodes),
            "daggerEpochs": int(args.dagger_epochs),
            "daggerTargetAgreement": float(args.dagger_target_agreement),
            "minibatch": int(args.minibatch),
            "maxGradNorm": args.max_grad_norm,
            "shapingCap": _default_shaping_cap(),
            "entropyCoef": args.entropy_coef,
            "evalSeed": args.eval_seed,
            "evalSampling": bool(args.eval_sampling),
            "evalBeforeTrain": bool(args.eval_before_train),
            "bcEpisodes": int(args.bc_episodes),
            "arenaJitter": [jitter_parts[0], jitter_parts[1]],
            # P1.7：对手池区间（结构化，供 dashboard/复盘直接展示，不必改 dashboard）。
            "opponentPool": {
                "modes": {
                    mode: {key: [low, high] for key, (low, high) in spec.items()}
                    for mode, spec in OPPONENT_PROFILES.items()
                },
                "reachOffset": list(OPPONENT_REACH_OFFSET),
            },
            "bc": {
                "episodes": bc_info["episodes"],
                "epochs": bc_info["epochs"],
                "samples": bc_info["samples"],
                "accuracy": bc_info["accuracy"],
                "agreementBefore": bc_info.get("agreementBefore"),
                "agreementAfter": bc_info.get("agreementAfter"),
                # G-710：v2 蒸馏的逐头明细（v1 为 null，旧读者不受影响）
                "agreementBeforeExact": bc_info.get("agreementBeforeExact"),
                "agreementAfterExact": bc_info.get("agreementAfterExact"),
                "headAgreementBefore": bc_info.get("headAgreementBefore"),
                "headAgreementAfter": bc_info.get("headAgreementAfter"),
                "perHeadAccuracy": bc_info.get("perHeadAccuracy"),
                "heads": bc_info.get("heads"),
                "totalWeight": bc_info.get("totalWeight"),
                "loss": bc_info.get("loss"),
                "maxSteps": int(args.bc_max_steps),
                # 老师版本与软目标参数（T>1 / eps>0 时 BC 不再把 softmax 压死）
                "teacher": bc_info.get("teacher", args.bc_teacher),
                "temperature": bc_info.get("temperature", float(args.bc_temperature)),
                "labelSmoothing": bc_info.get("labelSmoothing", float(args.bc_label_smoothing)),
                "teacherEntropy": bc_info.get("teacherEntropy"),
            },
            "gamma": args.gamma,
            "lr": args.lr,
            "hidden": args.hidden,
            "beta": args.beta,
            "seed": args.seed,
            "center": list(center),
            "spacing": args.spacing,
            "profiles": [
                {
                    "username": bot.username,
                    "cps": bot.cps,
                    "cpsJitter": bot.cps_jitter,
                    "reactionMs": bot.reaction_ms,
                    "reactionJitterMs": bot.reaction_jitter_ms,
                    "reach": bot.reach,
                    "model": bot.model,
                }
                for bot in bots
            ],
            "reviews": args.review_dir,
        },
        "history": history,
        # P2.2 块 B：BC 贡献单独成块（顶层 + config.bc 同源）。
        # agreementBefore → agreementAfter 是「老师一致率」的前后差，用来判断
        # 训练前的胜率到底有多少是 BC 打下的底子。
        "bc": {
            "episodes": bc_info["episodes"],
            "samples": bc_info["samples"],
            "epochs": bc_info["epochs"],
            "loss": bc_info.get("loss"),
            "agreementBefore": bc_info.get("agreementBefore"),
            "agreementAfter": bc_info.get("agreementAfter"),
            "accuracy": bc_info["accuracy"],
            "skipped": bc_info["episodes"] == 0 or bc_info["epochs"] == 0,
            # 块 B：老师版本与软目标参数（便于「同一批数据、不同 T」的对照）
            "teacher": bc_info.get("teacher", args.bc_teacher),
            "temperature": bc_info.get("temperature", float(args.bc_temperature)),
            "labelSmoothing": bc_info.get("labelSmoothing", float(args.bc_label_smoothing)),
            "teacherEntropy": bc_info.get("teacherEntropy"),
            # G-710：v2 蒸馏（十头）的逐头明细；v1 全部为 null。
            # agreementBefore/After 在 v2 下是「逐头平均」，exactMatch 是全体完全一致率。
            "agreementBeforeExact": bc_info.get("agreementBeforeExact"),
            "agreementAfterExact": bc_info.get("agreementAfterExact"),
            "headAgreementBefore": bc_info.get("headAgreementBefore"),
            "headAgreementAfter": bc_info.get("headAgreementAfter"),
            "perHeadAccuracy": bc_info.get("perHeadAccuracy"),
            "heads": bc_info.get("heads"),
            "totalWeight": bc_info.get("totalWeight"),
        },
        # P2.2 块 B：训练前评估（--eval-before-train 时非 null）。
        "evaluationBefore": eval_before,
        "evaluationGreedyBefore": eval_greedy_before,
        # P2.3 块 C：DAgger 每轮的样本量/一致率/评估（0 = 关闭时为空数组）。
        "dagger": dagger_report,
        # P1-1：分对手胜率。overall = 全部训练局；byOpponent 按 env.opponent_mode 分桶。
        # 滑动窗口（每 10 局）：均值会被个别长局拉偏，窗口胜率更适合判断趋势。
        "summary": {
            **train_summary,
            "windows": sliding_windows(history, bots[0].username, 10),
            # P2：PPO 诊断量均值（valueLoss 变大通常意味着价值头发散）。
            "ppo": _mean_ppo_metrics(history),
        },
        "evalSeed": args.eval_seed,
        "evaluation": eval_results,
        # 评估局的分桶（evaluation 本身是数组，分桶放在同级字段，避免破坏既有读取方）。
        "evaluationByOpponent": eval_summary["byOpponent"],
        "evaluationOverall": eval_summary["overall"],
        # P1.6：同一批 eval_seed 下的贪心评估（结构与 evaluation 完全一致）。
        "evaluationGreedy": eval_greedy,
        "evaluationGreedyByOpponent": eval_greedy_summary["byOpponent"],
        "evaluationGreedyOverall": eval_greedy_summary["overall"],
        "policyPath": args.save,
    }
    save_report(args.report, report)

    total_steps = sum(h["steps"] for h in history)
    mean_reward = sum(h["totalReward"] for h in history) / max(1, len(history))
    hits = sum((h.get("review") or {}).get("hits", 0) for h in history)
    whiffs = sum((h.get("review") or {}).get("whiffs", 0) for h in history)
    print(
        f"[train] 完成：{len(history)} episodes / {total_steps} steps / "
        f"mean reward {mean_reward:.3f} / 命中 {hits} 空挥 {whiffs} / 用时 {report['durationSeconds']}s"
    )
    by_opponent = report["summary"]["byOpponent"]
    if by_opponent:
        detail = "；".join(
            f"{mode} {bucket['wins']}/{bucket['episodes']}（{bucket['winRate']:.0%}）"
            for mode, bucket in by_opponent.items()
        )
        print(f"[train] 分对手胜率：{detail}")
    print(f"[train] 策略 {args.save}；报告 {args.report}；复盘 {args.review_dir}")
    if bc_info["episodes"] > 0:
        print(
            f"[train] BC 贡献：{bc_info['episodes']} 局 / 一致率 "
            f"{bc_info['agreementBefore']} → {bc_info['agreementAfter']}"
            f" / 样本 {bc_info['samples']}"
        )
    else:
        print("[train] BC：未启用（训练前策略为纯 PPO 起点）")
    if dagger_report:
        last = dagger_report[-1]
        print(
            f"[train] DAgger：{len(dagger_report) - 1} 轮 / 末轮一致率 {last.get('agreement')}"
            f" / 末轮评估 {_fmt_eval(last.get('evaluation'))}"
        )
    if eval_before:
        before = report["evaluationBefore"] and summarize_records(eval_before, bots[0].username)["overall"]
        after = summarize_records(eval_results, bots[0].username)["overall"]
        if before:
            print(
                f"[train] 训练前/后（采样）：{before['wins']}/{before['episodes']}"
                f" → {after['wins']}/{after['episodes']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
