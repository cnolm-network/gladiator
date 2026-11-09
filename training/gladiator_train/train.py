"""训练入口：真实服务端上的 REINFORCE + 课程学习 + 每局复盘。

用法（2 个 Bot、通道 B 端口 8766/8767）：

    python -m gladiator_train.train --users Bot_T1 Bot_T2 --ports 8766 8767 \
        --episodes 8 --max-steps 200 --auto-spawn

课程（对手从静止到会反击，逐阶段放开）：

    python -m gladiator_train.train --stages idle:3,wander:3,random:3,scripted:3

每个 Bot 的手感配置（G-507）默认 15±2 CPS / 220±40ms / 3.99 格：

    ... --cps 17 --cps-jitter 3 --reaction-ms 180 --reach 4.2 --model aggressive

前置：Paper 已启动并加载 Gladiator；每个 Bot 一个 `bot-runtime` 进程；
RCON 已开启（控制面：摆位/回血/清背包）。详见 docs/training-env-api.md。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

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
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.01, help="熵正则系数")
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
    p.add_argument("--reach", type=float, default=3.99)
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
    p.add_argument("--opp-reach", type=float, default=3.99)
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


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        from gladiator_env.reward_presets import load_preset
        from gladiator_env.spaces import NUM_ACTIONS, OBS_DIM
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可读提示
        print(f"无法导入 gladiator_env：{exc}（请在 training/ 下运行）", file=sys.stderr)
        return 2

    # 初始策略在创建任何环境之前校验：无效直接 exit 2，不产生目录/进程。
    init_path = args.init_policy or args.init_model
    loaded, init_error = validate_init_policy(
        init_path, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=args.hidden)
    if init_error:
        print(f"[train] {init_error}", file=sys.stderr)
        return 2

    bots = [BotConfig(
        username=user,
        env_url=f"ws://{args.env_host}:{port}",
        cps=args.cps,
        cps_jitter=args.cps_jitter,
        reaction_ms=args.reaction_ms,
        reaction_jitter_ms=args.reaction_jitter_ms,
        reach=args.reach,
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
        bot.reach = args.opp_reach
        bot.model = args.opp_model
        if args.opp_assist:
            bot.wtap = True
            bot.auto_rod = True
            bot.auto_block = True
            bot.aim_lock = True

    arena = ArenaConfig(center=center, spacing=args.spacing, prep_seconds=args.prep_seconds)
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
    )

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    policy = Policy(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=args.hidden,
        seed=args.seed,
        lr=args.lr,
        beta=args.beta,
    )
    if loaded is not None:
        # 只接管权重（不接管 optimizer/updates 状态），lr/beta 仍用本次参数。
        adopt_weights(policy, loaded)
        label = args.model_name or args.model_id or init_path
        print(f"[train] 已加载初始策略 {label}（obs={obs_dim} actions={n_actions} hidden={args.hidden}）")
    trainer = Trainer(env, policy, gamma=args.gamma, seed=args.seed, review_dir=args.review_dir)

    started = time.time()
    plan = stages or [(args.opponent_mode, args.episodes)]
    epochs = max(1, int(args.epochs))
    eval_epochs = args.eval_epochs if args.eval_epochs is not None else (
        args.eval_episodes if args.eval_episodes is not None else 1)
    eval_epochs = max(0, int(eval_epochs))
    per_epoch = sum(count for _mode, count in plan)
    print(
        f"[train] bots={args.users} ports={args.ports} plan={plan} "
        f"max_steps={args.max_steps} obs={env.observation_space.shape} actions={env.action_space.n}"
    )
    print(
        f"[train] 学习方 cps={bots[0].cps}±{bots[0].cps_jitter} "
        f"reaction={bots[0].reaction_ms}±{bots[0].reaction_jitter_ms}ms reach={bots[0].reach}"
    )
    print(
        f"[train] 训练回合={epochs} × 每回合 {per_epoch} 轮次 = {epochs * per_epoch} 局；"
        f"评估回合={eval_epochs}（共 {eval_epochs * per_epoch} 局，贪心不学习）"
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

    # 评估：同样按计划走，跑 eval_epochs 遍；贪心、不更新参数。
    eval_results = []
    for epoch in range(1, eval_epochs + 1):
        for mode, count in plan:
            env.opponent_mode = mode
            for record in trainer.evaluate(count):
                record["opponentMode"] = mode
                record["evalEpoch"] = epoch
                eval_results.append(record)
    env.close()

    # 产物目录可能不存在（artifacts/ 已被 .gitignore 忽略）
    for path in (args.save, args.report, args.review_dir):
        parent = os.path.dirname(os.path.abspath(path)) if os.path.splitext(path)[1] else path
        if parent:
            os.makedirs(parent, exist_ok=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    policy.save(args.save, extra={"episodes": len(history), "seed": args.seed})
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
            "evalEpochs": eval_epochs,
            "initModel": args.init_model or None,
            "initPolicy": init_path or None,
            "modelId": args.model_id or None,
            "modelName": args.model_name or None,
            "persistTo": args.persist_to or None,
            "maxSteps": args.max_steps,
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
        "evaluation": eval_results,
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
    print(f"[train] 策略 {args.save}；报告 {args.report}；复盘 {args.review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
