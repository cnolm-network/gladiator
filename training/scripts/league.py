r"""联赛评估：多模型循环赛（在 ``mirror_eval`` 的单对镜像机制上扩展）。

为什么需要它：镜像评估只能回答「新模型是否强于旧模型」，一次一对、人工挑选对手。
当模型有 3 个以上（BC 分支 / DAgger 各轮 / 自博弈快照）时，真正想知道的是**全局序**：
谁排第一、某对差距是否显著、某个模型是「只会打弱对手」还是「真的更强」。

做法与 ``mirror_eval`` 完全一致的对局口径（同一套 env、同一套左右位交替），只是把
「一对」扩展成「所有两两组合」：

- 每对打 ``--episodes-per-pair`` 局，**奇偶局交换左右位**（偶数局 A 在 bots[0]）；
- 每对输出 A 视角的胜/负/平、胜率、Wilson 95% 区间；
- 用**精确二项检验**（p=0.5，双侧）判断「A 是否显著强于 B」——小样本不做正态近似；
- 全局按总胜率排名，并打印 markdown 表格（可直接贴进 TASKS/HANDOVER；带 ``actionSpace`` 列）；
- 结果写 ``artifacts/evals/<YYYYMMDD-HHMM>-league.json``。

**动作空间（G-703）**：v1 / v2 模型可以同场混打，按**各自**的 ``_meta.actionSpace`` 分发：

- 学习者的动作空间决定 ``GladiatorEnv(action_space=...)``（v1 → ``Discrete(11)``，
  v2 → ``MultiDiscrete``）；
- 对手的动作空间由 env 自己读 ``opponent_policy.is_v2`` 判定，走 ``decode_intent`` 那条路径，
  不需要额外参数；
- 取动作用 ``gladiator_train.model_io.policy_action_for_env``：v1 → ``int`` 宏动作，
  v2 → 10 个头下标（带 action mask）。

因此「v1 学习者 + v2 对手」「v2 学习者 + v1 对手」「v2 + v2」都能跑，表格里的
``actionSpace`` 列会标出每个模型属于哪一边——混打的胜率必须带着这个上下文看。

用法::

    .venv\Scripts\python.exe scripts\league.py \\
        --policies ..\dashboard\var\models\v2-v4-dagger3\policy.npz \\
                   ..\dashboard\var\models\v2-v4-dagger2\policy.npz \\
                   ..\dashboard\var\models\dagger-1118b\policy.npz \\
        --episodes-per-pair 4 --seed 1200 --max-steps 400 --no-spawn \\
        --users Bot_V0A Bot_V0B --ports 8770 8771

    # 只校验参数与模型（服务器没开时先跑这个）
    .venv\Scripts\python.exe scripts\league.py --policies A.npz B.npz C.npz --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_train.eval_suite import SIGNIFICANCE, SUITE_DEFAULT, wilson_interval  # noqa: E402
from gladiator_train.model_io import (  # noqa: E402
    ModelLoadError,
    ModelRecord,
    env_action_space,
    load_model,
    model_label,
    normalize_action_mask,
    policy_action_for_env,
)

try:  # 复用 mirror_eval 的口径（同一份二项检验 / rcon 解析），避免两处漂移
    from scripts.mirror_eval import binom_p_two_sided, parse_rcon
except ImportError:  # pragma: no cover - 直接 `python scripts/league.py` 时 sys.path[0] 是 scripts/
    from mirror_eval import binom_p_two_sided, parse_rcon


LEAGUE_DEFAULT = {
    "episodesPerPair": 4,
    "seed": SUITE_DEFAULT["seed"],
    "maxSteps": 400,
    "out": "artifacts/evals",
}

#: 对局结果的三种取值（``play_episode`` 的约定返回值）。
OUTCOMES = ("left", "right", "draw")


# ---------------------------------------------------------------------------
# 模型与命名
# ---------------------------------------------------------------------------


def unique_labels(paths: list[str]) -> list[str]:
    """给每个模型起一个**唯一**短名（矩阵/表格的键）。

    ``.../branches/selfplay-final/policy.npz`` 这类路径的 stem 全是 ``policy``，
    直接拿 stem 会撞车；因此 stem 太通用时改用父目录名。
    """
    labels: list[str] = [model_label(path) for path in paths]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        count = seen.get(label, 0) + 1
        seen[label] = count
        unique.append(label if count == 1 else f"{label}#{count}")
    return unique


def load_policies(paths: list[str]) -> dict[str, "ModelRecord"]:
    """加载模型（v1/v2 都收），返回 ``label -> ModelRecord``。

    v2 的拒绝在 G-703 去掉：动作空间由每个模型自己的 ``_meta.actionSpace`` 决定，
    学习者按它建 env，对手由 env 自己按 ``is_v2`` 分流（见模块 docstring）。
    加载失败一律 ``SystemExit``（带路径与原因），不要跑到一半才发现有个坏文件。
    """
    labels = unique_labels(paths)
    loaded: dict[str, ModelRecord] = {}
    for label, path in zip(labels, paths):
        try:
            loaded[label] = load_model(path, label=label)
        except ModelLoadError as exc:
            raise SystemExit(str(exc)) from exc
    return loaded


# ---------------------------------------------------------------------------
# 赛程与对局
# ---------------------------------------------------------------------------


def build_schedule(labels: list[str], episodes_per_pair: int) -> list[dict]:
    """生成赛程：所有两两组合 × 每对局数，**奇偶局交换左右位**。

    返回 ``[{"a","b","episode","left","right"}, ...]``：``a`` 是「视角方」（矩阵的行），
    ``left`` 是这一局坐 bots[0]（左侧出生点）的模型——偶数局是 ``a``，奇数局是 ``b``。
    左右位交替是为了抵消出生点/朝向偏差，否则胜率里会混进「谁站在左边」。
    """
    pairs = list(itertools.combinations(labels, 2))
    schedule: list[dict] = []
    for a, b in pairs:
        for episode in range(int(episodes_per_pair)):
            left, right = (a, b) if episode % 2 == 0 else (b, a)
            schedule.append({"a": a, "b": b, "episode": episode,
                             "left": left, "right": right})
    return schedule


def run_league(paths: dict[str, str], episodes_per_pair: int, play_episode, seed: int,
               max_steps: int = LEAGUE_DEFAULT["maxSteps"],
               greedy: bool = False, verbose: bool = True,
               models: dict[str, "ModelRecord"] | None = None) -> dict:
    """跑完整个循环赛并汇总。

    ``play_episode(left_path, right_path, seed, max_steps, greedy)`` 返回
    ``"left"`` / ``"right"`` / ``"draw"``（**按座位**，不是按模型名）——
    这个注入点让单测可以不连服务端就能验证矩阵/排名/统计口径。

    ``models``（可选）只用于把 ``actionSpace`` / 头结构带进输出：**对局本身只看路径**，
    所以单测可以给假路径 + 不给 ``models``，统计口径完全一致。
    """
    labels = list(paths)
    if len(labels) < 2:
        raise ValueError("联赛至少需要 2 个模型")
    if models is not None and set(models) != set(labels):
        raise ValueError(
            f"models 的标签 {sorted(models)} 与 paths 的标签 {sorted(labels)} 不一致")
    schedule = build_schedule(labels, episodes_per_pair)
    rows: list[dict] = []
    for index, item in enumerate(schedule):
        episode_seed = int(seed) + index
        winner_side = play_episode(paths[item["left"]], paths[item["right"]],
                                   episode_seed, max_steps, greedy)
        if winner_side not in OUTCOMES:
            raise ValueError(f"play_episode 必须返回 {OUTCOMES} 之一，收到 {winner_side!r}")
        # 座位 → 模型名：play_episode 只回答「哪一边赢了」，映射成标签后才知道是谁赢。
        if winner_side == "left":
            winner_label = item["left"]
        elif winner_side == "right":
            winner_label = item["right"]
        else:
            winner_label = None
        if winner_label is None:
            outcome = "draw"
        elif winner_label == item["a"]:
            outcome = "a"
        else:
            outcome = "b"
        rows.append({"a": item["a"], "b": item["b"], "episode": item["episode"],
                     "left": item["left"], "right": item["right"], "seed": episode_seed,
                     "winner": winner_label, "outcome": outcome})
        if verbose:
            print(f"[league] {item['a']} vs {item['b']} 第 {item['episode'] + 1} 局 "
                  f"(left={item['left']}) seed={episode_seed} -> {outcome}", flush=True)
    summary = summarize_rows(labels, rows)
    summary.update({
        "episodesPerPair": int(episodes_per_pair),
        "seed": int(seed),
        "maxSteps": int(max_steps),
        "greedy": bool(greedy),
        # G-703：动作空间进输出，混打的胜率必须带着这个上下文看。
        "actionSpaces": ({label: models[label].action_space for label in labels}
                         if models is not None else {}),
        "policies": [
            ({"label": label, "path": models[label].path,
              "actionSpace": models[label].action_space,
              "headSizes": list(models[label].head_sizes),
              "obsDim": models[label].obs_dim}
             if models is not None and label in models
             else {"label": label, "path": paths[label]})
            for label in labels
        ],
        "rows": rows,
    })
    return summary


def summarize_rows(labels: list[str], rows: list[dict]) -> dict:
    """把逐局记录汇总成：胜负矩阵（双向对称）+ 每对统计 + 排名。"""
    matrix: dict[str, dict[str, dict]] = {
        a: {b: {"wins": 0, "losses": 0, "draws": 0, "episodes": 0, "winRate": 0.0}
            for b in labels if b != a}
        for a in labels
    }
    for row in rows:
        a, b, outcome = row["a"], row["b"], row["outcome"]
        cell, mirror = matrix[a][b], matrix[b][a]
        cell["episodes"] += 1
        mirror["episodes"] += 1
        if outcome == "a":
            cell["wins"] += 1
            mirror["losses"] += 1
        elif outcome == "b":
            cell["losses"] += 1
            mirror["wins"] += 1
        else:
            cell["draws"] += 1
            mirror["draws"] += 1
    for a in labels:
        for b in labels:
            if a == b:
                continue
            cell = matrix[a][b]
            cell["winRate"] = (round(cell["wins"] / cell["episodes"], 4)
                               if cell["episodes"] else 0.0)

    pairs: list[dict] = []
    for a, b in itertools.combinations(labels, 2):
        cell = matrix[a][b]
        wins, losses, draws = cell["wins"], cell["losses"], cell["draws"]
        episodes = cell["episodes"]
        # 精确二项检验只看「有胜负」的局：平局不提供 A/B 强弱的信息。
        p = binom_p_two_sided(wins, losses)
        significant = p < SIGNIFICANCE and wins != losses
        pairs.append({
            "a": a, "b": b,
            "aWins": wins, "bWins": losses, "draws": draws, "episodes": episodes,
            "aWinRate": cell["winRate"],
            "aWinRateCI": list(wilson_interval(wins, episodes)),
            "p": round(float(p), 6),
            "significant": bool(significant),
            "stronger": a if (significant and wins > losses)
                        else (b if (significant and losses > wins) else None),
        })

    ranking: list[dict] = []
    for label in labels:
        others = [other for other in labels if other != label]
        wins = sum(matrix[label][other]["wins"] for other in others)
        losses = sum(matrix[label][other]["losses"] for other in others)
        draws = sum(matrix[label][other]["draws"] for other in others)
        episodes = wins + losses + draws
        ranking.append({
            "label": label, "wins": wins, "losses": losses, "draws": draws,
            "episodes": episodes,
            "winRate": round(wins / episodes, 4) if episodes else 0.0,
            "winRateCI": list(wilson_interval(wins, episodes)),
        })
    # 胜率优先，其次胜场（同胜率时局数多的更可信），最后按名字稳定排序。
    ranking.sort(key=lambda item: (-item["winRate"], -item["wins"], item["label"]))
    for position, item in enumerate(ranking, 1):
        item["rank"] = position
    return {"matrix": matrix, "pairs": pairs, "ranking": ranking}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_ci(interval) -> str:
    return f"{_fmt_pct(interval[0])}–{_fmt_pct(interval[1])}"


def _space_of(summary: dict, label: str) -> str:
    """``label`` 的动作空间；没带（单测/旧 JSON）时返回 ``""``，表格里就不显示。"""
    return str((summary.get("actionSpaces") or {}).get(label) or "")


def _tagged(summary: dict, label: str) -> str:
    """``dagger3 [v2]`` 这种带动作空间的标签（矩阵/对阵表用）。"""
    space = _space_of(summary, label)
    return f"{label} [{space}]" if space else label


def format_markdown(summary: dict) -> str:
    """把汇总渲染成 markdown 表格（排名 / 对阵 / 胜负矩阵）。

    G-703：三张表都带动作空间——混打（v1/v2 同场）时，「谁更强」必须和「各自什么动作空间」
    一起看，否则同一个胜率会被读成「参数量差异」或「动作空间差异」。
    """
    labels = [item["label"] for item in summary["ranking"]]
    lines: list[str] = []
    lines.append(f"## 联赛结果（每对 {summary['episodesPerPair']} 局，seed={summary['seed']}）")
    lines.append("")
    lines.append("### 排名")
    lines.append("")
    lines.append("| # | 策略 | 动作空间 | 胜 | 负 | 平 | 胜率 | Wilson 95% |")
    lines.append("|---:|------|:--:|---:|---:|---:|---:|:-----------|")
    for item in summary["ranking"]:
        space = _space_of(summary, item["label"]) or "—"
        lines.append(
            f"| {item['rank']} | {item['label']} | {space} | {item['wins']} | "
            f"{item['losses']} | {item['draws']} | {_fmt_pct(item['winRate'])} | "
            f"{_fmt_ci(item['winRateCI'])} |")
    lines.append("")
    lines.append("### 对阵（A 视角，精确二项检验 p=0.5）")
    lines.append("")
    lines.append("| A | B | A 胜 | B 胜 | 平 | A 胜率 | 95% CI | p | 结论 |")
    lines.append("|:--|:--|---:|---:|---:|---:|:-------|---:|:-----|")
    for pair in summary["pairs"]:
        verdict = (f"{pair['stronger']} 显著更强" if pair["stronger"] else "无显著差异")
        lines.append(
            f"| {_tagged(summary, pair['a'])} | {_tagged(summary, pair['b'])} | "
            f"{pair['aWins']} | {pair['bWins']} | "
            f"{pair['draws']} | {_fmt_pct(pair['aWinRate'])} | "
            f"{_fmt_ci(pair['aWinRateCI'])} | {pair['p']:.4f} | {verdict} |")
    lines.append("")
    lines.append("### 胜负矩阵（行视角：胜-负-平）")
    lines.append("")
    header = "| A \\ B | " + " | ".join(_tagged(summary, label) for label in labels) + " |"
    lines.append(header)
    lines.append("|:------|" + "|".join([":--:" for _ in labels]) + "|")
    matrix = summary["matrix"]
    for row in labels:
        cells = []
        for column in labels:
            if row == column:
                cells.append("—")
            else:
                cell = matrix[row][column]
                cells.append(f"{cell['wins']}-{cell['losses']}-{cell['draws']}")
        lines.append(f"| **{_tagged(summary, row)}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(summary: dict, out_dir: str | Path, when: time.struct_time | None = None) -> Path:
    """写 ``<out_dir>/<YYYYMMDD-HHMM>-league.json``，返回路径。"""
    stamp = time.strftime("%Y%m%d-%H%M", when or time.localtime())
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stamp}-league.json"
    payload = dict(summary)
    payload.setdefault("generatedAt", time.strftime("%Y-%m-%dT%H:%M:%S%z", when or time.localtime()))
    payload.setdefault("kind", "league")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# 真机 runner
# ---------------------------------------------------------------------------


def make_env_play_episode(args) -> callable:
    """构造真机对局函数（口径与 ``mirror_eval`` 一致：左位当学习者，右位当对手策略）。

    G-703：左右两边的动作空间**各按自己的模型**来——
    - 学习者（``bots[0]``）：``action_space=`` 用左模型的（v1 → Discrete，v2 → MultiDiscrete），
      取动作走 :func:`policy_action_for_env`，v2 时带上 ``info['actionMask']``；
    - 对手（``bots[1]``）：交给 ``GladiatorEnv(opponent_policy=...)``，env 自己按
      ``is_v2`` 决定走 ``decode_intent`` 还是 v1 宏动作。

    因此「v1 学习者 + v2 对手」「v2 学习者 + v1 对手」「v2 + v2」都能跑，无需额外开关。
    """
    bots = [
        BotConfig(args.users[0], f"ws://{args.host}:{args.ports[0]}"),
        BotConfig(args.users[1], f"ws://{args.host}:{args.ports[1]}"),
    ]
    rcon = parse_rcon(args.rcon)
    left_name = bots[0].username
    #: 每个模型只加载一次：一次联赛会重复用到同一个 npz（N 对 × 局数），
    #: 重复 Policy.load 既慢又会把补零后的权重再拷一遍。
    cache: dict[str, ModelRecord] = {}

    def _model(path: str) -> "ModelRecord":
        record = cache.get(path)
        if record is None:
            record = load_model(path)
            cache[path] = record
        return record

    def play_episode(left_path: str, right_path: str, seed: int, max_steps: int,
                     greedy: bool = False) -> str:
        import numpy as np

        left = _model(left_path)
        right = _model(right_path)
        env = GladiatorEnv(
            bots,
            ArenaConfig(center=(0.5, 4.0, 0.5), spacing=6.0),
            single_agent=True,
            max_steps=max_steps,
            auto_spawn=not args.no_spawn,
            rcon=rcon,
            opponent_policy=right.policy,
            opponent_epsilon=0.0,
            action_space=env_action_space(left),
        )
        rng = np.random.default_rng(seed)
        try:
            observation, info = env.reset(seed=seed)
            mask = normalize_action_mask(left.policy, (info or {}).get("actionMask"))
            done = False
            steps = 0
            final_info = info
            while not done and steps < max_steps:
                action = policy_action_for_env(left.policy, observation, mask, rng,
                                               greedy=greedy)
                observation, _reward, terminated, truncated, final_info = env.step(action)
                mask = normalize_action_mask(left.policy, (final_info or {}).get("actionMask"))
                done = bool(terminated or truncated)
                steps += 1
            winner = (final_info or {}).get("winner")
        finally:
            env.close()
        if winner is None:
            return "draw"
        return "left" if winner == left_name else "right"

    return play_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="联赛评估：多模型循环赛（每对交替左右位）")
    parser.add_argument("--policies", nargs="+", required=True,
                        help="参赛模型 .npz（≥2 个，全部两两对局）")
    parser.add_argument("--episodes-per-pair", type=int, default=LEAGUE_DEFAULT["episodesPerPair"],
                        help="每对局数（奇偶局交换左右位；默认 4，建议偶数）")
    parser.add_argument("--seed", type=int, default=LEAGUE_DEFAULT["seed"])
    parser.add_argument("--max-steps", type=int, default=LEAGUE_DEFAULT["maxSteps"])
    parser.add_argument("--users", nargs=2, default=["Bot_V0A", "Bot_V0B"])
    parser.add_argument("--ports", nargs=2, type=int, default=[8770, 8771])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mc-port", type=int, default=25566)
    parser.add_argument("--rcon", default="127.0.0.1:25575:devpass")
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="两边都用 argmax 动作（去掉随机性）")
    parser.add_argument("--out", default=LEAGUE_DEFAULT["out"],
                        help="产物目录（默认 artifacts/evals，即 training/artifacts/evals）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做参数解析 + 模型加载 + 赛程校验，不开对局（服务器没开时用）")
    return parser


def dry_run_report(loaded: dict[str, "ModelRecord"], episodes_per_pair: int,
                   seed: int) -> str:
    """``--dry-run`` 的输出：模型表 + 赛程条数（**不连服务器**）。"""
    labels = list(loaded)
    schedule = build_schedule(labels, episodes_per_pair)
    lines = [f"[league] dry-run：{len(labels)} 个模型，"
             f"{len(loaded) * (len(loaded) - 1) // 2} 对 × {episodes_per_pair} 局 "
             f"= {len(schedule)} 局（seed={seed}）"]
    lines.append("")
    lines.append("| 策略 | 动作空间 | 头结构 | obsDim | 路径 |")
    lines.append("|:--|:--:|:--|---:|:--|")
    for label in labels:
        record = loaded[label]
        sizes = "/".join(str(size) for size in record.head_sizes)
        lines.append(f"| {label} | {record.action_space} | {sizes} | {record.obs_label()} "
                     f"| {record.path} |")
    lines.append("")
    fixed = sum(1 for index, item in enumerate(schedule) if item["episode"] % 2 == 0)
    lines.append(f"赛程校验：左右位交替（偶局 A 在左）{fixed}/{len(schedule)}，"
                 f"所有对局均为「学习者 + 对手策略」单智能体模式")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.policies) < 2:
        raise SystemExit("--policies 至少给 2 个模型")
    if args.episodes_per_pair < 1:
        raise SystemExit("--episodes-per-pair 至少为 1")
    # 先加载（v1/v2 都收，坏文件在这里就报），再开局：不要跑到一半才发现模型不对。
    loaded = load_policies(args.policies)
    paths = {label: record.path for label, record in loaded.items()}
    print(f"[league] {len(paths)} 个模型，每对 {args.episodes_per_pair} 局："
          + ", ".join(f"{label}({loaded[label].action_space})" for label in paths), flush=True)
    if args.dry_run:
        print(dry_run_report(loaded, args.episodes_per_pair, args.seed))
        print("[league] dry-run 结束（未开对局）")
        return 0

    summary = run_league(paths, args.episodes_per_pair, make_env_play_episode(args),
                         seed=args.seed, max_steps=args.max_steps, greedy=args.greedy,
                         models=loaded)
    markdown = format_markdown(summary)
    print()
    print(markdown)
    target = write_summary(summary, args.out)
    print()
    print(f"[league] 结果写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
