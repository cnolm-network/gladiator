"""权重合并 CLI：对齐 → 合并 → 评估门禁 → 蒸馏兜底。

典型用法::

    # 三支 merge 成一支（不跑门禁）
    python scripts/merge_policies.py \
        --inputs runs/a/policy.npz runs/b/policy.npz runs/c/policy.npz \
        --method rebasin-ties --trim 0.2 --out artifacts/merges/m1/policy.npz

    # 带门禁：合并模型 vs 基线，同一 seed 集配对，不通过直接非零退出
    python scripts/merge_policies.py --inputs a.npz b.npz \
        --calibration artifacts/bc/obs.npz \
        --gate-baseline ../dashboard/var/models/dagger-1118b/policy.npz \
        --gate-episodes 8 --no-spawn --users Bot_V50A Bot_V50B

    # 门禁不过时用集成蒸馏兜底
    ... --distill --distill-episodes 6

退出码：合并/门禁本身出错 ``2``；门禁不通过 ``1``（``--allow-regression`` 可放行）；
一切正常 ``0``。即使门禁不通过，``--distill`` 仍会写出兜底模型，便于人工复核。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_train.eval_suite import (  # noqa: E402
    SUITE_DEFAULT,
    SuiteConfig,
    format_interval,
    pair_rows,
    run_suite,
    summarize,
)
from gladiator_train.merge import (  # noqa: E402
    MERGE_METHODS,
    apply_permutation,
    distill,
    fisher_information,
    gate_verdict,
    merge,
    params_l2,
    params_of,
)
from gladiator_train.policy import Policy  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = TRAINING_DIR / "artifacts" / "merges"
OPPONENT_MODES = ("random", "wander", "idle", "scripted", "rusher", "aimbot")


# ---------------------------------------------------------------------------
# 参数与输入
# ---------------------------------------------------------------------------

def parse_weights(text: str | None) -> list[float] | None:
    """``"1,1,2"`` → ``[1.0, 1.0, 2.0]``；空串/None → ``None``（等权）。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    try:
        values = [float(item) for item in parts]
    except ValueError as exc:
        raise SystemExit(f"--weights 解析失败: {exc}") from exc
    if any(value < 0 for value in values):
        raise SystemExit("--weights 不能为负")
    if sum(values) <= 0:
        raise SystemExit("--weights 之和必须为正")
    return values


def parse_opponents(text: str) -> tuple:
    items = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not items:
        raise SystemExit("--gate-opponents 不能为空")
    unknown = [item for item in items if item not in OPPONENT_MODES]
    if unknown:
        raise SystemExit(
            f"未知对手模式: {', '.join(unknown)}（可选: {', '.join(OPPONENT_MODES)}）"
        )
    ordered, seen = [], set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def load_calibration(path: str | None) -> tuple[np.ndarray | None, str]:
    """读校准观测：``.npz``（键 ``obs``/``observations``）或 ``.jsonl``（每行 ``{"obs": [...]}``）。"""
    if not path:
        return None, "none"
    target = Path(path)
    if not target.is_file():
        raise SystemExit(f"校准集不存在: {target}")
    if target.suffix.lower() == ".npz":
        with np.load(target, allow_pickle=True) as data:
            for key in ("obs", "observations", "x"):
                if key in data:
                    matrix = np.array(data[key], dtype=np.float64)
                    break
            else:
                raise SystemExit(f"校准 npz 里没有 obs/observations 数组: {target}")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix, f"npz:{target.name}"
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"校准集第 {line_no} 行不是合法 JSON: {exc}") from exc
            obs = payload.get("obs") if isinstance(payload, dict) else payload
            if obs is None:
                raise SystemExit(f"校准集第 {line_no} 行缺少 obs 字段")
            rows.append(np.asarray(obs, dtype=np.float64).reshape(-1))
    if not rows:
        raise SystemExit(f"校准集为空: {target}")
    return np.stack(rows, axis=0), f"jsonl:{target.name}"


def synthetic_calibration(obs_dim: int, count: int = 256, seed: int = 1234) -> np.ndarray:
    """没有校准集时的确定性伪样本：正态 + 全 0 + 全 1（各 ``count`` 条）。

    正态样本负责区分方向，全 0 / 全 1 负责压住「只在边界区活得很好」的单元；
    固定 seed 保证同一组权重每次对齐结果一致（可复现）。
    """
    rng = np.random.default_rng(int(seed))
    normal = np.clip(rng.normal(0.0, 1.0, (count, obs_dim)), -5.0, 5.0)
    zeros = np.zeros((count, obs_dim), dtype=np.float64)
    ones = np.ones((count, obs_dim), dtype=np.float64)
    return np.vstack([normal, zeros, ones])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(paths: list[str]) -> tuple[list[Policy], int, list[dict]]:
    """加载全部输入，把 obs 维度小的模型右侧补零到最大维度（与 ``Policy.load`` 同语义）。"""
    raw: list[Policy] = []
    for path in paths:
        target = Path(path)
        if not target.is_file():
            raise SystemExit(f"输入模型不存在: {target}")
        try:
            raw.append(Policy.load(str(target)))
        except Exception as exc:  # noqa: BLE001 - npz 损坏应给出可读错误
            raise SystemExit(f"输入模型无法加载 {target}: {exc}") from exc
    hidden = {policy.hidden for policy in raw}
    actions = {policy.n_actions for policy in raw}
    if len(hidden) != 1 or len(actions) != 1:
        raise SystemExit(
            f"输入模型结构不一致：hidden={sorted(hidden)} n_actions={sorted(actions)}"
            "（合并要求同结构，否则无法逐元素对应）"
        )
    obs_dim = max(policy.obs_dim for policy in raw)
    padded = []
    for path, policy in zip(paths, raw):
        if policy.obs_dim < obs_dim:
            print(f"[merge] {Path(path).name}: obs_dim {policy.obs_dim} → {obs_dim}（W1 右侧补零）")
            policy = Policy.load(path, obs_dim=obs_dim)
        padded.append(policy)
    meta = [
        {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "obsDim": int(policy.obs_dim),
            "storedObsDim": int(getattr(policy, "stored_obs_dim", policy.obs_dim)),
            "updates": int(policy.updates),
        }
        for path, policy in zip(paths, padded)
    ]
    return padded, obs_dim, meta


def policy_from_params(params: dict, *, temperature: float = 1.0) -> Policy:
    """把参数表包成一个可 ``act``/``forward`` 的 ``Policy``（只用于评估与保存）。"""
    table = params_of(params)
    policy = Policy(
        obs_dim=int(np.asarray(table["W1"]).shape[1]),
        n_actions=int(np.asarray(table["W2"]).shape[0]),
        hidden=int(np.asarray(table["W1"]).shape[0]),
        temperature=float(temperature or 1.0),
    )
    for key in Policy.PARAM_NAMES:
        if key in table:
            policy.params[key] = np.array(table[key], dtype=np.float64, copy=True)
    return policy


# ---------------------------------------------------------------------------
# 评估门禁
# ---------------------------------------------------------------------------

def _eval_cli():
    """按文件路径加载 ``scripts/eval_suite.py``，复用它的 env 工厂（不复制粘贴）。"""
    path = Path(__file__).resolve().parent / "eval_suite.py"
    spec = importlib.util.spec_from_file_location("gladiator_eval_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gladiator_eval_cli", module)
    spec.loader.exec_module(module)
    return module


def run_gate(args, merged_policy: Policy, config: SuiteConfig) -> dict:
    """跑「合并模型 vs 基线」同 seed 配对，返回含 verdict 的完整结论。"""
    baseline_path = Path(args.gate_baseline)
    if not baseline_path.is_file():
        raise SystemExit(f"基线模型不存在: {baseline_path}")
    try:
        baseline = Policy.load(str(baseline_path), obs_dim=merged_policy.obs_dim)
    except Exception as exc:  # noqa: BLE001 - 维度不兼容/文件损坏都给出可读错误
        raise SystemExit(f"基线模型无法加载（obs_dim={merged_policy.obs_dim}）: {exc}") from exc
    if (baseline.n_actions, baseline.hidden) != (merged_policy.n_actions, merged_policy.hidden):
        raise SystemExit(
            f"基线与合并模型结构不一致：基线 n_actions={baseline.n_actions} "
            f"hidden={baseline.hidden}，合并 n_actions={merged_policy.n_actions} "
            f"hidden={merged_policy.hidden}"
        )
    cli = _eval_cli()
    # 复用 eval_suite 的 env 工厂：它只读 users/ports/host/rcon/no_spawn 这几个字段。
    env_args = argparse.Namespace(
        users=list(args.users), ports=list(args.ports), host=args.host,
        rcon=args.rcon, no_spawn=bool(args.no_spawn),
    )
    factory = cli.make_env_factory(env_args, config)
    print(f"[gate] 基线 {baseline_path.name} → 合并模型，"
          f"{len(config.opponents)} 对手 × {config.episodes} 局，seed={config.seed}")
    rows_baseline = run_suite(factory, baseline, config, logger=print)
    rows_merged = run_suite(factory, merged_policy, config, logger=print)
    summary_baseline = summarize(rows_baseline)
    summary_merged = summarize(rows_merged)
    comparison = pair_rows(rows_merged, rows_baseline)
    verdict = gate_verdict(summary_baseline, summary_merged, comparison["p"],
                           allow_regression=bool(args.allow_regression))

    print(f"\n[gate] 基线胜率 {verdict['baselineWinRate']:.1%}"
          f"（CI {format_interval(summary_baseline['overall']['winRateCI'])}）"
          f" vs 合并 {verdict['mergedWinRate']:.1%}"
          f"（CI {format_interval(summary_merged['overall']['winRateCI'])}）")
    print(f"[gate] b={comparison['b']}（合并独赢） c={comparison['c']}（基线独赢） "
          f"p={comparison['p']:.4f} → {verdict['status']}")
    print(f"[gate] {verdict['reason']}")
    return {
        "baseline": str(baseline_path.resolve()),
        "baselineSha256": sha256_file(baseline_path),
        "config": config.to_dict(),
        "rowsBaseline": rows_baseline,
        "rowsMerged": rows_merged,
        "summaryBaseline": summary_baseline,
        "summaryMerged": summary_merged,
        "comparison": comparison,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="权重合并 v2：REBASIN 对齐 + TIES/Fisher 合并 + 评估门禁 + 蒸馏兜底",
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="输入 .npz 列表（≥ 2 支）")
    parser.add_argument("--method", default="rebasin-ties", choices=list(MERGE_METHODS),
                        help="合并方法（默认 rebasin-ties：先对齐再 TIES）")
    parser.add_argument("--trim", type=float, default=0.2,
                        help="TIES 裁剪比例（0 = 退化为加权平均，默认 0.2）")
    parser.add_argument("--weights", default=None,
                        help="逗号分隔的权重（默认等权），如 1,1,2")
    parser.add_argument("--calibration", default=None,
                        help="校准观测 .npz（含 obs 数组）或 .jsonl（每行 {\"obs\": [...]}）")
    parser.add_argument("--out", default=None,
                        help="合并结果路径（默认 artifacts/merges/<时间>/merged.npz）")
    parser.add_argument("--summary", default=None,
                        help="summary.json 路径（默认与 --out 同目录）")
    parser.add_argument("--seed", type=int, default=1234,
                        help="伪样本/蒸馏的随机种子（默认 1234）")

    gate = parser.add_argument_group("评估门禁（合并模型 vs 基线，同 seed 配对）")
    gate.add_argument("--gate-baseline", default=None, help="基线 .npz；给出才跑门禁")
    gate.add_argument("--gate-episodes", type=int, default=SUITE_DEFAULT["episodes"],
                      help=f"每个对手的局数（默认 {SUITE_DEFAULT['episodes']}）")
    gate.add_argument("--gate-opponents", default=",".join(SUITE_DEFAULT["opponents"]),
                      help=f"逗号分隔对手（默认 {','.join(SUITE_DEFAULT['opponents'])}）")
    gate.add_argument("--gate-seed", type=int, default=SUITE_DEFAULT["seed"],
                      help=f"起始 seed（默认 {SUITE_DEFAULT['seed']}）")
    gate.add_argument("--gate-max-steps", type=int, default=SUITE_DEFAULT["maxSteps"],
                      help=f"每局最多步数（默认 {SUITE_DEFAULT['maxSteps']}）")
    gate.add_argument("--sampled", action="store_true",
                      help="按策略采样评估（默认贪心，可复现）")
    gate.add_argument("--no-spawn", action="store_true", help="不自动 spawn（Bot 已在跑）")
    gate.add_argument("--users", nargs=2, default=["Bot_V50A", "Bot_V50B"],
                      help="两个 Bot 用户名（第一个是被评估方）")
    gate.add_argument("--ports", nargs=2, type=int, default=[8770, 8771],
                      help="两个 Bot 的通道 B 端口")
    gate.add_argument("--host", default="127.0.0.1", help="通道 B 主机")
    gate.add_argument("--mc-port", type=int, default=25566, help="Minecraft 端口")
    gate.add_argument("--rcon", default="127.0.0.1:25575:devpass",
                      help="RCON host:port:password")
    gate.add_argument("--allow-regression", action="store_true",
                      help="门禁不通过也不非零退出（仍如实记录结论）")

    distill_group = parser.add_argument_group("蒸馏兜底")
    distill_group.add_argument("--distill", action="store_true",
                               help="门禁不通过时用集成当老师蒸馏出 merged.distill.npz")
    distill_group.add_argument("--distill-episodes", type=int, default=6,
                               help="蒸馏在样本上过的轮数（默认 6）")
    distill_group.add_argument("--distill-lr", type=float, default=5e-3,
                               help="蒸馏学习率（默认 5e-3）")
    distill_group.add_argument("--distill-batch", type=int, default=64,
                               help="蒸馏 batch（默认 64）")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.inputs) < 2:
        raise SystemExit("--inputs 至少两支（单支合并没有意义）")
    weights = parse_weights(args.weights)
    if weights is not None and len(weights) != len(args.inputs):
        raise SystemExit(f"--weights 数量 {len(weights)} 与 --inputs 数量 {len(args.inputs)} 不一致")

    policies, obs_dim, input_meta = load_inputs(list(args.inputs))
    params_list = [policy.params for policy in policies]
    calibration, calibration_source = load_calibration(args.calibration)
    if calibration is None:
        calibration = synthetic_calibration(obs_dim, seed=args.seed)
        print("[merge] 警告：未提供校准集，对齐质量可能下降"
              "（已用确定性伪样本：正态/全 0/全 1 各 256 条）", file=sys.stderr)
    if calibration.shape[1] != obs_dim:
        raise SystemExit(
            f"校准集维度 {calibration.shape[1]} 与模型 obs_dim {obs_dim} 不一致"
        )
    print(f"[merge] {len(policies)} 支输入，method={args.method}，"
          f"校准样本 {calibration.shape[0]} 条（{calibration_source}）")

    fisher_list = None
    if args.method == "rebasin-fisher":
        # 没有动作标签时用各模型自己的贪心动作当样本动作（近似 Fisher）。
        fisher_list = []
        for policy in policies:
            actions = [int(np.argmax(policy.probs(row))) for row in calibration]
            fisher_list.append(
                fisher_information(policy, list(zip(calibration, actions)), obs_dim=obs_dim)
            )
        print("[merge] rebasin-fisher：Fisher 用各模型在校准样本上的贪心动作估计")

    merged, info = merge(
        params_list, method=args.method, calibration_obs=calibration,
        weights=weights, trim=float(args.trim), fisher=fisher_list, return_info=True,
    )
    merged_policy = policy_from_params(merged, temperature=policies[0].temperature)

    stamp = time.strftime("%Y%m%d-%H%M")
    out_path = Path(args.out) if args.out else DEFAULT_OUT / stamp / "merged.npz"
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_policy.save(str(out_path), extra={
        "source": "merge_policies",
        "method": args.method,
        "trim": float(args.trim),
        "inputs": [item["sha256"] for item in input_meta],
        "calibrationSamples": int(calibration.shape[0]),
    })
    print(f"[merge] 已写出 {out_path}")

    perms = info.get("perms") or []
    perm_changed = sum(1 for perm in perms if list(perm) != list(range(len(perm))))
    l2_rows = []
    for index, (policy, meta) in enumerate(zip(policies, input_meta), start=1):
        perm = perms[index - 1] if index - 1 < len(perms) else list(range(policy.hidden))
        aligned = apply_permutation(policy.params, perm)
        l2_rows.append({
            "index": index,
            **meta,
            "permChanged": list(perm) != list(range(policy.hidden)),
            "l2ToMerged": params_l2(policy.params, merged),
            "l2ToMergedAligned": params_l2(aligned, merged),
        })
        print(f"[merge]   #{index} {Path(meta['path']).name}: "
              f"L2={l2_rows[-1]['l2ToMerged']:.4f}（对齐后 "
              f"{l2_rows[-1]['l2ToMergedAligned']:.4f}）")

    payload: dict = {
        "schema": "gladiator.merge/v1",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": args.method,
        "trim": float(args.trim),
        "weights": weights,
        "calibration": {
            "source": calibration_source,
            "samples": int(calibration.shape[0]),
            "obsDim": int(obs_dim),
        },
        "inputs": input_meta,
        "permCount": int(perm_changed),
        "permIdentity": int(len(perms) - perm_changed),
        "l2": l2_rows,
        "mergedPath": str(out_path),
        "mergedSha256": sha256_file(out_path),
    }

    exit_code = 0
    gate_result = None
    if args.gate_baseline:
        config = SuiteConfig(
            opponents=parse_opponents(args.gate_opponents),
            episodes=int(args.gate_episodes),
            seed=int(args.gate_seed),
            max_steps=int(args.gate_max_steps),
            greedy=not args.sampled,
        )
        if config.total_episodes < 24:
            print(f"[gate] 提示：总样本 {config.total_episodes} 局 < 24，按口径不下结论")
        gate_result = run_gate(args, merged_policy, config)
        payload["gate"] = gate_result
        if not gate_result["verdict"]["passed"]:
            exit_code = 1

    # 兜底看的是「门禁结论」而不是「是否阻断」：--allow-regression 只放行退出码，
    # 结论仍然是掉点，这种情况更应该产出蒸馏模型。
    failed_status = {"significant-worse", "regression"}
    need_distill = gate_result is None or gate_result["verdict"]["status"] in failed_status
    if args.distill and need_distill:
        print(f"[distill] 用 {len(policies)} 个老师的集成蒸馏 "
              f"{args.distill_episodes} 轮（{calibration.shape[0]} 条样本）")
        teacher_params = [policy.params for policy in policies]
        student, distill_info = distill(
            teacher_params, calibration, init_params=merged,
            epochs=int(args.distill_episodes), lr=float(args.distill_lr),
            batch_size=int(args.distill_batch), seed=int(args.seed), logger=print,
        )
        student_policy = policy_from_params(student, temperature=policies[0].temperature)
        distill_path = out_path.with_name(out_path.stem + ".distill" + out_path.suffix)
        student_policy.save(str(distill_path), extra={
            "source": "merge_policies.distill",
            "method": args.method,
            "teachers": [item["sha256"] for item in input_meta],
        })
        payload["distill"] = {"path": str(distill_path), **distill_info}
        print(f"[distill] 已写出 {distill_path}（KL {distill_info['klBefore']:.6f} → "
              f"{distill_info['klAfter']:.6f}）")

    summary_path = Path(args.summary) if args.summary else out_path.parent / "summary.json"
    if not summary_path.is_absolute():
        summary_path = (Path.cwd() / summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] 汇总写入 {summary_path}")

    if gate_result is not None and not gate_result["verdict"]["passed"]:
        print(f"[gate] 门禁不通过：{gate_result['verdict']['reason']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
