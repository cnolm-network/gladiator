"""量「模型与真人录像的一致率」：拿一份已导入的数据集，对每帧算模型 argmax 与人类动作是否一致。

为什么单独写：训练侧的 `teacher_agreement` 比的是「模型 vs 脚本老师」，而真人录像这边的「老师」是录制下来的
人。两者口径不同（一个是 v4 的十头输出、一个是推断出来的宏动作），所以这里同时支持两种动作空间：

- 数据集是 v2（`actions` 形状 `(N, 10)`）：逐头比，并给「十头全对」的比例；
- 数据集是 v1（`actions` 形状 `(N,)`）：比宏动作下标。

用法::

    cd training
    .venv\\Scripts\\python.exe scripts\\human_agreement.py \\
        --dataset artifacts/human-awr-1/dataset.npz \\
        --policy ../dashboard/var/models/dagger-1118b/policy.npz

输出（同时写 `--out` 指定的 JSON，默认与数据集同目录的 `agreement.json`）：
`overall`（逐样本一致率）、`exact`（v2 十头全对率）、`perHead`（v2 逐头一致率）、
`perAction`（人类每个动作被模型猜中的比例）、`confusions`（人类→模型最常见的错法）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "training"))

from gladiator_train.model_io import load_model  # noqa: E402
from gladiator_train.policy import Policy  # noqa: E402


def model_logits(policy: Policy, obs: np.ndarray) -> np.ndarray:
    """v1 返回 `(N,)` 的 argmax；v2 返回 `(N, 10)` 的逐头 argmax。

    注意 `Policy.forward/logits_all` 只吃**单帧**向量（内部会 `reshape(obs_dim)`），
    所以这里显式按行循环，不要整批传进去。
    """
    if not policy.is_v2:
        return np.asarray(
            [int(np.argmax(policy.probs(obs[index]))) for index in range(obs.shape[0])],
            dtype=np.int64,
        )
    rows = [policy.logits_all(obs[index]) for index in range(obs.shape[0])]
    return np.asarray([[int(np.argmax(head)) for head in row] for row in rows], dtype=np.int64)


def agreement(actions: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    """`actions` 与 `predicted` 形状一致（`(N,)` 或 `(N, heads)`），返回结构化统计。"""
    actions = np.asarray(actions)
    predicted = np.asarray(predicted)
    if actions.shape != predicted.shape:
        raise ValueError("动作形状不一致：%s vs %s" % (actions.shape, predicted.shape))

    if actions.ndim == 1:
        matched = actions == predicted
        pairs: Dict[str, int] = {}
        for truth, guess in zip(actions.tolist(), predicted.tolist()):
            if truth == guess:
                continue
            key = "%d->%d" % (truth, guess)
            pairs[key] = pairs.get(key, 0) + 1
        return {
            "overall": float(matched.mean()) if matched.size else 0.0,
            "samples": int(matched.size),
            "exact": float(matched.mean()) if matched.size else 0.0,
            "perHead": [],
            "perAction": _per_action(actions, matched),
            "confusions": _top_pairs(pairs),
        }

    per_head = (actions == predicted).mean(axis=0)
    exact = (actions == predicted).all(axis=1)
    confusions: Dict[str, int] = {}
    for index, head in enumerate(range(actions.shape[1])):
        del index
        for truth, guess in zip(actions[:, head].tolist(), predicted[:, head].tolist()):
            if truth == guess:
                continue
            key = "head%d:%d->%d" % (head, truth, guess)
            confusions[key] = confusions.get(key, 0) + 1
    return {
        "overall": float(per_head.mean()),
        "samples": int(actions.shape[0]),
        "exact": float(exact.mean()),
        "perHead": [round(float(value), 4) for value in per_head],
        "perAction": _per_action(actions[:, 0], actions[:, 0] == predicted[:, 0]),
        "confusions": _top_pairs(confusions),
    }


def _per_action(truth: np.ndarray, matched: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in sorted(set(truth.tolist())):
        mask = truth == value
        total = int(mask.sum())
        rows.append({
            "action": int(value),
            "samples": total,
            "hit": round(float(matched[mask].mean()), 4) if total else 0.0,
        })
    return rows


def _top_pairs(counter: Dict[str, int], limit: int = 8) -> List[Dict[str, Any]]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"pair": key, "count": count} for key, count in ordered[:limit]]


def evaluate(dataset: str, policy_path: str, out: Optional[str] = None) -> Dict[str, Any]:
    with np.load(dataset, allow_pickle=True) as data:
        obs = np.asarray(data["obs"], dtype=np.float64)
        actions = np.asarray(data["actions"])
        meta = json.loads(str(data["meta"].item())) if "meta" in data else {}

    model = load_model(policy_path, obs_dim=obs.shape[1])
    predicted = model_logits(model.policy, obs)
    report = agreement(actions, predicted)
    report.update({
        "dataset": os.path.abspath(dataset),
        "policy": os.path.abspath(policy_path),
        "policyActionSpace": model.action_space,
        "datasetActionSpace": meta.get("actionSpace") or ("v2" if actions.ndim == 2 else "v1"),
        "obsDim": int(obs.shape[1]),
        "nativeObsDim": int(model.native_obs_dim),
    })
    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="量模型与真人录像的一致率")
    parser.add_argument("--dataset", required=True, help="`import_human_match.py` 产出的 dataset.npz")
    parser.add_argument("--policy", required=True, help="要评估的模型（.npz）")
    parser.add_argument("--out", help="报告 JSON 路径；默认写数据集同目录的 agreement.json")
    args = parser.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dataset)), "agreement.json")
    report = evaluate(args.dataset, args.policy, out)
    print(json.dumps({key: value for key, value in report.items() if key != "confusions"},
                     ensure_ascii=False))
    print("[agreement] 最常错的 %d 种：" % len(report["confusions"]))
    for row in report["confusions"]:
        print("[agreement]   %s × %d" % (row["pair"], row["count"]))
    print("[agreement] 报告写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
