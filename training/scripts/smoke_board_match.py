"""人机对战端到端冒烟：起一场板子对局，等它打完，再核对录制与数据集。

这条脚本把 2026-11-28 手工做过的验收（"板子引擎 + Arena 录制 + 赛后导入"）固化成一条命令，
避免每次改完 dashboard/arena 都靠人工点网页验证。

用法（dashboard 与 Paper 都在跑、目标 Bot 已在服内）::

    cd training
    .venv\\Scripts\\python.exe scripts\\smoke_board_match.py \\
        --player Glad_E1 --bot Glad_E5 --env-port 8791 \\
        --rounds 2 --round-seconds 20 --brain dagger-1118b

判据（任一不满足即非 0 退出）：
  1. 对局在 `--timeout` 秒内到达 `finished`；
  2. `roundsDetail` 的条数等于请求的局数；
  3. 录制目录里 `arena-records/` 的份数 == 局数（Arena 每局导出一份）；
  4. `dataset.npz` 的 `meta.rewardSource == "arena-record"`；
  5. `rewards` 非全零，且 `rewards.sum < 0`（受伤为负）；
  6. `obs` 的第二维等于训练侧当前观测维度。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MATCHES = os.path.join(ROOT, "training", "artifacts", "matches")


def post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def match_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    """`/api/match` 既可能直接返回快照，也可能包在 `match` 里。"""
    if isinstance(payload.get("match"), dict):
        return payload["match"]
    return payload


def wait_finished(base: str, timeout: float, poll: float = 2.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        try:
            last = match_snapshot(get_json(base + "/api/match"))
        except (urllib.error.URLError, TimeoutError) as exc:  # dashboard 抖动不算失败
            last = {"status": "unreachable", "error": str(exc)}
        status = str(last.get("status") or "")
        if status in ("finished", "failed", "stopped"):
            return last
        time.sleep(poll)
    return last


def latest_match_dir(matches: str, since: float) -> Optional[str]:
    """挑出本场开始之后创建的最新目录（按目录名的时间戳排序）。"""
    if not os.path.isdir(matches):
        return None
    candidates: List[str] = []
    for name in os.listdir(matches):
        path = os.path.join(matches, name)
        if not os.path.isdir(path):
            continue
        if os.path.getmtime(path) + 1.0 < since:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def record_files(match_dir: str) -> List[str]:
    folder = os.path.join(match_dir, "arena-records")
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, name) for name in os.listdir(folder))


def dataset_report(match_dir: str) -> Dict[str, Any]:
    """读 `dataset.npz`，只取判据需要的几个标量。"""
    import numpy as np

    path = os.path.join(match_dir, "dataset.npz")
    if not os.path.exists(path):
        return {"ok": False, "error": "dataset.npz 不存在"}
    with np.load(path, allow_pickle=True) as data:
        rewards = data["rewards"]
        meta = json.loads(str(data["meta"].item()))
        obs = data["obs"]
        rounds = data["rounds"] if "rounds" in data else None
        return {
            "ok": True,
            "obsShape": list(obs.shape),
            "rounds": sorted({int(value) for value in rounds}) if rounds is not None else [],
            "rewardSource": meta.get("rewardSource"),
            "rewardSum": round(float(rewards.sum()), 2),
            "nonZeroFrames": int((rewards != 0).sum()),
            "frames": int(obs.shape[0]),
        }


def verify(match: Dict[str, Any], match_dir: Optional[str], want_rounds: int,
           want_obs_dim: int) -> Dict[str, Any]:
    """把判据算成结构化结果，便于打印或单测。"""
    problems: List[str] = []
    details = (match.get("roundsDetail") or []) if isinstance(match, dict) else []

    if str(match.get("status")) != "finished":
        problems.append("对局没有 finished：%s（%s）" % (match.get("status"), match.get("error") or "无错误信息"))
    if len(details) != want_rounds:
        problems.append("逐局记录 %d 条，期望 %d 条" % (len(details), want_rounds))

    records = record_files(match_dir) if match_dir else []
    if len(records) != want_rounds:
        problems.append("Arena 录制 %d 份，期望 %d 份" % (len(records), want_rounds))

    report = dataset_report(match_dir) if match_dir else {"ok": False, "error": "没有找到本场的产物目录"}
    if not report.get("ok"):
        problems.append("数据集不可读：%s" % report.get("error"))
    else:
        if report["rewardSource"] != "arena-record":
            problems.append("奖励来源是 %r，期望 arena-record" % report["rewardSource"])
        if report["nonZeroFrames"] <= 0:
            problems.append("rewards 全是 0——事件没有对齐到帧")
        if report["rewardSum"] >= 0:
            problems.append("rewards.sum=%s，期望为负（受伤为负）" % report["rewardSum"])
        if report["obsShape"][1] != want_obs_dim:
            problems.append("obs 维度 %d，期望 %d" % (report["obsShape"][1], want_obs_dim))

    return {
        "ok": not problems,
        "problems": problems,
        "matchDir": match_dir,
        "records": [os.path.basename(path) for path in records],
        "dataset": report,
    }


def observation_dim() -> int:
    sys.path.insert(0, os.path.join(ROOT, "training"))
    from gladiator_env.spaces import OBS_DIM  # noqa: PLC0415  (延迟导入，避免 --help 也拖 numpy)

    return int(OBS_DIM)


def main(argv: Optional[List[str]] = None) -> int:
    # Windows 控制台默认 cp936，输出里的「✓/✗/中文」会抛 UnicodeEncodeError；
    # 这里强制 UTF-8 + 替换，保证脚本在 cmd/PowerShell/pwsh 下都能正常退出。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="人机对战端到端冒烟（板子引擎 + Arena 录制 + 赛后导入）")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8080")
    parser.add_argument("--player", required=True, help="玩家侧 Bot 名（需已在服内）")
    parser.add_argument("--bot", required=True, help="对局 Bot 名")
    parser.add_argument("--env-port", type=int, required=True, help="通道 B 端口（空闲的那个）")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-seconds", type=int, default=20)
    parser.add_argument("--kit", default="diamond")
    parser.add_argument("--brain", default="dagger-1118b")
    parser.add_argument("--mode", default="spawn", choices=("spawn", "attach"))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--matches", default=DEFAULT_MATCHES)
    parser.add_argument("--no-train", action="store_true", help="不勾赛后训练（只看录制）")
    args = parser.parse_args(argv)

    started = time.time()
    payload = {
        "mode": args.mode,
        "player": args.player,
        "bot": args.bot,
        "envPort": args.env_port,
        "rounds": args.rounds,
        "roundSeconds": args.round_seconds,
        "kit": args.kit,
        "brain": args.brain,
        "record": True,
        "trainAfter": not args.no_train,
        "engine": "board",
    }
    print("[smoke] start %s" % json.dumps(payload, ensure_ascii=False), flush=True)
    response = post_json(args.dashboard + "/api/match/start", payload)
    if not response.get("ok"):
        print("[smoke] 启动失败：%s" % json.dumps(response, ensure_ascii=False))
        return 1

    match = wait_finished(args.dashboard, args.timeout)
    print("[smoke] status=%s score=%s" % (match.get("status"), json.dumps(match.get("score"), ensure_ascii=False)))
    for entry in match.get("roundsDetail") or []:
        print("[smoke] 第 %s 局 winner=%s reason=%s seconds=%s"
              % (entry.get("round"), entry.get("winner"), entry.get("reason"), entry.get("seconds")))

    match_dir = latest_match_dir(args.matches, started)
    result = verify(match, match_dir, args.rounds, observation_dim())
    print("[smoke] result %s" % json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        for problem in result["problems"]:
            print("[smoke] [FAIL] %s" % problem)
        return 1
    print("[smoke] [PASS] 端到端通过：%d 局、录制 %d 份、rewards.sum=%s"
          % (args.rounds, len(result["records"]), result["dataset"].get("rewardSum")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
