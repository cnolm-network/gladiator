"""G-503 故障注入：杀掉一个 bot-runtime / 掐断链路，检查插件能自愈且不留残留会话。

前置：服务端已开 RCON，4 个 bot-runtime 已启动并 spawn 完毕（可用 ``stress.py`` 或手动拉起）。
脚本自己会：

1. 记录基线（``/gladiator status`` 的 ``bots`` / ``sessions``）；
2. **链路闪断**：重启其中一个 runtime，并在启动时设置 ``GLADIATOR_TEST_DROP_MS``，
   让它连上后主动掐断一次 socket —— 断言运行时会自动重连、插件会话数恢复且 Bot 实体不被回收；
3. **进程崩溃**：直接 ``kill`` 一个 runtime —— 断言玩家下线后插件回收 BotHandle、会话数下降；
   然后用同样的命令重启该 runtime，重新 ``/gladiator spawn``，断言 Bot 回到 Ready。

用法::

    python scripts/chaos.py --users Glad_Bot_1,Glad_Bot_2,Glad_Bot_3,Glad_Bot_4 \
        --ports 8766,8767,8768,8769 --pids 1111,2222,3333,4444
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gladiator_env.rcon import RconClient

BOT_RUNTIME = Path(__file__).resolve().parents[2] / "bot-runtime"


def parse_status(text: str) -> dict[str, int]:
    """把 ``/gladiator status`` 的多行输出解析成计数。"""
    result: dict[str, int] = {}
    for key in ("bots", "games", "training", "sessions"):
        match = re.search(rf"{key}=(\d+)", text)
        if match:
            result[key] = int(match.group(1))
    return result


class Chaos:
    def __init__(self, rcon: RconClient, users: list[str], ports: list[int], pids: list[int]):
        self.rcon = rcon
        self.users = users
        self.ports = ports
        self.pids = pids
        self.failures: list[str] = []
        self.child: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # 断言与等待
    # ------------------------------------------------------------------

    def check(self, ok: bool, message: str) -> None:
        print(("[PASS] " if ok else "[FAIL] ") + message)
        if not ok:
            self.failures.append(message)

    def status(self) -> dict[str, int]:
        return parse_status(self.rcon.command("gladiator status"))

    def wait_for(self, predicate, timeout: float, description: str) -> bool:
        deadline = time.time() + timeout
        last: dict[str, int] = {}
        while time.time() < deadline:
            try:
                last = self.status()
            except Exception as exc:  # noqa: BLE001 - RCON 偶发超时不影响重试
                print(f"[warn] status 查询失败: {exc}")
                time.sleep(1.0)
                continue
            if predicate(last):
                print(f"[chaos] 满足条件：{description} → {last}")
                return True
            time.sleep(1.0)
        print(f"[chaos] 超时：{description}（最后一次 {last}）")
        return False

    # ------------------------------------------------------------------
    # 场景
    # ------------------------------------------------------------------

    def kill(self, pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, 9)

    def start_runtime(self, index: int, extra_env: dict[str, str] | None = None) -> subprocess.Popen:
        env = os.environ.copy()
        env.setdefault("GLADIATOR_BRIDGE_HOST", "127.0.0.1")
        env.setdefault("GLADIATOR_BRIDGE_PORT", "8765")
        if extra_env:
            env.update(extra_env)
        return subprocess.Popen(
            [
                "node",
                "src/index.js",
                "--username",
                self.users[index],
                "--env-port",
                str(self.ports[index]),
            ],
            cwd=str(BOT_RUNTIME),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def scenario_link_flap(self, index: int) -> None:
        """链路闪断：runtime 连上后主动掐一次 socket，必须自动重连且不产生重复会话。"""
        user = self.users[index]
        print(f"\n=== 场景 1：{user} 链路闪断（自动重连 + 插件 rebind） ===")
        baseline = self.status()
        print(f"[chaos] 基线 {baseline}")

        self.kill(self.pids[index])
        ok = self.wait_for(
            lambda s: s.get("sessions", 99) < baseline.get("sessions", 0)
            and s.get("bots", 99) < baseline.get("bots", 0),
            timeout=30.0,
            description="杀掉 runtime 后会话/BotHandle 被回收",
        )
        self.check(ok, "杀掉 runtime 后插件无残留会话")

        self.child = self.start_runtime(index, {"GLADIATOR_TEST_DROP_MS": "3000"})
        self.pids[index] = self.child.pid
        print(f"[chaos] 重启 {user} runtime pid={self.child.pid}，3s 后主动掐断一次链路")

        ok = self.wait_for(
            lambda s: s.get("sessions") == baseline.get("sessions"),
            timeout=30.0,
            description="自动重连后会话数恢复",
        )
        self.check(ok, "链路闪断后自动重连、会话数恢复")

        self.rcon.command(f"gladiator spawn {user}")
        ok = self.wait_for(
            lambda s: s.get("bots") == baseline.get("bots"),
            timeout=60.0,
            description="重新 spawn 后 Bot 回到 Ready",
        )
        self.check(ok, "链路闪断后重新 spawn 可回到 Ready")

        # 再等一个完整的重连周期，确认会话不会被重复计数（插件侧 replaced-by-reconnect）。
        time.sleep(12.0)
        after = self.status()
        self.check(
            after.get("sessions") == baseline.get("sessions"),
            f"重连后没有重复会话（sessions={after.get('sessions')}）",
        )
        self.check(
            after.get("bots") == baseline.get("bots"),
            f"重连后没有重复 BotHandle（bots={after.get('bots')}）",
        )

    def scenario_crash_restart(self, index: int) -> None:
        """进程崩溃：玩家下线 → 插件回收；重启并重新 spawn → Ready（干净重启，无 drop）。"""
        user = self.users[index]
        print(f"\n=== 场景 2：{user} 进程崩溃 + 干净重启 ===")
        baseline = self.status()
        print(f"[chaos] kill runtime pid={self.pids[index]}")
        self.kill(self.pids[index])

        ok = self.wait_for(
            lambda s: s.get("sessions", 99) < baseline.get("sessions", 0)
            and s.get("bots", 99) < baseline.get("bots", 0),
            timeout=30.0,
            description="崩溃后会话与 BotHandle 都被回收",
        )
        self.check(ok, "崩溃后插件回收会话与 BotHandle（无残留）")

        self.child = self.start_runtime(index)
        self.pids[index] = self.child.pid
        print(f"[chaos] 已重启 {user} runtime pid={self.child.pid}")

        ok = self.wait_for(
            lambda s: s.get("sessions") == baseline.get("sessions"),
            timeout=30.0,
            description="重启后链路会话恢复",
        )
        self.check(ok, "重启后链路会话恢复")

        self.rcon.command(f"gladiator spawn {user}")
        ok = self.wait_for(
            lambda s: s.get("bots") == baseline.get("bots"),
            timeout=60.0,
            description="重新 spawn 后 Bot 回到 Ready",
        )
        self.check(ok, "重新 spawn 后 Bot 回到 Ready")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gladiator 故障注入（G-503）")
    parser.add_argument("--users", default="Glad_Bot_1,Glad_Bot_2,Glad_Bot_3,Glad_Bot_4")
    parser.add_argument("--ports", default="8766,8767,8768,8769")
    parser.add_argument("--pids", required=True, help="与 --users 一一对应的 node 进程 PID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rcon-port", type=int, default=25575)
    parser.add_argument("--rcon-password", default="devpass")
    parser.add_argument("--flap-index", type=int, default=0)
    parser.add_argument("--crash-index", type=int, default=1)
    args = parser.parse_args(argv)

    users = [item.strip() for item in args.users.split(",") if item.strip()]
    ports = [int(item) for item in args.ports.split(",") if item.strip()]
    pids = [int(item) for item in args.pids.split(",") if item.strip()]
    if not (len(users) == len(ports) == len(pids)):
        raise SystemExit("--users / --ports / --pids 数量必须一致")

    rcon = RconClient(host=args.host, port=args.rcon_port, password=args.rcon_password)
    rcon.connect()
    chaos = Chaos(rcon, users, ports, pids)
    try:
        baseline = chaos.status()
        print(f"[chaos] 基线 {baseline}")
        if baseline.get("bots", 0) < len(users):
            print(f"[warn] 基线 bots={baseline.get('bots')} < {len(users)}，请先 spawn 全部 Bot")

        chaos.scenario_link_flap(args.flap_index)
        chaos.scenario_crash_restart(args.crash_index)
    finally:
        rcon.close()

    print("\n[chaos] 结果: " + ("全部通过" if not chaos.failures else f"{len(chaos.failures)} 项失败"))
    for failure in chaos.failures:
        print("  - " + failure)
    return 0 if not chaos.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
