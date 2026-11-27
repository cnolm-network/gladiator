"""人机对战编排器（`dashboard/match_runner.py`）的测试。

G-715 之后编排器的规则部分是「发一条 `/gladiator board start` + 每 500ms 读 status」，
所以这里的假连接模拟的是**服务器侧板子**：`board status` 每次调用推进一格脚本，
编排器必须只靠 status 的 `phase`/`score` 收敛结果。

覆盖：配置校验、attach/spawn 选择、局数推进与比分、板子视角的判死（不轮询血量）、
录制（帧 + rewards + meta/summary）、赛后导入、停止、快照旧字段兼容。

配套的纯函数测试（status 解析、rewards 计算、board start 命令拼装）在
`test_match_board.py`。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from config import DashboardConfig  # noqa: E402
from errors import BadRequest, Conflict  # noqa: E402
from match_runner import (  # noqa: E402
    BOARD_ID,
    MatchPlan,
    MatchRunner,
    argv_brain_spec,
    build_import_command,
    choose_runtime,
    parse_stats_line,
)

# --- 假实现 ----------------------------------------------------------------


class FakeClock:
    """sleep 直接推进时间：一场几分钟的对局在毫秒内跑完。"""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class FakeBoardConn:
    """脚本化的「板子 + RCON」。

    `steps` 里每一格描述一次 `board status` 采样：

    ``{"phase": "fighting", "round": 1, "rounds": 2, "score": (1, 0), "draws": 0,
       "reason": "kill", "health": {"player": 20, "bot": 0}, "session": "s-1"}``

    缺省 ``session="s-1"``；显式 ``"session": None`` 表示 `session=none`（无对局）。
    脚本走完后一直重复最后一格（和真实板子在 ended 后保持不变一致）。
    """

    def __init__(self, steps: list[dict], player: str, bot: str, *,
                 board_start_reply: str | None = None,
                 status_text: str | None = None) -> None:
        self.steps = list(steps)
        self.player_name = player
        self.bot_name = bot
        self.commands: list[str] = []
        self.index = 0
        self.closed = False
        self.started = False
        self.start_command: str | None = None
        self.board_start_reply = board_start_reply
        self.status_text = status_text
        self.health: dict[str, Any] = {"player": 20.0, "bot": 20.0}
        self.online = {"player": True, "bot": True}
        self.pos = {"player": [0.0, 4.0, 0.0], "bot": [6.0, 4.0, 0.0]}

    def _key(self, name: str) -> str:
        return "player" if str(name).lower() == self.player_name.lower() else "bot"

    def _status_line(self, step: dict) -> str:
        session = step.get("session", "s-1")
        if session is None:
            sid, board = "none", ""
        else:
            sid, board = str(session), BOARD_ID
        score = step.get("score") or (0, 0)
        return (f"session={sid} board={board} round={step.get('round', 0)}"
                f" rounds={step.get('rounds', 1)} phase={step.get('phase', 'idle')}"
                f" score={score[0]}:{score[1]} draws={step.get('draws', 0)}"
                f" reason={step.get('reason') or 'none'}"
                f" player={self.player_name} bot={self.bot_name}")

    def command(self, text: str) -> str:
        self.commands.append(text)
        if text.startswith("gladiator board start"):
            self.started = True
            self.start_command = text
            if self.board_start_reply is not None:
                return self.board_start_reply
            rounds = self.steps[0].get("rounds", 1) if self.steps else 1
            return f"board 已开始 session=s-1 rounds={rounds} seconds=10"
        if text == "gladiator board status":
            if self.status_text is not None:
                return self.status_text
            step = self.steps[min(self.index, len(self.steps) - 1)]
            for side, value in (step.get("health") or {}).items():
                self.health[side] = float(value)
            self.index += 1
            return self._status_line(step)
        if text == "gladiator board stop":
            return "board 已停止 session=s-1 reason=command"
        if text.startswith("gladiator stats "):
            name = text.split()[-1]
            key = self._key(name)
            if not self.online.get(key, True) or self.health[key] is None:
                return "stats 失败：玩家不在线"
            x, y, z = self.pos[key]
            return (f"name={name} health={float(self.health[key]):.1f} food=20"
                    f" saturation=5.0 armor=0.0 x={x:.2f} y={y:.2f} z={z:.2f}"
                    " yaw=0.0 pitch=0.0 sprinting=false blocking=false"
                    " onGround=true hurt=0.0")
        if text == "gladiator list":
            return f"  - glad-{self.bot_name} name={self.bot_name} uuid=x ready=true state=Alive"
        if text == "gladiator spawn " + self.bot_name:
            return f"spawn 已就绪：{self.bot_name}"
        if text == "gladiator status":
            return ("bots=1 games=0 training=0 pending=0 spawn=runtime\n"
                    "[Gladiator] link=127.0.0.1:8765 sessions=1")
        return ""

    def close(self) -> None:
        self.closed = True


class FakeRuntimes:
    def __init__(self, items: list[dict] | None = None) -> None:
        self.items = [dict(item) for item in (items or [])]
        self.started: list[dict] = []
        self.stopped: list[str] = []

    def list(self) -> list[dict]:
        return [dict(item) for item in self.items]

    def start(self, username: str, env_port: int, *, extra_args: list[str] | None = None,
              **_: Any) -> dict:
        argv = ["node", "src/index.js", "--username", username, "--env-port", str(env_port)]
        argv.extend(extra_args or [])
        view = {
            "id": f"runtime-{env_port}",
            "pid": 4242,
            "envPort": env_port,
            "alive": True,
            "external": False,
            "username": username,
            "command": argv,
        }
        self.started.append({"username": username, "envPort": env_port,
                             "extraArgs": list(extra_args or [])})
        self.items.append(view)
        return dict(view)

    def stop(self, runtime_id: str) -> dict:
        self.stopped.append(runtime_id)
        self.items = [item for item in self.items if item.get("id") != runtime_id]
        return {"id": runtime_id, "alive": False}

    def stop_all(self) -> list:
        return []


class FakeModels:
    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = mapping

    def policy_path(self, model_id: str) -> Path | None:
        return self.mapping.get(model_id)


def make_cfg(root: Path) -> DashboardConfig:
    return DashboardConfig(
        host="127.0.0.1",
        port=8080,
        artifacts_dir=root / "artifacts",
        var_dir=root / "var",
        server_dir=root / "server",
        bot_runtime_dir=root / "bot-runtime",
        training_dir=root / "training",
        python_exe=sys.executable,
        java_exe="java",
        node_exe="node",
        allow_process_control=True,
        scan_external=False,
        server_jar="missing.jar",
        mc_port=25998,
        rcon_port=25575,
        runtime_port_min=8770,
        runtime_port_max=8789,
    )


def simple_rounds(rounds: int = 1, *, seconds_per_round: int = 10) -> list[dict]:
    """一边倒的脚本：第 1 局玩家击杀 Bot，其余局打平，最后 ended。"""
    steps: list[dict] = [{"phase": "countdown", "round": 1, "rounds": rounds}]
    for index in range(1, rounds + 1):
        steps.append({"phase": "fighting", "round": index, "rounds": rounds,
                      "score": (0, 0), "draws": index - 1})
        if index == 1:
            steps.append({"phase": "fighting", "round": index, "rounds": rounds,
                          "score": (1, 0), "draws": 0, "reason": "kill"})
        else:
            steps.append({"phase": "fighting", "round": index, "rounds": rounds,
                          "score": (1, 0), "draws": index - 1})
    steps.append({"phase": "ended", "round": rounds, "rounds": rounds,
                  "score": (1, 0), "draws": rounds - 1, "reason": "completed"})
    return steps


# --- 配置与纯函数 -----------------------------------------------------------


class MatchPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_defaults(self) -> None:
        plan = MatchPlan.from_payload({"player": "Steve"}, self.cfg)
        self.assertEqual(plan.mode, "spawn")
        self.assertEqual(plan.bot, "Glad_Bot_1")
        self.assertEqual(plan.env_port, 8770)
        self.assertEqual(plan.rounds, 3)
        self.assertEqual(plan.round_seconds, 120.0)
        self.assertEqual(plan.kit, "diamond")
        self.assertTrue(plan.record)
        self.assertFalse(plan.train_after)
        self.assertIsNone(plan.brain)
        self.assertEqual(plan.respawn_delay_ms, 800)
        data = plan.to_dict()
        self.assertEqual(data["player"], "Steve")
        self.assertEqual(data["rounds"], 3)
        self.assertEqual(data["respawnDelayMs"], 800)

    def test_respawn_delay_is_configurable_and_bounded(self) -> None:
        # 旧字段：重生归板子后 dashboard 不再用它，但请求体必须继续接受（前端还在发）。
        plan = MatchPlan.from_payload({"player": "Steve", "respawnDelayMs": 0}, self.cfg)
        self.assertEqual(plan.respawn_delay_ms, 0)
        plan = MatchPlan.from_payload({"player": "Steve", "respawnDelayMs": 1500}, self.cfg)
        self.assertEqual(plan.respawn_delay_ms, 1500)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "respawnDelayMs": 6000}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "respawnDelayMs": -1}, self.cfg)

    def test_train_after_forces_recording(self) -> None:
        plan = MatchPlan.from_payload(
            {"player": "Steve", "record": False, "trainAfter": True}, self.cfg)
        self.assertTrue(plan.record, "赛后训练必须录制，否则导入拿到空数据集")
        self.assertTrue(plan.record_forced)
        self.assertTrue(plan.to_dict()["recordForced"])

    def test_recording_not_forced_without_train_after(self) -> None:
        plan = MatchPlan.from_payload({"player": "Steve", "record": False}, self.cfg)
        self.assertFalse(plan.record)
        self.assertFalse(plan.record_forced)
        self.assertFalse(plan.to_dict()["recordForced"])

    def test_player_is_required(self) -> None:
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({}, self.cfg)

    def test_invalid_mode_kit_and_ranges(self) -> None:
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "mode": "duel"}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "kit": "netherite"}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "rounds": 0}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "rounds": 21}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "roundSeconds": 5}, self.cfg)
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "envPort": 25575}, self.cfg)

    def test_spawns_optional_and_partial(self) -> None:
        plan = MatchPlan.from_payload({"player": "Steve", "spawns": None}, self.cfg)
        self.assertIsNone(plan.spawns)
        plan = MatchPlan.from_payload(
            {"player": "Steve", "spawns": {"player": [0, 4, 0], "bot": [5, 4, 0]}}, self.cfg)
        self.assertEqual(plan.spawns["Steve"], (0.0, 4.0, 0.0))
        self.assertEqual(plan.spawns["Glad_Bot_1"], (5.0, 4.0, 0.0))
        partial = MatchPlan.from_payload(
            {"player": "Steve", "spawns": {"bot": {"x": 1, "y": 4, "z": 2}}}, self.cfg)
        self.assertEqual(list(partial.spawns), ["Glad_Bot_1"])
        with self.assertRaises(BadRequest):
            MatchPlan.from_payload({"player": "Steve", "spawns": {"player": "abc"}}, self.cfg)

    def test_parse_stats_line(self) -> None:
        line = ("name=Steve health=18.5 food=20 saturation=5.0 armor=16.0 x=0.5 y=4.0 z=0.5"
                " yaw=90.0 pitch=0.0 sprinting=false blocking=true onGround=true hurt=0.0")
        stats = parse_stats_line(line)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["name"], "Steve")
        self.assertAlmostEqual(stats["health"], 18.5)
        self.assertIs(stats["blocking"], True)
        self.assertIs(stats["sprinting"], False)
        self.assertIsNone(parse_stats_line("stats 失败：玩家不在线"))
        self.assertIsNone(parse_stats_line(""))


class RuntimeChoiceTests(unittest.TestCase):
    def test_name_match_prefers_username(self) -> None:
        runtimes = [
            {"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1", "envPort": 8775,
             "external": True, "command": ["node", "src/index.js"]},
            {"id": "runtime-8776", "alive": True, "username": "Glad_Bot_2", "envPort": 8776,
             "external": False, "command": ["node", "src/index.js"]},
        ]
        decision = choose_runtime(runtimes, "Glad_Bot_2", 8775, None)
        self.assertEqual(decision["action"], "reuse")
        self.assertEqual(decision["runtime"]["id"], "runtime-8776")

    def test_port_match_includes_external(self) -> None:
        runtimes = [{"id": "external-8775", "alive": True, "username": None, "envPort": 8775,
                     "external": True, "command": ["node", "src/index.js"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, None)
        self.assertEqual(decision["action"], "reuse")
        self.assertTrue(decision["runtime"]["external"])

    def test_missing_runtime_starts_new(self) -> None:
        self.assertEqual(choose_runtime([], "Glad_Bot_1", 8770, None)["action"], "start")

    def test_brain_requires_restart_when_argv_has_none(self) -> None:
        runtimes = [{"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1",
                     "envPort": 8775, "external": False,
                     "command": ["node", "src/index.js", "--username", "Glad_Bot_1"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "/m/policy.npz")
        self.assertEqual(decision["action"], "restart")

    def test_brain_same_model_is_reused(self) -> None:
        runtimes = [{"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1",
                     "envPort": 8775, "external": False,
                     "command": ["node", "src/index.js", "--brain", "/m/policy.npz"]}]
        self.assertEqual(choose_runtime(runtimes, "Glad_Bot_1", 8775, "/m/policy.npz")["action"],
                         "reuse")

    def test_external_runtime_cannot_be_restarted(self) -> None:
        runtimes = [{"id": "external-8775", "alive": True, "username": None, "envPort": 8775,
                     "external": True, "command": ["node", "src/index.js"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "/m/policy.npz")
        self.assertEqual(decision["action"], "blocked")

    def test_external_runtime_can_be_reused_when_brain_assumed(self) -> None:
        """attach + 用户声明「外部运行时已在跑同一模型」时不要 blocked。"""
        runtimes = [{"id": "external-8775", "alive": True, "username": None, "envPort": 8775,
                     "external": True, "command": ["node", "src/index.js"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "/m/policy.npz",
                                  assume_brain=True)
        self.assertEqual(decision["action"], "reuse")
        self.assertIn("brainAssumed", decision["reason"])

    def test_brain_assumed_does_not_override_missing_runtime(self) -> None:
        decision = choose_runtime([], "Glad_Bot_1", 8775, "/m/policy.npz", assume_brain=True)
        self.assertEqual(decision["action"], "start")

    def test_argv_brain_spec(self) -> None:
        self.assertEqual(argv_brain_spec(["node", "--brain", "/x.npz", "--host", "1"]), "/x.npz")
        self.assertIsNone(argv_brain_spec(["node", "src/index.js"]))
        self.assertIsNone(argv_brain_spec([]))


class ImportCommandTests(unittest.TestCase):
    def test_command_with_bc(self) -> None:
        command = build_import_command(
            "python.exe", Path("C:/t/import_human_match.py"), Path("C:/rec"),
            bc=True, base_model="C:/m/policy.npz", dataset=Path("C:/rec/dataset.npz"),
        )
        self.assertEqual(command[0], "python.exe")
        self.assertEqual(command[1], str(Path("C:/t/import_human_match.py")))
        self.assertEqual(command[2], str(Path("C:/rec")))
        self.assertIn("--out", command)
        self.assertEqual(command[command.index("--out") + 1], str(Path("C:/rec/dataset.npz")))
        self.assertIn("--bc", command)
        self.assertEqual(command[command.index("--base-model") + 1], "C:/m/policy.npz")

    def test_command_without_bc(self) -> None:
        command = build_import_command("py", Path("s.py"), Path("d"), bc=False,
                                       base_model=None)
        self.assertNotIn("--bc", command)
        self.assertNotIn("--base-model", command)
        self.assertEqual(command[:3], ["py", str(Path("s.py")), str(Path("d"))])


# --- 端到端（假板子 + 假时钟） ---------------------------------------------


class MatchFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.cfg.ensure_dirs()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _runner(self, steps: list[dict], runtimes: FakeRuntimes | None = None,
                models: FakeModels | None = None, **kwargs: Any
                ) -> tuple[MatchRunner, FakeBoardConn, FakeClock]:
        clock = FakeClock()
        conn = FakeBoardConn(steps, "Steve", "Glad_Bot_1", **kwargs)
        runner = MatchRunner(
            self.cfg, server=mock.Mock(), runtimes=runtimes or FakeRuntimes(), models=models,
            rcon_factory=lambda: conn, clock=clock.now, sleeper=clock.sleep,
        )
        return runner, conn, clock

    def _start_and_wait(self, runner: MatchRunner, payload: dict) -> dict:
        # 补丁必须覆盖整个对局线程：线程里会调 is_port_open/_wait_listening，
        # 提前退出 with 会让它去打真实端口、白等 15~20 秒。
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start(payload)
            thread = runner._thread
            self.assertIsNotNone(thread)
            thread.join(timeout=20)
            self.assertFalse(thread.is_alive(), "对局线程未在 20s 内结束")
        return runner.current()

    # --- 起局 -----------------------------------------------------------
    def test_board_start_command_and_lifecycle(self) -> None:
        runner, conn, _ = self._runner(simple_rounds(1))
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "bot": "Glad_Bot_1", "envPort": 8775,
            "rounds": 1, "roundSeconds": 10, "record": False,
        })
        self.assertEqual(conn.start_command,
                         "gladiator board start flat-1v1 Steve Glad_Bot_1 1 10 kit=classic")
        self.assertEqual(snapshot["phase"], "finished")
        self.assertEqual(snapshot["boardSession"], "s-1")
        # 只有一条 start + 若干 status，没有 stop（板子自己收尾）。
        self.assertEqual(
            [item for item in conn.commands if item.startswith("gladiator board")][0],
            conn.start_command)
        self.assertNotIn("gladiator board stop", conn.commands)

    def test_board_start_carries_spawn_coordinates(self) -> None:
        runner, conn, _ = self._runner(simple_rounds(1))
        self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False, "spawns": {"player": [0, 4, 0], "bot": [6, 4, 0]},
        })
        self.assertEqual(conn.start_command,
                         "gladiator board start flat-1v1 Steve Glad_Bot_1 1 10"
                         " 0.00 4.00 0.00 6.00 4.00 0.00 kit=classic")

    def test_no_rule_commands_from_dashboard(self) -> None:
        """套件/清包/传送/回血全部归板子：dashboard 一条都不许发。"""
        runner, conn, _ = self._runner(simple_rounds(1))
        self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False, "spawns": {"player": [0, 4, 0], "bot": [6, 4, 0]},
        })
        forbidden = ("give ", "clear ", "effect ", "tp ", "spawnpoint ", "gamemode ",
                     "gladiator respawn ")
        offenders = [item for item in conn.commands if item.startswith(forbidden)]
        self.assertEqual(offenders, [], "dashboard 不应再直接下规则命令")

    def test_board_start_failure_marks_failed(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1),
                                    board_start_reply="board 启动失败：已有对局在进行中")
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["phase"], "failed")
        self.assertIn("板子启动失败", snapshot["error"])

    def test_missing_board_command_marks_failed(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1),
                                    board_start_reply="未知的命令，请用 /help 查看")
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["phase"], "failed")
        self.assertIn("未实现", snapshot["error"])

    def test_unparseable_status_marks_failed(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1), status_text="这不是 status")
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["phase"], "failed")
        self.assertIn("无法解析", snapshot["error"])

    def test_player_must_join_before_start(self) -> None:
        """玩家不自己进服就别开局：失败信息要直接告诉他去哪个地址。"""
        clock = FakeClock()
        conn = FakeBoardConn(simple_rounds(1), "Steve", "Glad_Bot_1")
        conn.online["player"] = False
        runner = MatchRunner(self.cfg, server=mock.Mock(), runtimes=FakeRuntimes(),
                             rcon_factory=lambda: conn, clock=clock.now, sleeper=clock.sleep)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["phase"], "failed")
        self.assertIn("请先自己进服", snapshot["error"])
        # 玩家不在线就绝不 board start：否则板子只会回一句含糊的「启动失败」。
        self.assertFalse(any(item.startswith("gladiator board start") for item in conn.commands))

    # --- 判定与比分 -----------------------------------------------------
    def test_rounds_progress_and_score(self) -> None:
        # 两局都打满时间、双方同血量 → 板子记两个平局；dashboard 只跟着 status 走。
        steps = [
            {"phase": "countdown", "round": 1, "rounds": 2},
            {"phase": "fighting", "round": 1, "rounds": 2, "draws": 1},
            {"phase": "countdown", "round": 2, "rounds": 2, "draws": 1},
            {"phase": "fighting", "round": 2, "rounds": 2, "draws": 2, "reason": "timeout"},
            {"phase": "ended", "round": 2, "rounds": 2, "draws": 2, "reason": "completed"},
        ]
        runner, _, _ = self._runner(steps)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "bot": "Glad_Bot_1", "envPort": 8775,
            "rounds": 2, "roundSeconds": 10, "record": False,
        })
        self.assertEqual(snapshot["phase"], "finished")
        self.assertEqual(len(snapshot["roundsDetail"]), 2)
        self.assertEqual([item["winner"] for item in snapshot["roundsDetail"]],
                         ["draw", "draw"])
        self.assertEqual([item["reason"] for item in snapshot["roundsDetail"]],
                         ["unknown", "timeout"])
        self.assertEqual(snapshot["score"], {"player": 0, "bot": 0, "draws": 2})
        self.assertEqual(snapshot["round"], 2)
        self.assertEqual(snapshot["rounds"], 2)
        # spawn 模式：自己拉起的运行时在对局结束后必须停掉。
        self.assertEqual(runner.runtimes.stopped, ["runtime-8775"])

    def test_kill_is_judged_by_board_events_not_health_polling(self) -> None:
        """1.8 死亡到重生只有 2 tick：血量永远是 20/20，只有板子知道谁死了。

        这就是旧 `judge_round` 漏判（三局全 timeout）的回归样本：status 里血量没变，
        但比分 +1，dashboard 必须立刻记一局 bot 胜（reason=kill）。
        """
        steps = [
            {"phase": "countdown", "round": 1, "rounds": 1},
            {"phase": "fighting", "round": 1, "rounds": 1, "health": {"player": 20, "bot": 20}},
            {"phase": "fighting", "round": 1, "rounds": 1, "health": {"player": 20, "bot": 20},
             "score": (0, 1), "reason": "kill"},
            {"phase": "ended", "round": 1, "rounds": 1, "score": (0, 1), "reason": "completed"},
        ]
        runner, conn, _ = self._runner(steps)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 120,
            "record": False, "spawns": {"player": [0, 4, 0], "bot": [6, 4, 0]},
        })
        entry = snapshot["roundsDetail"][0]
        self.assertEqual(entry["winner"], "bot")
        self.assertEqual(entry["reason"], "kill")
        self.assertEqual(snapshot["score"], {"player": 0, "bot": 1, "draws": 0})
        self.assertEqual(snapshot["phase"], "finished")
        # 旧实现的死亡恢复痕迹必须一条都没有：不轮询判死、不重生、不重新摆位。
        log_text = " ".join(item["text"] for item in snapshot["log"])
        self.assertNotIn("已强制重生", log_text)
        self.assertNotIn("未回到站位", log_text)
        self.assertFalse(any(item.startswith("tp ") for item in conn.commands))

    def test_session_none_ends_match(self) -> None:
        steps = [
            {"phase": "countdown", "round": 1, "rounds": 1},
            {"phase": "fighting", "round": 1, "rounds": 1, "score": (1, 0), "reason": "kill"},
            {"session": None, "phase": "idle", "round": 0, "rounds": 0,
             "score": (1, 0), "reason": "command"},
        ]
        runner, _, _ = self._runner(steps)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["phase"], "finished")
        self.assertEqual(snapshot["roundsDetail"][0]["winner"], "player")
        self.assertEqual(snapshot["score"]["player"], 1)

    # --- 录制 -----------------------------------------------------------
    def test_recording_writes_frames_meta_summary(self) -> None:
        steps = [
            {"phase": "countdown", "round": 1, "rounds": 1},
            {"phase": "fighting", "round": 1, "rounds": 1, "health": {"player": 20, "bot": 20}},
            {"phase": "fighting", "round": 1, "rounds": 1, "health": {"player": 18, "bot": 20}},
            {"phase": "fighting", "round": 1, "rounds": 1, "health": {"player": 10, "bot": 14}},
            {"phase": "ended", "round": 1, "rounds": 1, "score": (0, 1), "reason": "kill"},
        ]
        runner, _, _ = self._runner(steps)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": True,
        })
        record_dir = Path(snapshot["recordDir"])
        self.assertTrue(record_dir.is_dir(), record_dir)
        self.assertIn("matches", record_dir.parts)
        self.assertRegex(record_dir.name, r"^\d{8}-\d{6}-Steve-vs-Glad_Bot_1$")

        frames_path = record_dir / "frames.jsonl"
        self.assertTrue(frames_path.is_file())
        lines = [json.loads(line) for line in
                 frames_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(lines, "frames.jsonl 不能为空")
        frame = lines[0]
        for key in ("t", "round", "player", "bot", "rewards"):
            self.assertIn(key, frame)
        for key in ("health", "food", "saturation", "armor", "x", "y", "z",
                    "yaw", "pitch", "sprinting", "blocking", "onGround", "hurt"):
            self.assertIn(key, frame["player"])
            self.assertIn(key, frame["bot"])
        self.assertEqual(frame["round"], 1)
        # 帧里必须带本帧伤害（AWR 回报）：Bot 20→14 → playerDamage=6，玩家 18→10 → botDamage=8。
        self.assertIn({"playerDamage": 6.0, "botDamage": 8.0},
                      [item["rewards"] for item in lines])

        meta = json.loads((record_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["plan"]["player"], "Steve")
        self.assertEqual(meta["board"], BOARD_ID)
        self.assertEqual(meta["boardSession"], "s-1")
        self.assertEqual(meta["framePeriod"], 0.5)
        summary = json.loads((record_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["frames"], len(lines))
        self.assertGreater(summary["frameHz"], 0)
        self.assertEqual(summary["score"]["bot"], 1)
        self.assertEqual(summary["boardSession"], "s-1")
        self.assertEqual(len(summary["rounds"]), 1)

    def test_train_after_records_intent(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1))
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": True, "trainAfter": True,
        })
        self.assertTrue(snapshot["trainAfter"])
        # 测试环境没有 training/scripts/import_human_match.py，编排器应记录跳过而不是失败。
        log_text = " ".join(entry["text"] for entry in snapshot["log"])
        self.assertIn("跳过", log_text)

    # --- 运行时与 attach ------------------------------------------------
    def test_attach_reuses_existing_runtime_without_starting(self) -> None:
        runtimes = FakeRuntimes([{
            "id": "external-8775", "pid": 20568, "envPort": 8775, "alive": True,
            "external": True, "username": None, "command": ["node", "src/index.js"],
        }])
        runner, _, _ = self._runner(simple_rounds(1), runtimes=runtimes)
        snapshot = self._start_and_wait(runner, {
            "mode": "attach", "player": "Steve", "bot": "Glad_Bot_1", "envPort": 8775,
            "rounds": 1, "roundSeconds": 10, "record": False,
        })
        self.assertEqual(runtimes.started, [], "attach 模式不应重复启动运行时")
        self.assertEqual(snapshot["runtime"]["id"], "external-8775")
        self.assertTrue(snapshot["runtime"]["external"])
        # attach 模式不停止用户的进程。
        self.assertEqual(runtimes.stopped, [])

    def test_attach_with_brain_restarts_managed_runtime(self) -> None:
        model = self.root / "policy.npz"
        model.write_bytes(b"fake")
        runtimes = FakeRuntimes([{
            "id": "runtime-8775", "pid": 1, "envPort": 8775, "alive": True,
            "external": False, "username": "Glad_Bot_1",
            "command": ["node", "src/index.js", "--username", "Glad_Bot_1"],
        }])
        runner, _, _ = self._runner(simple_rounds(1), runtimes=runtimes,
                                    models=FakeModels({"dagger-1118b": model}))
        snapshot = self._start_and_wait(runner, {
            "mode": "attach", "player": "Steve", "bot": "Glad_Bot_1", "envPort": 8775,
            "rounds": 1, "roundSeconds": 10, "record": False, "brain": "dagger-1118b",
        })
        self.assertEqual(snapshot["phase"], "finished")
        self.assertEqual(runtimes.stopped, ["runtime-8775"])
        self.assertEqual(len(runtimes.started), 1)
        extra = runtimes.started[0]["extraArgs"]
        self.assertIn("--brain", extra)
        self.assertIn(str(model.resolve()), extra)
        self.assertIn("--target-player", extra)
        self.assertIn("Steve", extra)
        self.assertIn("--rcon", extra)
        # 20Hz：5Hz 时同一策略的出手频率只有训练时的 1/4（真机 2026-11-18 实测近身几乎不攻击）。
        self.assertEqual(extra[extra.index("--brain-hz") + 1], "20")

    def test_attach_with_brain_on_external_runtime_is_rejected(self) -> None:
        model = self.root / "policy.npz"
        model.write_bytes(b"fake")
        runtimes = FakeRuntimes([{
            "id": "external-8775", "pid": 20568, "envPort": 8775, "alive": True,
            "external": True, "username": None, "command": ["node", "src/index.js"],
        }])
        runner, _, _ = self._runner(simple_rounds(1), runtimes=runtimes,
                                    models=FakeModels({"dagger-1118b": model}))
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            with self.assertRaises(Conflict):
                runner.start({
                    "mode": "attach", "player": "Steve", "envPort": 8775,
                    "rounds": 1, "roundSeconds": 10, "brain": "dagger-1118b",
                })

    def test_brain_model_without_weights_is_rejected(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1),
                                    models=FakeModels({"empty": self.root / "missing.npz"}))
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            with self.assertRaises(BadRequest):
                runner.start({"player": "Steve", "brain": "empty"})

    # --- 停止 -----------------------------------------------------------
    def test_stop_sends_board_stop_and_cleans_runtime(self) -> None:
        steps = [{"phase": "fighting", "round": 1, "rounds": 5}] * 600
        runner, conn, _ = self._runner(steps)
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start({"mode": "spawn", "player": "Steve", "rounds": 5,
                          "roundSeconds": 1000, "record": False, "envPort": 8775})
            snapshot = runner.stop()
        self.assertEqual(snapshot["phase"], "stopped")
        self.assertFalse(snapshot["running"])
        self.assertIn("gladiator board stop", conn.commands)
        self.assertIn("runtime-8775", runner.runtimes.stopped)

    def test_second_start_is_rejected_while_running(self) -> None:
        steps = [{"phase": "fighting", "round": 1, "rounds": 5}] * 600
        runner, _, _ = self._runner(steps)
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start({"mode": "spawn", "player": "Steve", "rounds": 5,
                          "roundSeconds": 1000, "record": False})
            with self.assertRaises(Conflict):
                runner.start({"mode": "spawn", "player": "Steve", "rounds": 1})
            runner.stop()

    def test_idle_snapshot_shape(self) -> None:
        runner, _, _ = self._runner(simple_rounds(1))
        snapshot = runner.current()
        self.assertEqual(snapshot["phase"], "idle")
        self.assertFalse(snapshot["running"])
        for key in ("phase", "round", "rounds", "score", "roundsDetail", "brain",
                    "boardSession", "defaultSpawns", "countdown"):
            self.assertIn(key, snapshot)


if __name__ == "__main__":
    unittest.main()
