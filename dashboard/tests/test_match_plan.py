"""人机对战编排器（`dashboard/match_runner.py`）的测试。

覆盖任务书要求的五项：attach 选择、局数推进、超时判负、录制字段、trainAfter 参数；
另外覆盖配置校验、单局判定与运行时重启策略。

全部用注入的假 RCON / 假时钟 / 假运行时管理器，不启动 Paper，也不碰真实端口。

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
    KITS,
    SHIELD_AMPLIFIER,
    SHIELD_EFFECT_ID,
    SHIELD_SECONDS,
    MatchPlan,
    MatchRunner,
    argv_brain_spec,
    build_import_command,
    choose_runtime,
    judge_round,
    parse_stats_line,
    round_place_commands,
    round_setup_commands,
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


class FakeConn:
    """脚本化的 RCON 连接。

    ``script`` 是逐次"决策采样"的血量序列：``[{"player": 20, "bot": 20}, ...]``，
    取完后一直重复最后一项。``None`` 表示该方读不到 stats（离线）。
    """

    def __init__(self, script: list[dict], player: str, bot: str) -> None:
        self.script = list(script) or [{"player": 20.0, "bot": 20.0}]
        self.player = player.lower()
        self.bot = bot.lower()
        self.bot_name = bot
        self.commands: list[str] = []
        self.index = 0
        self.current = self.script[0]
        self.pending_player = False
        self.closed = False

    def command(self, text: str) -> str:
        self.commands.append(text)
        if text.startswith("gladiator stats "):
            name = text.split()[-1].lower()
            if name == self.player:
                self.current = self.script[min(self.index, len(self.script) - 1)]
                self.pending_player = True
                health = self.current.get("player")
            else:
                health = self.current.get("bot")
                if self.pending_player:
                    self.index += 1
                    self.pending_player = False
            if health is None:
                return "stats 失败：玩家不在线"
            return (f"name={name} health={float(health):.1f} food=20 saturation=5.0 armor=0.0"
                    " x=0.0 y=4.0 z=0.0 yaw=0.0 pitch=0.0 sprinting=false blocking=false"
                    " onGround=true hurt=0.0")
        if text == "gladiator list":
            return f"  - glad-{self.bot} name={self.bot_name} uuid=x ready=true state=Alive"
        if text == "gladiator status":
            return "bots=1 games=0 training=0 pending=0 spawn=runtime\n[Gladiator] link=127.0.0.1:8765 sessions=1"
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
        data = plan.to_dict()
        self.assertEqual(data["player"], "Steve")
        self.assertEqual(data["rounds"], 3)

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

    def test_round_setup_commands_order_and_kit(self) -> None:
        spawns = {"Steve": (0.0, 4.0, 0.0), "Glad_Bot_1": (6.0, 4.0, 0.0)}
        commands = round_setup_commands("Steve", "Glad_Bot_1", "diamond", spawns)
        self.assertEqual(commands[0], "tp Steve 0.00 4.00 0.00")
        self.assertEqual(commands[1], "tp Glad_Bot_1 6.00 4.00 0.00")
        self.assertEqual(commands[2], "clear Steve")
        self.assertIn("gamemode 0 Steve", commands)
        self.assertIn("give Steve diamond_sword", commands)
        self.assertIn("give Glad_Bot_1 golden_apple 5", commands)
        self.assertIn("effect Steve clear", commands)
        # clear 必须在 give 之前、effect clear 在 give 之后：顺序错了会吞掉套件。
        self.assertLess(commands.index("clear Steve"), commands.index("give Steve diamond_sword"))
        self.assertLess(commands.index("give Steve diamond_sword"),
                        commands.index("effect Steve clear"))
        none_kit = round_setup_commands("Steve", "Glad_Bot_1", "none", None)
        self.assertFalse(any(item.startswith("give ") for item in none_kit))
        self.assertFalse(any(item.startswith("tp ") for item in none_kit))
        self.assertIn("diamond_sword", KITS["diamond"])

    def test_place_commands_run_after_countdown(self) -> None:
        """摆位必须能单独下发：准备阶段不传送，否则 Bot 会在倒计时里打死玩家。"""
        spawns = {"Steve": (0.0, 4.0, 0.0), "Glad_Bot_1": (6.0, 4.0, 0.0)}
        place = round_place_commands("Steve", "Glad_Bot_1", spawns)
        self.assertEqual(place[0], "tp Steve 0.00 4.00 0.00")
        self.assertEqual(place[1], "spawnpoint Steve 0.00 4.00 0.00")
        self.assertIn("tp Glad_Bot_1 6.00 4.00 0.00", place)
        self.assertIn("spawnpoint Glad_Bot_1 6.00 4.00 0.00", place)
        # 倒计时免伤必须在开局时解除，否则整局无敌。
        self.assertEqual(place[-2:], ["effect Steve clear", "effect Glad_Bot_1 clear"])

        setup = round_setup_commands("Steve", "Glad_Bot_1", "diamond", spawns, place=False)
        self.assertFalse(any(item.startswith("tp ") for item in setup))
        self.assertIn("give Steve diamond_sword", setup)
        self.assertIn(f"effect Steve {SHIELD_EFFECT_ID} {SHIELD_SECONDS} {SHIELD_AMPLIFIER} true",
                      setup)

        self.assertEqual(round_place_commands("Steve", "Glad_Bot_1", None),
                         ["effect Steve clear", "effect Glad_Bot_1 clear"])
        # 只给一侧时另一侧不传送，避免把没有坐标的一方丢到默认出生点。
        half = round_place_commands("Steve", "Glad_Bot_1", {"Steve": (0.0, 4.0, 0.0)})
        self.assertEqual(half, ["tp Steve 0.00 4.00 0.00", "spawnpoint Steve 0.00 4.00 0.00",
                                "effect Steve clear", "effect Glad_Bot_1 clear"])


class RuntimeChoiceTests(unittest.TestCase):
    def test_name_match_prefers_username(self) -> None:
        runtimes = [
            {"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1", "envPort": 8775,
             "external": True, "command": ["node", "src/index.js"]},
            {"id": "runtime-8770", "alive": True, "username": "Glad_Bot_2", "envPort": 8770,
             "external": False, "command": ["node", "src/index.js"]},
        ]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8770, None)
        self.assertEqual(decision["action"], "reuse")
        self.assertEqual(decision["runtime"]["id"], "runtime-8775")

    def test_port_match_includes_external(self) -> None:
        runtimes = [
            {"id": "external-8775", "alive": True, "username": None, "envPort": 8775,
             "external": True, "command": []},
        ]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, None)
        self.assertEqual(decision["action"], "reuse")
        self.assertTrue(decision["runtime"]["external"])

    def test_missing_runtime_starts_new(self) -> None:
        decision = choose_runtime([], "Glad_Bot_1", 8775, None)
        self.assertEqual(decision["action"], "start")
        self.assertIsNone(decision["runtime"])

    def test_brain_requires_restart_when_argv_has_none(self) -> None:
        runtimes = [{"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1",
                     "envPort": 8775, "external": False,
                     "command": ["node", "src/index.js", "--username", "Glad_Bot_1"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "dagger-1118b")
        self.assertEqual(decision["action"], "restart")
        self.assertIn("大脑模式", decision["reason"])

    def test_brain_same_model_is_reused(self) -> None:
        runtimes = [{"id": "runtime-8775", "alive": True, "username": "Glad_Bot_1",
                     "envPort": 8775, "external": False,
                     "command": ["node", "src/index.js", "--brain", "/m/dagger.npz"]}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "/m/dagger.npz")
        self.assertEqual(decision["action"], "reuse")

    def test_external_runtime_cannot_be_restarted(self) -> None:
        runtimes = [{"id": "external-8775", "alive": True, "username": None, "envPort": 8775,
                     "external": True, "command": []}]
        decision = choose_runtime(runtimes, "Glad_Bot_1", 8775, "dagger-1118b")
        self.assertEqual(decision["action"], "blocked")
        self.assertIn("外部", decision["reason"])

    def test_argv_brain_spec(self) -> None:
        self.assertEqual(argv_brain_spec(["node", "--brain", "/x.npz", "--host", "1"]), "/x.npz")
        self.assertIsNone(argv_brain_spec(["node", "src/index.js"]))
        self.assertIsNone(argv_brain_spec([]))


class JudgeRoundTests(unittest.TestCase):
    def test_kill_by_health_zero(self) -> None:
        verdict = judge_round({"health": 0.0}, {"health": 12.0}, 1.0, 120.0)
        self.assertEqual(verdict["winner"], "bot")
        self.assertEqual(verdict["reason"], "kill")
        verdict = judge_round({"health": 12.0}, {"health": 0.0}, 1.0, 120.0)
        self.assertEqual(verdict["winner"], "player")

    def test_offline_side_loses(self) -> None:
        self.assertEqual(judge_round(None, {"health": 20.0}, 0.0, 120.0)["winner"], "bot")
        self.assertEqual(judge_round({"health": 20.0}, None, 0.0, 120.0)["winner"], "player")

    def test_timeout_lower_health_loses(self) -> None:
        verdict = judge_round({"health": 14.0}, {"health": 6.0}, 120.0, 120.0)
        self.assertEqual(verdict["winner"], "player")
        self.assertEqual(verdict["reason"], "timeout")
        verdict = judge_round({"health": 3.0}, {"health": 15.0}, 121.0, 120.0)
        self.assertEqual(verdict["winner"], "bot")
        verdict = judge_round({"health": 9.0}, {"health": 9.0}, 121.0, 120.0)
        self.assertEqual(verdict["winner"], "draw")

    def test_still_fighting_returns_none(self) -> None:
        self.assertIsNone(judge_round({"health": 20.0}, {"health": 20.0}, 3.0, 120.0))

    def test_respawn_is_detected_as_death(self) -> None:
        verdict = judge_round({"health": 20.0}, {"health": 20.0}, 5.0, 120.0,
                              prev_player={"health": 3.0}, prev_bot={"health": 20.0})
        self.assertEqual(verdict["winner"], "bot")
        self.assertEqual(verdict["reason"], "player-respawn")


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


# --- 端到端（假 RCON + 假时钟） --------------------------------------------


class MatchFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_cfg(self.root)
        self.cfg.ensure_dirs()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _runner(self, script: list[dict], runtimes: FakeRuntimes | None = None,
                models: FakeModels | None = None) -> tuple[MatchRunner, FakeConn, FakeClock]:
        clock = FakeClock()
        conn = FakeConn(script, "Steve", "Glad_Bot_1")
        runner = MatchRunner(
            self.cfg, server=mock.Mock(), runtimes=runtimes or FakeRuntimes(), models=models,
            rcon_factory=lambda: conn, clock=clock.now, sleeper=clock.sleep,
        )
        return runner, conn, clock

    def _start_and_wait(self, runner: MatchRunner, payload: dict) -> dict:
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start(payload)
        thread = runner._thread
        self.assertIsNotNone(thread)
        thread.join(timeout=20)
        self.assertFalse(thread.is_alive(), "对局线程未在 20s 内结束")
        return runner.current()

    def test_rounds_progress_and_score(self) -> None:
        # 第 1 局：Bot 把玩家打死（玩家血量掉到 0）；第 2 局：双方僵持到超时，Bot 血更少。
        script = [
            {"player": 20.0, "bot": 20.0},
            {"player": 12.0, "bot": 18.0},
            {"player": 0.0, "bot": 18.0},
            {"player": 20.0, "bot": 20.0},
            {"player": 20.0, "bot": 12.0},
        ]
        runner, conn, _ = self._runner(script)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "bot": "Glad_Bot_1", "envPort": 8775,
            "rounds": 2, "roundSeconds": 10, "record": False,
        })
        self.assertEqual(snapshot["phase"], "finished")
        self.assertEqual(len(snapshot["roundsDetail"]), 2)
        first, second = snapshot["roundsDetail"]
        self.assertEqual(first["winner"], "bot")
        self.assertEqual(first["reason"], "kill")
        self.assertEqual(second["winner"], "player")
        self.assertEqual(second["reason"], "timeout")
        self.assertEqual(snapshot["score"], {"player": 1, "bot": 1, "draws": 0})
        self.assertEqual(snapshot["round"], 2)
        # spawn 模式：自己拉起的运行时在对局结束后必须停掉。
        self.assertEqual(runner.runtimes.stopped, ["runtime-8775"])

    def test_timeout_with_equal_health_is_draw(self) -> None:
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}])
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": False,
        })
        self.assertEqual(snapshot["roundsDetail"][0]["winner"], "draw")
        self.assertEqual(snapshot["score"]["draws"], 1)

    def test_attach_reuses_existing_runtime_without_starting(self) -> None:
        runtimes = FakeRuntimes([{
            "id": "external-8775", "pid": 20568, "envPort": 8775, "alive": True,
            "external": True, "username": None, "command": ["node", "src/index.js"],
        }])
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}], runtimes=runtimes)
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
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}], runtimes=runtimes,
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

    def test_attach_with_brain_on_external_runtime_is_rejected(self) -> None:
        model = self.root / "policy.npz"
        model.write_bytes(b"fake")
        runtimes = FakeRuntimes([{
            "id": "external-8775", "pid": 20568, "envPort": 8775, "alive": True,
            "external": True, "username": None, "command": ["node", "src/index.js"],
        }])
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}], runtimes=runtimes,
                                    models=FakeModels({"dagger-1118b": model}))
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            with self.assertRaises(Conflict):
                runner.start({
                    "mode": "attach", "player": "Steve", "envPort": 8775,
                    "rounds": 1, "roundSeconds": 10, "brain": "dagger-1118b",
                })

    def test_brain_model_without_weights_is_rejected(self) -> None:
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}],
                                    models=FakeModels({"empty": self.root / "missing.npz"}))
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            with self.assertRaises(BadRequest):
                runner.start({"player": "Steve", "brain": "empty"})

    def test_recording_writes_frames_meta_summary(self) -> None:
        script = [
            {"player": 20.0, "bot": 20.0},
            {"player": 18.0, "bot": 20.0},
            {"player": 10.0, "bot": 20.0},
            {"player": 0.0, "bot": 20.0},
        ]
        runner, _, _ = self._runner(script)
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": True,
            "spawns": {"player": [0, 4, 0], "bot": [6, 4, 0]},
        })
        record_dir = Path(snapshot["recordDir"])
        self.assertTrue(record_dir.is_dir(), record_dir)
        self.assertIn("matches", record_dir.parts)
        self.assertRegex(record_dir.name, r"^\d{8}-\d{6}-Steve-vs-Glad_Bot_1$")

        frames_path = record_dir / "frames.jsonl"
        self.assertTrue(frames_path.is_file())
        lines = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(lines, "frames.jsonl 不能为空")
        frame = lines[0]
        for key in ("t", "round", "player", "bot"):
            self.assertIn(key, frame)
        for key in ("health", "food", "saturation", "armor", "x", "y", "z",
                    "yaw", "pitch", "sprinting", "blocking", "onGround", "hurt"):
            self.assertIn(key, frame["player"])
            self.assertIn(key, frame["bot"])
        self.assertEqual(frame["round"], 1)

        meta = json.loads((record_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["plan"]["player"], "Steve")
        self.assertEqual(meta["frameHz"], 10.0)
        summary = json.loads((record_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["score"]["bot"], 1)
        self.assertEqual(len(summary["rounds"]), 1)
        self.assertEqual(summary["rounds"][0]["player"]["healthAfter"], 0.0)

    def test_train_after_records_intent(self) -> None:
        runner, _, _ = self._runner([{"player": 0.0, "bot": 20.0}])
        snapshot = self._start_and_wait(runner, {
            "mode": "spawn", "player": "Steve", "rounds": 1, "roundSeconds": 10,
            "record": True, "trainAfter": True,
        })
        self.assertTrue(snapshot["trainAfter"])
        # 测试环境没有 training/scripts/import_human_match.py，编排器应记录跳过而不是失败。
        log_text = " ".join(entry["text"] for entry in snapshot["log"])
        self.assertIn("跳过", log_text)

    def test_stop_marks_phase_stopped_and_cleans_runtime(self) -> None:
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}])
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start({"mode": "spawn", "player": "Steve", "rounds": 5,
                          "roundSeconds": 600, "record": False, "envPort": 8775})
            snapshot = runner.stop()
        self.assertEqual(snapshot["phase"], "stopped")
        self.assertFalse(snapshot["running"])
        self.assertIn("runtime-8775", runner.runtimes.stopped)

    def test_second_start_is_rejected_while_running(self) -> None:
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}])
        with mock.patch("match_runner.is_port_open", return_value=True), \
                mock.patch("match_runner.is_port_free", return_value=True):
            runner.start({"mode": "spawn", "player": "Steve", "rounds": 5,
                          "roundSeconds": 600, "record": False})
            with self.assertRaises(Conflict):
                runner.start({"mode": "spawn", "player": "Steve", "rounds": 1})
            runner.stop()

    def test_idle_snapshot_shape(self) -> None:
        runner, _, _ = self._runner([{"player": 20.0, "bot": 20.0}])
        snapshot = runner.current()
        self.assertEqual(snapshot["phase"], "idle")
        self.assertFalse(snapshot["running"])
        for key in ("phase", "round", "rounds", "score", "roundsDetail", "brain"):
            self.assertIn(key, snapshot)


if __name__ == "__main__":
    unittest.main()
