"""板子判定（G-715）的纯函数测试。

只测「没有 RCON / 没有时钟」的部分：`/gladiator board status` 的解析、阶段映射、
`board start` 命令拼装、帧内伤害（rewards）计算、以及把连续 status 聚合成逐局结果的
`BoardTracker`。端到端流程在 `test_match_plan.py`。

运行：
    cd D:\\cbot-gladiator\\dashboard
    ..\\training\\.venv\\Scripts\\python.exe -m unittest discover -s tests -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from match_runner import (  # noqa: E402
    BOARD_ID,
    BoardTracker,
    MatchPlan,
    board_finished,
    board_start_command,
    frame_rewards,
    map_board_phase,
    parse_board_status,
)

IDLE_LINE = ("session=none board= round=0 rounds=0 phase=idle score=0:0 draws=0"
             " reason=none player= bot=")


class BoardStatusParseTests(unittest.TestCase):
    def test_idle_line_without_session(self) -> None:
        status = parse_board_status(IDLE_LINE)
        self.assertIsNotNone(status)
        self.assertIsNone(status["session"])
        self.assertEqual(status["board"], "")
        self.assertEqual(status["round"], 0)
        self.assertEqual(status["rounds"], 0)
        self.assertEqual(status["phase"], "idle")
        self.assertEqual(status["score"], {"player": 0, "bot": 0, "draws": 0})
        self.assertIsNone(status["reason"])
        self.assertEqual(status["player"], "")
        self.assertEqual(status["bot"], "")

    def test_running_line(self) -> None:
        status = parse_board_status(
            "session=3f9c1a board=flat-1v1 round=2 rounds=3 phase=fighting"
            " score=1:1 draws=1 reason=kill player=Steve bot=Glad_Bot_1")
        self.assertEqual(status["session"], "3f9c1a")
        self.assertEqual(status["board"], "flat-1v1")
        self.assertEqual(status["round"], 2)
        self.assertEqual(status["rounds"], 3)
        self.assertEqual(status["phase"], "fighting")
        self.assertEqual(status["score"], {"player": 1, "bot": 1, "draws": 1})
        self.assertEqual(status["reason"], "kill")
        self.assertEqual(status["player"], "Steve")
        self.assertEqual(status["bot"], "Glad_Bot_1")

    def test_every_phase_is_accepted(self) -> None:
        for phase in ("idle", "countdown", "fighting", "ended"):
            status = parse_board_status(
                f"session=s1 board=flat-1v1 round=1 rounds=1 phase={phase}"
                " score=0:0 draws=0 reason=none player=Steve bot=Glad_Bot_1")
            self.assertIsNotNone(status, phase)
            self.assertEqual(status["phase"], phase)

    def test_score_parsing_is_not_digit_limited(self) -> None:
        status = parse_board_status(
            "session=s1 board=flat-1v1 round=10 rounds=12 phase=fighting"
            " score=11:9 draws=7 reason=none player=Steve bot=Glad_Bot_1")
        self.assertEqual(status["score"], {"player": 11, "bot": 9, "draws": 7})

    def test_reason_none_becomes_null(self) -> None:
        status = parse_board_status(
            "session=s1 board=flat-1v1 round=1 rounds=1 phase=fighting"
            " score=0:0 draws=0 reason=none player=Steve bot=Glad_Bot_1")
        self.assertIsNone(status["reason"])

    def test_reason_with_spaces_is_kept_whole(self) -> None:
        status = parse_board_status(
            "session=s1 board=flat-1v1 round=1 rounds=1 phase=ended"
            " score=0:1 draws=0 reason=bot offline during round player=Steve bot=Glad_Bot_1")
        self.assertEqual(status["reason"], "bot offline during round")

    def test_colour_codes_are_stripped(self) -> None:
        status = parse_board_status(
            "§a session=none §r board= round=0 rounds=0 phase=idle score=0:0 draws=0"
            " reason=none player= bot=")
        self.assertIsNotNone(status)
        self.assertIsNone(status["session"])

    def test_status_is_picked_out_of_noisy_output(self) -> None:
        raw = ("[Gladiator] board status\n" + IDLE_LINE + "\nsome trailing line")
        status = parse_board_status(raw)
        self.assertIsNotNone(status)
        self.assertEqual(status["phase"], "idle")

    def test_reordered_keys_fall_back_to_token_parse(self) -> None:
        status = parse_board_status(
            "phase=fighting session=s2 round=1 rounds=1 score=2:3 draws=1"
            " reason=kill player=Steve bot=Glad_Bot_1")
        self.assertIsNotNone(status)
        self.assertEqual(status["session"], "s2")
        self.assertEqual(status["score"], {"player": 2, "bot": 3, "draws": 1})
        self.assertEqual(status["phase"], "fighting")

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(parse_board_status(""))
        self.assertIsNone(parse_board_status("这不是 status"))
        self.assertIsNone(parse_board_status("session=s1 缺少 phase"))

    def test_real_server_lines_use_dash_placeholders(self) -> None:
        """服务端实际输出（`GladiatorCommand#boardStatusLine`）：空位用 `-`，对局中 reason=running。"""
        idle = parse_board_status(
            "[Gladiator] session=none board=- round=0 rounds=0 phase=idle score=0:0 draws=0"
            " reason=none player=- bot=-")
        self.assertIsNone(idle["session"])
        self.assertEqual(idle["board"], "")
        self.assertEqual(idle["player"], "")
        self.assertEqual(idle["bot"], "")

        running = parse_board_status(
            "[Gladiator] session=9f2c board=flat-1v1 round=2 rounds=3 phase=fighting"
            " score=1:0 draws=0 reason=running player=Steve bot=Glad_Bot_1")
        self.assertEqual(running["session"], "9f2c")
        self.assertEqual(running["phase"], "fighting")
        self.assertEqual(running["reason"], "running")
        self.assertEqual(running["score"], {"player": 1, "bot": 0, "draws": 0})

    def test_starting_phase_from_board(self) -> None:
        status = parse_board_status(
            "[Gladiator] session=9f2c board=flat-1v1 round=1 rounds=1 phase=starting"
            " score=0:0 draws=0 reason=running player=Steve bot=Glad_Bot_1")
        self.assertEqual(status["phase"], "starting")
        self.assertEqual(map_board_phase(status["phase"]), "preparing")

    def test_default_snapshot_board_id_matches_contract(self) -> None:
        self.assertEqual(BOARD_ID, "flat-1v1")


class BoardPhaseTests(unittest.TestCase):
    def test_phase_mapping_keeps_dashboard_vocabulary(self) -> None:
        self.assertEqual(map_board_phase("idle"), "preparing")
        self.assertEqual(map_board_phase("starting"), "preparing")
        self.assertEqual(map_board_phase("countdown"), "countdown")
        self.assertEqual(map_board_phase("fighting"), "fighting")
        self.assertEqual(map_board_phase("ended"), "finished")
        self.assertEqual(map_board_phase(""), "preparing")
        # 未知阶段原样透出（前端按原文显示，不静默吞掉新阶段）。
        self.assertEqual(map_board_phase("sudden-death"), "sudden-death")

    def test_board_finished_only_on_ended_or_no_session(self) -> None:
        running = {"session": "s1", "phase": "fighting"}
        self.assertFalse(board_finished(running))
        self.assertFalse(board_finished({"session": "s1", "phase": "countdown"}))
        self.assertTrue(board_finished({"session": "s1", "phase": "ended"}))
        self.assertTrue(board_finished({"session": None, "phase": "idle"}))
        self.assertTrue(board_finished({"session": None, "phase": ""}))
        self.assertFalse(board_finished(None))


class BoardStartCommandTests(unittest.TestCase):
    def test_without_spawns(self) -> None:
        plan = MatchPlan(player="Steve", bot="Glad_Bot_1", rounds=3, round_seconds=120.0)
        self.assertEqual(board_start_command(plan),
                         "gladiator board start flat-1v1 Steve Glad_Bot_1 3 120 kit=classic")

    def test_with_both_spawns(self) -> None:
        plan = MatchPlan(player="Steve", bot="Glad_Bot_1", rounds=2, round_seconds=90.0,
                         spawns={"Steve": (-4.0, 4.0, 0.5),
                                 "Glad_Bot_1": (4.0, 4.0, 0.5)})
        self.assertEqual(board_start_command(plan),
                         "gladiator board start flat-1v1 Steve Glad_Bot_1 2 90"
                         " -4.00 4.00 0.50 4.00 4.00 0.50 kit=classic")

    def test_partial_spawns_are_not_sent(self) -> None:
        # 只有一侧有坐标时不带 6 个数：缺一半的坐标会把对手丢到板子的默认点。
        plan = MatchPlan(player="Steve", bot="Glad_Bot_1", rounds=1, round_seconds=30.0,
                         spawns={"Steve": (0.0, 4.0, 0.0)})
        self.assertEqual(board_start_command(plan),
                         "gladiator board start flat-1v1 Steve Glad_Bot_1 1 30 kit=classic")

    def test_custom_board_id_and_integer_seconds(self) -> None:
        # 服务端按 `parseInt` 解析秒数：这里必须发整数，不能把 12.4 原样发过去。
        plan = MatchPlan(player="Steve", bot="Glad_Bot_1", rounds=1, round_seconds=12.4)
        self.assertEqual(board_start_command(plan, "arena-2v2"),
                         "gladiator board start arena-2v2 Steve Glad_Bot_1 1 12 kit=classic")


class FrameRewardsTests(unittest.TestCase):
    def test_damage_from_health_drop(self) -> None:
        rewards = frame_rewards({"health": 20.0}, {"health": 20.0},
                                {"health": 15.0}, {"health": 12.0})
        # playerDamage = Bot 掉的血；botDamage = 玩家掉的血。
        self.assertEqual(rewards, {"playerDamage": 8.0, "botDamage": 5.0})

    def test_no_change_is_zero(self) -> None:
        self.assertEqual(frame_rewards({"health": 20.0}, {"health": 20.0},
                                       {"health": 20.0}, {"health": 20.0}),
                         {"playerDamage": 0.0, "botDamage": 0.0})

    def test_healing_is_not_negative_damage(self) -> None:
        rewards = frame_rewards({"health": 5.0}, {"health": 3.0},
                                {"health": 20.0}, {"health": 20.0})
        self.assertEqual(rewards, {"playerDamage": 0.0, "botDamage": 0.0})

    def test_missing_stats_give_zero(self) -> None:
        self.assertEqual(frame_rewards(None, None, {"health": 10.0}, {"health": 10.0}),
                         {"playerDamage": 0.0, "botDamage": 0.0})
        self.assertEqual(frame_rewards({"health": 20.0}, {"health": 20.0}, None, None),
                         {"playerDamage": 0.0, "botDamage": 0.0})

    def test_malformed_health_gives_zero(self) -> None:
        self.assertEqual(frame_rewards({"health": None}, {"name": "bot"},
                                       {"health": 10.0}, {"health": 10.0}),
                         {"playerDamage": 0.0, "botDamage": 0.0})

    def test_partial_damage_only_on_one_side(self) -> None:
        rewards = frame_rewards({"health": 20.0}, {"health": 20.0},
                                {"health": 20.0}, {"health": 0.0})
        self.assertEqual(rewards, {"playerDamage": 20.0, "botDamage": 0.0})


class BoardTrackerTests(unittest.TestCase):
    def _status(self, *, score=(0, 0), draws=0, round_no=1, phase="fighting",
                reason=None, session="s1") -> dict:
        return {
            "session": session,
            "board": BOARD_ID,
            "round": round_no,
            "rounds": 3,
            "phase": phase,
            "score": {"player": score[0], "bot": score[1], "draws": draws},
            "reason": reason,
            "player": "Steve",
            "bot": "Glad_Bot_1",
        }

    def test_score_delta_becomes_round_entry(self) -> None:
        tracker = BoardTracker()
        self.assertEqual(tracker.update(self._status(), 0.0), [])
        entries = tracker.update(self._status(score=(1, 0), reason="kill"), 42.0)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["round"], 1)
        self.assertEqual(entries[0]["winner"], "player")
        self.assertEqual(entries[0]["reason"], "kill")
        self.assertEqual(entries[0]["seconds"], 42.0)
        # 向后兼容字段：duration 与 seconds 同义。
        self.assertEqual(entries[0]["duration"], entries[0]["seconds"])
        self.assertEqual(tracker.score, {"player": 1, "bot": 0, "draws": 0})

    def test_repeated_status_does_not_duplicate_entries(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        tracker.update(self._status(score=(0, 1)), 5.0)
        self.assertEqual(tracker.update(self._status(score=(0, 1)), 6.0), [])
        self.assertEqual(tracker.update(self._status(score=(0, 1)), 7.0), [])
        self.assertEqual(len(tracker.entries), 1)

    def test_draw_increment_counts_as_draw(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        entries = tracker.update(self._status(draws=1, reason="timeout"), 60.0)
        self.assertEqual([item["winner"] for item in entries], ["draw"])
        self.assertEqual(entries[0]["reason"], "timeout")

    def test_two_rounds_between_polls_are_both_recorded(self) -> None:
        # 板子 500ms 内连打两局（秒杀）：比分 +2 时必须记两局，不能吞掉一局。
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        entries = tracker.update(self._status(score=(2, 0)), 3.0)
        self.assertEqual([item["round"] for item in entries], [1, 2])
        self.assertEqual([item["winner"] for item in entries], ["player", "player"])

    def test_merged_round_keeps_next_round_clock(self) -> None:
        # 真机复现过的坑：一次采样里「比分 +1」且局号已经跳到下一局（板子 500ms 内就换局），
        # 下一局的秒表必须在同一次采样里重新起算，否则第 2 局会记成 0 秒。
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        entries = tracker.update(self._status(draws=1, round_no=2), 25.3)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["seconds"], 25.3)
        # 下一次采样（同一局）应能从 25.3 起算，而不是 None/0。
        self.assertAlmostEqual(tracker.round_elapsed(30.3), 5.0)

    def test_round_advance_restarts_the_clock(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        tracker.update(self._status(score=(1, 0)), 30.0)
        tracker.update(self._status(score=(1, 0), round_no=2), 31.0)
        self.assertAlmostEqual(tracker.round_elapsed(33.0), 2.0)
        entries = tracker.update(self._status(score=(1, 1), round_no=2), 51.0)
        self.assertEqual(entries[0]["round"], 2)
        self.assertEqual(entries[0]["seconds"], 20.0)

    def test_reason_is_sticky_until_the_board_reports_a_new_one(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(reason="kill"), 0.0)
        # 板子随后把 reason 清成 none：本局仍记最近一次非 none 的原因。
        entries = tracker.update(self._status(score=(1, 0), reason=None), 10.0)
        self.assertEqual(entries[0]["reason"], "kill")

    def test_unknown_reason_when_board_never_reports_one(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(), 0.0)
        entries = tracker.update(self._status(score=(0, 1)), 10.0)
        self.assertEqual(entries[0]["reason"], "unknown")

    def test_running_reason_is_ignored(self) -> None:
        # 服务端对局中固定 reason=running：不能把它当成「本局为什么结束」。
        tracker = BoardTracker()
        tracker.update(self._status(reason="running"), 0.0)
        entries = tracker.update(self._status(score=(1, 0), reason="running"), 20.0)
        self.assertEqual(entries[0]["reason"], "unknown")
        # 结束那一刻才拿到会话级 lastReason（如 rounds-finished），此时若有增量才会用它。
        entries = tracker.update(self._status(score=(1, 0), phase="ended",
                                              reason="rounds-finished"), 21.0)
        self.assertEqual(entries, [])
        self.assertEqual(tracker.reason, "rounds-finished")

    def test_seen_session_flag(self) -> None:
        tracker = BoardTracker()
        tracker.update(self._status(session=None), 0.0)
        self.assertFalse(tracker.seen_session)
        tracker.update(self._status(session="s1"), 0.5)
        self.assertTrue(tracker.seen_session)


class RoundReasonTests(unittest.TestCase):
    """逐局 reason 只能靠时长推断（status 的 reason 是会话级）。"""

    def _tracker(self, round_seconds: float = 20.0) -> BoardTracker:
        tracker = BoardTracker()
        tracker.round_seconds = round_seconds
        return tracker

    def _status(self, round_no: int, player: int, bot: int, draws: int = 0,
                reason: str = "running") -> dict:
        return {
            "session": "s-1", "round": round_no, "rounds": 2, "phase": "fighting",
            "score": {"player": player, "bot": bot, "draws": draws},
            "reason": reason, "player": "P", "bot": "B",
        }

    def test_kill_when_round_ends_early(self) -> None:
        tracker = self._tracker(20.0)
        tracker.update(self._status(1, 0, 0), 100.0)
        entries = tracker.update(self._status(2, 0, 1), 108.0)  # 8 秒就分胜负
        self.assertEqual(entries[0]["winner"], "bot")
        self.assertEqual(entries[0]["reason"], "kill")

    def test_timeout_when_round_runs_full(self) -> None:
        tracker = self._tracker(20.0)
        tracker.update(self._status(1, 0, 0), 100.0)
        entries = tracker.update(self._status(1, 0, 0, draws=1), 120.4)
        self.assertEqual(entries[0]["winner"], "draw")
        self.assertEqual(entries[0]["reason"], "timeout")

    def test_session_reason_kept_when_no_clock(self) -> None:
        tracker = self._tracker(20.0)
        entries = tracker.update(self._status(1, 1, 0, reason="rounds-finished"), 100.0)
        # 没有秒表就别猜「击杀」：保留服务端给的会话级原因。
        self.assertEqual(entries[0]["reason"], "rounds-finished")


if __name__ == "__main__":
    unittest.main()
