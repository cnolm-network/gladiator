"""G-736：`--records`（Arena Recorder 权威伤害来源）的单测。

全部用**合成 JSONL**：录制格式（`{seq, at, category, from, content}` + `key=value` 文本）
与 `frames.jsonl`（10Hz 采样、相对秒 `t`）都在本地造，不依赖真机产物。

覆盖的判据（对应任务书 10 条）：
落帧 / 同帧累加 / 边界归属后一帧 / 大小写不敏感 / 无法匹配的攻击者计数 /
击杀只加末帧 / 路径不存在与零事件都要回退 / 不给 `--records` 时逐位一致 /
`damageDealt/damageTaken` 与 `obs` 等长。另外补了「目录里挑对文件」「时间基准优先级」
「首帧之前的事件钳到首帧」三条真机上真会遇到的边界。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

import import_human_match as ihm  # noqa: E402

PLAYER = "Glad_P9"
BOT = "Glad_M3"
ORIGIN_MS = 1_795_807_974_138          # 与真机样例同一个量级的毫秒时间戳


# ---------------------------------------------------------------- 合成数据

def make_frame(t: float, *, round_no: int = 1, player_health: float = 20.0,
               bot_health: float = 20.0) -> dict:
    """一帧 10Hz 采样：字段与 `dashboard` 录制的真实 frames.jsonl 同形。"""
    body = {
        "health": 20.0, "food": 20.0, "saturation": 5.0, "armor": 20.0,
        "x": 0.0, "y": 4.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0,
        "sprinting": False, "blocking": False, "onGround": True, "hurt": 0.0,
    }
    player = dict(body, health=player_health)
    bot = dict(body, health=bot_health, x=3.0)
    return {"t": t, "round": round_no, "player": player, "bot": bot,
            "rewards": {"playerDamage": 0.0, "botDamage": 0.0}}


def timeline(count: int = 20, dt: float = 0.1) -> list[dict]:
    """t = 0.1, 0.2, … 的等间隔帧（与真机 10Hz 一致）。"""
    return [make_frame(round((index + 1) * dt, 3)) for index in range(count)]


def damage_row(at_ms: int, victim: str = PLAYER, attacker: str = BOT,
               damage: float = 1.6, seq: int = 1) -> dict:
    return {
        "seq": seq, "at": at_ms, "category": "event",
        "from": "EventHandler:top.cnolm.arena.simplify.FlatManagerSimplified",
        "content": f"victim={victim} attacker={attacker} damage={damage} cause=ENTITY_ATTACK",
    }


def kill_row(at_ms: int, victim: str = PLAYER, killer: str = BOT, seq: int = 2) -> dict:
    return {
        "seq": seq, "at": at_ms, "category": "event",
        "from": "EventHandler:top.cnolm.arena.simplify.FlatManagerSimplified",
        "content": f"victim={victim} killer={killer} cause=KILL",
    }


def setup_row(at_ms: int = ORIGIN_MS) -> dict:
    return {
        "seq": 0, "at": at_ms, "category": "event",
        "from": "Custom:top.cnolm.arena.simplify.FlatManagerSimplified",
        "content": "kit=classic shield=3 period=2",
    }


def position_row(at_ms: int = ORIGIN_MS) -> dict:
    """真实录制里最多的 `position` 行——解析时必须整条忽略。"""
    return {
        "seq": 3, "at": at_ms, "category": "position",
        "from": "PositionTracker:top.cnolm.arena.simplify.FlatManagerSimplified",
        "content": "{loc=Location{world=world,x=0.0,y=4.0,z=0.0}}",
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    return path


def write_match(root: Path, frames: list[dict], *, board_session: str | None = None,
                started_at: str = "2026-11-28T03:32:54") -> Path:
    match_dir = root / "match"
    match_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(match_dir / "frames.jsonl", frames)
    meta = {"schema": "gladiator.match/v1", "startedAt": started_at,
            "plan": {"player": PLAYER, "bot": BOT, "roundSeconds": 20.0}}
    if board_session is not None:
        meta["boardSession"] = board_session
    (match_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return match_dir


def t_ms(seconds: float) -> int:
    """相对秒 → 绝对毫秒（本文件的测试都用 ORIGIN_MS 当 t=0）。"""
    return int(ORIGIN_MS + round(seconds * 1000))


def load_arena(tmp: Path, frames: list[dict], records: Path, *, win_bonus: float = 0.0,
               board_session: str | None = f"flat-1v1-{ORIGIN_MS}") -> dict:
    match_dir = write_match(tmp, frames, board_session=board_session)
    meta = ihm.load_meta(match_dir)
    return ihm.load_arena_records(records, frames=frames, meta=meta,
                                  player_name=PLAYER, bot_name=BOT,
                                  win_bonus=win_bonus)


# ---------------------------------------------------------------- 用例

class AlignmentTest(unittest.TestCase):
    """事件 → 帧的落位规则。"""

    def test_single_damage_lands_in_covering_frame(self) -> None:
        frames = timeline(20)                     # t = 0.1 … 2.0
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                setup_row(), damage_row(t_ms(1.25), victim=BOT, attacker=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertIsNotNone(arena)
        dealt = arena["damageDealt"]
        self.assertEqual(int(np.count_nonzero(dealt)), 1)
        # t=1.25 落在 [1.2, 1.3) → 下标 11（帧序列 0.1 起、步长 0.1）
        self.assertEqual(int(np.argmax(dealt)), 11)
        self.assertAlmostEqual(float(dealt[11]), 1.6, places=6)
        self.assertAlmostEqual(float(arena["rewards"][11]), 1.6, places=6)

    def test_same_frame_damage_accumulates(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(1.21), victim=BOT, attacker=PLAYER, seq=1),
                damage_row(t_ms(1.24), victim=BOT, attacker=PLAYER, seq=2),
                damage_row(t_ms(1.28), victim=BOT, attacker=PLAYER, seq=3),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertAlmostEqual(float(arena["damageDealt"][11]), 4.8, places=6)
        self.assertAlmostEqual(float(arena["rewards"][11]), 4.8, places=6)
        self.assertEqual(arena["stats"]["damageEvents"], 3)

    def test_event_on_boundary_belongs_to_next_frame(self) -> None:
        """事件恰好落在 `t+dt`：按 [t, t+dt) 应归**后一帧**（1.3 → 下标 12）。"""
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(1.30), victim=BOT, attacker=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertEqual(int(np.argmax(arena["damageDealt"])), 12)

    def test_event_before_first_frame_clamps_to_first(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(-0.4), victim=BOT, attacker=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertEqual(float(arena["damageDealt"][0]), 1.6)
        self.assertEqual(arena["stats"]["beforeFirstFrame"], 1)

    def test_event_after_last_frame_clamps_to_last(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(9.0), victim=BOT, attacker=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertEqual(float(arena["damageDealt"][-1]), 1.6)
        self.assertEqual(arena["stats"]["afterLastFrame"], 1)


class AttributionTest(unittest.TestCase):
    """谁打谁：大小写、无法归属、未知受害者。"""

    def test_names_are_case_insensitive(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(1.25), victim=BOT.upper(), attacker=PLAYER.lower()),
                damage_row(t_ms(1.45), victim=PLAYER.lower(), attacker=BOT.upper(), seq=2),
            ])
            arena = load_arena(Path(tmp), frames, records)

        self.assertAlmostEqual(float(arena["damageDealt"][11]), 1.6, places=6)
        self.assertAlmostEqual(float(arena["damageTaken"][13]), 1.6, places=6)
        self.assertEqual(arena["stats"]["unattributed"], 0)
        # 一进一出，净回报为 0
        self.assertAlmostEqual(float(arena["rewards"].sum()), 0.0, places=6)

    def test_unknown_attacker_is_unattributed_and_excluded(self) -> None:
        """纯对齐函数：别人的伤害只计数，**不进** rewards。"""
        frames = timeline(20)
        events = [{"at": t_ms(1.25), "kind": "damage", "victim": PLAYER,
                   "attacker": "Random_Stranger", "damage": 3.0, "cause": "ENTITY_ATTACK"}]
        aligned = ihm.align_record_rewards(
            frames, events, origin_ms=float(ORIGIN_MS),
            player_name=PLAYER, bot_name=BOT)

        self.assertEqual(aligned["stats"]["unattributed"], 1)
        self.assertAlmostEqual(aligned["stats"]["unattributedDamage"], 3.0, places=6)
        self.assertAlmostEqual(float(np.abs(aligned["rewards"]).sum()), 0.0, places=6,
                               msg="无法归属的伤害不能进 rewards")

    def test_pipeline_falls_back_when_nothing_is_attributable(self) -> None:
        """一条都归不到（名字对不上）→ 必须回退，不能静默产出全 0。"""
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(1.25), victim=PLAYER, attacker="Random_Stranger"),
            ])
            warnings = io.StringIO()
            with contextlib.redirect_stderr(warnings):
                arena = load_arena(Path(tmp), frames, records)
        self.assertIsNone(arena)
        self.assertIn("没有一条能归属", warnings.getvalue())

    def test_damage_without_attacker_is_unattributed(self) -> None:
        """摔伤/虚空：录制里只有 `victim=… damage=… cause=FALL`，没有 attacker。"""
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            row = {"seq": 9, "at": t_ms(1.25), "category": "event", "from": "EventHandler:x",
                   "content": f"victim={PLAYER} damage=2.0 cause=FALL"}
            records = write_jsonl(Path(tmp) / "r.jsonl", [row])
            arena = load_arena(Path(tmp), frames, records)

        # 全部无法归属 → load_arena_records 判定「没有可用事件」→ 回退
        self.assertIsNone(arena)

    def test_kill_bonus_only_on_last_frame_of_that_round(self) -> None:
        frames = [make_frame(0.1 + index * 0.1, round_no=1) for index in range(5)]
        frames += [make_frame(0.6 + index * 0.1, round_no=2) for index in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                # 一条可归属伤害：否则「全线归不到」会触发回退（见上一个用例）
                damage_row(t_ms(0.15), victim=BOT, attacker=PLAYER, damage=1.0),
                kill_row(t_ms(0.45), victim=BOT, killer=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, records, win_bonus=5.0)

        rewards = arena["rewards"]
        self.assertAlmostEqual(float(rewards[4]), 5.0, places=6, msg="第 1 局末帧拿到 bonus")
        self.assertEqual(int(np.count_nonzero(rewards[5:])), 0, "其它局/帧不该有 bonus")
        self.assertAlmostEqual(arena["stats"]["killBonus"], 5.0, places=6)

    def test_kill_by_bot_gives_player_no_bonus(self) -> None:
        frames = timeline(10)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(0.15), victim=BOT, attacker=PLAYER, damage=1.0),
                kill_row(t_ms(0.55), victim=PLAYER, killer=BOT),
            ])
            arena = load_arena(Path(tmp), frames, records, win_bonus=5.0)

        # 只留下那 1.0 的伤害，没有击杀 bonus
        self.assertAlmostEqual(float(np.sum(arena["rewards"])), 1.0, places=6)
        self.assertAlmostEqual(arena["stats"]["killBonus"], 0.0, places=6)


class SourceSelectionTest(unittest.TestCase):
    """时间基准与目录挑文件。"""

    def test_origin_prefers_board_session(self) -> None:
        origin, source = ihm.resolve_record_origin({"boardSession": f"flat-1v1-{ORIGIN_MS}"})
        self.assertEqual(origin, float(ORIGIN_MS))
        self.assertEqual(source, "boardSession")

    def test_origin_falls_back_to_started_at(self) -> None:
        origin, source = ihm.resolve_record_origin({"startedAt": "2026-11-28T03:32:54"})
        self.assertEqual(source, "startedAt")
        self.assertIsNotNone(origin)
        self.assertGreater(origin, 1.7e12)

    def test_missing_board_session_cannot_align(self) -> None:
        """两个时间基准都没有 → 回退，且**不**产出 arena-record。"""
        frames = timeline(10)
        with tempfile.TemporaryDirectory() as tmp:
            match_dir = tmp
            frames_path = Path(tmp) / "frames.jsonl"
            write_jsonl(frames_path, frames)
            (Path(tmp) / "meta.json").write_text(json.dumps({"plan": {}}), encoding="utf-8")
            records = write_jsonl(Path(tmp) / "rec" / "r.jsonl", [damage_row(t_ms(1.0))])
            warnings = io.StringIO()
            with contextlib.redirect_stderr(warnings):
                arena = ihm.load_arena_records(records, frames=frames,
                                               meta=ihm.load_meta(Path(match_dir)),
                                               player_name=PLAYER, bot_name=BOT)
        self.assertIsNone(arena)
        self.assertIn("时间基准", warnings.getvalue())

    def test_directory_picks_only_this_match(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp) / "records"
            write_jsonl(rec_dir / "other-people.jsonl", [
                damage_row(t_ms(1.25), victim="Stranger_A", attacker="Stranger_B"),
            ])
            write_jsonl(rec_dir / "too-late.jsonl", [
                damage_row(t_ms(500.0), victim=PLAYER, attacker=BOT),
            ])
            write_jsonl(rec_dir / "this-match.jsonl", [
                damage_row(t_ms(1.25), victim=BOT, attacker=PLAYER),
            ])
            arena = load_arena(Path(tmp), frames, rec_dir)

        self.assertIsNotNone(arena)
        self.assertEqual([Path(item).name for item in arena["files"]], ["this-match.jsonl"])
        self.assertAlmostEqual(float(arena["damageDealt"].sum()), 1.6, places=6)


class DatasetIntegrationTest(unittest.TestCase):
    """与 build_dataset / CLI 的接线。"""

    def test_without_records_is_bit_identical_to_health_delta(self) -> None:
        frames = timeline(20)
        frames[5]["bot"]["health"] = 18.4        # 第 6 帧对手掉 1.6
        frames[9]["player"]["health"] = 18.4     # 第 10 帧自己掉 1.6
        dataset = ihm.build_dataset(frames, player_name=PLAYER, bot_name=BOT,
                                    round_seconds=20.0)
        # 旧口径：相邻帧血量差（对手掉血 − 自己掉血）
        expected = np.zeros(len(frames), dtype=np.float32)
        expected[5] = 1.6
        expected[9] = -1.6
        np.testing.assert_allclose(dataset["rewards"], expected, rtol=0, atol=1e-6)
        self.assertNotIn("damageDealt", dataset, "不给 --records 时不该多出键")
        self.assertNotIn("damageTaken", dataset)

    def test_damage_arrays_match_obs_length(self) -> None:
        frames = timeline(30)
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [damage_row(t_ms(1.25), victim=BOT,
                                                                      attacker=PLAYER)])
            arena = load_arena(Path(tmp), frames, records)
        dataset = ihm.build_dataset(frames, player_name=PLAYER, bot_name=BOT,
                                    round_seconds=20.0, arena=arena)
        self.assertEqual(len(dataset["obs"]), len(frames))
        self.assertEqual(len(dataset["damageDealt"]), len(dataset["obs"]))
        self.assertEqual(len(dataset["damageTaken"]), len(dataset["obs"]))
        self.assertEqual(len(dataset["rewards"]), len(dataset["obs"]))

    def test_arena_rewards_override_health_delta(self) -> None:
        """血量差因「吃金苹果」漏掉的那一刀，事件来源必须还在。"""
        frames = timeline(20)
        for frame in frames[10:]:                # 第 11 帧起血量一直没掉（被治疗/采样吞了）
            frame["player"]["health"] = 20.0
        with tempfile.TemporaryDirectory() as tmp:
            records = write_jsonl(Path(tmp) / "r.jsonl", [
                damage_row(t_ms(1.25), victim=PLAYER, attacker=BOT, damage=1.6),
            ])
            arena = load_arena(Path(tmp), frames, records)
        dataset = ihm.build_dataset(frames, player_name=PLAYER, bot_name=BOT,
                                    round_seconds=20.0, arena=arena)
        self.assertAlmostEqual(float(dataset["rewards"].sum()), -1.6, places=6,
                               msg="血量差看不出这一刀，事件来源必须记上")

    def test_cli_falls_back_when_path_missing(self) -> None:
        frames = timeline(10)
        with tempfile.TemporaryDirectory() as tmp:
            match_dir = write_match(Path(tmp), frames, board_session=f"flat-1v1-{ORIGIN_MS}")
            out_dir = Path(tmp) / "out"
            warnings = io.StringIO()
            with contextlib.redirect_stderr(warnings):
                code = ihm.main([str(match_dir), "--out", str(out_dir),
                                 "--records", str(Path(tmp) / "nope")])
            self.assertEqual(code, 0)
            self.assertIn("不存在", warnings.getvalue())
            with np.load(out_dir / "dataset.npz") as data:
                meta = json.loads(str(data["meta"]))
            self.assertEqual(meta["rewardSource"], "health-delta")

    def test_cli_falls_back_when_no_events(self) -> None:
        frames = timeline(10)
        with tempfile.TemporaryDirectory() as tmp:
            match_dir = write_match(Path(tmp), frames, board_session=f"flat-1v1-{ORIGIN_MS}")
            records = write_jsonl(Path(tmp) / "rec" / "r.jsonl",
                                  [setup_row(), position_row()])
            out_dir = Path(tmp) / "out"
            warnings = io.StringIO()
            with contextlib.redirect_stderr(warnings):
                code = ihm.main([str(match_dir), "--out", str(out_dir),
                                 "--records", str(records)])
            self.assertEqual(code, 0)
            self.assertIn("没有解析出", warnings.getvalue())
            with np.load(out_dir / "dataset.npz") as data:
                meta = json.loads(str(data["meta"]))
                self.assertEqual(meta["rewardSource"], "health-delta")
                self.assertNotIn("damageDealt", data.files)

    def test_cli_end_to_end_uses_arena_records(self) -> None:
        frames = timeline(20)
        with tempfile.TemporaryDirectory() as tmp:
            match_dir = write_match(Path(tmp), frames, board_session=f"flat-1v1-{ORIGIN_MS}")
            records = write_jsonl(Path(tmp) / "rec" / "r.jsonl", [
                setup_row(), position_row(),
                damage_row(t_ms(1.25), victim=BOT, attacker=PLAYER),
                damage_row(t_ms(1.45), victim=PLAYER, attacker=BOT, seq=2),
            ])
            out_dir = Path(tmp) / "out"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = ihm.main(["--record", str(match_dir), "--out", str(out_dir),
                                 "--records", str(records)])
            self.assertEqual(code, 0)
            report = json.loads(stdout.getvalue().strip().splitlines()[-1])
            self.assertEqual(report["rewardSource"], "arena-record")
            self.assertEqual(report["arenaRecord"]["stats"]["unattributed"], 0)
            with np.load(out_dir / "dataset.npz") as data:
                meta = json.loads(str(data["meta"]))
                self.assertEqual(meta["rewardSource"], "arena-record")
                self.assertIn("damageDealt", data.files)
                self.assertAlmostEqual(float(np.sum(data["damageDealt"])), 1.6, places=6)
                self.assertAlmostEqual(float(np.sum(data["damageTaken"])), 1.6, places=6)


if __name__ == "__main__":
    unittest.main()
