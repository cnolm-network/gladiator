"""G-703：联赛 / 镜像评估支持 v1+v2 混打。

这一组盯的是「动作空间混打」这条链路的每一段（**全程离线**，不连服务器）：

1. ``model_io.load_model``：把 ``_meta.actionSpace`` / ``headSizes`` 读出来，坏文件报错；
2. ``model_io.policy_action_for_env``：v1 → ``int``，v2 → 10 个头（带 mask）；
3. ``GladiatorEnv``：三种组合都要真的能 step ——

   - v1 学习者 + v2 对手（学习方发宏动作、对手发 intent 帧）；
   - v2 学习者 + v1 对手（学习方发 intent、对手发宏动作）；
   - v2 学习者 + v2 对手（两边都是 intent，且**不能**退化成随机动作——
     这正是修之前的 bug：``Policy.act()`` 在 v2 上抛错，异常被 env 吞掉后
     每帧走「随机头」分支，表面能跑、实则全程乱打）；

4. ``league.run_league`` / ``format_markdown``：把 ``actionSpace`` 带进 summary 与表格；
5. 两个脚本的 ``--dry-run``：服务器关着也能完成「参数解析 + 模型加载」。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import numpy as np  # noqa: E402

from gladiator_env import ArenaConfig, BotConfig, GladiatorEnv  # noqa: E402
from gladiator_env.spaces import ACTION_HEADS_V2, OBS_DIM, decode_intent  # noqa: E402
from gladiator_env.testing import FakeLink, make_observation, make_target  # noqa: E402
from gladiator_train.model_io import (  # noqa: E402
    ModelLoadError,
    ModelRecord,
    env_action_space,
    load_model,
    policy_action_for_env,
)
from gladiator_train.policy import Policy  # noqa: E402
from scripts.league import (  # noqa: E402
    build_schedule,
    dry_run_report,
    format_markdown,
    load_policies,
    run_league,
)
from scripts import mirror_eval  # noqa: E402

HEADS = len(ACTION_HEADS_V2)


def make_v1(path: Path, *, seed: int = 1) -> str:
    Policy(obs_dim=OBS_DIM, n_actions=11, hidden=8, seed=seed).save(str(path))
    return str(path)


def make_v2(path: Path, *, seed: int = 2) -> str:
    Policy(obs_dim=OBS_DIM, action_space="v2", hidden=8, seed=seed).save(str(path))
    return str(path)


def observation(username: str = "Glad_Bot_1") -> dict:
    """最小可用观测：手里有剑、对面 2 格外（够得着，便于真的出手）。"""
    other = "Glad_Bot_2" if username == "Glad_Bot_1" else "Glad_Bot_1"
    raw = make_observation(
        username, health=20.0,
        targets=[make_target(2 if username == "Glad_Bot_1" else 1, other,
                             x=1.0, z=0.0, distance=2.0)])
    raw["self"]["inventory"] = [{"slot": 36, "name": "diamond_sword", "count": 1},
                                {"slot": 37, "name": "golden_apple", "count": 5}]
    raw["self"]["heldItem"] = {"name": "diamond_sword", "count": 1}
    return raw


def build_mixed_env(learner_space: str, opponent) -> tuple[GladiatorEnv, list[FakeLink]]:
    links = [FakeLink("Glad_Bot_1", observation("Glad_Bot_1")),
             FakeLink("Glad_Bot_2", observation("Glad_Bot_2"))]
    env = GladiatorEnv(
        [BotConfig("Glad_Bot_1", "ws://127.0.0.1:8766"),
         BotConfig("Glad_Bot_2", "ws://127.0.0.1:8767")],
        links=links,
        connect=True,
        control_hz=200.0,
        action_ticks=1,
        arena=ArenaConfig(give_kit=False),
        single_agent=True,
        action_space=learner_space,
        opponent_policy=opponent,
        logger=lambda _message: None,   # 测试里不打印 [env] 横幅
    )
    env._raw = [observation("Glad_Bot_1"), observation("Glad_Bot_2")]
    return env, links


def intents_of(link: FakeLink) -> list[dict]:
    return [action for action in link.actions if action.get("type") == "intent"]


class LoadModelTest(unittest.TestCase):
    def test_reads_v1_and_v2_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            v1 = load_model(make_v1(Path(folder) / "a.npz"))
            v2 = load_model(make_v2(Path(folder) / "b.npz"))
        self.assertEqual(v1.action_space, "v1")
        self.assertEqual(v1.head_sizes, (11,))
        self.assertEqual(v2.action_space, "v2")
        self.assertEqual(v2.head_sizes, tuple(ACTION_HEADS_V2.values()))
        self.assertEqual(v2.obs_dim, OBS_DIM)
        self.assertTrue(v2.is_v2)
        self.assertIn("[v2", v2.describe())

    def test_old_obs_dim_is_padded_and_actable(self):
        """现役模型是 25/23 维，而环境现在是 37 维：加载时必须右侧补零，否则第一步就 reshape 炸。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old25.npz"
            Policy(obs_dim=25, n_actions=11, hidden=8, seed=7).save(str(path))
            record = load_model(path)
        self.assertEqual(record.native_obs_dim, 25)
        self.assertEqual(record.obs_dim, OBS_DIM)
        self.assertEqual(record.obs_label(), f"25→{OBS_DIM}")
        self.assertEqual(record.policy.params["W1"].shape, (8, OBS_DIM))
        self.assertTrue(np.all(record.policy.params["W1"][:, 25:] == 0))
        action = policy_action_for_env(record, np.zeros(OBS_DIM), rng=np.random.default_rng(0))
        self.assertTrue(0 <= action < 11)

    def test_old_v2_model_padded_and_actable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old25v2.npz"
            Policy(obs_dim=25, action_space="v2", hidden=8, seed=8).save(str(path))
            record = load_model(path)
        self.assertEqual(record.action_space, "v2")
        self.assertEqual(record.obs_label(), f"25→{OBS_DIM}")
        heads = policy_action_for_env(record, np.zeros(OBS_DIM), rng=np.random.default_rng(0))
        self.assertEqual(len(heads), HEADS)

    def test_missing_file_raises(self):
        with self.assertRaises(ModelLoadError) as caught:
            load_model("/no/such/policy.npz")
        self.assertIn("模型不存在", str(caught.exception))

    def test_broken_npz_raises_with_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.npz"
            path.write_bytes(b"definitely not an npz")
            with self.assertRaises(ModelLoadError) as caught:
                load_model(path)
            message = str(caught.exception)
            self.assertIn("无法加载", message)
            self.assertIn("broken.npz", message)


class ActionForEnvTest(unittest.TestCase):
    def test_v1_returns_int(self):
        with tempfile.TemporaryDirectory() as folder:
            record = load_model(make_v1(Path(folder) / "a.npz"))
        action = policy_action_for_env(record, np.zeros(OBS_DIM), rng=np.random.default_rng(0))
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 11)

    def test_v2_returns_head_list(self):
        with tempfile.TemporaryDirectory() as folder:
            record = load_model(make_v2(Path(folder) / "b.npz"))
        action = policy_action_for_env(record, np.zeros(OBS_DIM), rng=np.random.default_rng(0))
        self.assertIsInstance(action, list)
        self.assertEqual(len(action), HEADS)
        for value, size in zip(action, record.head_sizes):
            self.assertTrue(0 <= value < size, f"{value} 超出头大小 {size}")

    def test_v2_respects_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            record = load_model(make_v2(Path(folder) / "b.npz"))
        # 每个头只允许最后一个档位
        mask = {}
        for head, size in ACTION_HEADS_V2.items():
            row = [False] * size
            row[-1] = True
            mask[head] = row
        action = policy_action_for_env(record, np.zeros(OBS_DIM), mask=mask,
                                       rng=np.random.default_rng(0), greedy=True)
        self.assertEqual(action, [size - 1 for size in record.head_sizes])

    def test_env_action_space_accepts_record_or_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            record = load_model(make_v2(Path(folder) / "b.npz"))
        self.assertEqual(env_action_space(record), "v2")
        self.assertEqual(env_action_space(record.policy), "v2")


class MixedEnvTest(unittest.TestCase):
    """三种动作空间组合都要真的能 step（离线，FakeLink 提供观测）。"""

    def test_v1_learner_v2_opponent(self):
        with tempfile.TemporaryDirectory() as folder:
            opponent = load_model(make_v2(Path(folder) / "opp.npz")).policy
        env, links = build_mixed_env("v1", opponent)
        try:
            links[0].actions.clear()
            links[1].actions.clear()
            env.step(1)                              # v1 学习者：1 = approach
            self.assertNotIn("intent", [a.get("type") for a in links[0].actions])
            enemy = intents_of(links[1])
            self.assertEqual(len(enemy), 1)
            self.assertEqual(len(enemy[0]["intent"].get("look", {})), 2)
            self.assertEqual(env.opponent_policy_stats["errors"], 0)
            self.assertEqual(env.opponent_policy_stats["invalidActions"], 0)
        finally:
            env.close()

    def test_v2_learner_v1_opponent(self):
        with tempfile.TemporaryDirectory() as folder:
            opponent = load_model(make_v1(Path(folder) / "opp.npz")).policy
        env, links = build_mixed_env("v2", opponent)
        try:
            links[0].actions.clear()
            links[1].actions.clear()
            env.step([1, 3, 2, 1, 0, 0, 0, 0, 0, 1])   # v2 学习者：10 头
            learner = intents_of(links[0])
            self.assertEqual(len(learner), 1, links[0].actions)
            self.assertTrue(learner[0]["intent"]["attack"])
            # 对手是 v1：只发宏动作，不发 intent
            self.assertNotIn("intent", [a.get("type") for a in links[1].actions])
            self.assertGreaterEqual(len(links[1].actions), 1)
        finally:
            env.close()

    def test_v2_learner_v2_opponent(self):
        with tempfile.TemporaryDirectory() as folder:
            opponent = load_model(make_v2(Path(folder) / "opp.npz")).policy
        env, links = build_mixed_env("v2", opponent)
        try:
            links[0].actions.clear()
            links[1].actions.clear()
            env.step([1, 3, 2, 1, 0, 0, 0, 0, 0, 1])
            self.assertEqual(len(intents_of(links[0])), 1)
            self.assertEqual(len(intents_of(links[1])), 1)
        finally:
            env.close()

    def test_v2_opponent_is_not_silently_random(self):
        """回归：``Policy.act()`` 在 v2 上抛错 → 异常被吞 → 每帧走「随机头」兜底。

        判定方式：给对手包一层**只实现 ``sample()``** 的记录器——
        修之前 env 只会调 ``act()``，这里没有 ``act`` → 抛错 → 被吞成随机动作，
        ``errors`` 会 >0、``calls`` 为空；修之后 env 调 ``sample``，
        发出去的 intent 必须与 ``decode_intent(采样结果)`` 完全一致。
        """
        with tempfile.TemporaryDirectory() as folder:
            policy = load_model(make_v2(Path(folder) / "opp.npz")).policy

        class RecordingV2Policy:
            is_v2 = True
            action_space = "v2"

            def __init__(self, inner):
                self.inner = inner
                self.calls = []

            def act(self, vector, rng=None, greedy=False):
                # 与 v2 Policy 一致：``act`` 是 v1 的接口，v2 上必须炸——
                # 这样只要 env 还在调 act，「被吞成随机」就会体现在 errors 上。
                raise ValueError("v2 策略的动作是 (H,) 数组：请用 sample() / act_with_mask()")

            def sample(self, vector, rng=None, greedy=False):
                picked = self.inner.sample(vector, rng, greedy=greedy)
                self.calls.append([int(value) for value in np.asarray(picked).reshape(-1)])
                return picked

        recorder = RecordingV2Policy(policy)
        env, links = build_mixed_env("v1", recorder)
        try:
            links[1].actions.clear()
            env.step(1)
            self.assertEqual(env.opponent_policy_stats["errors"], 0,
                             "v2 对手不该抛异常（抛了就会被吞成随机）")
            self.assertEqual(len(recorder.calls), 1, "对手必须走 sample()，而不是 act()")
            frames = intents_of(links[1])
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["intent"], decode_intent(recorder.calls[0]),
                             "发出的 intent 必须就是采样结果的解码")
        finally:
            env.close()


class LeagueMixedTest(unittest.TestCase):
    """联赛编排层：混打时 summary / markdown 要带动作空间，赛程仍交替左右位。"""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.paths = {"d3": make_v2(root / "d3.npz", seed=3),
                      "d2": make_v2(root / "d2.npz", seed=4),
                      "old": make_v1(root / "old.npz", seed=5)}
        self.models = load_policies(list(self.paths.values()))
        # load_policies 按路径顺序起名（d3/d2/old），这里对齐成本测试的键
        self.by_path = {record.path: record for record in self.models.values()}

    def tearDown(self):
        self.folder.cleanup()

    def _play(self, left: str, right: str, _seed: int, _max_steps: int,
              _greedy: bool = False) -> str:
        """「v2 更强」的假对局：v2 一律赢 v1，同空间按路径序。"""
        left_v2 = self.by_path[left].action_space == "v2"
        right_v2 = self.by_path[right].action_space == "v2"
        if left_v2 == right_v2:
            return "left" if Path(left).name < Path(right).name else "right"
        return "left" if left_v2 else "right"

    def test_schedule_alternates_with_mixed_spaces(self):
        schedule = build_schedule(list(self.paths), 4)
        for label in self.paths:
            # 3 个模型 → 每个进 2 对 × 4 局 = 8 局，其中左右各 4 次（每对 2/2）。
            sides = [item["left"] for item in schedule if label in (item["a"], item["b"])]
            self.assertEqual(len(sides), 8, label)
            self.assertEqual(sides.count(label), 4, f"{label} 应左右各坐两次/对")

    def test_summary_carries_action_spaces(self):
        summary = run_league(self.paths, 4, self._play, seed=11, verbose=False,
                             models=self.models)
        self.assertEqual(summary["actionSpaces"], {"d3": "v2", "d2": "v2", "old": "v1"})
        spaces = {item["label"]: item["actionSpace"] for item in summary["policies"]}
        self.assertEqual(spaces, {"d3": "v2", "d2": "v2", "old": "v1"})
        # 每对 4 局：v2 全胜 v1；两个 v2 之间按路径名。d2 全胜（8-0），old 全负（0-8）。
        ranking = {item["label"]: item for item in summary["ranking"]}
        self.assertEqual(ranking["old"]["wins"], 0)
        self.assertEqual(ranking["d2"]["winRate"], 1.0)
        self.assertEqual(ranking["d3"]["winRate"], 0.5)

    def test_markdown_shows_action_space_column(self):
        summary = run_league(self.paths, 2, self._play, seed=12, verbose=False,
                             models=self.models)
        markdown = format_markdown(summary)
        self.assertIn("| 动作空间 |", markdown)
        self.assertIn("[v2]", markdown)          # 对阵/矩阵里带空间标签
        self.assertIn("[v1]", markdown)

    def test_models_labels_must_match_paths(self):
        with self.assertRaises(ValueError):
            run_league({"a": "x", "b": "y"}, 2, lambda *args: "draw", seed=1,
                       verbose=False, models={"a": None})

    def test_dry_run_report_lists_models(self):
        report = dry_run_report(self.models, 4, seed=99)
        self.assertIn("dry-run", report)
        self.assertIn("| 动作空间 |", report)
        self.assertIn("v2", report)
        self.assertIn("5/7/5/2/3/2/2/2/9/3", report)
        # 3 个模型 → 3 对 × 4 局 = 12 局
        self.assertIn("= 12 局", report)


class DryRunMainTest(unittest.TestCase):
    """验收判据：服务器没开时，两个脚本至少要能完成参数解析 + 模型加载。"""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.v1 = make_v1(root / "old.npz")
        self.v2a = make_v2(root / "d3.npz", seed=3)
        self.v2b = make_v2(root / "d2.npz", seed=4)

    def tearDown(self):
        self.folder.cleanup()

    def test_league_dry_run_with_three_mixed_models(self):
        from scripts import league

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = league.main(["--policies", self.v2a, self.v2b, self.v1, "--dry-run",
                                "--episodes-per-pair", "4"])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("dry-run", output)
        self.assertIn("= 12 局", output)
        self.assertIn("| d3 | v2 |", output)

    def test_mirror_dry_run_with_mixed_spaces(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mirror_eval.main(["--policy", self.v2a, "--policy-b", self.v1,
                                     "--dry-run"])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("新模型 d3（v2）", output)
        self.assertIn("旧模型 old（v1）", output)

    def test_mirror_model_table_has_both_spaces(self):
        new = load_model(self.v2a, label="d3")
        old = load_model(self.v1, label="old")
        table = mirror_eval.model_table(new, old)
        self.assertIn("| 新 | d3 | v2 |", table)
        self.assertIn("| 旧 | old | v1 |", table)

    def test_generic_stem_uses_parent_dir_label(self):
        """``.../models/<名字>/policy.npz`` 的 stem 都是 ``policy``：标签必须换成父目录名。"""
        from gladiator_train.model_io import model_label

        self.assertEqual(model_label("/x/models/v2-v4-dagger3/policy.npz"), "v2-v4-dagger3")
        self.assertEqual(model_label("/x/stage1.npz"), "stage1")
        # mirror 的 dry-run 也要用这个标签（否则表里两行都叫 policy，分不清谁是谁）
        parent = Path(self.folder.name) / "v2-v4-dagger3"
        parent.mkdir()
        model = make_v2(parent / "policy.npz", seed=9)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mirror_eval.main(["--policy", model, "--policy-b", self.v1, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("新模型 v2-v4-dagger3（v2）", buffer.getvalue())

    def test_mirror_rejects_broken_model(self):
        buffer = io.StringIO()
        broken = Path(self.folder.name) / "broken.npz"
        broken.write_bytes(b"nope")
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                mirror_eval.main(["--policy", str(broken), "--policy-b", self.v1,
                                  "--dry-run"])
        self.assertIn("无法加载", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
