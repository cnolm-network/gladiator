"""评估套件 v2 的统计与编排测试（纯离线，不需要 RCON / 服务端）。

覆盖：Wilson 区间、精确 McNemar、汇总口径、配对对齐（含乱序与不一致的 seed 集）、
reset 重试、run_suite 的 seed/对手模式编排，以及 rollout 的伤害回退口径。
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from gladiator_train.eval_suite import (  # noqa: E402
    SUITE_DEFAULT,
    SuiteConfig,
    mcnemar_exact,
    pair_rows,
    reset_with_retry,
    rollout,
    run_suite,
    summarize,
    verdict,
    wilson_interval,
)


class WilsonTests(unittest.TestCase):
    def test_all_wins_has_upper_one_and_sane_lower(self) -> None:
        low, high = wilson_interval(10, 10)
        self.assertEqual(high, 1.0)
        self.assertGreater(low, 0.6)
        self.assertLess(low, 0.8)

    def test_empty_sample_is_uninformative(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_interval_contains_point_estimate(self) -> None:
        for wins, n in ((0, 8), (3, 8), (8, 8), (50, 100)):
            low, high = wilson_interval(wins, n)
            rate = wins / n
            self.assertLessEqual(low, rate)
            self.assertGreaterEqual(high, rate)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)

    def test_wider_interval_for_smaller_samples(self) -> None:
        low_small, high_small = wilson_interval(4, 8)
        low_big, high_big = wilson_interval(50, 100)
        self.assertGreater(high_small - low_small, high_big - low_big)


class McNemarTests(unittest.TestCase):
    def test_no_discordant_pairs_is_one(self) -> None:
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_ten_to_zero_is_highly_significant(self) -> None:
        p = mcnemar_exact(10, 0)
        self.assertLess(p, 0.01)
        self.assertAlmostEqual(p, 2.0 / 1024.0, places=12)

    def test_eight_two_matches_hand_computed_binomial(self) -> None:
        expected = 2.0 * sum(math.comb(10, k) for k in range(0, 3)) / (2 ** 10)
        self.assertAlmostEqual(mcnemar_exact(8, 2), expected, places=12)
        self.assertAlmostEqual(expected, 0.109375, places=12)

    def test_symmetric(self) -> None:
        self.assertAlmostEqual(mcnemar_exact(8, 2), mcnemar_exact(2, 8), places=12)
        self.assertAlmostEqual(mcnemar_exact(9, 1), mcnemar_exact(1, 9), places=12)


def _row(opponent: str, index: int, seed: int, outcome: str,
         reward: float = 0.0, steps: int = 10,
         dealt: float = 0.0, taken: float = 0.0) -> dict:
    return {
        "opponent": opponent,
        "episodeIndex": index,
        "seed": seed,
        "outcome": outcome,
        "win": outcome == "win",
        "reward": reward,
        "steps": steps,
        "damageDealt": dealt,
        "damageTaken": taken,
    }


class SummarizeTests(unittest.TestCase):
    def test_three_episodes_one_win_two_losses(self) -> None:
        rows = [
            _row("scripted", 0, 1000, "win", reward=30.0, steps=80, dealt=10.0, taken=5.0),
            _row("scripted", 1, 1001, "loss", reward=-20.0, steps=120, dealt=2.0, taken=8.0),
            _row("scripted", 2, 1002, "loss", reward=-10.0, steps=100, dealt=3.0, taken=9.0),
        ]
        summary = summarize(rows)
        overall = summary["overall"]
        self.assertEqual(overall["episodes"], 3)
        self.assertEqual((overall["wins"], overall["losses"], overall["draws"]), (1, 2, 0))
        self.assertAlmostEqual(overall["winRate"], 1.0 / 3.0, places=12)
        low, high = overall["winRateCI"]
        self.assertLessEqual(low, overall["winRate"])
        self.assertGreaterEqual(high, overall["winRate"])
        self.assertAlmostEqual(overall["meanReward"], 0.0, places=12)
        self.assertAlmostEqual(overall["meanSteps"], 100.0, places=12)
        self.assertAlmostEqual(overall["damageRatio"], 15.0 / 22.0, places=12)
        self.assertEqual(set(summary["opponents"]), {"scripted"})

    def test_groups_by_opponent_and_counts_draws(self) -> None:
        rows = [
            _row("scripted", 0, 1000, "win"),
            _row("rusher", 0, 1000, "draw"),
            _row("rusher", 1, 1001, "loss"),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["overall"]["draws"], 1)
        self.assertEqual(summary["opponents"]["scripted"]["wins"], 1)
        self.assertEqual(summary["opponents"]["rusher"]["losses"], 1)
        self.assertEqual(summary["opponents"]["rusher"]["draws"], 1)

    def test_empty_rows(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["overall"]["episodes"], 0)
        self.assertEqual(summary["overall"]["winRateCI"], (0.0, 1.0))
        self.assertEqual(summary["opponents"], {})


class PairRowsTests(unittest.TestCase):
    def test_shuffled_b_still_pairs_by_key(self) -> None:
        rows_a = [
            _row("scripted", 0, 1000, "win"),
            _row("scripted", 1, 1001, "win"),
            _row("rusher", 0, 1000, "loss"),
            _row("rusher", 1, 1001, "win"),
        ]
        rows_b = [
            _row("rusher", 1, 1001, "loss"),
            _row("scripted", 1, 1001, "loss"),
            _row("rusher", 0, 1000, "win"),
            _row("scripted", 0, 1000, "win"),
        ]
        result = pair_rows(rows_a, rows_b)
        self.assertEqual(result["pairs"], 4)
        # A 独赢：scripted#1、rusher#1；B 独赢：rusher#0
        self.assertEqual((result["b"], result["c"]), (2, 1))
        self.assertAlmostEqual(result["p"], mcnemar_exact(2, 1), places=12)
        self.assertEqual(result["byOpponent"]["scripted"]["b"], 1)
        self.assertEqual(result["byOpponent"]["rusher"]["c"], 1)
        self.assertEqual(result["byOpponent"]["scripted"]["pairs"], 2)

    def test_mismatched_seed_sets_raise(self) -> None:
        rows_a = [_row("scripted", 0, 1000, "win"), _row("scripted", 1, 1001, "win")]
        rows_b = [_row("scripted", 0, 1000, "loss")]
        with self.assertRaises(ValueError) as ctx:
            pair_rows(rows_a, rows_b)
        self.assertIn("seed 集不一致", str(ctx.exception))

    def test_duplicate_keys_raise(self) -> None:
        rows = [_row("scripted", 0, 1000, "win"), _row("scripted", 0, 1000, "loss")]
        with self.assertRaises(ValueError):
            pair_rows(rows, rows)

    def test_verdict_wording(self) -> None:
        self.assertIn("不显著", verdict({"p": 0.42, "b": 3, "c": 1}))
        self.assertIn("A 显著更好", verdict({"p": 0.01, "b": 9, "c": 1}, "A", "B"))
        self.assertIn("B 显著更好", verdict({"p": 0.01, "b": 1, "c": 9}, "A", "B"))


class FakeEnv:
    """最小 env：固定步数、固定赢家，记录 reset 的 seed 与 opponent_mode。"""

    def __init__(self, mode: str, *, steps: int = 2, winner: str = "Bot_A") -> None:
        self.opponent_mode = mode
        self.bots = [SimpleNamespace(username="Bot_A"), SimpleNamespace(username="Bot_B")]
        self.steps = steps
        self.winner = winner
        self.reset_seeds: list = []
        self.closed = False
        self._t = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._t = 0
        return np.zeros(4, dtype=np.float32), {"step": 0}

    def step(self, action):
        self._t += 1
        done = self._t >= self.steps
        info = {
            "step": self._t,
            "winner": self.winner if done else None,
            "health": {"Bot_A": 20.0, "Bot_B": 20.0},
        }
        return np.zeros(4, dtype=np.float32), 1.0, done, False, info

    def close(self) -> None:
        self.closed = True


class StubPolicy:
    def __init__(self) -> None:
        self.greedy_flags: list = []
        self.calls = 0

    def act(self, obs, rng=None, greedy=False):
        self.calls += 1
        self.greedy_flags.append(bool(greedy))
        return 0


class ResetRetryTests(unittest.TestCase):
    class FlakyEnv:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.seeds: list = []

        def reset(self, seed=None):
            self.seeds.append(seed)
            if len(self.seeds) <= self.failures:
                raise TimeoutError("Bot 未就绪")
            return np.zeros(2, dtype=np.float32), {}

    def test_retry_uses_incrementing_seed(self) -> None:
        env = self.FlakyEnv(failures=2)
        obs, _info = reset_with_retry(env, 1000, attempts=4, delay=0.0,
                                      sleeper=lambda _s: None)
        self.assertEqual(env.seeds, [1000, 1001, 1002])
        self.assertEqual(tuple(obs.shape), (2,))

    def test_exhausted_retries_raise_runtime_error(self) -> None:
        env = self.FlakyEnv(failures=4)
        with self.assertRaises(RuntimeError):
            reset_with_retry(env, 1000, attempts=4, delay=0.0, sleeper=lambda _s: None)


class RunSuiteTests(unittest.TestCase):
    def test_calls_each_opponent_with_shared_seed_set(self) -> None:
        created: list = []

        def factory(mode: str) -> FakeEnv:
            env = FakeEnv(mode, steps=2)
            created.append(env)
            return env

        config = SuiteConfig(opponents=("scripted", "rusher"), episodes=3,
                             seed=1000, max_steps=5)
        policy = StubPolicy()
        rows = run_suite(factory, policy, config, sleeper=lambda _s: None)

        self.assertEqual([env.opponent_mode for env in created], ["scripted", "rusher"])
        self.assertEqual([env.reset_seeds for env in created],
                         [[1000, 1001, 1002], [1000, 1001, 1002]])
        self.assertTrue(all(env.closed for env in created))
        self.assertEqual(len(rows), 6)
        self.assertEqual([row["seed"] for row in rows[:3]], [1000, 1001, 1002])
        self.assertEqual([row["opponent"] for row in rows], ["scripted"] * 3 + ["rusher"] * 3)
        self.assertTrue(all(row["outcome"] == "win" for row in rows))
        self.assertTrue(all(row["steps"] == 2 for row in rows))
        self.assertAlmostEqual(rows[0]["reward"], 2.0, places=12)
        # 贪心评估默认开启
        self.assertTrue(all(policy.greedy_flags))
        # rows 必须能直接写 JSON
        json.dumps(rows)

    def test_draw_when_winner_missing(self) -> None:
        def factory(mode: str) -> FakeEnv:
            return FakeEnv(mode, steps=1, winner=None)

        config = SuiteConfig(opponents=("idle",), episodes=1, seed=7, max_steps=3)
        rows = run_suite(factory, StubPolicy(), config, sleeper=lambda _s: None)
        self.assertEqual(rows[0]["outcome"], "draw")
        self.assertFalse(rows[0]["win"])


class RolloutDamageTests(unittest.TestCase):
    class HealthEnv:
        """血量逐 step 下降：Bot_A 20→14→8（受伤 12），Bot_B 20→17（对手受伤 3）。"""

        def __init__(self) -> None:
            self.bots = [SimpleNamespace(username="Bot_A"), SimpleNamespace(username="Bot_B")]
            self._step = 0
            self.health_plan = [
                {"Bot_A": 20.0, "Bot_B": 20.0},
                {"Bot_A": 14.0, "Bot_B": 17.0},
                {"Bot_A": 8.0, "Bot_B": 17.0},
            ]

        def step(self, action):
            self._step += 1
            done = self._step >= len(self.health_plan)
            info = {"winner": None, "health": dict(self.health_plan[self._step - 1])}
            return np.zeros(2, dtype=np.float32), 0.0, done, False, info

    def test_health_fallback_when_env_has_no_events(self) -> None:
        env = self.HealthEnv()
        result = rollout(env, StubPolicy(), np.zeros(2, dtype=np.float32),
                         max_steps=3, username="Bot_A")
        self.assertEqual(result["steps"], 3)
        # 第一帧只建立基线，之后的下落量才计入
        self.assertAlmostEqual(result["damageTaken"], 6.0 + 6.0, places=12)
        self.assertAlmostEqual(result["damageDealt"], 3.0, places=12)


class SuiteConfigTests(unittest.TestCase):
    def test_defaults_match_suite_default(self) -> None:
        config = SuiteConfig()
        self.assertEqual(tuple(config.opponents), SUITE_DEFAULT["opponents"])
        self.assertEqual(config.episodes, SUITE_DEFAULT["episodes"])
        self.assertEqual(config.seed, SUITE_DEFAULT["seed"])
        self.assertEqual(config.max_steps, SUITE_DEFAULT["maxSteps"])
        self.assertEqual(config.maxSteps, config.max_steps)
        self.assertEqual(config.total_episodes, 24)
        self.assertTrue(config.greedy)

    def test_to_dict_round_trip(self) -> None:
        config = SuiteConfig(opponents=("idle",), episodes=2, seed=5, max_steps=10)
        data = config.to_dict()
        self.assertEqual(data["opponents"], ["idle"])
        self.assertEqual(data["maxSteps"], 10)
        self.assertEqual(data["totalEpisodes"], 2)


if __name__ == "__main__":
    unittest.main()
