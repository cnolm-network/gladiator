"""G-509：每局复盘的统计口径与落盘格式。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gladiator_env.review import EpisodeTracker, to_markdown, write_review
from gladiator_env.spaces import NUM_ACTIONS


def obs(health: float, x: float = 0.0, state: str = "Alive") -> dict:
    return {"self": {"health": health, "pos": {"x": x, "y": 4.0, "z": 0.0}}, "state": state}


class EpisodeTrackerTest(unittest.TestCase):
    def make(self) -> EpisodeTracker:
        return EpisodeTracker(episode=1, username="Bot_A", opponent="Bot_B")

    def test_counts_hits_whiffs_and_damage(self) -> None:
        tracker = self.make()
        tracker.update(
            action_index=2, previous=obs(20), current=obs(20),
            opponent_previous=obs(20), opponent_current=obs(14), reward=15.0,
        )
        tracker.update(
            action_index=2, previous=obs(20), current=obs(20),
            opponent_previous=obs(14), opponent_current=obs(14), reward=-0.3,
        )
        review = tracker.finish(winner=None, terminated=False, truncated=True)
        self.assertEqual(review.hits, 1)
        self.assertEqual(review.whiffs, 1)
        self.assertAlmostEqual(review.damage_dealt, 6.0)
        self.assertAlmostEqual(review.hit_rate, 0.5)
        self.assertEqual(review.attack_actions, 2)
        self.assertAlmostEqual(review.total_reward, 14.7)

    def test_damage_taken_and_distance(self) -> None:
        tracker = self.make()
        tracker.update(
            action_index=0, previous=obs(20, 0), current=obs(15, 0),
            opponent_previous=obs(20, 4), opponent_current=obs(20, 4), reward=-6.0,
        )
        review = tracker.finish(winner=None, terminated=False, truncated=False)
        self.assertAlmostEqual(review.damage_taken, 5.0)
        self.assertAlmostEqual(review.mean_distance, 4.0)
        self.assertAlmostEqual(review.damage_ratio, 0.0)  # 只挨打没输出 → 0

    def test_combo_counts_consecutive_hits(self) -> None:
        tracker = self.make()
        for i in range(3):
            tracker.update(
                action_index=2, previous=obs(20), current=obs(20),
                opponent_previous=obs(20 - i * 6), opponent_current=obs(14 - i * 6), reward=15.0,
            )
        review = tracker.finish(winner="Bot_A", terminated=True, truncated=False)
        self.assertEqual(review.hits, 3)
        self.assertGreaterEqual(review.max_combo, 3)
        self.assertEqual(review.winner, "Bot_A")

    def test_action_counts_cover_all_seven(self) -> None:
        tracker = self.make()
        tracker.update(
            action_index=0, previous=obs(20), current=obs(20),
            opponent_previous=obs(20), opponent_current=obs(20), reward=0.0,
        )
        review = tracker.finish(winner=None, terminated=False, truncated=False)
        self.assertEqual(len(review.action_counts), NUM_ACTIONS)
        self.assertEqual(review.action_counts["idle"], 1)

    def test_markdown_and_json_are_written(self) -> None:
        tracker = self.make()
        tracker.update(
            action_index=2, previous=obs(20), current=obs(20),
            opponent_previous=obs(20), opponent_current=obs(12), reward=20.0,
            events=[{"type": "combo", "value": 2, "bonus": 1.2}],
        )
        review = tracker.finish(
            winner="Bot_A", terminated=True, truncated=False,
            reward_breakdown={"damageDealt": 20.0, "combo": 1.2},
        )
        markdown = to_markdown(review)
        self.assertIn("# Episode 1", markdown)
        self.assertIn("## 伤害交换", markdown)
        self.assertIn("## 动作分布", markdown)
        self.assertIn("## 奖励分解", markdown)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_review(tmp, review)
            self.assertTrue(path.exists())
            data = json.loads(Path(tmp, "episode-001.json").read_text(encoding="utf-8"))
            self.assertEqual(data["hits"], 1)
            self.assertEqual(data["winner"], "Bot_A")
            self.assertEqual(data["actionCounts"]["attack"], 1)
            self.assertIn("combo", data["rewardBreakdown"])


if __name__ == "__main__":
    unittest.main()
