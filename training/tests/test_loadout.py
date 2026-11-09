from __future__ import annotations

import unittest

from gladiator_env.env import loadout_report


def observation(*, inventory=None, armor=None, held="diamond_sword"):
    return {
        "self": {
            "inventory": inventory if inventory is not None else [
                {"slot": 36, "name": "diamond_sword", "count": 1},
                {"slot": 37, "name": "golden_apple", "count": 5},
                {"slot": 38, "name": "fishing_rod", "count": 1},
            ],
            "armor": armor if armor is not None else {
                "head": {"name": "diamond_helmet"},
                "torso": {"name": "diamond_chestplate"},
                "legs": {"name": "diamond_leggings"},
                "feet": {"name": "diamond_boots"},
            },
            "heldItem": {"name": held, "count": 1},
        }
    }


class LoadoutReportTest(unittest.TestCase):
    def test_complete_kit_is_ok(self):
        report = loadout_report(observation())
        self.assertTrue(report["ok"])
        self.assertTrue(report["holding_sword"])
        self.assertEqual(report["armor"], 4)
        self.assertEqual(report["missing"], [])

    def test_missing_gapples_are_reported_with_count(self):
        raw = observation(inventory=[
            {"slot": 36, "name": "diamond_sword", "count": 1},
            {"slot": 37, "name": "golden_apple", "count": 2},
        ])
        report = loadout_report(raw)
        self.assertFalse(report["ok"])
        self.assertIn(("golden_apple", 3), report["missing"])
        self.assertIn(("fishing_rod", 1), report["missing"])

    def test_armor_counted_from_worn_slots(self):
        raw = observation(armor={"head": {"name": "diamond_helmet"}, "torso": None, "legs": None, "feet": None})
        report = loadout_report(raw)
        self.assertEqual(report["armor"], 1)
        self.assertFalse(report["ok"])

    def test_holding_rod_is_not_holding_sword(self):
        report = loadout_report(observation(held="fishing_rod"))
        self.assertTrue(report["ok"])
        self.assertFalse(report["holding_sword"])

    def test_empty_observation_does_not_crash(self):
        report = loadout_report(None)
        self.assertFalse(report["ok"])
        self.assertEqual(report["armor"], 0)
        self.assertEqual(report["held"], "")
        self.assertEqual(len(report["missing"]), 3)


if __name__ == "__main__":
    unittest.main()
