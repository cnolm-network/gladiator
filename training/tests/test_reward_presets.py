"""奖励预设：schema 校验、store CRUD 与条件项结算。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from gladiator_env.reward_presets import (
    PresetError,
    PresetStore,
    Term,
    build_terms,
    default_preset,
    normalize_preset,
    preset_for_env,
)


def _obs(username, z=0.0, health=20.0, state="Alive"):
    return {
        "self": {"pos": {"x": 0.0, "y": 4.0, "z": z}, "health": health},
        "state": state,
        "entities": [{"id": 7, "kind": "player", "name": "Opp", "pos": {"x": 0.0, "y": 4.0, "z": z + 3.0}, "distance": 3.0}],
        "tick": 1,
        "_username": username,
    }


class NormalizeTests(unittest.TestCase):
    def test_default_preset_is_valid(self):
        preset = normalize_preset(default_preset())
        self.assertEqual(preset["format"], "gladiator.reward-preset/v1")
        self.assertEqual(preset["terms"], [])
        # P2：胜负由 victory 承担（±50），kill/death 只作小额补充
        self.assertAlmostEqual(preset["base"]["kill"], 2.0)
        self.assertAlmostEqual(preset["base"]["victory"], 50.0)
        self.assertAlmostEqual(preset["base"]["pbrsAlpha"], 0.5)

    def test_unknown_base_key_rejected(self):
        preset = default_preset()
        preset["base"]["nope"] = 1.0
        with self.assertRaises(PresetError):
            normalize_preset(preset)

    def test_unknown_condition_rejected(self):
        preset = default_preset()
        preset["terms"] = [{"id": "x", "weight": 1.0, "unit": "step", "when": {"magic": 1}}]
        with self.assertRaises(PresetError):
            normalize_preset(preset)

    def test_duplicate_term_id_rejected(self):
        preset = default_preset()
        preset["terms"] = [
            {"id": "same", "weight": 1.0, "unit": "step", "when": {}},
            {"id": "same", "weight": 2.0, "unit": "step", "when": {}},
        ]
        with self.assertRaises(PresetError):
            normalize_preset(preset)

    def test_bad_unit_rejected(self):
        preset = default_preset()
        preset["terms"] = [{"id": "x", "weight": 1.0, "unit": "hour", "when": {}}]
        with self.assertRaises(PresetError):
            normalize_preset(preset)

    def test_nan_weight_rejected(self):
        preset = default_preset()
        preset["base"]["kill"] = float("nan")
        with self.assertRaises(PresetError):
            normalize_preset(preset)


class TermTests(unittest.TestCase):
    def test_distance_window(self):
        term = Term("threat", 0.3, "second", {"distanceMin": 2.8, "distanceMax": 4.2})
        self.assertAlmostEqual(term.evaluate({"distance": 3.5}, 0.2), 0.06)
        self.assertEqual(term.evaluate({"distance": 5.0}, 0.2), 0.0)
        self.assertEqual(term.evaluate({"distance": None}, 0.2), 0.0)

    def test_action_and_distance(self):
        term = Term("punish", -0.2, "step", {"actionIn": ["block"], "distanceMin": 4.2})
        self.assertEqual(term.evaluate({"action": "block", "distance": 5.0}, 0.2), -0.2)
        self.assertEqual(term.evaluate({"action": "block", "distance": 3.0}, 0.2), 0.0)
        self.assertEqual(term.evaluate({"action": "attack", "distance": 5.0}, 0.2), 0.0)

    def test_still_and_after_hit(self):
        still = Term("still", -0.1, "second", {"stillSecondsMin": 3.0})
        self.assertAlmostEqual(still.evaluate({"stillSeconds": 3.2}, 0.5), -0.05)
        self.assertEqual(still.evaluate({"stillSeconds": 2.0}, 0.5), 0.0)
        combo = Term("follow", 0.5, "step", {"afterHitSeconds": 1.0, "comboMin": 2})
        self.assertEqual(combo.evaluate({"sinceHit": 0.5, "combo": 2}, 0.2), 0.5)
        self.assertEqual(combo.evaluate({"sinceHit": 2.0, "combo": 2}, 0.2), 0.0)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PresetStore(self.tmp.name)
        self.store.ensure()

    def tearDown(self):
        self.tmp.cleanup()

    def test_ensure_creates_default(self):
        items = self.store.list()
        self.assertEqual([item["id"] for item in items], ["default"])

    def test_save_list_get_delete(self):
        preset = default_preset("我的预设")
        preset["base"]["kill"] = 25.0
        self.store.save("mine", preset)
        self.assertEqual(self.store.get("mine")["base"]["kill"], 25.0)
        self.assertIn("mine", [item["id"] for item in self.store.list()])
        self.assertTrue(self.store.delete("mine"))
        self.assertFalse(self.store.delete("mine"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "mine.json")))

    def test_overwrite_makes_backup(self):
        self.store.save("dup", default_preset("第一版"))
        self.store.save("dup", default_preset("第二版"))
        trash = os.listdir(os.path.join(self.tmp.name, "trash"))
        self.assertTrue(any(name.startswith("dup-") for name in trash))

    def test_path_traversal_rejected(self):
        with self.assertRaises(PresetError):
            self.store.get("../evil")

    def test_invalid_json_on_disk_skipped(self):
        with open(os.path.join(self.tmp.name, "broken.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        ids = [item["id"] for item in self.store.list()]
        self.assertIn("default", ids)
        self.assertNotIn("broken", ids)


class P2PresetTest(unittest.TestCase):
    """P2：victory 符号权重与 won/lost 条件。"""

    def _preset(self, terms):
        return {
            "format": "gladiator.reward-preset/v1",
            "name": "p2",
            "base": {},
            "terms": terms,
        }

    def test_victory_symbol_weight(self):
        preset = normalize_preset(self._preset([
            {"id": "win", "weight": "victory", "unit": "step", "when": {"won": True}},
        ]))
        self.assertEqual(preset["terms"][0]["weight"], 50.0)

    def test_unknown_symbol_rejected(self):
        with self.assertRaises(PresetError):
            normalize_preset(self._preset([
                {"id": "x", "weight": "nope", "unit": "step", "when": {}},
            ]))

    def test_won_lost_conditions(self):
        kwargs, terms = preset_for_env(self._preset([
            {"id": "win", "weight": 5, "unit": "step", "when": {"won": True}},
            {"id": "lose", "weight": -5, "unit": "step", "when": {"lost": True}},
        ]))
        self.assertEqual(len(terms), 2)
        self.assertEqual(terms[0].evaluate({"won": True, "lost": False}, 0.2), 5)
        self.assertEqual(terms[0].evaluate({"won": False, "lost": True}, 0.2), 0)
        self.assertEqual(terms[1].evaluate({"won": False, "lost": True}, 0.2), -5)

    def test_won_must_be_bool(self):
        with self.assertRaises(PresetError):
            normalize_preset(self._preset([
                {"id": "x", "weight": 1, "unit": "step", "when": {"won": 1}},
            ]))

    def test_old_preset_defaults_new_keys(self):
        kwargs, _terms = preset_for_env({
            "format": "gladiator.reward-preset/v1", "name": "old",
            "base": {"kill": 20.0}, "terms": [],
        })
        self.assertEqual(kwargs["victory"], 50.0)
        self.assertEqual(kwargs["pbrs_alpha"], 0.5)
        self.assertEqual(kwargs["shaping_cap"], 5.0)
        self.assertEqual(kwargs["kill"], 20.0, "旧预设显式给的值优先")


class PresetForEnvTests(unittest.TestCase):
    def test_base_maps_to_shaping_fields(self):
        weights, terms = preset_for_env(default_preset())
        self.assertAlmostEqual(weights["kill"], 2.0)
        self.assertAlmostEqual(weights["victory"], 50.0)
        self.assertAlmostEqual(weights["pbrs_gamma"], 1.0)
        self.assertAlmostEqual(weights["damage_dealt"], 2.5)
        self.assertAlmostEqual(weights["still"], -0.1)
        self.assertEqual(terms, [])

    def test_terms_built(self):
        preset = default_preset()
        preset["terms"] = [{"id": "threat", "weight": 0.3, "unit": "second", "when": {"distanceMin": 2.8}}]
        _, terms = preset_for_env(preset)
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].id, "threat")


if __name__ == "__main__":
    unittest.main()
