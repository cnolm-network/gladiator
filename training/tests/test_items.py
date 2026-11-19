"""物品能力表（``items.json``）的加载、匹配与未知物品兜底。

对应规格 §2.3：匹配支持通配/正则，未知物品落到 ``misc`` 且**不报错**——
这是「加物品零改代码」的前提，所以这几条必须是硬测试。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from gladiator_env.items import (
    CATEGORIES,
    DEFAULT_ITEMS_PATH,
    MISC_CAPABILITY,
    USE_MODES,
    ItemTable,
    canonical_name,
    default_table,
)


class DefaultTableTest(unittest.TestCase):
    """仓库自带那份 items.json 必须能直接回答内置条目的能力。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = ItemTable.load()

    def test_default_path_exists(self):
        self.assertTrue(os.path.isfile(DEFAULT_ITEMS_PATH), DEFAULT_ITEMS_PATH)

    def test_diamond_sword_is_sword_with_block(self):
        capability = self.table.capability("diamond_sword")
        self.assertEqual(capability["category"], "sword")
        self.assertEqual(capability["use"], {"mode": "hold_block", "ticks": 5})
        self.assertEqual(capability["cooldownTicks"], 10)
        self.assertAlmostEqual(self.table.reach("diamond_sword"), 3.99, places=2)

    def test_glob_matches_other_swords(self):
        for name in ("iron_sword", "stone_sword", "minecraft:gold_sword"):
            self.assertEqual(self.table.category(name), "sword", name)

    def test_golden_apple_is_hold_food(self):
        spec = self.table.use_spec("golden_apple")
        self.assertEqual(spec, {"mode": "hold", "ticks": 32})
        self.assertEqual(self.table.category("golden_apple"), "food")
        # 食物不参与攻击距离
        self.assertIsNone(self.table.reach("golden_apple"))

    def test_fishing_rod_is_click_projectile(self):
        spec = self.table.use_spec("fishing_rod")
        self.assertEqual(spec["mode"], "click")
        self.assertTrue(self.table.capability("fishing_rod")["projectile"])

    def test_bow_is_charge(self):
        self.assertEqual(self.table.use_spec("bow")["mode"], "charge")
        self.assertEqual(self.table.category("bow"), "bow")

    def test_potion_glob_matches_only_suffixed_names(self):
        # *_potion 只匹配「xx_potion」；裸 potion 是未知物品，要显式写一条才认
        self.assertEqual(self.table.category("splash_potion"), "potion")
        self.assertEqual(self.table.category("lingering_potion"), "potion")

    def test_block_entries(self):
        self.assertEqual(self.table.category("cobblestone"), "block")
        self.assertEqual(self.table.use_spec("cobblestone")["mode"], "click")
        self.assertEqual(self.table.category("grass_block"), "block")

    def test_unknown_item_falls_back_to_misc(self):
        capability = self.table.capability("mystery_widget")
        self.assertEqual(capability["category"], "misc")
        self.assertEqual(capability["use"], dict(MISC_CAPABILITY["use"]))
        self.assertIsNone(capability["matched"])
        self.assertFalse(self.table.is_usable("mystery_widget"))

    def test_namespace_and_case_are_normalised(self):
        self.assertEqual(canonical_name("  Minecraft:Diamond_Sword "), "diamond_sword")
        self.assertEqual(self.table.category("MINECRAFT:DIAMOND_SWORD"), "sword")

    def test_empty_name_is_misc(self):
        self.assertEqual(self.table.category(""), "misc")
        self.assertEqual(self.table.category(None), "misc")

    def test_use_spec_returns_copy(self):
        spec = self.table.use_spec("golden_apple")
        spec["ticks"] = 999
        self.assertEqual(self.table.use_spec("golden_apple")["ticks"], 32)

    def test_default_table_is_cached(self):
        self.assertIs(default_table(), default_table())


class CustomTableTest(unittest.TestCase):
    """自定义 items.json：正则、扁平写法、错误处理。"""

    def _write(self, payload: dict) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(payload, handle, ensure_ascii=False)
        finally:
            handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_exact_wins_over_glob(self):
        path = self._write({"items": {
            "*_sword": {"category": "sword", "use": {"mode": "hold_block", "ticks": 5}},
            "minecraft:wooden_sword": {"category": "misc", "use": {"mode": "none", "ticks": 0}},
        }})
        table = ItemTable.load(path)
        self.assertEqual(table.category("wooden_sword"), "misc")
        self.assertEqual(table.category("iron_sword"), "sword")

    def test_regex_prefix(self):
        path = self._write({"items": {
            "re:^.*_(pickaxe|axe)$": {"category": "tool", "use": {"mode": "click", "ticks": 2}},
        }})
        table = ItemTable.load(path)
        self.assertEqual(table.category("diamond_pickaxe"), "tool")
        self.assertEqual(table.category("iron_axe"), "tool")
        self.assertEqual(table.category("diamond_sword"), "misc")

    def test_flat_form_is_accepted(self):
        # 规格 §2.3 里的示例就是扁平写法（顶层直接是 name -> 配置）
        path = self._write({
            "minecraft:potion": {"category": "potion", "use": {"mode": "click", "ticks": 2}},
        })
        table = ItemTable.load(path)
        self.assertEqual(table.category("potion"), "potion")
        self.assertEqual(table.use_spec("potion")["mode"], "click")

    def test_unknown_category_is_rejected(self):
        path = self._write({"items": {"x": {"category": "banana"}}})
        with self.assertRaises(ValueError) as ctx:
            ItemTable.load(path)
        self.assertIn("banana", str(ctx.exception))

    def test_unknown_use_mode_is_rejected(self):
        path = self._write({"items": {"x": {"category": "tool", "use": {"mode": "wiggle"}}}})
        with self.assertRaises(ValueError):
            ItemTable.load(path)

    def test_bad_json_reports_path(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        handle.write("{ not json")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with self.assertRaises(ValueError) as ctx:
            ItemTable.load(handle.name)
        self.assertIn("items.json", str(ctx.exception))

    def test_missing_file_reports_absolute_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ItemTable.load("no-such-dir/items.json")
        self.assertIn("items.json", str(ctx.exception))

    def test_defaults_fill_missing_fields(self):
        path = self._write({"items": {"thing": {"category": "misc"}}})
        table = ItemTable.load(path)
        capability = table.capability("thing")
        self.assertEqual(capability["use"], {"mode": "none", "ticks": 0})
        self.assertEqual(capability["cooldownTicks"], 0)
        self.assertEqual(capability["stackGoal"], 1)
        self.assertFalse(capability["projectile"])

    def test_patterns_preserve_file_order(self):
        path = self._write({"items": {
            "minecraft:apple": {"category": "food"},
            "*_sword": {"category": "sword"},
            "re:.*potion": {"category": "potion"},
        }})
        table = ItemTable.load(path)
        self.assertEqual(table.patterns()[0], "minecraft:apple")
        self.assertIn("*_sword", table.patterns())
        self.assertEqual(len(table), 3)


class ConstantsTest(unittest.TestCase):
    def test_categories_cover_spec(self):
        for name in ("sword", "tool", "food", "block", "bow", "potion", "misc"):
            self.assertIn(name, CATEGORIES)

    def test_use_modes_cover_spec(self):
        for mode in ("none", "click", "hold", "hold_block", "charge"):
            self.assertIn(mode, USE_MODES)


if __name__ == "__main__":
    unittest.main()
