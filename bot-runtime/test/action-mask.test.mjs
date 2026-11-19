/**
 * action mask（运行时侧）与 items.json 加载的单元测试。
 *
 * 判据与 `training/gladiator_env/spaces.py::action_mask` 必须一致：
 * 空槽 → slot 屏蔽；手持物品不能右键 → use 只剩「不按」。
 *
 * 最后一个用例是「一票判据」的运行时版本：**只在 items.json 里加一条**
 * `minecraft:potion`，不改任何代码，mask 就必须认得它。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  CATEGORIES,
  DEFAULT_ITEMS_PATH,
  HOTBAR_SLOTS,
  USE_MODES,
  canonicalName,
  diffMasks,
  loadItemTable,
  matchingMask,
  resolveItemsPath,
} from '../src/action-mask.js';

/** 造一个「背包里有东西」的假 bot：slots 是「槽号 → 物品」数组。 */
function makeBot({ hotbar = {}, held = null, extraSlots = [] } = {}) {
  const slots = new Array(46).fill(null);
  for (const [index, name] of Object.entries(hotbar)) {
    slots[Number(index)] = { name, count: 1 };
  }
  for (const [index, name] of extraSlots) {
    slots[Number(index)] = { name, count: 1 };
  }
  return {
    username: 'Glad_Bot_1',
    inventory: { slots },
    heldItem: held ? { name: held, count: 1 } : null,
  };
}

function tempTable(payload) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glad-items-'));
  const file = path.join(dir, 'items.json');
  fs.writeFileSync(file, JSON.stringify(payload), 'utf8');
  return file;
}

test('默认能力表存在且能读到内置条目', () => {
  assert.ok(fs.existsSync(DEFAULT_ITEMS_PATH), DEFAULT_ITEMS_PATH);
  const table = loadItemTable();
  assert.equal(table.path, DEFAULT_ITEMS_PATH);
  assert.ok(table.size >= 8);
  assert.equal(table.category('diamond_sword'), 'sword');
  assert.equal(table.useSpec('golden_apple').mode, 'hold');
  assert.equal(table.useSpec('golden_apple').ticks, 32);
  assert.equal(table.useSpec('fishing_rod').mode, 'click');
  assert.equal(table.useSpec('bow').mode, 'charge');
  assert.equal(table.category('mystery_widget'), 'misc');
  assert.equal(table.useSpec('mystery_widget').mode, 'none');
});

test('通配与正则都能匹配', () => {
  const table = loadItemTable();
  assert.equal(table.category('iron_sword'), 'sword');
  assert.equal(table.category('stone_sword'), 'sword');
  assert.equal(table.category('splash_potion'), 'potion');

  const file = tempTable({ items: {
    're:^.*_(pickaxe|axe)$': { category: 'tool', use: { mode: 'click', ticks: 2 } },
  } });
  const custom = loadItemTable(file);
  assert.equal(custom.category('diamond_pickaxe'), 'tool');
  assert.equal(custom.category('diamond_sword'), 'misc');
});

test('canonicalName 归一命名空间与大小写', () => {
  assert.equal(canonicalName('  Minecraft:Diamond_Sword '), 'diamond_sword');
  assert.equal(canonicalName(null), '');
});

test('缺文件时报错里带绝对路径', () => {
  const missing = path.join(os.tmpdir(), 'no-such-dir-items.json');
  assert.throws(() => loadItemTable(missing), (error) => {
    assert.match(error.message, /找不到物品能力表/);
    assert.ok(error.message.includes(path.resolve(missing)), error.message);
    assert.match(error.message, /--items-json/);
    return true;
  });
});

test('resolveItemsPath 支持显式路径与默认路径', () => {
  assert.equal(resolveItemsPath(), DEFAULT_ITEMS_PATH);
  assert.equal(resolveItemsPath('../training/gladiator_env/items.json'),
    path.resolve('../training/gladiator_env/items.json'));
});

test('非法分类/使用方式立刻报错（别让 mask 悄悄算错）', () => {
  assert.throws(() => loadItemTable(tempTable({ items: { x: { category: 'banana' } } })), /banana/);
  assert.throws(() => loadItemTable(tempTable({ items: {
    x: { category: 'tool', use: { mode: 'wiggle' } },
  } })), /wiggle/);
});

test('扁平写法（规格 §2.3 的示例）也能读', () => {
  const file = tempTable({
    'minecraft:potion': { category: 'potion', use: { mode: 'click', ticks: 2 } },
  });
  const table = loadItemTable(file);
  assert.equal(table.category('potion'), 'potion');
});

test('mask 形状与 Python 的 ACTION_HEADS_V2 一致', () => {
  const mask = matchingMask(makeBot(), loadItemTable());
  assert.deepEqual(Object.keys(mask.heads), [
    'move', 'look_yaw', 'look_pitch', 'attack', 'use', 'jump', 'sneak', 'sprint', 'slot',
  ]);
  assert.deepEqual(mask.heads.move.length, 5);
  assert.deepEqual(mask.heads.look_yaw.length, 7);
  assert.deepEqual(mask.heads.look_pitch.length, 5);
  assert.deepEqual(mask.heads.attack.length, 2);
  assert.deepEqual(mask.heads.use.length, 3);
  assert.deepEqual(mask.heads.jump.length, 2);
  assert.deepEqual(mask.heads.sneak.length, 2);
  assert.deepEqual(mask.heads.sprint.length, 2);
  assert.deepEqual(mask.heads.slot.length, HOTBAR_SLOTS);
});

test('空背包：所有槽位被屏蔽，use 只剩「不按」', () => {
  const mask = matchingMask(makeBot(), loadItemTable());
  assert.deepEqual(mask.slot, new Array(HOTBAR_SLOTS).fill(false));
  assert.deepEqual(mask.use, [true, false, false]);
});

test('热键栏有物品就点亮对应槽位，背包里不算', () => {
  const bot = makeBot({
    hotbar: { 36: 'diamond_sword', 37: 'golden_apple' },
    held: 'diamond_sword',
    extraSlots: [[20, 'fishing_rod']],
  });
  const mask = matchingMask(bot, loadItemTable());
  assert.deepEqual(mask.slot, [true, true, false, false, false, false, false, false, false]);
  assert.deepEqual(mask.use, [true, true, true]);
});

test('手持不可用物品时 use 被屏蔽（即使背包里有别的）', () => {
  const bot = makeBot({ hotbar: { 36: 'mystery_widget', 37: 'diamond_sword' }, held: 'mystery_widget' });
  const mask = matchingMask(bot, loadItemTable());
  assert.deepEqual(mask.use, [true, false, false]);
});

test('空手时 use 被屏蔽', () => {
  const mask = matchingMask(makeBot({ hotbar: { 36: 'diamond_sword' }, held: 'air' }), loadItemTable());
  assert.deepEqual(mask.use, [true, false, false]);
});

test('diffMasks 能指出环境与运行时的差异', () => {
  const bot = makeBot({ hotbar: { 36: 'golden_apple' }, held: 'golden_apple' });
  const runtime = matchingMask(bot, loadItemTable());
  const env = { heads: { ...runtime.heads, use: [true, false, false] } };
  const diffs = diffMasks(env, runtime);
  assert.deepEqual(diffs, [
    { head: 'use', index: 1, env: false, runtime: true },
    { head: 'use', index: 2, env: false, runtime: true },
  ]);
  assert.deepEqual(diffMasks(runtime, runtime), []);
});

test('一票判据：只在 items.json 加一条 potion，mask 立刻认得（零改代码）', () => {
  // 用默认表（不含 potion 精确项）→ 裸 potion 是 misc、不能用右键
  const base = loadItemTable();
  const bot = makeBot({ hotbar: { 40: 'potion' }, held: 'potion' });
  assert.equal(base.category('potion'), 'misc');
  assert.deepEqual(matchingMask(bot, base).use, [true, false, false]);

  // 临时表 = 默认表条目 + 一条 minecraft:potion（模拟「临时加一条」）
  const raw = JSON.parse(fs.readFileSync(DEFAULT_ITEMS_PATH, 'utf8'));
  raw.items['minecraft:potion'] = {
    category: 'potion', use: { mode: 'click', ticks: 2 }, cooldownTicks: 0,
  };
  const table = loadItemTable(tempTable(raw));

  assert.equal(table.category('potion'), 'potion');
  assert.deepEqual(table.useSpec('potion'), { mode: 'click', ticks: 2 });
  const mask = matchingMask(bot, table);
  assert.deepEqual(mask.slot, [false, false, false, false, true, false, false, false, false]);
  assert.deepEqual(mask.use, [true, true, true]);
});

test('常量表覆盖规格里的分类与使用方式', () => {
  for (const name of ['sword', 'tool', 'food', 'block', 'bow', 'potion', 'misc']) {
    assert.ok(CATEGORIES.includes(name), name);
  }
  for (const mode of ['none', 'click', 'hold', 'hold_block', 'charge']) {
    assert.ok(USE_MODES.includes(mode), mode);
  }
});
