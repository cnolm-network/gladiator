/**
 * action mask（v2）：按背包算 `slot` / `use` 的可用性，与训练侧 `spaces.action_mask` 对齐。
 *
 * <p>为什么运行时也要算一遍：环境侧的 mask 来自**观测**（200ms 一帧），运行时的背包是实时的，
 * 两者偶发不一致（刚 give 完、刚吃掉一个苹果）。这里提供同一套判据，供：
 * - 诊断日志（"训练侧说 use 可用、运行时说不可用" 就是数据不同步的证据）；
 * - 一致性校验（自博弈/评测时可以按差异计数，而不是等策略学歪）。
 *
 * <p>判据与 Python 完全一致（见 `training/gladiator_env/spaces.py::action_mask`）：
 * - `slot[i]`：热键栏第 i 格（原始槽 36..44）有物品；
 * - `use[0]` 恒可用；`use[1..2]` 需要**手持物品**在能力表里 `mode != none`；
 * - 其余头恒可用（reach 屏蔽是 attack-guard 的职责，不在这里）。
 *
 * <p>`items.json` 由 `--items-json` 指定，默认 `../training/gladiator_env/items.json`；
 * 文件缺失时抛出的错误里带上**解析后的绝对路径**——这是排查「改了 items.json 不生效」
 * 的唯一线索。
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HEAD_ORDER_V2 } from './action-space-v2.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** 允许的分类（与 Python 的 `items.CATEGORIES` 同序）。 */
export const CATEGORIES = Object.freeze(['sword', 'tool', 'food', 'block', 'bow', 'potion', 'misc']);
/** 允许的使用方式（与 Python 的 `items.USE_MODES` 同序）。 */
export const USE_MODES = Object.freeze(['none', 'click', 'hold', 'hold_block', 'charge']);
/** 热键栏 9 格在协议观测/背包里的起始原始槽号。 */
export const HOTBAR_SLOT_BASE = 36;
/** 热键栏格数。 */
export const HOTBAR_SLOTS = 9;
/** 默认能力表路径（`bot-runtime/src` → 仓库根 → `training/gladiator_env/items.json`）。 */
export const DEFAULT_ITEMS_PATH = path.resolve(HERE, '..', '..', 'training', 'gladiator_env', 'items.json');

/** 未匹配任何条目时的能力（与 Python 的 `MISC_CAPABILITY` 一致）。 */
const MISC_CAPABILITY = Object.freeze({
  category: 'misc',
  use: Object.freeze({ mode: 'none', ticks: 0 }),
  cooldownTicks: 0,
  reach: null,
  stackGoal: 1,
  projectile: false,
});

/** 物品名归一：去命名空间、小写、去空白（观测用裸名，items.json 用带命名空间的名字）。 */
export function canonicalName(name) {
  let text = String(name === undefined || name === null ? '' : name).trim().toLowerCase();
  if (text.startsWith('minecraft:')) {
    text = text.slice('minecraft:'.length);
  }
  return text;
}

function compilePattern(pattern) {
  const text = String(pattern || '').trim();
  if (text.startsWith('re:')) {
    return new RegExp(text.slice(3));
  }
  const canonical = canonicalName(text);
  if (!canonical) {
    return null;
  }
  if (canonical.includes('*') || canonical.includes('?')) {
    // 先把所有元字符转义，再把 \* / \? 还原成通配（否则 `^*_sword$` 这种是非法正则）。
    const escaped = canonical
      .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      .replace(/\\\*/g, '.*')
      .replace(/\\\?/g, '.');
    return new RegExp(`^${escaped}$`);
  }
  return null;
}

function coerceEntry(label, raw) {
  if (!raw || typeof raw !== 'object') {
    throw new Error(`items.json 的 ${label} 必须是对象`);
  }
  const category = String(raw.category || 'misc').trim().toLowerCase();
  if (!CATEGORIES.includes(category)) {
    throw new Error(`items.json 的 ${label}.category 不认识：${category}（允许 ${CATEGORIES.join(', ')}）`);
  }
  const useRaw = raw.use && typeof raw.use === 'object' ? raw.use : {};
  const mode = String(useRaw.mode || 'none').trim().toLowerCase();
  if (!USE_MODES.includes(mode)) {
    throw new Error(`items.json 的 ${label}.use.mode 不认识：${mode}（允许 ${USE_MODES.join(', ')}）`);
  }
  const ticks = Math.max(0, Math.trunc(Number(useRaw.ticks) || 0));
  const reach = raw.reach === undefined || raw.reach === null ? null : Number(raw.reach);
  return {
    category,
    use: { mode, ticks },
    cooldownTicks: Math.max(0, Math.trunc(Number(raw.cooldownTicks) || 0)),
    reach: Number.isFinite(reach) ? reach : null,
    stackGoal: Math.max(1, Math.trunc(Number(raw.stackGoal) || 1)),
    projectile: Boolean(raw.projectile),
  };
}

/**
 * 解析 items.json 路径：显式参数优先，否则用默认路径。
 *
 * @param {string|null} explicit `--items-json` 的值
 * @returns {string} 绝对路径
 */
export function resolveItemsPath(explicit = null) {
  const target = explicit && String(explicit).trim() ? String(explicit).trim() : DEFAULT_ITEMS_PATH;
  return path.resolve(target);
}

/**
 * 读能力表。
 *
 * @param {string|null} filePath `--items-json` 的值（可空）
 * @returns {{path:string, size:number, capability:Function, useSpec:Function,
 *            reach:Function, category:Function, patterns:Function}}
 */
export function loadItemTable(filePath = null) {
  const absolute = resolveItemsPath(filePath);
  if (!fs.existsSync(absolute) || !fs.statSync(absolute).isFile()) {
    throw new Error(
      `找不到物品能力表 items.json：${absolute}`
      + (filePath ? `（来自 --items-json ${filePath}）` : '（默认路径，可用 --items-json 指定）'),
    );
  }
  let data;
  try {
    data = JSON.parse(fs.readFileSync(absolute, 'utf8'));
  } catch (error) {
    throw new Error(`items.json 解析失败（${absolute}）：${error.message}`);
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error(`items.json 顶层必须是对象（${absolute}）`);
  }
  // 兼容规格 §2.3 的扁平写法：顶层直接是 name -> 配置。
  const entries = data.items && typeof data.items === 'object' && !Array.isArray(data.items)
    ? data.items
    : Object.fromEntries(Object.entries(data).filter(
      ([key]) => !key.startsWith('_') && key !== 'version',
    ));

  const exact = new Map();
  const patterns = [];
  for (const [label, raw] of Object.entries(entries)) {
    const capability = coerceEntry(label, raw);
    const matcher = compilePattern(label);
    if (matcher) {
      patterns.push({ label, matcher, capability });
    } else {
      const key = canonicalName(label);
      if (!key) {
        throw new Error(`items.json 里有空键名：${JSON.stringify(label)}`);
      }
      exact.set(key, capability);
    }
  }

  const lookup = (name) => {
    const key = canonicalName(name);
    if (key && exact.has(key)) {
      return { capability: exact.get(key), matched: key };
    }
    if (key) {
      for (const entry of patterns) {
        if (entry.matcher.test(key)) {
          return { capability: entry.capability, matched: entry.label };
        }
      }
    }
    return { capability: MISC_CAPABILITY, matched: null };
  };

  return {
    path: absolute,
    size: exact.size + patterns.length,
    /** 完整能力（拷贝；未知物品回 misc，不抛错）。 */
    capability(name) {
      const { capability, matched } = lookup(name);
      return {
        ...capability,
        use: { ...capability.use },
        name: canonicalName(name),
        matched,
      };
    },
    /** `{mode, ticks}`。 */
    useSpec(name) {
      return { ...lookup(name).capability.use };
    },
    /** 攻击/使用距离；表里没写就是 null。 */
    reach(name) {
      const value = lookup(name).capability.reach;
      return value === null || value === undefined ? null : Number(value);
    },
    /** 分类字符串（未知物品为 `misc`）。 */
    category(name) {
      return lookup(name).capability.category;
    },
    /** 表里声明的全部键（诊断用）。 */
    patterns() {
      return [...exact.keys(), ...patterns.map((entry) => entry.label)];
    },
  };
}

/** 手持物品名（没有就是空串）。 */
export function heldItemName(bot) {
  const held = bot && bot.heldItem;
  if (!held || typeof held.name !== 'string') {
    return '';
  }
  return held.name === 'air' ? '' : held.name;
}

/** 遍历背包里的物品（mineflayer 的 `inventory.slots` 是「槽号 → 物品」的数组）。 */
function inventorySlots(bot) {
  const inventory = bot && bot.inventory;
  if (!inventory) {
    return [];
  }
  if (Array.isArray(inventory.slots)) {
    return inventory.slots.map((item, index) => (item ? { slot: index, item } : null)).filter(Boolean);
  }
  if (typeof inventory.items === 'function') {
    const items = inventory.items();
    return Array.isArray(items) ? items : [];
  }
  return [];
}

/**
 * 按当前背包算 mask。
 *
 * @param {object} bot mineflayer Bot（测试用假对象：只需要 `inventory.slots` 与 `heldItem`）
 * @param {object} [itemTable] `loadItemTable()` 的返回值；省略时按默认路径加载
 * @returns {{slot:boolean[], use:boolean[], heads:object}}
 */
export function matchingMask(bot, itemTable = null) {
  const table = itemTable || loadItemTable();
  const slot = new Array(HOTBAR_SLOTS).fill(false);
  for (const entry of inventorySlots(bot)) {
    const raw = Number(entry.slot);
    if (!Number.isFinite(raw)) {
      continue;
    }
    const index = raw - HOTBAR_SLOT_BASE;
    if (index >= 0 && index < HOTBAR_SLOTS) {
      slot[index] = true;
    }
  }
  const name = heldItemName(bot);
  const usable = Boolean(name) && table.useSpec(name).mode !== 'none';
  const use = [true, usable, usable];
  const heads = {
    move: new Array(5).fill(true),
    look_yaw: new Array(7).fill(true),
    look_pitch: new Array(5).fill(true),
    attack: [true, true],
    use: [...use],
    jump: [true, true],
    sneak: [true, true],
    sprint: [true, true],
    slot: [...slot],
    // G-707/G-712：第 10 头恒定全开（与 Python `action_mask` 一致）。三种取值都不依赖
    // 物品：`release_w` 只动 forward 键，`block_hit` 由运行时自己先切剑。
    sprint_reset: [true, true, true],
  };
  return {
    slot,
    use,
    // 与 Python 的 info["actionMask"] 同形（十个头，顺序 = ACTION_HEADS_V2）。
    heads,
    // 十头数组形式（顺序 = HEAD_ORDER_V2），可直接当 `policy.actWithMask(vec, array)` 的入参。
    array: HEAD_ORDER_V2.map((head) => [...heads[head]]),
  };
}

/** 两套 mask 的差异（诊断/一致性校验用）；完全一致返回空数组。 */
export function diffMasks(left, right) {
  const diffs = [];
  const leftHeads = (left && left.heads) || left || {};
  const rightHeads = (right && right.heads) || right || {};
  for (const head of Object.keys(rightHeads)) {
    const a = leftHeads[head] || [];
    const b = rightHeads[head] || [];
    for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
      if (Boolean(a[index]) !== Boolean(b[index])) {
        diffs.push({ head, index, env: Boolean(a[index]), runtime: Boolean(b[index]) });
      }
    }
  }
  return diffs;
}
