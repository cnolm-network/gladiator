/**
 * 观测编码（协议 §4.2，json-v1）。
 *
 * <p>把 mineflayer 的世界状态压成固定字段，Layer 3 只依赖这里输出的结构，
 * 不接触 mineflayer 的对象图。blockWindow 固定 7x4x7，palette 去重后按索引压平。
 */

import { Vec3 } from 'vec3';

export const OBSERVATION_FIELDS = Object.freeze([
  'tick', 'gameId', 'state', 'self', 'entities', 'blockWindow', 'restrictions', 'chat', 'pendingNotices',
]);

const WINDOW = Object.freeze({ width: 7, height: 4, depth: 7 });

function blockName(block) {
  if (!block) {
    return 'unknown';
  }
  if (typeof block.name === 'string' && block.name.length > 0) {
    return block.name;
  }
  return `id:${block.type}`;
}

function durabilityOf(item) {
  if (!item) {
    return null;
  }
  const max = Number.isFinite(item.maxDurability) && item.maxDurability > 0 ? item.maxDurability : null;
  const used = Number.isFinite(item.durabilityUsed) ? item.durabilityUsed : 0;
  return {
    used,
    max,
    // 没有耐久概念的物品（方块、苹果）按满耐久处理。
    remaining: max === null ? 1 : Math.max(0, Math.min(1, (max - used) / max)),
  };
}

function heldItemOf(bot) {
  const item = bot.heldItem;
  if (!item) {
    return null;
  }
  return { name: item.name, count: item.count, slot: item.slot, durability: durabilityOf(item) };
}

const ARMOR_SLOTS = Object.freeze({ head: 5, torso: 6, legs: 7, feet: 8 });

function armorOf(bot) {
  const slots = bot.inventory && Array.isArray(bot.inventory.slots) ? bot.inventory.slots : [];
  const armor = {};
  for (const [part, slot] of Object.entries(ARMOR_SLOTS)) {
    const item = slots[slot];
    armor[part] = item
      ? { name: item.name, durability: durabilityOf(item) }
      : null;
  }
  return armor;
}

function armorRemaining(armor) {
  const values = Object.values(armor)
    .filter(Boolean)
    .map((entry) => entry.durability.remaining);
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function inventoryOf(bot) {
  const slots = bot.inventory && Array.isArray(bot.inventory.slots) ? bot.inventory.slots : [];
  const items = [];
  for (let index = 0; index < slots.length; index += 1) {
    const item = slots[index];
    if (item) {
      items.push({ slot: index, name: item.name, count: item.count });
    }
  }
  return items;
}

function entityKind(entity) {
  if (entity.type === 'player') {
    return 'player';
  }
  if (entity.kind) {
    return entity.kind;
  }
  return entity.type || 'unknown';
}

function entityName(entity) {
  return entity.username || entity.displayName || entity.name || String(entity.id);
}

function encodeBlockWindow(bot) {
  const position = bot.entity && bot.entity.position ? bot.entity.position : new Vec3(0, 0, 0);
  const origin = new Vec3(
    Math.floor(position.x) - Math.floor(WINDOW.width / 2),
    Math.floor(position.y) - 1,
    Math.floor(position.z) - Math.floor(WINDOW.depth / 2),
  );
  const palette = [];
  const paletteIndex = new Map();
  const data = [];
  for (let y = 0; y < WINDOW.height; y += 1) {
    for (let z = 0; z < WINDOW.depth; z += 1) {
      for (let x = 0; x < WINDOW.width; x += 1) {
        const block = bot.blockAt(new Vec3(origin.x + x, origin.y + y, origin.z + z));
        const name = blockName(block);
        let index = paletteIndex.get(name);
        if (index === undefined) {
          index = palette.length;
          palette.push(name);
          paletteIndex.set(name, index);
        }
        data.push(index);
      }
    }
  }
  return {
    origin: { x: origin.x, y: origin.y, z: origin.z },
    size: [WINDOW.width, WINDOW.height, WINDOW.depth],
    palette,
    data,
  };
}

/**
 * 生成一帧观测。
 *
 * @param {object} bot      mineflayer Bot
 * @param {object} options  {tick, gameId, state, restrictions, chat, pendingNotices}
 */
export function buildObservation(bot, {
  tick = 0,
  gameId = '',
  state = 'Alive',
  restrictions = [],
  chat = [],
  pendingNotices = [],
} = {}) {
  const self = bot.entity || {};
  const position = self.position || new Vec3(0, 0, 0);
  const entities = [];
  const all = bot.entities || {};
  for (const id of Object.keys(all)) {
    const entity = all[id];
    if (!entity || entity === bot.entity) {
      continue;
    }
    const pos = entity.position || new Vec3(0, 0, 0);
    // 对手手持物：训练方需要它来判断“对面是不是握着剑”，否则同一距离下
    // 对手拿剑/空手的威胁完全不同，策略会学到互相矛盾的经验。
    const holding = entity.heldItem && entity.heldItem.name && entity.heldItem.name !== 'air'
      ? { name: entity.heldItem.name, count: entity.heldItem.count || 1 }
      : null;
    entities.push({
      id: entity.id,
      kind: entityKind(entity),
      name: entityName(entity),
      pos: { x: pos.x, y: pos.y, z: pos.z },
      distance: position.distanceTo(pos),
      heldItem: holding,
    });
  }
  entities.sort((a, b) => a.distance - b.distance);
  const armor = armorOf(bot);
  const held = heldItemOf(bot);
  return {
    tick,
    gameId,
    state,
    self: {
      pos: { x: position.x, y: position.y, z: position.z },
      yaw: self.yaw || 0,
      pitch: self.pitch || 0,
      onGround: Boolean(self.onGround),
      health: bot.health,
      food: bot.food,
      saturation: bot.foodSaturation,
      heldItem: held,
      armor,
      armorRemaining: armorRemaining(armor),
      inventory: inventoryOf(bot),
    },
    entities,
    blockWindow: encodeBlockWindow(bot),
    restrictions,
    chat: chat.slice(-20),
    pendingNotices: pendingNotices.slice(-10),
  };
}
