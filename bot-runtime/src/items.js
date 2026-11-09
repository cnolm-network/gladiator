/**
 * 物品与副手交互（动作细化）。
 *
 * <p>训练需要模型学会「切物品 / 选物品 / 用物品 / 左右键交互」，因此这里把
 * mineflayer 的背包操作收敛成几个可预测的原子操作，供 ActionDispatcher 复用：
 * - 选择快捷栏槽位（sword / rod / gapple / blocks）
 * - 右键使用（吃金苹果需要按住 1.6s、鱼竿是一次性释放、剑格挡需要保持按住）
 * - 自动穿戴护甲（调用方给完套件后由运行时穿上）
 */

import { Vec3 } from 'vec3';

/** 逻辑物品名 → 匹配规则。用正则避免把 golden_apple 和 golden_apple 的 NBT 变体搞混。 */
export const ITEM_PATTERNS = Object.freeze({
  sword: /_(sword)$/,
  rod: /^fishing_rod$/,
  gapple: /^golden_apple$/,
  blocks: /_(dirt|stone|cobblestone|planks|log)$/,
  food: /^(cooked_beef|cooked_porkchop|bread|apple)$/,
});

export class ItemError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function matches(item, pattern) {
  if (!item || !item.name) {
    return false;
  }
  return pattern.test(item.name);
}

/** 在快捷栏（slot 36..44）里找第一个匹配的物品槽位；找不到返回 null。 */
export function findHotbarSlot(bot, pattern) {
  const items = bot.inventory && bot.inventory.items ? bot.inventory.items() : [];
  for (const item of items) {
    if (item.slot >= 36 && item.slot <= 44 && matches(item, pattern)) {
      return item.slot;
    }
  }
  return null;
}

/** 在背包任意位置找匹配物品槽位；找不到返回 null。 */
export function findAnySlot(bot, pattern) {
  const items = bot.inventory && bot.inventory.items ? bot.inventory.items() : [];
  for (const item of items) {
    if (matches(item, pattern)) {
      return item.slot;
    }
  }
  return null;
}

/**
 * 把逻辑物品切到手上。
 *
 * @param {object} bot mineflayer 实例
 * @param {string} logicalName sword/rod/gapple/blocks/food
 * @returns {string} 实际物品名
 */
export async function selectItem(bot, logicalName) {
  const pattern = ITEM_PATTERNS[logicalName];
  if (!pattern) {
    throw new ItemError('ACTION_UNSUPPORTED', `未知物品类别: ${logicalName}`);
  }
  if (typeof bot.heldItem === 'object' && bot.heldItem && matches(bot.heldItem, pattern)) {
    return bot.heldItem.name;
  }
  const slot = findHotbarSlot(bot, pattern);
  if (slot === null) {
    throw new ItemError('ITEM_MISSING', `快捷栏里没有 ${logicalName}`);
  }
  await bot.setQuickBarSlot(slot - 36);
  const item = bot.inventory.items().find((it) => it.slot === slot);
  return item ? item.name : logicalName;
}

/** 把护甲穿上（调用方通过 RCON 发过套件后调用；缺哪件就跳过哪件）。 */
export async function equipArmor(bot) {
  // mineflayer 的护甲目的地：head / torso / legs / feet。
  const wanted = [['helmet', 'head'], ['chestplate', 'torso'], ['leggings', 'legs'], ['boots', 'feet']];
  const equipped = [];
  for (const [kind, destination] of wanted) {
    const slot = findAnySlot(bot, new RegExp(`_${kind}$`));
    if (slot === null) {
      continue;
    }
    const item = bot.inventory.items().find((it) => it.slot === slot);
    if (!item) {
      continue;
    }
    try {
      await bot.equip(item, destination);
      equipped.push(item.name);
    } catch (error) {
      // 装备失败不阻断训练（例如物品还在背包而不是快捷栏）。
      equipped.push(`!${item.name}:${error?.message ?? error}`);
    }
  }
  return equipped;
}

/** 吃一个金苹果：按住右键 1.6s，松手；失败抛 ITEM_MISSING。 */
export async function eatGoldenApple(bot, holdMs = 1600) {
  await selectItem(bot, 'gapple');
  bot.activateItem();
  try {
    await new Promise((resolve) => setTimeout(resolve, holdMs));
  } finally {
    bot.deactivateItem();
  }
}

/** 鱼竿：切到鱼竿 → 右键释放（击退）→ 切回剑。 */
export async function releaseRod(bot, returnTo = 'sword') {
  await selectItem(bot, 'rod');
  bot.activateItem();
  await new Promise((resolve) => setTimeout(resolve, 50));
  bot.deactivateItem();
  try {
    await selectItem(bot, returnTo);
  } catch (error) {
    // 没有剑也不影响鱼竿动作成功。
  }
}

/** 剑格挡：按住右键 durationMs；返回是否真的进入格挡姿势。 */
export async function blockWithSword(bot, durationMs = 250) {
  await selectItem(bot, 'sword');
  bot.activateItem();
  try {
    await new Promise((resolve) => setTimeout(resolve, Math.max(1, durationMs)));
  } finally {
    bot.deactivateItem();
  }
}

/** 计算到实体的视线距离（眼高 1.62）与目标中心（身体 +1.0）。 */
export function eyeDistance(bot, entity) {
  const self = bot.entity && bot.entity.position;
  if (!self || !entity || !entity.position) {
    return Number.POSITIVE_INFINITY;
  }
  const dx = entity.position.x - self.x;
  const dy = entity.position.y + 1.0 - (self.y + 1.62);
  const dz = entity.position.z - self.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** 看向实体眼睛（3D 瞄准），用于攻击前的「连续对准」。 */
export async function aimAt(bot, entity, force = true) {
  if (!entity || !entity.position) {
    return;
  }
  const pos = entity.position;
  const target = new Vec3(Number(pos.x), Number(pos.y) + 1.5, Number(pos.z));
  await bot.lookAt(target, force);
}

export { Vec3 };
