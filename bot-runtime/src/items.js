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

/** 错误信息里的中文名：训练日志给人看，写“背包里没有剑”比“没有 sword”更好读。 */
export const ITEM_LABELS = Object.freeze({
  sword: '剑',
  rod: '鱼竿',
  gapple: '金苹果',
  blocks: '方块',
  food: '食物',
});

const HOTBAR_FIRST = 36;
const HOTBAR_LAST = 44;

function inventoryItems(bot) {
  const inventory = bot && bot.inventory;
  if (!inventory || typeof inventory.items !== 'function') {
    return null;
  }
  const items = inventory.items();
  return Array.isArray(items) ? items : null;
}

/**
 * 选一个快捷栏目标槽：优先空槽（从 36 起），全满时返回 null（调用方改为与手持槽互换）。
 */
function pickFreeHotbarSlot(bot) {
  const items = inventoryItems(bot) || [];
  const used = new Set(items.filter((it) => it.slot >= HOTBAR_FIRST && it.slot <= HOTBAR_LAST)
    .map((it) => it.slot));
  for (let slot = HOTBAR_FIRST; slot <= HOTBAR_LAST; slot += 1) {
    if (!used.has(slot)) {
      return slot;
    }
  }
  return null;
}

function currentHandSlot(bot) {
  const index = Number.isInteger(bot.quickBarSlot) ? bot.quickBarSlot : 0;
  return HOTBAR_FIRST + Math.min(Math.max(index, 0), 8);
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
  const items = inventoryItems(bot);
  const hotbarSlot = findHotbarSlot(bot, pattern);
  if (hotbarSlot !== null) {
    await bot.setQuickBarSlot(hotbarSlot - 36);
    const item = items ? items.find((it) => it.slot === hotbarSlot) : null;
    return item ? item.name : logicalName;
  }
  // 快捷栏没有但背包里有：先挪进快捷栏再选中，否则训练里会出现“有剑却报 ITEM_MISSING”。
  const anySlot = findAnySlot(bot, pattern);
  if (anySlot !== null) {
    const free = pickFreeHotbarSlot(bot);
    const target = free === null ? currentHandSlot(bot) : free;
    if (typeof bot.moveSlotItem === 'function') {
      await bot.moveSlotItem(anySlot, target);
    } else if (typeof bot.moveSlotItem === 'undefined') {
      // 没有窗口操作能力（少数只读封装）：只能如实报“不在快捷栏”。
      throw new ItemError('ITEM_MISSING', `${ITEM_LABELS[logicalName] || logicalName}在背包里但不在快捷栏`);
    }
    await bot.setQuickBarSlot(target - HOTBAR_FIRST);
    const item = items ? items.find((it) => it.slot === target) : null;
    return item ? item.name : logicalName;
  }
  const label = ITEM_LABELS[logicalName] || logicalName;
  throw new ItemError('ITEM_MISSING', `背包里没有${label}`);
}

/**
 * 护甲穿戴的时序参数（导出便于运行时调参与测试缩短等待）。
 *
 * <p>背景（真机）：1.8 的服务端在客户端点击背包时回一个窗口事务确认，mineflayer 等不到就
 * 报 `Server didn't respond to transaction for clicking on slot N`。此时客户端本地库存已被
 * 改写（所以运行时日志说「已穿戴护甲」），服务端却是空的——护甲实际没穿上。
 * 因此每次 equip 必须有超时兜底、失败后复位窗口再重试，并对同一 bot 做冷却限流。
 */
export const ARMOR_TIMING = Object.freeze({
  timeoutMs: 3000,
  retryDelayMs: 300,
  cooldownMs: 2000,
});

/** mineflayer 的护甲目的地：head / torso / legs / feet。 */
const ARMOR_SLOTS = Object.freeze([
  ['helmet', 'head'],
  ['chestplate', 'torso'],
  ['leggings', 'legs'],
  ['boots', 'feet'],
]);

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resolveArmorTiming(options) {
  const opts = options || {};
  const pick = (value, fallback) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
  return {
    timeoutMs: Math.max(1, pick(opts.timeoutMs, ARMOR_TIMING.timeoutMs)),
    retryDelayMs: Math.max(0, pick(opts.retryDelayMs, ARMOR_TIMING.retryDelayMs)),
    cooldownMs: Math.max(0, pick(opts.cooldownMs, ARMOR_TIMING.cooldownMs)),
  };
}

/** 每 bot 一份的「已警告过的失败原因」集合，用来去重刷屏。 */
function armorWarnedSet(bot) {
  if (!(bot.__armorWarned instanceof Set)) {
    bot.__armorWarned = new Set();
  }
  return bot.__armorWarned;
}

function warnArmorOnce(bot, key, message) {
  const warned = armorWarnedSet(bot);
  if (warned.has(key)) {
    return false;
  }
  warned.add(key);
  console.warn(message);
  return true;
}

/**
 * 失败归类：日志去重按「件 + 类别」做键。
 * 不能直接用原始 message——事务错误里带槽位号（`slot 11`），换个槽位就会被当成新原因刷屏。
 */
function classifyEquipFailure(error) {
  if (error && error.code === 'EQUIP_TIMEOUT') {
    return 'timeout';
  }
  const message = String(error?.message ?? error ?? '');
  if (/respond to transaction/i.test(message)) {
    return 'transaction';
  }
  return 'error';
}

/**
 * 单次 equip，带硬超时。
 *
 * <p>超时后不抛到调用方外面（由调用方捕获），且迟到 settle 的 equip 也已挂空 catch，
 * 不会以 unhandled rejection 的形式打崩进程。
 */
async function equipWithTimeout(bot, item, destination, timeoutMs) {
  let timer = null;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(`服务端 ${timeoutMs}ms 内未确认穿 ${item.name}`);
      error.code = 'EQUIP_TIMEOUT';
      reject(error);
    }, timeoutMs);
  });
  // 谁先 settle 谁说话：equip 迟到 reject 时 race 已结束，这里必须自己吞掉。
  const attempt = Promise.resolve().then(() => bot.equip(item, destination));
  attempt.catch(() => {});
  try {
    await Promise.race([attempt, guard]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

/**
 * 尽力复位窗口状态。
 *
 * <p>1.8 的窗口点击带「事务确认」，服务端不回确认后两侧状态就不一致；仅靠重试会一直踩同一个
 * 不一致状态。复位手段有两个：
 * 1. `closeWindow(currentWindow)`——有打开的容器窗口时把它关掉，让 mineflayer 重建 inventory；
 * 2. `clickWindow(-999, 0, 0)`——mineflayer 自己关窗后也这么干，用途是清光标栈并同步 state id。
 *
 * <p>两种调用都可能抛（`closeWindow(undefined)` 会读 `window.id` 直接 TypeError；对自家背包
 * 窗口调 closeWindow 在 1.8 下也会抛），这里全部忽略——复位是尽力而为，失败不算穿戴失败。
 */
function closeWindowQuietly(bot) {
  if (typeof bot.closeWindow === 'function') {
    try {
      const result = bot.closeWindow(bot.currentWindow || undefined);
      // 有的封装把 closeWindow 写成 async：失败形式是 rejected promise 而非同步抛，
      // 同样不能漏成 unhandled rejection 打崩进程。
      if (result && typeof result.catch === 'function') {
        result.catch(() => {});
      }
    } catch (error) {
      // 忽略：见上。
    }
  }
  if (typeof bot.clickWindow === 'function') {
    try {
      const result = bot.clickWindow(-999, 0, 0);
      if (result && typeof result.catch === 'function') {
        result.catch(() => {});
      }
    } catch (error) {
      // 忽略：复位尽力而为。
    }
  }
}

/**
 * 穿一件护甲：失败后复位窗口、等 retryDelayMs、重试一次。
 * @returns {Promise<{ok: boolean, error?: Error, attempts: number}>}
 */
async function equipOne(bot, item, destination, timing) {
  let attempts = 0;
  let lastError = null;
  for (let round = 0; round < 2; round += 1) {
    if (round > 0) {
      // 窗口事务超时意味着两侧状态已经不一致：先关窗口让 mineflayer 重建 inventory，
      // 等一下再试，否则重试必然踩同一个不一致状态。
      closeWindowQuietly(bot);
      await delay(timing.retryDelayMs);
    }
    attempts += 1;
    try {
      // 每轮都按名字重新解析槽位：关闭窗口后槽位可能变化，旧引用可能已失效。
      const current = bot.inventory && bot.inventory.items
        ? bot.inventory.items().find((it) => it.name === item.name)
        : null;
      await equipWithTimeout(bot, current || item, destination, timing.timeoutMs);
      return { ok: true, attempts };
    } catch (error) {
      lastError = error;
    }
  }
  return { ok: false, error: lastError, attempts };
}

/**
 * 把护甲穿上（调用方通过 RCON 发过套件后调用；缺哪件就跳过哪件）。
 *
 * <p>永不抛异常：单件的失败一律折算成返回值里的 `!物品名:原因` 条目，避免穿戴失败
 * 冒泡打断 ActionDispatcher 的动作循环。
 *
 * <p>同一 bot 两次穿甲尝试至少间隔 `cooldownMs`（默认 2000ms），冷却期内直接返回空数组
 * （空操作、不报错）——训练里每 800ms 轮询一次，不节流会持续撞事务超时。
 *
 * @returns {Promise<string[]>} 已穿上的物品名；失败项形如 `!diamond_boots:原因`
 */
export async function equipArmor(bot, options = {}) {
  const timing = resolveArmorTiming(options);
  const equipped = [];
  if (!bot || !bot.inventory || typeof bot.inventory.items !== 'function') {
    return equipped;
  }

  // 先算出这轮要穿哪几件：没有可穿物品时不该消耗冷却窗口（否则 RCON give 后的
  // 第一次有效尝试会被前一次空轮询挡掉最多 cooldownMs）。
  const pending = [];
  for (const [kind, destination] of ARMOR_SLOTS) {
    let item = null;
    try {
      const slot = findAnySlot(bot, new RegExp(`_${kind}$`));
      item = slot === null ? null : bot.inventory.items().find((it) => it.slot === slot) || null;
    } catch (error) {
      item = null;
    }
    if (item) {
      pending.push({ item, destination });
    }
  }
  if (pending.length === 0) {
    return equipped;
  }

  const now = Date.now();
  const lastAttemptAt = Number(bot.__armorAttemptAt) || 0;
  if (lastAttemptAt > 0 && now - lastAttemptAt < timing.cooldownMs) {
    return equipped;
  }
  bot.__armorAttemptAt = now;

  for (const { item, destination } of pending) {
    const result = await equipOne(bot, item, destination, timing);
    if (result.ok) {
      equipped.push(item.name);
      continue;
    }
    const reason = String(result.error?.message ?? result.error ?? '未知原因');
    equipped.push(`!${item.name}:${reason}`);
    warnArmorOnce(
      bot,
      `${item.name}:${classifyEquipFailure(result.error)}`,
      `[armor] 穿戴 ${item.name} 失败（已重试 ${result.attempts} 次，本轮跳过）：${reason}`,
    );
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
