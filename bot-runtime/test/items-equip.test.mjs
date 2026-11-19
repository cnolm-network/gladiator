/**
 * 护甲穿戴的事务超时（真机 P0）。
 *
 * <p>现象：1.8 服务端不回窗口点击事务确认，mineflayer 的 `equip` 报
 * `Server didn't respond to transaction for clicking on slot 11 on window with id 0`，
 * 但客户端本地库存已被改写——运行时日志说「已穿戴护甲」，服务端 `armor` 仍是 0。
 *
 * <p>这里用假 bot 覆盖四条关键路径，不启动 mineflayer：
 * 超时不挂死、失败后复位窗口并只重试一次、冷却期内空操作、同一失败原因只警告一次。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { ActionDispatcher } from '../src/actions.js';
import { ARMOR_TIMING, equipArmor } from '../src/items.js';

const ARMORY = ['diamond_helmet', 'diamond_chestplate', 'diamond_leggings', 'diamond_boots'];
const DESTINATIONS = { diamond_helmet: 'head', diamond_chestplate: 'torso', diamond_leggings: 'legs', diamond_boots: 'feet' };
// 测试用参数：把等待压到毫秒级，避免用例被真实的 3s/300ms 拖慢。
const FAST = { timeoutMs: 60, retryDelayMs: 5, cooldownMs: 0 };

/** 假 bot：inventory 是静态快照，equip/closeWindow 记调用序列，行为由 hook 决定。 */
function makeBot({ armors = ARMORY, equip, closeWindow, clickWindow, currentWindow = { id: 0 } } = {}) {
  const calls = [];
  const inventory = armors.map((name, index) => ({ name, slot: index, count: 1 }));
  const bot = {
    calls,
    currentWindow,
    inventory: {
      items: () => inventory.map((it) => ({ ...it })),
      on: () => {},
    },
    async closeWindow(window) {
      calls.push(['closeWindow', window]);
      if (closeWindow) {
        return closeWindow(window);
      }
      return undefined;
    },
    async clickWindow(mode, button, slot) {
      calls.push(['clickWindow', mode, button, slot]);
      if (clickWindow) {
        return clickWindow(mode, button, slot);
      }
      return undefined;
    },
    async equip(item, destination) {
      const attempt = calls.filter((entry) => entry[0] === 'equip').length + 1;
      calls.push(['equip', item.name, destination]);
      return equip ? equip(item, destination, attempt) : undefined;
    },
  };
  return bot;
}

const equipCalls = (bot) => bot.calls.filter((entry) => entry[0] === 'equip');
const closeCalls = (bot) => bot.calls.filter((entry) => entry[0] === 'closeWindow');
const clickCalls = (bot) => bot.calls.filter((entry) => entry[0] === 'clickWindow');

/** console.warn 捕获：断言「只打一次」，同时避免测试输出被刷屏。 */
function captureWarn() {
  const original = console.warn;
  const lines = [];
  console.warn = (...args) => {
    lines.push(args.map((arg) => String(arg)).join(' '));
  };
  return {
    lines,
    restore: () => {
      console.warn = original;
    },
  };
}

const transactionError = () => new Error("Server didn't respond to transaction for clicking on slot 11 on window with id 0.");

test('正常路径：四件护甲各穿一次，目的地映射正确，且不产生警告', async () => {
  const bot = makeBot();
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, FAST);
  } finally {
    warn.restore();
  }

  assert.deepEqual(worn, ARMORY);
  assert.equal(equipCalls(bot).length, 4);
  for (const [, name, destination] of equipCalls(bot)) {
    assert.equal(destination, DESTINATIONS[name], `${name} 应穿到 ${DESTINATIONS[name]}`);
  }
  assert.equal(closeCalls(bot).length, 0, '成功路径不该关窗口');
  assert.deepEqual(warn.lines, []);
});

test('事务错误：复位窗口后只重试一次，重试成功则视为穿上', async () => {
  const bot = makeBot({
    armors: ['diamond_chestplate'],
    equip: (item, destination, attempt) => {
      if (attempt === 1) {
        throw transactionError();
      }
      return undefined;
    },
  });
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, FAST);
  } finally {
    warn.restore();
  }

  assert.deepEqual(worn, ['diamond_chestplate']);
  assert.equal(equipCalls(bot).length, 2, '失败后应重试一次');
  assert.equal(closeCalls(bot).length, 1, '重试前应复位一次窗口');
  assert.deepEqual(closeCalls(bot)[0][1], bot.currentWindow, 'closeWindow 应拿到 currentWindow');
  assert.deepEqual(warn.lines, [], '重试成功后不该告警');
});

test('持续失败：重试仍失败就放弃这件，返回失败原因且不再尝试第三次', async () => {
  const bot = makeBot({
    armors: ['diamond_boots'],
    equip: () => {
      throw transactionError();
    },
  });
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, FAST);
  } finally {
    warn.restore();
  }

  assert.equal(worn.length, 1);
  assert.match(worn[0], /^!diamond_boots:/, '失败项应以 ! 前缀标注');
  assert.match(worn[0], /respond to transaction/);
  assert.equal(equipCalls(bot).length, 2, '总共两次尝试（首发 + 一次重试）');
  assert.equal(closeCalls(bot).length, 1);
  assert.equal(warn.lines.length, 1, '失败原因应告警一次');
  assert.match(warn.lines[0], /diamond_boots/);
});

test('挂起不返回：超时兜底，不挂死调用方', async () => {
  const hang = () => new Promise(() => {});
  const bot = makeBot({ armors: ['diamond_helmet'], equip: hang });
  const warn = captureWarn();
  const startedAt = Date.now();
  let worn;
  try {
    worn = await equipArmor(bot, { timeoutMs: 50, retryDelayMs: 5, cooldownMs: 0 });
  } finally {
    warn.restore();
  }
  const elapsed = Date.now() - startedAt;

  assert.ok(elapsed < 1500, `超时后应尽快返回，实际 ${elapsed}ms`);
  assert.equal(worn.length, 1);
  assert.match(worn[0], /^!diamond_helmet:/);
  assert.match(worn[0], /50ms 内未确认/);
  assert.equal(equipCalls(bot).length, 2, '超时后应重试一次');
  assert.equal(warn.lines.length, 1);
});

test('冷却：同一 bot 两次尝试间隔不足则空操作返回，不报错也不碰 equip', async () => {
  const bot = makeBot();
  bot.__armorAttemptAt = Date.now();
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, { cooldownMs: 2000 });
  } finally {
    warn.restore();
  }

  assert.deepEqual(worn, []);
  assert.equal(equipCalls(bot).length, 0);
  assert.deepEqual(warn.lines, []);

  // 冷却过期后应恢复真实尝试。
  bot.__armorAttemptAt = Date.now() - 5000;
  const after = await equipArmor(bot, { timeoutMs: 60, retryDelayMs: 5, cooldownMs: 2000 });
  assert.deepEqual(after, ARMORY);
  assert.equal(equipCalls(bot).length, 4);
});

test('无护甲可穿时不消耗冷却窗口（RCON give 后第一次有效尝试不被挡住）', async () => {
  const bot = makeBot({ armors: [] });
  assert.deepEqual(await equipArmor(bot, FAST), []);
  assert.equal(bot.__armorAttemptAt, undefined, '空盘点不该写冷却时间戳');

  // 同一 bot 换上护甲后立刻应能穿上（若空盘写了时间戳，这里会被 2000ms 冷却挡掉）。
  bot.inventory.items = () => ARMORY.map((name, index) => ({ name, slot: index, count: 1 }));
  assert.deepEqual(await equipArmor(bot, { ...FAST, cooldownMs: 2000 }), ARMORY);
});

test('同一件护甲的同一失败原因只警告一次（跨多次调用去重）', async () => {
  const bot = makeBot({
    armors: ['diamond_leggings'],
    equip: () => {
      throw transactionError();
    },
  });
  const warn = captureWarn();
  try {
    const first = await equipArmor(bot, FAST);
    const second = await equipArmor(bot, FAST);
    const third = await equipArmor(bot, FAST);
    assert.match(first[0], /^!diamond_leggings:/);
    assert.match(second[0], /^!diamond_leggings:/);
    assert.match(third[0], /^!diamond_leggings:/);
  } finally {
    warn.restore();
  }

  assert.equal(warn.lines.length, 1, `应只告警一次，实际 ${warn.lines.length} 次`);
  assert.equal(bot.__armorWarned.size, 1, '去重集合按「件 + 原因类别」只留一条');

  // 换一件护甲属于新的失败原因，应允许各自告警一次。
  const other = makeBot({
    armors: ['diamond_boots'],
    equip: () => {
      throw transactionError();
    },
  });
  other.__armorWarned = bot.__armorWarned;
  const warn2 = captureWarn();
  try {
    await equipArmor(other, FAST);
  } finally {
    warn2.restore();
  }
  assert.equal(warn2.lines.length, 1);
  assert.equal(bot.__armorWarned.size, 2);
});

test('closeWindow 抛错（对 inventory 窗口）被忽略，不影响重试', async () => {
  const bot = makeBot({
    armors: ['diamond_chestplate'],
    closeWindow: () => {
      throw new Error('Window is not open / cannot close inventory window');
    },
    equip: (item, destination, attempt) => {
      if (attempt === 1) {
        throw transactionError();
      }
      return undefined;
    },
  });
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, FAST);
  } finally {
    warn.restore();
  }

  assert.deepEqual(worn, ['diamond_chestplate']);
  assert.equal(equipCalls(bot).length, 2);
  assert.deepEqual(warn.lines, []);
});

test('无打开窗口时（currentWindow=null）closeWindow 抛 TypeError，仍能靠 -999 点击复位', async () => {
  // 真实 mineflayer 的 closeWindow 会读 window.id，传 undefined 直接 TypeError。
  const bot = makeBot({
    armors: ['diamond_boots'],
    currentWindow: null,
    closeWindow: (window) => {
      if (!window) {
        throw new TypeError("Cannot read properties of undefined (reading 'id')");
      }
      return undefined;
    },
    equip: (item, destination, attempt) => {
      if (attempt === 1) {
        throw transactionError();
      }
      return undefined;
    },
  });
  const warn = captureWarn();
  let worn;
  try {
    worn = await equipArmor(bot, FAST);
  } finally {
    warn.restore();
  }

  assert.deepEqual(worn, ['diamond_boots']);
  assert.equal(closeCalls(bot).length, 1);
  assert.equal(closeCalls(bot)[0][1], undefined);
  assert.deepEqual(clickCalls(bot), [['clickWindow', -999, 0, 0]], '应发一次 -999 光标/state 复位');
  assert.deepEqual(warn.lines, []);
});

test('ActionDispatcher 的 use(armor) 不把穿甲异常冒泡出去', async () => {
  const bot = makeBot({
    armors: ['diamond_chestplate'],
    equip: () => {
      throw transactionError();
    },
  });
  const warn = captureWarn();
  let result;
  try {
    const dispatcher = new ActionDispatcher({ bot });
    result = await dispatcher.use({ item: 'armor' });
  } finally {
    warn.restore();
  }

  assert.equal(result, undefined, 'use 应正常返回而不是抛错');
  assert.ok(Array.isArray(bot.armorWorn), 'armorWorn 应拿到结构化结果');
  assert.match(bot.armorWorn[0], /^!diamond_chestplate:/);
  assert.equal(warn.lines.length, 1, '失败告警一次，但不打断动作循环');
});

test('ARMOR_TIMING 默认值符合验收口径（3s 超时 / 300ms 复位 / 2s 冷却）', () => {
  assert.deepEqual({ ...ARMOR_TIMING }, { timeoutMs: 3000, retryDelayMs: 300, cooldownMs: 2000 });
  assert.equal(Object.isFrozen(ARMOR_TIMING), true);
});
