/**
 * 开局等套件（G-717-A）的测试。
 *
 * <p>背景（真机 2026-11-24 230121 那局）：板子在倒计时结束才发套件，而模型开局就选
 * `attack`，`ensureSwordInHand()` 立刻抛 `ITEM_MISSING`，一局里连打 4 条
 * `[brain] 意图执行失败`。现在改成「背包里根本没有剑 → 200ms × 10 等 2 秒」，
 * 这段等待单独计数（`kitWaits`），不算失败。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { ActionDispatcher, KIT_WAIT } from '../src/actions.js';

/** 最小假 Bot：背包内容可变，够 selectItem 把剑切到手上。 */
function makeBot(items = []) {
  const state = { items: items.slice() };
  const bot = {
    inventory: { items: () => state.items, slots: [] },
    quickBarSlot: 0,
    heldItem: null,
    entities: {},
    async setQuickBarSlot(index) {
      bot.quickBarSlot = index;
    },
    async moveSlotItem() {
      throw new Error('本用例不该走到背包内移动');
    },
    /** 测试用：把一件物品放进背包（模拟板子发套件）。 */
    give(item) {
      state.items.push(item);
    },
  };
  return bot;
}

const SWORD = { slot: 36, name: 'diamond_sword', count: 1, type: 276 };

/** 快节奏的等套件参数（真实默认是 200ms × 10 = 2s）。 */
const FAST_WAIT = { intervalMs: 5, attempts: 6 };

test('默认节奏与契约一致：200ms × 10 = 2 秒', () => {
  assert.equal(KIT_WAIT.intervalMs, 200);
  assert.equal(KIT_WAIT.attempts, 10);
  const dispatcher = new ActionDispatcher({ bot: makeBot() });
  assert.deepEqual(dispatcher.kitWait, { intervalMs: 200, attempts: 10 });
});

test('开局没剑：等到套件就继续，不抛错且只通知一次', async () => {
  const bot = makeBot();
  const notices = [];
  const dispatcher = new ActionDispatcher({
    bot, kitWait: FAST_WAIT, onKitWait: (info) => notices.push(info),
  });

  // 第 3 次轮询前把剑放进背包（模拟板子发套件）。
  let polls = 0;
  const originalItems = bot.inventory.items;
  bot.inventory.items = () => {
    polls += 1;
    if (polls > 3) {
      return [SWORD];
    }
    return originalItems();
  };

  await dispatcher.ensureSwordInHand();
  assert.equal(dispatcher.kitWaits, 1, '进入等套件的次数');
  assert.equal(dispatcher.kitWaitGaveUp, 0, '等到了就不该记「放弃」');
  assert.equal(notices.length, 1, 'onKitWait 只通知一次（外层据此打一行日志）');
  assert.deepEqual(notices[0], { attempts: FAST_WAIT.attempts, intervalMs: FAST_WAIT.intervalMs });
  assert.equal(dispatcher.combatStats.kitWaits, 1);
  assert.equal(dispatcher.combatStats.kitWaitGaveUp, 0);
});

test('等满窗口仍没剑：抛 ITEM_MISSING 并记一次放弃', async () => {
  const dispatcher = new ActionDispatcher({ bot: makeBot(), kitWait: FAST_WAIT });
  await assert.rejects(
    () => dispatcher.ensureSwordInHand(),
    (error) => {
      assert.equal(error.code, 'ITEM_MISSING');
      assert.match(error.message, /板子仍未发套件/);
      return true;
    },
  );
  assert.equal(dispatcher.kitWaits, 1);
  assert.equal(dispatcher.kitWaitGaveUp, 1, '2 秒没等到才是真失败');
});

test('等待期间不抛错（2 秒内的失败不进 failures）', async () => {
  // 只要背包里"还没有剑"就进入等待，而不是当场抛：这里验证等待期间没有提前 reject。
  const bot = makeBot();
  const dispatcher = new ActionDispatcher({ bot, kitWait: { intervalMs: 5, attempts: 3 } });
  let settled = false;
  const pending = dispatcher.ensureSwordInHand().then(
    () => { settled = true; },
    () => { settled = true; },
  );
  await new Promise((resolve) => setTimeout(resolve, 8));
  assert.equal(settled, false, '窗口没走完之前不能 settle（既不 resolve 也不抛）');
  bot.give(SWORD);
  await pending;
});

test('手里已经是剑：不等、不计次', async () => {
  const bot = makeBot([SWORD]);
  bot.heldItem = SWORD;
  const dispatcher = new ActionDispatcher({ bot, kitWait: FAST_WAIT });
  await dispatcher.ensureSwordInHand();
  assert.equal(dispatcher.kitWaits, 0);
});

test('有剑但不在快捷栏时走原来的切换重试，不进入等套件', async () => {
  const bot = makeBot([{ slot: 10, name: 'iron_sword', count: 1 }]);
  bot.moveSlotItem = async () => {};
  const dispatcher = new ActionDispatcher({ bot, kitWait: FAST_WAIT });
  await dispatcher.ensureSwordInHand();
  assert.equal(dispatcher.kitWaits, 0, '有剑只是不在手上 → 不算「等套件」');
});

test('背包读不到（只读封装）时不做判定，也不等', async () => {
  const bot = { heldItem: null, entities: {}, inventory: null };
  const dispatcher = new ActionDispatcher({ bot, kitWait: FAST_WAIT });
  await dispatcher.ensureSwordInHand();
  assert.equal(dispatcher.kitWaits, 0);
});
