/**
 * A1（P0-1）：物品前置条件。
 *
 * <p>训练日志里出现过「[kit] 手持剑」之后仍然 `ITEM_MISSING 快捷栏里没有 sword/gapple/rod`，
 * 根因是运行时只看快捷栏、不把背包里的物品挪进来，且 attack 不检查主手是不是剑。
 *
 * <p>用假 bot + 假 inventory 覆盖四条路径，不启动 mineflayer。
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { ActionDispatcher } from '../src/actions.js';

function item(name, slot, count = 1) {
  return { name, slot, count, type: 0 };
}

/**
 * 假 bot：inventory.items() 每次返回当前背包快照；moveSlotItem 改写槽位；
 * setQuickBarSlot 同步 heldItem，便于断言“切剑后再打”。
 */
function fakeBot({ items = [], held = null, quickBarSlot = 0 } = {}) {
  const calls = [];
  const inventory = items.map((it) => ({ ...it }));
  const bot = {
    calls,
    entity: { position: { x: 0, y: 64, z: 0 } },
    entities: { 7: { id: 7, position: { x: 1, y: 64, z: 0 } } },
    quickBarSlot,
    heldItem: held,
    inventory: {
      items: () => inventory.map((it) => ({ ...it })),
      on: () => {},
    },
    setControlState: () => {},
    async look() {},
    async lookAt() {},
    async setQuickBarSlot(slot) {
      calls.push(['quickbar', slot]);
      bot.quickBarSlot = slot;
      bot.heldItem = inventory.find((it) => it.slot === slot + 36) || null;
    },
    async moveSlotItem(source, dest) {
      calls.push(['move', source, dest]);
      const it = inventory.find((entry) => entry.slot === source);
      if (it) {
        it.slot = dest;
      }
    },
    attack(entity) {
      calls.push(['attack', entity.id]);
    },
    activateItem(offhand) {
      calls.push(['activate', offhand]);
    },
    deactivateItem() {
      calls.push(['deactivate']);
    },
    chat() {},
  };
  return bot;
}

test('剑在背包不在快捷栏：attack 先切剑再攻击', async () => {
  const bot = fakeBot({
    items: [item('diamond_sword', 20), item('golden_apple', 36, 5)],
    held: { name: 'golden_apple', slot: 36 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });

  const move = bot.calls.find((c) => c[0] === 'move');
  assert.ok(move, '应该把剑从背包挪进快捷栏');
  assert.equal(move[1], 20);
  assert.ok(move[2] >= 36 && move[2] <= 44);
  // 先切剑、再攻击
  const attackIndex = bot.calls.findIndex((c) => c[0] === 'attack');
  const quickbarIndex = bot.calls.findIndex((c) => c[0] === 'quickbar');
  assert.ok(quickbarIndex >= 0 && quickbarIndex < attackIndex, '切剑必须发生在攻击之前');
  assert.equal(bot.heldItem.name, 'diamond_sword');
  assert.deepEqual(bot.calls[attackIndex], ['attack', 7]);
});

test('完全没有剑：attack 回 ITEM_MISSING 且不挥砍', async () => {
  const bot = fakeBot({
    items: [item('golden_apple', 36, 5)],
    held: { name: 'golden_apple', slot: 36 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await assert.rejects(
    () => dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } }),
    (error) => {
      assert.equal(error.code, 'ITEM_MISSING');
      assert.match(error.message, /背包里没有剑/);
      return true;
    },
  );
  assert.equal(bot.calls.some((c) => c[0] === 'attack'), false);
});

test('鱼竿在背包不在快捷栏：use rod 先挪进快捷栏再抛竿', async () => {
  const bot = fakeBot({
    items: [item('fishing_rod', 15), item('diamond_sword', 36)],
    held: { name: 'diamond_sword', slot: 36 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await dispatcher.dispatch({ type: 'use', params: { item: 'rod' } });

  const move = bot.calls.find((c) => c[0] === 'move');
  assert.ok(move, '鱼竿应该从背包挪进快捷栏');
  assert.equal(move[1], 15);
  const activate = bot.calls.findIndex((c) => c[0] === 'activate');
  const quickbar = bot.calls.findIndex((c) => c[0] === 'quickbar');
  assert.ok(quickbar >= 0 && quickbar < activate, '切鱼竿必须发生在右键之前');
});

test('金苹果只在背包里：use gapple 也能吃到（自动挪位）', async () => {
  const bot = fakeBot({
    items: [item('golden_apple', 10, 3), item('diamond_sword', 36)],
    held: { name: 'diamond_sword', slot: 36 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await dispatcher.dispatch({ type: 'use', params: { item: 'gapple', holdMs: 1 } });

  assert.ok(bot.calls.some((c) => c[0] === 'move'), '金苹果应该被挪进快捷栏');
  assert.ok(bot.calls.some((c) => c[0] === 'activate'), '应该真的右键吃了');
});

test('完全没有鱼竿：use rod 回 ITEM_MISSING 且不右键', async () => {
  const bot = fakeBot({
    items: [item('diamond_sword', 36)],
    held: { name: 'diamond_sword', slot: 36 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await assert.rejects(
    () => dispatcher.dispatch({ type: 'use', params: { item: 'rod' } }),
    (error) => {
      assert.equal(error.code, 'ITEM_MISSING');
      assert.match(error.message, /背包里没有鱼竿/);
      return true;
    },
  );
  assert.equal(bot.calls.some((c) => c[0] === 'activate'), false);
});

test('select:true 只切手持，不产生挥砍/右键', async () => {
  const bot = fakeBot({
    items: [item('diamond_sword', 36), item('golden_apple', 37, 5)],
    held: { name: 'golden_apple', slot: 37 },
  });
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await dispatcher.dispatch({ type: 'use', params: { item: 'sword', select: true } });

  assert.equal(bot.heldItem.name, 'diamond_sword');
  assert.deepEqual(bot.calls.filter((c) => c[0] === 'quickbar'), [['quickbar', 0]]);
  assert.equal(bot.calls.some((c) => c[0] === 'attack'), false);
  assert.equal(bot.calls.some((c) => c[0] === 'activate'), false);
});

test('背包读不到时不做剑校验（只读封装不误伤）', async () => {
  const bot = fakeBot({ items: [], held: null });
  delete bot.inventory;
  const dispatcher = new ActionDispatcher({ bot, profile: { reactionMs: 0, reactionJitterMs: 0 } });

  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  assert.ok(bot.calls.some((c) => c[0] === 'attack'));
});
