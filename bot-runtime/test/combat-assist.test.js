/**
 * 战斗辅助（自动鱼竿 / 自动格挡 / W-tap / 3D 对准）与物品交互的单元测试。
 *
 * <p>用假 bot 记录调用序列，不启动 mineflayer。
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { ActionDispatcher, sleep } from '../src/actions.js';
import { normalizeProfile } from '../src/bot-profile.js';
import { eyeDistance } from '../src/items.js';

function item(name, slot) {
  return { name, slot, count: 1, type: 0 };
}

/** 记录所有控制状态与交互调用。 */
function fakeBot({ position = { x: 0, y: 4, z: 0 }, entityPosition = { x: 0, y: 4, z: 3 } } = {}) {
  const calls = [];
  const inventoryItems = [
    item('diamond_sword', 36),
    item('fishing_rod', 37),
    item('golden_apple', 38),
    item('diamond_helmet', 39),
    item('dirt', 40),
  ];
  const bot = {
    calls,
    entity: { position },
    heldItem: { name: 'diamond_sword', slot: 36 },
    controlState: {},
    entities: {
      7: { id: 7, name: 'Target', position: entityPosition },
    },
    inventory: {
      items: () => inventoryItems.slice(),
      on: () => {},
    },
    setControlState(control, state) {
      calls.push(['control', control, state]);
      bot.controlState[control] = state;
    },
    async setQuickBarSlot(slot) {
      calls.push(['quickbar', slot]);
      bot.heldItem = inventoryItems.find((it) => it.slot === slot + 36) || bot.heldItem;
    },
    async equip(it, dest) {
      calls.push(['equip', it.name, dest]);
    },
    async lookAt(target, force) {
      calls.push(['lookAt', target.y, force]);
    },
    attack(entity) {
      calls.push(['attack', entity.id]);
    },
    activateItem(offhand = false) {
      calls.push(['activate', offhand]);
    },
    deactivateItem() {
      calls.push(['deactivate']);
    },
    chat(message) {
      calls.push(['chat', message]);
    },
  };
  return bot;
}

function profile(overrides = {}) {
  return normalizeProfile({
    cps: 15, cpsJitter: 2, reactionMs: 0, reactionJitterMs: 0, reach: 3.99,
    ...overrides,
  });
}

test('assist 默认全关，显式 true 才打开', () => {
  assert.deepEqual(profile().assist, {
    autoRod: false, autoBlock: false, wtap: false, aimLock: false,
  });
  const on = profile({ assist: { autoRod: true, autoBlock: true, wtap: true, aimLock: true } });
  assert.equal(on.assist.autoRod, true);
  assert.equal(on.assist.aimLock, true);
});

test('aimLock 在攻击前额外调用 lookAt', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot, profile: profile({ assist: { aimLock: true } }) });
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  assert.ok(bot.calls.some(([kind]) => kind === 'lookAt'));
  assert.ok(bot.calls.some(([kind]) => kind === 'attack'));
});

test('autoRod 在中距离先放鱼竿再攻击', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot, profile: profile({ assist: { autoRod: true } }) });
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  const activateCount = bot.calls.filter(([kind]) => kind === 'activate').length;
  // 鱼竿一次 + 攻击不触发 activate
  assert.equal(activateCount, 1);
  assert.ok(bot.calls.some(([kind, slot]) => kind === 'quickbar' && slot === 1));
  assert.equal(dispatcher.combatStats.rodReleases, 1);
});

test('wtap 在命中后松开再按下前进键', async () => {
  const bot = fakeBot();
  bot.controlState.forward = true;
  bot.controlState.sprint = true;
  const dispatcher = new ActionDispatcher({ bot, profile: profile({ assist: { wtap: true } }) });
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  const controlSequence = bot.calls
    .filter(([kind]) => kind === 'control')
    .map(([, control, state]) => `${control}:${state}`);
  assert.ok(controlSequence.includes('sprint:false'));
  assert.ok(controlSequence.includes('forward:false'));
  assert.ok(controlSequence.includes('forward:true'));
  assert.equal(dispatcher.combatStats.wtaps, 1);
});

test('use gapple 需要按住再松开', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot, profile: profile() });
  const started = Date.now();
  await dispatcher.dispatch({ type: 'use', params: { item: 'gapple', holdMs: 60 } });
  assert.ok(Date.now() - started >= 55);
  assert.ok(bot.calls.some(([kind]) => kind === 'activate'));
  assert.ok(bot.calls.some(([kind]) => kind === 'deactivate'));
  assert.ok(bot.calls.some(([kind, slot]) => kind === 'quickbar' && slot === 2));
});

test('use rod 会切回剑，use sword 会格挡指定时长', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot, profile: profile() });
  await dispatcher.dispatch({ type: 'use', params: { item: 'rod' } });
  assert.equal(dispatcher.combatStats.rodReleases, 1);
  const callsBefore = bot.calls.length;
  await dispatcher.dispatch({ type: 'use', params: { item: 'sword', holdMs: 40 } });
  assert.ok(bot.calls.length > callsBefore);
  assert.ok(bot.calls.slice(callsBefore).some(([kind]) => kind === 'deactivate'));
});

test('缺少物品时返回 ITEM_MISSING', async () => {
  const bot = fakeBot();
  delete bot.inventory;
  bot.inventory = { items: () => [], on: () => {} };
  bot.heldItem = null;
  const dispatcher = new ActionDispatcher({ bot, profile: profile() });
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'use', params: { item: 'gapple' } }),
    (error) => error.code === 'ITEM_MISSING',
  );
});

test('eyeDistance 以眼睛高度计算', () => {
  const bot = fakeBot({ position: { x: 0, y: 4, z: 0 }, entityPosition: { x: 0, y: 4, z: 3 } });
  const distance = eyeDistance(bot, bot.entities[7]);
  // 水平 3 格，垂直 1.0 - 1.62 = -0.62
  assert.ok(Math.abs(distance - Math.sqrt(9 + 0.62 * 0.62)) < 1e-9);
});

test('sleep 至少等待指定毫秒', async () => {
  const started = Date.now();
  await sleep(30);
  assert.ok(Date.now() - started >= 25);
});
