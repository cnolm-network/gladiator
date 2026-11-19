/**
 * 意图执行器（动作空间 v2）的单元测试：假 bot 对象，不连服务器。
 *
 * 覆盖审稿人最关心的四件事：
 * 1. `look` 是**增量**且每 tick clamp（±45°/±20°）；
 * 2. `move/jump/sneak/sprint` 打的是 setControlState，`idle` 会松键；
 * 3. `attack` 走 attack-guard（切剑 + 500ms 节流）；
 * 4. `use` 的按法由 items.json 的 mode 决定（click 一次 / hold 到 ticks 后松开）。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { loadItemTable } from '../src/action-mask.js';
import {
  MAX_DYAWDEG_PER_TICK,
  applyIntent,
  describeIntent,
  intentStats,
  nearestOpponent,
  normalizeIntent,
  resetIntentState,
} from '../src/intent.js';

const DEG = Math.PI / 180;

function makeBot({
  held = { name: 'diamond_sword', count: 1 },
  quickBarSlot = 0,
  yaw = 0,
  pitch = 0,
  withOpponent = true,
} = {}) {
  const calls = { controls: [], looks: [], attacks: [], slots: [], activates: 0, deactivates: 0 };
  const bot = {
    username: 'Glad_Bot_1',
    heldItem: held,
    quickBarSlot,
    controlState: {},
    entity: {
      yaw,
      pitch,
      position: { x: 0, y: 64, z: 0, distanceTo: (other) => Math.abs(other.x) },
    },
    entities: withOpponent
      ? { 7: { id: 7, type: 'player', username: 'Foe', position: { x: 2, y: 64, z: 0 } } }
      : {},
    inventory: { slots: [] },
    setControlState(name, value) {
      calls.controls.push([name, value]);
      bot.controlState[name] = value;
    },
    async look(nextYaw, nextPitch) {
      calls.looks.push([nextYaw, nextPitch]);
      bot.entity.yaw = nextYaw;
      bot.entity.pitch = nextPitch;
    },
    async setQuickBarSlot(slot) {
      calls.slots.push(slot);
      bot.quickBarSlot = slot;
    },
    activateItem() {
      calls.activates += 1;
    },
    deactivateItem() {
      calls.deactivates += 1;
    },
    attack(entity) {
      calls.attacks.push(entity.id);
    },
  };
  return { bot, calls };
}

function controlOf(calls, name) {
  return calls.controls.filter(([key]) => key === name).map(([, value]) => value);
}

test('normalizeIntent 补齐缺省值并拒绝非法取值', () => {
  const intent = normalizeIntent({});
  assert.deepEqual(intent, {
    move: 'idle',
    look: { dyaw: 0, dpitch: 0 },
    attack: false,
    use: null,
    slot: null,
    jump: false,
    sneak: false,
    sprint: false,
  });
  assert.throws(() => normalizeIntent({ move: 'teleport' }), /move/);
  assert.throws(() => normalizeIntent({ use: 'double' }), /use/);
  assert.throws(() => normalizeIntent({ slot: 9 }), /slot/);
  assert.throws(() => normalizeIntent(null), /对象/);
  // Python 侧的 "none" 兼容成 null
  assert.equal(normalizeIntent({ use: 'none' }).use, null);
});

test('describeIntent 是紧凑的一行回显', () => {
  const text = describeIntent({ move: 'forward', look: { dyaw: 10, dpitch: -5 }, attack: true, slot: 2 });
  assert.match(text, /move=forward/);
  assert.match(text, /look=\+10\/-5/);
  assert.match(text, /attack=1/);
  assert.match(text, /slot=2/);
});

test('move 用 setControlState，idle 松开所有方向', async () => {
  const { bot, calls } = makeBot();
  await applyIntent(bot, { move: 'forward' });
  assert.deepEqual(controlOf(calls, 'forward'), [true]);
  // 已经是 false 的方向不重复下发（省包），但状态必须是「没按下」
  assert.equal(bot.controlState.back, undefined);
  assert.equal(bot.controlState.left, undefined);
  assert.equal(bot.controlState.right, undefined);

  calls.controls.length = 0;
  await applyIntent(bot, { move: 'left' });
  assert.deepEqual(controlOf(calls, 'left'), [true]);
  assert.deepEqual(controlOf(calls, 'forward'), [false]);

  calls.controls.length = 0;
  await applyIntent(bot, { move: 'idle' });
  assert.deepEqual(controlOf(calls, 'left'), [false]);
  assert.equal(bot.controlState.forward, false);
});

test('look 是增量旋转，并按每 tick 上限 clamp', async () => {
  const { bot, calls } = makeBot({ yaw: 0, pitch: 0 });
  await applyIntent(bot, { look: { dyaw: 10, dpitch: 5 } });
  assert.equal(calls.looks.length, 1);
  assert.ok(Math.abs(calls.looks[0][0] - 10 * DEG) < 1e-9);
  assert.ok(Math.abs(calls.looks[0][1] - 5 * DEG) < 1e-9);

  // 增量超过 ±45°/±20° 时被夹住（不是绝对朝向被夹）
  calls.looks.length = 0;
  await applyIntent(bot, { look: { dyaw: 90, dpitch: -60 } });
  assert.ok(Math.abs(calls.looks[0][0] - (10 + MAX_DYAWDEG_PER_TICK) * DEG) < 1e-9);
  assert.ok(Math.abs(calls.looks[0][1] - (5 - 20) * DEG) < 1e-9);

  // 俯仰有 ±90° 硬边界
  calls.looks.length = 0;
  bot.entity.pitch = 85 * DEG;
  await applyIntent(bot, { look: { dyaw: 0, dpitch: 15 } });
  assert.ok(calls.looks[0][1] <= Math.PI / 2 + 1e-9);
});

test('视角不动时不发包', async () => {
  const { bot, calls } = makeBot();
  await applyIntent(bot, { look: { dyaw: 0, dpitch: 0 } });
  assert.equal(calls.looks.length, 0);
});

test('slot 只在变化时调用 setQuickBarSlot', async () => {
  const { bot, calls } = makeBot({ quickBarSlot: 0 });
  await applyIntent(bot, { slot: 3 });
  assert.deepEqual(calls.slots, [3]);
  await applyIntent(bot, { slot: 3 });
  assert.deepEqual(calls.slots, [3]);
});

test('jump / sneak / sprint 与 move 正交', async () => {
  const { bot, calls } = makeBot();
  await applyIntent(bot, { move: 'forward', jump: true, sneak: true, sprint: true });
  assert.deepEqual(controlOf(calls, 'jump'), [true]);
  assert.deepEqual(controlOf(calls, 'sneak'), [true]);
  assert.deepEqual(controlOf(calls, 'sprint'), [true]);
  assert.deepEqual(controlOf(calls, 'forward'), [true]);

  calls.controls.length = 0;
  await applyIntent(bot, { move: 'forward', sneak: false, sprint: false });
  assert.deepEqual(controlOf(calls, 'sneak'), [false]);
  assert.deepEqual(controlOf(calls, 'sprint'), [false]);
  resetIntentState(bot);
});

test('attack 打最近玩家，并在 500ms 内节流', async () => {
  const { bot, calls } = makeBot();
  let now = 1000;
  const ctx = { now: () => now };

  await applyIntent(bot, { attack: true }, ctx);
  assert.deepEqual(calls.attacks, [7]);

  now += 100; // 100ms < 500ms 无敌帧
  const second = await applyIntent(bot, { attack: true }, ctx);
  assert.equal(second.applied.attack.attack, 'throttled');
  assert.deepEqual(calls.attacks, [7]);

  now += 500;
  await applyIntent(bot, { attack: true }, ctx);
  assert.deepEqual(calls.attacks, [7, 7]);
  assert.equal(intentStats(bot).attacks, 2);
  assert.equal(intentStats(bot).throttled, 1);
});

test('attack 走 dispatcher（复用 reach/追击/切剑）', async () => {
  const { bot, calls } = makeBot({ held: { name: 'golden_apple', count: 1 } });
  const seen = [];
  const dispatcher = {
    async attack(params) {
      seen.push(params);
      return { ok: true, code: 'OUT_OF_REACH_CHASED' };
    },
  };
  const result = await applyIntent(bot, { attack: true }, { dispatcher, now: () => 5000 });
  assert.deepEqual(seen, [{ targetEntityId: 7 }]);
  assert.equal(result.applied.attack.code, 'OUT_OF_REACH_CHASED');
  assert.deepEqual(calls.attacks, []); // 交给 dispatcher，不再重复 bot.attack
});

test('没有对手时 attack 不清冷却', async () => {
  const { bot, calls } = makeBot({ withOpponent: false });
  const result = await applyIntent(bot, { attack: true }, { now: () => 1000 });
  assert.equal(result.applied.attack.attack, 'no-target');
  assert.deepEqual(calls.attacks, []);
  assert.equal(intentStats(bot).attacks, 0);
});

test('nearestOpponent 只认玩家实体', () => {
  const { bot } = makeBot();
  bot.entities[9] = { id: 9, type: 'zombie', username: 'Zombie', position: { x: 1, y: 64, z: 0 } };
  assert.equal(nearestOpponent(bot).entity.id, 7);
});

test('use=click 对 click 类物品：按一下再松开', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'fishing_rod', count: 1 } });
  const result = await applyIntent(bot, { use: 'click' }, { itemTable: table });
  assert.equal(result.applied.use.mode, 'click');
  assert.equal(calls.activates, 1);
  assert.equal(calls.deactivates, 1);
});

test('use=hold 对 hold 类物品：按 ticks 后由计时器松开', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'golden_apple', count: 5 } });
  const result = await applyIntent(bot, { use: 'hold' }, { itemTable: table });
  assert.equal(result.applied.use.mode, 'hold');
  assert.equal(result.applied.use.ticks, 32);        // items.json 里金苹果是 32 tick = 1.6s
  assert.equal(result.applied.use.holdMs, 1600);
  assert.equal(calls.activates, 1);
  assert.equal(calls.deactivates, 0);

  // 连续 hold 不重启计时（否则永远松不了手）
  await applyIntent(bot, { use: 'hold' }, { itemTable: table });
  assert.equal(calls.activates, 1);

  // use=null 立即松手
  await applyIntent(bot, { use: null }, { itemTable: table });
  assert.equal(calls.deactivates, 1);
  resetIntentState(bot);
});

test('use=click 对 hold 类物品只按 1 tick（点一下就撤）', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'golden_apple', count: 5 } });
  const result = await applyIntent(bot, { use: 'click' }, { itemTable: table });
  assert.equal(result.applied.use.ticks, 1);
  assert.equal(result.applied.use.holdMs, 50);
  await applyIntent(bot, { use: null }, { itemTable: table });
  assert.equal(calls.deactivates, 1);
});

test('能力表说不能用右键时什么都不做（mask 之外的兜底）', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'mystery_widget', count: 1 } });
  const result = await applyIntent(bot, { use: 'hold' }, { itemTable: table });
  assert.equal(result.applied.use.skipped, 'no-usable-item');
  assert.equal(calls.activates, 0);
});

test('空手 use 不报错也不发包', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'air', count: 0 } });
  const result = await applyIntent(bot, { use: 'click' }, { itemTable: table });
  assert.equal(result.applied.use.skipped, 'no-usable-item');
  assert.equal(calls.activates, 0);
});

test('完整意图按 选槽 → 视角 → 移动 → 右键 → 攻击 的顺序落地', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'diamond_sword', count: 1 } });
  const result = await applyIntent(bot, {
    move: 'forward',
    look: { dyaw: 3, dpitch: 0 },
    attack: true,
    use: 'hold',
    slot: 4,
    sprint: true,
  }, { itemTable: table, now: () => 1000 });

  assert.equal(result.ok, true);
  assert.equal(result.type, 'intent');
  assert.deepEqual(calls.slots, [4]);
  assert.equal(calls.looks.length, 1);
  assert.deepEqual(calls.attacks, [7]);
  assert.equal(calls.activates, 1);
  assert.equal(result.applied.slot, 4);
  assert.equal(result.applied.move, 'forward');
  assert.equal(result.applied.sprint, true);
  assert.equal(result.applied.attack.attack, 'hit');
  resetIntentState(bot);
});

test('resetIntentState 清掉按住状态', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'golden_apple', count: 5 } });
  await applyIntent(bot, { use: 'hold', move: 'forward' }, { itemTable: table });
  resetIntentState(bot);
  // 清状态后再次 hold 会重新按下（说明旧状态确实被丢了）
  await applyIntent(bot, { use: 'hold' }, { itemTable: table });
  assert.equal(calls.activates, 2);
  resetIntentState(bot);
});
