/**
 * G-707：`sprintReset` 动作维度的执行语义（假 bot，不连服务器）。
 *
 * 三条判据（与任务书 A-4 一一对应）：
 * 1. `release_w` —— **同一帧内**先 `forward=false`、再 attack、最后 `forward=true`；
 * 2. `block_hit` —— 同一帧内先举剑（activateItem）再出手，举剑时长 300ms；
 * 3. 缺省（没有 `sprintReset` 字段）—— 按键流与九头时代**完全一致**（回归）。
 *
 * 为什么顺序要断言而不是只看「都调用了」：1.8 的击退大小由**命中帧**是否疾跑决定，
 * 先砍后松手等于什么都没做。同理 block_hit 必须先举剑，砍完再举就只吃到「举剑」而没有减伤。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { loadItemTable } from '../src/action-mask.js';
import {
  BLOCK_HIT_HOLD_MS,
  SPRINT_RESET_VALUES,
  applyIntent,
  normalizeIntent,
  resetIntentState,
} from '../src/intent.js';

function makeBot({
  held = { name: 'diamond_sword', count: 1 },
  withOpponent = true,
} = {}) {
  const calls = { controls: [], attacks: [], activates: 0, deactivates: 0, slots: [] };
  const bot = {
    username: 'Glad_Bot_1',
    heldItem: held,
    quickBarSlot: 0,
    controlState: {},
    entity: {
      yaw: 0,
      pitch: 0,
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
    async look() {},
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

/** 只保留某个控制键的取值序列（用来断言「先关后开」的时序）。 */
function controlOf(calls, name) {
  return calls.controls.filter(([key]) => key === name).map(([, value]) => value);
}

test('SPRINT_RESET_VALUES 与 Python 侧同序（none / release_w / block_hit）', () => {
  assert.deepEqual([...SPRINT_RESET_VALUES], ['none', 'release_w', 'block_hit']);
  assert.equal(BLOCK_HIT_HOLD_MS, 300);
  // 非法取值要被拦下，而不是悄悄退化成 none
  assert.throws(() => normalizeIntent({ sprintReset: 'wtap' }), /sprintReset/);
});

test('sprintReset 缺省时归一化结果不含该键（intent 形状回归）', () => {
  assert.equal('sprintReset' in normalizeIntent({ move: 'forward' }), false);
  assert.equal('sprintReset' in normalizeIntent({ move: 'forward', sprintReset: null }), false);
  assert.equal('sprintReset' in normalizeIntent({ move: 'forward', sprintReset: 'none' }), false);
  assert.equal(normalizeIntent({ sprintReset: 'release_w' }).sprintReset, 'release_w');
});

test('release_w：同一帧内 forward 先关、attack 之后马上开回来', async () => {
  const { bot, calls } = makeBot();
  let clock = 1000;
  const ctx = { now: () => clock };

  const result = await applyIntent(
    bot,
    { move: 'forward', attack: true, sprintReset: 'release_w' },
    ctx,
  );

  // 按键流：forward true（applyMove）→ false（松手）→ true（按回）
  assert.deepEqual(controlOf(calls, 'forward'), [true, false, true]);
  // attack 仍然发出，而且落在「松手之后」
  assert.deepEqual(calls.attacks, [7]);
  assert.equal(result.applied.attack.attack, 'hit');
  // 松手发生在 attack 之前：把 controls 与 attacks 摊平看相对顺序
  assert.equal(result.applied.sprintReset.reset, 'release_w');
  assert.equal(result.applied.sprintReset.restored, true);
  // 收尾后仍然是按住状态（下一帧同方向不会再重复下发）
  assert.equal(bot.controlState.forward, true);

  calls.controls.length = 0;
  clock += 500;
  await applyIntent(bot, { move: 'forward', attack: true, sprintReset: 'release_w' }, ctx);
  assert.deepEqual(controlOf(calls, 'forward'), [false, true], '第二帧不应再重复按下');
  resetIntentState(bot);
});

test('release_w 在出手前重发疾跑位（重置疾跑速度 = 这一刀给对手更大 KB）', async () => {
  const { bot, calls } = makeBot();
  const result = await applyIntent(
    bot,
    { move: 'forward', attack: true, sprint: true, sprintReset: 'release_w' },
    { now: () => 2000 },
  );
  // true（本帧要求疾跑）→ false → true 重发 START_SPRINTING，全部发生在 attack 之前：
  // 依据 `PVP_overview.md`「攻击后松 W 再按，重置疾跑速度，给对面更大击退」。
  assert.deepEqual(controlOf(calls, 'sprint'), [true, false, true]);
  assert.equal(result.applied.sprintResent, true);
  // back/left/right 没有被 release_w 动过
  assert.deepEqual(controlOf(calls, 'back'), []);
  assert.deepEqual(controlOf(calls, 'left'), []);
  resetIntentState(bot);
});

test('release_w 在意图未要求疾跑时不碰 sprint（不替模型做决定）', async () => {
  const { bot, calls } = makeBot();
  await applyIntent(
    bot,
    { move: 'forward', attack: true, sprintReset: 'release_w' },
    { now: () => 2100 },
  );
  assert.deepEqual(controlOf(calls, 'sprint'), []);
  resetIntentState(bot);
});

test('release_w 在非前进帧上不发多余的按键（如实回报 released=null）', async () => {
  const { bot, calls } = makeBot();
  const result = await applyIntent(
    bot,
    { move: 'idle', attack: true, sprintReset: 'release_w' },
    { now: () => 3000 },
  );
  assert.deepEqual(controlOf(calls, 'forward'), []);
  assert.equal(result.applied.sprintReset.released, null);
  assert.deepEqual(calls.attacks, [7]);
  resetIntentState(bot);
});
test('block_hit：同一帧内先举剑（activate）再出手，举剑 300ms', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'diamond_sword', count: 1 } });
  const result = await applyIntent(
    bot,
    { move: 'forward', attack: true, sprintReset: 'block_hit' },
    { itemTable: table, now: () => 4000 },
  );

  assert.equal(calls.activates, 1, '举剑必须先发生');
  assert.deepEqual(calls.attacks, [7]);
  assert.equal(result.applied.sprintReset.reset, 'block_hit');
  assert.equal(result.applied.sprintReset.holdMs, BLOCK_HIT_HOLD_MS);
  // 300ms 之内不松手（举剑要盖住命中帧）
  assert.equal(calls.deactivates, 0);
  assert.equal(bot.controlState.forward, true);

  // use=null 之后必须能正常收起（共用 state.useHeld / useTimer）
  await applyIntent(bot, { use: null }, { itemTable: table });
  assert.equal(calls.deactivates, 1);
  resetIntentState(bot);
});

test('block_hit 手里不是剑时会先切剑（needsSwordSwap 路径）', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot({ held: { name: 'air', count: 0 } });
  // 背包里有剑（假 inventory：items() 返回热键栏里的剑）
  bot.inventory = {
    items: () => [{ name: 'diamond_sword', slot: 36, count: 1 }],
    slots: [],
  };
  bot.setQuickBarSlot = async (slot) => {
    calls.slots.push(slot);
    bot.quickBarSlot = slot;
  };
  await applyIntent(bot, { attack: true, sprintReset: 'block_hit' },
    { itemTable: table, now: () => 5000 });
  assert.deepEqual(calls.attacks, [7]);
  resetIntentState(bot);
});

test('没有 sprintReset 时按键流与九头时代完全一致（回归）', async () => {
  const table = loadItemTable();
  const { bot, calls } = makeBot();
  await applyIntent(
    bot,
    { move: 'forward', look: { dyaw: 3, dpitch: 0 }, attack: true, sprint: true },
    { itemTable: table, now: () => 6000 },
  );
  // 只有一次 forward=true，没有「先关后开」
  assert.deepEqual(controlOf(calls, 'forward'), [true]);
  assert.equal(calls.activates, 0, '不该自己举剑');
  assert.deepEqual(calls.attacks, [7]);
  resetIntentState(bot);
});

test('sprintReset 与 jump 可以同帧共存（reset 不吞掉跳劈时序）', async () => {
  const { bot, calls } = makeBot();
  await applyIntent(
    bot,
    { move: 'forward', attack: true, jump: true, sprintReset: 'release_w' },
    { now: () => 7000 },
  );
  assert.deepEqual(controlOf(calls, 'jump'), [true]);
  assert.deepEqual(controlOf(calls, 'forward'), [true, false, true]);
  assert.deepEqual(calls.attacks, [7]);
  resetIntentState(bot);
});
