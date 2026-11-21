/**
 * G-701 复合动作回归测试：**同一次 `applyIntent` 调用里**既起跳又出手。
 *
 * <p>背景（ADR-020）：v2 的 9 个头是**同帧独立取值**，「跳劈」不是一个新动作，
 * 而是 `jump=1` 与 `attack=1` 同时出现在一条意图里。`applyIntent` 的步骤 3 打
 * `setControlState('jump', true)`、步骤 5 走 attack-guard，两者在同一个 async 函数里
 * 顺序执行，所以天然可以共存——本文件的作用是**把这条通路钉住**，任何人重排/提前
 * return/把 attack 包进 try 里回滚 jump，都会在这里失败。
 *
 * <p>三个不变量：
 * 1. 同 tick 双原语都落地（jump 按下 + attack 计数 +1）；
 * 2. jump 是 60ms 脉冲（`JUMP_PULSE_MS`），松开由定时器完成，**不会**在 attack 之前松掉；
 * 3. attack 抛错时 jump 不被回滚（不因为「后一步失败」把「前一步已生效的按键」撤销）。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { loadItemTable } from '../src/action-mask.js';
import { JUMP_PULSE_MS, applyIntent, intentStats, resetIntentState } from '../src/intent.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * 假 Bot：记录每次 `setControlState` 的 `[名字, 值, 时刻]`，以及 attack 目标。
 * 不连服务器，`bot.entity.velocity` 等物理字段不需要（intent 层不读它们）。
 */
function makeBot({ held = { name: 'diamond_sword', count: 1 } } = {}) {
  const calls = { controls: [], attacks: [], dispatcherHits: 0, activates: 0, deactivates: 0 };
  const bot = {
    username: 'Glad_Combo',
    heldItem: held,
    quickBarSlot: 0,
    controlState: {},
    entity: {
      yaw: 0,
      pitch: 0,
      position: { x: 0, y: 64, z: 0, distanceTo: (other) => Math.abs(other.x) },
    },
    entities: { 7: { id: 7, type: 'player', username: 'Foe', position: { x: 2, y: 64, z: 0 } } },
    inventory: { slots: [] },
    setControlState(name, value) {
      calls.controls.push([name, value, Date.now()]);
      bot.controlState[name] = value;
    },
    async look(nextYaw, nextPitch) {
      bot.entity.yaw = nextYaw;
      bot.entity.pitch = nextPitch;
    },
    async setQuickBarSlot(slot) {
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
  const dispatcher = {
    async attack(params) {
      calls.dispatcherHits += 1;
      calls.attacks.push(params.targetEntityId);
      return { ok: true, code: 'HIT' };
    },
  };
  return { bot, calls, dispatcher };
}

/** 某个控制键的全部取值序列（按发生顺序）。 */
function controlOf(calls, name) {
  return calls.controls.filter(([key]) => key === name).map(([, value]) => value);
}

/** 某个控制键第一次被设成 `value` 的时刻；没发生则返回 null。 */
function controlAt(calls, name, value) {
  const hit = calls.controls.find(([key, seen]) => key === name && seen === value);
  return hit ? hit[2] : null;
}

test('JUMP_PULSE_MS 是 60ms，跳劈的「按住」不会盖住一整帧', () => {
  assert.equal(JUMP_PULSE_MS, 60);
});

test('同一次 applyIntent 里 jump 与 attack 都执行（跳劈的运行时前提）', async () => {
  const { bot, calls, dispatcher } = makeBot();
  const result = await applyIntent(bot, {
    move: 'forward',
    jump: true,
    attack: true,
    use: { item: 'sword' },
  }, { dispatcher, itemTable: loadItemTable(), now: () => 1000 });

  // ① jump 真的按下去了
  assert.deepEqual(controlOf(calls, 'jump'), [true]);
  assert.equal(bot.controlState.jump, true);
  assert.equal(result.applied.jump, true);

  // ② 同一次调用里 attack 也执行了（走 dispatcher，计数 +1）
  assert.equal(calls.dispatcherHits, 1);
  assert.deepEqual(calls.attacks, [7]);
  assert.equal(result.applied.attack.attack, 'hit');
  // ③ 移动头也照常落地：三个原语互不挤占
  assert.deepEqual(controlOf(calls, 'forward'), [true]);
  assert.equal(intentStats(bot).attacks, 1);
  // ④ v1 形状的 `use:{item}` 在 v2 这一层不构成按键（v2 的 use 是 click/hold 模式，
  //    用哪个物品由 slot 头 + 能力表决定）；它也不应吞掉 jump/attack。
  assert.equal(calls.activates, 0);

  resetIntentState(bot);
});

test('jump 的松开是 60ms 后的脉冲，不会在 attack 之前吃掉出手', async () => {
  const { bot, calls, dispatcher } = makeBot();
  await applyIntent(bot, { move: 'forward', jump: true, attack: true }, {
    dispatcher,
    now: () => 2000,
  });

  // applyIntent 返回时仍在空中按键窗口内：attack 已经落地，jump 还没松
  assert.equal(calls.dispatcherHits, 1);
  assert.equal(bot.controlState.jump, true);
  const pressedAt = controlAt(calls, 'jump', true);
  const releasedBeforeReturn = controlOf(calls, 'jump').includes(false);
  assert.equal(releasedBeforeReturn, false, 'attack 执行前不应该松 jump');

  // 等过脉冲窗口：由定时器松开，且只松一次
  await sleep(JUMP_PULSE_MS + 40);
  assert.equal(bot.controlState.jump, false);
  assert.deepEqual(controlOf(calls, 'jump'), [true, false]);
  const releasedAt = controlAt(calls, 'jump', false);

  // 脉冲时长量级正确（定时器精度 + CI 抖动，给宽松上界）
  const pulse = releasedAt - pressedAt;
  assert.ok(pulse >= JUMP_PULSE_MS - 5, `脉冲过短: ${pulse}ms`);
  assert.ok(pulse <= JUMP_PULSE_MS + 400, `脉冲过长: ${pulse}ms`);

  // 松开之后不再有新的攻击被凭空触发
  assert.equal(calls.dispatcherHits, 1);
  resetIntentState(bot);
});

test('反例：attack 抛错不会把已经生效的 jump 回滚', async () => {
  const { bot, calls } = makeBot();
  const failing = {
    async attack() {
      throw new Error('dispatcher exploded');
    },
  };
  calls.dispatcherHits = 0;

  await assert.rejects(
    applyIntent(bot, { move: 'forward', jump: true, attack: true }, {
      dispatcher: failing,
      now: () => 3000,
    }),
    /dispatcher exploded/,
  );

  // 关键断言：jump 仍是按下状态，没有被「后一步失败」撤销
  assert.equal(bot.controlState.jump, true);
  assert.deepEqual(controlOf(calls, 'jump'), [true]);
  // 水平控制也没有被回滚
  assert.equal(bot.controlState.forward, true);

  // 松开定时器照旧生效：脉冲通路没有被异常打断
  await sleep(JUMP_PULSE_MS + 40);
  assert.equal(bot.controlState.jump, false);
  assert.deepEqual(controlOf(calls, 'jump'), [true, false]);
  resetIntentState(bot);
});

test('v2 的 use 头是模式字符串：slot + use=click 与 jump/attack 仍可同帧共存', async () => {
  // 意图里的 use 不是 `{item}`（那是 v1 宏动作的形状），而是 click/hold；
  // 「用哪个物品」由 slot 头 + 能力表决定。这里一次性把四个头都用上。
  const { bot, calls, dispatcher } = makeBot({ held: { name: 'fishing_rod', count: 1 } });
  const result = await applyIntent(bot, {
    move: 'forward',
    jump: true,
    attack: true,
    slot: 2,
    use: 'click',
  }, { dispatcher, itemTable: loadItemTable(), now: () => 4000 });

  assert.equal(result.applied.slot, 2);
  assert.equal(result.applied.jump, true);
  assert.equal(calls.dispatcherHits, 1);
  assert.equal(result.applied.use.mode, 'click');
  assert.deepEqual(controlOf(calls, 'jump'), [true]);
  resetIntentState(bot);
});
