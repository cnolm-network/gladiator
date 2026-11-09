/**
 * 单 Bot 行为配置（G-507）：CPS 节拍、反应延时、攻击距离。
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ActionDispatcher } from '../src/actions.js';
import {
  ClickGate,
  DEFAULT_PROFILE,
  ReactionDelay,
  normalizeProfile,
  usesReactionDelay,
} from '../src/bot-profile.js';

function fakeBot({ position = { x: 0, y: 64, z: 0 }, entityPosition = { x: 2, y: 64, z: 0 } } = {}) {
  const attacks = [];
  const controls = [];
  return {
    attacks,
    controls,
    entity: { position },
    entities: { 7: { position: entityPosition } },
    setControlState: (name, value) => controls.push([name, value]),
    attack: (entity) => attacks.push(entity),
    chat: () => {},
    look: async () => {},
  };
}

test('默认配置是 15±2 CPS / 220±40ms / 3.99 格', () => {
  assert.equal(DEFAULT_PROFILE.cps, 15);
  assert.equal(DEFAULT_PROFILE.cpsJitter, 2);
  assert.equal(DEFAULT_PROFILE.reactionMs, 220);
  assert.equal(DEFAULT_PROFILE.reactionJitterMs, 40);
  assert.equal(DEFAULT_PROFILE.reach, 3.99);
  assert.equal(DEFAULT_PROFILE.model, 'default');
});

test('normalizeProfile 对脏值回落到默认并告警', () => {
  const warnings = [];
  const profile = normalizeProfile(
    { cps: -1, cpsJitter: 99, reactionMs: 'abc', reach: 12, model: '  ' },
    { warn: (m) => warnings.push(m) },
  );
  assert.equal(profile.cps, 15);
  assert.equal(profile.cpsJitter, 2);
  assert.equal(profile.reactionMs, 220);
  assert.equal(profile.reach, 3.99);
  assert.equal(profile.model, 'default');
  assert.ok(warnings.length >= 4);
});

test('ClickGate 按 cps±jitter 生成间隔并记录下一次窗口', () => {
  let now = 0;
  const gate = new ClickGate({ cps: 10, cpsJitter: 0 }, { now: () => now, random: () => 0.5 });
  assert.equal(gate.canClick(), true);
  gate.recordClick();
  assert.equal(gate.nextAllowedAt, 100);
  assert.equal(gate.canClick(), false);
  now = 60;
  assert.equal(gate.remainingMs(), 40);
  now = 100;
  assert.equal(gate.canClick(), true);
});

test('ClickGate 抖动落在 ±jitter 内', () => {
  const low = new ClickGate({ cps: 15, cpsJitter: 2 }, { now: () => 0, random: () => 0 });
  const high = new ClickGate({ cps: 15, cpsJitter: 2 }, { now: () => 0, random: () => 1 });
  const lowInterval = low.nextIntervalMs();
  const highInterval = high.nextIntervalMs();
  // random=0 → 13 cps → 76.9ms；random=1 → 17 cps → 58.8ms
  assert.ok(Math.abs(lowInterval - 1000 / 13) < 1e-6);
  assert.ok(Math.abs(highInterval - 1000 / 17) < 1e-6);
});

test('ClickGate.waitForSlot 会等待到下一个窗口', async () => {
  const gate = new ClickGate({ cps: 20, cpsJitter: 0 });
  gate.recordClick();
  const started = Date.now();
  const waited = await gate.waitForSlot();
  const elapsed = Date.now() - started;
  assert.ok(waited > 0);
  assert.ok(elapsed >= 35, `elapsed=${elapsed}`);
});

test('ReactionDelay 用注入随机数产出 220±40ms', () => {
  const mid = new ReactionDelay({ reactionMs: 220, reactionJitterMs: 40 }, { random: () => 0.5 });
  assert.equal(mid.nextDelayMs(), 220);
  const low = new ReactionDelay({ reactionMs: 220, reactionJitterMs: 40 }, { random: () => 0 });
  assert.equal(low.nextDelayMs(), 180);
  const high = new ReactionDelay({ reactionMs: 220, reactionJitterMs: 40 }, { random: () => 1 });
  assert.equal(high.nextDelayMs(), 260);
});

test('talk/idle 不计反应延时，战斗动作计入', () => {
  assert.equal(usesReactionDelay('talk'), false);
  assert.equal(usesReactionDelay('idle'), false);
  assert.equal(usesReactionDelay('attack'), true);
  // 移动是连续控制，不能每步加反应延时（否则走一下停一下）。
  assert.equal(usesReactionDelay('move'), false);
  assert.equal(usesReactionDelay('look'), false);
});

test('attack 超出 reach 直接回 OUT_OF_REACH 且不调用 bot.attack', async () => {
  const bot = fakeBot({ entityPosition: { x: 5, y: 64, z: 0 } });
  const dispatcher = new ActionDispatcher({
    bot,
    profile: { reach: 3.99, reactionMs: 0, reactionJitterMs: 0 },
  });
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } }),
    (error) => error.code === 'OUT_OF_REACH',
  );
  assert.equal(bot.attacks.length, 0);
});

test('attack 在 reach 内会受 CPS 节流，第二次点击等待', async () => {
  const bot = fakeBot({ entityPosition: { x: 2, y: 64, z: 0 } });
  const dispatcher = new ActionDispatcher({
    bot,
    profile: { cps: 20, cpsJitter: 0, reactionMs: 0, reactionJitterMs: 0, reach: 3.99 },
  });
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  const started = Date.now();
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  const elapsed = Date.now() - started;
  assert.equal(bot.attacks.length, 2);
  assert.ok(elapsed >= 35, `第二次点击应等待约 50ms，实际 ${elapsed}ms`);
  assert.equal(dispatcher.combatStats.clicks, 2);
});

test('talk 不受反应延时影响', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({
    bot,
    profile: { reactionMs: 300, reactionJitterMs: 0 },
  });
  const started = Date.now();
  await dispatcher.dispatch({ type: 'talk', params: { message: 'hi' } });
  assert.ok(Date.now() - started < 100, 'talk 不应等待 300ms');
});
