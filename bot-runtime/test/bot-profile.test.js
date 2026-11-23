/**
 * 单 Bot 行为配置（G-507）：CPS 节拍、反应延时、攻击距离。
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ActionDispatcher } from '../src/actions.js';
import {
  ClickGate,
  DEFAULT_PROFILE,
  LEGIT_REACH_MAX,
  MAX_REACH,
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

test('默认配置是 15±2 CPS / 220±40ms / 3.0 格（原版水平，非 3.99）', () => {
  assert.equal(DEFAULT_PROFILE.cps, 15);
  assert.equal(DEFAULT_PROFILE.cpsJitter, 2);
  assert.equal(DEFAULT_PROFILE.reactionMs, 220);
  assert.equal(DEFAULT_PROFILE.reactionJitterMs, 40);
  assert.equal(DEFAULT_PROFILE.reach, 3.0);
  assert.equal(DEFAULT_PROFILE.cheatReach, false);
  assert.equal(DEFAULT_PROFILE.model, 'default');
  assert.equal(LEGIT_REACH_MAX, 3.0);
  assert.equal(MAX_REACH, 6.0);
});

test('reach 超限但没开 cheat：降回 3.0 并告警（不静默接受）', () => {
  const warnings = [];
  const profile = normalizeProfile({ reach: 3.99 }, { warn: (m) => warnings.push(m), error: () => {} });
  assert.equal(profile.reach, 3.0);
  assert.equal(profile.cheatReach, false);
  assert.ok(warnings.some((m) => m.includes('3.99')), warnings.join('|'));
});

test('显式 cheatReach 才允许超限，并打 [cheat] 标记', () => {
  const warnings = [];
  const profile = normalizeProfile({ reach: 3.5, cheatReach: true }, { warn: (m) => warnings.push(m) });
  assert.equal(profile.reach, 3.5);
  assert.equal(profile.cheatReach, true);
  assert.ok(warnings.some((m) => m.startsWith('[cheat] reach 超限=3.5')), warnings.join('|'));
});

test('reach 越界回落默认；>3 但 ≤6 只有 cheat 才保住', () => {
  const logger = { warn: () => {} };
  // 低于下限 / 高于硬上限：一律回落默认
  assert.equal(normalizeProfile({ reach: 0.5 }, logger).reach, 3.0);
  assert.equal(normalizeProfile({ reach: 7 }, logger).reach, 3.0);
  // 合法区间内
  assert.equal(normalizeProfile({ reach: 1.0 }, logger).reach, 1.0);
  assert.equal(normalizeProfile({ reach: 2.95 }, logger).reach, 2.95);
  // 3~6 需要 cheat：没有就降回 3.0，有就保留
  assert.equal(normalizeProfile({ reach: 6 }, logger).reach, 3.0);
  assert.equal(normalizeProfile({ reach: 6, cheatReach: true }, logger).reach, 6.0);
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
  assert.equal(profile.reach, 3.0);
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

test('attack 超出 reach 不抛错，改为追击并返回 OUT_OF_REACH_CHASED', async () => {
  const bot = fakeBot({ entityPosition: { x: 5, y: 64, z: 0 } });
  const dispatcher = new ActionDispatcher({
    bot,
    profile: { reach: 3.99, reactionMs: 0, reactionJitterMs: 0 },
  });
  // 硬打只会吃 OUT_OF_REACH，所以仍然不许真的挥；但也不再抛错（环境侧不再刷失败日志）。
  const result = await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } });
  assert.equal(result.ok, true);
  assert.equal(result.code, 'OUT_OF_REACH_CHASED');
  assert.ok(result.distance > result.reach, `distance=${result.distance}`);
  assert.equal(bot.attacks.length, 0);
  // 追一步 = 按下前进 + 疾跑
  assert.deepEqual(bot.controls.filter(([name]) => name === 'forward' || name === 'sprint'),
    [['forward', true], ['sprint', true]]);
  assert.equal(dispatcher.hold.direction, 'forward');
  assert.equal(dispatcher.hold.sprint, true);
  // 延迟释放：宽限期过后必须松开，不能把键按死
  await new Promise((resolve) => setTimeout(resolve, 220));
  assert.deepEqual(bot.controls.slice(2), [['forward', false], ['sprint', false]]);
  assert.equal(dispatcher.hold, null);
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
