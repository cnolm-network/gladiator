/**
 * 大脑遥测（G-717-B/C）的测试：心跳里的 dist/armor、远距离「赶路中」节流、
 * 目标离线/回来的一次性日志、以及护甲点数换算。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ARMOR_POINTS,
  FarChaseNotifier,
  TargetPresence,
  armorPoints,
  formatHeartbeat,
  horizontalDistance,
} from '../src/brain-telemetry.js';
import { debugLog, isDebugEnabled, setDebugEnabled } from '../src/debug.js';

/** 一帧"正常在打"的 brainState 快照（各用例按需覆盖字段）。 */
function state(overrides = {}) {
  return {
    decisions: 2486,
    attacks: 132,
    actionFailures: 3,
    waitingKit: 2,
    actionTimeouts: 0,
    busy: false,
    busySkips: 512,
    notReadySkips: 0,
    // G-721：盲决策计数与原始实体数（心跳字段名是 blind= / entities=）。
    blind: 0,
    entitiesSeen: 0,
    lastAction: null,
    lastActionAt: 0,
    lastTickAt: 1000,
    target: 'lvsm_',
    opponentHealth: 18.4,
    dist: 3.24,
    armor: 20,
    lastError: null,
    ...overrides,
  };
}

test('心跳行包含 dist / armor / waitingKit，且顺序稳定', () => {
  const line = formatHeartbeat(state(), 1046);
  assert.match(line, /^\[brain\] 心跳 decisions=2486 attacks=132 failures=3 waitingKit=2 timeouts=0 /);
  assert.match(line, / busy=0 skips=512 notReady=0 blind=0 entities=0 dist=3\.2 armor=20 /);
  assert.match(line, / last=idle lastTick=46ms target=lvsm_ oppHp=18\.4$/);
});

test('没有目标时 dist=-', () => {
  const line = formatHeartbeat(state({ dist: null, target: null }), 1000);
  assert.match(line, /dist=- /);
  assert.match(line, /target=- /);
});

test('正在执行动作时 last=动作(耗时)', () => {
  const line = formatHeartbeat(state({ lastAction: 'attack', lastActionAt: 700 }), 1000);
  assert.match(line, /last=attack\(300ms\)/);
});

test('lastError 只在有值时追加在末尾', () => {
  assert.doesNotMatch(formatHeartbeat(state(), 1000), /lastError/);
  assert.match(formatHeartbeat(state({ lastError: '动作超时 attack 3001ms' }), 1000),
    /lastError=动作超时 attack 3001ms$/);
});

test('护甲点数与 stats 的 armor= 同口径（钻石全套 20）', () => {
  assert.equal(ARMOR_POINTS.diamond_chestplate, 8);
  const full = [
    { slot: 5, name: 'diamond_helmet' },
    { slot: 6, name: 'diamond_chestplate' },
    { slot: 7, name: 'diamond_leggings' },
    { slot: 8, name: 'diamond_boots' },
  ];
  assert.equal(armorPoints(full), 20);
  assert.equal(armorPoints([{ slot: 6, name: 'iron_chestplate' }]), 6);
  // 背包里的护甲（非 5–8 槽）不算已穿戴。
  assert.equal(armorPoints([{ slot: 20, name: 'diamond_chestplate' }]), 0);
  assert.equal(armorPoints([]), 0);
  assert.equal(armorPoints(null), 0);
  assert.equal(armorPoints([{ slot: 5, name: 'not_an_armor' }]), 0);
});

test('水平距离只算 x/z', () => {
  assert.equal(horizontalDistance({ x: 0, y: 0, z: 0 }, { x: 3, y: 0, z: 4 }), 5);
  // 竖直差（跳/台阶）不影响
  assert.equal(horizontalDistance({ x: 0, y: 4, z: 0 }, { x: 0, y: 40, z: 2 }), 2);
  assert.equal(horizontalDistance(null, { x: 0, y: 0, z: 0 }), null);
  assert.equal(horizontalDistance({ x: NaN, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }), null);
});

test('>24 格时每 5 秒报一条「赶路中」，回到近处重置', () => {
  const notifier = new FarChaseNotifier({ threshold: 24, everyMs: 5000 });
  // 第 1 局开局实测 1114 格
  const line = notifier.update(1_000_000, 1114.4, 1114.4);
  assert.equal(line, '[brain] 目标在 1114 格之外，赶路中（本局最长 1114 格）');
  // 5 秒内不重复
  assert.equal(notifier.update(1_003_000, 1000, 1114.4), null);
  // 5 秒后再说一句（括号里仍是本局最远）
  assert.equal(notifier.update(1_005_100, 968.2, 1114.4),
    '[brain] 目标在 968 格之外，赶路中（本局最长 1114 格）');
  // 进 24 格内：不报，并且重置节流（下一帧再拉开要立刻报）
  assert.equal(notifier.update(1_006_000, 20, 1114.4), null);
  assert.match(notifier.update(1_006_500, 30, 1114.4), /目标在 30 格之外/);
});

test('刚好 24 格不报（阈值是「超过」）', () => {
  const notifier = new FarChaseNotifier({ threshold: 24, everyMs: 5000 });
  assert.equal(notifier.update(0, 24, 24), null);
  assert.match(notifier.update(1, 24.5, 24.5), /目标在 25 格之外/);
});

test('目标不在线超过 5 秒只报一次，回来再报一次', () => {
  const presence = new TargetPresence({ offlineAfterMs: 5000, name: 'lvsm_' });
  const t0 = 10_000;
  assert.equal(presence.update(t0, false), null, '刚消失不报（可能在传送/重连）');
  assert.equal(presence.update(t0 + 4_900, false), null, '未满 5 秒');
  assert.equal(presence.update(t0 + 5_100, false), 'offline', '满 5 秒报一次');
  assert.equal(presence.update(t0 + 6_000, false), null, '之后保持静默');
  assert.equal(presence.update(t0 + 60_000, false), null, '一直不在线也不重复报');
  assert.equal(presence.update(t0 + 61_000, true), 'back', '回来说明一次');
  assert.equal(presence.update(t0 + 62_000, true), null, '之后保持静默');
});

test('reset 后重新计时（每局重新算）', () => {
  const presence = new TargetPresence({ offlineAfterMs: 5000, name: 'lvsm_' });
  assert.equal(presence.update(0, false), null);
  assert.equal(presence.update(6000, false), 'offline');
  presence.reset('lvsm_');
  assert.equal(presence.update(7000, false), null, '清零后要重新等 5 秒');
  assert.equal(presence.update(12_100, false), 'offline');
});

test('调试日志默认静默，打开后带 [debug] 前缀', () => {
  const lines = [];
  const original = console.log;
  console.log = (...args) => lines.push(args.join(' '));
  try {
    setDebugEnabled(false);
    assert.equal(isDebugEnabled(), false);
    debugLog('[intent] #1 chase');
    assert.deepEqual(lines, [], '默认不该输出');

    setDebugEnabled(true);
    debugLog('[intent] #1 chase');
    assert.equal(lines.length, 1);
    assert.match(lines[0], /^\[debug\] \[intent\] #1 chase$/);
  } finally {
    console.log = original;
    setDebugEnabled(false);
  }
});
