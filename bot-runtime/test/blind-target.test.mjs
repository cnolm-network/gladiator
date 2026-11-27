/**
 * G-721：盲决策自检的测试。
 *
 * <p>背景（真机 2026-11-25）：板子开局 tp 落在 bot 登录窗口内，被客户端位置预测吃掉，
 * 服务端坐标正确而客户端还停在原地 → `observation.entities` 只有自己一个 →
 * `has_target=0`，`target_dx/dy/dz/distance/target_health/opponent_speed/in_reach`
 * 全部静默变 0，策略照常决策却毫无告警。这组用例锁住三件事：
 * 1. `has_target` 下标与 `features.OBS_FIELDS_JS` 同序（改观测定义时立刻红）；
 * 2. 「大脑有目标 + 观测没目标」的判定是纯函数，0/1/NaN/越界都稳；
 * 3. 告警文案与心跳字段（`blind=` / `entities=`）的形状。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { OBS_FIELDS_JS } from '../src/features.js';
import {
  BlindWatch,
  HAS_TARGET_INDEX,
  TARGET_BACK_LINE,
  formatBlindAlert,
  formatHeartbeat,
  hasVisibleTarget,
} from '../src/brain-telemetry.js';

/** 造一个 39 维观测向量，只改需要的位（G-731 从 37 扩到 39）。 */
function features(overrides = {}) {
  const vector = new Float64Array(39);
  for (const [index, value] of Object.entries(overrides)) {
    vector[Number(index)] = value;
  }
  return vector;
}

test('HAS_TARGET_INDEX 指向 has_target（与 spaces.py / features.js 同序）', () => {
  assert.equal(HAS_TARGET_INDEX, 13);
  assert.equal(OBS_FIELDS_JS[HAS_TARGET_INDEX], 'has_target');
});

test('hasVisibleTarget 只认 has_target 这一维', () => {
  assert.equal(hasVisibleTarget(features({ 13: 1 })), true);
  assert.equal(hasVisibleTarget(features({ 13: 0 })), false);
  // 「贴着站」也可能让 dx/dz 是 0，但那不算看得见——必须靠 has_target。
  assert.equal(hasVisibleTarget(features({ 7: 0, 8: 0, 9: 0, 10: 0, 22: 1 })), false);
  assert.equal(hasVisibleTarget(features({ 13: 1, 7: -0.2 })), true);
  assert.equal(hasVisibleTarget(features({ 13: Number.NaN })), false);
  assert.equal(hasVisibleTarget(null), false);
  assert.equal(hasVisibleTarget(undefined), false);
  assert.equal(hasVisibleTarget(new Float64Array(0)), false);
  assert.equal(hasVisibleTarget(features({ 13: 1 }), 99), false, '越界一律当看不见');
});

test('告警文案固定为 [brain] 目标不可见（entities=N）', () => {
  assert.equal(formatBlindAlert(0), '[brain] 目标不可见（entities=0）');
  assert.equal(formatBlindAlert(1), '[brain] 目标不可见（entities=1）');
  assert.equal(formatBlindAlert(7), '[brain] 目标不可见（entities=7）');
  assert.equal(formatBlindAlert(3.9), '[brain] 目标不可见（entities=3）');
  // 缺值/异常值不能把 N 打成 NaN 或负数
  assert.equal(formatBlindAlert(), '[brain] 目标不可见（entities=0）');
  assert.equal(formatBlindAlert(null), '[brain] 目标不可见（entities=0）');
  assert.equal(formatBlindAlert(-5), '[brain] 目标不可见（entities=0）');
  assert.equal(formatBlindAlert('abc'), '[brain] 目标不可见（entities=0）');
});

test('盲：首次立刻报，5 秒内不重复，之后每 5 秒一条', () => {
  const watch = new BlindWatch({ everyMs: 5000 });
  const t0 = 1_000_000;
  assert.deepEqual(watch.update(t0, { expected: true, visible: false, entities: 1 }),
    { blind: true, line: '[brain] 目标不可见（entities=1）' });
  assert.deepEqual(watch.update(t0 + 50, { expected: true, visible: false, entities: 1 }),
    { blind: true, line: null }, '50ms 后（20Hz 下一帧）不刷屏');
  assert.deepEqual(watch.update(t0 + 4_900, { expected: true, visible: false, entities: 1 }),
    { blind: true, line: null });
  assert.deepEqual(watch.update(t0 + 5_000, { expected: true, visible: false, entities: 2 }),
    { blind: true, line: '[brain] 目标不可见（entities=2）' }, '满 5 秒再报一条（带最新实体数）');
});

test('恢复可见只报一次，再盲再报', () => {
  const watch = new BlindWatch({ everyMs: 5000 });
  assert.match(watch.update(0, { expected: true, visible: false, entities: 1 }).line,
    /目标不可见/);
  assert.deepEqual(watch.update(100, { expected: true, visible: true, entities: 3 }),
    { blind: false, line: TARGET_BACK_LINE });
  assert.deepEqual(watch.update(200, { expected: true, visible: true, entities: 3 }),
    { blind: false, line: null }, '可见时保持静默');
  assert.deepEqual(watch.update(300, { expected: true, visible: false, entities: 1 }),
    { blind: true, line: '[brain] 目标不可见（entities=1）' }, '重新进入盲状态要立刻报');
});

test('大脑本来就没目标时不算盲（那是 idle(no-target) 分支，不是盲决策）', () => {
  const watch = new BlindWatch({ everyMs: 5000 });
  assert.deepEqual(watch.update(0, { expected: false, visible: false, entities: 1 }),
    { blind: false, line: null });
  assert.deepEqual(watch.update(10, { expected: false, visible: false, entities: 0 }),
    { blind: false, line: null });
});

test('reset 后重新计节流（每局重新开始）', () => {
  const watch = new BlindWatch({ everyMs: 5000 });
  assert.match(watch.update(0, { expected: true, visible: false, entities: 1 }).line,
    /目标不可见/);
  assert.equal(watch.update(100, { expected: true, visible: false, entities: 1 }).line, null);
  watch.reset();
  assert.match(watch.update(200, { expected: true, visible: false, entities: 1 }).line,
    /目标不可见/, 'reset 后第一条立刻报');
});

test('心跳带 blind=N 与 entities=N，位置在 notReady 之后', () => {
  const line = formatHeartbeat({
    decisions: 120,
    attacks: 3,
    actionFailures: 0,
    waitingKit: 0,
    actionTimeouts: 0,
    busy: false,
    busySkips: 0,
    notReadySkips: 0,
    blind: 47,
    entitiesSeen: 1,
    lastAction: null,
    lastActionAt: 0,
    lastTickAt: 1000,
    target: 'lvsm_',
    opponentHealth: null,
    dist: null,
    armor: 0,
    lastError: null,
  }, 1046);
  assert.match(line, /busy=0 skips=0 notReady=0 blind=47 entities=1 dist=- /);
  assert.match(line, /oppHp=-$/);
});

test('心跳没有 blind 字段时退化为 0（老快照/老测试不受影响）', () => {
  const line = formatHeartbeat({ decisions: 1, lastTickAt: 0 }, 0);
  assert.match(line, /blind=0 entities=0 /);
});
