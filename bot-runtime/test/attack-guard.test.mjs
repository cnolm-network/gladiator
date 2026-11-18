/**
 * `attack-guard.js` 单测：换剑判定 + 无敌帧节流边界。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ATTACK_STREAK_LIMIT,
  DEFAULT_ATTACK_COOLDOWN_MS,
  isSwordItem,
  needsSwordSwap,
  shouldThrottleAttack,
} from '../src/attack-guard.js';

test('isSwordItem 只认以 _sword 结尾的物品', () => {
  assert.equal(isSwordItem({ name: 'diamond_sword' }), true);
  assert.equal(isSwordItem({ name: 'iron_sword' }), true);
  assert.equal(isSwordItem({ name: 'golden_apple' }), false);
  assert.equal(isSwordItem({ name: 'air' }), false);
  assert.equal(isSwordItem({}), false);
  assert.equal(isSwordItem(null), false);
  assert.equal(isSwordItem(undefined), false);
});

test('needsSwordSwap：空手/非剑需要切，持剑不需要', () => {
  assert.equal(needsSwordSwap(null), true);
  assert.equal(needsSwordSwap({ name: 'fishing_rod' }), true);
  assert.equal(needsSwordSwap({ name: 'diamond_sword' }), false);
});

test('连续不足 3 次时从不节流', () => {
  for (let streak = 1; streak < ATTACK_STREAK_LIMIT; streak += 1) {
    assert.equal(shouldThrottleAttack({
      streak, lastAttackAt: 1000, now: 1010, cooldownMs: DEFAULT_ATTACK_COOLDOWN_MS,
    }), false);
  }
});

test('连续 3 次后在 500ms 内节流，过了窗口放行', () => {
  const base = { streak: ATTACK_STREAK_LIMIT, lastAttackAt: 10_000 };
  assert.equal(shouldThrottleAttack({ ...base, now: 10_100 }), true);
  assert.equal(shouldThrottleAttack({ ...base, now: 10_499 }), true);
  assert.equal(shouldThrottleAttack({ ...base, now: 10_500 }), false);
  assert.equal(shouldThrottleAttack({ ...base, now: 10_900 }), false);
});

test('还没出过手 / cooldown=0 时不节流', () => {
  assert.equal(shouldThrottleAttack({ streak: 9, lastAttackAt: 0, now: 5_000 }), false);
  assert.equal(shouldThrottleAttack({ streak: 9, lastAttackAt: -1, now: 5_000 }), false);
  assert.equal(shouldThrottleAttack({
    streak: 9, lastAttackAt: 1_000, now: 1_001, cooldownMs: 0,
  }), false);
});

test('自定义窗口与阈值生效', () => {
  assert.equal(shouldThrottleAttack({
    streak: 2, lastAttackAt: 1_000, now: 1_050, cooldownMs: 200, streakLimit: 2,
  }), true);
  assert.equal(shouldThrottleAttack({
    streak: 2, lastAttackAt: 1_000, now: 1_250, cooldownMs: 200, streakLimit: 2,
  }), false);
});
