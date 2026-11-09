import assert from 'node:assert/strict';
import test from 'node:test';

import { Backoff } from '../src/connection.js';

test('默认序列为 1s→2s→4s→8s→16s→30s 封顶', () => {
  const backoff = new Backoff({ jitter: 0 });
  const delays = [];
  for (let i = 0; i < 7; i += 1) {
    delays.push(backoff.nextDelay());
  }
  assert.deepEqual(delays, [1000, 2000, 4000, 8000, 16000, 30000, 30000]);
});

test('reset 后重新从 initialMs 开始', () => {
  const backoff = new Backoff({ jitter: 0 });
  backoff.nextDelay();
  backoff.nextDelay();
  assert.equal(backoff.attempt, 2);
  backoff.reset();
  assert.equal(backoff.nextDelay(), 1000);
});

test('抖动保持在基准值 ±20% 范围内', () => {
  const backoff = new Backoff({ jitter: 0.2, random: () => 0.999 });
  for (let i = 0; i < 6; i += 1) {
    const base = Math.min(30000, 1000 * 2 ** i);
    const delay = backoff.nextDelay();
    assert.ok(delay >= base * 0.79 && delay <= base * 1.21, `${delay} 超出 ${base} 的抖动范围`);
  }
});

test('自定义上限与倍率生效', () => {
  const backoff = new Backoff({ initialMs: 100, maxMs: 500, factor: 3, jitter: 0 });
  assert.deepEqual(
    [backoff.nextDelay(), backoff.nextDelay(), backoff.nextDelay(), backoff.nextDelay()],
    [100, 300, 500, 500],
  );
});
