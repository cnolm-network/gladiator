import test from 'node:test';
import assert from 'node:assert/strict';

import { ActionScheduler } from '../src/scheduler.js';

test('允许窗口内动作', async () => {
  const scheduler = new ActionScheduler({ maxPerSecond: 3, now: () => 1000 });
  const results = await Promise.all([
    scheduler.run(1, async () => {}),
    scheduler.run(2, async () => {}),
    scheduler.run(3, async () => {}),
  ]);
  assert.deepEqual(results.map((r) => r.ok), [true, true, true]);
  assert.equal(scheduler.stats.accepted, 3);
});

test('超过 maxPerSecond 回 ACTION_RATE_LIMITED 且不执行', async () => {
  const scheduler = new ActionScheduler({ maxPerSecond: 2, now: () => 1000 });
  let executed = 0;
  await scheduler.run(1, async () => { executed += 1; });
  await scheduler.run(2, async () => { executed += 1; });
  const blocked = await scheduler.run(3, async () => { executed += 1; });
  assert.equal(blocked.ok, false);
  assert.equal(blocked.code, 'ACTION_RATE_LIMITED');
  assert.equal(executed, 2);
  assert.equal(scheduler.stats.rateLimited, 1);
});

test('1 秒窗口滑动后恢复', async () => {
  let now = 1000;
  const scheduler = new ActionScheduler({ maxPerSecond: 1, now: () => now });
  assert.equal((await scheduler.run(1, async () => {})).ok, true);
  assert.equal((await scheduler.run(2, async () => {})).ok, false);
  now = 2100;
  assert.equal((await scheduler.run(3, async () => {})).ok, true);
});

test('重复 seq 回放缓存结果并标记 deduped', async () => {
  const scheduler = new ActionScheduler({ maxPerSecond: 10, now: () => 1000 });
  let executed = 0;
  const first = await scheduler.run(7, async () => { executed += 1; });
  const second = await scheduler.run(7, async () => { executed += 1; });
  assert.equal(first.ok, true);
  assert.equal(first.deduped, undefined);
  assert.equal(second.ok, true);
  assert.equal(second.deduped, true);
  assert.equal(executed, 1);
  assert.equal(scheduler.stats.deduped, 1);
});

test('执行异常转结构化 code', async () => {
  const scheduler = new ActionScheduler({ now: () => 1000 });
  const error = new Error('区域禁止');
  error.code = 'RULE_DENIED';
  const result = await scheduler.run(1, async () => { throw error; });
  assert.deepEqual(result, { ok: false, code: 'RULE_DENIED', reason: '区域禁止' });
});
