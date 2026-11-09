import test from 'node:test';
import assert from 'node:assert/strict';

import { RestrictionCache } from '../src/restriction-cache.js';

test('put 归一化矩形并只在矩形内 deny', () => {
  const cache = new RestrictionCache({ now: () => 1000 });
  cache.put({
    regionId: 'arena-deny',
    box: { minX: 10, minY: 60, minZ: 10, maxX: 0, maxY: 64, maxZ: 0 },
  });
  assert.equal(cache.check(5, 62, 5).denied, true);
  assert.equal(cache.check(30, 62, 30).denied, false);
  assert.equal(cache.check(5, 70, 5).denied, false);
});

test('离开矩形即失效，不把 A 区域的规则带到 B 区域', () => {
  const cache = new RestrictionCache({ now: () => 1000 });
  cache.put({ regionId: 'a', box: { minX: 0, minY: 0, minZ: 0, maxX: 5, maxY: 5, maxZ: 5 } });
  assert.equal(cache.check(2, 2, 2).denied, true);
  assert.equal(cache.check(50, 2, 50).denied, false);
});

test('TTL 过期后自动放行', () => {
  let now = 1000;
  const cache = new RestrictionCache({ now: () => now, defaultTtlMs: 500 });
  cache.put({ regionId: 'tmp', box: { minX: 0, minY: 0, minZ: 0, maxX: 5, maxY: 5, maxZ: 5 } });
  assert.equal(cache.check(1, 1, 1).denied, true);
  now = 1600;
  assert.equal(cache.check(1, 1, 1).denied, false);
  assert.equal(cache.size, 0);
});

test('allowPvp=true 的条目不会 deny，snapshot 只返回当前条目', () => {
  let now = 1000;
  const cache = new RestrictionCache({ now: () => now });
  cache.put({ regionId: 'pvp-ok', box: { minX: 0, minY: 0, minZ: 0, maxX: 5, maxY: 5, maxZ: 5 }, allowPvp: true });
  assert.equal(cache.check(1, 1, 1).denied, false);
  cache.put({ regionId: 'deny', box: { minX: 0, minY: 0, minZ: 0, maxX: 5, maxY: 5, maxZ: 5 } });
  const snapshot = cache.snapshot();
  assert.equal(snapshot.length, 2);
  assert.equal(snapshot.find((item) => item.regionId === 'deny').allowPvp, false);
});

test('replaceAll 以服务端列表为真源并丢弃 allow 条目', () => {
  const cache = new RestrictionCache({ now: () => 1000 });
  cache.put({ regionId: 'stale', box: { minX: 0, minY: 0, minZ: 0, maxX: 1, maxY: 1, maxZ: 1 } });
  cache.replaceAll([
    { regionId: 'new', box: { minX: 10, minY: 10, minZ: 10, maxX: 20, maxY: 20, maxZ: 20 }, allowPvp: false },
    { regionId: 'free', allowPvp: true },
  ]);
  assert.equal(cache.size, 1);
  assert.equal(cache.check(15, 15, 15).denied, true);
});

test('maxEntries 超出时淘汰最旧条目', () => {
  const cache = new RestrictionCache({ now: () => 1000, maxEntries: 2 });
  cache.put({ regionId: 'a', box: { minX: 0, minY: 0, minZ: 0, maxX: 1, maxY: 1, maxZ: 1 } });
  cache.put({ regionId: 'b', box: { minX: 2, minY: 2, minZ: 2, maxX: 3, maxY: 3, maxZ: 3 } });
  cache.put({ regionId: 'c', box: { minX: 4, minY: 4, minZ: 4, maxX: 5, maxY: 5, maxZ: 5 } });
  assert.equal(cache.size, 2);
  assert.equal(cache.check(0, 0, 0).denied, false);
  assert.equal(cache.check(4, 4, 4).denied, true);
});
