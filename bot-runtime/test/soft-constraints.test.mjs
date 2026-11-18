import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BLOCK_HOLD_MS,
  DEFAULT_SOFT,
  applySoftActions,
  describeSoft,
  parseSoft,
  softDropReason,
} from '../src/soft-constraints.js';

const look = () => ({ type: 'look', params: { yaw: 90, pitch: 0 } });
const useOf = (item, holdMs) => ({ type: 'use', params: holdMs ? { item, holdMs } : { item } });
const jump = () => ({ type: 'jump', params: {} });
const attack = () => ({ type: 'attack', params: { targetEntityId: 7 } });

const softFrom = (spec) => {
  const parsed = parseSoft(spec);
  assert.equal(parsed.ok, true, `parseSoft(${JSON.stringify(spec)}) 应该成功: ${parsed.error}`);
  return parsed.soft;
};

test('DEFAULT_SOFT 全部允许，且是冻结的', () => {
  assert.deepEqual({ ...DEFAULT_SOFT }, {
    allowRod: true, allowGapple: true, allowBlock: true, allowJumpCrit: true,
    maxCps: null, minReachOffset: null,
  });
  assert.equal(Object.isFrozen(DEFAULT_SOFT), true);
});

test('空串/undefined = 默认全开', () => {
  for (const spec of ['', '   ', undefined, null]) {
    const parsed = parseSoft(spec);
    assert.equal(parsed.ok, true);
    assert.deepEqual({ ...parsed.soft }, { ...DEFAULT_SOFT });
    assert.deepEqual(parsed.entries, []);
  }
});

test('布尔三种写法：name / name=on|off / noName', () => {
  assert.equal(softFrom('rod').allowRod, true);
  assert.equal(softFrom('rod=off').allowRod, false);
  assert.equal(softFrom('rod=on').allowRod, true);
  assert.equal(softFrom('noRod').allowRod, false);
  assert.equal(softFrom('noGapple').allowGapple, false);
  assert.equal(softFrom('noBlock').allowBlock, false);
  assert.equal(softFrom('noJump').allowJumpCrit, false);
  // 大小写 / 连字符都应等价
  assert.equal(softFrom('NO-ROD').allowRod, false);
  assert.equal(softFrom('Rod=OFF').allowRod, false);
});

test('数值键：cps / reachOffset（含串行混写）', () => {
  const soft = softFrom('noRod,noGapple,cps=12,reachOffset=0.5');
  assert.equal(soft.allowRod, false);
  assert.equal(soft.allowGapple, false);
  assert.equal(soft.maxCps, 12);
  assert.equal(soft.minReachOffset, 0.5);
  assert.equal(soft.allowBlock, true);
});

test('entries 记录生效项（供启动日志/回显）', () => {
  const parsed = parseSoft('noRod,cps=8');
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.entries.map((entry) => [entry.key, entry.value]),
    [['allowRod', false], ['maxCps', 8]]);
});

test('未知键报错而不是静默忽略', () => {
  const parsed = parseSoft('noRod,fly=on');
  assert.equal(parsed.ok, false);
  assert.match(parsed.error, /未知软约束键: fly/);
  assert.match(parsed.error, /rod, gapple, block, jump, cps, reachOffset/);
});

test('非法值报错', () => {
  for (const [spec, pattern] of [
    ['cps=99', /1\.\.40/],
    ['cps=0', /1\.\.40/],
    ['cps=abc', /不是数字/],
    ['cps', /需要值/],
    ['rod=maybe', /不是布尔/],
    ['noRod=1', /不能同时带值/],
    ['noCps', /没有意义/],
    ['reachOffset=9', /0\.\.3/],
    ['rod,,', /^$|/],                       // 空段直接跳过，不报错
  ]) {
    const parsed = parseSoft(spec);
    if (spec === 'rod,,') {
      assert.equal(parsed.ok, true, '空段应被忽略');
      continue;
    }
    assert.equal(parsed.ok, false, `${spec} 应该报错`);
    assert.match(parsed.error, pattern, `${spec} 的报错信息不符: ${parsed.error}`);
  }
});

test('describeSoft：默认全开，禁用项按中文列出', () => {
  assert.equal(describeSoft(DEFAULT_SOFT), '动作全开');
  assert.equal(describeSoft(softFrom('noRod,noGapple')), '禁用 鱼竿、金苹果');
  assert.equal(describeSoft(softFrom('noBlock,noJump,cps=12')),
    '禁用 格挡、跳跃暴击；CPS 上限 12');
  assert.equal(describeSoft(softFrom('reachOffset=0.5')), '动作全开；攻击距离预留 0.5 格');
  // 部分字段缺失时按默认值补齐
  assert.equal(describeSoft({ allowRod: false }), '禁用 鱼竿');
});

test('剔除 rod / gapple / block / jump，其余保留且顺序不变', () => {
  const actions = [
    look(), attack(), useOf('rod'), useOf('gapple', 1600), useOf('sword', BLOCK_HOLD_MS), jump(),
    useOf('sword', 50),                  // 切剑：allowBlock=false 也不能误杀
  ];
  const filtered = applySoftActions(actions, softFrom('noRod,noGapple,noBlock,noJump'));
  assert.deepEqual(filtered.map((action) => action.type), ['look', 'attack', 'use']);
  assert.deepEqual(filtered[2].params, { item: 'sword', holdMs: 50 });
});

test('默认软约束（全开）不改变动作数组内容', () => {
  const actions = [look(), useOf('rod'), useOf('gapple', 1600), jump(), attack()];
  const filtered = applySoftActions(actions, DEFAULT_SOFT);
  assert.deepEqual(filtered, actions);
  assert.notEqual(filtered, actions, '应该返回新数组');
});

test('全被禁后补一个 idle，避免空 tick', () => {
  assert.deepEqual(applySoftActions([jump()], softFrom('noJump')),
    [{ type: 'idle', params: { durationMs: 180 } }]);
  assert.deepEqual(applySoftActions([useOf('rod')], softFrom('noRod')),
    [{ type: 'idle', params: { durationMs: 180 } }]);
  assert.deepEqual(applySoftActions([useOf('gapple', 1600)], softFrom('noGapple')),
    [{ type: 'idle', params: { durationMs: 180 } }]);
});

test('只剩 look 时也补 idle（否则这一 tick 空转）', () => {
  const filtered = applySoftActions([look(), useOf('rod')], softFrom('noRod'));
  assert.deepEqual(filtered.map((action) => action.type), ['look', 'idle']);
});

test('有位移时不再补 idle', () => {
  const move = { type: 'move', params: { direction: 'forward', durationMs: 180 } };
  const filtered = applySoftActions([look(), move, useOf('rod')], softFrom('noRod'));
  assert.deepEqual(filtered.map((action) => action.type), ['look', 'move']);
});

test('applySoftActions 不修改入参', () => {
  const actions = [look(), useOf('rod'), jump()];
  const snapshot = JSON.stringify(actions);
  applySoftActions(actions, softFrom('noRod,noJump'));
  assert.equal(JSON.stringify(actions), snapshot);
});

test('softDropReason 逐项判定（含 block 与切剑的区分）', () => {
  const soft = softFrom('noRod,noGapple,noBlock,noJump');
  assert.equal(softDropReason(useOf('rod'), soft), 'rod');
  assert.equal(softDropReason(useOf('gapple', 1600), soft), 'gapple');
  assert.equal(softDropReason(useOf('sword', BLOCK_HOLD_MS), soft), 'block');
  assert.equal(softDropReason(useOf('block'), soft), 'block');
  assert.equal(softDropReason(jump(), soft), 'jump');
  assert.equal(softDropReason(useOf('sword', 50), soft), null);
  assert.equal(softDropReason(look(), soft), null);
  assert.equal(softDropReason(attack(), soft), null);
  assert.equal(softDropReason(null, soft), null);
});

test('maxCps 只影响描述，不影响动作集合', () => {
  const slow = softFrom('cps=2');
  const actions = [look(), attack(), useOf('rod'), jump()];
  assert.deepEqual(applySoftActions(actions, slow), actions);
  assert.equal(slow.maxCps, 2);
  assert.match(describeSoft(slow), /CPS 上限 2/);
});
