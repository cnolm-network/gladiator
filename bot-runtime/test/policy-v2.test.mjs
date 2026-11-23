/**
 * v2（十头 MultiDiscrete）策略的载入与前向单测（G-712）。
 *
 * <h3>为什么需要这一组</h3>
 * `bot-runtime/src/index.js` 原来硬校验 `numActions === 11`，v2 模型（`n_actions=40`、十个头）
 * 一加载就被拒。这里把「v2 能用」拆成可复现的判据：
 *
 * 1. **真文件载入**：`dashboard/var/models/v2-v4-dagger3/policy.npz`（当前最强陪练）的
 *    `_meta.actionSpace/headSizes/headNames` 必须被正确解析；
 * 2. **与 Python 逐头相等**：同一输入向量 + 同一 mask 下，`actWithMask(...,{greedy:true})`
 *    的结果必须与 `gladiator_train.policy.Policy.sample(..., mask=..., greedy=True)`
 *    （即 `act_with_mask` 的实现）**逐头一致**；
 * 3. **mask 语义**：屏蔽值概率归零、重归一；全关时回退全开并告警（或 strictMask 抛错）；
 * 4. **v1 回归**：`dagger-1118b` 的 `probs` 与 8 位小数逐位一致（动作空间改造不许碰老路）；
 * 5. **意图帧结构**：20Hz 下产出的 `{type:'intent', payload:{action:[...10]}}` 能被
 *    `intent.js::decodeIntentVector` + `normalizeIntent` 接住（= `link-server.js::onIntent` 的期望）。
 *
 * <h3>对照数值怎么复算（Python 侧）</h3>
 * 向量与 mask 都是**公式生成**的，不依赖随机数：
 * ```python
 * import numpy as np
 * from gladiator_train.policy import Policy
 * p = Policy.load('../dashboard/var/models/v2-v4-dagger3/policy.npz')
 * v = np.array([(((i * 37) % 23) - 11) / 7.0 for i in range(25)])
 * maskA = [[True]*5, [True]*7, [True]*5, [True, True], [True, False, False],
 *          [True, True], [True, True], [True, True],
 *          [False, True, True, False, False, False, False, False, False],
 *          [True, False, True]]
 * print(p.sample(v, np.random.default_rng(0), mask=maskA, greedy=True))  # [3 2 0 0 0 0 0 0 1 2]
 * ```
 */

import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';

import { DEFAULT_ITEMS_PATH, loadItemTable, matchingMask } from '../src/action-mask.js';
import { ACTION_DIM_V2, HEAD_ORDER_V2, HEAD_SIZES_V2 } from '../src/action-space-v2.js';
import { decodeIntentVector, normalizeIntent } from '../src/intent.js';
import { loadPolicy, detectActionSpace } from '../src/policy.js';

/** 当前最强的 v2 陪练模型（十头 / obs 25）。 */
const V2_MODEL = path.resolve('..', 'dashboard', 'var', 'models', 'v2-v4-dagger3', 'policy.npz');
/** v1 基线模型（单头 / 11 宏动作 / obs 23）。 */
const V1_MODEL = path.resolve('..', 'dashboard', 'var', 'models', 'dagger-1118b', 'policy.npz');

/** 固定输入向量（与 Python 侧同一条公式；25 维 = 当前运行时特征）。 */
function fixedVector(length = 25) {
  return Array.from({ length }, (_, index) => (((index * 37) % 23) - 11) / 7);
}

/** maskA：slot 只开 1/2，use 只开「不按」，sprint_reset 只开 none/block_hit。 */
function maskA() {
  return [
    Array(5).fill(true),
    Array(7).fill(true),
    Array(5).fill(true),
    [true, true],
    [true, false, false],
    [true, true],
    [true, true],
    [true, true],
    [false, true, true, false, false, false, false, false, false],
    [true, false, true],
  ];
}

/** maskB：在 maskA 上把 attack 只留 1、slot 只留 0/3（检验「同一向量、不同 mask」）。 */
function maskB() {
  const mask = maskA().map((row) => [...row]);
  mask[3] = [false, true];
  mask[8] = [true, false, false, true, false, false, false, false, false];
  return mask;
}

const ATTACK_HEAD = HEAD_ORDER_V2.indexOf('attack');
const SLOT_HEAD = HEAD_ORDER_V2.indexOf('slot');

test('v2 真模型：meta 的 actionSpace / 头数 / 头名解析正确', () => {
  const policy = loadPolicy(V2_MODEL);
  assert.equal(policy.actionSpace, 'v2');
  assert.deepEqual(policy.headNames, [...HEAD_ORDER_V2]);
  assert.deepEqual(policy.headSizes, [...HEAD_SIZES_V2]);
  assert.deepEqual(policy.headSizes, [5, 7, 5, 2, 3, 2, 2, 2, 9, 3]);
  assert.equal(policy.numActions, ACTION_DIM_V2);
  assert.equal(policy.numActions, 40);
  assert.equal(policy.obsDim, 25);
  assert.equal(policy.hidden, 64);
  assert.equal(policy.temperature, 1);
  assert.equal(policy.meta.actionSpace, 'v2');
});

test('v2 argmax 与 Python act_with_mask(greedy=True) 逐头相等（maskA）', () => {
  const policy = loadPolicy(V2_MODEL);
  const heads = policy.actWithMask(fixedVector(), maskA(), { greedy: true });
  // 对照值来自 Python：Policy.sample(v, rng, mask=maskA, greedy=True)
  assert.deepEqual(heads, [3, 2, 0, 0, 0, 0, 0, 0, 1, 2]);
  // slot 的裸 argmax 是 3，mask 把它关掉后必须落到允许集合里
  assert.equal(heads[SLOT_HEAD], 1);
});

test('v2 argmax 与 Python 逐头相等（maskB：attack 只开 1、slot 只开 0/3）', () => {
  const policy = loadPolicy(V2_MODEL);
  const heads = policy.actWithMask(fixedVector(), maskB(), { greedy: true });
  assert.deepEqual(heads, [3, 2, 0, 1, 0, 0, 0, 0, 3, 2]);
  assert.equal(heads[ATTACK_HEAD], 1);
  assert.equal(heads[SLOT_HEAD], 3);
});

test('v2 无 mask 时等于纯 argmax（与 Python 的 mask=None 一致）', () => {
  const policy = loadPolicy(V2_MODEL);
  assert.deepEqual(policy.actWithMask(fixedVector(), null, { greedy: true }),
    [3, 2, 0, 0, 0, 0, 0, 0, 3, 2]);
  // headProbs 的每头 argmax 必须与 actWithMask 的 greedy 结果一致（同一条前向）
  const probs = policy.headProbs(fixedVector());
  assert.deepEqual(probs.map((row) => row.indexOf(Math.max(...row))),
    [3, 2, 0, 0, 0, 0, 0, 0, 3, 2]);
  for (const row of probs) {
    assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  }
});

test('mask 全关：回退全开 + 只告警一次 + 计数递增', () => {
  const policy = loadPolicy(V2_MODEL);
  const mask = maskA();
  mask[SLOT_HEAD] = new Array(9).fill(false);
  const warnings = [];
  const original = console.warn;
  console.warn = (message) => warnings.push(String(message));
  try {
    const first = policy.actWithMask(fixedVector(), mask, { greedy: true });
    const second = policy.actWithMask(fixedVector(), mask, { greedy: true });
    assert.deepEqual(first, [3, 2, 0, 0, 0, 0, 0, 0, 3, 2], '全关的 slot 应回退到全开');
    assert.deepEqual(second, first);
  } finally {
    console.warn = original;
  }
  assert.equal(policy.maskFallbacks, 2);
  assert.equal(warnings.length, 1, '同一个头只提醒一次，别刷屏');
  assert.match(warnings[0], /mask 全关（slot）/);
});

test('mask 全关 + strictMask：抛明确错误（不静默回退）', () => {
  const policy = loadPolicy(V2_MODEL);
  const mask = maskA();
  mask[ATTACK_HEAD] = [false, false];
  assert.throws(
    () => policy.actWithMask(fixedVector(), mask, { greedy: true, strictMask: true }),
    /第 3 个头（attack）的 mask 全为 false/,
  );
  assert.equal(policy.maskFallbacks, 0, 'strictMask 不该走回退分支');
});

test('mask 只留一个值：每个头都必须落在那唯一允许的档位上', () => {
  const policy = loadPolicy(V2_MODEL);
  const mask = policy.headSizes.map((size) => {
    const row = new Array(size).fill(false);
    row[size - 1] = true;
    return row;
  });
  const heads = policy.actWithMask(fixedVector(), mask, { greedy: true });
  assert.deepEqual(heads, policy.headSizes.map((size) => size - 1));
});

test('matchingMask() 的 heads 与 array 两种写法等价（都是十头）', () => {
  const slots = new Array(46).fill(null);
  slots[36] = { name: 'diamond_sword', count: 1 };
  slots[37] = { name: 'golden_apple', count: 5 };
  const bot = { inventory: { slots }, heldItem: { name: 'diamond_sword', count: 1 } };
  const table = loadItemTable(DEFAULT_ITEMS_PATH);
  const mask = matchingMask(bot, table);
  assert.deepEqual(mask.array.map((row) => row.length), [5, 7, 5, 2, 3, 2, 2, 2, 9, 3]);
  const policy = loadPolicy(V2_MODEL);
  assert.deepEqual(
    policy.actWithMask(fixedVector(), mask, { greedy: true }),
    policy.actWithMask(fixedVector(), mask.array, { greedy: true }),
  );
});

test('epsilon=1 时按头随机，但只从允许值里抽（= v1 的探索口径）', () => {
  const policy = loadPolicy(V2_MODEL);
  const mask = maskA();
  let seed = 12345;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const seen = policy.headSizes.map(() => new Set());
  for (let tick = 0; tick < 60; tick += 1) {   // 20Hz × 3 秒
    const heads = policy.actWithMask(fixedVector(), mask, { greedy: true, epsilon: 1, rng });
    heads.forEach((value, index) => {
      assert.ok(value >= 0 && value < policy.headSizes[index], `第 ${index} 头越界: ${value}`);
      const allowed = [];
      mask[index].forEach((flag, slot) => {
        if (flag) {
          allowed.push(slot);
        }
      });
      assert.ok(allowed.includes(value), `第 ${index} 头选了被屏蔽的 ${value}`);
      seen[index].add(value);
    });
  }
  // 至少有若干个头真的在变（否则「探索」就是假的）
  const varying = seen.filter((values) => values.size > 1).length;
  assert.ok(varying >= 3, `只有 ${varying} 个头出现多个取值，探索没生效`);
  // slot 头被 mask 限死成 {1,2}，不该出现第三个值
  assert.deepEqual([...seen[SLOT_HEAD]].sort(), [1, 2]);
});

test('epsilon=0 时不进探索分支（rng 一次都不该被调用）', () => {
  const policy = loadPolicy(V2_MODEL);
  let calls = 0;
  const heads = policy.actWithMask(fixedVector(), maskA(), {
    greedy: true,
    epsilon: 0,
    rng: () => {
      calls += 1;
      return 0.5;
    },
  });
  assert.equal(calls, 0);
  assert.deepEqual(heads, [3, 2, 0, 0, 0, 0, 0, 0, 1, 2]);
});

test('v2 上调用单头接口必须抛错（防止悄悄只发一个头）', () => {
  const policy = loadPolicy(V2_MODEL);
  for (const name of ['greedy', 'sample', 'forward', 'logits']) {
    assert.throws(() => policy[name](fixedVector()), /v2 策略有 10 个头/, name);
  }
});

test('v1 回归：dagger-1118b 的 probs 与 Python 逐位一致（8 位小数）', () => {
  const policy = loadPolicy(V1_MODEL);
  assert.equal(policy.actionSpace, 'v1');
  assert.deepEqual(policy.headSizes, [11]);
  assert.equal(policy.obsDim, 23);
  // 期望值来自 Python：Policy.load('dagger-1118b/policy.npz').probs(vec[:23])
  const expected = [
    0.00009679, 0.99720976, 0.00008365, 0.00020551, 0.00007243, 0.00042681,
    0.00075284, 0.00036682, 0.00036598, 0.00034348, 0.00007593,
  ];
  const probs = policy.forward(fixedVector());
  assert.deepEqual(probs.map((value) => Number(value.toFixed(8))), expected);
  assert.equal(policy.greedy(fixedVector()), 1);
  // 25 维特征喂 23 维老模型：多出来的列忽略（与改造前同口径）
  assert.equal(policy.greedy(fixedVector(23)), policy.greedy(fixedVector(25)));
});

test('detectActionSpace：meta 缺字段时按权重形状推断', () => {
  assert.equal(detectActionSpace({}, new Map([['W2', {}]])), 'v1');
  assert.equal(detectActionSpace({}, new Map([['W2_move', {}]])), 'v2');
  assert.equal(detectActionSpace({ actionSpace: 'V2' }, new Map()), 'v2');
  assert.equal(detectActionSpace({ actionSpace: 'v1' }, new Map([['W2_move', {}]])), 'v1');
  assert.throws(() => detectActionSpace({ actionSpace: 'v3' }, new Map()), /不认识/);
});

test('20Hz 下的 intent 帧结构符合 intent.js 期望（link-server::onIntent 同形）', () => {
  const policy = loadPolicy(V2_MODEL);
  const mask = maskA();
  let seed = 999;
  const rng = () => {
    seed = (seed * 48271) % 2147483647;
    return seed / 2147483647;
  };
  for (let tick = 0; tick < 20; tick += 1) {   // --brain-hz 20 → 一秒 20 帧
    const heads = policy.actWithMask(fixedVector(), mask, { greedy: true, epsilon: 0.25, rng });
    const frame = { type: 'intent', payload: { action: heads } };
    assert.equal(frame.type, 'intent');
    assert.equal(frame.payload.action.length, 10);
    for (const value of frame.payload.action) {
      assert.ok(Number.isInteger(value), `intent 帧里出现非整数档位: ${value}`);
    }
    // 解码 → 归一：任何一帧都必须能直接被 applyIntent 接住
    const intent = decodeIntentVector(frame.payload.action);
    const normalized = normalizeIntent(intent);
    assert.equal(normalized.move, intent.move);
    assert.equal(normalized.slot, intent.slot);
    assert.equal(Boolean(normalized.attack), Boolean(intent.attack));
    assert.ok(HEAD_ORDER_V2.length === 10);
  }
});
