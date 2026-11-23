/**
 * G-711 运行时生产端：`BrainCrossFrame` → 观测 25–36 维。
 *
 * 这一组测的是**训练/推理输入一致性**：`features.js` 早就会读 `crossFrame`，
 * 但运行时没人产它，37 维模型的后 12 维就恒为 0（= 「不知道上一帧干了什么」）。
 * 这里用真实的 `encodeObservation` 断言「状态 → 向量」这条链路真的通。
 *
 * 口径提醒：计数单位是**大脑决策帧**（`--brain-hz`，默认 20Hz），不是 mineflayer tick。
 * 编码发生在决策之前，所以第 t 帧拿到的是「第 t−1 帧出手了没有」——与训练侧
 * 「第 t 步的观测记录第 t−1 步动作」等价（见 `src/brain-cross-frame.js` 头注释）。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { BrainCrossFrame, FRAME_CAP } from '../src/brain-cross-frame.js';
import { OBS_DIM, encodeObservation } from '../src/features.js';

/** 最小观测帧：只有 self（血量/坐标）与空实体表。 */
function baseObservation(health = 20) {
  return {
    tick: 1,
    gameId: 'g',
    state: 'Alive',
    self: {
      pos: { x: 0, y: 4, z: 0 },
      yaw: 0,
      pitch: 0,
      onGround: true,
      velocity: { x: 0, y: 0, z: 0 },
      health,
      food: 20,
      saturation: 5,
      heldItem: null,
      armor: {},
      armorRemaining: 0,
      inventory: [],
    },
    entities: [],
    blockWindow: { origin: { x: 0, y: 0, z: 0 }, size: [7, 4, 7], palette: [], data: [] },
    restrictions: [],
    chat: [],
    pendingNotices: [],
  };
}

/** 大脑一帧：先读血量，再编码（= index.js 里 `observeHealth` → `encodeObservation`）。 */
function encodeTick(state, { health = 20, observation = null } = {}) {
  state.observeHealth(health);
  return encodeObservation(observation || baseObservation(health), {
    username: 'Bot_A',
    ...state.options(),
  });
}

function headSlices(vector) {
  return Array.from(vector.slice(25, 35));
}

test('首帧：prevAction 为 null → 25–34 全 0，两个帧计数饱和 1.0', () => {
  const state = new BrainCrossFrame();
  assert.equal(state.prevAction, null);
  const vector = encodeTick(state);
  assert.equal(vector.length, OBS_DIM);
  assert.deepEqual(headSlices(vector), new Array(10).fill(0),
    '没有上一帧动作时必须全 0（不能假装「上一帧选了 0 档」）');
  assert.equal(vector[35], 1, 'frames_since_attack 应从饱和值开始（很久没出手）');
  assert.equal(vector[36], 1);
  assert.equal(FRAME_CAP, 50);
});

test('决策一次后：25–34 = 上一帧十头 / headSize', () => {
  const state = new BrainCrossFrame();
  encodeTick(state);                       // 第 0 帧（没有上一帧）
  state.commit({ prevAction: [1, 5, 2, 1, 0, 0, 0, 0, 8, 2], attacked: true });
  const vector = encodeTick(state);        // 第 1 帧
  const expected = [1 / 5, 5 / 7, 2 / 5, 1 / 2, 0, 0, 0, 0, 8 / 9, 2 / 3];
  headSlices(vector).forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 1e-12,
      `第 ${25 + index} 维 ${value} != ${expected[index]}`);
  });
});

test('attack 帧把 35 置 0，之后逐帧增长', () => {
  const state = new BrainCrossFrame();
  state.commit({ prevAction: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], attacked: true });
  const afterAttack = encodeTick(state);
  assert.equal(afterAttack[35], 0, '上一帧出手了 → 0');

  state.advance();                         // 一帧没出手
  const next = encodeTick(state);
  assert.ok(Math.abs(next[35] - 1 / FRAME_CAP) < 1e-12, `下一帧应为 1/${FRAME_CAP}`);
  state.advance();
  const third = encodeTick(state);
  assert.ok(Math.abs(third[35] - 2 / FRAME_CAP) < 1e-12);
});

test('长时间不出手：35 涨到封顶后停在 1.0', () => {
  const state = new BrainCrossFrame();
  state.commit({ attacked: true });
  let vector = encodeTick(state);
  assert.equal(vector[35], 0);
  for (let frame = 0; frame < FRAME_CAP + 10; frame += 1) {
    vector = encodeTick(state);
    state.advance();
  }
  assert.equal(vector[35], 1, '超过 cap 之后必须饱和在 1.0，不能溢出');
});

test('受伤帧把 36 置 0（血量下降即判定，与训练侧同一判据）', () => {
  const state = new BrainCrossFrame();
  encodeTick(state, { health: 20 });       // 建立基线
  const hurt = encodeTick(state, { health: 16.5 });
  assert.equal(hurt[36], 0, '本帧掉血 → frames_since_hit=0');

  const calm = encodeTick(state, { health: 16.5 });
  assert.ok(Math.abs(calm[36] - 1 / FRAME_CAP) < 1e-12, '没掉血的下一帧开始增长');
});

test('血量回升（重生 0 → 20）不算受伤', () => {
  const state = new BrainCrossFrame();
  encodeTick(state, { health: 20 });       // 基线
  const hurt = encodeTick(state, { health: 0 });
  assert.equal(hurt[36], 0, '掉到 0 是受伤');
  const respawn = encodeTick(state, { health: 20 });
  assert.ok(Math.abs(respawn[36] - 1 / FRAME_CAP) < 1e-12,
    '回升不是受伤：计数应继续增长而不是再次清零');
});

test('读不到血量时不推进受伤计数（不能把「不知道」当受伤或安全）', () => {
  const state = new BrainCrossFrame();
  encodeTick(state, { health: 20 });
  const before = state.framesSinceHit;
  state.observeHealth(undefined);
  state.observeHealth(Number.NaN);
  assert.equal(state.framesSinceHit, before, '非法血量必须原样不动计数');
  assert.equal(state.lastHealth, 20, '也不该污染基线');
});

test('v1 路径：25–34 恒 0，但两个帧计数照常工作', () => {
  const state = new BrainCrossFrame();
  state.commit({ prevAction: null, attacked: false });
  let vector = encodeTick(state);
  assert.deepEqual(headSlices(vector), new Array(10).fill(0), 'v1 没有十头标签');

  state.commit({ prevAction: null, attacked: true });
  vector = encodeTick(state);
  assert.equal(vector[35], 0, 'v1 出手同样清零 35');
  assert.deepEqual(headSlices(vector), new Array(10).fill(0));
});

test('reset() 复位：prevAction 清空、两个计数回到饱和', () => {
  const state = new BrainCrossFrame();
  state.commit({ prevAction: [4, 4, 4, 1, 1, 1, 1, 1, 4, 1], attacked: true });
  encodeTick(state, { health: 12 });
  state.reset();
  assert.equal(state.prevAction, null);
  assert.equal(state.framesSinceAttack, FRAME_CAP);
  assert.equal(state.framesSinceHit, FRAME_CAP);
  const vector = encodeTick(state);
  assert.deepEqual(headSlices(vector), new Array(10).fill(0));
  assert.equal(vector[35], 1);
});

test('cap 可注入（测试用），且不影响默认值', () => {
  const state = new BrainCrossFrame({ cap: 4 });
  state.commit({ attacked: true });
  assert.equal(state.framesSinceAttack, 0);
  state.advance();
  state.advance();
  state.advance();
  assert.equal(state.framesSinceAttack, 3);
  state.advance();
  assert.equal(state.framesSinceAttack, 4, '自定义 cap 生效');
  state.advance();
  assert.equal(state.framesSinceAttack, 4, '不会超过 cap');
  assert.equal(new BrainCrossFrame().framesSinceAttack, 50, '默认仍是 50');
});

test('编码器优先用显式传参（不被 observation.crossFrame 覆盖）', () => {
  const state = new BrainCrossFrame();
  const observation = baseObservation();
  observation.crossFrame = { prevAction: [4, 0, 0, 0, 0, 0, 0, 0, 8, 0], framesSinceAttack: 3 };
  const vector = encodeTick(state, { observation });
  assert.deepEqual(headSlices(vector), new Array(10).fill(0),
    'state 里的 null 必须压过帧里的旧值（否则两处状态会打架）');
  assert.equal(vector[35], 1, 'state 的饱和值优先于帧里的 3/50');
});
