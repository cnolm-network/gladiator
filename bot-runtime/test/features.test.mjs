/**
 * `features.js` 单测：37 维编码逐项对齐 `spaces.py`，以及动作解码。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ACTION_NAMES,
  MOVE_MS,
  OBS_DIM,
  OBS_FIELDS_JS,
  decodeAction,
  encodeObservation,
  entityHealth,
  nearestPlayer,
} from '../src/features.js';

function baseObservation(overrides = {}) {
  const self = {
    pos: { x: 1.5, y: 4.0, z: -2.5 },
    yaw: Math.PI / 2,
    pitch: -Math.PI / 4,
    onGround: true,
    health: 14.0,
    food: 18,
    saturation: 2.5,
    heldItem: { name: 'diamond_sword', count: 1, slot: 0, durability: { used: 25, max: 1561, remaining: 0.984 } },
    armorRemaining: 0.75,
    inventory: [
      { slot: 0, name: 'diamond_sword', count: 1 },
      { slot: 1, name: 'golden_apple', count: 3 },
      { slot: 2, name: 'golden_apple', count: 4 },
    ],
  };
  const entities = [
    {
      id: 42,
      kind: 'player',
      name: 'Steve',
      pos: { x: 4.5, y: 4.0, z: 1.5 },
      velocity: { x: 0.3, y: -0.2, z: 0.4 },
      distance: 5.0,
      onGround: false,
      health: 11.0,
    },
    {
      id: 7,
      kind: 'player',
      name: 'Glad_Bot_9',
      pos: { x: 2.0, y: 4.0, z: -2.0 },
      velocity: { x: 0, y: 0, z: 0 },
      distance: 0.7,
      onGround: true,
    },
    { id: 99, kind: 'zombie', name: 'Zombie', pos: { x: 1, y: 4, z: 1 }, distance: 1.0 },
  ];
  return {
    tick: 7,
    gameId: 'game-1',
    state: 'Alive',
    self,
    entities,
    blockWindow: null,
    restrictions: [],
    chat: [],
    pendingNotices: [],
    ...overrides,
  };
}

test('观测字段与 spaces.py 同序同长度', () => {
  assert.equal(OBS_DIM, 39);
  assert.equal(OBS_FIELDS_JS.length, 39);
  assert.equal(OBS_FIELDS_JS[0], 'self_health');
  assert.equal(OBS_FIELDS_JS[18], 'self_gapples');
  assert.equal(OBS_FIELDS_JS[22], 'in_reach');
  // G-702：只在末尾追加，0–22 一个都不许动
  assert.equal(OBS_FIELDS_JS[23], 'self_vertical_speed');
  assert.equal(OBS_FIELDS_JS[24], 'self_attack_cooldown');
  // G-711：25–34 是上一帧十头动作（顺序 = ACTION_HEADS_V2），35/36 是两个帧计数
  assert.deepEqual(OBS_FIELDS_JS.slice(25, 35), [
    'prev_move', 'prev_look_yaw', 'prev_look_pitch', 'prev_attack', 'prev_use',
    'prev_jump', 'prev_sneak', 'prev_sprint', 'prev_slot', 'prev_sprint_reset',
  ]);
  assert.equal(OBS_FIELDS_JS[35], 'frames_since_attack');
  assert.equal(OBS_FIELDS_JS[36], 'frames_since_hit');
  // G-731：37/38 = 朝向误差的 sin/cos（37 之前一位不动）
  assert.equal(OBS_FIELDS_JS[37], 'target_bearing_sin');
  assert.equal(OBS_FIELDS_JS[38], 'target_bearing_cos');
  assert.equal(ACTION_NAMES.length, 11);
  assert.equal(ACTION_NAMES[2], 'attack');
});

test('编码 39 维：逐项对齐 spaces.encode_observation', () => {
  const vector = encodeObservation(baseObservation(), {
    username: 'Glad_Bot_1',
    stepFraction: 0.5,
    reach: 3.99,
    sinceAttack: 2.0,
    attackCooldownRemaining: 0.25,
  });
  assert.equal(vector.length, 39);
  const near = (index, expected) => assert.ok(
    Math.abs(vector[index] - expected) < 1e-9,
    `第 ${index} 维 ${vector[index]} != ${expected}`,
  );

  near(0, 14.0 / 20.0);                       // self_health
  near(1, 18 / 20.0);                         // self_food
  near(2, 2.5 / 5.0);                         // self_saturation
  near(3, 1.0);                               // onGround
  near(4, Math.sin(Math.PI / 2));             // yaw sin
  near(5, Math.cos(Math.PI / 2));             // yaw cos
  near(6, -0.5);                              // pitch / (pi/2)
  near(7, (2.0 - 1.5) / 16.0);                // dx（最近的是 Glad_Bot_9）
  near(13, 1.0);                              // has_target
  near(14, 0.5);                              // step_fraction
  near(15, 1.0);                              // state_alive
  near(16, 0.984);                            // 手持耐久
  near(17, 0.75);                             // armorRemaining
  near(18, 1.0);                              // (3+4)/5 = 1.4 → clip 1
  near(21, 2.0 / 5.0);                        // since_attack
  near(23, 0.0);                              // self_vertical_speed（本帧 self 无 velocity → 0）
  near(24, 0.25 / 0.5);                       // self_attack_cooldown
});

test('G-702：自身 vy 优先取 options，缺省回落 self.velocity.y 并裁剪', () => {
  const observation = baseObservation();
  observation.self.velocity = { x: 0.1, y: -1.2, z: 0.3 };

  const fromPayload = encodeObservation(observation, { username: 'Glad_Bot_1' });
  assert.ok(Math.abs(fromPayload[23] - (-1.2 / 5.0)) < 1e-12);

  const fromOption = encodeObservation(observation, { username: 'Glad_Bot_1', selfVelocityY: 3.0 });
  assert.ok(Math.abs(fromOption[23] - 0.6) < 1e-12);

  // clip：自由下落速度会超过 5，归一化后最多 1
  const clipped = encodeObservation(observation, { username: 'Glad_Bot_1', selfVelocityY: -40 });
  assert.equal(clipped[23], -1.0);

  // 冷却裁剪：负值→0（可打），超 0.5→1（刚打完）
  assert.equal(encodeObservation(observation, { username: 'Glad_Bot_1', attackCooldownRemaining: -1 })[24], 0.0);
  assert.equal(encodeObservation(observation, { username: 'Glad_Bot_1', attackCooldownRemaining: 9 })[24], 1.0);
  // 不传 = 「可以打」
  assert.equal(encodeObservation(observation, { username: 'Glad_Bot_1' })[24], 0.0);
});

test('最近的非自己玩家胜出，Bot 也在候选里（由 allowed 决定）', () => {
  const observation = baseObservation();
  const nearest = nearestPlayer(observation, 'Glad_Bot_1');
  assert.equal(nearest.name, 'Glad_Bot_9');
  const onlyHuman = nearestPlayer(observation, 'Glad_Bot_1', ['Steve']);
  assert.equal(onlyHuman.name, 'Steve');
  // 自己永远被排除
  assert.equal(nearestPlayer(observation, 'Glad_Bot_9').name, 'Steve');
});

test('目标字段：相对位置、距离、血量、速度与 in_reach', () => {
  const observation = baseObservation();
  const vector = encodeObservation(observation, {
    username: 'Glad_Bot_1',
    allowed: ['Steve'],
    reach: 3.99,
  });
  const near = (index, expected) => assert.ok(
    Math.abs(vector[index] - expected) < 1e-9,
    `第 ${index} 维 ${vector[index]} != ${expected}`,
  );
  near(7, (4.5 - 1.5) / 16.0);
  near(8, (4.0 - 4.0) / 8.0);
  near(9, (1.5 + 2.5) / 16.0);
  near(10, 5.0 / 32.0);
  near(11, 11.0 / 20.0);
  near(12, 0.0);                              // onGround=false
  near(19, Math.sqrt(0.3 ** 2 + 0.4 ** 2) / 10.0);
  near(20, -0.2 / 5.0);
  near(22, 0.0);                              // 水平距离 5 > 3.99
});

test('无目标时 has_target/in_reach 为 0，其余目标字段留 0', () => {
  const observation = baseObservation({ entities: [] });
  const vector = encodeObservation(observation, { username: 'Glad_Bot_1' });
  assert.equal(vector[13], 0);
  assert.equal(vector[22], 0);
  for (const index of [7, 8, 9, 10, 11, 12, 19, 20]) {
    assert.equal(vector[index], 0, `第 ${index} 维应为 0`);
  }
});

test('整向量按 OBS_CLIP=5 裁剪', () => {
  const observation = baseObservation({ entities: [] });
  const far = {
    id: 5,
    kind: 'player',
    name: 'Steve',
    pos: { x: 500, y: 500, z: 500 },
    velocity: { x: 100, y: 100, z: 100 },
    distance: 1000,
    onGround: false,
  };
  observation.entities = [far];
  const vector = encodeObservation(observation, { username: 'Glad_Bot_1' });
  assert.equal(vector[10], 5);                       // 1000/32 = 31.25 → 5
  assert.equal(vector[19], 1);                       // 速度先 clip 0..10 再 /10
  assert.equal(vector[20], 1);                       // vy clip 到 5 → 1
  for (const value of vector) {
    assert.ok(value <= 5 && value >= -5);
  }
});

test('opponentHealth 优先于实体血量；读不到时回落实体字段', () => {
  const observation = baseObservation();
  const withOverride = encodeObservation(observation, {
    username: 'Glad_Bot_1', allowed: ['Steve'], opponentHealth: 4.0,
  });
  assert.ok(Math.abs(withOverride[11] - 0.2) < 1e-9);
  const fallback = encodeObservation(observation, { username: 'Glad_Bot_1', allowed: ['Steve'] });
  assert.ok(Math.abs(fallback[11] - 11 / 20) < 1e-9);
  const noHealth = baseObservation();
  delete noHealth.entities[0].health;
  const defaulted = encodeObservation(noHealth, { username: 'Glad_Bot_1', allowed: ['Steve'] });
  assert.ok(Math.abs(defaulted[11] - 1.0) < 1e-9);   // 默认 20 → 1.0
});

test('entityHealth 支持直连字段与 metadata（数组/对象）', () => {
  assert.equal(entityHealth({ health: 7.5 }), 7.5);
  assert.equal(entityHealth({ metadata: [{ key: 6, value: 3.5 }] }), 3.5);
  assert.equal(entityHealth({ metadata: { 6: 2.5 } }), 2.5);
  assert.equal(entityHealth({ metadata: { health: 9 } }), 9);
  assert.equal(entityHealth({}), null);
  assert.equal(entityHealth(null), null);
});

test('玩家实体的 health=0 视为「未知」而不是「已死亡」', () => {
  // 1.8 不下发其他玩家血量，mineflayer 给 player 实体的默认 health 是 0。
  assert.equal(entityHealth({ kind: 'player', health: 0 }), null);
  assert.equal(entityHealth({ kind: 'player', metadata: [{ key: 6, value: 0 }] }), null);
  assert.equal(entityHealth({ kind: 'player', health: 0, metadata: { 6: 0 } }), null);
  // 玩家血量正常时报真实值。
  assert.equal(entityHealth({ kind: 'player', health: 12.5 }), 12.5);
  // 非玩家实体（僵尸等）0 就是真的死了，保留原值。
  assert.equal(entityHealth({ kind: 'zombie', health: 0 }), 0);
});

test('decodeAction：attack 在射程内先转向再挥击', () => {
  const observation = baseObservation();
  const actions = decodeAction(ACTION_NAMES.indexOf('attack'), observation, 'Glad_Bot_1', {
    allowed: ['Steve'], reach: 6.0,
  });
  assert.deepEqual(actions.map((action) => action.type), ['look', 'attack', 'look']);
  assert.equal(actions[1].params.targetEntityId, 42);
});

test('decodeAction：超出射程退化为前冲，不空挥', () => {
  const observation = baseObservation();
  const actions = decodeAction(ACTION_NAMES.indexOf('attack'), observation, 'Glad_Bot_1', {
    allowed: ['Steve'], reach: 3.99,
  });
  assert.deepEqual(actions.map((action) => action.type), ['look', 'move']);
  assert.equal(actions[1].params.direction, 'forward');
  assert.equal(actions[1].params.sprint, true);
  assert.equal(actions[1].params.durationMs, MOVE_MS);
});

test('decodeAction：无目标待机，满血不浪费金苹果，越界报错', () => {
  const observation = baseObservation({ entities: [] });
  assert.deepEqual(decodeAction(0, observation, 'Glad_Bot_1'), [{ type: 'idle', params: { durationMs: MOVE_MS } }]);
  assert.equal(decodeAction(ACTION_NAMES.indexOf('approach'), observation, 'Glad_Bot_1')[0].type, 'idle');

  const fullHealth = baseObservation();
  fullHealth.self.health = 20;
  assert.equal(
    decodeAction(ACTION_NAMES.indexOf('eat_gapple'), fullHealth, 'Glad_Bot_1')[0].type,
    'idle',
  );

  const hurt = baseObservation();
  hurt.self.health = 8;
  const eat = decodeAction(ACTION_NAMES.indexOf('eat_gapple'), hurt, 'Glad_Bot_1');
  assert.deepEqual(eat[0].params, { item: 'gapple', holdMs: 1600 });

  assert.throws(() => decodeAction(11, observation, 'Glad_Bot_1'), /动作编号越界/);
});

// --- G-711：跨帧 12 维（25..36） -------------------------------------------

test('G-711 缺省：prev_action 全 0（不知道），帧计数置饱和 1.0', () => {
  const vector = encodeObservation(baseObservation(), { username: 'Glad_Bot_1' });
  for (let index = 25; index <= 34; index += 1) {
    assert.equal(vector[index], 0, `第 ${index} 维应为 0（没有上一帧动作）`);
  }
  assert.equal(vector[35], 1);
  assert.equal(vector[36], 1);
});

test('G-711 prevAction 按 head_size 归一化，顺序 = ACTION_HEADS_V2', () => {
  const vector = encodeObservation(baseObservation(), {
    username: 'Glad_Bot_1',
    // move=1/5, look_yaw=5/7, look_pitch=2/5, attack=1/2, use=0, jump=0,
    // sneak=0, sprint=0, slot=0, sprint_reset=2/3
    prevAction: [1, 5, 2, 1, 0, 0, 0, 0, 0, 2],
  });
  const expected = [1 / 5, 5 / 7, 2 / 5, 1 / 2, 0, 0, 0, 0, 0, 2 / 3];
  expected.forEach((value, offset) => {
    assert.ok(Math.abs(vector[25 + offset] - value) < 1e-12,
      `第 ${25 + offset} 维 ${vector[25 + offset]} != ${value}`);
  });
});

test('G-711 prevAction 可从 observation.crossFrame 缺省读取，显式参数优先', () => {
  const observation = baseObservation({
    crossFrame: { prevAction: [4, 0, 0, 0, 0, 0, 0, 0, 8, 0], framesSinceAttack: 10, framesSinceHit: 20 },
  });
  const fromFrame = encodeObservation(observation, { username: 'Glad_Bot_1' });
  assert.ok(Math.abs(fromFrame[25] - 4 / 5) < 1e-12);
  assert.ok(Math.abs(fromFrame[33] - 8 / 9) < 1e-12);
  assert.ok(Math.abs(fromFrame[35] - 10 / 50) < 1e-12);
  assert.ok(Math.abs(fromFrame[36] - 20 / 50) < 1e-12);

  const explicit = encodeObservation(observation, {
    username: 'Glad_Bot_1', prevAction: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], framesSinceAttack: 0,
  });
  assert.ok(Math.abs(explicit[25] - 1 / 5) < 1e-12, '显式 prevAction 应覆盖帧里的值');
  assert.equal(explicit[35], 0, '显式 framesSinceAttack 应覆盖帧里的值');
});

test('G-711 帧计数按 50 归一化并裁剪到 0..1', () => {
  const cases = [[0, 0], [25, 0.5], [50, 1], [500, 1], [-3, 0]];
  for (const [frames, expected] of cases) {
    const vector = encodeObservation(baseObservation(), {
      username: 'Glad_Bot_1', framesSinceAttack: frames, framesSinceHit: frames,
    });
    assert.ok(Math.abs(vector[35] - expected) < 1e-12, `${frames} 帧 → ${vector[35]} != ${expected}`);
    assert.ok(Math.abs(vector[36] - expected) < 1e-12);
  }
});

test('G-711 越界的 prevAction 被裁到 0（不让坏索引把观测推到负区间）', () => {
  const vector = encodeObservation(baseObservation(), {
    username: 'Glad_Bot_1', prevAction: [-1, 99, 0, 0, 0, 0, 0, 0, 0, 0],
  });
  assert.equal(vector[25], 0);
  assert.equal(vector[26], 1);
});
