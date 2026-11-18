/**
 * `features.js` 单测：23 维编码逐项对齐 `spaces.py`，以及动作解码。
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
  assert.equal(OBS_DIM, 23);
  assert.equal(OBS_FIELDS_JS.length, 23);
  assert.equal(OBS_FIELDS_JS[0], 'self_health');
  assert.equal(OBS_FIELDS_JS[18], 'self_gapples');
  assert.equal(OBS_FIELDS_JS[22], 'in_reach');
  assert.equal(ACTION_NAMES.length, 11);
  assert.equal(ACTION_NAMES[2], 'attack');
});

test('编码 23 维：逐项对齐 spaces.encode_observation', () => {
  const vector = encodeObservation(baseObservation(), {
    username: 'Glad_Bot_1',
    stepFraction: 0.5,
    reach: 3.99,
    sinceAttack: 2.0,
  });
  assert.equal(vector.length, 23);
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
