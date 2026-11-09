import test from 'node:test';
import assert from 'node:assert/strict';
import { Vec3 } from 'vec3';

import { OBSERVATION_FIELDS, buildObservation } from '../src/observation.js';

function fakeBot() {
  return {
    entity: { position: new Vec3(0.5, 64, 0.5), yaw: 1.25, pitch: -10, onGround: true },
    entities: {
      1: { id: 1, type: 'player', username: 'Foe', position: new Vec3(3, 64, 0) },
      2: { id: 2, type: 'mob', name: 'Zombie', position: new Vec3(1, 64, 1) },
    },
    health: 20,
    food: 18,
    foodSaturation: 5,
    heldItem: { name: 'iron_sword', count: 1, slot: 36, maxDurability: 250, durabilityUsed: 50 },
    inventory: {
      slots: [
        { name: 'bread', count: 3 }, null, { name: 'stone', count: 64 },
        null, null,
        { name: 'diamond_helmet', count: 1, maxDurability: 363, durabilityUsed: 0 },
        { name: 'diamond_chestplate', count: 1, maxDurability: 528, durabilityUsed: 264 },
        { name: 'diamond_leggings', count: 1, maxDurability: 495, durabilityUsed: 495 },
        null,
      ],
    },
    blockAt(position) {
      return { name: position.y < 64 ? 'stone' : 'air', type: position.y < 64 ? 1 : 0 };
    },
  };
}

test('观测字段与协议 §4.2 一致', () => {
  assert.deepEqual([...OBSERVATION_FIELDS], [
    'tick', 'gameId', 'state', 'self', 'entities', 'blockWindow', 'restrictions', 'chat', 'pendingNotices',
  ]);
});

test('self 携带坐标/朝向/生命/背包', () => {
  const observation = buildObservation(fakeBot(), { tick: 3, gameId: 'game-1', state: 'Alive' });
  assert.equal(observation.tick, 3);
  assert.equal(observation.gameId, 'game-1');
  assert.deepEqual(observation.self.pos, { x: 0.5, y: 64, z: 0.5 });
  assert.equal(observation.self.yaw, 1.25);
  assert.equal(observation.self.onGround, true);
  assert.equal(observation.self.health, 20);
  assert.deepEqual(observation.self.heldItem, {
    name: 'iron_sword', count: 1, slot: 36,
    durability: { used: 50, max: 250, remaining: 0.8 },
  });
  assert.equal(observation.self.armor.head.name, 'diamond_helmet');
  assert.equal(observation.self.armor.legs.durability.remaining, 0);
  assert.equal(observation.self.armor.feet, null);
  assert.ok(Math.abs(observation.self.armorRemaining - 0.5) < 1e-6);
  assert.deepEqual(observation.self.inventory.map((item) => item.name), [
    'bread', 'stone', 'diamond_helmet', 'diamond_chestplate', 'diamond_leggings',
  ]);
});

test('entities 排除自身并按距离升序', () => {
  const observation = buildObservation(fakeBot(), {});
  assert.equal(observation.entities.length, 2);
  assert.equal(observation.entities[0].id, 2);
  assert.equal(observation.entities[0].kind, 'mob');
  assert.equal(observation.entities[1].name, 'Foe');
  assert.ok(observation.entities[0].distance <= observation.entities[1].distance);
});

test('blockWindow 固定 7x4x7 且 palette/data 长度匹配', () => {
  const observation = buildObservation(fakeBot(), {});
  const window = observation.blockWindow;
  assert.deepEqual(window.size, [7, 4, 7]);
  assert.equal(window.data.length, 7 * 4 * 7);
  assert.ok(window.data.every((index) => index >= 0 && index < window.palette.length));
  assert.deepEqual(window.origin, { x: -3, y: 63, z: -3 });
});

test('restrictions / chat / pendingNotices 透传并截断', () => {
  const observation = buildObservation(fakeBot(), {
    restrictions: [{ regionId: 'spawn', allowPvp: false }],
    chat: Array.from({ length: 30 }, (_, i) => ({ message: `m${i}` })),
    pendingNotices: Array.from({ length: 15 }, (_, i) => ({ kind: 'EVENT', at: i })),
  });
  assert.deepEqual(observation.restrictions, [{ regionId: 'spawn', allowPvp: false }]);
  assert.equal(observation.chat.length, 20);
  assert.equal(observation.chat[0].message, 'm10');
  assert.equal(observation.pendingNotices.length, 10);
});
