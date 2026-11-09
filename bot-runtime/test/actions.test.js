import test from 'node:test';
import assert from 'node:assert/strict';

import { ActionDispatcher, ACTION_TYPES } from '../src/actions.js';
import { RestrictionCache } from '../src/restriction-cache.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fakeBot() {
  const controls = new Map();
  const calls = [];
  return {
    controls,
    calls,
    entity: { position: { x: 0, y: 64, z: 0 } },
    entities: { 42: { id: 42, type: 'player', username: 'Foe', position: { x: 1, y: 64, z: 0 } } },
    setControlState(name, value) {
      controls.set(name, value);
      calls.push(['control', name, value]);
    },
    async look(yaw, pitch, force) {
      calls.push(['look', yaw, pitch, force]);
    },
    attack(entity) {
      calls.push(['attack', entity.id]);
    },
    blockAt(position) {
      calls.push(['blockAt', position.x, position.y, position.z]);
      if (position.y === 63) {
        return { name: 'air', type: 0, boundingBox: 'empty' };
      }
      return { name: 'stone', type: 1, boundingBox: 'block' };
    },
    async dig(block) {
      calls.push(['dig', block.name]);
    },
    async placeBlock(reference, offset) {
      calls.push(['placeBlock', reference.name, offset.x, offset.y, offset.z]);
    },
    activateItem(offhand) {
      calls.push(['activateItem', offhand]);
    },
    chat(message) {
      calls.push(['chat', message]);
    },
  };
}

test('覆盖协议全部动作类型', () => {
  assert.deepEqual([...ACTION_TYPES], ['move', 'look', 'jump', 'attack', 'dig', 'place', 'use', 'talk', 'idle']);
});

test('未知动作类型回 ACTION_UNSUPPORTED', async () => {
  const dispatcher = new ActionDispatcher({ bot: fakeBot() });
  await assert.rejects(() => dispatcher.dispatch({ type: 'explode' }), /未知动作类型/);
});

test('move 在宽限期后松开控制状态', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'forward', durationMs: 1 } });
  // 连续 move 之间故意保持按住，避免每步松键导致一卡一卡；宽限期过后才释放。
  assert.equal(bot.controls.get('forward'), true);
  await sleep(200);
  assert.equal(bot.controls.get('forward'), false);
  assert.equal(dispatcher.hold, null);
});

test('连续同方向 move 不松键（运动连续）', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'forward', durationMs: 30 } });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'forward', durationMs: 30 } });
  const offCount = bot.calls.filter((c) => c[0] === 'control' && c[2] === false).length;
  assert.equal(offCount, 0);
  assert.equal(bot.controls.get('forward'), true);
});

test('move 带 sprint 时同时按下并释放疾跑', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'forward', durationMs: 1, sprint: true } });
  assert.equal(bot.controls.get('sprint'), true);
  await sleep(200);
  assert.equal(bot.controls.get('sprint'), false);
  const controls = bot.calls.filter((c) => c[0] === 'control');
  assert.equal(controls.length, 4); // forward on/off + sprint on/off
});

test('切换方向会立刻松开上一个方向', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'forward', durationMs: 20 } });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'left', durationMs: 20 } });
  assert.equal(bot.controls.get('forward'), false);
  assert.equal(bot.controls.get('left'), true);
});

test('后退不会误触疾跑', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'move', params: { direction: 'back', durationMs: 1, sprint: true } });
  assert.equal(bot.controls.get('sprint'), undefined);
});

test('move 拒绝未知方向', async () => {
  const dispatcher = new ActionDispatcher({ bot: fakeBot() });
  await assert.rejects(() => dispatcher.dispatch({ type: 'move', params: { direction: 'up' } }), /未知方向/);
});

test('look 夹紧 pitch 并强制刷新', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({ type: 'look', params: { yaw: 1.5, pitch: 180 } });
  const call = bot.calls.find((c) => c[0] === 'look');
  assert.deepEqual(call, ['look', 1.5, 90, true]);
});

test('attack 找不到实体时报错，找到实体时调用 mineflayer', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 7 } }),
    /实体不存在/,
  );
  await dispatcher.dispatch({ type: 'attack', params: { targetEntityId: 42 } });
  assert.deepEqual(bot.calls.find((c) => c[0] === 'attack'), ['attack', 42]);
});

test('dig 空方块报错；命中缓存 deny 时回 RULE_DENIED 且计数', async () => {
  const cache = new RestrictionCache({ now: () => 1000 });
  cache.put({ regionId: 'spawn', box: { minX: 0, minY: 60, minZ: 0, maxX: 5, maxY: 70, maxZ: 5 } });
  const dispatcher = new ActionDispatcher({ bot: fakeBot(), restrictionCache: cache });
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'dig', params: { position: { x: 0, y: 63, z: 0 } } }),
    /禁止该动作/,
  );
  assert.equal(dispatcher.ruleDenials, 1);
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'dig', params: { position: { x: 100, y: 63, z: 100 } } }),
    /目标方块为空/,
  );
});

test('place 使用 against+face 计算目标格并检查规则', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await dispatcher.dispatch({
    type: 'place',
    params: { against: { x: 10, y: 64, z: 10 }, face: 'up' },
  });
  const call = bot.calls.find((c) => c[0] === 'placeBlock');
  assert.deepEqual(call, ['placeBlock', 'stone', 0, 1, 0]);
  await assert.rejects(
    () => dispatcher.dispatch({ type: 'place', params: { against: { x: 10, y: 64, z: 10 }, face: 'sideways' } }),
    /未知朝向/,
  );
});

test('talk 校验长度并调用 chat；use/idle 走对应 API', async () => {
  const bot = fakeBot();
  const dispatcher = new ActionDispatcher({ bot });
  await assert.rejects(() => dispatcher.dispatch({ type: 'talk', params: { message: '' } }), /1\.\.256/);
  await dispatcher.dispatch({ type: 'talk', params: { message: 'hi' } });
  assert.deepEqual(bot.calls.find((c) => c[0] === 'chat'), ['chat', 'hi']);
  await dispatcher.dispatch({ type: 'use', params: { offhand: true } });
  assert.deepEqual(bot.calls.find((c) => c[0] === 'activateItem'), ['activateItem', true]);
  await dispatcher.dispatch({ type: 'idle', params: { durationMs: 1 } });
  assert.equal(bot.controls.get('forward'), false);
});

test('Bot 未登录时回 BOT_NOT_READY', async () => {
  const dispatcher = new ActionDispatcher({ bot: null });
  await assert.rejects(() => dispatcher.dispatch({ type: 'jump' }), /尚未登录/);
});
