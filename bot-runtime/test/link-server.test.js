import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { WebSocket } from 'ws';

import { ActionDispatcher } from '../src/actions.js';
import { EnvLinkServer } from '../src/link-server.js';
import { RestrictionCache } from '../src/restriction-cache.js';

const logger = { info() {}, warn() {}, error() {} };

// 收尾时兜底清理：ws 的 close 握手在测试进程里可能等不到对端，残留句柄会让 runner 不退出。
const servers = new Set();
const clients = new Set();

function startServer(options) {
  const server = new EnvLinkServer({ port: 0, logger, ...options }).start();
  servers.add(server);
  return server;
}

function connect(port) {
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  clients.add(socket);
  const inbox = [];
  socket.on('message', (raw) => inbox.push(JSON.parse(String(raw))));
  return new Promise((resolve, reject) => {
    socket.once('open', () => resolve({ socket, inbox }));
    socket.once('error', reject);
  });
}

function send(socket, type, payload, seq = 1) {
  socket.send(JSON.stringify({ v: 1, type, seq, ts: Date.now(), payload }));
}

async function waitFor(inbox, type, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const found = inbox.find((frame) => frame.type === type);
    if (found) {
      return found;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(`等待 ${type} 超时，已收到: ${inbox.map((f) => f.type).join(',')}`);
}

function fakeBot() {
  return {
    entity: { position: { x: 0, y: 64, z: 0 }, yaw: 0, pitch: 0, onGround: true },
    entities: {},
    health: 20,
    food: 20,
    foodSaturation: 5,
    heldItem: null,
    inventory: { slots: [] },
    blockAt: () => ({ name: 'air', type: 0 }),
    setControlState() {},
    chat() {},
    async look() {},
  };
}

after(() => {
  for (const socket of clients) {
    try {
      socket.terminate();
    } catch {
      // ignore
    }
  }
  clients.clear();
  for (const server of servers) {
    server.stop();
  }
  servers.clear();
});

test('HELLO 回 WELCOME，带 actionSpec 与 controlHz', async () => {
  const server = startServer({ controlHz: 5, resolveSession: () => null });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'HELLO', { v: 1, protocol: 'gladiator-env', botId: 'glad-a' }, 1);
  const welcome = await waitFor(inbox, 'WELCOME');
  assert.equal(welcome.replyTo, 1);
  assert.equal(welcome.payload.controlHz, 5);
  assert.deepEqual(welcome.payload.actionSpec.types,
    ['move', 'look', 'jump', 'attack', 'dig', 'place', 'use', 'talk', 'idle']);
  assert.ok(welcome.payload.observationSpec.fields.includes('blockWindow'));
  server.stop();
});

test('HELLO 之前发 ACTION 被 SESSION_UNKNOWN 拒绝', async () => {
  const server = startServer({ resolveSession: () => null });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'ACTION', { type: 'jump' }, 1);
  const reject = await waitFor(inbox, 'REJECT');
  assert.equal(reject.payload.code, 'SESSION_UNKNOWN');
  server.stop();
});

test('无 Bot 时 ACTION 回 BOT_NOT_READY；有 Bot 时执行并回 ok', async () => {
  let session = null;
  const server = startServer({ resolveSession: () => session });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  await waitFor(inbox, 'WELCOME');

  send(socket, 'ACTION', { type: 'jump' }, 2);
  const notReady = await waitFor(inbox, 'ACTION_RESULT');
  assert.equal(notReady.replyTo, 2);
  assert.equal(notReady.payload.code, 'BOT_NOT_READY');

  const cache = new RestrictionCache();
  const bot = fakeBot();
  session = {
    bot,
    dispatcher: new ActionDispatcher({ bot, restrictionCache: cache }),
    state: () => 'Alive',
    gameId: () => 'game-1',
  };
  send(socket, 'ACTION', { type: 'talk', params: { message: 'gg' } }, 3);
  const deadline = Date.now() + 2000;
  let ok;
  while (Date.now() < deadline) {
    ok = inbox.find((frame) => frame.type === 'ACTION_RESULT' && frame.replyTo === 3);
    if (ok) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal(ok.payload.ok, true);
  server.stop();
});

test('同一 seq 重复发送只执行一次（幂等）', async () => {
  let executed = 0;
  const bot = fakeBot();
  bot.chat = () => { executed += 1; };
  const session = {
    bot,
    dispatcher: new ActionDispatcher({ bot }),
    state: () => 'Alive',
    gameId: () => '',
  };
  const server = startServer({ resolveSession: () => session });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  await waitFor(inbox, 'WELCOME');
  send(socket, 'ACTION', { type: 'talk', params: { message: 'hi' } }, 5);
  send(socket, 'ACTION', { type: 'talk', params: { message: 'hi' } }, 5);
  const deadline = Date.now() + 2000;
  while (Date.now() < deadline && inbox.filter((f) => f.type === 'ACTION_RESULT').length < 2) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  const results = inbox.filter((f) => f.type === 'ACTION_RESULT' && f.replyTo === 5);
  assert.equal(results.length, 2);
  assert.equal(executed, 1);
  assert.equal(results.some((frame) => frame.payload.deduped === true), true);
  server.stop();
});

test('HELLO 后按 controlHz 推送 OBSERVATION', async () => {

  const bot = fakeBot();
  const session = {
    bot,
    dispatcher: new ActionDispatcher({ bot }),
    state: () => 'Alive',
    gameId: () => 'game-9',
    chatLines: () => [],
    notices: () => [],
  };
  const server = startServer({ controlHz: 20, resolveSession: () => session });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  const observation = await waitFor(inbox, 'OBSERVATION');
  assert.equal(observation.payload.gameId, 'game-9');
  assert.equal(observation.payload.state, 'Alive');
  assert.deepEqual(observation.payload.blockWindow.size, [7, 4, 7]);
  server.stop();
});

test('intent 帧交给 applyIntent（v1 宏动作仍走 dispatcher）', async () => {
  const seen = [];
  const session = {
    bot: fakeBot(),
    dispatcher: new ActionDispatcher({ bot: fakeBot() }),
    applyIntent: async (intent) => {
      seen.push(intent);
      return { ok: true, type: 'intent', applied: { move: intent.move } };
    },
    state: () => 'Alive',
    gameId: () => '',
  };
  const server = startServer({ resolveSession: () => session });
  await new Promise((resolve) => server.server.once('listening', resolve));
  const { socket, inbox } = await connect(server.address.port);
  send(socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  await waitFor(inbox, 'WELCOME');

  send(socket, 'ACTION', { type: 'intent', intent: { move: 'forward', attack: true } }, 2);
  const result = await waitFor(inbox, 'ACTION_RESULT');
  assert.equal(result.replyTo, 2);
  assert.equal(result.payload.ok, true);
  assert.equal(result.payload.type, 'intent');
  assert.deepEqual(seen, [{ move: 'forward', attack: true }]);

  // 缺 intent 对象 → 明确错误，不进 applyIntent
  socket.send(JSON.stringify({
    v: 1, type: 'ACTION', seq: 3, ts: Date.now(),
    payload: { type: 'intent', params: {} },
  }));
  const deadline = Date.now() + 2000;
  let failed;
  while (Date.now() < deadline) {
    failed = inbox.find((frame) => frame.type === 'ACTION_RESULT' && frame.replyTo === 3);
    if (failed) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal(failed.payload.code, 'INTENT_INVALID');
  assert.equal(seen.length, 1);
  server.stop();
});

test('新连接的 seq 从 1 重来：去重缓存不跨连接', async () => {
  let executed = 0;
  const bot = fakeBot();
  bot.chat = () => { executed += 1; };
  const session = {
    bot,
    dispatcher: new ActionDispatcher({ bot }),
    state: () => 'Alive',
    gameId: () => '',
  };
  const server = startServer({ resolveSession: () => session });
  await new Promise((resolve) => server.server.once('listening', resolve));

  const first = await connect(server.address.port);
  send(first.socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  await waitFor(first.inbox, 'WELCOME');
  send(first.socket, 'ACTION', { type: 'talk', params: { message: 'a' } }, 1);
  await waitFor(first.inbox, 'ACTION_RESULT');
  first.socket.terminate();

  // 第二个连接（= 训练侧重开一个进程）seq 又从 1 开始，必须真的执行
  const second = await connect(server.address.port);
  send(second.socket, 'HELLO', { v: 1, botId: 'glad-a' }, 1);
  await waitFor(second.inbox, 'WELCOME');
  send(second.socket, 'ACTION', { type: 'talk', params: { message: 'b' } }, 1);
  await waitFor(second.inbox, 'ACTION_RESULT');
  assert.equal(executed, 2);
  server.stop();
});

