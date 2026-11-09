import assert from 'node:assert/strict';
import test from 'node:test';

import { WebSocketServer } from 'ws';

import { Backoff, LinkClient } from '../src/connection.js';

const logger = { info() {}, warn() {}, error() {} };

function startServer() {
  const wss = new WebSocketServer({ port: 0 });
  const state = {
    wss,
    hellos: [],
    sockets: [],
    messages: [],
    port: 0,
  };
  wss.on('connection', (socket) => {
    state.sockets.push(socket);
    socket.on('message', (raw) => {
      const envelope = JSON.parse(raw.toString());
      state.messages.push(envelope);
      if (envelope.type === 'HELLO') {
        state.hellos.push(envelope.payload);
        socket.send(JSON.stringify({
          v: 1,
          type: 'WELCOME',
          seq: 1,
          ts: Date.now(),
          botId: envelope.payload.botId,
          replyTo: envelope.seq,
          payload: {
            sessionId: `session-${state.hellos.length}`,
            playerUuid: '',
            serverVersion: 'test',
            features: [],
            limits: {},
          },
        }));
      } else if (envelope.type === 'PING') {
        socket.send(JSON.stringify({
          v: 1, type: 'PONG', seq: 2, ts: Date.now(), botId: envelope.botId, replyTo: envelope.seq, payload: {},
        }));
      }
    });
  });
  return new Promise((resolve) => {
    wss.on('listening', () => {
      state.port = wss.address().port;
      resolve(state);
    });
  });
}

function waitFor(predicate, timeoutMs = 3000) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      if (predicate()) {
        resolve();
        return;
      }
      if (Date.now() - started > timeoutMs) {
        reject(new Error('waitFor timeout'));
        return;
      }
      setTimeout(tick, 10);
    };
    tick();
  });
}

test('连接后发送 HELLO，收到 WELCOME，并回 PONG', async () => {
  const server = await startServer();
  const client = new LinkClient({
    url: `ws://127.0.0.1:${server.port}/gladiator`,
    token: 's3cr3t',
    botId: 'glad-bot-1',
    username: 'Glad_Bot_1',
    logger,
  });
  try {
    const welcome = new Promise((resolve) => client.once('welcome', resolve));
    client.start();
    const payload = await welcome;

    assert.equal(payload.sessionId, 'session-1');
    assert.equal(server.hellos.length, 1);
    assert.deepEqual(server.hellos[0], {
      token: 's3cr3t',
      botId: 'glad-bot-1',
      username: 'Glad_Bot_1',
      mcVersion: '1.8.8',
      runtimeVersion: '0.1.0',
    });

    client.send('PING', {});
    await waitFor(() => server.messages.some((m) => m.type === 'PING'));
    assert.ok(server.messages.find((m) => m.type === 'PING').botId === 'glad-bot-1');
  } finally {
    client.stop();
    server.wss.close();
  }
});

test('断线后按退避重连并重新 HELLO', async () => {
  const server = await startServer();
  const client = new LinkClient({
    url: `ws://127.0.0.1:${server.port}/gladiator`,
    botId: 'glad-bot-2',
    username: 'Glad_Bot_2',
    backoff: new Backoff({ initialMs: 20, maxMs: 100, jitter: 0 }),
    logger,
  });
  try {
    client.start();
    await waitFor(() => server.hellos.length === 1);

    server.sockets[0].terminate();

    await waitFor(() => server.hellos.length === 2);
    assert.equal(server.hellos[1].botId, 'glad-bot-2');
    assert.equal(client.backoff.attempt, 0, '重连成功后 backoff 必须复位');
  } finally {
    client.stop();
    server.wss.close();
  }
});

test('AUTH_FAILED 属于致命错误，停止重连', async () => {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise((resolve) => wss.on('listening', resolve));
  wss.on('connection', (socket) => {
    socket.on('message', () => {
      socket.send(JSON.stringify({
        v: 1,
        type: 'REJECT',
        seq: 1,
        ts: Date.now(),
        botId: '',
        replyTo: 1,
        payload: { code: 'AUTH_FAILED', message: 'token mismatch' },
      }));
    });
  });

  const client = new LinkClient({
    url: `ws://127.0.0.1:${wss.address().port}/gladiator`,
    botId: 'glad-bot-3',
    backoff: new Backoff({ initialMs: 10, jitter: 0 }),
    logger,
  });
  try {
    const rejected = new Promise((resolve) => client.once('reject', resolve));
    client.start();
    const payload = await rejected;
    assert.equal(payload.code, 'AUTH_FAILED');
    assert.equal(client.stopped, true);
    assert.equal(client.fatal, 'AUTH_FAILED');
  } finally {
    client.stop();
    wss.close();
  }
});
