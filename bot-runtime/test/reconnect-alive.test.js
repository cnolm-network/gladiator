import assert from 'node:assert/strict';
import test from 'node:test';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

import { LinkClient } from '../src/connection.js';

const CONNECTION_URL = new URL('../src/connection.js', import.meta.url).href;

test('服务端不可达时进程保持存活并持续重连（G-106 回归）', async () => {
  const script = `
    import { Backoff, LinkClient } from ${JSON.stringify(CONNECTION_URL)};
    const client = new LinkClient({
      url: 'ws://127.0.0.1:1/gladiator',
      botId: 'glad-offline',
      backoff: new Backoff({ initialMs: 20, maxMs: 40, jitter: 0 }),
      logger: { info() {}, warn() {}, error() {} },
    });
    client.start();
  `;
  const child = spawn(process.execPath, ['--input-type=module', '-e', script], {
    stdio: 'ignore',
  });
  try {
    await delay(600);
    assert.equal(child.exitCode, null, '重连等待期间进程不得静默退出');
  } finally {
    child.kill();
  }
});

test('stop() 之后清掉重连计时器，进程可以自然退出', () => {
  const client = new LinkClient({
    url: 'ws://127.0.0.1:1/gladiator',
    botId: 'glad-stopped',
    logger: { info() {}, warn() {}, error() {} },
  });
  client.start();
  client.stop();
  assert.equal(client.reconnectTimer, null);
  assert.equal(client.stopped, true);
});
