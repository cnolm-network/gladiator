/**
 * RCON 客户端回归测试（G-716）。
 *
 * <p>背景（真机 2026-11-24）：`RconClient.connect()` 认证成功后把 socket 的 `data`
 * 监听器一起 removeListener 了，于是认证之后**所有响应都读不到**——`command()` 每次等满
 * 3 秒超时并 resolve 空串，`/gladiator stats` 采样恒为 null，策略侧 `opponent_health`
 * 退化成「玩家实体默认血量 0」，Bot 站着不出手。
 *
 * <p>这里用一个说 Source RCON 协议的假服务器把「认证后还能收包」钉死。
 */

import assert from 'node:assert/strict';
import net from 'node:net';
import test from 'node:test';

import {
  RconClient,
  SERVERDATA_AUTH,
  SERVERDATA_AUTH_RESPONSE,
  SERVERDATA_EXECCOMMAND,
  SERVERDATA_RESPONSE_VALUE,
  parseRconSpec,
  parseStatsLine,
} from '../src/rcon.js';

/** 按协议编码一帧（长度不含自身 4 字节）。 */
function frame(id, kind, body) {
  const payload = Buffer.from(String(body), 'utf8');
  const buf = Buffer.alloc(14 + payload.length);
  buf.writeInt32LE(10 + payload.length, 0);
  buf.writeInt32LE(id, 4);
  buf.writeInt32LE(kind, 8);
  payload.copy(buf, 12);
  buf.writeInt16LE(0, 12 + payload.length);
  return buf;
}

/**
 * 假 RCON 服务器：认证通过后把 EXECCOMMAND 的内容回显成 `echo:<命令>`。
 * `replyTimes` 可以指定某个命令连续发几包（模拟 Bukkit 的「正文 + 空终止包」）。
 */
function startFakeServer({ password = 'devpass', silent = false } = {}) {
  const received = [];
  const server = net.createServer((socket) => {
    let buffer = Buffer.alloc(0);
    socket.on('data', (chunk) => {
      buffer = Buffer.concat([buffer, chunk]);
      for (;;) {
        if (buffer.length < 4) {
          return;
        }
        const length = buffer.readInt32LE(0);
        if (length < 10) {
          // 长度不合法的包：丢掉已缓冲的数据，避免死循环。
          buffer = Buffer.alloc(0);
          return;
        }
        if (buffer.length < length + 4) {
          return;
        }
        const body = buffer.subarray(12, length + 2).toString('utf8');
        const id = buffer.readInt32LE(4);
        const kind = buffer.readInt32LE(8);
        buffer = buffer.subarray(length + 4);
        if (kind === SERVERDATA_AUTH) {
          const ok = body === password;
          socket.write(frame(ok ? id : -1, SERVERDATA_AUTH_RESPONSE, ''));
          continue;
        }
        if (kind === SERVERDATA_EXECCOMMAND) {
          received.push(body);
          if (silent) {
            continue;
          }
          socket.write(frame(id, SERVERDATA_RESPONSE_VALUE, `echo:${body}`));
          // Spigot 1.8 常常在正文后再补一个空包（同一 id）：客户端不能因此错位。
          socket.write(frame(id, SERVERDATA_RESPONSE_VALUE, ''));
        }
      }
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        port,
        received,
        close: () => new Promise((done) => server.close(done)),
      });
    });
  });
}

test('认证之后仍能收到命令响应（G-716 回归）', async () => {
  const fake = await startFakeServer();
  const client = new RconClient({ host: '127.0.0.1', port: fake.port, password: 'devpass',
    timeout: 1000 });
  try {
    await client.connect();
    assert.equal(client.ready, true);

    const reply = await client.command('gladiator stats Steve');
    assert.equal(fake.received[0], 'gladiator stats Steve');
    // 修复前这里恒为空串（data 监听器已被摘掉，只能等超时）。
    assert.equal(reply, 'echo:gladiator stats Steve');
  } finally {
    client.close();
    await fake.close();
  }
});

test('连续两条命令各自拿到自己的响应（按 id 匹配）', async () => {
  const fake = await startFakeServer();
  const client = new RconClient({ host: '127.0.0.1', port: fake.port, timeout: 1000 });
  try {
    await client.connect();
    const [first, second] = await Promise.all([
      client.command('one'),
      client.command('two'),
    ]);
    assert.equal(first, 'echo:one');
    assert.equal(second, 'echo:two');
  } finally {
    client.close();
    await fake.close();
  }
});

test('服务器不回包时按超时返回空串，连接仍可继续用', async () => {
  const fake = await startFakeServer({ silent: true });
  const client = new RconClient({ host: '127.0.0.1', port: fake.port, timeout: 120 });
  try {
    await client.connect();
    assert.equal(await client.command('nobody answers'), '');
    assert.equal(client.ready, true, '一次超时不该让连接失效');
  } finally {
    client.close();
    await fake.close();
  }
});

test('密码错误时 connect 拒绝', async () => {
  const fake = await startFakeServer();
  const client = new RconClient({ host: '127.0.0.1', port: fake.port, password: 'wrong',
    timeout: 1000 });
  await assert.rejects(() => client.connect());
  client.close();
  await fake.close();
});

test('parseRconSpec 与 parseStatsLine 保持原契约', () => {
  assert.deepEqual(parseRconSpec('127.0.0.1:25575:devpass'),
    { host: '127.0.0.1', port: 25575, password: 'devpass' });
  assert.deepEqual(parseRconSpec('25575'), { host: '127.0.0.1', port: 25575, password: 'devpass' });
  assert.equal(parseRconSpec(''), null);
  const stats = parseStatsLine(
    'name=Glad_P1 health=12.0 food=20 saturation=5.0 armor=20.0 x=-1.0 y=4.0 z=-2.0'
    + ' yaw=0.0 pitch=0.0 sprinting=false blocking=false onGround=true hurt=0.6 dead=false effects=-');
  assert.equal(stats.health, 12);
  assert.equal(stats.armor, 20);
  assert.equal(stats.sprinting, false);
  assert.equal(parseStatsLine('stats 失败：玩家不在线'), null);
});
