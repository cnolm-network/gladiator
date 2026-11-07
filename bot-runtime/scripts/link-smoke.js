#!/usr/bin/env node
/**
 * 通道 A 冒烟测试：用真实的 ws 客户端验证插件侧 LinkServer 的握手与心跳。
 *
 * 需要插件已启用且监听回环端口（config.yml 的 bridge.port / bridge.token）。
 * 不改变任何文件、不连接 Minecraft 服务端，只验证 Layer 1 ↔ Layer 2 链路。
 *
 * 用法：
 *   node scripts/link-smoke.js
 *   node scripts/link-smoke.js --port 8765 --token <token> --bot glad-smoke-1
 *
 * 退出码：0 = HELLO→WELCOME→PING→PONG 全部通过；1 = 链路不可用或握手被拒。
 */
import WebSocket from 'ws';

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : fallback;
};

const port = getArg('port', '8765');
const token = getArg('token', '');
const botId = getArg('bot', 'glad-smoke-1');
const url = `ws://127.0.0.1:${port}/gladiator`;

let seq = 0;
let settled = false;
const deadline = setTimeout(() => finish(1, 'timeout: no complete handshake within 10s'), 10000);

function send(ws, type, payload, replyTo = null) {
  seq += 1;
  ws.send(JSON.stringify({
    v: 1,
    type,
    seq,
    ts: Date.now(),
    botId: type === 'HELLO' ? '' : botId,
    replyTo,
    payload,
  }));
}

function finish(code, message) {
  if (settled) return;
  settled = true;
  clearTimeout(deadline);
  console.log(`${code === 0 ? 'PASS' : 'FAIL'}: ${message}`);
  process.exit(code);
}

const ws = new WebSocket(url, { handshakeTimeout: 5000 });

ws.on('open', () => {
  send(ws, 'HELLO', {
    token,
    botId,
    username: `Smoke_${botId}`,
    mcVersion: '1.8.8',
    runtimeVersion: '0.1.0',
  });
});

ws.on('message', (raw) => {
  let env;
  try {
    env = JSON.parse(raw.toString());
  } catch (err) {
    finish(1, `malformed JSON from server: ${err.message}`);
    return;
  }
  if (env.type === 'WELCOME') {
    const p = env.payload ?? {};
    console.log(`WELCOME session=${p.sessionId} serverVersion=${p.serverVersion ?? 'n/a'}`);
    console.log(`features=${JSON.stringify(p.features ?? [])} limits=${JSON.stringify(p.limits ?? {})}`);
    send(ws, 'PING', {}, env.seq);
  } else if (env.type === 'PONG') {
    finish(0, `HELLO → WELCOME → PING → PONG completed for ${botId}`);
  } else if (env.type === 'REJECT') {
    finish(1, `REJECT ${env.payload?.code}: ${env.payload?.message}`);
  }
});

ws.on('error', (err) => finish(1, `connection error: ${err.message}`));
ws.on('close', () => {
  if (!settled) finish(1, 'connection closed before handshake completed');
});
