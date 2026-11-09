#!/usr/bin/env node
/**
 * 通道 B 冒烟探针（Layer 3 侧最小客户端）。
 *
 * 用法：node scripts/env-probe.mjs [--bot glad-glad_bot_1] [--port 8766]
 * 验证：HELLO→WELCOME、OBSERVATION 推送、ACTION_RESULT（含限流与 seq 幂等）。
 */
import { WebSocket } from 'ws';

function arg(name, fallback) {
  const argv = process.argv.slice(2);
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1]) {
    return argv[i + 1];
  }
  return fallback;
}

const botId = arg('bot', 'glad-glad_bot_1');
const port = Number(arg('port', 8766));
const socket = new WebSocket(`ws://127.0.0.1:${port}`);
const inbox = [];
let seq = 0;

function send(type, payload, replySeq = null) {
  seq += 1;
  socket.send(JSON.stringify({ v: 1, type, seq, ts: Date.now(), replyTo: replySeq, payload }));
  return seq;
}

function waitFor(predicate, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      const found = inbox.find(predicate);
      if (found) return resolve(found);
      if (Date.now() > deadline) return reject(new Error('等待超时'));
      setTimeout(tick, 20);
    };
    tick();
  });
}

socket.on('message', (raw) => inbox.push(JSON.parse(String(raw))));
socket.on('error', (error) => {
  console.error(`[probe] socket 错误: ${error.message}`);
  process.exit(1);
});

await new Promise((resolve, reject) => {
  socket.once('open', resolve);
  socket.once('error', reject);
});

send('HELLO', { v: 1, protocol: 'gladiator-env', botId });
const welcome = await waitFor((f) => f.type === 'WELCOME');
console.log(`[probe] WELCOME controlHz=${welcome.payload.controlHz} types=${welcome.payload.actionSpec.types.join(',')}`);

const observation = await waitFor((f) => f.type === 'OBSERVATION');
console.log(`[probe] OBSERVATION tick=${observation.payload.tick} gameId=${observation.payload.gameId || '-'}`
  + ` pos=${JSON.stringify(observation.payload.self.pos)} entities=${observation.payload.entities.length}`
  + ` palette=${observation.payload.blockWindow.palette.length}`);

const talkSeq = send('ACTION', { type: 'talk', params: { message: 'hello from layer3' } });
const talkResult = await waitFor((f) => f.type === 'ACTION_RESULT' && f.replyTo === talkSeq);
console.log(`[probe] ACTION talk → ${JSON.stringify(talkResult.payload)}`);

const dupSeq = send('ACTION', { type: 'talk', params: { message: 'hello from layer3' } });
await waitFor((f) => f.type === 'ACTION_RESULT' && f.replyTo === dupSeq);
// 用同一 seq 重发，期望 deduped
socket.send(JSON.stringify({ v: 1, type: 'ACTION', seq: talkSeq, ts: Date.now(), payload: { type: 'talk', params: { message: 'dup' } } }));
const dupResult = await waitFor((f) => f.type === 'ACTION_RESULT' && f.replyTo === talkSeq
  && f.payload.deduped === true);
console.log(`[probe] 重复 seq → ${JSON.stringify(dupResult.payload)}`);

const burst = [];
for (let i = 0; i < 20; i += 1) {
  burst.push(send('ACTION', { type: 'idle', params: { durationMs: 5 } }));
}
await waitFor((f) => f.type === 'ACTION_RESULT' && burst.includes(f.replyTo)
  && f.payload.code === 'ACTION_RATE_LIMITED');
const results = inbox.filter((f) => f.type === 'ACTION_RESULT' && burst.includes(f.replyTo));
const limited = results.filter((f) => f.payload.code === 'ACTION_RATE_LIMITED').length;
console.log(`[probe] 限流：${limited}/${results.length} 个动作被丢弃（maxActionsPerSecond=10）`);

socket.close();
process.exit(0);
