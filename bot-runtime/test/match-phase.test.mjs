/**
 * `match-phase.js` 单测：阶段归一化、放行表、准备期动作裁剪、剩余秒数递减。
 *
 * 背景（2026-11-28 用户实测）：Bot 不会利用 countdown 时间准备，甚至提前出手。
 * 插件侧改成每秒推一帧 `MATCH_PHASE`，运行时据此在准备期只「朝向对手 + 待命」。
 * 这里把那条策略钉死，避免以后改 index.js 时又把进攻动作放回准备期。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { LinkClient } from '../src/connection.js';
import {
  MATCH_PHASES,
  countdownLine,
  facingAction,
  filterPhaseActions,
  normalizeMatchPhase,
  phaseGate,
  remainingSeconds,
} from '../src/match-phase.js';

/** 一帧完整的 countdown（插件侧的真实形态，见 docs/protocol.md §2.11）。 */
function countdownFrame(nowMs = 1_000, seconds = 5) {
  return normalizeMatchPhase({
    phase: 'countdown', round: 1, seconds, endsAt: nowMs + seconds * 1000,
  }, nowMs);
}

test('countdown 期间不允许进攻：attack 被裁掉，只留 look/idle', () => {
  const actions = [
    { type: 'attack', params: { targetEntityId: 7 } },
    { type: 'move', params: { direction: 'forward' } },
    { type: 'look', params: { yaw: 0, pitch: 0 } },
    { type: 'idle', params: { durationMs: 200 } },
  ];
  const kept = filterPhaseActions(actions, 'countdown');
  assert.deepEqual(kept.map((action) => action.type), ['look', 'idle']);
  assert.equal(phaseGate('countdown').allowAttack, false);
  assert.equal(phaseGate('countdown').hold, true);
  // starting 是「已开赛但还没摆位」，同样不许出手。
  assert.equal(phaseGate('starting').allowAttack, false);
  assert.deepEqual(filterPhaseActions(actions, 'starting').map((a) => a.type), ['look', 'idle']);
});

test('fighting 起才放行 attack（原样返回，不做任何裁剪）', () => {
  const actions = [
    { type: 'attack', params: { targetEntityId: 7 } },
    { type: 'move', params: { direction: 'forward' } },
  ];
  assert.deepEqual(filterPhaseActions(actions, 'fighting'), actions);
  const gate = phaseGate('fighting');
  assert.equal(gate.allowAttack, true);
  assert.equal(gate.hold, false);
  assert.equal(gate.ended, false);
});

test('ended 后全停：任何动作都不发', () => {
  const actions = [
    { type: 'attack', params: { targetEntityId: 7 } },
    { type: 'look', params: { yaw: 0, pitch: 0 } },
    { type: 'idle', params: { durationMs: 200 } },
  ];
  assert.deepEqual(filterPhaseActions(actions, 'ended'), []);
  const gate = phaseGate('ended');
  assert.equal(gate.ended, true);
  assert.equal(gate.allowAttack, false);
});

test('缺字段/乱字段不崩：归一化成「不知道阶段」，且不抑制老服务端', () => {
  for (const payload of [undefined, null, {}, { phase: 42 }, { phase: '' }]) {
    const frame = normalizeMatchPhase(payload, 5_000);
    assert.equal(frame.phase, null);
    assert.equal(frame.round, 0);
    assert.equal(frame.seconds, -1);
    assert.equal(frame.endsAt, 0);
    assert.equal(frame.receivedAt, 5_000);
    // 关键：没收到过阶段 → 不限制（否则老服务端的 Bot 会被永久冻住）。
    assert.equal(phaseGate(frame.phase).allowAttack, true);
    assert.deepEqual(
      filterPhaseActions([{ type: 'attack' }], frame.phase).map((a) => a.type),
      ['attack'],
    );
  }
  // 未知阶段名同样不限制；只有协议里的四个词有语义。
  assert.equal(phaseGate('suddendeath').known, false);
  assert.equal(phaseGate('suddendeath').allowAttack, true);
  assert.deepEqual(MATCH_PHASES, ['starting', 'countdown', 'fighting', 'ended']);
});

test('准备期剩余秒数按 endsAt 递减，日志行跟着变', () => {
  const base = 1_000;
  const frame = countdownFrame(base, 5);
  assert.equal(remainingSeconds(frame, base), 5);
  assert.equal(remainingSeconds(frame, base + 1_000), 4);
  assert.equal(remainingSeconds(frame, base + 2_400), 3);
  assert.equal(remainingSeconds(frame, base + 5_000), 0);
  // 超过 endsAt 之后不许出现负数（endsAt 被时钟漂移甩在后面时也一样）。
  assert.equal(remainingSeconds(frame, base + 60_000), 0);
  assert.equal(countdownLine(remainingSeconds(frame, base)), '[brain] 准备期 5s：面向对手待命');
  assert.equal(countdownLine(remainingSeconds(frame, base + 1_000)), '[brain] 准备期 4s：面向对手待命');
  // 没有 endsAt 时退回帧里的 seconds（插件每帧都会带）。
  assert.equal(remainingSeconds({ seconds: 2 }, base), 2);
});

test('准备期间「面向对手」的朝向：yaw 朝向目标，拿不到坐标就只 idle', () => {
  // 1.8/mineflayer 的 yaw：0 = +Z，π/2 = −X，−π/2 = +X（与 arena 的 yawTowards 同口径）。
  const north = facingAction({ x: 0, z: 0 }, { x: 0, z: 5 });
  assert.equal(north.type, 'look');
  assert.ok(Math.abs(north.params.yaw) < 1e-9); // 目标在 +Z → yaw 0
  const minusX = facingAction({ x: 0, z: 0 }, { x: -5, z: 0 }); // 目标在 −X → yaw +90°
  assert.ok(Math.abs(minusX.params.yaw - Math.PI / 2) < 1e-9);
  const plusX = facingAction({ x: 0, z: 0 }, { x: 5, z: 0 }); // 目标在 +X → yaw −90°
  assert.ok(Math.abs(plusX.params.yaw + Math.PI / 2) < 1e-9);
  assert.equal(plusX.params.pitch, 0);
  // 目标不可见（实体表里没有对手）：不发 look，只待命。
  assert.equal(facingAction({ x: 0, z: 0 }, null), null);
  assert.equal(facingAction(null, { x: 1, z: 1 }), null);
});

test('链路层：MATCH_PHASE 存进 state.matchPhase 并单独 emit（不进 message 兜底）', () => {
  const client = new LinkClient({
    botId: 'glad-bot_1',
    logger: { info() {}, warn() {}, error() {} },
  });
  let emitted = null;
  let rawMessage = null;
  client.on('match-phase', (frame) => { emitted = frame; });
  client.on('message', (envelope) => { rawMessage = envelope; });
  const endsAt = Date.now() + 5_000;
  // 直接喂一帧（不建连接：_onMessage 只做解析 + 分派，start() 之前也能用）。
  client._onMessage(Buffer.from(JSON.stringify({
    v: 1, type: 'MATCH_PHASE', seq: 7, ts: Date.now(), botId: 'glad-bot_1',
    payload: { phase: 'countdown', round: 1, seconds: 5, endsAt },
  })));
  assert.equal(client.state.matchPhase.phase, 'countdown');
  assert.equal(client.state.matchPhase.round, 1);
  assert.equal(client.state.matchPhase.endsAt, endsAt);
  assert.equal(emitted.phase, 'countdown');
  // 每秒一帧是常态：不能落进 index.js 的「未知类型」兜底日志，否则会刷屏。
  assert.equal(rawMessage, null);
});
