/**
 * `policy.js` 单测：npz/npy 解析 + 前向数值 + 温度与维度兼容。
 *
 * 测试自己写一个最小 zip（stored）与 npy（float64/float32/文本），
 * 不依赖 numpy，也不需要仓库里存在模型文件。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadPolicy, readNpy, readNpz, resolvePolicyFile } from '../src/policy.js';

// --- 最小 npy / zip 写入器（仅测试用） --------------------------------------

function padHeader(headerText) {
  const magic = 10; // \x93NUMPY + version(2) + headerLen(2)
  let text = headerText;
  // numpy 要求 (magic + header) 是 64 的整数倍，且以 \n 结尾。
  while ((magic + text.length + 1) % 64 !== 0) {
    text += ' ';
  }
  return `${text}\n`;
}

function npy(values, shape, descr = '<f8') {
  const header = padHeader(
    `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.map((n) => String(n)).join(', ')}, ), }`,
  );
  const headerBuf = Buffer.from(header, 'latin1');
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  const lenBuf = Buffer.alloc(2);
  lenBuf.writeUInt16LE(headerBuf.length, 0);
  const isFloat32 = descr === '<f4';
  const body = Buffer.alloc(values.length * (isFloat32 ? 4 : 8));
  values.forEach((value, index) => {
    if (isFloat32) {
      body.writeFloatLE(value, index * 4);
    } else {
      body.writeDoubleLE(value, index * 8);
    }
  });
  return Buffer.concat([magic, lenBuf, headerBuf, body]);
}

function npyText(text) {
  const codepoints = Array.from(text);
  const header = padHeader(
    `{'descr': '<U${codepoints.length}', 'fortran_order': False, 'shape': (), }`,
  );
  const headerBuf = Buffer.from(header, 'latin1');
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  const lenBuf = Buffer.alloc(2);
  lenBuf.writeUInt16LE(headerBuf.length, 0);
  const body = Buffer.alloc(codepoints.length * 4);
  codepoints.forEach((char, index) => {
    body.writeUInt32LE(char.codePointAt(0), index * 4);
  });
  return Buffer.concat([magic, lenBuf, headerBuf, body]);
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

function crc32(buf) {
  let crc = 0xffffffff;
  for (const byte of buf) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function zipStored(entries) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const nameBuf = Buffer.from(name, 'utf8');
    const crc = crc32(data);
    const local = Buffer.alloc(30 + nameBuf.length);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(0, 8); // stored
    local.writeUInt16LE(0, 10);
    local.writeUInt16LE(0, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28);
    nameBuf.copy(local, 30);
    locals.push(local, data);

    const central = Buffer.alloc(46 + nameBuf.length);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 8);
    central.writeUInt16LE(0, 10);
    central.writeUInt16LE(0, 12);
    central.writeUInt16LE(0, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt16LE(0, 30);
    central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34);
    central.writeUInt16LE(0, 36);
    central.writeUInt32LE(0, 38);
    central.writeUInt32LE(offset, 42);
    nameBuf.copy(central, 46);
    centrals.push(central);

    offset += local.length + data.length;
  }
  const centralBuf = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(centralBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralBuf, eocd]);
}

function makeNpz({ W1, b1, W2, b2, obsDim, numActions, hidden, temperature }) {
  const entries = [
    ['W1.npy', npy(W1.flat(), [hidden, obsDim])],
    ['b1.npy', npy(b1, [hidden])],
    ['W2.npy', npy(W2.flat(), [numActions, hidden])],
    ['b2.npy', npy(b2, [numActions])],
  ];
  if (temperature !== undefined) {
    const meta = JSON.stringify({ obs_dim: obsDim, n_actions: numActions, hidden, temperature });
    entries.push(['_meta.npy', npyText(meta)]);
  }
  return zipStored(entries);
}

function tempFile(name, buffer) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gladiator-policy-'));
  const file = path.join(dir, name);
  fs.writeFileSync(file, buffer);
  return { dir, file };
}

// --- 用例 ------------------------------------------------------------------

test('readNpy 支持 float32 一维', () => {
  const buf = npy([1.5, -2.25, 0], [3], '<f4');
  const parsed = readNpy(buf);
  assert.deepEqual(parsed.shape, [3]);
  assert.equal(parsed.values.length, 3);
  assert.ok(Math.abs(parsed.values[0] - 1.5) < 1e-6);
  assert.ok(Math.abs(parsed.values[1] + 2.25) < 1e-6);
});

test('readNpz 解析参数名并保留文本 _meta', () => {
  const buffer = makeNpz({
    W1: [1, 2, 3, 4], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0, 0],
    obsDim: 2, numActions: 2, hidden: 2, temperature: 1.5,
  });
  const entries = readNpz(buffer);
  assert.deepEqual([...entries.keys()].sort(), ['W1', 'W2', '_meta', 'b1', 'b2']);
  assert.equal(entries.get('W1').shape.join('x'), '2x2');
  assert.match(entries.get('_meta').text, /"temperature":1\.5/);
});

test('forward 与手算 softmax 一致（temperature=2）', () => {
  // W1=I, b1=0 → h = tanh(x)；W2=I, b2=0 → logits = h；p = softmax(h/2)。
  const buffer = makeNpz({
    W1: [1, 0, 0, 1], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0, 0],
    obsDim: 2, numActions: 2, hidden: 2, temperature: 2,
  });
  const { file } = tempFile('hand.npz', buffer);
  const policy = loadPolicy(file);
  assert.equal(policy.obsDim, 2);
  assert.equal(policy.hidden, 2);
  assert.equal(policy.numActions, 2);
  assert.equal(policy.temperature, 2);

  const x = [0.5, -1.0];
  const h = [Math.tanh(0.5), Math.tanh(-1.0)];
  const z = h.map((value) => value / 2);
  const max = Math.max(...z);
  const exps = z.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const expected = exps.map((value) => value / total);

  const probs = policy.forward(x);
  assert.ok(Math.abs(probs[0] - expected[0]) < 1e-12, `${probs[0]} vs ${expected[0]}`);
  assert.ok(Math.abs(probs[1] - expected[1]) < 1e-12);
  assert.ok(Math.abs(probs[0] + probs[1] - 1) < 1e-12);

  const logits = policy.logits(x);
  assert.ok(Math.abs(logits[0] - Math.tanh(0.5)) < 1e-12);
  assert.equal(policy.greedy(x), 0);
});

test('greedy 取 argmax，sample 落在合法动作内', () => {
  // logits = [0.1, 0.9]，温度 1 → 动作 1 概率更高。
  const buffer = makeNpz({
    W1: [1, 0, 0, 1], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0.1, 0.9],
    obsDim: 2, numActions: 2, hidden: 2, temperature: 1,
  });
  const { file } = tempFile('argmax.npz', buffer);
  const policy = loadPolicy(file);
  assert.equal(policy.greedy([0, 0]), 1);
  for (let index = 0; index < 20; index += 1) {
    const action = policy.sample([0, 0]);
    assert.ok(action === 0 || action === 1);
  }
});

test('缺少 _meta 时温度按 1.0，维度由权重形状决定', () => {
  const buffer = makeNpz({
    W1: [0.5, 0, 0, 0.5, 0, 0], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0, 0],
    obsDim: 3, numActions: 2, hidden: 2,
  });
  const { file } = tempFile('nometa.npz', buffer);
  const policy = loadPolicy(file);
  assert.equal(policy.temperature, 1);
  assert.equal(policy.obsDim, 3);
});

test('特征维度与模型不一致时按 0 补齐/忽略多余列（与 policy.py 补零列等价）', () => {
  const buffer = makeNpz({
    W1: [1, 0, 0, 1], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0, 0],
    obsDim: 2, numActions: 2, hidden: 2, temperature: 1,
  });
  const { file } = tempFile('pad.npz', buffer);
  const policy = loadPolicy(file);
  // 多出来的特征被忽略：与只喂前两维完全一致。
  const longer = policy.forward([0.4, -0.2, 9, 9, 9]);
  const exact = policy.forward([0.4, -0.2]);
  assert.deepEqual(Array.from(longer), Array.from(exact));
  // 特征不足时缺列按 0。
  const shorter = policy.forward([0.4]);
  const zeroPadded = policy.forward([0.4, 0]);
  assert.deepEqual(Array.from(shorter), Array.from(zeroPadded));
});

test('权重维度不一致时给出中文错误', () => {
  const buffer = zipStored([
    ['W1.npy', npy([1, 0, 0, 1], [2, 2])],
    ['b1.npy', npy([0, 0, 0], [3])],
    ['W2.npy', npy([1, 0, 0, 1], [2, 2])],
    ['b2.npy', npy([0, 0], [2])],
  ]);
  const { file } = tempFile('bad.npz', buffer);
  assert.throws(() => loadPolicy(file), /W1\/b1 维度不一致/);
});

test('缺少必需权重时给出中文错误', () => {
  const buffer = zipStored([['W1.npy', npy([1], [1, 1])]]);
  const { file } = tempFile('missing.npz', buffer);
  assert.throws(() => loadPolicy(file), /缺少权重 b1/);
});

test('resolvePolicyFile 支持路径、模型 id 与 .npz 直连', () => {
  const buffer = makeNpz({
    W1: [1, 0, 0, 1], b1: [0, 0], W2: [1, 0, 0, 1], b2: [0, 0],
    obsDim: 2, numActions: 2, hidden: 2, temperature: 1,
  });
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gladiator-models-'));
  fs.mkdirSync(path.join(root, 'dagger-1118b'), { recursive: true });
  fs.writeFileSync(path.join(root, 'dagger-1118b', 'policy.npz'), buffer);
  fs.writeFileSync(path.join(root, 'flat.npz'), buffer);

  assert.equal(resolvePolicyFile('dagger-1118b', { modelsDir: root }),
    path.join(root, 'dagger-1118b', 'policy.npz'));
  assert.equal(resolvePolicyFile('flat', { modelsDir: root }), path.join(root, 'flat.npz'));
  assert.equal(resolvePolicyFile(path.join(root, 'flat.npz'), { modelsDir: root }),
    path.join(root, 'flat.npz'));
  assert.equal(resolvePolicyFile('nope', { modelsDir: root }), null);
  assert.equal(resolvePolicyFile('', { modelsDir: root }), null);
});
