/**
 * 策略加载与前向推理（纯 JS，无第三方依赖）。
 *
 * <p>模型来源是训练侧 `Policy.save()` 产出的 `.npz`：一个 zip，里面是若干 `.npy`，
 * 参数名 `W1,b1,W2,b2`（价值头 `Wv,bv` 忽略）外加 `_meta`（JSON 字符串）。
 *
 * <p>网络结构必须与 `training/gladiator_train/policy.py` 完全同构：
 * ```
 * h      = tanh(W1·x + b1)
 * logits = W2·h + b2
 * p      = softmax(logits / temperature)
 * ```
 * `temperature` 从 `_meta` 读（缺失按 1.0），与 `Policy.load()` 的兼容规则一致。
 *
 * <p>为什么自己解 zip/npy：运行时只依赖 mineflayer/ws，不能为一个只读的权重文件
 * 引入 numpy 或额外 npm 包；zip 中央目录 + npy header 都是极小、稳定的格式。
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { ACTION_HEADS_V2, HEAD_ORDER_V2, HEAD_SIZES_V2 } from './action-space-v2.js';

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
const ZIP_EOCD_SIG = 0x06054b50;
const ZIP_CENTRAL_SIG = 0x02014b50;
const ZIP_LOCAL_SIG = 0x04034b50;

function fail(message) {
  throw new Error(message);
}

// ---------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------

function product(shape) {
  return shape.reduce((acc, value) => acc * value, 1);
}

const DTYPE_RE = /^([<>=|])([fiuSUab])(\d+)$/;

/**
 * 解析一个 `.npy` 字节串。
 *
 * @returns {{descr:string, shape:number[], fortran:boolean, dtype:string,
 *            values:Float64Array|string[], text:string}}
 *   `values` 对浮点是 Float64Array（行优先展开），对字符串是逐元素数组；
 *   `text` 是字符串数组拼接结果（`_meta` 用）。
 */
export function readNpy(source) {
  const buf = Buffer.isBuffer(source) ? source : Buffer.from(source);
  if (buf.length < 10) {
    fail('npy 文件过短，无法解析');
  }
  for (let index = 0; index < NPY_MAGIC.length; index += 1) {
    if (buf[index] !== NPY_MAGIC[index]) {
      fail('不是合法的 .npy 文件（magic 不匹配）');
    }
  }
  const major = buf[6];
  let headerLen;
  let headerStart;
  if (major === 1) {
    headerLen = buf.readUInt16LE(8);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLen = buf.readUInt32LE(8);
    headerStart = 12;
  } else {
    fail(`不支持的 npy 版本 ${major}`);
  }
  const headerEnd = headerStart + headerLen;
  if (headerEnd > buf.length) {
    fail('npy header 长度越界');
  }
  // header 是 Python dict 的字面量（ASCII），latin1 足够且不会因非 ASCII 抛错。
  const header = buf.subarray(headerStart, headerEnd).toString('latin1');
  const descr = (/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(header) || [])[1];
  if (!descr) {
    fail('npy header 缺少 descr');
  }
  const fortranRaw = (/['"]fortran_order['"]\s*:\s*(True|False)/.exec(header) || [])[1];
  const shapeRaw = (/['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(header) || [])[1];
  if (shapeRaw === undefined) {
    fail('npy header 缺少 shape');
  }
  const shape = shapeRaw
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => {
      const value = Number(part);
      if (!Number.isInteger(value) || value < 0) {
        fail(`npy shape 非法: ${shapeRaw}`);
      }
      return value;
    });
  const fortran = fortranRaw === 'True';

  const match = DTYPE_RE.exec(descr);
  if (!match) {
    fail(`不支持的 npy dtype: ${descr}`);
  }
  const [, order, kind, sizeText] = match;
  const itemsize = Number(sizeText);
  // 注意：`<U44` 的 44 是**字符数**，每个码点占 4 字节（numpy 的字符串 dtype 约定）。
  const bytesPerElement = kind === 'U' ? itemsize * 4 : itemsize;
  const count = product(shape);
  const need = count * bytesPerElement;
  const data = buf.subarray(headerEnd);
  if (data.length < need) {
    fail(`npy 数据长度不足：需要 ${need} 字节，实际 ${data.length}`);
  }
  const little = order !== '>';

  if (kind === 'f') {
    if (itemsize !== 4 && itemsize !== 8 && itemsize !== 2) {
      fail(`不支持的浮点位宽: ${descr}`);
    }
    if (fortran && shape.length > 1) {
      fail(`不支持 Fortran 序的多维权重: ${descr} ${shape.join('x')}`);
    }
    const values = new Float64Array(count);
    const view = new DataView(data.buffer, data.byteOffset, need);
    for (let index = 0; index < count; index += 1) {
      const offset = index * itemsize;
      if (itemsize === 8) {
        values[index] = view.getFloat64(offset, little);
      } else if (itemsize === 4) {
        values[index] = view.getFloat32(offset, little);
      } else {
        values[index] = halfToFloat(view.getUint16(offset, little));
      }
    }
    return { descr, shape, fortran, dtype: descr, values, text: '' };
  }

  if (kind === 'U') {
    // UTF-32 定长码点；字符串数组只用于 `_meta`（JSON 文本）。
    const per = itemsize;
    const out = [];
    const view = new DataView(data.buffer, data.byteOffset, need);
    for (let index = 0; index < count; index += 1) {
      let text = '';
      for (let unit = 0; unit < per; unit += 1) {
        const code = view.getUint32((index * per + unit) * 4, little);
        if (code === 0) {
          break;
        }
        text += String.fromCodePoint(code);
      }
      out.push(text);
    }
    return { descr, shape, fortran, dtype: descr, values: out, text: out.join('') };
  }

  if (kind === 'S' || kind === 'a') {
    const out = [];
    for (let index = 0; index < count; index += 1) {
      const chunk = data.subarray(index * itemsize, (index + 1) * itemsize);
      const zero = chunk.indexOf(0);
      out.push(chunk.subarray(0, zero < 0 ? chunk.length : zero).toString('utf8'));
    }
    return { descr, shape, fortran, dtype: descr, values: out, text: out.join('') };
  }

  if (kind === 'i' || kind === 'u' || kind === 'b') {
    if (fortran && shape.length > 1) {
      fail(`不支持 Fortran 序的多维数组: ${descr} ${shape.join('x')}`);
    }
    const values = new Float64Array(count);
    const view = new DataView(data.buffer, data.byteOffset, need);
    for (let index = 0; index < count; index += 1) {
      const offset = index * itemsize;
      if (kind === 'b') {
        values[index] = view.getUint8(offset) ? 1 : 0;
      } else if (kind === 'u') {
        values[index] = itemsize === 1 ? view.getUint8(offset)
          : itemsize === 2 ? view.getUint16(offset, little)
            : itemsize === 4 ? view.getUint32(offset, little)
              : Number(view.getBigUint64(offset, little));
      } else {
        values[index] = itemsize === 1 ? view.getInt8(offset)
          : itemsize === 2 ? view.getInt16(offset, little)
            : itemsize === 4 ? view.getInt32(offset, little)
              : Number(view.getBigInt64(offset, little));
      }
    }
    return { descr, shape, fortran, dtype: descr, values, text: '' };
  }

  return fail(`不支持的 npy dtype: ${descr}`);
}

function halfToFloat(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * fraction * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---------------------------------------------------------------------------
// .npz（zip）
// ---------------------------------------------------------------------------

/**
 * 读取 zip 内的所有条目（只支持 stored / deflate，足以覆盖 `np.savez`）。
 *
 * @returns {Map<string, Buffer>}
 */
export function readZip(buf) {
  let eocd = -1;
  const scanStart = Math.max(0, buf.length - 65557);
  for (let offset = buf.length - 22; offset >= scanStart; offset -= 1) {
    if (buf.readUInt32LE(offset) === ZIP_EOCD_SIG) {
      eocd = offset;
      break;
    }
  }
  if (eocd < 0) {
    fail('不是合法的 zip（找不到中央目录结束记录）');
  }
  const count = buf.readUInt16LE(eocd + 10);
  let cursor = buf.readUInt32LE(eocd + 16);
  const entries = new Map();
  for (let index = 0; index < count; index += 1) {
    if (cursor + 46 > buf.length || buf.readUInt32LE(cursor) !== ZIP_CENTRAL_SIG) {
      fail(`zip 中央目录损坏（第 ${index} 项）`);
    }
    const method = buf.readUInt16LE(cursor + 10);
    const compressedSize = buf.readUInt32LE(cursor + 20);
    const nameLen = buf.readUInt16LE(cursor + 28);
    const extraLen = buf.readUInt16LE(cursor + 30);
    const commentLen = buf.readUInt16LE(cursor + 32);
    const localOffset = buf.readUInt32LE(cursor + 42);
    const name = buf.subarray(cursor + 46, cursor + 46 + nameLen).toString('utf8');
    cursor += 46 + nameLen + extraLen + commentLen;

    if (buf.readUInt32LE(localOffset) !== ZIP_LOCAL_SIG) {
      fail(`zip 本地头损坏: ${name}`);
    }
    const localNameLen = buf.readUInt16LE(localOffset + 26);
    const localExtraLen = buf.readUInt16LE(localOffset + 28);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    const raw = buf.subarray(dataStart, dataStart + compressedSize);
    if (method === 0) {
      entries.set(name, raw);
    } else if (method === 8) {
      entries.set(name, zlib.inflateRawSync(raw));
    } else {
      fail(`不支持的 zip 压缩方式 ${method}: ${name}`);
    }
  }
  return entries;
}

/** 读取 `.npz`（路径或 Buffer），返回 `{参数名: readNpy 结果}`（去掉 `.npy` 后缀）。 */
export function readNpz(source) {
  const buf = typeof source === 'string' ? fs.readFileSync(source) : source;
  const out = new Map();
  for (const [name, data] of readZip(buf)) {
    out.set(name.replace(/\.npy$/i, ''), readNpy(data));
  }
  return out;
}

// ---------------------------------------------------------------------------
// 策略
// ---------------------------------------------------------------------------

function requireMatrix(entry, name) {
  if (!entry) {
    fail(`模型缺少权重 ${name}`);
  }
  if (entry.shape.length !== 2) {
    fail(`权重 ${name} 必须是二维，实际 shape=${JSON.stringify(entry.shape)}`);
  }
  return { rows: entry.shape[0], cols: entry.shape[1], data: entry.values };
}

function requireVector(entry, name) {
  if (!entry) {
    fail(`模型缺少权重 ${name}`);
  }
  if (entry.shape.length !== 1) {
    fail(`权重 ${name} 必须是一维，实际 shape=${JSON.stringify(entry.shape)}`);
  }
  return { length: entry.shape[0], data: entry.values };
}

/** 读 `_meta`（np.array(json_string)，dtype 为 `<U…`，shape 为 `()`）。坏 JSON 直接失败。 */
function readMeta(entries) {
  const entry = entries.get('_meta');
  if (!entry || !entry.text) {
    return {};
  }
  try {
    return JSON.parse(entry.text);
  } catch (error) {
    fail(`模型 _meta 不是合法 JSON: ${error.message}`);
  }
}

/**
 * 判定动作空间：`v1`（单头 11 宏动作）/ `v2`（十头 MultiDiscrete）。
 *
 * <p>以 `_meta.actionSpace` 为准（训练侧 `Policy.save()` 从 G-710 起会写这个字段）。
 * 老模型没有该字段时按权重形状推断——v2 只有 `W2_<head>`，没有单张 `W2`。
 *
 * @returns {'v1'|'v2'}
 */
export function detectActionSpace(meta, entries) {
  const raw = meta && meta.actionSpace !== undefined && meta.actionSpace !== null
    ? String(meta.actionSpace).trim().toLowerCase()
    : '';
  if (raw === 'v1' || raw === 'v2') {
    return raw;
  }
  if (raw) {
    fail(`模型 _meta.actionSpace=${meta.actionSpace} 不认识（允许 v1 / v2）`);
  }
  if (entries.has('W2')) {
    return 'v1';
  }
  for (const name of entries.keys()) {
    if (name.startsWith('W2_')) {
      return 'v2';
    }
  }
  return 'v1';
}

/**
 * 加载策略（按 `_meta.actionSpace` 自动分流）。
 *
 * <p>两种返回值形状一致（`obsDim/numActions/hidden/temperature/file/meta`），区别在：
 * - **v1**：单头，`logits/forward/greedy/sample` 与改造前逐位一致；
 * - **v2**：十头，用 `headLogits/headProbs/actWithMask`；单头接口会抛错（避免悄悄丢头）。
 *
 * @param {string} file `.npz` 路径
 * @returns {object} 见 `buildV1Policy` / `buildV2Policy`
 */
export function loadPolicy(file) {
  const entries = readNpz(file);
  const meta = readMeta(entries);
  return detectActionSpace(meta, entries) === 'v2'
    ? buildV2Policy(file, entries, meta)
    : buildV1Policy(file, entries, meta);
}

/**
 * v1：单头 softmax（11 个宏动作）。
 *
 * <p>这段实现是**回归基线**：动作空间 v2 的接入不允许改变它的任何数值行为，
 * `test/policy.test.mjs` 与 `dashboard/var/models/dagger-1118b` 是判据。
 */
function buildV1Policy(file, entries, meta) {
  const W1 = requireMatrix(entries.get('W1'), 'W1');
  const b1 = requireVector(entries.get('b1'), 'b1');
  const W2 = requireMatrix(entries.get('W2'), 'W2');
  const b2 = requireVector(entries.get('b2'), 'b2');

  if (W1.rows !== b1.length) {
    fail(`W1/b1 维度不一致: W1=${W1.rows}x${W1.cols} b1=${b1.length}`);
  }
  if (W2.rows !== b2.length) {
    fail(`W2/b2 维度不一致: W2=${W2.rows}x${W2.cols} b2=${b2.length}`);
  }
  if (W2.cols !== W1.rows) {
    fail(`W2 的输入维度必须等于隐层宽度: W2=${W2.cols} hidden=${W1.rows}`);
  }

  const obsDim = W1.cols;
  const hidden = W1.rows;
  const numActions = W2.rows;

  for (const key of ['obs_dim', 'hidden', 'n_actions']) {
    if (meta[key] !== undefined) {
      const expected = key === 'obs_dim' ? obsDim : key === 'hidden' ? hidden : numActions;
      if (Number(meta[key]) !== expected) {
        fail(`模型 _meta.${key}=${meta[key]} 与权重形状（${expected}）不一致: ${file}`);
      }
    }
  }
  const rawTemperature = Number(meta.temperature);
  const temperature = Number.isFinite(rawTemperature) && rawTemperature > 0 ? rawTemperature : 1;

  const hiddenBuffer = new Float64Array(hidden);
  const logitBuffer = new Float64Array(numActions);

  /**
   * 输入向量对齐到模型的 obsDim。
   *
   * <p>两种偏差都能安全处理，语义与 `policy.py` 的「W1 右侧补零列」等价：
   * - 特征比模型**多**（例如 23 维特征喂 19 维旧模型）→ 多出来的列权重当作 0，直接忽略；
   * - 特征比模型**少** → 缺的列按 0 处理。
   * 两种情况下模型只看到它训练时见过的那些特征，不会因为维度不同直接崩掉。
   */
  function normalizeInput(x) {
    const source = x || [];
    const input = new Float64Array(obsDim);
    const shared = Math.min(obsDim, source.length);
    for (let index = 0; index < shared; index += 1) {
      const value = Number(source[index]);
      input[index] = Number.isFinite(value) ? value : 0;
    }
    return input;
  }

  function logits(x) {
    const input = normalizeInput(x);
    for (let row = 0; row < hidden; row += 1) {
      let sum = b1.data[row];
      const base = row * obsDim;
      for (let col = 0; col < obsDim; col += 1) {
        sum += W1.data[base + col] * input[col];
      }
      hiddenBuffer[row] = Math.tanh(sum);
    }
    for (let row = 0; row < numActions; row += 1) {
      let sum = b2.data[row];
      const base = row * hidden;
      for (let col = 0; col < hidden; col += 1) {
        sum += W2.data[base + col] * hiddenBuffer[col];
      }
      logitBuffer[row] = sum;
    }
    return Array.from(logitBuffer);
  }

  function forward(x) {
    const raw = logits(x);
    let max = Number.NEGATIVE_INFINITY;
    const scaled = new Float64Array(numActions);
    for (let index = 0; index < numActions; index += 1) {
      scaled[index] = raw[index] / temperature;
      if (scaled[index] > max) {
        max = scaled[index];
      }
    }
    let total = 0;
    const exp = new Float64Array(numActions);
    for (let index = 0; index < numActions; index += 1) {
      const value = Math.exp(scaled[index] - max);
      exp[index] = value;
      total += value;
    }
    const probs = new Array(numActions);
    for (let index = 0; index < numActions; index += 1) {
      probs[index] = total > 0 ? exp[index] / total : 1 / numActions;
    }
    return probs;
  }

  function greedy(x) {
    const raw = logits(x);
    let best = 0;
    for (let index = 1; index < raw.length; index += 1) {
      if (raw[index] > raw[best]) {
        best = index;
      }
    }
    return best;
  }

  function sample(x, random = Math.random) {
    const probs = forward(x);
    let threshold = random();
    for (let index = 0; index < probs.length; index += 1) {
      threshold -= probs[index];
      if (threshold <= 0) {
        return index;
      }
    }
    return probs.length - 1;
  }

  return {
    file,
    //: 动作空间标识：`--brain` 据此决定走宏动作还是意图帧（G-712）。
    actionSpace: 'v1',
    //: v1 只有一个头，名字固定 `action`（与 Policy.head_names() 的口径一致）。
    headNames: ['action'],
    headSizes: [numActions],
    obsDim,
    numActions,
    hidden,
    temperature,
    meta,
    logits,
    forward,
    greedy,
    sample,
  };
}

/** 取某头某行的起始下标（行优先展开）。 */
function rowOffset(row, cols) {
  return row * cols;
}

/**
 * 把 `actWithMask` 的 mask 参数规范成「每头一个允许表」。
 *
 * <p>三种写法都收（调用方不用记形状）：
 * - `null/undefined` → 不掩码；
 * - `各头数组的数组`（长度 = 头数，按 `HEAD_ORDER_V2`）；
 * - `matchingMask()` 的返回值（`{slot, use, heads:{...}}`）。
 *
 * <p>数组比该头档位短时，缺的位按「允许」补（与 Python `normalize_mask` 一致）；
 * 只有显式 `false` / `0` 才算屏蔽。
 */
function normalizeMask(mask, heads) {
  if (mask === null || mask === undefined) {
    return null;
  }
  const source = Array.isArray(mask)
    ? mask
    : (mask.heads && typeof mask.heads === 'object' ? heads.map((head) => mask.heads[head.name]) : null);
  if (!source) {
    fail('actWithMask 的 mask 需要「每头一个数组」或 matchingMask() 的返回值');
  }
  return heads.map((head, index) => {
    const row = source[index];
    if (row === undefined || row === null) {
      return null;
    }
    const flags = new Array(head.size).fill(true);
    const list = Array.from(row);
    for (let value = 0; value < head.size && value < list.length; value += 1) {
      flags[value] = !(list[value] === false || list[value] === 0);
    }
    return flags;
  });
}

/**
 * v2：十头 factorized 策略（`W1/b1` 共享隐层 + 每头 `W2_<head>/b2_<head>`）。
 *
 * <p>结构必须与 `training/gladiator_train/policy.py` 一致：
 * `h = tanh(W1·x + b1)`，每头 `p_head = softmax((W2_head·h + b2_head)/T)`。
 * 头顺序/档位来自 `src/action-space-v2.js`（= `spaces.py::ACTION_HEADS_V2`）。
 */
function buildV2Policy(file, entries, meta) {
  const W1 = requireMatrix(entries.get('W1'), 'W1');
  const b1 = requireVector(entries.get('b1'), 'b1');
  if (W1.rows !== b1.length) {
    fail(`W1/b1 维度不一致: W1=${W1.rows}x${W1.cols} b1=${b1.length}`);
  }
  const obsDim = W1.cols;
  const hidden = W1.rows;

  // 头顺序是**契约**：meta 里写了 headNames 就必须与运行时逐位一致，否则权重会挂错头。
  if (Array.isArray(meta.headNames)) {
    const declared = meta.headNames.map(String);
    if (declared.join(',') !== HEAD_ORDER_V2.join(',')) {
      fail(`模型 _meta.headNames=[${declared.join(',')}] 与运行时顺序`
        + ` [${HEAD_ORDER_V2.join(',')}] 不一致：权重会挂到错的头`);
    }
  }
  if (Array.isArray(meta.headSizes)) {
    const declared = meta.headSizes.map(Number);
    if (declared.join(',') !== HEAD_SIZES_V2.join(',')) {
      fail(`模型 _meta.headSizes=[${declared.join(',')}] 与动作空间`
        + ` [${HEAD_SIZES_V2.join(',')}] 不一致`);
    }
  }

  const heads = HEAD_ORDER_V2.map((name) => {
    const W2 = requireMatrix(entries.get(`W2_${name}`), `W2_${name}`);
    const b2 = requireVector(entries.get(`b2_${name}`), `b2_${name}`);
    if (W2.rows !== b2.length) {
      fail(`W2_${name}/b2_${name} 维度不一致: W2=${W2.rows}x${W2.cols} b2=${b2.length}`);
    }
    if (W2.cols !== hidden) {
      fail(`W2_${name} 的输入维度必须等于隐层宽度: W2=${W2.cols} hidden=${hidden}`);
    }
    if (W2.rows !== ACTION_HEADS_V2[name]) {
      fail(`头 ${name} 的权重行数 ${W2.rows} 与动作空间档位 ${ACTION_HEADS_V2[name]} 不一致`);
    }
    return { name, size: W2.rows, W2: W2.data, b2: b2.data };
  });

  const numActions = heads.reduce((sum, head) => sum + head.size, 0);
  if (meta.n_actions !== undefined && Number(meta.n_actions) !== numActions) {
    fail(`模型 _meta.n_actions=${meta.n_actions} 与十头之和（${numActions}）不一致: ${file}`);
  }
  if (meta.obs_dim !== undefined && Number(meta.obs_dim) !== obsDim) {
    fail(`模型 _meta.obs_dim=${meta.obs_dim} 与权重形状（${obsDim}）不一致: ${file}`);
  }
  if (meta.hidden !== undefined && Number(meta.hidden) !== hidden) {
    fail(`模型 _meta.hidden=${meta.hidden} 与权重形状（${hidden}）不一致: ${file}`);
  }
  const rawTemperature = Number(meta.temperature);
  const temperature = Number.isFinite(rawTemperature) && rawTemperature > 0 ? rawTemperature : 1;

  const hiddenBuffer = new Float64Array(hidden);
  const logitBuffers = heads.map((head) => new Float64Array(head.size));
  /** mask 全关时回退到全开的次数（诊断用，等价于 Python 的 `mask_fallbacks`）。 */
  let maskFallbacks = 0;
  const warnedHeads = new Set();

  /** 输入对齐到模型的 obsDim（多出来的特征丢掉、缺的按 0 补，与 v1 同口径）。 */
  function normalizeInput(x) {
    const source = x || [];
    const input = new Float64Array(obsDim);
    const shared = Math.min(obsDim, source.length);
    for (let index = 0; index < shared; index += 1) {
      const value = Number(source[index]);
      input[index] = Number.isFinite(value) ? value : 0;
    }
    return input;
  }

  /** 只算隐层（多头的 logits 共用它，别重复算）。 */
  function fillHidden(x) {
    const input = normalizeInput(x);
    for (let row = 0; row < hidden; row += 1) {
      let sum = b1.data[row];
      const base = row * obsDim;
      for (let col = 0; col < obsDim; col += 1) {
        sum += W1.data[base + col] * input[col];
      }
      hiddenBuffer[row] = Math.tanh(sum);
    }
    return hiddenBuffer;
  }

  /** 每头未除温度的 logits；返回**新数组**（调用方可能改写）。 */
  function headLogits(x) {
    const h = fillHidden(x);
    return heads.map((head, headIndex) => {
      const out = logitBuffers[headIndex];
      for (let row = 0; row < head.size; row += 1) {
        let sum = head.b2[row];
        const base = rowOffset(row, hidden);
        for (let col = 0; col < hidden; col += 1) {
          sum += head.W2[base + col] * h[col];
        }
        out[row] = sum;
      }
      return Array.from(out);
    });
  }

  /** 每头带温度的 softmax（数值稳定）。 */
  function headProbs(x) {
    return headLogits(x).map((logits) => softmaxWithTemperature(logits, temperature));
  }

  /**
   * 十头动作：可选 mask、可选「按头随机探索」。
   *
   * @param {ArrayLike<number>} x 观测向量
   * @param {any} mask `null` / 每头允许表 / `matchingMask()` 返回值
   * @param {{greedy?:boolean, rng?:()=>number, epsilon?:number, strictMask?:boolean}} [options]
   *   `greedy` 默认 true（argmax）；`epsilon>0` 时以该概率**逐头**均匀随机（只从允许值里抽），
   *   与 v1 的 `--brain-epsilon` 同一探索口径；`strictMask` 时 mask 全关直接抛错。
   * @returns {number[]} 长度 10 的下标数组（顺序 = `HEAD_ORDER_V2`）
   */
  function actWithMask(x, mask = null, options = {}) {
    const opts = options || {};
    const greedy = opts.greedy !== false;
    const rng = typeof opts.rng === 'function' ? opts.rng : Math.random;
    const epsilon = Number(opts.epsilon) > 0 ? Number(opts.epsilon) : 0;
    const strictMask = Boolean(opts.strictMask);
    const masks = normalizeMask(mask, heads);
    const explore = epsilon > 0 && rng() < epsilon;
    const picked = new Array(heads.length);
    // 隐层只算一次：十个头共用它，逐头再算一遍是 O(10×) 的浪费（20Hz 下会拖慢 tick）。
    const allLogits = headLogits(x);
    for (let headIndex = 0; headIndex < heads.length; headIndex += 1) {
      const head = heads[headIndex];
      const logits = allLogits[headIndex];
      let allowed = masks ? masks[headIndex] : null;
      if (allowed) {
        const anyAllowed = allowed.some(Boolean);
        if (!anyAllowed) {
          if (strictMask) {
            fail(`第 ${headIndex} 个头（${head.name}）的 mask 全为 false：没有可执行的动作`);
          }
          maskFallbacks += 1;
          if (!warnedHeads.has(head.name)) {
            warnedHeads.add(head.name);
            console.warn(`[policy] mask 全关（${head.name}）：回退到全开，交给策略自身偏好`);
          }
          allowed = null;
        }
      }
      if (explore) {
        const pool = [];
        for (let value = 0; value < head.size; value += 1) {
          if (!allowed || allowed[value]) {
            pool.push(value);
          }
        }
        picked[headIndex] = pool[Math.min(pool.length - 1, Math.floor(rng() * pool.length))];
        continue;
      }
      const probs = softmaxWithTemperature(logits, temperature);
      if (greedy) {
        // argmax 与 numpy 一致：并列取最小下标。
        let best = -1;
        for (let value = 0; value < head.size; value += 1) {
          if (allowed && !allowed[value]) {
            continue;
          }
          if (best < 0 || probs[value] > probs[best]) {
            best = value;
          }
        }
        picked[headIndex] = best < 0 ? 0 : best;
      } else {
        picked[headIndex] = sampleWithMask(probs, allowed, rng);
      }
    }
    return picked;
  }

  /** v2 没有单头接口：调用即抛错，避免调用方拿一个下标当「整帧动作」。 */
  function notForV2(name) {
    fail(`v2 策略有 ${heads.length} 个头，没有单一 ${name}；请用 actWithMask() / headProbs()`);
  }

  return {
    file,
    actionSpace: 'v2',
    headNames: heads.map((head) => head.name),
    headSizes: heads.map((head) => head.size),
    obsDim,
    numActions,
    hidden,
    temperature,
    meta,
    headLogits,
    headProbs,
    actWithMask,
    get maskFallbacks() {
      return maskFallbacks;
    },
    // 单头接口在 v2 上会抛错：静默返回「某个下标」会丢掉另外 9 个头（v1 调用方必须改）。
    logits: () => notForV2('logits'),
    forward: () => notForV2('forward'),
    greedy: () => notForV2('greedy'),
    sample: () => notForV2('sample'),
  };
}

/** `p = softmax(logits / temperature)`（数值稳定）。 */
function softmaxWithTemperature(logits, temperature) {
  const scale = Number.isFinite(temperature) && temperature > 0 ? temperature : 1;
  const size = logits.length;
  let max = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < size; index += 1) {
    const value = logits[index] / scale;
    if (value > max) {
      max = value;
    }
  }
  const probs = new Array(size);
  let total = 0;
  for (let index = 0; index < size; index += 1) {
    const value = Math.exp(logits[index] / scale - max);
    probs[index] = value;
    total += value;
  }
  if (total <= 0) {
    return new Array(size).fill(1 / size);
  }
  for (let index = 0; index < size; index += 1) {
    probs[index] /= total;
  }
  return probs;
}

/** 在允许集合上按概率抽样（掩掉的项概率视为 0 后重归一）。 */
function sampleWithMask(probs, allowed, rng) {
  let total = 0;
  for (let index = 0; index < probs.length; index += 1) {
    if (!allowed || allowed[index]) {
      total += probs[index];
    }
  }
  if (total <= 0) {
    for (let index = 0; index < probs.length; index += 1) {
      if (!allowed || allowed[index]) {
        return index;
      }
    }
    return 0;
  }
  let threshold = rng() * total;
  let last = 0;
  for (let index = 0; index < probs.length; index += 1) {
    if (allowed && !allowed[index]) {
      continue;
    }
    threshold -= probs[index];
    last = index;
    if (threshold <= 0) {
      return index;
    }
  }
  return last;
}

/** 把 `--brain` 的取值解析成 `.npz` 路径：先按路径试，再按 `<modelsDir>/<id>/policy.npz`、`<id>.npz`。 */
export function resolvePolicyFile(spec, { modelsDir = null, cwd = process.cwd() } = {}) {
  const raw = String(spec || '').trim();
  if (!raw) {
    return null;
  }
  const candidates = [];
  candidates.push(raw);
  if (!raw.match(/[\\/]/) && !raw.endsWith('.npz')) {
    if (modelsDir) {
      candidates.push(path.join(modelsDir, raw, 'policy.npz'));
      candidates.push(path.join(modelsDir, `${raw}.npz`));
    }
    candidates.push(path.join(cwd, raw, 'policy.npz'));
  }
  for (const candidate of candidates) {
    try {
      if (fs.statSync(candidate).isFile()) {
        return candidate;
      }
    } catch {
      // 继续试下一个候选
    }
  }
  return null;
}
