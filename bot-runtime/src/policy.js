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

/**
 * 加载策略。
 *
 * @param {string} file `.npz` 路径
 * @returns {{obsDim:number, numActions:number, hidden:number, temperature:number,
 *            file:string, logits:(x:ArrayLike<number>)=>number[],
 *            forward:(x:ArrayLike<number>)=>number[], greedy:(x:ArrayLike<number>)=>number,
 *            sample:(x:ArrayLike<number>, random?:()=>number)=>number}}
 */
export function loadPolicy(file) {
  const entries = readNpz(file);
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

  // `_meta` 是 np.array(json_string)，dtype 为 <U…，shape 为 ()。
  let meta = {};
  const metaEntry = entries.get('_meta');
  if (metaEntry && metaEntry.text) {
    try {
      meta = JSON.parse(metaEntry.text);
    } catch (error) {
      fail(`模型 _meta 不是合法 JSON: ${error.message}`);
    }
  }
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
