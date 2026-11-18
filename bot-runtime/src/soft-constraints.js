/**
 * 软约束：把「Bot 能不能用某个招式」做成**运行参数**，而不是写进模型。
 *
 * <p>为什么是运行参数：同一个 `.npz` 会在不同场合被打（人机对战、演示、评测）。
 * 用户想临时禁用鱼竿/金苹果/格挡/跳跃，只需改命令行或前端表单，**模型文件一个字不动**，
 * 评测也能拿来当「消融开关」。写进模型会让这些选择变成不可逆的训练事实。
 *
 * <p>与硬约束的分工：
 * <ul>
 *   <li>硬约束（`--cps`、`--reach`、动作空间 11 个宏动作）决定「物理上能做什么」，
 *       改它要动 profile / 模型；</li>
 *   <li>软约束只做**事后剔除**：策略选了被禁的宏动作，就在这里把它换成 idle/保留位移，
 *       不改变动作空间大小、不改变观测与模型前向。</li>
 * </ul>
 *
 * <p>用法（逗号分隔）：
 * <pre>
 *   --soft noRod,noGapple
 *   --soft rod=off,gapple=off,cps=12
 *   --soft cps=12,reachOffset=0.5
 * </pre>
 */

import { MOVE_MS } from './features.js';

/** 全部允许的默认值：软约束的语义就是「默认全开，按需关闭」。 */
export const DEFAULT_SOFT = Object.freeze({
  allowRod: true,
  allowGapple: true,
  allowBlock: true,
  allowJumpCrit: true,
  //: 每秒点击上限（null = 不限制）。只用于节流，不改变动作集合。
  maxCps: null,
  //: 攻击距离至少留出的余量（null = 不预留）。
  minReachOffset: null,
});

/**
 * `block` 在协议里是 `use{sword, holdMs:300}`（见 features.js 的 `block` 分支），
 * 与「切剑」的 `use{sword, holdMs:50}` 共用 item 名，只能按 holdMs 区分。
 * 300 = 格挡；50 = 切剑（必须放行，否则攻击前的换剑会被自己禁掉）。
 */
export const BLOCK_HOLD_MS = 300;

/** 键别名（全部小写、去掉 `-`/`_` 后比较）。 */
const KEY_ALIASES = Object.freeze({
  rod: 'allowRod',
  allowrod: 'allowRod',
  userod: 'allowRod',
  gapple: 'allowGapple',
  apple: 'allowGapple',
  goldenapple: 'allowGapple',
  allowgapple: 'allowGapple',
  block: 'allowBlock',
  swordblock: 'allowBlock',
  allowblock: 'allowBlock',
  jump: 'allowJumpCrit',
  jumpcrit: 'allowJumpCrit',
  crit: 'allowJumpCrit',
  allowjumpcrit: 'allowJumpCrit',
  cps: 'maxCps',
  maxcps: 'maxCps',
  reachoffset: 'minReachOffset',
  minreachoffset: 'minReachOffset',
});

const BOOL_KEYS = new Set(['allowRod', 'allowGapple', 'allowBlock', 'allowJumpCrit']);
const NUMBER_KEYS = new Set(['maxCps', 'minReachOffset']);

/** 用户可读的键名（报错与日志用）。 */
const KEY_LABELS = Object.freeze({
  allowRod: 'rod',
  allowGapple: 'gapple',
  allowBlock: 'block',
  allowJumpCrit: 'jump',
  maxCps: 'cps',
  minReachOffset: 'reachOffset',
});

const TRUE_WORDS = new Set(['1', 'true', 'yes', 'on', 'enable', 'enabled']);
const FALSE_WORDS = new Set(['0', 'false', 'no', 'off', 'disable', 'disabled']);

function normalizeToken(text) {
  return String(text ?? '').toLowerCase().replace(/[-_\s]+/g, '');
}

function parseBoolWord(text) {
  const value = normalizeToken(text);
  if (TRUE_WORDS.has(value)) return true;
  if (FALSE_WORDS.has(value)) return false;
  return null;
}

function fail(message) {
  return { ok: false, error: message };
}

/**
 * 解析软约束串。
 *
 * 支持三种写法（可混用，逗号分隔）：
 * - `name`          —— 布尔开（`rod`）；
 * - `no<Name>`      —— 布尔关（`noRod` / `nogapple`）；
 * - `name=value`    —— 布尔 `on/off/true/false/1/0`，数值 `cps=12`、`reachOffset=0.5`。
 *
 * @param {string} raw 空串/undefined = 全部允许。
 * @returns {{ok:true, soft:object, entries:Array<{key:string,value:any,token:string}>}
 *          | {ok:false, error:string}} 非法键/值返回 `{ok:false,error}`，不抛异常。
 */
export function parseSoft(raw) {
  const soft = { ...DEFAULT_SOFT };
  const entries = [];
  const text = String(raw ?? '').trim();
  if (!text) {
    return { ok: true, soft, entries };
  }
  for (const chunk of text.split(',')) {
    const token = chunk.trim();
    if (!token) continue;

    let name = token;
    let value = null;                 // null = 没写 `=value`
    const eq = token.indexOf('=');
    if (eq >= 0) {
      name = token.slice(0, eq).trim();
      value = token.slice(eq + 1).trim();
    }
    const normalized = normalizeToken(name);
    if (!normalized) {
      return fail(`软约束里有空键（${token}）`);
    }

    // `noRod` / `no-rod`：布尔取反；对数值键（noCps）没有意义。
    let negated = false;
    let key = KEY_ALIASES[normalized];
    if (key === undefined && normalized.startsWith('no')) {
      const base = KEY_ALIASES[normalized.slice(2)];
      if (base !== undefined) {
        if (NUMBER_KEYS.has(base)) {
          return fail(`${name} 没有意义（${KEY_LABELS[base]} 是数值键，请用 `
            + `${KEY_LABELS[base]}=值）`);
        }
        key = base;
        negated = true;
      }
    }
    if (key === undefined) {
      return fail(`未知软约束键: ${name}`
        + `（可用: rod, gapple, block, jump, cps, reachOffset；取反写 noRod）`);
    }
    if (negated && value !== null) {
      return fail(`${name} 不能同时带值（写成 ${KEY_LABELS[key]}=off）`);
    }

    if (BOOL_KEYS.has(key)) {
      let flag;
      if (negated) {
        flag = false;
      } else if (value === null) {
        flag = true;                  // 裸写键名 = 打开
      } else {
        flag = parseBoolWord(value);
        if (flag === null) {
          return fail(`${KEY_LABELS[key]} 的值 "${value}" 不是布尔（on/off/true/false/1/0）`);
        }
      }
      soft[key] = flag;
      entries.push({ key, value: flag, token });
      continue;
    }

    // 数值键
    if (negated) {
      return fail(`${name} 没有意义（数值键请用 ${KEY_LABELS[key]}=值）`);
    }
    if (value === null) {
      return fail(`${KEY_LABELS[key]} 需要值（例如 ${KEY_LABELS[key]}=12）`);
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return fail(`${KEY_LABELS[key]} 的值 ${value} 不是数字`);
    }
    if (key === 'maxCps') {
      if (number < 1 || number > 40) {
        return fail(`cps 需要在 1..40 之间（收到 ${number}）`);
      }
      soft.maxCps = number;
    } else {
      if (number < 0 || number > 3) {
        return fail(`reachOffset 需要在 0..3 之间（收到 ${number}）`);
      }
      soft.minReachOffset = number;
    }
    entries.push({ key, value: soft[key], token });
  }
  return { ok: true, soft, entries };
}

/** 中文摘要（启动日志 / 前端回显用）。 */
export function describeSoft(soft) {
  const value = { ...DEFAULT_SOFT, ...(soft || {}) };
  const disabled = [];
  if (value.allowRod === false) disabled.push('鱼竿');
  if (value.allowGapple === false) disabled.push('金苹果');
  if (value.allowBlock === false) disabled.push('格挡');
  if (value.allowJumpCrit === false) disabled.push('跳跃暴击');
  const parts = [disabled.length ? `禁用 ${disabled.join('、')}` : '动作全开'];
  if (value.maxCps != null) parts.push(`CPS 上限 ${value.maxCps}`);
  if (value.minReachOffset != null) parts.push(`攻击距离预留 ${value.minReachOffset} 格`);
  return parts.join('；');
}

/**
 * 这条动作是否被软约束禁掉。
 *
 * @returns {string|null} 被禁原因（`rod`/`gapple`/`block`/`jump`），允许则 `null`。
 */
export function softDropReason(action, soft) {
  const value = soft || DEFAULT_SOFT;
  if (!action || typeof action !== 'object') return null;
  const type = String(action.type || '');
  const params = action.params || {};
  if (type === 'jump') {
    return value.allowJumpCrit === false ? 'jump' : null;
  }
  if (type !== 'use') return null;
  const item = String(params.item || '').toLowerCase();
  if (item === 'rod') {
    return value.allowRod === false ? 'rod' : null;
  }
  if (item === 'gapple') {
    return value.allowGapple === false ? 'gapple' : null;
  }
  if (item === 'block') {
    // 协议里没有 item:"block"，保留这个分支兼容未来的写法。
    return value.allowBlock === false ? 'block' : null;
  }
  if (item === 'sword') {
    const holdMs = Number(params.holdMs ?? 0);
    if (holdMs >= BLOCK_HOLD_MS) {
      return value.allowBlock === false ? 'block' : null;
    }
  }
  return null;
}

/**
 * 剔除被禁用的宏动作，返回**新数组**（不修改入参）。
 *
 * 规则：只要有动作被剔除，就保证这一 tick 至少有一个真正的动作——
 * 全被剔光时补 `idle`；只剩 `look`（转身）时也在后面补一个 `idle`，
 * 免得这一 tick 空转、观感上像卡住。
 *
 * 注意 `maxCps` **不**在这里生效：它只做节流（`index.js` 换算成攻击冷却），
 * 不改变动作集合。
 */
export function applySoftActions(actions, soft) {
  const list = Array.isArray(actions) ? actions : [];
  const value = soft || DEFAULT_SOFT;
  const kept = [];
  let removed = 0;
  for (const action of list) {
    if (softDropReason(action, value) !== null) {
      removed += 1;
      continue;
    }
    kept.push(action);
  }
  if (removed === 0) {
    return kept;
  }
  const hasBodyAction = kept.some((action) => String(action?.type || '') !== 'look');
  if (!hasBodyAction) {
    kept.push({ type: 'idle', params: { durationMs: MOVE_MS } });
  }
  return kept;
}
