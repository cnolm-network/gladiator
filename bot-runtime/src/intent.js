/**
 * 动作空间 v2 的意图执行器：`{type:"intent", intent:{...}}` → mineflayer 调用。
 *
 * <p>背景见 `docs/identity-and-action-space.md` §2.2：v1 的 11 个宏动作把
 * 「选槽 + 右键 + 按住 + 切回剑」写死，加物品就要重训。v2 把这些拆成 10 个正交头
 * （G-707 追加 `sprintReset`），环境只发意图，**怎么按**由 `items.json` 的能力表决定，
 * 因此加物品零改代码。
 *
 * <p>分工与 v1 保持一致：
 * - `move` / `jump` / `sneak` / `sprint` / `look` 直接打 mineflayer 的控制状态；
 * - `attack` 走现有 attack-guard（切剑 + 500ms 无敌帧节流 + reach 判定）；
 * - `slot` 是**唯一的**物品切换入口，`use` 是**唯一的**右键入口；
 * - `BotProfile.assist`（W-tap / 自动鱼竿 / 自动格挡 / 三维对准）仍留在执行层，
 *   不进动作空间——机制不该让策略重新学（规格 §2.5）。
 *
 * <p>本节只处理意图本身，不碰协议帧格式（`link-server.js` 负责解帧）。
 */

import { DEFAULT_ATTACK_COOLDOWN_MS, needsSwordSwap, shouldThrottleAttack } from './attack-guard.js';
import { selectItem } from './items.js';

/** 意图里合法的移动取值（与 `spaces.py::MOVE_VALUES_V2` 同序）。 */
export const MOVE_VALUES = Object.freeze(['idle', 'forward', 'back', 'left', 'right']);
/** 真正要按下的方向键（`idle` 不在其中）。 */
export const MOVE_DIRECTIONS = Object.freeze(['forward', 'back', 'left', 'right']);
/** 意图里合法的 use 取值（`none` 用 `null` 表示）。 */
export const USE_VALUES = Object.freeze([null, 'click', 'hold']);
/**
 * 意图里合法的 `sprintReset` 取值（与 `spaces.py::SPRINT_RESET_VALUES_V2` 同序，G-707）。
 *
 * - `none`：不重置疾跑（缺省，行为与九头时代完全一致）；
 * - `release_w`：**同一帧内**先松 `forward` → 发 attack → 立刻按回 `forward`（wtap）；
 * - `block_hit`：**同一帧内**先举剑（右键按住 300ms）再 attack（防砍 / block-hit）。
 *
 * 组合权在模型手里（ADR-020）：这里**不**做「命中后自动 reset」的隐式合成，
 * 只负责把意图里已经写明的那一步按物理顺序派发出去。
 */
export const SPRINT_RESET_VALUES = Object.freeze(['none', 'release_w', 'block_hit']);
/** `block_hit` 举剑的保持时长（毫秒）。1.8 的「防砍」至少要举过命中帧。 */
export const BLOCK_HIT_HOLD_MS = 300;

/** 一个 tick 的毫秒数（1.8 固定 20 TPS）。 */
export const INTENT_TICK_MS = 50;
/** 单个 tick 允许的偏航增量上限（度）。表里最大 30°，这里兜住手写 intent。 */
export const MAX_DYAWDEG_PER_TICK = 45;
/** 单个 tick 允许的俯仰增量上限（度）。 */
export const MAX_DPITCH_DEG_PER_TICK = 20;
/** 意图停发后多久自动松键（毫秒）。略大于一个控制周期，避免「走一下停一下」。 */
export const MOVE_RELEASE_GRACE_MS = 120;
/** 跳跃脉冲：按住 `jump` 的时长（毫秒）。 */
export const JUMP_PULSE_MS = 60;

const DEG = Math.PI / 180;

/** 每个 Bot 一份执行状态（意图是「流」，按住/计时必须跨 tick 记住）。 */
const STATES = new WeakMap();

function stateFor(bot) {
  let state = STATES.get(bot);
  if (!state) {
    state = {
      pressed: { forward: false, back: false, left: false, right: false },
      moveTimer: null,
      jumpTimer: null,
      useHeld: false,
      useTimer: null,
      lastAttackAt: 0,
      stats: { intents: 0, attacks: 0, throttled: 0, presses: 0, holds: 0, releases: 0 },
    };
    STATES.set(bot, state);
  }
  return state;
}

/** 清掉某个 Bot 的执行状态（断线/重连/测试收尾用）。 */
export function resetIntentState(bot) {
  const state = STATES.get(bot);
  if (!state) {
    return;
  }
  for (const timer of [state.moveTimer, state.jumpTimer, state.useTimer]) {
    if (timer) {
      clearTimeout(timer);
    }
  }
  STATES.delete(bot);
}

/** 只读诊断快照：本局下发过多少意图、节流过多少刀、按住过多少次右键。 */
export function intentStats(bot) {
  const state = STATES.get(bot);
  return state ? { ...state.stats } : { intents: 0, attacks: 0, throttled: 0, presses: 0, holds: 0, releases: 0 };
}

function clamp(value, low, high) {
  if (value < low) {
    return low;
  }
  return value > high ? high : value;
}

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/**
 * 校验并归一一条意图，返回固定形状的对象（缺字段按「不动」补）。
 *
 * @param {object} raw 意图字典
 * @returns {{move:string, look:{dyaw:number, dpitch:number}, attack:boolean,
 *            use:string|null, slot:number|null, jump:boolean, sneak:boolean, sprint:boolean,
 *            sprintReset?:string}}
 */
export function normalizeIntent(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new Error('intent 必须是对象');
  }
  const move = raw.move === undefined || raw.move === null ? 'idle' : String(raw.move).toLowerCase();
  if (!MOVE_VALUES.includes(move)) {
    throw new Error(`intent.move 不认识：${raw.move}（允许 ${MOVE_VALUES.join(' / ')}）`);
  }
  const look = raw.look && typeof raw.look === 'object' ? raw.look : {};
  let use = raw.use === undefined ? null : raw.use;
  if (typeof use === 'string') {
    const text = use.trim().toLowerCase();
    // 兼容 Python 侧可能发来的 "none"（虽然规范用 null）。
    use = text === '' || text === 'none' ? null : text;
  } else if (use !== null) {
    use = null;
  }
  if (use !== null && !USE_VALUES.includes(use)) {
    throw new Error(`intent.use 不认识：${raw.use}（允许 null / click / hold）`);
  }
  let slot = raw.slot === undefined || raw.slot === null ? null : Math.trunc(num(raw.slot, -1));
  if (slot !== null) {
    if (slot < 0 || slot > 8) {
      throw new Error(`intent.slot 越界：${raw.slot}（热键栏 0..8）`);
    }
  }
  // G-707：sprintReset 缺省 = none，且**不写进返回对象**——这样九头时代的 intent
  // 归一化结果逐字段不变（回归口径），只有真的要重置时才多一个键。
  const resetRaw = raw.sprintReset === undefined || raw.sprintReset === null
    ? 'none'
    : String(raw.sprintReset).trim().toLowerCase();
  if (!SPRINT_RESET_VALUES.includes(resetRaw)) {
    throw new Error(
      `intent.sprintReset 不认识：${raw.sprintReset}（允许 ${SPRINT_RESET_VALUES.join(' / ')}）`);
  }
  const normalized = {
    move,
    look: { dyaw: num(look.dyaw, 0), dpitch: num(look.dpitch, 0) },
    attack: Boolean(raw.attack),
    use,
    slot,
    jump: Boolean(raw.jump),
    sneak: Boolean(raw.sneak),
    sprint: Boolean(raw.sprint),
  };
  if (resetRaw !== 'none') {
    normalized.sprintReset = resetRaw;
  }
  return normalized;
}

/** 一行紧凑回显（运行时日志 / 复盘用）。 */
export function describeIntent(raw) {
  let intent;
  try {
    intent = normalizeIntent(raw);
  } catch (error) {
    return `<非法 intent: ${error.message}>`;
  }
  const parts = [`move=${intent.move}`];
  if (intent.look.dyaw !== 0 || intent.look.dpitch !== 0) {
    parts.push(`look=${intent.look.dyaw >= 0 ? '+' : ''}${intent.look.dyaw}`
      + `/${intent.look.dpitch >= 0 ? '+' : ''}${intent.look.dpitch}`);
  }
  if (intent.attack) {
    parts.push('attack=1');
  }
  if (intent.use) {
    parts.push(`use=${intent.use}`);
  }
  if (intent.slot !== null) {
    parts.push(`slot=${intent.slot}`);
  }
  const keys = ['jump', 'sneak', 'sprint'].filter((key) => intent[key]);
  if (keys.length) {
    parts.push(`keys=${keys.join('+')}`);
  }
  if (intent.sprintReset) {
    parts.push(`reset=${intent.sprintReset}`);
  }
  return parts.join(' ');
}

function releaseMove(bot, state) {
  for (const direction of MOVE_DIRECTIONS) {
    try {
      bot.setControlState(direction, false);
    } catch {
      // Bot 掉线时 setControlState 会抛，交给上层的连接处理。
    }
    state.pressed[direction] = false;
  }
}

/** 重新起「没有新意图就松键」的宽限计时（`release_w` 松手后再按回来也要重新计时）。 */
function armMoveRelease(bot, state, graceMs) {
  if (state.moveTimer) {
    clearTimeout(state.moveTimer);
    state.moveTimer = null;
  }
  state.moveTimer = setTimeout(() => {
    state.moveTimer = null;
    releaseMove(bot, state);
  }, Math.max(1, graceMs));
  if (typeof state.moveTimer.unref === 'function') {
    state.moveTimer.unref();
  }
}

/**
 * 应用移动头：与 v1 的 `move` 动作同语义（按住 + 宽限期后自动松开）。
 *
 * <p>意图是每 tick 一条的流，所以这里**不**在结尾松键：同方向连续意图之间保持按住，
 * 只有换方向或超过 `MOVE_RELEASE_GRACE_MS` 没有新意图才松开。
 */
function applyMove(bot, state, move, graceMs) {
  for (const direction of MOVE_DIRECTIONS) {
    const wanted = move === direction;
    if (state.pressed[direction] !== wanted) {
      bot.setControlState(direction, wanted);
      state.pressed[direction] = wanted;
    }
  }
  if (state.moveTimer) {
    clearTimeout(state.moveTimer);
    state.moveTimer = null;
  }
  if (move === 'idle') {
    return;
  }
  armMoveRelease(bot, state, graceMs);
}

/** 应用视角头：**增量**旋转（度 → 弧度），每 tick 各自 clamp。 */
async function applyLook(bot, look) {
  const dyaw = clamp(num(look.dyaw), -MAX_DYAWDEG_PER_TICK, MAX_DYAWDEG_PER_TICK);
  const dpitch = clamp(num(look.dpitch), -MAX_DPITCH_DEG_PER_TICK, MAX_DPITCH_DEG_PER_TICK);
  if (dyaw === 0 && dpitch === 0) {
    return null;
  }
  const entity = bot.entity || {};
  const yaw = num(entity.yaw, 0) + dyaw * DEG;
  // mineflayer 的 pitch 正值向下；±90° 是硬边界（超出服务器会拒绝/回弹）。
  const pitch = clamp(num(entity.pitch, 0) + dpitch * DEG, -Math.PI / 2, Math.PI / 2);
  await bot.look(yaw, pitch, true);
  return { yaw, pitch, dyaw, dpitch };
}

function scheduleJumpRelease(bot, state) {
  if (state.jumpTimer) {
    clearTimeout(state.jumpTimer);
  }
  state.jumpTimer = setTimeout(() => {
    state.jumpTimer = null;
    try {
      bot.setControlState('jump', false);
    } catch {
      // 掉线竞态：忽略
    }
  }, JUMP_PULSE_MS);
  if (typeof state.jumpTimer.unref === 'function') {
    state.jumpTimer.unref();
  }
}

function releaseUse(bot, state) {
  if (!state.useHeld) {
    return false;
  }
  if (state.useTimer) {
    clearTimeout(state.useTimer);
    state.useTimer = null;
  }
  state.useHeld = false;
  state.stats.releases += 1;
  try {
    bot.deactivateItem();
  } catch {
    // 掉线竞态：忽略
  }
  return true;
}

/**
 * 应用右键头。**物理方式由能力表的 mode 决定**（规格 §2.3 / 块 B 第 1 条）：
 *
 * - `click`：一次右键（按下后一个 tick 松开）；
 * - `hold` / `hold_block` / `charge`：按住 `ticks` × 50ms 后松开；
 * - `none`：什么都不做（正常路径下 mask 已经把 use 屏蔽了）。
 *
 * <p>意图里的 `use=click` 对 hold 类物品仍然有意义：只按一个 tick，等于「点一下就撤」
 * （例如举盾一下就走）。`use=hold` 才按表里的 `ticks` 走完。
 */
async function applyUse(bot, state, use, table) {
  if (use === null) {
    // 不按 = 立即松手，允许策略中途取消按住（比如吃到一半被打断）。
    return { released: releaseUse(bot, state) };
  }
  const held = bot.heldItem;
  const name = held && typeof held.name === 'string' ? held.name : '';
  const spec = table && typeof table.useSpec === 'function'
    ? table.useSpec(name)
    : { mode: 'none', ticks: 0 };
  const mode = String(spec.mode || 'none');
  if (!name || mode === 'none') {
    return { mode, held: name || null, skipped: 'no-usable-item' };
  }
  if (mode === 'click') {
    releaseUse(bot, state);
    bot.activateItem(Boolean(spec.offhand));
    state.stats.presses += 1;
    // 停一个 tick 再松：1.8 的 use_item 与 release_use_item 同 tick 发送时，
    // 服务端偶尔只处理前者，鱼竿会「甩出去但没收线」。
    await new Promise((resolve) => setTimeout(resolve, INTENT_TICK_MS));
    bot.deactivateItem();
    return { mode, held: name, ticks: 1 };
  }
  if (state.useHeld) {
    // 已经是按住状态：不重启计时，否则连续 hold 意图会让它永远不松手。
    return { mode, held: name, holding: true };
  }
  bot.activateItem(Boolean(spec.offhand));
  state.useHeld = true;
  state.stats.holds += 1;
  const ticks = use === 'click' ? 1 : Math.max(1, Math.trunc(num(spec.ticks, 1)));
  const holdMs = ticks * INTENT_TICK_MS;
  state.useTimer = setTimeout(() => {
    state.useTimer = null;
    releaseUse(bot, state);
  }, holdMs);
  if (typeof state.useTimer.unref === 'function') {
    state.useTimer.unref();
  }
  return { mode, held: name, ticks, holdMs };
}

/**
 * `sprintReset = block_hit`：举剑（防砍）。
 *
 * <p>复用 `use{sword, holdMs=300}` 那条既有路径：先按 `items.js` 切到剑
 * （手里不是剑时 `needsSwordSwap` 才动背包），再 `activateItem`，300ms 后自动松开。
 * 与 `applyUse` 共用 `state.useHeld` / `state.useTimer`，因此：
 * - 本帧随后若还有 `use` 意图，不会再 `activateItem` 一次（不会把举剑点掉）；
 * - `use: null` 或下一个 tick 的普通 hold 仍能正常接管，计时也不会叠成「永不松手」。
 *
 * <p>举剑与出手在**同一次 applyIntent 调用里**完成（先举后砍），这正是 1.8「防砍」
 * 的物理顺序：命中帧手持剑且处于格挡姿态，吃 0.5× 伤害并重置疾跑。
 */
async function applyBlockHit(bot, state, table) {
  if (needsSwordSwap(bot.heldItem)) {
    try {
      await selectItem(bot, 'sword');
    } catch {
      // 没剑也照样出手：空手挥击的伤害由服务端决定，不该吞掉意图。
    }
  }
  const held = bot.heldItem;
  const name = held && typeof held.name === 'string' ? held.name : '';
  const spec = table && typeof table.useSpec === 'function'
    ? table.useSpec(name)
    : { mode: 'none' };
  const mode = String(spec.mode || 'none');
  if (!name || mode === 'none') {
    return { reset: 'block_hit', held: name || null, skipped: 'no-usable-item' };
  }
  if (!state.useHeld) {
    bot.activateItem(Boolean(spec.offhand));
    state.useHeld = true;
    state.stats.holds += 1;
  }
  if (state.useTimer) {
    clearTimeout(state.useTimer);
    state.useTimer = null;
  }
  state.useTimer = setTimeout(() => {
    state.useTimer = null;
    releaseUse(bot, state);
  }, BLOCK_HIT_HOLD_MS);
  if (typeof state.useTimer.unref === 'function') {
    state.useTimer.unref();
  }
  return { reset: 'block_hit', held: name, mode, holdMs: BLOCK_HIT_HOLD_MS };
}

/** 找最近的其他玩家（战斗目标的唯一来源；意图里不带 entityId）。 */
export function nearestOpponent(bot) {
  const self = bot.entity && bot.entity.position;
  const selfName = String(bot.username || '').toLowerCase();
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const key of Object.keys(bot.entities || {})) {
    const entity = bot.entities[key];
    if (!entity || entity === bot.entity || !entity.position) {
      continue;
    }
    const kind = entity.type || entity.kind;
    if (kind !== 'player') {
      continue;
    }
    const name = String(entity.username || entity.name || '');
    if (!name || name.toLowerCase() === selfName) {
      continue;
    }
    const distance = self && typeof self.distanceTo === 'function'
      ? self.distanceTo(entity.position)
      : num(entity.distance, Number.POSITIVE_INFINITY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = entity;
    }
  }
  return best ? { entity: best, distance: bestDistance } : null;
}

/**
 * 应用攻击头：先过 attack-guard（切剑 + 500ms 节流 + reach），再交给 dispatcher。
 *
 * <p>节流用 `streakLimit: 1`：`shouldThrottleAttack` 的默认实现要连续 3 次才生效，
 * 那是给「宏动作 + 手写大脑」留的余量；v2 的意图每 tick 都可能带 attack，
 * 从第一次重复就该按 1.8 的受击无敌（500ms）压住。
 */
async function applyAttack(bot, state, ctx) {
  const now = typeof ctx.now === 'function' ? ctx.now() : Date.now();
  const cooldownMs = Number.isFinite(Number(ctx.attackCooldownMs))
    ? Number(ctx.attackCooldownMs)
    : DEFAULT_ATTACK_COOLDOWN_MS;
  if (shouldThrottleAttack({
    streak: 1,
    streakLimit: 1,
    lastAttackAt: state.lastAttackAt,
    now,
    cooldownMs,
  })) {
    state.stats.throttled += 1;
    return { attack: 'throttled', remainingMs: Math.max(0, cooldownMs - (now - state.lastAttackAt)) };
  }
  const target = nearestOpponent(bot);
  if (!target) {
    return { attack: 'no-target' };
  }
  const dispatcher = ctx.dispatcher;
  if (dispatcher && typeof dispatcher.attack === 'function') {
    // 复用 v1 的执行层：reach 判定、够不到先追一步、自动切剑、CPS 节奏都在里面。
    const result = await dispatcher.attack({ targetEntityId: target.entity.id });
    state.lastAttackAt = now;
    state.stats.attacks += 1;
    return { attack: 'hit', code: result && result.code, distance: target.distance };
  }
  if (needsSwordSwap(bot.heldItem)) {
    try {
      await selectItem(bot, 'sword');
    } catch {
      // 没有剑也让它挥下去：空手伤害由服务端决定，不要在这里吞掉意图。
    }
  }
  bot.attack(target.entity);
  state.lastAttackAt = now;
  state.stats.attacks += 1;
  return { attack: 'hit', distance: target.distance };
}

/**
 * 执行一条意图。
 *
 * @param {object} bot     mineflayer Bot（测试里可以是假对象）
 * @param {object} intent  意图字典（见 `spaces.py::decode_intent`）
 * @param {object} [ctx]   `{dispatcher, itemTable, now, attackCooldownMs, moveGraceMs}`
 * @returns {Promise<{ok:true, type:'intent', applied:object}>} 供 ACTION_RESULT / 日志回显
 */
export async function applyIntent(bot, intent, ctx = {}) {
  if (!bot) {
    throw new Error('applyIntent 需要 bot 实例');
  }
  const action = normalizeIntent(intent);
  const state = stateFor(bot);
  const applied = {};

  // 1) 选槽：唯一的换物品入口。必须在 use 之前，否则「换槽 + 用」会用到旧物品。
  if (action.slot !== null && typeof bot.setQuickBarSlot === 'function') {
    if (Number(bot.quickBarSlot) !== action.slot) {
      await bot.setQuickBarSlot(action.slot);
      applied.slot = action.slot;
    }
  }

  // 2) 视角：增量旋转，让策略自己学瞄准。
  const look = await applyLook(bot, action.look);
  if (look) {
    applied.look = { dyaw: look.dyaw, dpitch: look.dpitch };
  }

  // 3) 移动 + 蹲/疾跑/跳（与 move 正交）。
  const graceMs = Number.isFinite(Number(ctx.moveGraceMs))
    ? Number(ctx.moveGraceMs)
    : MOVE_RELEASE_GRACE_MS;
  applyMove(bot, state, action.move, graceMs);
  if (action.move !== 'idle') {
    applied.move = action.move;
  }
  if (action.sneak !== Boolean(bot.controlState && bot.controlState.sneak)) {
    bot.setControlState('sneak', action.sneak);
    applied.sneak = action.sneak;
  }
  if (action.sprint !== Boolean(bot.controlState && bot.controlState.sprint)) {
    bot.setControlState('sprint', action.sprint);
    applied.sprint = action.sprint;
  }
  if (action.jump) {
    bot.setControlState('jump', true);
    scheduleJumpRelease(bot, state);
    applied.jump = true;
  }

  // 3.5) sprintReset = release_w（G-707）：**攻击前**松 W 再按回，把疾跑速度重置回最大值。
  // 依据 `docs/knowledge/pvp-1.8-mechanics.md` §2.5（用户整理的 `PVP_overview.md` 正本）：
  // 「疾跑速度越大，产生的击退越大；第一次疾跑攻击后速度会降下来——一次疾跑的第一次攻击最大」，
  // 所以 reset 的目的是让**这一刀**吃满疾跑速度 → 对手吃更大 KB（配上高低差就是 combo），
  // 而不是把这一刀变成非疾跑小 KB（那是早期理解，已纠正）。
  // 顺序必须是 release → re-press → attack，全部在同一帧内完成。
  const reset = action.sprintReset || 'none';
  const holdingForward = action.move === 'forward';
  let releasedForward = false;
  let didRelease = false;
  if (reset === 'release_w' && holdingForward) {
    bot.setControlState('forward', false);
    state.pressed.forward = false;
    releasedForward = true;
    didRelease = true;
  }
  if (reset === 'release_w' && releasedForward) {
    // 立刻按回：这一对 false→true 就是「重置疾跑速度」本身。
    bot.setControlState('forward', true);
    state.pressed.forward = true;
    armMoveRelease(bot, state, graceMs);
    releasedForward = false;
  }
  // 3.6) 疾跑位同样要走一次 false→true：mineflayer 只在 controlState.sprint 变化时才发
  // START_SPRINTING，而 1.8 的 sprint 是服务端状态位（命中后清掉），客户端状态还是 true，
  // 于是「重置」不会真的生效。只在意图本身要求疾跑时才动，绝不替模型做决定。
  if (reset === 'release_w' && action.sprint) {
    bot.setControlState('sprint', false);
    bot.setControlState('sprint', true);
    applied.sprint = true;
    applied.sprintResent = true;
  }

  // 4) 右键：能力表决定按多久。
  const useResult = await applyUse(bot, state, action.use, ctx.itemTable);
  if (useResult && (useResult.mode || useResult.released)) {
    applied.use = useResult;
  }

  // 4.5) sprintReset = block_hit：同一帧内先举剑，紧接的 5) 再出手。
  if (reset === 'block_hit') {
    applied.sprintReset = await applyBlockHit(bot, state, ctx.itemTable);
  }

  // 5) 左键：attack-guard（切剑 + 500ms 节流）。
  if (action.attack) {
    applied.attack = await applyAttack(bot, state, ctx);
  }

  // 6) release_w 回报：reset 已经在 attack 之前完成（release → re-press → attack）。
  if (reset === 'release_w') {
    applied.sprintReset = applied.sprintReset
      || { reset: 'release_w', released: didRelease ? 'forward' : null, restored: didRelease };
    if (applied.sprintResent) {
      applied.sprintReset.sprintResent = true;
    }
  }

  state.stats.intents += 1;
  return { ok: true, type: 'intent', applied };
}
