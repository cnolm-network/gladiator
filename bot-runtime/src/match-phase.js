/**
 * 对局阶段（`MATCH_PHASE`）的纯逻辑：归一化、剩余秒数、动作放行表。
 *
 * <p>为什么要单独一个模块：插件从 2026-11-28 起会在阶段变化时（以及倒计时/战斗中每 1 秒）
 * 推一帧 `MATCH_PHASE`，运行时据此决定「这一帧能不能出手」。这条策略必须能单测
 * （`test/match-phase.test.mjs`），而 `index.js` 里全是 mineflayer 与全局状态，测不动。
 *
 * <p>兼容性：**没收到过 `MATCH_PHASE` 的老服务端一切照旧**——{@link phaseGate} 对
 * `null` / 未知阶段一律返回「不限制」，不要因为缺字段把 Bot 冻住。
 */

/** 协议约定的四个阶段（见 docs/protocol.md §2.11）。 */
export const MATCH_PHASES = Object.freeze(['starting', 'countdown', 'fighting', 'ended']);

/** 准备期默认时长（秒），与插件侧 `DEFAULT_COUNTDOWN_SECONDS` 同口径。 */
export const DEFAULT_COUNTDOWN_SECONDS = 5;

/** 准备期允许的动作：只朝向对手 + 站着。其余（move/attack/use/dig/place/jump/talk）全部压掉。 */
const HOLD_ACTION_TYPES = Object.freeze(['look', 'idle']);

/**
 * 把一帧 `MATCH_PHASE.payload` 归一化成内部形态；缺字段/类型不对都不抛异常。
 *
 * @param {object} [payload] 帧的 payload（可能是 `undefined`/`null`/空对象）
 * @param {number} [receivedAt] 收到该帧的本地时间（毫秒），用于 endsAt 缺失时的兜底
 * @returns {{phase: string|null, round: number, seconds: number, endsAt: number, receivedAt: number}}
 *   `phase=null` 表示「没有可用阶段」；`seconds=-1` / `endsAt=0` 表示插件没给。
 */
export function normalizeMatchPhase(payload, receivedAt = Date.now()) {
  const data = payload && typeof payload === 'object' ? payload : {};
  const rawPhase = typeof data.phase === 'string' ? data.phase.trim().toLowerCase() : '';
  const round = Number.isInteger(Number(data.round)) ? Number(data.round) : 0;
  const secondsRaw = Number(data.seconds);
  const endsAtRaw = Number(data.endsAt);
  return {
    phase: rawPhase === '' ? null : rawPhase,
    round,
    // 负数（-1）是「插件不知道」，保留原值让上层区分「0 秒」与「未知」。
    seconds: Number.isFinite(secondsRaw) ? secondsRaw : -1,
    endsAt: Number.isFinite(endsAtRaw) && endsAtRaw > 0 ? endsAtRaw : 0,
    receivedAt: Number.isFinite(Number(receivedAt)) ? Number(receivedAt) : Date.now(),
  };
}

/**
 * 本阶段剩余秒数（向上取整）。
 *
 * <p>优先用 `endsAt` 算：它不依赖帧的到达间隔，掉帧也不会把倒计时算歪；
 * 没有 `endsAt` 时退回帧里的 `seconds`（插件每秒推一帧，够用）。
 *
 * @param {{phase?: string|null, seconds?: number, endsAt?: number}} frame
 * @param {number} [nowMs]
 * @returns {number} 剩余秒数；`-1` = 未知
 */
export function remainingSeconds(frame, nowMs = Date.now()) {
  const data = frame && typeof frame === 'object' ? frame : {};
  const endsAt = Number(data.endsAt);
  const declared = Number(data.seconds);
  if (Number.isFinite(endsAt) && endsAt > 0) {
    const left = Math.ceil((endsAt - nowMs) / 1000);
    const bounded = Math.max(0, left);
    // 声明了总时长就夹一下：endsAt 与 seconds 万一不同步（旧帧/时钟漂移），不要报出超过总时长。
    return Number.isFinite(declared) && declared >= 0 ? Math.min(bounded, declared) : bounded;
  }
  if (Number.isFinite(declared) && declared >= 0) {
    return declared;
  }
  return -1;
}

/**
 * 阶段 → 动作放行表。
 *
 * - `countdown` / `starting`：**禁止进攻**（也只能 look/idle）；
 * - `fighting`：正常决策；
 * - `ended`：全停（连 look 都不发，避免结束后还在空转）；
 * - `null` / 未知阶段：不限制（老服务端兼容）。
 *
 * @param {string|null|undefined} phase
 * @returns {{known: boolean, ended: boolean, hold: boolean, allowAttack: boolean}}
 */
export function phaseGate(phase) {
  const value = typeof phase === 'string' ? phase.trim().toLowerCase() : '';
  if (value === 'countdown' || value === 'starting') {
    return { known: true, ended: false, hold: true, allowAttack: false };
  }
  if (value === 'fighting') {
    return { known: true, ended: false, hold: false, allowAttack: true };
  }
  if (value === 'ended') {
    return { known: true, ended: true, hold: true, allowAttack: false };
  }
  return { known: false, ended: false, hold: false, allowAttack: true };
}

/**
 * 按阶段裁剪一批动作（防御式第二道闸）。
 *
 * <p>正常路径上 `index.js` 在准备期根本不会走策略（直接 early-return），这里是兜底：
 * 任何来源（v1 宏动作、v2 意图、软约束补充）最终都要经过它，避免漏网的一刀。
 *
 * @param {Array<{type: string}>} actions
 * @param {string|null|undefined} phase
 * @returns {Array<{type: string}>}
 */
export function filterPhaseActions(actions, phase) {
  const list = Array.isArray(actions) ? actions : [];
  const gate = phaseGate(phase);
  if (gate.ended) {
    return [];
  }
  if (!gate.hold) {
    return list;
  }
  return list.filter((action) => action && HOLD_ACTION_TYPES.includes(action.type));
}

/**
 * 准备期日志行（每秒一条，由调用方按秒数变化节流）。
 *
 * @param {number} seconds 剩余秒数；负数表示未知
 * @returns {string} 形如 `[brain] 准备期 3s：面向对手待命`
 */
export function countdownLine(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return '[brain] 准备期：面向对手待命';
  }
  return `[brain] 准备期 ${Math.max(0, Math.round(value))}s：面向对手待命`;
}

/**
 * 「面朝对手」的 `look` 动作（yaw 单位是弧度，与 mineflayer 一致）。
 *
 * <p>换算用 {@code atan2(-dx, +dz)}，即 1.8 的 yaw 约定：0 = +Z、π/2 = −X、π = −Z、
 * −π/2 = +X（与 `arena/FlatBoard.yawTowards` 同一公式；`describeFacing` 的 `bearing`
 * 也是这个口径）。注意**不要**照抄 `features.js::yawToTarget`——那个是编码用的向量，
 * 符号与操控口径相反，直接拿去 `bot.look` 会让 Bot 背对对手。
 *
 * <p>pitch 取 0 = 平视：准备期只是站好看着对手，不替策略做瞄准。
 *
 * @param {{x:number,z:number}} selfPos
 * @param {{x:number,z:number}} targetPos
 * @returns {{type:'look', params:{yaw:number, pitch:number}}|null} 拿不到坐标返回 null
 */
export function facingAction(selfPos, targetPos) {
  if (!selfPos || !targetPos) {
    return null;
  }
  const dx = Number(targetPos.x) - Number(selfPos.x);
  const dz = Number(targetPos.z) - Number(selfPos.z);
  if (!Number.isFinite(dx) || !Number.isFinite(dz)) {
    return null;
  }
  if (Math.abs(dx) < 1e-6 && Math.abs(dz) < 1e-6) {
    // 已经站在对手身上：不用转身，避免 atan2(0,0) 把朝向拽到某个固定方向。
    return null;
  }
  return { type: 'look', params: { yaw: Math.atan2(-dx, dz), pitch: 0 } };
}
