/**
 * 大脑运行期遥测（G-717）：心跳行、目标距离、远距离提示节流、目标离线状态机。
 *
 * <p>全部不依赖 mineflayer（只吃纯数据），所以可以单测；`index.js` 只负责把
 * `bot` / `brainState` 里的值喂进来。抽出来的动机：真机上「Bot 不动也不打」有三类
 * 完全不同的原因（在赶路 / 等套件 / 目标不在线），以前心跳里看不出来，只能靠猜。
 */

/** 1.8 各护甲件的防御点数（与 `/gladiator stats` 的 `armor=` 同口径，钻石全套 = 3+8+6+3 = 20）。 */
export const ARMOR_POINTS = Object.freeze({
  leather_helmet: 1, leather_chestplate: 3, leather_leggings: 2, leather_boots: 1,
  golden_helmet: 2, golden_chestplate: 5, golden_leggings: 3, golden_boots: 1,
  chainmail_helmet: 2, chainmail_chestplate: 5, chainmail_leggings: 4, chainmail_boots: 1,
  iron_helmet: 2, iron_chestplate: 6, iron_leggings: 5, iron_boots: 2,
  diamond_helmet: 3, diamond_chestplate: 8, diamond_leggings: 6, diamond_boots: 3,
});

/** 1.8 背包里的护甲槽（head / torso / legs / feet）。 */
export const ARMOR_SLOTS = Object.freeze([5, 6, 7, 8]);

/**
 * 已穿戴护甲的防御点数（0–20）。
 *
 * @param {Array<{slot:number, name:string}>} items `bot.inventory.items()` 的形态（带 slot）
 * @returns {number}
 */
export function armorPoints(items) {
  const list = Array.isArray(items) ? items : [];
  let total = 0;
  for (const item of list) {
    if (!item || !ARMOR_SLOTS.includes(Number(item.slot))) {
      continue;
    }
    const points = ARMOR_POINTS[String(item.name || '')];
    if (Number.isFinite(points)) {
      total += points;
    }
  }
  return total;
}

/** 水平距离（只算 x/z：跳跃/台阶造成的 y 差不算「远近」）。读不到返回 null。 */
export function horizontalDistance(from, to) {
  if (!from || !to) {
    return null;
  }
  const dx = Number(to.x) - Number(from.x);
  const dz = Number(to.z) - Number(from.z);
  if (!Number.isFinite(dx) || !Number.isFinite(dz)) {
    return null;
  }
  return Math.sqrt(dx * dx + dz * dz);
}

/**
 * 拼一行心跳。
 *
 * <p>判读方式（真机排障顺序）：
 * - `decisions` 不涨 + `busy=1` → 有动作卡住，看 `last=`；
 * - `decisions` 不涨 + `notReady` 涨 → 没 Bot / 没会话；
 * - `dist` 远 → 在赶路（不是不决策）；`dist` 近但 `attacks` 不涨 → 模型没选攻击；
 * - `waitingKit` 涨 → 板子还没发套件（不是模型问题）。
 *
 * @param {object} s   brainState 的快照 + `dist` / `armor` / `lastTickMs`
 * @param {number} nowMs
 */
export function formatHeartbeat(s, nowMs) {
  const state = s || {};
  const num = (value) => (Number.isFinite(Number(value)) ? Number(value) : 0);
  const lastAction = state.lastAction
    ? `${state.lastAction}(${Math.max(0, nowMs - num(state.lastActionAt))}ms)`
    : 'idle';
  const lastTick = state.lastTickAt ? `${nowMs - num(state.lastTickAt)}ms` : 'never';
  const dist = state.dist === null || state.dist === undefined || !Number.isFinite(Number(state.dist))
    ? '-'
    : Number(state.dist).toFixed(1);
  const parts = [
    `decisions=${num(state.decisions)}`,
    `attacks=${num(state.attacks)}`,
    `failures=${num(state.actionFailures)}`,
    `waitingKit=${num(state.waitingKit)}`,
    `timeouts=${num(state.actionTimeouts)}`,
    `busy=${state.busy ? 1 : 0}`,
    `skips=${num(state.busySkips)}`,
    `notReady=${num(state.notReadySkips)}`,
    `dist=${dist}`,
    `armor=${num(state.armor)}`,
    `last=${lastAction}`,
    `lastTick=${lastTick}`,
    `target=${state.target || '-'}`,
    `oppHp=${state.opponentHealth === null || state.opponentHealth === undefined
      ? '-' : state.opponentHealth}`,
  ];
  if (state.lastError) {
    parts.push(`lastError=${state.lastError}`);
  }
  return `[brain] 心跳 ${parts.join(' ')}`;
}

/**
 * 目标在线状态机（C）。
 *
 * <p>`--target-player` 指定的玩家从 tab 列表里消失超过 `offlineAfterMs`（默认 5s）时
 * 报一次「不在线」，回来报一次「回来了」；中间无论跳多少次都只报这两次。
 * 用「消失就立刻报」会在他传送/重连的 1 秒里刷屏，所以必须带迟滞。
 */
export class TargetPresence {
  constructor({ offlineAfterMs = 5000, name = '' } = {}) {
    this.offlineAfterMs = Math.max(0, Number(offlineAfterMs) || 0);
    this.name = name;
    this.missingSince = null;
    this.reportedOffline = false;
  }

  /** 新的一场对局：状态清零（否则上一局的「不在线」会带过来）。 */
  reset(name = this.name) {
    this.name = name;
    this.missingSince = null;
    this.reportedOffline = false;
  }

  /**
   * @param {number} nowMs
   * @param {boolean} online 目标是否在玩家列表里
   * @returns {'offline'|'back'|null} 需要打日志的事件
   */
  update(nowMs, online) {
    if (online) {
      const wasOffline = this.reportedOffline;
      this.missingSince = null;
      this.reportedOffline = false;
      return wasOffline ? 'back' : null;
    }
    if (this.missingSince === null) {
      this.missingSince = nowMs;
    }
    if (this.reportedOffline) {
      return null;
    }
    if (nowMs - this.missingSince < this.offlineAfterMs) {
      return null;
    }
    this.reportedOffline = true;
    return 'offline';
  }
}

/**
 * 远距离「赶路中」提示（B）：`dist > threshold` 时每 `everyMs` 报一条，进圈后重置。
 *
 * <p>为什么需要：玩家与 Bot 相距上千格时，模型会一路 `approach`，日志里看着像「不决策」；
 * 这条把「在赶路」和「贴身不打」区分开。**不做任何动作抑制**（打不打由模型决定，ADR-020）。
 */
export class FarChaseNotifier {
  constructor({ threshold = 24, everyMs = 5000 } = {}) {
    this.threshold = Number(threshold) > 0 ? Number(threshold) : 24;
    this.everyMs = Math.max(0, Number(everyMs) || 0);
    this.lastReportAt = 0;
  }

  reset() {
    this.lastReportAt = 0;
  }

  /**
   * @param {number} nowMs
   * @param {number|null} dist 当前水平距离
   * @param {number|null} maxDist 本局最远距离
   * @returns {string|null} 需要打的日志文本（不需要打时 null）
   */
  update(nowMs, dist, maxDist) {
    if (!Number.isFinite(Number(dist)) || Number(dist) <= this.threshold) {
      // 回到 24 格内：下次再拉开时立刻能报一条。
      this.lastReportAt = 0;
      return null;
    }
    if (this.lastReportAt && nowMs - this.lastReportAt < this.everyMs) {
      return null;
    }
    this.lastReportAt = nowMs;
    const current = Math.round(Number(dist));
    const farthest = Math.round(Math.max(Number(dist), Number(maxDist) || 0));
    return `[brain] 目标在 ${current} 格之外，赶路中（本局最长 ${farthest} 格）`;
  }
}
