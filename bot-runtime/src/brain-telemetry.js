/**
 * 大脑运行期遥测（G-717）：心跳行、目标距离、远距离提示节流、目标离线状态机；
 * G-721 追加「目标不可见」检测（盲决策告警）。
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
 * - `blind` 涨 → 大脑认为有目标但观测里 `has_target=0`（盲决策，G-721），看 `entities=`：
 *   1 = 世界没同步进来 / 客户端坐标还停在旧位置；
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
  // G-735：`lastTick` 的含义从「距上次决策多久」改成「整帧决策耗时」——拆开决策/执行之后，
  // 「距上次」恒等于 50ms（定时器周期），没有信息量；真正要看的是这一帧算得多慢。
  // 仍兼容旧调用方：没给 `lastTick` 就用 `lastTickAt` 推「距上次」。
  const frameMs = Number(state.lastTick);
  const lastTick = Number.isFinite(frameMs) && frameMs >= 0
    ? `${Math.round(frameMs)}ms`
    : (state.lastTickAt ? `${nowMs - num(state.lastTickAt)}ms` : 'never');
  const dist = state.dist === null || state.dist === undefined || !Number.isFinite(Number(state.dist))
    ? '-'
    : Number(state.dist).toFixed(1);
  const parts = [
    `decisions=${num(state.decisions)}`,
    // G-735：最近 10 秒的实际决策率（Hz）。20Hz 目标下这条直接回答「有效决策率够不够」。
    ...(state.hz === null || state.hz === undefined
      ? []
      : [`hz=${Number(state.hz).toFixed(1)}`]),
    `attacks=${num(state.attacks)}`,
    `failures=${num(state.actionFailures)}`,
    `waitingKit=${num(state.waitingKit)}`,
    `timeouts=${num(state.actionTimeouts)}`,
    `busy=${state.busy ? 1 : 0}`,
    // 被顶掉的待执行计划数（队列容量 1）：涨得快说明执行比决策慢得多——这是**设计如此**，
    // 只要 `hz` 达标就不是问题。
    ...(state.superseded === null || state.superseded === undefined
      ? []
      : [`superseded=${num(state.superseded)}`]),
    `skips=${num(state.busySkips)}`,
    `notReady=${num(state.notReadySkips)}`,
    // G-721：`blind` 涨 = 大脑以为有目标、观测里 has_target 却是 0（盲决策帧数）；
    // `entities` = 最近一帧 mineflayer 的原始实体数（含自己），两者一起看能立刻区分
    // 「世界没同步进来（entities=1）」与「目标名字对不上（entities=N 但 blind 仍涨）」。
    `blind=${num(state.blind)}`,
    `entities=${num(state.entitiesSeen)}`,
    `dist=${dist}`,
    `armor=${num(state.armor)}`,
    `last=${lastAction}`,
    // G-735：分段耗时。`obsMs` 观测+编码、`inferMs` 前向、`applyMs` 上一次计划执行耗时、
    // `applyMsMean` 最近 50 次均值。四个数一起看就知道瓶颈在「算」还是在「执行」。
    ...(state.obsMs === null || state.obsMs === undefined ? [] : [`obsMs=${num(state.obsMs)}`]),
    ...(state.inferMs === null || state.inferMs === undefined ? [] : [`inferMs=${num(state.inferMs)}`]),
    ...(state.applyMs === null || state.applyMs === undefined ? [] : [`applyMs=${num(state.applyMs)}`]),
    ...(state.applyMsMean === null || state.applyMsMean === undefined
      ? []
      : [`applyMsMean=${Number(state.applyMsMean).toFixed(1)}`]),
    `lastTick=${lastTick}`,
    `target=${state.target || '-'}`,
    `oppHp=${state.opponentHealth === null || state.opponentHealth === undefined
      ? '-' : state.opponentHealth}`,
  ];
  // G-723：朝向诊断。只在有值时才加，老调用方/老测试的期望行不受影响。
  if (state.yaw !== null && state.yaw !== undefined) {
    parts.push(`yaw=${Number(state.yaw).toFixed(1)}`);
  }
  if (state.bearing !== null && state.bearing !== undefined) {
    parts.push(`bearing=${Number(state.bearing).toFixed(1)}`);
  }
  if (state.yawErr !== null && state.yawErr !== undefined) {
    parts.push(`yawErr=${Number(state.yawErr).toFixed(1)}`);
  }
  // G-723：世界同步诊断——`raw` 是 mineflayer 跟踪到的实体数（含自己），
  // `seen` 表示目标实体是否在本地实体表里。`seen=0 && raw<=1` 就是「客户端还没同步」。
  if (state.rawEntities !== null && state.rawEntities !== undefined) {
    parts.push(`raw=${num(state.rawEntities)}`);
  }
  if (state.targetVisible !== null && state.targetVisible !== undefined) {
    parts.push(`seen=${state.targetVisible ? 1 : 0}`);
  }
  // 对局阶段（MATCH_PHASE）：`countdown` 期间 decisions 不涨是预期行为（准备期不出手）。
  if (state.phase) {
    parts.push(`phase=${state.phase}`);
  }
  if (state.lastError) {
    parts.push(`lastError=${state.lastError}`);
  }
  return `[brain] 心跳 ${parts.join(' ')}`;
}

// --- G-721：目标不可见（盲决策） -------------------------------------------

/** `has_target` 在 39 维观测里的下标（与 `features.OBS_FIELDS_JS[13]` 同序；G-731 后仍是 13）。 */
export const HAS_TARGET_INDEX = 13;

/**
 * 观测是否「看见了目标」。
 *
 * <p>只看 `has_target` 这一维，不看 `target_dx` 之类——后者在没目标时也是 0，
 * 分不清「贴着站」和「看不见」。
 *
 * @param {ArrayLike<number>|null|undefined} features `features.encodeObservation` 的输出
 * @param {number} [index] 测试用；默认 {@link HAS_TARGET_INDEX}
 * @returns {boolean}
 */
export function hasVisibleTarget(features, index = HAS_TARGET_INDEX) {
  if (!features || typeof features.length !== 'number' || index < 0 || index >= features.length) {
    return false;
  }
  const value = Number(features[index]);
  return Number.isFinite(value) && value >= 0.5;
}

/** 目标恢复可见时的一句话（与告警配对，避免只报「不可见」不报「好了」）。 */
export const TARGET_BACK_LINE = '[brain] 目标恢复可见';

/**
 * 盲决策告警行（G-721）。
 *
 * <p>格式固定为 `[brain] 目标不可见（entities=N）`，N = mineflayer 原始实体数（含自己）。
 * `entities=1` 基本可以判定「世界没同步进来 / 客户端位置还是旧的」（真机 G-721 就是这个形态）；
 * `entities>1` 但目标仍不可见，则更可能是目标名字过滤（`allowed` / `isLikelyBot`）对不上。
 *
 * @param {number} entities
 * @returns {string}
 */
export function formatBlindAlert(entities = 0) {
  const count = Number(entities);
  const shown = Number.isFinite(count) && count > 0 ? Math.trunc(count) : 0;
  return `[brain] 目标不可见（entities=${shown}）`;
}

/**
 * 盲决策状态机（G-721）：**大脑以为有目标、观测里 `has_target==0`** 时告警。
 *
 * <p>为什么需要它：`pickTarget()` 在 `--target-player` 下无条件返回指定名字，于是
 * 「目标在线但看不见」（登录窗口内 tp 被客户端预测吃掉、换世界、实体未同步）会让
 * 观测里 7–13、19–22 维全部静默变成 0，策略在「盲」状态里照常决策，日志里毫无异常。
 *
 * <p>告警节流：首次立刻报一条，之后每 `everyMs`（默认 5s）一条；恢复可见报一条。
 * 与 {@link TargetPresence} 的分工：那个判「目标在不在线」（tab 列表），这个判
 * 「目标看不看得见」（实体表）——两者都成立时才是真的盲。
 */
export class BlindWatch {
  /**
   * @param {{everyMs?: number}} [options]
   */
  constructor({ everyMs = 5000 } = {}) {
    this.everyMs = Math.max(0, Number(everyMs) || 0);
    this.reset();
  }

  reset() {
    this.lastAlertAt = 0;
    this.blind = false;
  }

  /**
   * @param {number} nowMs
   * @param {{expected?: boolean, visible?: boolean, entities?: number}} frame
   *   `expected`：大脑这一帧是否认为有目标；`visible`：观测里 `has_target`；
   *   `entities`：mineflayer 原始实体数（进告警行）
   * @returns {{blind: boolean, line: string|null}} `blind` 用于累加计数器，`line` 需要打印时非空
   */
  update(nowMs, { expected = false, visible = false, entities = 0 } = {}) {
    if (!expected || visible) {
      const recovered = this.blind;
      this.blind = false;
      this.lastAlertAt = 0;
      return { blind: false, line: recovered ? TARGET_BACK_LINE : null };
    }
    const first = !this.blind;
    this.blind = true;
    // 注意：不能用 `this.lastAlertAt` 的真假值判「报过没有」——首次告警可能落在 nowMs=0
    // （单测就是这样），0 是假值，会退化成每帧都报。`first` 已经是权威判据。
    if (!first && nowMs - this.lastAlertAt < this.everyMs) {
      return { blind: true, line: null };
    }
    this.lastAlertAt = nowMs;
    return { blind: true, line: formatBlindAlert(entities) };
  }
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
