/**
 * 攻击守卫：换剑判定与「无敌帧节流」（纯函数，可单测）。
 *
 * <p>背景（真机 2026-11-18 23:0x）：brain 循环里策略每 tick 都可能选 `attack`，
 * 但 1.8 的受击无敌帧是 10 tick（500ms），500ms 内的重复点击不会造成任何伤害，
 * 只会刷日志、挤占动作队列。这里把「是否该跳过这一刀」抽成纯函数，便于单测覆盖。
 *
 * <p>另一个问题是手上不一定是剑（可能拿着金苹果/鱼竿），直接 attack 只会空手挥；
 * `needsSwordSwap` 让调用方先切剑再攻击。
 */

/** 1.8 受击无敌帧 = 10 tick = 500ms。 */
export const DEFAULT_ATTACK_COOLDOWN_MS = 500;

/** 连续这么多次「选了 attack 且已在射程内」之后开始按无敌帧节流。 */
export const ATTACK_STREAK_LIMIT = 3;

/** 手持物是不是剑（`diamond_sword` / `iron_sword` …）。 */
export function isSwordItem(item) {
  return Boolean(item && typeof item.name === 'string' && /_sword$/.test(item.name));
}

/** 需要先切剑吗（空手/拿别的都算需要）。 */
export function needsSwordSwap(heldItem) {
  return !isSwordItem(heldItem);
}

/**
 * 这一 tick 的 `attack` 是否应该被节流掉。
 *
 * @param {object} options
 * @param {number} options.streak       连续选择 attack 的次数（含本次）
 * @param {number} options.lastAttackAt 上次真正出手的时间戳（ms）；0/非法视为还没出手
 * @param {number} options.now          当前时间戳（ms）
 * @param {number} [options.cooldownMs] 节流窗口，默认 500ms
 * @param {number} [options.streakLimit] 连续多少次后开始节流，默认 3
 * @returns {boolean} true = 这一刀别发
 */
export function shouldThrottleAttack({
  streak,
  lastAttackAt,
  now,
  cooldownMs = DEFAULT_ATTACK_COOLDOWN_MS,
  streakLimit = ATTACK_STREAK_LIMIT,
} = {}) {
  const tried = Number(streak) || 0;
  if (tried < Math.max(1, Number(streakLimit) || ATTACK_STREAK_LIMIT)) {
    return false;
  }
  const last = Number(lastAttackAt);
  if (!Number.isFinite(last) || last <= 0) {
    return false;
  }
  const window = Math.max(0, Number(cooldownMs) || 0);
  if (window === 0) {
    return false;
  }
  return Number(now) - last < window;
}
