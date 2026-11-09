/**
 * 每个 Bot 独立的行为配置（G-507）。
 *
 * <p>配置来源优先级：SPAWN.payload.profile &gt; CLI 参数 &gt; 内置默认值
 * （15±2 CPS、220±40ms 反应、3.99 格攻击距离、default 模型）。
 * 这里只做“手感”建模：CPS 限制点击节奏、反应延时给动作加排队延迟、
 * attack 距离用于本地预检（服务端仍会自行校验）。
 */

export const DEFAULT_PROFILE = Object.freeze({
  cps: 15,
  cpsJitter: 2,
  reactionMs: 220,
  reactionJitterMs: 40,
  reach: 3.99,
  model: 'default',
  /**
   * 战斗辅助（“外挂级”开关，默认全关，训练里逐步打开）：
   * - autoRod：距离在 2.5~4.5 格时自动切鱼竿右键，击退后再切回剑
   * - autoBlock：非攻击间隙自动举剑格挡（降低受到伤害）
   * - wtap：每次命中后松开再按下前进键（sprint reset），打断击退距离
   * - aimLock：攻击前 3D 对准目标眼睛，避免背对空挥
   */
  assist: Object.freeze({
    autoRod: false,
    autoBlock: false,
    wtap: false,
    aimLock: false,
  }),
});

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * 归一化并校验一份配置；非法值回落到默认值并告警，不抛异常
 * （运行时不应该因为调用方传了脏配置而无法启动）。
 */
export function normalizeProfile(raw, logger = null) {
  const warnings = [];
  const src = raw && typeof raw === "object" ? raw : {};
  const base = DEFAULT_PROFILE;

  const pick = (key, fallback, valid, format = String) => {
    if (!(key in src)) {
      return fallback;
    }
    const value = num(src[key], NaN);
    if (!Number.isFinite(value) || !valid(value)) {
      warnings.push(`${key}=${format(src[key])} 非法，回落 ${fallback}`);
      return fallback;
    }
    return value;
  };

  const cps = pick("cps", base.cps, (v) => v > 0 && v <= 100);
  const cpsJitter = pick("cpsJitter", base.cpsJitter, (v) => v >= 0 && v < cps);
  const reactionMs = pick("reactionMs", base.reactionMs, (v) => v >= 0 && v <= 5000);
  const reactionJitterMs = pick(
    "reactionJitterMs", base.reactionJitterMs, (v) => v >= 0 && v <= 5000,
  );
  const reach = pick("reach", base.reach, (v) => v > 0 && v <= 6);

  let model = base.model;
  if ("model" in src) {
    if (typeof src.model === "string" && src.model.trim()) {
      model = src.model.trim();
    } else {
      warnings.push(`model=${JSON.stringify(src.model)} 非法，回落 ${base.model}`);
    }
  }

  const assistSrc = src.assist && typeof src.assist === "object" ? src.assist : {};
  const assist = Object.freeze({
    autoRod: assistSrc.autoRod === true,
    autoBlock: assistSrc.autoBlock === true,
    wtap: assistSrc.wtap === true,
    aimLock: assistSrc.aimLock === true,
  });

  if (warnings.length && logger) {
    for (const w of warnings) logger.warn(`[profile] ${w}`);
  }

  return Object.freeze({
    cps,
    cpsJitter,
    reactionMs,
    reactionJitterMs,
    reach,
    model,
    assist,
  });
}

/**
 * CPS 节拍器：每次点击后随机出 1000/(cps±jitter) 的冷却。
 * 用注入的 now/random 保证测试确定性。
 */
export class ClickGate {
  constructor(profile, { now = () => Date.now(), random = Math.random } = {}) {
    this.profile = normalizeProfile(profile);
    this.now = now;
    this.random = random;
    this.nextAllowedAt = 0;
    this.clicks = 0;
    this.waits = 0;
  }

  /** 每次点击的间隔（毫秒），在 cps±jitter 范围内均匀抖动。 */
  nextIntervalMs() {
    const { cps, cpsJitter } = this.profile;
    const spread = cpsJitter * (this.random() * 2 - 1);
    const rate = Math.max(0.1, cps + spread);
    return 1000 / rate;
  }

  /** 距离下一次允许点击还有多少毫秒（&lt;=0 表示现在就可以）。 */
  remainingMs() {
    return Math.max(0, this.nextAllowedAt - this.now());
  }

  canClick() {
    return this.now() >= this.nextAllowedAt;
  }

  /** 记录一次点击并安排下一次窗口。 */
  recordClick() {
    const at = this.now();
    this.nextAllowedAt = at + this.nextIntervalMs();
    this.clicks += 1;
    return this.nextAllowedAt;
  }

  /** 等待到下一个可用窗口（用于“宁可慢一点也要按 CPS 出手”）。 */
  async waitForSlot() {
    const remaining = this.remainingMs();
    if (remaining <= 0) {
      return 0;
    }
    this.waits += 1;
    await new Promise((resolve) => setTimeout(resolve, remaining));
    return remaining;
  }

  get stats() {
    return { clicks: this.clicks, waits: this.waits, nextAllowedAt: this.nextAllowedAt };
  }
}

/**
 * 反应延时：给每个动作加 220±40ms 的“看到→动手”延迟。
 *
 * <p>talk/idle 不走这条路：说话和发呆不体现战斗反应。
 */
export class ReactionDelay {
  constructor(profile, { random = Math.random } = {}) {
    this.profile = normalizeProfile(profile);
    this.random = random;
    this.samples = 0;
    this.totalMs = 0;
  }

  nextDelayMs() {
    const { reactionMs, reactionJitterMs } = this.profile;
    const jitter = reactionJitterMs * (this.random() * 2 - 1);
    return Math.max(0, reactionMs + jitter);
  }

  async wait() {
    const delay = this.nextDelayMs();
    this.samples += 1;
    this.totalMs += delay;
    if (delay > 0) {
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
    return delay;
  }

  get meanDelayMs() {
    return this.samples === 0 ? 0 : this.totalMs / this.samples;
  }
}

/** 动作是否计入反应延时（说话/发呆/跳一下不算）。 */
// 反应延时只用于“需要人做出判断”的动作（攻击/使用/挖掘/放置）。
// 移动与视角是连续控制，每步都加 200ms 延迟会让 Bot 走一下停一下。
const REACTION_ACTIONS = new Set(['attack', 'use', 'dig', 'place']);

export function usesReactionDelay(type) {
  return REACTION_ACTIONS.has(type);
}
