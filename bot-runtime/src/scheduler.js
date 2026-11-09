/**
 * 动作节流与幂等去重（G-202）。
 *
 * <p>协议 §4.3：seq 单调递增，重复 seq 必须丢弃并回放缓存结果；协议 §5：
 * 超过 maxActionsPerSecond 的动作丢弃并回 ACTION_RATE_LIMITED。
 *
 * <p>本模块不依赖 mineflayer，可独立单测。
 */

export class ActionScheduler {
  constructor({ maxPerSecond = 10, dedupeCapacity = 256, now = () => Date.now() } = {}) {
    this.maxPerSecond = Math.max(1, maxPerSecond);
    this.dedupeCapacity = Math.max(1, dedupeCapacity);
    this.now = now;
    this.acceptedAt = [];
    this.results = new Map();
    this.pending = new Map();
    this.stats = { accepted: 0, rateLimited: 0, deduped: 0 };
  }

  /** 清理 1 秒窗口外的旧时间戳。 */
  prune() {
    const cutoff = this.now() - 1000;
    while (this.acceptedAt.length > 0 && this.acceptedAt[0] <= cutoff) {
      this.acceptedAt.shift();
    }
  }

  /**
   * 执行一个动作。
   *
   * @param {number} seq      协议序号（用于幂等）
   * @param {() => Promise<any>} execute 真正的动作实现
   * @returns {Promise<{ok:boolean, code?:string, reason?:string, deduped?:boolean}>}
   */
  async run(seq, execute) {
    if (Number.isInteger(seq)) {
      const cached = this.results.get(seq);
      if (cached) {
        this.stats.deduped += 1;
        return { ...cached, deduped: true };
      }
      const inflight = this.pending.get(seq);
      if (inflight) {
        // 同一 seq 并发到达：等第一个执行完，回放它的结果（不能重复执行）。
        const shared = await inflight;
        this.stats.deduped += 1;
        return { ...shared, deduped: true };
      }
    }

    this.prune();
    if (this.acceptedAt.length >= this.maxPerSecond) {
      this.stats.rateLimited += 1;
      return { ok: false, code: 'ACTION_RATE_LIMITED', reason: `超过 ${this.maxPerSecond} actions/s` };
    }
    this.acceptedAt.push(this.now());
    this.stats.accepted += 1;

    const task = this.execute(seq, execute);
    if (Number.isInteger(seq)) {
      this.pending.set(seq, task);
    }
    try {
      return await task;
    } finally {
      if (Number.isInteger(seq)) {
        this.pending.delete(seq);
      }
    }
  }

  async execute(seq, execute) {
    let result;
    try {
      await execute();
      result = { ok: true };
    } catch (error) {
      result = {
        ok: false,
        code: error && error.code ? error.code : 'ACTION_FAILED',
        reason: error && error.message ? error.message : String(error),
      };
    }

    if (Number.isInteger(seq)) {
      this.results.set(seq, result);
      if (this.results.size > this.dedupeCapacity) {
        const oldest = this.results.keys().next().value;
        this.results.delete(oldest);
      }
    }
    return result;
  }
}
