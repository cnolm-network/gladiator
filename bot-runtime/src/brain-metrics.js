/**
 * 大脑运行期指标（G-735）：决策率与执行耗时的滚动统计。
 *
 * 为什么单独抽出来：这两个数要进心跳（`hz=` / `applyMsMean=`），也必须能单测。
 * 直接写在 `index.js` 里就要靠真机日志才能验，正是 G-735 想摆脱的调试方式。
 */

/**
 * 最近窗口内的决策率（Hz）。
 *
 * 口径：`(样本数 - 1) / (最后 - 最初)`——用**首末间隔**而不是 `样本数 / 窗口长度`，
 * 因为窗口还没填满时后者会低估（开局 1 秒内 20 帧应该显示 20Hz，而不是 20/10=2Hz）。
 * 少于 2 个样本时返回 0（没法算间隔）。
 */
export class DecisionRate {
  constructor({ windowMs = 10000, now = () => Date.now() } = {}) {
    this.windowMs = Math.max(1, Number(windowMs) || 10000);
    this.now = now;
    this.samples = [];
  }

  /** 记一次决策（默认用本地时钟）。 */
  mark(nowMs = this.now()) {
    const at = Number(nowMs);
    if (!Number.isFinite(at)) {
      return this;
    }
    this.samples.push(at);
    this.prune(at);
    return this;
  }

  /** 丢掉窗口外的样本（保持数组短，心跳每 5 秒读一次不会拖慢）。 */
  prune(nowMs = this.now()) {
    const cutoff = Number(nowMs) - this.windowMs;
    let drop = 0;
    while (drop < this.samples.length && this.samples[drop] <= cutoff) {
      drop += 1;
    }
    if (drop > 0) {
      this.samples.splice(0, drop);
    }
    return drop;
  }

  /** 窗口内决策率（Hz，保留 1 位小数由调用方决定）。 */
  hz(nowMs = this.now()) {
    this.prune(nowMs);
    const count = this.samples.length;
    if (count < 2) {
      return 0;
    }
    const span = (this.samples[count - 1] - this.samples[0]) / 1000;
    if (!(span > 0)) {
      return 0;
    }
    return (count - 1) / span;
  }

  /** 窗口内样本数（诊断用）。 */
  count(nowMs = this.now()) {
    this.prune(nowMs);
    return this.samples.length;
  }

  reset() {
    this.samples.length = 0;
    return this;
  }
}

/**
 * 最近 N 个样本的算术平均（`applyMsMean=`）。
 *
 * 用滑动窗口而不是累计平均：真机上偶发一次 2.9s 的动作超时会把累计均值永久拉高，
 * 之后就再也看不出「平时到底多快」了。
 */
export class RollingMean {
  constructor({ size = 50 } = {}) {
    this.capacity = Math.max(1, Math.trunc(Number(size) || 50));
    this.samples = [];
  }

  push(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return this.mean();
    }
    this.samples.push(num);
    while (this.samples.length > this.capacity) {
      this.samples.shift();
    }
    return this.mean();
  }

  mean() {
    if (this.samples.length === 0) {
      return 0;
    }
    let total = 0;
    for (const value of this.samples) {
      total += value;
    }
    return total / this.samples.length;
  }

  get count() {
    return this.samples.length;
  }

  reset() {
    this.samples.length = 0;
    return this;
  }
}
