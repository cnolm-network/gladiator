/**
 * 动作计划队列与串行执行器（G-735：把「决策」和「执行」拆开）。
 *
 * 背景：旧实现里 `brainTick` 一路 `await` 到动作执行完（一次 `move` 就 200ms），
 * 于是 20Hz 的定时器有 3/4 的帧被 `busy` 丢掉，有效决策率掉到 ~4Hz。
 * 现在决策侧只产计划（毫秒级），执行侧串行消费：
 *
 * - 队列容量默认 **1**：新计划到达时，若旧的还没开始执行就直接替换（丢弃旧的），
 *   计入 `superseded`——永远执行「最贴当前观测」的那一帧；
 * - **已经在执行的计划不打断**（否则会留下半截 move），它的后续步骤照常跑完；
 * - 执行器同一时刻只跑一个计划（串行），失败只记 `failed`，不阻塞后续。
 *
 * 本模块不依赖 mineflayer / 定时器，可独立单测（`test/plan-queue.test.mjs`）。
 */

/** 一个待执行的动作计划。 */
export function makePlan({ planId, kind = 'intent', intents = [], createdAt = Date.now(), meta = {} } = {}) {
  return {
    planId,
    kind,
    //: 串行执行的单元列表；形状由 `kind` 决定（`intent` → 意图字典，`action` → 协议动作）。
    intents: Array.isArray(intents) ? intents.slice() : [],
    createdAt,
    meta,
  };
}

/**
 * 容量固定的计划队列（FIFO，但满员时丢**最旧**的）。
 *
 * 为什么丢最旧而不是丢最新：新计划基于更新的观测，旧计划的价值随时间线性衰减；
 * 保留新的才能让「决策 20Hz、执行 ~5Hz」时的动作始终对应最近的画面。
 */
export class PlanQueue {
  constructor({ size = 1 } = {}) {
    this.capacity = Math.max(1, Math.trunc(Number(size) || 1));
    this.items = [];
    //: `{ pushed, superseded, taken, dropped }`——`superseded` 是本模块的核心指标。
    this.stats = { pushed: 0, superseded: 0, taken: 0, dropped: 0 };
  }

  get length() {
    return this.items.length;
  }

  get empty() {
    return this.items.length === 0;
  }

  /**
   * 入队；满员时丢最旧的。
   *
   * @returns {{accepted: boolean, superseded: number}} `superseded` = 本次因满员被丢弃的旧计划数
   */
  push(plan) {
    if (!plan) {
      return { accepted: false, superseded: 0 };
    }
    let superseded = 0;
    while (this.items.length >= this.capacity) {
      this.items.shift();
      superseded += 1;
      this.stats.dropped += 1;
    }
    this.items.push(plan);
    this.stats.pushed += 1;
    this.stats.superseded += superseded;
    return { accepted: true, superseded };
  }

  /** 取最早的一个（执行器消费入口）；空队列返回 `null`。 */
  shift() {
    const plan = this.items.shift();
    if (!plan) {
      return null;
    }
    this.stats.taken += 1;
    return plan;
  }

  /** 丢弃全部待执行计划（对局结束 / 断线时用）；返回丢弃个数。 */
  clear() {
    const count = this.items.length;
    this.items.length = 0;
    this.stats.dropped += count;
    return count;
  }

  size() {
    return this.items.length;
  }
}

/**
 * 串行执行器：同一时刻只执行一个计划，执行期间新计划只入队不打断。
 *
 * @param {{execute: (plan: object) => Promise<any>, capacity?: number,
 *          onError?: (error: any, plan: object) => void,
 *          onResult?: (result: any, plan: object) => void}} options
 */
export class PlanExecutor {
  constructor({ execute, capacity = 1, onError = null, onResult = null } = {}) {
    if (typeof execute !== 'function') {
      throw new Error('PlanExecutor 需要 execute(plan)');
    }
    this.execute = execute;
    this.queue = new PlanQueue({ size: capacity });
    this.onError = onError;
    this.onResult = onResult;
    this.running = false;
    //: `{ submitted, executed, failed, superseded, cleared }`
    this.stats = { submitted: 0, executed: 0, failed: 0, superseded: 0, cleared: 0 };
  }

  /** 正在执行计划？（心跳/诊断用；与旧 `busy` 不同，它不阻塞决策） */
  get busy() {
    return this.running;
  }

  /** 待执行队列长度（0 或 1）。 */
  get pending() {
    return this.queue.length;
  }

  /**
   * 提交一个计划并立刻返回（**不 await 执行**——这就是非阻塞的那一半）。
   *
   * @returns {{accepted: boolean, superseded: number}}
   */
  submit(plan) {
    const outcome = this.queue.push(plan);
    if (outcome.accepted) {
      this.stats.submitted += 1;
      this.stats.superseded += outcome.superseded;
    }
    this._kick();
    return outcome;
  }

  /** 丢弃待执行计划（不影响正在执行的那个）。 */
  clear() {
    const dropped = this.queue.clear();
    this.stats.cleared += dropped;
    return dropped;
  }

  /**
   * 等到队列空且没有在执行的计划（测试与 shutdown 用；生产路径不 await 它）。
   */
  async idle() {
    // 单线程 + 微任务：`running` 变 false 之后再检查一次队列即可收敛。
    for (;;) {
      if (!this.running && this.queue.empty) {
        return;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, 1);
      });
    }
  }

  _kick() {
    if (this.running) {
      return;
    }
    this.running = true;
    Promise.resolve()
      .then(() => this._drain())
      .catch((error) => {
        // `_drain` 自己吞执行错误，这里只兜「循环本身」的意外，避免未捕获拒绝。
        this.onError?.(error, null);
      })
      .finally(() => {
        this.running = false;
        // 关窗：`_drain` 因队列空退出、而 submit 恰好落在 running=true 的窗口里时，
        // 那一帧会留在队列里没人取——这里补一次。
        if (!this.queue.empty) {
          this._kick();
        }
      });
  }

  async _drain() {
    for (;;) {
      const plan = this.queue.shift();
      if (!plan) {
        return;
      }
      try {
        const result = await this.execute(plan);
        this.stats.executed += 1;
        this.onResult?.(result, plan);
      } catch (error) {
        this.stats.failed += 1;
        this.onError?.(error, plan);
      }
    }
  }
}
