/**
 * 本地大脑的跨帧状态（G-711 的**运行时生产端**）。
 *
 * <p>训练侧（`GladiatorEnv._update_cross_frame`）在每步末尾维护「上一帧十头动作 /
 * 距上次出手多少帧 / 距上次受伤多少帧」，编码进观测的 25–36 维。运行时这边如果不维护，
 * 只有 25 维的模型拿到的最后 12 维就恒为 0——等于「不知道上一帧干了什么」，
 * 训练与推理的输入分布直接错位（和「只走不打」是同一类病）。
 *
 * <h3>计数口径（与训练侧一致，但有一帧的相位差，这里写清楚）</h3>
 *
 * 训练：`step()` 先下发动作 → 更新计数 → 编码观测 → 返回给策略。
 * 运行时：一个 tick 是「读观测 → 编码 → 决策 → 下发」，编码在决策之前，所以
 * **编码时用的是上一帧决策的结果**。这恰好等价：训练里策略在第 t 步用的观测
 * 记录的是「第 t−1 步出手了没有」，运行时在第 t 帧编码的也是同一件事。
 *
 * - 计数单位是**大脑决策帧**（`--brain-hz`，默认 20Hz），不是 mineflayer 的 tick，
 *   也不是服务端 tick；与训练侧按 `action_ticks/control_hz` 的步长口径对齐。
 * - 饱和值 50 帧（= `features.CROSS_FRAME_NORMALIZER`，20Hz 下 2.5 秒）：
 *   再久就没有分辨意义，也让归一化落在 0..1。
 * - 「受伤」用**自身血量下降**判定，与训练侧同一条判据（`previous.health − now > 1e-3`）；
 *   血量回升（0 → 20 的重生）不算受伤。
 */
import { CROSS_FRAME_NORMALIZER } from './features.js';

/** 帧计数饱和值（与 `features.js` 的 `CROSS_FRAME_NORMALIZER` 同一个数）。 */
export const FRAME_CAP = CROSS_FRAME_NORMALIZER;

/** 判定阈值：血量差超过它才算「刚挨打」，避免浮点抖动把每一帧都判成受伤。 */
const HEALTH_EPSILON = 1e-3;

export class BrainCrossFrame {
  /**
   * @param {{cap?: number}} [options] `cap` 只给测试用（默认 50，与训练侧一致）
   */
  constructor({ cap = FRAME_CAP } = {}) {
    this.cap = Number.isFinite(Number(cap)) && Number(cap) > 0 ? Number(cap) : FRAME_CAP;
    this.reset();
  }

  /** 开局/重连时复位：没有上一帧动作（观测全 0），两个计数取饱和（= 很久以前）。 */
  reset() {
    this.prevAction = null;
    this.framesSinceAttack = this.cap;
    this.framesSinceHit = this.cap;
    this.lastHealth = null;
  }

  /**
   * 每帧读一次自身血量，推进「距上次受伤」。
   *
   * @param {number|null|undefined} health `observation.self.health`
   * @returns {boolean} 本帧是否判定为「刚受伤」
   */
  observeHealth(health) {
    const value = Number(health);
    if (!Number.isFinite(value)) {
      // 拿不到血量就不动计数（不能把「读不到」当成受伤，也不能当成没受伤后 +1）。
      return false;
    }
    const hurt = this.lastHealth !== null && value < this.lastHealth - HEALTH_EPSILON;
    this.framesSinceHit = hurt ? 0 : Math.min(this.cap, this.framesSinceHit + 1);
    this.lastHealth = value;
    return hurt;
  }

  /**
   * 提交一次真实决策：记下「这一帧选了什么」，并按是否出手推进出手计数。
   *
   * @param {{prevAction?: number[]|null, attacked?: boolean}} decision
   *   `prevAction`：v2 的十头下标（v1 传 `null` 或不传）；`attacked`：本帧是否下发 attack
   */
  commit({ prevAction, attacked = false } = {}) {
    if (prevAction !== undefined) {
      this.prevAction = Array.isArray(prevAction) ? prevAction.slice() : null;
    }
    this.framesSinceAttack = attacked
      ? 0
      : Math.min(this.cap, this.framesSinceAttack + 1);
  }

  /** 没有决策的帧（例如找不到目标只发了 idle）：只推进出手计数。 */
  advance() {
    this.framesSinceAttack = Math.min(this.cap, this.framesSinceAttack + 1);
  }

  /** 直接交给 `encodeObservation(observation, {...options, ...this.options()})`。 */
  options() {
    return {
      prevAction: this.prevAction,
      framesSinceAttack: this.framesSinceAttack,
      framesSinceHit: this.framesSinceHit,
    };
  }
}
