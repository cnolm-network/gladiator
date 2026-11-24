/**
 * 动作分发器（G-201）：ACTION 帧 → mineflayer 调用。
 *
 * <p>覆盖协议 §4.3 的 move/look/jump/attack/dig/place/use/talk/idle。
 * 失败返回结构化错误（带 code），绝不重试同一动作；规则拒绝由 RestrictionCache 预检，
 * 返回 RULE_DENIED 而不是把动作发到服务器后再被取消。
 */

import { Vec3 } from 'vec3';
import { isSwordItem } from './attack-guard.js';
import { ClickGate, ReactionDelay, normalizeProfile, usesReactionDelay } from './bot-profile.js';
import {
  ItemError, aimAt, blockWithSword, eatGoldenApple, equipArmor, eyeDistance, releaseRod, selectItem,
} from './items.js';

export const ACTION_TYPES = Object.freeze([
  'move', 'look', 'jump', 'attack', 'dig', 'place', 'use', 'talk', 'idle',
]);

const DIRECTIONS = Object.freeze(['forward', 'back', 'left', 'right', 'sprint', 'sneak']);
// 连续 move 之间允许“按住不放”的宽限期：略大于控制周期（默认 10Hz、2 tick/步 = 200ms），
// 让运动连续而不是每步松键一次；超过这个时间没有新 move 才真正停下。
const MOVE_RELEASE_GRACE_MS = 120;
// 打不到时「追一步」的按键时长：略短于一个控制周期（200ms），既产生实际位移，
// 又不会让 Bot 冲过头。
const CHASE_STEP_MS = 180;

/**
 * 等板子发套件的节奏（G-717-A）：背包里**根本没有剑**时按 200ms 看一次、最多 10 次（2 秒）。
 *
 * <p>背景（真机 2026-11-24）：开局板子还没发套件，模型已经选 `attack`，`ensureSwordInHand()`
 * 立刻抛 `ITEM_MISSING`，一局里连打 4 条 `[brain] 意图执行失败`，把真正该看的失败淹掉。
 * 这 2 秒是「还没开始」而不是「执行异常」，所以单独计数、不记失败。
 */
export const KIT_WAIT = Object.freeze({ intervalMs: 200, attempts: 10 });

const FACES = Object.freeze({
  up: [0, 1, 0],
  down: [0, -1, 0],
  north: [0, 0, -1],
  south: [0, 0, 1],
  east: [1, 0, 0],
  west: [-1, 0, 0],
});

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class ActionError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function requireNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ActionError('ACTION_UNSUPPORTED', `${field} 必须是数字`);
  }
  return value;
}

function requirePosition(value, field) {
  if (!value || typeof value !== 'object') {
    throw new ActionError('ACTION_UNSUPPORTED', `${field} 必须是 {x,y,z}`);
  }
  return {
    x: Math.floor(requireNumber(value.x, `${field}.x`)),
    y: Math.floor(requireNumber(value.y, `${field}.y`)),
    z: Math.floor(requireNumber(value.z, `${field}.z`)),
  };
}

export class ActionDispatcher {
  constructor({ bot, restrictionCache = null, maxMoveMs = 5000, profile = null,
    now = () => Date.now(), kitWait = null, onKitWait = null } = {}) {
    this.bot = bot;
    this.restrictionCache = restrictionCache;
    this.maxMoveMs = maxMoveMs;
    this.ruleDenials = 0;
    this.now = now;
    this.lastRodAt = 0;
    this.rodReleases = 0;
    this.wtaps = 0;
    this.profile = normalizeProfile(profile);
    this.clickGate = new ClickGate(this.profile);
    this.reaction = new ReactionDelay(this.profile);
    this.hold = null;
    this.holdTimer = null;
    //: 等套件的节奏（默认 200ms × 10 = 2s；测试可缩短）。
    this.kitWait = {
      intervalMs: Math.max(1, Number(kitWait && kitWait.intervalMs) || KIT_WAIT.intervalMs),
      attempts: Math.max(1, Number(kitWait && kitWait.attempts) || KIT_WAIT.attempts),
    };
    //: 进入「等套件」的次数 / 2 秒内没等到的次数（后者才是真失败）。
    this.kitWaits = 0;
    this.kitWaitGaveUp = 0;
    //: 通知外层（index.js 用它打「每局第一次」的日志并累加 waitingKit）。
    this.onKitWait = typeof onKitWait === 'function' ? onKitWait : null;
  }

  setBot(bot) {
    this.bot = bot;
  }

  /** 运行时重连/重配时替换行为配置（保留点击统计）。 */
  setProfile(profile) {
    this.profile = normalizeProfile(profile);
    this.clickGate = new ClickGate(this.profile);
    this.reaction = new ReactionDelay(this.profile);
  }

  /** 暴露给观测/复盘：本局累计的空挥、点击节奏等。 */
  get combatStats() {
    return {
      profile: this.profile,
      clicks: this.clickGate.clicks,
      cpsWaits: this.clickGate.waits,
      reactionSamples: this.reaction.samples,
      reactionMeanMs: this.reaction.meanDelayMs,
      rodReleases: this.rodReleases,
      wtaps: this.wtaps,
      ruleDenials: this.ruleDenials,
      // G-717-A：等套件的次数（不算失败）/ 2 秒没等到的次数（算真失败）。
      kitWaits: this.kitWaits,
      kitWaitGaveUp: this.kitWaitGaveUp,
    };
  }

  /** 预检规则：拒绝返回 RULE_DENIED，并累加计数。 */
  checkAllowed(position) {
    if (!this.restrictionCache || !position) {
      return;
    }
    const verdict = this.restrictionCache.check(position.x, position.y, position.z);
    if (verdict.denied) {
      this.ruleDenials += 1;
      throw new ActionError('RULE_DENIED',
        `区域 ${verdict.entry.regionId} 禁止该动作（${verdict.entry.reason}）`);
    }
  }

  async dispatch(action) {
    if (!action || typeof action.type !== 'string') {
      throw new ActionError('ACTION_UNSUPPORTED', '缺少 action.type');
    }
    if (!ACTION_TYPES.includes(action.type)) {
      throw new ActionError('ACTION_UNSUPPORTED', `未知动作类型: ${action.type}`);
    }
    const bot = this.bot;
    if (!bot) {
      throw new ActionError('BOT_NOT_READY', 'Bot 尚未登录');
    }
    const params = action.params || {};
    if (usesReactionDelay(action.type)) {
      await this.reaction.wait();
    }
    switch (action.type) {
      case 'move':
        return this.move(params);
      case 'look':
        return this.look(params);
      case 'jump':
        return this.jump();
      case 'attack':
        return this.attack(params);
      case 'dig':
        return this.dig(params);
      case 'place':
        return this.place(params);
      case 'use':
        return this.use(params);
      case 'talk':
        return this.talk(params);
      case 'idle':
        return this.idle(params);
      default:
        throw new ActionError('ACTION_UNSUPPORTED', `未知动作类型: ${action.type}`);
    }
  }

  async move(params) {
    const direction = String(params.direction || 'forward');
    if (!DIRECTIONS.includes(direction)) {
      throw new ActionError('ACTION_UNSUPPORTED', `未知方向: ${direction}`);
    }
    const durationMs = Math.min(Math.max(Number(params.durationMs) || 200, 1), this.maxMoveMs);
    const bot = this.bot;
    // sprint 不是移动方向而是一个控制状态：必须与前进同时按下才生效，
    // 只 setControlState('sprint', true) 不会让 Bot 真的加速（实测该机器人此前从不疾跑）。
    const sprint = params.sprint === true && direction === 'forward';
    // 关键：不能让每个 move 动作“按下→松开”。训练环境每 200ms 发一次 move，
    // 如果每次都在结尾松键，Bot 会走一下停一下（用户看到的一卡一卡）。
    // 这里改成“保持按住 + 延迟释放”：同方向的连续 move 之间不松键，
    // 只有方向/疾跑变化或超过宽限期没有新 move 时才释放。
    if (this.hold && (this.hold.direction !== direction || this.hold.sprint !== sprint)) {
      this._releaseHold();
    }
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    bot.setControlState(direction, true);
    if (sprint) {
      bot.setControlState('sprint', true);
    }
    this.hold = { direction, sprint };
    await sleep(durationMs);
    this._scheduleHoldRelease(MOVE_RELEASE_GRACE_MS);
  }

  _releaseHold() {
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    if (!this.hold) {
      return;
    }
    const { direction, sprint } = this.hold;
    this.hold = null;
    try {
      this.bot.setControlState(direction, false);
      if (sprint) {
        this.bot.setControlState('sprint', false);
      }
    } catch (error) {
      // Bot 掉线时 setControlState 会抛，这里吞掉由上层统一处理连接问题。
    }
  }

  _scheduleHoldRelease(graceMs) {
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
    }
    this.holdTimer = setTimeout(() => {
      this.holdTimer = null;
      this._releaseHold();
    }, Math.max(1, graceMs));
    if (typeof this.holdTimer.unref === 'function') {
      this.holdTimer.unref();
    }
  }

  async look(params) {
    const yaw = requireNumber(params.yaw, 'yaw');
    const pitch = Math.max(-90, Math.min(90, requireNumber(params.pitch, 'pitch')));
    await this.bot.look(yaw, pitch, true);
  }

  async jump() {
    const bot = this.bot;
    bot.setControlState('jump', true);
    try {
      await sleep(50);
    } finally {
      bot.setControlState('jump', false);
    }
  }

  async attack(params) {
    const id = Number(params.targetEntityId);
    if (!Number.isInteger(id)) {
      throw new ActionError('ACTION_UNSUPPORTED', 'targetEntityId 必须是整数');
    }
    const entity = this.bot.entities[id];
    if (!entity) {
      throw new ActionError('ACTION_UNSUPPORTED', `实体不存在: ${id}`);
    }
    const distance = eyeDistance(this.bot, entity);
    if (distance > this.profile.reach) {
      // 打不到就追上去，不报错。
      //
      // 训练里对手被判「能打」之后，常被学习方那一击的击退推开 1.0~1.6 格
      // （击退初速 0.4 格/tick 衰减 + 5.4 格/s × 0.2s 位移），dispatch 时真实 eyeDistance
      // 已经 > reach。抛 OUT_OF_REACH 只会刷日志，这一 tick 还是白白浪费（下一 tick 仍要靠近）。
      // 所以改成「先转身 + 疾跑追一步」，返回 ok=true，环境侧不再记失败，
      // 但仍可用 code=OUT_OF_REACH_CHASED 统计「有多少次是贴上去的」。
      await this.chaseTarget(entity, distance);
      return {
        ok: true,
        code: 'OUT_OF_REACH_CHASED',
        distance,
        reach: this.profile.reach,
      };
    }
    this.checkAllowed({
      x: Math.floor(entity.position.x),
      y: Math.floor(entity.position.y),
      z: Math.floor(entity.position.z),
    });
    // A1：主手必须是剑。策略经常在拿着金苹果/鱼竿（甚至空手）时发 attack，
    // 结果白白吃一发伤害。这里先自动切剑；只有背包里确实没有剑才报 ITEM_MISSING。
    await this.ensureSwordInHand();
    // 攻击前 3D 对准目标眼睛：训练里模型只给朝向，这里补一次精确对准。
    if (this.profile.assist.aimLock) {
      try {
        await aimAt(this.bot, entity, true);
      } catch (error) {
        // 对准失败不阻断攻击（例如目标刚消失）。
      }
    }
    // 自动放鱼竿：中距离先钓一下把对手击退，再切回剑。
    if (this.profile.assist.autoRod && distance > 2.5 && distance <= 4.5
        && this.now() - this.lastRodAt > 1500) {
      try {
        await releaseRod(this.bot, 'sword');
        this.lastRodAt = this.now();
        this.rodReleases += 1;
      } catch (error) {
        // 没有鱼竿时静默跳过，不影响正常攻击。
      }
    }
    // CPS 节奏：宁可等下一次窗口，也不在冷却里连点。
    await this.clickGate.waitForSlot();
    this.bot.attack(entity);
    this.clickGate.recordClick();
    if (this.profile.assist.wtap) {
      await this.sprintReset();
    }
  }

  /**
   * 够不到目标时先追一步：转身对准 + 按住前进/疾跑 CHASE_STEP_MS，然后交给既有的延迟释放。
   *
   * <p>复用 {@link #move} 的「按住 + 延迟释放」机制（{@code this.hold} + {@link #_scheduleHoldRelease}），
   * 不会把前进键按死；环境下一 tick 若继续靠近，`move` 会接着保持按住，反之宽限期后自动松开。
   */
  async chaseTarget(entity, distance) {
    const bot = this.bot;
    if (!entity || !entity.position) {
      throw new ActionError('ENTITY_NOT_FOUND', '追击时目标已消失');
    }
    try {
      // 未开 aimLock 也至少把视角转过去，否则疾跑是「盲跑」，下一 tick 依然打不到。
      await aimAt(bot, entity, this.profile.assist.aimLock === true);
    } catch (error) {
      // 目标刚消失 / 测试替身没有 lookAt：不阻断追击，下一 tick 的 look 动作会纠正朝向。
    }
    if (this.hold && (this.hold.direction !== 'forward' || this.hold.sprint !== true)) {
      this._releaseHold();
    }
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    bot.setControlState('forward', true);
    bot.setControlState('sprint', true);
    this.hold = { direction: 'forward', sprint: true };
    await sleep(CHASE_STEP_MS);
    this._scheduleHoldRelease(MOVE_RELEASE_GRACE_MS);
    return { distance, reach: this.profile.reach };
  }

  /**
   * 确保主手是剑：手持不是剑但快捷栏/背包里有剑时先切过去。
   *
   * <p>背包读不到（只读封装或测试替身）时不做判定，避免误伤没有 inventory API 的调用方；
   * 真实运行时 mineflayer 一定有 inventory，所以训练路径上这条判定始终生效。
   *
   * <p>G-717-A：背包里**根本没有剑**（开局板子还没发套件）不再立刻抛 `ITEM_MISSING`，
   * 而是等最多 2 秒（200ms × 10）——这段等待由 {@link #waitForKit} 计数，外层不记失败。
   */
  async ensureSwordInHand() {
    const bot = this.bot;
    const hasSword = this._inventoryHasSword();
    if (hasSword === null) {
      return;
    }
    if (isSwordItem(bot.heldItem)) {
      return;
    }
    if (!hasSword) {
      const ready = await this.waitForKit();
      if (!ready) {
        throw new ActionError('ITEM_MISSING',
          `背包里没有剑（等了 ${this.kitWait.attempts * this.kitWait.intervalMs}ms 板子仍未发套件）`);
      }
    }
    // 每次死亡/传送后 mineflayer 会重开背包窗口，期间 items() 会短暂为空：
    // 立刻判定会误报 ITEM_MISSING（真机 23:48 每杀一次报一条）。给两个短重试窗口。
    for (let attempt = 0; ; attempt += 1) {
      try {
        await selectItem(bot, 'sword');
        return;
      } catch (error) {
        if (error instanceof ItemError) {
          if (error.code === 'ITEM_MISSING' && attempt < 2) {
            await sleep(120);
            continue;
          }
          throw new ActionError(error.code, error.message);
        }
        throw error;
      }
    }
  }

  /** 背包（任意格子）里有没有剑；读不到背包返回 null（调用方不做判定）。 */
  _inventoryHasSword() {
    const bot = this.bot;
    const items = bot && bot.inventory && typeof bot.inventory.items === 'function'
      ? bot.inventory.items() : null;
    if (!Array.isArray(items)) {
      return null;
    }
    return items.some((item) => item && /_sword$/.test(String(item.name || '')));
  }

  /**
   * 等板子发套件：每 `intervalMs` 看一次背包，最多 `attempts` 次（默认 200ms × 10 = 2s）。
   *
   * <p>**期间不抛错**：这是「还没开始」而不是「执行异常」，所以只累加 `kitWaits`；
   * 只有 2 秒都没等到才由调用方抛 `ITEM_MISSING`（那时才算真失败）。
   *
   * @returns {Promise<boolean>} 是否等到剑
   */
  async waitForKit() {
    this.kitWaits += 1;
    if (this.onKitWait) {
      try {
        this.onKitWait({ attempts: this.kitWait.attempts, intervalMs: this.kitWait.intervalMs });
      } catch (error) {
        // 通知（打日志/计数）失败不能影响动作本身。
      }
    }
    for (let attempt = 0; attempt < this.kitWait.attempts; attempt += 1) {
      if (this._inventoryHasSword() === true) {
        return true;
      }
      await sleep(this.kitWait.intervalMs);
    }
    if (this._inventoryHasSword() === true) {
      return true;
    }
    this.kitWaitGaveUp += 1;
    return false;
  }

  /**
   * W-tap：命中后松开前进/疾跑 1 tick 再按回，抵消自身被击退的距离。
   * 只在当前确实在前进时执行，避免把站立状态改成前进。
   */
  async sprintReset() {
    const bot = this.bot;
    const state = bot.controlState || {};
    const wasForward = state.forward === true;
    if (!wasForward) {
      return;
    }
    const wasSprint = state.sprint === true;
    bot.setControlState('sprint', false);
    bot.setControlState('forward', false);
    await sleep(50);
    bot.setControlState('forward', true);
    if (wasSprint) {
      bot.setControlState('sprint', true);
    }
    this.hold = { direction: 'forward', sprint: wasSprint };
    this.wtaps += 1;
  }

  async dig(params) {
    const position = requirePosition(params.position, 'position');
    this.checkAllowed(position);
    const block = this.bot.blockAt(new Vec3(position.x, position.y, position.z));
    if (!block || block.name === 'air') {
      throw new ActionError('ACTION_UNSUPPORTED', `目标方块为空: ${position.x},${position.y},${position.z}`);
    }
    await this.bot.dig(block);
  }

  async place(params) {
    const against = requirePosition(params.against, 'against');
    this.checkAllowed(against);
    const face = String(params.face || 'up');
    const offset = FACES[face];
    if (!offset) {
      throw new ActionError('ACTION_UNSUPPORTED', `未知朝向: ${face}`);
    }
    const reference = this.bot.blockAt(new Vec3(against.x, against.y, against.z));
    if (!reference || reference.boundingBox === 'empty') {
      throw new ActionError('ACTION_UNSUPPORTED', '参考方块不可放置');
    }
    this.checkAllowed({
      x: against.x + offset[0],
      y: against.y + offset[1],
      z: against.z + offset[2],
    });
    await this.bot.placeBlock(reference, new Vec3(offset[0], offset[1], offset[2]));
  }

  /**
   * 右键交互（动作细化）：`item` 可以是 rod / gapple / sword / blocks / food。
   * - rod：切鱼竿→右键释放→切回剑
   * - gapple：切金苹果→按住 1.6s 吃完
   * - sword：举剑格挡 holdMs
   * - 其它：切到该类别后 activateItem
   */
  async use(params) {
    const item = params.item ? String(params.item) : null;
    if (!item) {
      this.bot.activateItem(Boolean(params.offhand));
      return;
    }
    try {
      if (params.select === true) {
        // 只切换手持物，不使用：用于每次开局把双方都固定成“手握剑”，
        // 否则拿着金苹果/鱼竿的对手会显著改变对局强度，训练数据会剧烈波动。
        await selectItem(this.bot, item);
        return;
      }
      if (item === 'armor') {
        // 由 Layer 3 在发完套件后显式调用：不依赖 inventory 事件（实测 1.8.8 下
        // 自己背包的 update 事件并不总是触发，导致护甲一直躺在背包里）。
        this.bot.armorWorn = await equipArmor(this.bot);
        return;
      }
      if (item === 'rod') {
        this.checkAllowedEntity(params);
        await releaseRod(this.bot, 'sword');
        this.rodReleases += 1;
        return;
      }
      if (item === 'gapple') {
        await eatGoldenApple(this.bot, Number(params.holdMs) || 1600);
        return;
      }
      if (item === 'sword') {
        const holdMs = Math.min(Math.max(Number(params.holdMs) || 250, 1), 2000);
        await blockWithSword(this.bot, holdMs);
        return;
      }
      if (item === 'blocks' || item === 'food') {
        await selectItem(this.bot, item);
        this.bot.activateItem();
        return;
      }
      throw new ActionError('ACTION_UNSUPPORTED', `未知物品类别: ${item}`);
    } catch (error) {
      if (error instanceof ItemError) {
        throw new ActionError(error.code, error.message);
      }
      throw error;
    }
  }

  /** 使用物品前若带了位置，先做规则预检。 */
  checkAllowedEntity(params) {
    if (params && params.position) {
      this.checkAllowed(requirePosition(params.position, 'position'));
    }
  }

  async talk(params) {
    const message = params.message;
    if (typeof message !== 'string' || message.length === 0 || message.length > 256) {
      throw new ActionError('ACTION_UNSUPPORTED', 'message 长度必须是 1..256');
    }
    this.bot.chat(message);
  }

  async idle(params) {
    const durationMs = Math.min(Math.max(Number(params.durationMs) || 100, 1), this.maxMoveMs);
    const bot = this.bot;
    // 先清掉“按住不放”的移动状态，否则 idle 之后 hold 仍是旧方向，下一步 move 会以为键还按着。
    this.hold = null;
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    for (const control of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
      bot.setControlState(control, false);
    }
    // 自动防砍：空闲时举剑格挡（1.8.9 格挡能减少正面伤害）。
    if (this.profile.assist.autoBlock && durationMs >= 100) {
      try {
        await blockWithSword(bot, durationMs);
        return;
      } catch (error) {
        // 没剑就退化成普通待机。
      }
    }
    await sleep(durationMs);
  }
}
