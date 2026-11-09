/**
 * 寻路集成（G-207）。
 *
 * <p>pathfinder 生成路径前先查 RestrictionCache：目标点在 deny 矩形内时直接拒绝，
 * 不发路径请求；路径中途的 deny 区域由 events 监听在进入时取消路径并回 RULE_DENIED。
 *
 * <p>mineflayer-pathfinder 按需加载：未安装时降到"不可用"，不阻塞其余动作。
 */

import { Vec3 } from 'vec3';

let cachedModule = null;

export async function loadPathfinder() {
  if (cachedModule) {
    return cachedModule;
  }
  try {
    cachedModule = await import('mineflayer-pathfinder');
    return cachedModule;
  } catch (error) {
    cachedModule = { unavailable: true, error };
    return cachedModule;
  }
}

export class NavigationError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export class Navigator {
  constructor({ bot, restrictionCache = null, logger = console } = {}) {
    this.bot = bot;
    this.restrictionCache = restrictionCache;
    this.logger = logger;
    this.loaded = false;
  }

  setBot(bot) {
    this.bot = bot;
    this.loaded = false;
  }

  async ensureReady() {
    if (this.loaded) {
      return cachedModule;
    }
    const module = await loadPathfinder();
    if (module.unavailable) {
      throw new NavigationError('ACTION_UNSUPPORTED', 'mineflayer-pathfinder 不可用');
    }
    if (!this.bot) {
      throw new NavigationError('BOT_NOT_READY', 'Bot 尚未登录');
    }
    if (typeof this.bot.pathfinder === 'undefined') {
      this.bot.loadPlugin(module.pathfinder);
    }
    const { Movements } = module;
    if (Movements && this.bot.pathfinder) {
      this.bot.pathfinder.setMovements(new Movements(this.bot));
    }
    this.loaded = true;
    return module;
  }

  /**
   * 走到目标点附近。
   *
   * @param {{x:number,y:number,z:number}} position 目标方块坐标
   * @param {{range?:number, timeoutMs?:number}} options
   */
  async goTo(position, { range = 1, timeoutMs = 10000 } = {}) {
    const target = {
      x: Math.floor(position.x),
      y: Math.floor(position.y),
      z: Math.floor(position.z),
    };
    // 目的地本身被禁行：不发路径请求（deny 区域作为禁行节点）
    if (this.restrictionCache) {
      const verdict = this.restrictionCache.check(target.x, target.y, target.z);
      if (verdict.denied) {
        throw new NavigationError('RULE_DENIED',
          `目标区域 ${verdict.entry.regionId} 禁止进入（${verdict.entry.reason}）`);
      }
    }
    const module = await this.ensureReady();
    const { goals } = module;
    return new Promise((resolve, reject) => {
      const bot = this.bot;
      const goal = new goals.GoalNear(target.x, target.y, target.z, range);
      let settled = false;
      const finish = (error) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        bot.removeListener('goal_reached', onReached);
        bot.removeListener('path_update', onPathUpdate);
        if (error) {
          reject(error);
        } else {
          resolve();
        }
      };
      const onReached = () => finish(null);
      const onPathUpdate = (result) => {
        if (result && result.status === 'noPath') {
          finish(new NavigationError('ACTION_UNSUPPORTED', '无法生成到目标的路径'));
        }
      };
      const timer = setTimeout(() => {
        try {
          bot.pathfinder.setGoal(null);
        } catch {
          // ignore
        }
        finish(new NavigationError('ACTION_TIMEOUT', `寻路超时（>${timeoutMs}ms）`));
      }, timeoutMs);
      if (typeof timer.unref === 'function') {
        timer.unref();
      }
      bot.on('goal_reached', onReached);
      bot.on('path_update', onPathUpdate);
      bot.pathfinder.setGoal(goal);
    });
  }

  stop() {
    if (this.bot && this.bot.pathfinder) {
      try {
        this.bot.pathfinder.setGoal(null);
      } catch {
        // ignore
      }
    }
  }
}

export function positionOf(value) {
  return new Vec3(value.x, value.y, value.z);
}
