#!/usr/bin/env node
/**
 * Bot 运行时入口（Layer 2）。
 *
 * 两条链路：
 *  - 通道 A（connection.js）：连插件 8765，收 SPAWN → mineflayer 真实登录（minecraft.js）
 *    → 服务端 PlayerJoin 绑定 BotHandle 并回 BOT_BOUND → 插件派发 GladiatorBotReadyEvent。
 *  - 通道 B（link-server.js）：起 WS 8766 给 Layer 3，按 controlHz 推 OBSERVATION、
 *    接 ACTION / RESET，动作经限流与 seq 去重后打到 mineflayer。
 *
 * 用法示例：
 *   node src/index.js --username Glad_Bot_1 --host 127.0.0.1 --mc-port 25565
 * 环境变量：GLADIATOR_BRIDGE_PORT / GLADIATOR_BRIDGE_TOKEN / GLADIATOR_ENV_PORT /
 *          MC_HOST / MC_PORT / BOT_USERNAME / MC_VERSION / OBSERVATION_HZ
 */
import path from 'node:path';
import { ActionDispatcher } from './actions.js';
import {
  DEFAULT_ATTACK_COOLDOWN_MS,
  needsSwordSwap,
  shouldThrottleAttack,
} from './attack-guard.js';
import { LinkClient } from './connection.js';
import { decodeAction, encodeObservation, entityHealth, nearestPlayer } from './features.js';
import { EnvLinkServer } from './link-server.js';
import { describeLoginFailure, launchBot } from './minecraft.js';
import { buildObservation } from './observation.js';
import { loadPolicy, resolvePolicyFile } from './policy.js';
import { RconClient, parseRconSpec, parseStatsLine } from './rcon.js';
import { ActionScheduler } from './scheduler.js';
import { equipArmor } from './items.js';
import { Navigator } from './navigation.js';
import { RestrictionCache } from './restriction-cache.js';
import { DEFAULT_PROFILE, normalizeProfile } from './bot-profile.js';

function arg(name, fallback) {
  const argv = process.argv.slice(2);
  const flag = `--${name}`;
  const i = argv.indexOf(flag);
  if (i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--')) {
    return argv[i + 1];
  }
  const envName = name.replace(/-/g, '_').toUpperCase();
  return process.env[envName] ?? process.env[`GLADIATOR_${envName}`] ?? fallback;
}

const username = arg('username', process.env.BOT_USERNAME ?? 'Glad_Bot_1');
const bridgeHost = arg('bridge-host', '127.0.0.1');
const bridgePort = Number(arg('bridge-port', process.env.GLADIATOR_BRIDGE_PORT ?? 8765));
const token = arg('token', process.env.GLADIATOR_BRIDGE_TOKEN ?? '');
const mcHost = arg('host', process.env.MC_HOST ?? '127.0.0.1');
const mcPort = Number(arg('mc-port', process.env.MC_PORT ?? 25565));
const mcVersion = arg('mc-version', process.env.MC_VERSION ?? '1.8.8');
const envPort = Number(arg('env-port', process.env.GLADIATOR_ENV_PORT ?? 8766));
const controlHz = Number(arg('observation-hz', process.env.OBSERVATION_HZ ?? 10));

// --- 本地大脑（块 A）--------------------------------------------------------
// 空字符串 = 旧行为：只执行通道 B 下发的动作（Layer 3 训练时就是这个模式）。
// 给模型后运行时自己闭环决策，人机对战不需要训练环境在跑。
const brainSpec = String(arg('brain', process.env.GLADIATOR_BRAIN ?? '') || '').trim();
const modelsDir = path.resolve(String(arg('models-dir', path.resolve('../dashboard/var/models'))));
// 20Hz 与训练/BC 的控制频率对齐（`controlHz=10` 的每步 0.05s 决策步长）；
// 5Hz 时同一策略的出手频率只有训练时的 1/4，近身几乎打不出伤害（真机 2026-11-18 实测）。
const brainHzRaw = Number(arg('brain-hz', 20));
const brainHz = Math.max(1, Math.min(20, Number.isFinite(brainHzRaw) && brainHzRaw > 0 ? brainHzRaw : 20));
const brainEpsilon = Math.max(0, Math.min(1, Number(arg('brain-epsilon', 0)) || 0));
// 1.8 受击无敌 500ms：连续攻击时按该窗口节流，避免刷无效点击。
const attackCooldownRaw = Number(arg('attack-cooldown-ms', DEFAULT_ATTACK_COOLDOWN_MS));
const attackCooldownMs = Math.max(
  0,
  Math.min(5000, Number.isFinite(attackCooldownRaw) && attackCooldownRaw >= 0
    ? attackCooldownRaw
    : DEFAULT_ATTACK_COOLDOWN_MS),
);
const targetPlayerArg = String(arg('target-player', '') || '').trim();
const rconSpec = parseRconSpec(
  arg('rcon', process.env.GLADIATOR_RCON ?? ''),
  String(arg('rcon-password', process.env.GLADIATOR_RCON_PASSWORD ?? 'devpass')),
);
const brainEnabled = brainSpec.length > 0;

// 每个 Bot 独立的“手感”配置（G-507）：CLI/env 给基线，SPAWN 里的 profile 可整体覆盖。
const cliProfile = normalizeProfile({
  cps: arg('cps', DEFAULT_PROFILE.cps),
  cpsJitter: arg('cps-jitter', DEFAULT_PROFILE.cpsJitter),
  reactionMs: arg('reaction-ms', DEFAULT_PROFILE.reactionMs),
  reactionJitterMs: arg('reaction-jitter-ms', DEFAULT_PROFILE.reactionJitterMs),
  reach: arg('reach', DEFAULT_PROFILE.reach),
  model: arg('model', DEFAULT_PROFILE.model),
}, console);

// 与服务端 BotRegistry.botIdFor 保持同一公式，两端才能对上同一个 botId。
const botId = `glad-${username.toLowerCase()}`;

const client = new LinkClient({
  url: `ws://${bridgeHost}:${bridgePort}/gladiator`,
  token,
  botId,
  username,
  mcVersion,
  runtimeVersion: '0.1.0',
});

/** 当前会话：Bot 与各类按 botId 复用的对象。 */
let session = null;
let bot = null;
let spawnInFlight = false;
let quitting = false;
let gameState = 'Alive';
let gameId = '';
const chatLines = [];
const notices = [];

const restrictionCache = new RestrictionCache();
const navigator = new Navigator({ restrictionCache });

const envServer = new EnvLinkServer({
  port: envPort,
  controlHz: Number.isFinite(controlHz) ? controlHz : 10,
  maxActionsPerSecond: Math.max(10, Math.ceil(cliProfile.cps + cliProfile.cpsJitter)),
  resolveSession: (requestedBotId) => {
    if (!session || requestedBotId !== botId) {
      return null;
    }
    return session;
  },
  // 本地大脑模式下通道 B 的 ACTION 一律忽略：两套指令源同时下发会让 Bot 抽风。
  acceptActions: () => !brainEnabled,
  logger: console,
});

/** 当前生效的配置：SPAWN.profile 覆盖 CLI 基线。 */
let activeProfile = cliProfile;

function makeSession(currentBot) {
  return {
    bot: currentBot,
    restrictionCache,
    dispatcher: new ActionDispatcher({ bot: currentBot, restrictionCache, profile: activeProfile }),
    navigator,
    state: () => gameState,
    gameId: () => gameId,
    chatLines: () => chatLines,
    notices: () => notices,
    onRuleDenied: (action, reason) => {
      console.warn(`[gladiator] 动作被规则拒绝 ${action.type}: ${reason}`);
    },
  };
}

// --- 本地大脑（块 A）------------------------------------------------------
// 加载策略：(1) `--brain` 为空 → 完全不启用，保持「只执行通道 B 动作」的旧行为；
// (2) 给了模型 → 按 brainHz 自己决策，并把通道 B 的 ACTION 忽略掉（避免两套指令打架）。
let policy = null;
if (brainEnabled) {
  const policyFile = resolvePolicyFile(brainSpec, { modelsDir });
  if (!policyFile) {
    console.error(`[brain] 找不到模型 ${brainSpec}（已试路径与 --models-dir=${modelsDir}），退出`);
    process.exit(2);
  }
  try {
    policy = loadPolicy(policyFile);
  } catch (error) {
    console.error(`[brain] 模型加载失败: ${error.message}`);
    process.exit(2);
  }
  if (policy.numActions !== 11) {
    console.error(`[brain] 模型动作空间 ${policy.numActions} != 11，无法与 ACTION_NAMES 对齐`);
    process.exit(2);
  }
  if (policy.obsDim !== 23) {
    console.warn(`[brain] 模型 obsDim=${policy.obsDim}（运行时特征 23 维）：窄的字段按 0 补齐，`
      + '相当于忽略新增特征，效果可能弱于 23 维模型');
  }
  console.log(`[brain] 模型 ${policyFile} obsDim=${policy.obsDim} actions=${policy.numActions}`
    + ` hidden=${policy.hidden} temperature=${policy.temperature}`);
}

const brainState = {
  timer: null,
  poller: null,
  busy: false,
  decisions: 0,
  attacks: 0,
  damage: 0,
  kills: 0,
  actionFailures: 0,
  attackStreak: 0,
  attacksThrottled: 0,
  swordSwaps: 0,
  startedAt: Date.now(),
  seq: 0,
  tick: 0,
  lastAttackAt: 0,
  target: targetPlayerArg || null,
  targetPickedAt: 0,
  opponentHealth: null,
  lastOpponentHealth: null,
  killCounted: false,
  deathSummaryPrinted: false,
  rcon: null,
  rconLastPollAt: 0,
  rconWarned: false,
};

const brainScheduler = new ActionScheduler({
  // 大脑每 tick 最多发 3 条动作（look/move/attack），20Hz 就是 60 条/秒。
  // 沿用训练用的 17/s 上限会把 look 丢掉（真机 2026-11-18 23:34 复现 ACTION_RATE_LIMITED），
  // 所以只在启用大脑时放宽到「每秒动作预算」，通道 B 训练路径的上限不变。
  maxPerSecond: Math.max(
    10,
    Math.ceil(cliProfile.cps + cliProfile.cpsJitter),
    brainEnabled ? brainHz * 3 : 0,
  ),
});

/** 用户名像 Bot 的一律不当对手（人机对战里要打的是真人）。 */
function isLikelyBot(name) {
  return /^(glad[_-]|gladbot|bot[_-]|goblin|ai_)/i.test(String(name || ''));
}

function selfName() {
  return (bot && bot.username) || username;
}

/**
 * 选对手：`--target-player` 显式指定时固定用它；否则取最近的、看起来不是 Bot 的玩家。
 * 锁定后 3 秒内不换人（避免观战者路过时策略来回切目标）。
 */
function pickTarget(observation, nowMs) {
  if (targetPlayerArg) {
    return targetPlayerArg;
  }
  const self = selfName().toLowerCase();
  const entities = observation.entities || [];
  const keep = brainState.target;
  if (keep && nowMs - brainState.targetPickedAt < 3000) {
    const stillThere = entities.some((entity) => entity
      && entity.kind === 'player'
      && String(entity.name || '').toLowerCase() === keep.toLowerCase());
    if (stillThere) {
      return keep;
    }
  }
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const entity of entities) {
    if (!entity || entity.kind !== 'player') {
      continue;
    }
    const name = String(entity.name || '');
    if (!name || name.toLowerCase() === self || isLikelyBot(name)) {
      continue;
    }
    const distance = Number(entity.distance);
    if (Number.isFinite(distance) && distance < bestDistance) {
      bestDistance = distance;
      best = name;
    }
  }
  if (best && best !== brainState.target) {
    console.log(`[brain] 目标锁定 ${best}（距离 ${bestDistance.toFixed(1)} 格）`);
    brainState.target = best;
    brainState.targetPickedAt = nowMs;
    brainState.lastOpponentHealth = null;
    brainState.opponentHealth = null;
    brainState.killCounted = false;
  }
  return brainState.target;
}

/** 记录一次对手血量采样：掉的血记成 damage，归零记一次 kill。 */
function recordOpponentHealth(health) {
  const value = Number(health);
  if (!Number.isFinite(value)) {
    return;
  }
  const previous = brainState.lastOpponentHealth;
  if (previous !== null && value < previous - 0.05) {
    brainState.damage += previous - value;
  }
  if (value <= 0 && !brainState.killCounted) {
    brainState.killCounted = true;
    brainState.kills += 1;
    console.log('[brain] 对手血量归零，记一次击杀');
  }
  if (value > 0) {
    brainState.killCounted = false;
  }
  brainState.lastOpponentHealth = value;
  brainState.opponentHealth = value;
}

/**
 * RCON 采样对手血量（1.8 的玩家血量不通过实体元数据稳定暴露）。
 * 失败只告警一次并丢弃连接，下次 tick 自动重连——不能因为采样失败停掉决策。
 */
async function pollOpponentHealth() {
  if (!rconSpec || !brainState.target) {
    return;
  }
  const nowMs = Date.now();
  if (nowMs - brainState.rconLastPollAt < 900) {
    return;
  }
  brainState.rconLastPollAt = nowMs;
  try {
    if (!brainState.rcon || !brainState.rcon.ready) {
      if (brainState.rcon) {
        brainState.rcon.close();
      }
      brainState.rcon = new RconClient(rconSpec);
      await brainState.rcon.connect();
      console.log(`[brain] RCON 已连接 ${rconSpec.host}:${rconSpec.port}（轮询 /gladiator stats）`);
    }
    const raw = await brainState.rcon.command(`gladiator stats ${brainState.target}`);
    const stats = parseStatsLine(raw);
    if (stats) {
      recordOpponentHealth(stats.health);
    }
  } catch (error) {
    if (!brainState.rconWarned) {
      brainState.rconWarned = true;
      console.warn(`[brain] RCON 采样失败（会重试）: ${error?.message ?? error}`);
    }
    if (brainState.rcon) {
      brainState.rcon.close();
      brainState.rcon = null;
    }
  }
}

/** 一次决策：编码 23 维 → 策略 → 解码成协议动作 → 经 ActionDispatcher 下发。 */
async function brainTick() {
  if (!policy || brainState.busy || !bot || !session || !bot.entity) {
    return;
  }
  brainState.busy = true;
  const currentSession = session;
  try {
    brainState.tick += 1;
    const observation = buildObservation(bot, {
      tick: brainState.tick,
      gameId,
      state: gameState,
      restrictions: restrictionCache.snapshot(),
      chat: chatLines,
      pendingNotices: notices,
    });
    const nowMs = Date.now();
    const targetName = pickTarget(observation, nowMs);
    if (!targetName) {
      await currentSession.dispatcher.dispatch({ type: 'idle', params: { durationMs: 180 } });
      return;
    }
    // 没有 RCON 时退化成读实体血量（有就用，读不到就保持上次的采样值）。
    if (!rconSpec) {
      const entity = (observation.entities || []).find((item) => item
        && item.kind === 'player'
        && String(item.name || '').toLowerCase() === targetName.toLowerCase());
      recordOpponentHealth(entityHealth(entity));
    }
    const features = encodeObservation(observation, {
      username: selfName(),
      stepFraction: 0,
      opponentHealth: brainState.opponentHealth,
      allowed: [targetName],
      reach: activeProfile.reach,
      sinceAttack: brainState.lastAttackAt ? (nowMs - brainState.lastAttackAt) / 1000 : 5,
    });
    const index = brainEpsilon > 0 && Math.random() < brainEpsilon
      ? Math.floor(Math.random() * policy.numActions)
      : policy.greedy(features);
    let actions = decodeAction(index, observation, selfName(), {
      allowed: [targetName],
      reach: activeProfile.reach,
    });
    // 连续选 attack 说明已经贴身：按 1.8 的无敌帧（500ms）节流，别每 tick 刷空点击。
    const choseAttack = actions.some((action) => action.type === 'attack');
    brainState.attackStreak = choseAttack ? brainState.attackStreak + 1 : 0;
    const throttled = choseAttack && shouldThrottleAttack({
      streak: brainState.attackStreak,
      lastAttackAt: brainState.lastAttackAt,
      now: nowMs,
      cooldownMs: attackCooldownMs,
    });
    if (throttled) {
      actions = actions.filter((action) => action.type !== 'attack');
      if (actions.length === 0) {
        actions = [{ type: 'idle', params: { durationMs: 40 } }];
      }
      brainState.attacksThrottled += 1;
      if (brainState.attacksThrottled === 1) {
        console.log(`[brain] 敌人仍在无敌帧内，攻击节流到 ${attackCooldownMs}ms 一刀`);
      }
    } else if (choseAttack && needsSwordSwap(bot.heldItem)) {
      // 手上不是剑就先切剑再打：直接 attack 只会空手挥（训练里模型看不到手持物）。
      actions = [{ type: 'use', params: { item: 'sword', holdMs: 50 } }, ...actions];
      brainState.swordSwaps += 1;
    }
    brainState.decisions += 1;
    for (const action of actions) {
      brainState.seq += 1;
      const result = await brainScheduler.run(
        brainState.seq,
        () => currentSession.dispatcher.dispatch(action),
      );
      if (result && result.ok === false) {
        brainState.actionFailures += 1;
        if (brainState.actionFailures <= 5) {
          console.warn(`[brain] 动作失败 ${action.type}: ${result.code || '?'} ${result.reason || ''}`);
        }
      }
      if (action.type === 'attack') {
        brainState.attacks += 1;
        brainState.lastAttackAt = Date.now();
      }
    }
  } catch (error) {
    console.warn(`[brain] 决策失败: ${error?.message ?? error}`);
  } finally {
    brainState.busy = false;
  }
}

function startBrain() {
  if (!brainEnabled || !policy) {
    return;
  }
  if (brainState.timer) {
    clearInterval(brainState.timer);
  }
  const interval = Math.max(50, Math.floor(1000 / brainHz));
  brainState.timer = setInterval(() => {
    brainTick().catch((error) => console.warn(`[brain] tick 异常: ${error?.message ?? error}`));
  }, interval);
  if (brainState.timer.unref) {
    brainState.timer.unref();
  }
  console.log(`[brain] 已启动 ${brainHz}Hz，目标=${targetPlayerArg || '自动（最近的非 Bot 玩家）'}`
    + ` epsilon=${brainEpsilon} attackCooldown=${attackCooldownMs}ms`
    + ` rcon=${rconSpec ? `${rconSpec.host}:${rconSpec.port}` : 'off'}`);
  if (rconSpec) {
    const poller = setInterval(() => {
      pollOpponentHealth().catch(() => {});
    }, 1000);
    if (poller.unref) {
      poller.unref();
    }
    brainState.poller = poller;
  }
}

function stopBrain() {
  if (brainState.timer) {
    clearInterval(brainState.timer);
    brainState.timer = null;
  }
  if (brainState.poller) {
    clearInterval(brainState.poller);
    brainState.poller = null;
  }
  if (brainState.rcon) {
    brainState.rcon.close();
    brainState.rcon = null;
  }
}

/** 对局结束/退出时的机器可读汇总（Dashboard 直接读该行）。 */
function printMatchSummary(reason) {
  if (!brainEnabled) {
    return;
  }
  const duration = Math.round((Date.now() - brainState.startedAt) / 100) / 10;
  console.log(JSON.stringify({
    event: 'match-summary',
    decisions: brainState.decisions,
    attacks: brainState.attacks,
    damage: Math.round(brainState.damage * 10) / 10,
    kills: brainState.kills,
    duration,
    reason: reason || 'exit',
    model: brainSpec,
    target: brainState.target || null,
    actionFailures: brainState.actionFailures,
    attacksThrottled: brainState.attacksThrottled,
    swordSwaps: brainState.swordSwaps,
    brainHz,
    attackCooldownMs,
  }));
}

client.on('welcome', (payload) => {
  const uuid = payload.playerUuid || '(尚未绑定，等待 PlayerJoin)';
  console.log(`[gladiator] ready session=${payload.sessionId} playerUuid=${uuid}`);
});

client.on('message', async (envelope) => {
  const payload = envelope.payload ?? {};
  if (envelope.type === 'SPAWN') {
    activeProfile = normalizeProfile({ ...cliProfile, ...(payload.profile ?? {}) }, console);
    console.log(
      `[gladiator] SPAWN request=${payload.requestId} bot=${payload.botId}`
      + ` target=${payload.server?.host}:${payload.server?.port}`
      + ` profile=cps ${activeProfile.cps}±${activeProfile.cpsJitter},`
      + ` reaction ${activeProfile.reactionMs}±${activeProfile.reactionJitterMs}ms,`
      + ` reach ${activeProfile.reach}, model ${activeProfile.model}`,
    );
    await handleSpawn(payload);
    return;
  }
  if (envelope.type === 'BOT_BOUND') {
    console.log(`[gladiator] BOT_BOUND user=${payload.username} uuid=${payload.playerUuid} state=${payload.state}`);
    return;
  }
  if (envelope.type === 'STATE') {
    const previous = payload.state ?? gameState;
    gameState = payload.state ?? gameState;
    if (payload.gameId) {
      gameId = payload.gameId;
    }
    console.log(`[gladiator] STATE ${payload.previous ?? '?'} → ${gameState}${payload.reason ? ` (${payload.reason})` : ''}`);
    // 对局结束信号：自己被淘汰或死亡时输出一次汇总（回到 Alive 后允许再次输出）。
    if (brainEnabled) {
      const ended = gameState === 'Dead' || gameState === 'Eliminated' || previous === 'Eliminated';
      if (ended && gameState !== 'Alive' && !brainState.deathSummaryPrinted) {
        brainState.deathSummaryPrinted = true;
        printMatchSummary(`state:${gameState}`);
      }
      if (gameState === 'Alive' || gameState === 'Respawning') {
        brainState.deathSummaryPrinted = false;
      }
    }
    return;
  }
  if (envelope.type === 'NOTICE') {
    handleNotice(payload);
    return;
  }
  console.log(`[gladiator] ${envelope.type} ${JSON.stringify(payload)}`);
});

function handleNotice(payload) {
  const data = payload.data ?? {};
  if (payload.kind === 'RESTRICTED') {
    const expiresAt = Number(data.expiresAt) || undefined;
    const regionIds = Array.isArray(data.regionIds) && data.regionIds.length > 0 ? data.regionIds : [''];
    for (const regionId of regionIds) {
      restrictionCache.put({
        regionId,
        box: data.box ?? null,
        allowPvp: false,
        expiresAt,
        action: data.action ?? 'pvp',
        reason: data.reason ?? 'pvp-denied',
      });
    }
    console.log(`[gladiator] NOTICE RESTRICTED region=${regionIds.join(',')} reason=${data.reason ?? '-'}`);
  } else if (payload.kind === 'CHAT') {
    chatLines.push({ at: payload.at ?? Date.now(), username: data.username, message: data.message });
    if (chatLines.length > 100) {
      chatLines.shift();
    }
  } else if (payload.kind === 'IDLE') {
    console.log(`[gladiator] NOTICE IDLE since=${data.sinceSeconds}s`);
  }
  notices.push({ kind: payload.kind, at: payload.at ?? Date.now(), data });
  if (notices.length > 50) {
    notices.shift();
  }
}

client.on('reject', (payload) => {
  console.error(`[gladiator] 链路被拒绝 ${payload.code}: ${payload.message}`);
});

client.on('reconnecting', ({ delayMs, attempt }) => {
  console.warn(`[gladiator] 链路断开，第 ${attempt} 次重连将在 ${delayMs}ms 后开始`);
});

client.on('stopped', () => {
  console.log('[gladiator] 链路已停止（不再重连）');
});

async function handleSpawn(spawn) {
  if (bot || spawnInFlight) {
    console.warn('[gladiator] 已有 Bot 在运行或登录中，忽略重复 SPAWN（链路重连会重放 SPAWN）');
    return;
  }
  spawnInFlight = true;
  const targetHost = spawn.server?.host && spawn.server.host !== '' ? spawn.server.host : mcHost;
  const targetPort = Number(spawn.server?.port ?? mcPort);
  try {
    bot = await launchBot({
      host: targetHost,
      port: targetPort,
      username: spawn.username ?? username,
      version: mcVersion,
    });
    console.log(`[gladiator] mineflayer 登录成功 user=${bot.username}`);
    const current = bot;
    session = makeSession(current);
    navigator.setBot(current);
    // 本地大脑在 Bot 实体就绪后开始决策（不需要 Layer 3 接入通道 B）。
    if (brainEnabled) {
      brainState.deathSummaryPrinted = false;
      startBrain();
    }
    // 调用方（训练场/对局）给完套件后自动穿上护甲。
    // 注意：不能只靠 bot.inventory 的 'update' 事件——1.8.8 实测 RCON give 之后
    // 该事件不一定触发，护甲会一直躺在背包里；这里用轮询兜底，穿上 4 件后停止。
    let equipTimer = null;
    let equipAttempts = 0;
    const tryEquip = () => {
      if (equipTimer) {
        return;
      }
      equipTimer = setInterval(() => {
        equipAttempts += 1;
        equipArmor(current)
          .then((worn) => {
            // 不能穿上就停：人机对战每局都会 clear + give，死后护甲也会掉落，
            // 停掉的定时器会让第二局开始就裸奔。equipArmor 在没有可穿物品时本就是空操作。
            if (worn.length && worn.every((name) => !name.startsWith('!'))) {
              if (worn.join(',') !== lastWorn) {
                lastWorn = worn.join(',');
                console.log(`[gladiator] 已穿戴护甲: ${worn.join(', ')}`);
              }
            }
          })
          .catch((error) => console.warn(`[gladiator] 穿护甲失败：${error?.message ?? error}`));
        // 800ms 而不是 2s：RCON give 之后 mineflayer 的背包同步可能要等一个 tick，
        // 2s 的间隔会让大脑在开局头两秒一直报 ITEM_MISSING（真机 23:48 实测）。
      }, 800);
      if (equipTimer.unref) {
        equipTimer.unref();
      }
    };
    let lastWorn = '';
    current.inventory.on('update', tryEquip);
    current.once('spawn', tryEquip);
    tryEquip();
    current.on('error', (err) => {
      // 必须常驻监听：mineflayer 的登录期 once('error') 已被消费，之后的错误若无监听会打崩进程。
      console.warn(`[gladiator] Bot 连接错误：${err?.message ?? err}`);
    });
    current.on('chat', (user, message) => {
      chatLines.push({ at: Date.now(), username: user, message });
      if (chatLines.length > 100) {
        chatLines.shift();
      }
    });
    current.on('message', (message) => {
      const text = typeof message === 'string' ? message : message?.toString?.() ?? '';
      if (text) {
        console.log(`[mc] ${text}`);
      }
    });
    current.once('end', (reason) => {
      console.warn(`[gladiator] Bot 断线：${reason ?? 'unknown'}`);
      if (bot === current) {
        stopBrain();
        printMatchSummary(`disconnect:${reason ?? 'unknown'}`);
        bot = null;
        session = null;
        restrictionCache.clear();
      }
    });
    current.once('kicked', (reason) => {
      console.warn(`[gladiator] Bot 被踢出：${reason}`);
    });
  } catch (err) {
    const failure = describeLoginFailure(err);
    bot = null;
    session = null;
    console.error(`[gladiator] 登录失败 ${failure.code}: ${failure.message}`);
    client.send('SPAWN_FAILED', {
      requestId: spawn.requestId ?? '',
      botId: spawn.botId ?? botId,
      code: failure.code,
      message: failure.message,
    });
  } finally {
    spawnInFlight = false;
  }
}

async function shutdown(signal) {
  if (quitting) return;
  quitting = true;
  console.log(`[gladiator] 收到 ${signal}，正在退出`);
  stopBrain();
  printMatchSummary(`signal:${signal}`);
  envServer.stop();
  client.stop();
  if (bot) {
    try {
      bot.quit('runtime shutdown');
    } catch {
      // ignore
    }
  }
  process.exit(0);
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

console.log(`[gladiator] bot=${botId} username=${username} link=ws://${bridgeHost}:${bridgePort}/gladiator env=ws://127.0.0.1:${envPort}`);
envServer.start();
client.start();

// 故障注入（仅测试用）：GLADIATOR_TEST_DROP_MS 在启动后掐一次连接，
// GLADIATOR_TEST_DROP_EVERY_MS 周期性掐断，用来验证退避重连与插件侧会话替换。
const dropOnceMs = Number(process.env.GLADIATOR_TEST_DROP_MS || 0);
const dropEveryMs = Number(process.env.GLADIATOR_TEST_DROP_EVERY_MS || 0);
if (dropOnceMs > 0) {
  setTimeout(() => {
    console.warn(`[test] 主动断开链路（${dropOnceMs}ms 后，模拟闪断）`);
    client.debugDropConnection();
  }, dropOnceMs).unref();
}
if (dropEveryMs > 0) {
  setInterval(() => {
    console.warn(`[test] 主动断开链路（每 ${dropEveryMs}ms，模拟抖动）`);
    client.debugDropConnection();
  }, dropEveryMs).unref();
}
