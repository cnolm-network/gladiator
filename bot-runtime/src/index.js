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
import fs from 'node:fs';
import path from 'node:path';
import { loadItemTable, matchingMask } from './action-mask.js';
import { ActionDispatcher } from './actions.js';
import {
  DEFAULT_ATTACK_COOLDOWN_MS,
  needsSwordSwap,
  shouldThrottleAttack,
} from './attack-guard.js';
import { LinkClient } from './connection.js';
import { decodeAction, encodeObservation, entityHealth, nearestPlayer, OBS_DIM } from './features.js';
import { applyIntent, decodeIntentVector, describeIntent, resetIntentState } from './intent.js';
import {
  countdownLine,
  facingAction,
  filterPhaseActions,
  phaseGate,
  remainingSeconds,
} from './match-phase.js';
import { BrainCrossFrame } from './brain-cross-frame.js';
import {
  BlindWatch,
  FarChaseNotifier,
  TargetPresence,
  armorPoints,
  formatHeartbeat,
  hasVisibleTarget,
  horizontalDistance,
} from './brain-telemetry.js';
import { debugLog, setDebugEnabled } from './debug.js';
import { DecisionRate, RollingMean } from './brain-metrics.js';
import { PlanExecutor, makePlan } from './plan-queue.js';
import { EnvLinkServer } from './link-server.js';
import { describeLoginFailure, launchBot } from './minecraft.js';
import { buildObservation } from './observation.js';
import { loadPolicy, resolvePolicyFile } from './policy.js';
import { RconClient, parseRconSpec, parseStatsLine } from './rcon.js';
import { ActionScheduler } from './scheduler.js';
import {
  applySoftActions,
  describeSoft,
  parseSoft,
  softDropReason,
} from './soft-constraints.js';
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
// --- 观测落盘（G-721 排查）---------------------------------------------------
// `--dump-obs <file>` 把每个大脑决策帧的编码向量写成 JSONL（最多 DUMP_OBS_LIMIT 帧），
// 用来和训练侧 `spaces.encode_observation` 逐维 diff——「运行时看到的世界」不该靠猜。
const dumpObsPath = String(arg('dump-obs', '') || '').trim();
const DUMP_OBS_LIMIT = 60;
//: 每多少个决策帧落一帧（20Hz 下 40 帧 ≈ 2 秒采样一件，覆盖整局而不是只抓开头）。
const DUMP_OBS_EVERY = Math.max(1, Number(arg('dump-obs-every', 40)) || 40);
let dumpObsCount = 0;
if (dumpObsPath) {
  try {
    fs.writeFileSync(dumpObsPath, '', 'utf8');
    console.log(`[brain] 观测落盘：${dumpObsPath}（上限 ${DUMP_OBS_LIMIT} 帧）`);
  } catch (error) {
    console.warn(`[brain] 观测落盘初始化失败：${error.message}`);
  }
}
// --- 调试日志（G-717-B）-----------------------------------------------------
// `--debug` 或 GLADIATOR_DEBUG=1 才打开；打开后高频细节（远距离追击的
// OUT_OF_REACH_CHASED 等）以 `[debug]` 前缀输出，默认静默不刷屏。
setDebugEnabled(
  process.argv.slice(2).some((item) => item === '--debug' || item.startsWith('--debug='))
  || /^(1|true|yes|on)$/i.test(String(process.env.GLADIATOR_DEBUG ?? '').trim()),
);
// --- 遥测阈值（G-717-B/C）---------------------------------------------------
//: 目标玩家从 tab 列表消失多久算「不在线」。
const TARGET_OFFLINE_AFTER_MS = 5000;
//: 距离超过多少格算「在赶路」，以及这种提示的节流间隔。
const FAR_CHASE_THRESHOLD = 24;
const FAR_CHASE_LOG_MS = 5000;
// --- 软约束（运行参数，**不写进模型**）----------------------------------------
// 只影响「本进程怎么用这个模型」：同一份 .npz 在演示/评测/对战时可以开不同约束。
// 任何情况下都不要把它写进 policy.npz 或 model.json —— 那是运行期选择，不是模型事实。
const softSpec = String(arg('soft', process.env.GLADIATOR_SOFT ?? '') || '').trim();
const softParsed = parseSoft(softSpec);
if (!softParsed.ok) {
  console.error(`[soft] ${softParsed.error}`);
  process.exit(1);
}
let soft = softParsed.soft;
const softMaxCpsArg = String(arg('soft-max-cps', '') || '').trim();
if (softMaxCpsArg) {
  const value = Number(softMaxCpsArg);
  if (!Number.isFinite(value) || value < 1 || value > 40) {
    console.error(`[soft] --soft-max-cps 需要在 1..40 之间（收到 ${softMaxCpsArg}）`);
    process.exit(1);
  }
  soft = { ...soft, maxCps: value };
}
// CPS 上限换算成攻击冷却：20Hz 下每 tick 都能挥，但 1.8 的受击无敌只有 500ms，
// 超过 2 次/秒的点击本来就没有收益。只节流攻击，不裁剪动作集合。
// 注意 max() 的语义：cps ≥ 2 时被 500ms 保护盖住（只记录意图），cps=1 才真的变慢。
const softAttackCooldownMs = soft.maxCps
  ? Math.max(attackCooldownMs, Math.round(1000 / soft.maxCps))
  : attackCooldownMs;
const rconSpec = parseRconSpec(
  arg('rcon', process.env.GLADIATOR_RCON ?? ''),
  String(arg('rcon-password', process.env.GLADIATOR_RCON_PASSWORD ?? 'devpass')),
);
const brainEnabled = brainSpec.length > 0;
console.log(`[soft] ${describeSoft(soft)}`);
if (!brainEnabled && softSpec) {
  console.warn('[soft] 软约束只作用于本地大脑（--brain）；当前未启用大脑，'
    + '本进程仍只执行通道 B 下发的动作，约束不会生效');
}

// --- 动作空间 v2：物品能力表 ------------------------------------------------
// 环境只发 {type:"intent"}，**怎么按**（点一下 / 按住多少 tick）由这份表决定。
// 默认 ../training/gladiator_env/items.json；文件缺失/写错直接退出并打印绝对路径，
// 否则会变成「intent 到了但 use 永远不生效」这种很难查的静默故障。
const itemsJsonArg = String(arg('items-json', process.env.GLADIATOR_ITEMS_JSON ?? '') || '').trim();
let itemTable = null;
try {
  itemTable = loadItemTable(itemsJsonArg || null);
} catch (error) {
  console.error(`[intent] ${error.message}`);
  process.exit(3);
}
console.log(`[intent] 物品能力表 ${itemTable.path}（${itemTable.size} 条）`);

// G-708：reach 默认 3.0（原版水平）。超限值只能显式开作弊：`--cheat-reach <值>`
// 或 SPAWN profile 里的 `cheatReach: true`；两者都会打 `[cheat] reach 超限=…`。
// 只给 `--reach 3.99` 而不给作弊开关时，normalizeProfile 会降回 3.0 并告警。
const cheatReachArg = Number(arg('cheat-reach', 0)) || 0;
const cliReachRaw = arg('reach', DEFAULT_PROFILE.reach);
const cliProfile = normalizeProfile({
  cps: arg('cps', DEFAULT_PROFILE.cps),
  cpsJitter: arg('cps-jitter', DEFAULT_PROFILE.cpsJitter),
  reactionMs: arg('reaction-ms', DEFAULT_PROFILE.reactionMs),
  reactionJitterMs: arg('reaction-jitter-ms', DEFAULT_PROFILE.reactionJitterMs),
  reach: cheatReachArg > 0 ? cheatReachArg : cliReachRaw,
  cheatReach: cheatReachArg > 0,
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
  const dispatcher = new ActionDispatcher({
    bot: currentBot, restrictionCache, profile: activeProfile,
    // G-717-A：开局板子还没发套件时只「等」，不记失败；日志每局只打一次。
    onKitWait: ({ attempts, intervalMs }) => {
      brainState.waitingKit += 1;
      if (!brainState.kitWaitLogged) {
        brainState.kitWaitLogged = true;
        console.log(`[brain] 还没拿到剑，等板子发套件（最多 ${Math.round(attempts * intervalMs / 1000)}s）`);
      }
    },
  });
  return {
    bot: currentBot,
    restrictionCache,
    dispatcher,
    navigator,
    state: () => gameState,
    gameId: () => gameId,
    chatLines: () => chatLines,
    notices: () => notices,
    // 动作空间 v2：意图帧走这里（v1 的宏动作仍走 dispatcher.dispatch，两条路互不影响）。
    applyIntent: async (intent) => {
      const result = await applyIntent(currentBot, intent, {
        dispatcher,
        itemTable,
        // 软约束的 cps 上限同样作用于意图路径：1.8 的受击无敌就是 500ms，多按没收益。
        attackCooldownMs: softAttackCooldownMs,
      });
      logIntent(currentBot, intent, result);
      return result;
    },
    onRuleDenied: (action, reason) => {
      console.warn(`[gladiator] 动作被规则拒绝 ${action.type}: ${reason}`);
    },
  };
}

// --- 意图回显 -------------------------------------------------------------
// 训练侧排查「策略发了什么、运行时执行成什么」时只认这行日志：前 5 条全打，
// 之后每 20 条打一条（5Hz 下节奏刚好，quota 也不会被日志吃掉）。
const intentEcho = { received: 0 };

/**
 * 这一帧的执行结果是不是**只有**「够不到、追了一步」（G-717-B）。
 *
 * <p>它是 `ok:true` 的战术结果，不是失败；远距离赶路时每刀一条，留在 info 里会把
 * 真机日志刷满（实测 1114 格追击 50 秒），所以整行降到 debug。
 */
function isChaseResult(applied) {
  const attack = applied && applied.attack;
  return Boolean(attack && attack.code === 'OUT_OF_REACH_CHASED');
}

function logIntent(currentBot, intent, result) {
  intentEcho.received += 1;
  const count = intentEcho.received;
  if (count > 5 && count % 20 !== 0) {
    return;
  }
  const applied = (result && result.applied) || {};
  const detail = Object.keys(applied).length ? JSON.stringify(applied) : '-';
  const line = `[intent] #${count} ${describeIntent(intent)} → ${detail}`;
  if (isChaseResult(applied)) {
    // 追击帧：`--debug` 才看得到（`OUT_OF_REACH_CHASED` 不再是噪音）。
    debugLog(line);
  } else {
    console.log(line);
  }
  if (itemTable && (count === 1 || count % 100 === 0)) {
    // 运行时侧的 mask（实时背包）与训练侧（观测 200ms 一帧）偶发不一致，
    // 打出来是为了让「策略明明选了 use 却没反应」这类问题一眼可见。
    const mask = matchingMask(currentBot, itemTable);
    const bits = (list) => list.map((value) => (value ? '1' : '0')).join('');
    console.log(`[intent] mask slot=${bits(mask.slot)} use=${bits(mask.use)}`
      + ` 手持=${describeHeld(currentBot)}`);
  }
}

function describeHeld(currentBot) {
  const held = currentBot && currentBot.heldItem;
  return held && typeof held.name === 'string' && held.name !== 'air'
    ? `${held.name}x${held.count || 1}`
    : '空手';
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
  if (policy.actionSpace !== 'v2' && policy.numActions !== 11) {
    console.error(`[brain] 模型动作空间 ${policy.numActions} != 11，无法与 ACTION_NAMES 对齐`);
    process.exit(2);
  }
  if (policy.obsDim !== OBS_DIM) {
    console.warn(`[brain] 模型 obsDim=${policy.obsDim}（运行时特征 ${OBS_DIM} 维）：窄的字段按 0 补齐，`
      + `相当于忽略新增特征，效果可能弱于 ${OBS_DIM} 维模型`);
  }
  if (policy.actionSpace === 'v2') {
    // G-712：v2 = 十头意图模型（numActions 是各头之和 40，不能按 11 校验）。
    console.log(`[brain] 模型 ${policyFile} actionSpace=v2`
      + ` heads=[${policy.headSizes.join(',')}]`
      + ` obsDim=${policy.obsDim} hidden=${policy.hidden} temperature=${policy.temperature}`);
  } else {
    console.log(`[brain] 模型 ${policyFile} obsDim=${policy.obsDim} actions=${policy.numActions}`
      + ` hidden=${policy.hidden} temperature=${policy.temperature}`);
  }
}

//: v2 的 attack 头下标（顺序由模型头表决定，这里取一次，逐 tick 用来记攻击统计）。
const ATTACK_HEAD_INDEX = policy && policy.actionSpace === 'v2'
  ? policy.headNames.indexOf('attack')
  : -1;

/**
 * G-711：本地大脑的跨帧状态（上一帧动作 + 两个帧计数）。
 *
 * <p>训练侧把这三项编码进观测的 25–36 维；运行时如果不产，模型的后 12 维恒为 0，
 * 训练与推理的输入分布就错位了。计数单位是**大脑决策帧**（`--brain-hz`，默认 20Hz），
 * 与训练侧 `action_ticks/control_hz` 的步长口径对齐——不是 mineflayer tick，也不是服务端 tick。
 */
const brainCrossFrame = new BrainCrossFrame();

const brainState = {
  timer: null,
  poller: null,
  heartbeat: null,
  busy: false,
  decisions: 0,
  attacks: 0,
  damage: 0,
  kills: 0,
  actionFailures: 0,
  attackStreak: 0,
  attacksThrottled: 0,
  swordSwaps: 0,
  //: 被软约束剔除的宏动作次数（运行期统计，不进模型）。
  softDropped: 0,
  // --- G-716 可观测性：定位「只决策一次就停」 -------------------------------
  //: 上一次真正进入决策的时间戳（ms）；心跳里显示距今多少毫秒。
  lastTickAt: 0,
  //: 最近一次失败/异常的一句话（心跳会带出来）。
  lastError: null,
  //: 当前正在 await 的动作标签（`attack`/`look`/`intent`/…）与开始时间。
  lastAction: null,
  lastActionAt: 0,
  //: 动作超过 ACTION_TIMEOUT_MS 未返回的次数（放行后继续决策）。
  actionTimeouts: 0,
  //: 因为 busy 而整帧跳过的次数：如果它一直涨，就是有动作卡住了。
  busySkips: 0,
  //: 因为 bot/session/bot.entity 缺失而跳过的次数（另一种「静默空转」）。
  notReadySkips: 0,
  // --- G-735：非阻塞派发（决策侧耗时 / 执行侧耗时拆开看） ---------------------
  //: 一帧决策的总耗时（ms）= 读观测 + 编码 + 推理 + 入队；心跳 `lastTick=`。
  lastFrameMs: 0,
  //: 上一帧的分段耗时（心跳 `obsMs/inferMs/applyMs`）。
  obsMs: 0,
  inferMs: 0,
  //: 最近一次「计划执行」的耗时（ms），由执行器写回；心跳 `applyMs=`。
  applyMs: 0,
  //: 因为「计划还没执行完」被丢掉的计划数（心跳 `superseded=`）。
  superseded: 0,
  // --- G-721：盲决策（大脑认为有目标，观测里 has_target=0） -------------------
  //: 盲决策帧数（进心跳 `blind=`）。它一直涨说明目标「看不见」：登录窗口内 tp 被客户端
  //: 位置预测吃掉、换世界/重生后实体未同步、或目标名字被过滤器挡掉。
  blind: 0,
  //: 最近一帧 mineflayer 的原始实体数（含自己，进心跳 `entities=`）。
  //: `entities=1` + `blind` 涨 = 世界还没同步进来，不是模型问题。
  entitiesSeen: 0,
  // --- G-717：把「在赶路 / 等套件 / 目标不在线」从「不动也不打」里区分开 ---------
  //: 进入「等板子发套件」的次数（不算失败；2 秒没等到才由调用方算失败）。
  waitingKit: 0,
  //: 本次对局是否已经打过「等套件」的提示（每局只打一次）。
  kitWaitLogged: false,
  // --- 对局阶段（MATCH_PHASE，插件每秒推一帧）-------------------------------
  //: 最近一帧归一化后的阶段；`null` = 没收到过（老服务端）→ 不做任何抑制。
  matchPhase: null,
  //: 准备期日志的节流哨兵：只在「剩余秒数变了」时打一行，避免 5Hz 刷屏。
  countdownLoggedSecond: null,
  //: `ended` 只提示一次。
  endedLogged: false,
  //: 与目标的水平距离（心跳里的 dist）与本局最远距离（远距离提示用）。
  targetDist: null,
  maxTargetDist: 0,
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

/** 目标在线状态机（G-717-C）：只有显式 `--target-player` 时才跟踪。 */
const targetPresence = new TargetPresence({
  offlineAfterMs: TARGET_OFFLINE_AFTER_MS, name: targetPlayerArg,
});

// --- G-735：决策侧 / 执行侧拆开的两个基础设施 ---------------------------------
//: 最近 10 秒的决策率（心跳 `hz=`）。用首末间隔算，开局也能显示真实频率。
const decisionRate = new DecisionRate({ windowMs: 10000 });
//: 最近 50 个计划的执行耗时（心跳 `applyMsMean=`）；不用累计均值，免得一次超时把基线拉高。
const applyMean = new RollingMean({ size: 50 });

/**
 * 动作计划执行器：**串行**消费，队列最多 1 个待执行计划。
 *
 * <p>决策侧 `submit()` 立刻返回（这就是非阻塞的那一半）；正在执行的计划不打断，
 * 新计划只在队列里等——所以「半截 move」不会出现，而决策仍能按 20Hz 跑满。
 */
const planExecutor = new PlanExecutor({
  capacity: 1,
  execute: (plan) => executePlan(plan),
  onError: (error, plan) => {
    brainState.lastError = `计划执行失败: ${error?.message ?? error}`;
    console.warn(`[brain] 计划执行失败 plan=${plan?.planId ?? '?'}:`
      + ` ${error?.message ?? error}`);
  },
});

/** 远距离「赶路中」提示（G-717-B）：>24 格每 5 秒一条，回到近处重置。 */
const farChaseNotifier = new FarChaseNotifier({
  threshold: FAR_CHASE_THRESHOLD, everyMs: FAR_CHASE_LOG_MS,
});
/** 盲决策告警（G-721）：首次立刻报，之后每 5 秒一条；恢复可见报一条。 */
const blindWatch = new BlindWatch({ everyMs: 5000 });

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

/**
 * 单个动作的硬超时（G-716）。默认 3 秒；`GLADIATOR_ACTION_TIMEOUT_MS` 可覆盖。
 *
 * <p>为什么需要：`brainDecide()` 用 `brainState.busy` 防重入，`finally` 里清零。只要有一条
 * `await` 永不 settle（mineflayer 的窗口事务确认、`equip`、`look` 的 physicsTick 都可能），
 * `busy` 就永远是 true，之后**每一帧都直接 return**——表现是「只决策一次、不动不打、无失败日志」。
 * 超时后放行不是修根因，而是把「静默卡死」变成「可见的告警 + 继续决策」。
 */
const ACTION_TIMEOUT_MS = Math.max(
  500, Number(process.env.GLADIATOR_ACTION_TIMEOUT_MS) || 3000,
);

/** 超时哨兵（普通对象，靠身份比较，永远不会与动作结果撞上）。 */
const ACTION_TIMED_OUT = Object.freeze({ code: 'ACTION_TIMEOUT' });

/** 打一条超时告警并返回结构化失败，供调用方当作一次普通失败统计。 */
function reportActionTimeout(label, startedAt) {
  brainState.actionTimeouts += 1;
  const cost = Date.now() - startedAt;
  console.warn(`[brain] 动作超时 type=${label} ms=${cost}`
    + `（放行，不阻塞后续决策；累计 ${brainState.actionTimeouts} 次）`);
  brainState.lastError = `动作超时 ${label} ${cost}ms`;
  return {
    ok: false,
    code: 'ACTION_TIMEOUT',
    reason: `动作 ${label} 超过 ${ACTION_TIMEOUT_MS}ms 未返回（已放行）`,
  };
}

/**
 * 给一次动作套硬超时（G-716）：正常返回结果；抛错照旧抛；超时则告警并返回失败。
 *
 * <p>超时之后底层 promise 仍然挂着（mineflayer 的内部状态可能已经不一致），这里只保证
 * **调度层不会被它拖死**；`lastAction` 会保留标签与耗时，心跳里能看见到底卡在哪个动作上。
 */
async function runBrainAction(label, factory) {
  const startedAt = Date.now();
  brainState.lastAction = label;
  brainState.lastActionAt = startedAt;
  // 先包一层再 race：动作同步抛错也走 settled，不会变成未捕获异常。
  const settled = Promise.resolve()
    .then(factory)
    .then((value) => ({ value }), (error) => ({ error }));
  let timer = null;
  const guard = new Promise((resolve) => {
    timer = setTimeout(() => resolve(ACTION_TIMED_OUT), ACTION_TIMEOUT_MS);
    if (typeof timer.unref === 'function') {
      timer.unref();
    }
  });
  const outcome = await Promise.race([settled, guard]);
  if (timer) {
    clearTimeout(timer);
  }
  if (outcome === ACTION_TIMED_OUT) {
    return reportActionTimeout(label, startedAt);
  }
  brainState.lastAction = null;
  if (outcome.error) {
    throw outcome.error;
  }
  return outcome.value;
}

/** 当前背包物品（带 slot；读不到返回空数组——心跳只是展示，不值得抛）。 */
function currentInventoryItems() {
  try {
    if (bot && bot.inventory && typeof bot.inventory.items === 'function') {
      const items = bot.inventory.items();
      return Array.isArray(items) ? items : [];
    }
  } catch (error) {
    // 背包窗口重建期间 items() 可能抛：心跳不能因此失败。
  }
  return [];
}

/**
 * 带护甲槽的背包快照（G-722）。
 *
 * `bot.inventory.items()` 只遍历 `inventoryStart..inventoryEnd`（9–44），
 * **不含 5–8 的护甲槽**，所以心跳用 `items()` 算护甲永远得到 0（服务端却是 20）。
 * 这里直接读 `bot.inventory.slots`（窗口 0 的完整 45 格）并补上 `slot` 下标。
 */
function currentInventorySlots() {
  try {
    const inventory = bot && bot.inventory;
    if (inventory && Array.isArray(inventory.slots)) {
      const out = [];
      for (let slot = 0; slot < inventory.slots.length; slot += 1) {
        const item = inventory.slots[slot];
        if (item) {
          out.push({ ...item, slot });
        }
      }
      return out;
    }
  } catch (error) {
    // 背包窗口重建期间可能抛：心跳不能因此失败，退化成 items()。
  }
  return currentInventoryItems();
}

/** 目标玩家实体（mineflayer 只跟踪视野内的实体：可能「在线但拿不到实体」）。 */
function findPlayerEntity(name) {
  if (!name || !bot) {
    return null;
  }
  const key = String(name).toLowerCase();
  const entities = bot.entities || {};
  for (const id of Object.keys(entities)) {
    const entity = entities[id];
    if (entity && entity.type === 'player'
      && String(entity.username || '').toLowerCase() === key) {
      return entity;
    }
  }
  return null;
}

/**
 * 目标是否在线：1.8 的 tab 列表（`bot.players`）含全部在线玩家，
 * 比「实体在不在视野里」可靠（1114 格外不会有实体，但人是真在线）。
 */
function isTargetOnline(name) {
  if (!name || !bot || !bot.players) {
    return false;
  }
  const key = String(name).toLowerCase();
  return Object.keys(bot.players).some((id) => {
    const entry = bot.players[id];
    return entry && String(entry.username || '').toLowerCase() === key;
  });
}

/** 目标离线/回来各报一次（G-717-C）。 */
function refreshTargetPresence(nowMs) {
  if (!targetPlayerArg) {
    return;
  }
  const event = targetPresence.update(nowMs, isTargetOnline(targetPlayerArg));
  if (event === 'offline') {
    console.log(`[brain] 目标玩家 ${targetPlayerArg} 不在线，先待机`);
  } else if (event === 'back') {
    console.log(`[brain] 目标玩家 ${targetPlayerArg} 回来了`);
  }
}

/**
 * 刷新「与目标的水平距离」（G-717-B），顺带维护本局最远距离，并结算两条提示：
 * 目标不在线（C）、远距离赶路（B）。返回当前距离或 null（没有目标/拿不到实体）。
 */
/**
 * G-723：朝向诊断。
 *
 * `yaw` 是 mineflayer 的自身朝向（弧度转度），`bearing` 是「从自己指向目标」的理想朝向，
 * 采用 1.8 的约定：0° 指 +Z、90° 指 −X（与 `arena` 的 `yawTowards` 同一公式）。
 * `move=forward` 时角色沿自身朝向走，所以 `yawErr` 接近 ±90/180 就等于「在绕圈」。
 */
function describeFacing() {
  const self = bot && bot.entity ? bot.entity : null;
  const entity = findPlayerEntity(brainState.target);
  const yaw = self && Number.isFinite(self.yaw) ? (self.yaw * 180) / Math.PI : null;
  if (!self || !self.position || !entity || !entity.position) {
    return { yaw, bearing: null, yawErr: null };
  }
  const dx = entity.position.x - self.position.x;
  const dz = entity.position.z - self.position.z;
  if (Math.abs(dx) < 1e-6 && Math.abs(dz) < 1e-6) {
    return { yaw, bearing: null, yawErr: null };
  }
  const bearing = (Math.atan2(-dx, dz) * 180) / Math.PI;
  let yawErr = bearing - (yaw === null ? 0 : yaw);
  while (yawErr > 180) {
    yawErr -= 360;
  }
  while (yawErr < -180) {
    yawErr += 360;
  }
  return { yaw: round1(yaw), bearing: round1(bearing), yawErr: round1(yawErr) };
}

function round1(value) {
  return Number.isFinite(value) ? Math.round(value * 10) / 10 : value;
}

function refreshTargetDistance(nowMs) {
  const self = bot && bot.entity ? bot.entity.position : null;
  const entity = findPlayerEntity(brainState.target);
  const dist = entity && entity.position && self
    ? horizontalDistance(self, entity.position)
    : null;
  brainState.targetDist = dist;
  if (dist !== null && dist > brainState.maxTargetDist) {
    brainState.maxTargetDist = dist;
  }
  refreshTargetPresence(nowMs);
  const farLine = farChaseNotifier.update(nowMs, dist, brainState.maxTargetDist);
  if (farLine) {
    console.log(farLine);
  }
  return dist;
}

/**
 * G-721 自检：大脑认为有目标、但观测里 `has_target==0` 时告警并计数。
 *
 * <p>为什么必须单独报：`encodeObservation` 在找不到目标时，`target_dx/dy/dz/distance`、
 * `target_health`、`opponent_speed`、`in_reach` 全部**静默 0**（`has_target` 是唯一的判别位）。
 * 策略会把「对手贴脸、满血、够得着」与「看不见」编码成不同向量，而「看不见」这一帧
 * 恰好长得像「对手在正中心 0 格、0 血」，于是做出任何动作都不会在日志里留下异常。
 *
 * <p>判读：`entities=1` → 世界没同步进来（客户端坐标还停在旧位置，G-721 真机形态）；
 * `entities>1` 而仍不可见 → 目标名字被 `allowed` / `isLikelyBot` 过滤掉了。
 *
 * @param {string|null} targetName 本帧大脑选定的目标（`pickTarget` 的返回值）
 * @param {ArrayLike<number>} features 本帧 39 维观测（G-731 从 37 扩到 39，前 37 位不变）
 * @param {number} nowMs
 * @returns {boolean} 本帧是否盲
 */
function recordBlindness(targetName, features, nowMs) {
  const rawEntities = bot && bot.entities ? Object.keys(bot.entities).length : 0;
  brainState.entitiesSeen = rawEntities;
  const { blind, line } = blindWatch.update(nowMs, {
    expected: Boolean(targetName),
    visible: hasVisibleTarget(features),
    entities: rawEntities,
  });
  if (blind) {
    brainState.blind += 1;
  }
  if (line) {
    console.warn(line);
  }
  return blind;
}

/**
 * 心跳（G-716/G-717）：每 5 秒一行，回答「大脑还在转吗、卡在哪、是在赶路还是在等套件」。
 *
 * - `decisions` 不涨 + `busy=1` → 有动作卡住（配合 `last=` 看是哪个）；
 * - `skips` 一直涨 → 每帧都因为 busy 被丢掉（就是「只决策一次就停」的形态）；
 * - `failures`/`timeouts` → 动作层真的在报错/超时；
 * - `blind`/`entities` → 盲决策帧数与原始实体数（G-721，见 `recordBlindness`）；
 * - `dist`/`armor` → 与目标的水平距离、身上护甲点数（`dist=-` 表示没有目标）；
 * - `waitingKit` → 板子还没发套件（不是模型问题）。
 */
function startBrainHeartbeat() {
  if (brainState.heartbeat) {
    return;
  }
  brainState.heartbeat = setInterval(() => {
    const now = Date.now();
    // 心跳里也要结算一次「赶路 / 目标不在线」：决策线程被动作占住时它同样要刷新。
    const dist = refreshTargetDistance(now);
    // G-723：朝向诊断——`yaw` 是自身朝向、`bearing` 是「指向目标」的理想朝向，
    // `yawErr` 是两者差（度，归一到 ±180）。move=forward 却贴不近目标时，
    // 先看 yawErr：接近 ±90/180 就说明它在绕圈而不是前进。
    const facing = describeFacing();
    console.log(formatHeartbeat({
      ...brainState,
      // 心跳里带 phase：`countdown` 时 decisions 不涨是**预期**（准备期不出手），
      // 没有这一列会被误读成「决策卡住」。
      phase: brainState.matchPhase ? brainState.matchPhase.phase : null,
      // G-735：`lastTick` 改成「整帧决策耗时」，另给最近 10 秒实际决策率 `hz`；
      // `superseded` 是被顶掉的待执行计划数（队列容量 1 的直接后果），
      // `applyMsMean` 是最近 50 个计划的执行耗时均值。
      lastTick: brainState.lastFrameMs,
      hz: decisionRate.hz(now),
      superseded: planExecutor.stats.superseded,
      applyMs: brainState.applyMs,
      applyMsMean: applyMean.mean(),
      dist,
      armor: armorPoints(currentInventorySlots()),
      yaw: facing.yaw,
      bearing: facing.bearing,
      yawErr: facing.yawErr,
      rawEntities: Object.keys((bot && bot.entities) || {}).length,
      targetVisible: Boolean(findPlayerEntity(brainState.target)),
    }, now));
  }, 5000);
  if (typeof brainState.heartbeat.unref === 'function') {
    brainState.heartbeat.unref();
  }
}

/** 用户名像 Bot 的一律不当对手（人机对战里要打的是真人）。 */
function isLikelyBot(name) {
  return /^(glad[_-]|gladbot|bot[_-]|goblin|ai_)/i.test(String(name || ''));
}

function selfName() {
  return (bot && bot.username) || username;
}

/**
 * 大脑本地决策用的有效 reach：软约束的 `reachOffset` 从这里扣。
 *
 * 只影响**自己的编码与解码**（观测里的 `in_reach` 与攻击判定），不改服务端的
 * `BotProfile.reach`——所以它是「自己给自己留余量」，不是改物理属性。
 */
function brainReach() {
  const offset = Number(soft.minReachOffset || 0);
  return Math.max(1.0, Number(activeProfile.reach || 0) - offset);
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
  // `Number(null) === 0`，直接转数会把「读不到血量」当成「对手 0 血」：
  // 无 RCON 时读实体血量失败 → 每 tick 记一次击杀、并把 target_health 压成 0。
  // 只有「有限数值且非 null/undefined/空串」才算真读数。
  if (health === null || health === undefined || health === '') {
    return;
  }
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

/** 计划序号（只用于日志/复盘对账，与协议 `seq` 无关）。 */
let planSeq = 0;

function nextPlanId() {
  planSeq += 1;
  return planSeq;
}

/**
 * 入队一个计划（G-735）：**不等待执行**，`superseded` 落到心跳。
 *
 * <p>队列容量 1：如果上一帧的计划还没开始执行，它会被这里顶掉——这是刻意的，
 * 新计划基于更新的观测，旧的只会让 Bot 对着两帧前的画面出手。
 */
function enqueuePlan(plan) {
  const outcome = planExecutor.submit(plan);
  brainState.superseded = planExecutor.stats.superseded;
  if (outcome.superseded > 0) {
    debugLog(`[brain] 计划替换 plan=${plan.planId}`
      + `（待执行被丢弃，累计 superseded=${planExecutor.stats.superseded}）`);
  }
  return outcome;
}

/**
 * v2 大脑一次决策（G-735：**只产计划，不执行**）。
 *
 * <p>为什么把执行摘出去：旧实现一路 `await` 到 `applyIntent` 返回，一次 `move` 就 200ms，
 * 20Hz 的定时器于是有 3/4 的帧被 `busy` 丢掉（真机 decisions=81/25s ≈ 4Hz）。
 * 现在这里只做「十头 → 意图字典」——`policy.actWithMask` 是纯 JS，毫秒级。
 *
 * <p>`brainCrossFrame.commit` 必须留在决策侧：下一帧编码要读「上一帧选了什么」，
 * 等执行完再写就会永远慢一拍（跨帧特征会整体错位）。
 */
function buildIntentPlan(features) {
  const mask = itemTable ? matchingMask(bot, itemTable) : null;
  const startedAt = Date.now();
  const heads = policy.actWithMask(features, mask ? mask.array : null, {
    greedy: true,
    epsilon: brainEpsilon,
    rng: Math.random,
  });
  brainState.inferMs = Date.now() - startedAt;
  const intent = decodeIntentVector(heads);
  const attackPicked = ATTACK_HEAD_INDEX >= 0 && Number(heads[ATTACK_HEAD_INDEX]) === 1;
  if (brainState.decisions === 1) {
    // 首帧留证：以后「模型到底发了什么」不用靠猜。
    console.log(`[brain] v2 首帧 payload.action=[${heads.join(',')}] → ${describeIntent(intent)}`);
  }
  brainCrossFrame.commit({ prevAction: heads, attacked: attackPicked });
  return makePlan({
    planId: nextPlanId(),
    kind: 'intent',
    intents: [intent],
    meta: { heads, attackPicked },
  });
}

/**
 * 执行一个计划（G-735）：串行跑完 `plan.intents`，同一时刻只会有一个。
 *
 * <p>这里保留 MATCH_PHASE 闸门作为**第二道**保险：计划是 50ms 前生成的，这期间阶段
 * 可能已经变成 `ended`（丢掉整个计划）或 `countdown`（只放行 look/idle）。
 * 「不打断正在执行的计划」与「阶段变了立刻停」在时间上重叠时，以**阶段**为准。
 */
async function executePlan(plan) {
  const current = session;
  const phase = brainState.matchPhase ? brainState.matchPhase.phase : null;
  const gate = phaseGate(phase);
  if (gate.ended) {
    planExecutor.clear();
    return { skipped: 'ended' };
  }
  if (!current || !bot) {
    return { skipped: 'no-session' };
  }
  // 协议动作按阶段裁剪（准备期只留 look/idle）；意图字典没有 type，整体按 gate 放行/丢弃。
  const steps = plan.kind === 'action'
    ? filterPhaseActions(plan.intents, phase)
    : (gate.hold ? [] : plan.intents);
  const startedAt = Date.now();
  let executed = 0;
  for (const step of steps) {
    brainState.seq += 1;
    const label = plan.kind === 'action' ? (step.type || 'action') : 'intent';
    const result = await runBrainAction(label, () => brainScheduler.run(brainState.seq, () => (
      plan.kind === 'action' ? current.dispatcher.dispatch(step) : current.applyIntent(step))));
    executed += 1;
    if (result && result.ok === false) {
      brainState.actionFailures += 1;
      if (brainState.actionFailures <= 5) {
        console.warn(`[brain] 动作失败 ${label}: ${result.code || '?'} ${result.reason || ''}`);
      }
    } else if (result && result.code === 'OUT_OF_REACH_CHASED') {
      // G-717-B：够不到先追一步是战术结果（ok=true），只在 --debug 下可见。
      const distance = Number(result.distance);
      debugLog(`[brain] 追击 ${label}`
        + ` distance=${Number.isFinite(distance) ? distance.toFixed(2) : '?'}`);
    }
    const attacked = plan.kind === 'action'
      ? step.type === 'attack'
      : Boolean(step && step.attack);
    if (attacked) {
      // 按**实际执行**计数（被顶掉的计划不算出手），口径与旧实现一致。
      brainState.attacks += 1;
      brainState.lastAttackAt = Date.now();
    }
  }
  if (executed === 0) {
    // 闸门把整条计划裁空了（准备期收到意图计划）：不算一次执行，也不要污染 applyMsMean。
    return { skipped: 'phase-hold' };
  }
  const cost = Date.now() - startedAt;
  brainState.applyMs = cost;
  applyMean.push(cost);
  return { steps: executed, ms: cost };
}

/**
 * 准备期的对手实体：优先已锁定的 `brainState.target`，其次实体表里最近的、看起来不是 Bot 的玩家。
 *
 * <p>准备期不走策略，只需要一个「朝谁站」的答案；拿不到就返回 null（只 idle）。
 */
function countdownTargetEntity() {
  const locked = findPlayerEntity(brainState.target);
  if (locked) {
    return locked;
  }
  const self = bot && bot.entity ? bot.entity : null;
  if (!self || !self.position) {
    return null;
  }
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const entity of Object.values(bot.entities || {})) {
    if (!entity || entity.type !== 'player' || !entity.position) {
      continue;
    }
    const name = String(entity.username || '');
    if (!name || name.toLowerCase() === selfName().toLowerCase() || isLikelyBot(name)) {
      continue;
    }
    const distance = entity.position.distanceTo(self.position);
    if (Number.isFinite(distance) && distance < bestDistance) {
      bestDistance = distance;
      best = entity;
    }
  }
  return best;
}

/**
 * 对局阶段闸门（`MATCH_PHASE`）——G-735 起只**产计划**，不再 `await` 执行。
 *
 * - `ended`：清空待执行计划并停手，返回 `true`（本帧不再决策）；
 * - `countdown` / `starting`：入队「朝向对手 + idle」的计划，返回 `true`；
 * - `fighting` / 没收到过阶段帧：返回 `false`，照常决策（老服务端行为不变）。
 *
 * <p>为什么准备期也要走计划队列：它的 look/idle 同样会 await（`bot.look` 走 physicsTick），
 * 留在决策线程里会把倒计时期间的决策率一起拖垮。
 */
function planMatchPhase() {
  const frame = brainState.matchPhase;
  if (!frame) {
    return false;
  }
  const gate = phaseGate(frame.phase);
  if (gate.ended) {
    if (!brainState.endedLogged) {
      brainState.endedLogged = true;
      console.log('[brain] 对局已结束：停止决策与出手');
    }
    // 待执行的计划立刻作废：它们可能是「还在 fighting 时」生成的。
    planExecutor.clear();
    return true;
  }
  if (!gate.hold) {
    if (brainState.countdownLoggedSecond !== null) {
      brainState.countdownLoggedSecond = null;
      console.log(`[brain] 阶段 ${frame.phase}（第 ${frame.round} 局）：恢复正常决策`);
    }
    return false;
  }
  const seconds = remainingSeconds(frame);
  const self = bot && bot.entity ? bot.entity.position : null;
  const target = countdownTargetEntity();
  const look = facingAction(self, target ? target.position : null);
  const wanted = look
    ? [look, { type: 'idle', params: { durationMs: 200 } }]
    : [{ type: 'idle', params: { durationMs: 200 } }];
  // filterPhaseActions 是第二道闸：即使 look/idle 之外的动作为将来加进来，也不会漏到准备期。
  const steps = filterPhaseActions(wanted, frame.phase);
  if (steps.length > 0) {
    enqueuePlan(makePlan({
      planId: nextPlanId(), kind: 'action', intents: steps, meta: { reason: 'phase' },
    }));
  }
  if (brainState.countdownLoggedSecond !== seconds) {
    brainState.countdownLoggedSecond = seconds;
    console.log(countdownLine(seconds));
  }
  return true;
}

/**
 * v1（11 路宏动作）的决策：解码 + 软约束 + 攻击节流，产出一个协议动作计划。
 *
 * <p>语义与 G-735 之前完全一致，唯一区别是「返回计划」而不是「就地 await 执行」。
 */
function buildActionPlan(observation, features, targetName, nowMs) {
  const inferStartedAt = Date.now();
  const index = brainEpsilon > 0 && Math.random() < brainEpsilon
    ? Math.floor(Math.random() * policy.numActions)
    : policy.greedy(features);
  brainState.inferMs = Date.now() - inferStartedAt;
  let actions = decodeAction(index, observation, selfName(), {
    allowed: [targetName],
    reach: brainReach(),
  });
  // 软约束：策略选了被禁的宏动作就在这里剔除（补 idle），不碰模型、不改变动作空间。
  const softDropped = actions.filter((action) => softDropReason(action, soft) !== null).length;
  if (softDropped > 0) {
    actions = applySoftActions(actions, soft);
    brainState.softDropped += softDropped;
    if (brainState.softDropped === softDropped) {
      console.log(`[soft] ${describeSoft(soft)}（被禁动作已改为 idle/位移）`);
    }
  }
  // 连续选 attack 说明已经贴身：按 1.8 的无敌帧（500ms）节流，别每 tick 刷空点击。
  const choseAttack = actions.some((action) => action.type === 'attack');
  brainState.attackStreak = choseAttack ? brainState.attackStreak + 1 : 0;
  const throttled = choseAttack && shouldThrottleAttack({
    streak: brainState.attackStreak,
    lastAttackAt: brainState.lastAttackAt,
    now: nowMs,
    cooldownMs: softAttackCooldownMs,
  });
  if (throttled) {
    // 只摘掉 attack，**不要**补 idle：`idle` 会清空前进/疾跑等控制状态，于是
    // 「贴身时每秒被节流十几次」的 Bot 会被钉在原地（G-716 真机：不动也不出手）。
    // 留空数组 = 这一帧什么都不发，既省动作预算，也保住正在按住的移动键。
    actions = actions.filter((action) => action.type !== 'attack');
    brainState.attacksThrottled += 1;
    if (brainState.attacksThrottled === 1) {
      console.log(`[brain] 敌人仍在无敌帧内，攻击节流到 ${softAttackCooldownMs}ms 一刀`
        + '（只摘掉攻击，保留移动/朝向）');
    }
  } else if (choseAttack && needsSwordSwap(bot.heldItem)) {
    // 手上不是剑就先切剑再打：直接 attack 只会空手挥（训练里模型看不到手持物）。
    actions = [{ type: 'use', params: { item: 'sword', holdMs: 50 } }, ...actions];
    brainState.swordSwaps += 1;
  }
  // G-711：v1 只维护两个帧计数（十头标签不适用 → prevAction 恒 null），
  // 「有没有出手」按**最终计划**算（节流/切剑之后），与训练侧口径一致。
  brainCrossFrame.commit({
    prevAction: null,
    attacked: actions.some((action) => action.type === 'attack'),
  });
  return makePlan({ planId: nextPlanId(), kind: 'action', intents: actions, meta: { index } });
}

/**
 * 一次决策（G-735：**同步、毫秒级**）：读观测 → 编码 → 推理 → 把计划丢进队列。
 *
 * <p>为什么要同步：只有决策侧不再 `await`，20Hz 的定时器才不会被「上一帧还没跑完」拖住。
 * 执行侧（{@link executePlan}）在自己的微任务里串行跑，两者通过容量 1 的队列解耦。
 */
function brainDecide() {
  if (!policy) {
    return;
  }
  // 三种「静默空转」分开计数（G-716）：否则「没日志、不动」到底是没 Bot、没会话，
  // 还是上一帧没回来，现场完全看不出来。
  if (!bot || !session || !bot.entity) {
    brainState.notReadySkips += 1;
    if (brainState.notReadySkips === 1 || brainState.notReadySkips % 200 === 0) {
      const why = !bot ? 'bot 未登录' : (!session ? '没有会话（通道 A 断开？）' : 'bot.entity 不存在');
      console.warn(`[brain] 跳过决策：${why}（累计 ${brainState.notReadySkips} 次）`);
      brainState.lastError = `跳过决策：${why}`;
    }
    return;
  }
  if (brainState.busy) {
    // 拆开之后这条基本不会触发（决策是同步的）；真触发了说明有人在决策里 await 了，
    // 必须打出来——否则又回到「只决策一次就停」的老毛病。
    brainState.busySkips += 1;
    if (brainState.busySkips === 1 || brainState.busySkips % 100 === 0) {
      console.warn(`[brain] 决策重入被跳过（上一帧还没结束，累计 ${brainState.busySkips} 次）`
        + ' —— 检查是否有新代码在决策路径里 await');
    }
    return;
  }
  brainState.busy = true;
  const frameStartedAt = Date.now();
  brainState.lastTickAt = frameStartedAt;
  // G-717：每帧结算「距离/目标在线/赶路提示」——它们与决策解耦，卡在动作里也要照常报。
  refreshTargetDistance(frameStartedAt);
  try {
    brainState.tick += 1;
    // 对局阶段闸门：准备期只站位、结束后全停。放在最前面——
    // 一旦进入准备期，策略连「选动作」的机会都不该有（用户实测：倒计时里 Bot 会提前出手）。
    if (planMatchPhase()) {
      return;
    }
    const observation = buildObservation(bot, {
      tick: brainState.tick,
      gameId,
      state: gameState,
      restrictions: restrictionCache.snapshot(),
      chat: chatLines,
      pendingNotices: notices,
    });
    const nowMs = Date.now();
    // G-711：先按「本帧血量」推进受伤计数，再编码——这样 36 维反映的是「这一帧挨打了吗」。
    brainCrossFrame.observeHealth((observation.self || {}).health);
    const targetName = pickTarget(observation, nowMs);
    if (!targetName) {
      // 没有目标 = 这一帧没有决策：只推进出手计数（不要谎称「刚出手」）。
      brainCrossFrame.advance();
      enqueuePlan(makePlan({
        planId: nextPlanId(),
        kind: 'action',
        intents: [{ type: 'idle', params: { durationMs: 180 } }],
        meta: { reason: 'no-target' },
      }));
      return;
    }
    // 没有 RCON 时退化成读实体血量（有就用，读不到就保持上次的采样值）。
    if (!rconSpec) {
      const entity = (observation.entities || []).find((item) => item
        && item.kind === 'player'
        && String(item.name || '').toLowerCase() === targetName.toLowerCase());
      recordOpponentHealth(entityHealth(entity));
    }
    const features = encodeObservation(observation, brainEncodeOptions(targetName, nowMs));
    brainState.obsMs = Date.now() - frameStartedAt;
    // G-721：编码完立刻自检「目标看不看得见」——这是 7–13/19–22 维静默 0 的唯一告警点。
    recordBlindness(targetName, features, nowMs);
    // G-721：把编码向量落盘，供与训练侧 `spaces.encode_observation` 逐维 diff。
    // 采样而不是「前 40 帧」：刚登录的世界还没同步实体，前几帧的 `entities` 常为空，
    // 只抓开头会把「启动竞态」误判成「编码错误」（2026-11-25 真机踩过）。
    if (dumpObsPath && dumpObsCount < DUMP_OBS_LIMIT
        && (brainState.tick % DUMP_OBS_EVERY === 1 || (observation.entities || []).length === 0)) {
      dumpObsCount += 1;
      try {
        fs.appendFileSync(dumpObsPath, `${JSON.stringify({
          t: nowMs,
          tick: brainState.tick,
          actionSpace: policy.actionSpace,
          target: targetName,
          distance: brainState.targetDistance ?? null,
          opponentHealth: brainState.opponentHealth,
          entityNames: (observation.entities || []).map((item) => `${item && item.kind}:${item && item.name}:${item && item.distance}`),
          rawEntities: Object.keys(bot.entities || {}).length,
          features: Array.from(features),
        })}\n`, 'utf8');
      } catch (error) {
        console.warn(`[brain] 观测落盘失败：${error.message}`);
      }
    }
    const plan = policy.actionSpace === 'v2'
      ? buildIntentPlan(features)
      : buildActionPlan(observation, features, targetName, nowMs);
    if (plan) {
      brainState.decisions += 1;
      decisionRate.mark(nowMs);
      enqueuePlan(plan);
    }
  } catch (error) {
    brainState.lastError = `决策失败: ${error?.message ?? error}`;
    console.warn(`[brain] 决策失败: ${error?.message ?? error}`);
  } finally {
    // 注意：不要清 `lastAction`——它现在描述的是**执行侧**正在跑的那一步。
    brainState.busy = false;
    brainState.lastFrameMs = Date.now() - frameStartedAt;
  }
}

/** 编码选项（v1/v2 共用，抽出来是为了让 buildActionPlan 也能复用同一份口径）。 */
function brainEncodeOptions(targetName, nowMs) {
  return {
    username: selfName(),
    stepFraction: 0,
    opponentHealth: brainState.opponentHealth,
    allowed: [targetName],
    reach: brainReach(),
    sinceAttack: brainState.lastAttackAt ? (nowMs - brainState.lastAttackAt) / 1000 : 5,
    // G-711：25–36 维（上一帧十头动作 + 距上次出手/受伤帧数）。
    // v1 下 `prevAction` 恒为 null（十头标签不适用），两个帧计数照常工作。
    ...brainCrossFrame.options(),
  };
}

function startBrain() {
  if (!brainEnabled || !policy) {
    return;
  }
  if (brainState.timer) {
    clearInterval(brainState.timer);
  }
  // G-711：新一局/重连：上一帧动作与两个帧计数都从头开始（与训练侧 reset() 同口径）。
  brainCrossFrame.reset();
  // G-717：每局重新开始统计（套件提示每局一次、最远距离每局归零、目标在线状态重算）。
  brainState.kitWaitLogged = false;
  brainState.maxTargetDist = 0;
  brainState.targetDist = null;
  targetPresence.reset(targetPlayerArg);
  farChaseNotifier.reset();
  // G-721：告警节流状态也随局复位（累计计数 `blind` 保留，用于看整场趋势）。
  blindWatch.reset();
  const interval = Math.max(50, Math.floor(1000 / brainHz));
  // G-735：**绝对时刻 + 漂移补偿**的调度器，而不是裸 `setInterval(50)`。
  //
  // 真机实测（Windows）：`setInterval(50)` 的实际周期是 **62.5ms**——Windows 的定时器
  // 量化粒度是 15.625ms，50ms 被向上取整到 4 个粒度。于是 20Hz 的脑子只跑到 16Hz，
  // 且误差逐帧累积（真机 decisions 81/5s）。这里改成「10ms 轮询 + 记住目标时刻」：
  //   - 到点就跑，跑完把目标时刻 +50ms（不受本次迟到影响，误差不累积）；
  //   - 迟到不足一个周期 → 下一次立刻补上；迟到超过一个周期（GC/长动作）→ 重同步，
  //     丢掉落后的帧（补跑一串只会让动作扎堆，不会让状态更准）。
  // 轮询间隔取 `interval/5`（50ms 时 = 10ms，≥5ms），CPU 代价可忽略。
  const pollMs = Math.max(5, Math.min(20, Math.floor(interval / 5) || 5));
  let nextAt = Date.now() + interval;
  brainState.timer = setInterval(() => {
    const now = Date.now();
    if (now < nextAt) {
      return;
    }
    if (now - nextAt > interval) {
      nextAt = now + interval;              // 落后太多：重同步
    } else {
      nextAt += interval;                   // 正常：按绝对时刻推进（补偿本次迟到）
    }
    // G-735：决策是**同步**的（毫秒级），异常直接 try/catch，没有 promise 要接。
    try {
      brainDecide();
    } catch (error) {
      brainState.lastError = `tick 异常: ${error?.message ?? error}`;
      console.warn(`[brain] tick 异常: ${error?.message ?? error}`);
    }
  }, pollMs);
  if (brainState.timer.unref) {
    brainState.timer.unref();
  }
  // G-716：心跳（decisions/attacks/failures/busy/last）——「只决策一次就停」靠它现形。
  startBrainHeartbeat();
  console.log(`[brain] 已启动 ${brainHz}Hz，目标=${targetPlayerArg || '自动（最近的非 Bot 玩家）'}`
    + ` epsilon=${brainEpsilon} attackCooldown=${softAttackCooldownMs}ms`
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
  if (brainState.heartbeat) {
    clearInterval(brainState.heartbeat);
    brainState.heartbeat = null;
  }
  if (brainState.poller) {
    clearInterval(brainState.poller);
    brainState.poller = null;
  }
  if (brainState.rcon) {
    brainState.rcon.close();
    brainState.rcon = null;
  }
  // G-735：停脑时把待执行的计划丢掉（否则断线后还在按旧计划挥刀），指标清零。
  planExecutor.clear();
  decisionRate.reset();
  applyMean.reset();
  brainState.busy = false;
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
    // G-716：卡死/超时也要进汇总，否则「0 失败但 0 伤害」看起来像策略不行为。
    actionTimeouts: brainState.actionTimeouts,
    busySkips: brainState.busySkips,
    // G-717-A：等套件的次数（不是失败；真失败在 actionFailures 里）。
    waitingKit: brainState.waitingKit,
    softDropped: brainState.softDropped,
    soft: describeSoft(soft),
    brainHz,
    attackCooldownMs: softAttackCooldownMs,
  }));
}

client.on('welcome', (payload) => {
  const uuid = payload.playerUuid || '(尚未绑定，等待 PlayerJoin)';
  console.log(`[gladiator] ready session=${payload.sessionId} playerUuid=${uuid}`);
});

/**
 * 对局阶段帧（插件每秒最多一帧）：只更新状态与日志，动作抑制在 {@link planMatchPhase}。
 *
 * <p>阶段**变化**时才打一行；每秒重复的同一阶段不刷屏（`planMatchPhase` 里另有按秒递减的
 * 准备期日志）。
 */
client.on('match-phase', (frame) => {
  const previous = brainState.matchPhase ? brainState.matchPhase.phase : null;
  brainState.matchPhase = frame;
  if (frame.phase === previous) {
    return;
  }
  brainState.countdownLoggedSecond = null;
  brainState.endedLogged = false;
  const seconds = remainingSeconds(frame);
  console.log(`[brain] 阶段 ${frame.phase ?? '-'}（第 ${frame.round} 局`
    + `${seconds >= 0 ? `，剩余 ${seconds}s` : ''}）`);
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
    // 训练侧靠这条事件判死：观测轮询会跳过 1.8 的「死→自动重生」（只有 2 tick）。
    envServer.broadcastState({
      state: gameState,
      previous: payload.previous ?? previous,
      reason: payload.reason,
      gameId,
      at: Date.now(),
    });
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
      // G-717-A：每局（含重生/重连后的新一轮）只提醒一次「还没拿到剑」。
      current.on('spawn', () => {
        brainState.kitWaitLogged = false;
      });
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
        // 意图执行状态（按住的方向键/右键计时器）随连接一起丢弃，重连后从干净状态开始。
        resetIntentState(current);
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
