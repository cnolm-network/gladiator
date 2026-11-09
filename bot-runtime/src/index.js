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
import { ActionDispatcher } from './actions.js';
import { LinkClient } from './connection.js';
import { EnvLinkServer } from './link-server.js';
import { describeLoginFailure, launchBot } from './minecraft.js';
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
    gameState = payload.state ?? gameState;
    if (payload.gameId) {
      gameId = payload.gameId;
    }
    console.log(`[gladiator] STATE ${payload.previous ?? '?'} → ${gameState}${payload.reason ? ` (${payload.reason})` : ''}`);
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
            if (worn.length && worn.every((name) => !name.startsWith('!'))) {
              if (worn.length >= 4) {
                clearInterval(equipTimer);
                equipTimer = null;
              }
              if (worn.join(',') !== lastWorn) {
                lastWorn = worn.join(',');
                console.log(`[gladiator] 已穿戴护甲: ${worn.join(', ')}`);
              }
            }
          })
          .catch((error) => console.warn(`[gladiator] 穿护甲失败：${error?.message ?? error}`))
          .finally(() => {
            if (equipAttempts > 60 && equipTimer) {
              clearInterval(equipTimer);
              equipTimer = null;
            }
          });
      }, 2000);
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
