/**
 * mineflayer 建连与登录失败归类（G-106）。
 *
 * 登录失败的原因对使用者必须可读：版本不符、online-mode 鉴权失败、白名单、
 * 服务器不可达是四类最常见的配置问题，直接映射成稳定的 code 回传插件
 * （SPAWN_FAILED），插件再以 GladiatorBotFailedEvent 交给调用方。
 */
import mineflayer from 'mineflayer';

/** SPAWN_FAILED 的稳定错误码；协议文档 docs/protocol.md §2.4。 */
export const SPAWN_FAILURE_CODES = Object.freeze({
  VERSION_MISMATCH: 'VERSION_MISMATCH',
  ONLINE_MODE: 'ONLINE_MODE',
  NOT_WHITELISTED: 'NOT_WHITELISTED',
  SERVER_UNREACHABLE: 'SERVER_UNREACHABLE',
  LOGIN_FAILED: 'LOGIN_FAILED',
});

const PATTERNS = [
  {
    code: SPAWN_FAILURE_CODES.ONLINE_MODE,
    test: /failed to verify username|invalid session|online mode|not authenticated|unauthenticated/i,
    hint: '服务端开启了 online-mode，但 Bot 以离线身份登录（或反之）。'
      + '请让服务端 online-mode=false，或为运行配置有效的正版账号。',
  },
  {
    code: SPAWN_FAILURE_CODES.NOT_WHITELISTED,
    test: /not white-?listed|white-?list/i,
    hint: 'Bot 不在服务端白名单中：把该用户名加入 whitelist.json 后重试。',
  },
  {
    code: SPAWN_FAILURE_CODES.VERSION_MISMATCH,
    test: /unsupported version|version mismatch|incompatible ?version|protocol version/i,
    hint: 'mineflayer 的 version 与服务端协议不一致：1.8.9 服务端请用 version "1.8.8"（协议 47）。',
  },
  {
    code: SPAWN_FAILURE_CODES.SERVER_UNREACHABLE,
    test: /ECONNREFUSED|EHOSTUNREACH|ENOTFOUND|ETIMEDOUT|getaddrinfo/i,
    hint: '无法连接到 Minecraft 服务端：确认地址/端口且服务端已启动。',
  },
];

/**
 * 把任意错误/踢出原因归类为 { code, message }；message 面向人阅读。
 * @param {unknown} error
 * @returns {{code: string, message: string}}
 */
export function describeLoginFailure(error) {
  const raw = typeof error === 'string'
    ? error
    : (error?.message ?? error?.reason ?? String(error ?? 'unknown'));
  const text = String(raw);
  for (const pattern of PATTERNS) {
    if (pattern.test.test(text)) {
      return { code: pattern.code, message: `${pattern.hint}（原始信息: ${text}）` };
    }
  }
  return {
    code: SPAWN_FAILURE_CODES.LOGIN_FAILED,
    message: `Bot 登录失败：${text}`,
  };
}

/**
 * 以真实协议登录服务器（不伪造任何服务端连接）。
 *
 * @returns {Promise<import('mineflayer').Bot>} mineflayer 实例
 */
export function launchBot({
  host,
  port = 25565,
  username,
  version = '1.8.8',
  auth = 'offline',
  logger = console,
  timeoutMs = 30000,
}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };

    const timer = setTimeout(() => {
      finish(reject, new Error(`登录超时（${timeoutMs}ms）：服务端未完成握手`));
    }, timeoutMs);

    let bot;
    try {
      bot = mineflayer.createBot({ host, port, username, version, auth });
    } catch (err) {
      finish(reject, err);
      return;
    }

    bot.once('spawn', () => {
      logger.info?.(`[mc] ${username} 已生成于 ${host}:${port}（version ${bot.version}）`);
      finish(resolve, bot);
    });
    bot.once('error', (err) => {
      logger.warn?.(`[mc] ${username} 连接错误: ${err.message}`);
      finish(reject, err);
    });
    bot.once('kicked', (reason) => {
      const text = typeof reason === 'string' ? reason : JSON.stringify(reason);
      finish(reject, new Error(text));
    });
    bot.once('end', (reason) => {
      finish(reject, new Error(`连接结束：${reason ?? 'unknown'}`));
    });
  });
}
