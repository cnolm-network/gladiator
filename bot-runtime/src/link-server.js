/**
 * 通道 B（Layer 2 ↔ Layer 3）WebSocket 服务端，协议 §4。
 *
 * <p>Layer 2 是服务端，Layer 3 训练环境作为客户端连入：HELLO → WELCOME，
 * 之后 L2 按 controlHz 推送 OBSERVATION，L3 发 ACTION / RESET。
 *
 * <p>动作经 ActionScheduler（限流 + seq 去重）后交给 ActionDispatcher，
 * 每个 ACTION 都回 ACTION_RESULT（replyTo=请求 seq），失败带结构化 code。
 */

import { WebSocketServer } from 'ws';
import { ACTION_TYPES } from './actions.js';
import { OBSERVATION_FIELDS, buildObservation } from './observation.js';
import { ActionScheduler } from './scheduler.js';

const PROTOCOL = 'gladiator-env';

function nowMs() {
  return Date.now();
}

export class EnvLinkServer {
  /**
   * @param {object} options
   * @param {number} options.port            监听端口，默认 8766
   * @param {string} [options.host]          绑定地址，默认 127.0.0.1（只允许回环）
   * @param {number} [options.controlHz]     观测频率，默认 10
   * @param {number} [options.maxActionsPerSecond] 每 Bot 动作上限，默认 10
   * @param {(botId:string)=>object|null} options.resolveSession 返回会话对象
   * @param {object} [options.logger]
   */
  constructor({
    port = 8766,
    host = '127.0.0.1',
    controlHz = 10,
    maxActionsPerSecond = 10,
    resolveSession,
    logger = console,
  } = {}) {
    this.port = port;
    this.host = host;
    this.controlHz = Math.max(1, Math.min(20, controlHz));
    this.maxActionsPerSecond = maxActionsPerSecond;
    this.resolveSession = resolveSession || (() => null);
    this.logger = logger;
    this.server = null;
    this.clients = new Set();
    this.tick = 0;
    this.scheds = new Map();
  }

  get address() {
    return this.server && this.server.address ? this.server.address() : null;
  }

  sched(botId) {
    if (!this.scheds.has(botId)) {
      this.scheds.set(botId, new ActionScheduler({ maxPerSecond: this.maxActionsPerSecond }));
    }
    return this.scheds.get(botId);
  }

  start() {
    if (this.host !== '127.0.0.1' && this.host !== '::1' && this.host !== 'localhost') {
      throw new Error(`EnvLinkServer 仅允许绑定回环地址，拒绝: ${this.host}`);
    }
    this.server = new WebSocketServer({ host: this.host, port: this.port });
    this.server.on('listening', () => {
      this.logger.info(`[env] 通道 B listening on ws://${this.host}:${this.port} (controlHz=${this.controlHz})`);
    });
    this.server.on('connection', (socket) => this.onConnection(socket));
    this.server.on('error', (error) => {
      this.logger.error(`[env] 通道 B 服务端错误: ${error.message}`);
    });
    return this;
  }

  onConnection(socket) {
    const client = {
      socket,
      botId: null,
      seq: 0,
      timer: null,
      lastActions: [],
    };
    this.clients.add(client);
    socket.on('message', (raw) => {
      let frame;
      try {
        frame = JSON.parse(String(raw));
      } catch {
        this.reject(client, null, 'INTERNAL', '帧不是合法 JSON');
        return;
      }
      this.onFrame(client, frame);
    });
    socket.on('close', () => {
      if (client.timer) {
        clearInterval(client.timer);
      }
      this.clients.delete(client);
      this.logger.info(`[env] 通道 B 客户端断开 botId=${client.botId || '-'}`);
    });
    socket.on('error', (error) => {
      this.logger.warn(`[env] 通道 B 客户端错误: ${error.message}`);
    });
  }

  onFrame(client, frame) {
    const type = frame && frame.type;
    const payload = (frame && frame.payload) || {};
    if (type === 'HELLO') {
      if (payload.v !== undefined && payload.v !== 1) {
        this.reject(client, frame.seq, 'VERSION_UNSUPPORTED', `不支持的 v=${payload.v}`);
        return;
      }
      if (payload.protocol && payload.protocol !== PROTOCOL) {
        this.reject(client, frame.seq, 'VERSION_UNSUPPORTED', `不支持的 protocol=${payload.protocol}`);
        return;
      }
      const botId = String(payload.botId || '');
      if (!botId) {
        this.reject(client, frame.seq, 'INTERNAL', 'HELLO 缺少 botId');
        return;
      }
      client.botId = botId;
      this.send(client, 'WELCOME', {
        observationSpec: { encoding: 'json-v1', fields: OBSERVATION_FIELDS },
        actionSpec: { encoding: 'json-v1', types: ACTION_TYPES },
        controlHz: this.controlHz,
      }, frame.seq);
      this.startObservation(client);
      return;
    }
    if (!client.botId) {
      this.reject(client, frame.seq, 'SESSION_UNKNOWN', 'HELLO 之前不能发送其他消息');
      return;
    }
    if (type === 'PING') {
      this.send(client, 'PONG', {}, frame.seq);
      return;
    }
    if (type === 'ACTION') {
      this.onAction(client, frame).catch((error) => {
        this.send(client, 'ACTION_RESULT', {
          ok: false, code: 'INTERNAL', reason: error.message,
        }, frame.seq);
      });
      return;
    }
    if (type === 'RESET') {
      const session = this.resolveSession(client.botId);
      this.send(client, 'RESET_DONE', {
        state: session ? session.state() : 'Alive',
        tick: this.tick,
        timeoutMs: 5000,
      }, frame.seq);
      return;
    }
    this.reject(client, frame.seq, 'ACTION_UNSUPPORTED', `未知消息类型: ${type}`);
  }

  async onAction(client, frame) {
    const seq = frame.seq;
    const payload = frame.payload || {};
    if (!Number.isInteger(seq)) {
      this.send(client, 'ACTION_RESULT', { ok: false, code: 'INTERNAL', reason: 'seq 必须是整数' }, null);
      return;
    }
    const session = this.resolveSession(client.botId);
    if (!session || !session.dispatcher) {
      this.send(client, 'ACTION_RESULT', { ok: false, code: 'BOT_NOT_READY', reason: 'Bot 尚未登录' }, seq);
      return;
    }
    const action = { type: payload.type, params: payload.params || {} };
    const result = await this.sched(client.botId).run(seq, () => session.dispatcher.dispatch(action));
    this.send(client, 'ACTION_RESULT', result, seq);
    if (!result.ok && result.code === 'RULE_DENIED' && typeof session.onRuleDenied === 'function') {
      session.onRuleDenied(action, result.reason);
    }
  }

  startObservation(client) {
    if (client.timer) {
      clearInterval(client.timer);
    }
    const interval = Math.max(50, Math.floor(1000 / this.controlHz));
    client.timer = setInterval(() => {
      const session = this.resolveSession(client.botId);
      if (!session || !session.bot) {
        return;
      }
      // 背压：客户端消费不过来时丢帧而不是堆积内存（协议 §5）
      if (client.socket.bufferedAmount > 1024 * 1024) {
        return;
      }
      this.tick += 1;
      const observation = buildObservation(session.bot, {
        tick: this.tick,
        gameId: session.gameId ? session.gameId() : '',
        state: session.state ? session.state() : 'Alive',
        restrictions: session.restrictionCache ? session.restrictionCache.snapshot() : [],
        chat: session.chatLines ? session.chatLines() : [],
        pendingNotices: session.notices ? session.notices() : [],
      });
      this.send(client, 'OBSERVATION', observation, null);
    }, interval);
    if (typeof client.timer.unref === 'function') {
      client.timer.unref();
    }
  }

  send(client, type, payload, replyTo = null) {
    if (!client.socket || client.socket.readyState !== 1) {
      return false;
    }
    client.seq += 1;
    const frame = { v: 1, type, seq: client.seq, ts: nowMs(), replyTo };
    frame.payload = payload;
    try {
      client.socket.send(JSON.stringify(frame));
      return true;
    } catch (error) {
      this.logger.warn(`[env] 发送 ${type} 失败: ${error.message}`);
      return false;
    }
  }

  reject(client, seq, code, message) {
    this.send(client, 'REJECT', { code, message }, seq);
    this.logger.warn(`[env] 拒绝: ${code} ${message}`);
  }

  stop() {
    for (const client of this.clients) {
      if (client.timer) {
        clearInterval(client.timer);
      }
      try {
        // terminate 而不是 close：close 要做握手，进程退出/测试收尾时等不到对端，句柄会一直挂着。
        client.socket.terminate();
      } catch {
        // 忽略关闭竞态
      }
    }
    this.clients.clear();
    this.scheds.clear();
    if (this.server) {
      for (const socket of this.server.clients || []) {
        try {
          socket.terminate();
        } catch {
          // ignore
        }
      }
      this.server.close();
      this.server = null;
    }
  }
}
