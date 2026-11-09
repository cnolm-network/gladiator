/**
 * 通道 A 客户端（G-106）：到 Gladiator 插件的 WebSocket 长连接 + 指数退避重连。
 *
 * 设计约束（见 docs/protocol.md 与 ARCHITECTURE.md「故障处理」）：
 * - 只负责链路：握手、信封编解码、心跳、断线重连；不碰任何游戏逻辑。
 * - 断线后按 1s → 2s → 4s … 30s 上限退避重连，重连成功必须重新 HELLO。
 * - AUTH_FAILED / VERSION_UNSUPPORTED 属于配置错误，重试没有意义，直接停止并上报。
 * - 服务端重启后端口暂时不可用，重连会自然退避等待，不需要特殊处理。
 */
import { EventEmitter } from 'node:events';
import WebSocket from 'ws';

/** 指数退避（带轻微抖动，避免多个进程同时重连打爆插件）。 */
export class Backoff {
  constructor({
    initialMs = 1000,
    maxMs = 30000,
    factor = 2,
    jitter = 0.2,
    random = Math.random,
  } = {}) {
    this.initialMs = Math.max(1, initialMs);
    this.maxMs = Math.max(this.initialMs, maxMs);
    this.factor = Math.max(1, factor);
    this.jitter = Math.max(0, Math.min(1, jitter));
    this.random = random;
    this.attempt = 0;
  }

  /** 返回本次等待毫秒数，并把尝试次数 +1。 */
  nextDelay() {
    const base = Math.min(
      this.maxMs,
      this.initialMs * this.factor ** this.attempt,
    );
    this.attempt += 1;
    const spread = base * this.jitter;
    const offset = (this.random() * 2 - 1) * spread;
    return Math.max(0, Math.round(base + offset));
  }

  reset() {
    this.attempt = 0;
  }
}

/** REJECT 中这些错误码意味着重连也不会成功。 */
const FATAL_CODES = new Set(['AUTH_FAILED', 'VERSION_UNSUPPORTED']);

export class LinkClient extends EventEmitter {
  constructor({
    url = 'ws://127.0.0.1:8765/gladiator',
    token = '',
    botId,
    username,
    mcVersion = '1.8.8',
    runtimeVersion = '0.1.0',
    heartbeatMs = 10000,
    backoff = new Backoff(),
    logger = console,
    WebSocketImpl = WebSocket,
  } = {}) {
    super();
    if (!botId) throw new Error('LinkClient 需要 botId');
    this.url = url;
    this.token = token;
    this.botId = botId;
    this.username = username ?? botId;
    this.mcVersion = mcVersion;
    this.runtimeVersion = runtimeVersion;
    this.heartbeatMs = heartbeatMs;
    this.backoff = backoff;
    this.logger = logger;
    this.WebSocketImpl = WebSocketImpl;

    this.seq = 0;
    this.ws = null;
    this.stopped = false;
    this.fatal = null;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
  }

  /** 开始连接；重复调用是幂等的。 */
  start() {
    if (this.ws || this.stopped) return this;
    this._connect();
    return this;
  }

  /** 主动关闭：不再重连、清掉所有定时器。 */
  stop() {
    this.stopped = true;
    this._clearTimers();
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      try {
        ws.close();
      } catch {
        // ignore
      }
    }
    return this;
  }

  /**
   * 仅用于故障注入测试：直接掐断底层 socket，模拟网络闪断。
   * 与 stop() 不同，这里不会置 stopped，所以会自动走退避重连。
   */
  debugDropConnection() {
    if (!this.ws) return false;
    try {
      this.ws.terminate();
    } catch {
      return false;
    }
    return true;
  }

  send(type, payload = {}, replyTo = null) {
    if (!this.ws || this.ws.readyState !== this.WebSocketImpl.OPEN) {
      return false;
    }
    this.seq += 1;
    this.ws.send(JSON.stringify({
      v: 1,
      type,
      seq: this.seq,
      ts: Date.now(),
      botId: type === 'HELLO' ? '' : this.botId,
      replyTo,
      payload,
    }));
    return true;
  }

  // ------------------------------------------------------------------
  // 内部
  // ------------------------------------------------------------------

  _connect() {
    if (this.stopped) return;
    this.logger.info?.(`[link] connecting ${this.url}`);
    const ws = new this.WebSocketImpl(this.url, { handshakeTimeout: 5000 });
    this.ws = ws;

    ws.on('open', () => this._onOpen(ws));
    ws.on('message', (raw) => this._onMessage(raw));
    ws.on('error', (err) => {
      this.logger.warn?.(`[link] socket error: ${err.message}`);
      this.emit('socket-error', err);
    });
    ws.on('close', (code, reason) => this._onClose(code, reason));
  }

  _onOpen(ws) {
    if (ws !== this.ws) return;
    this.backoff.reset();
    this.emit('open');
    this.send('HELLO', {
      token: this.token,
      botId: this.botId,
      username: this.username,
      mcVersion: this.mcVersion,
      runtimeVersion: this.runtimeVersion,
    });
    this._startHeartbeat();
  }

  _startHeartbeat() {
    this._clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.send('PING', {});
    }, this.heartbeatMs);
    if (typeof this.heartbeatTimer.unref === 'function') {
      this.heartbeatTimer.unref();
    }
  }

  _onMessage(raw) {
    let envelope;
    try {
      envelope = JSON.parse(raw.toString());
    } catch (err) {
      this.logger.warn?.(`[link] malformed JSON: ${err.message}`);
      this.emit('protocol-error', err);
      return;
    }
    this.emit('envelope', envelope);
    switch (envelope.type) {
      case 'PONG':
        this.emit('pong', envelope);
        return;
      case 'WELCOME':
        this.logger.info?.(
          `[link] WELCOME session=${envelope.payload?.sessionId}`
          + ` playerUuid=${envelope.payload?.playerUuid || '(pending)'}`,
        );
        this.emit('welcome', envelope.payload ?? {}, envelope);
        return;
      case 'REJECT': {
        const payload = envelope.payload ?? {};
        this.logger.error?.(
          `[link] REJECT ${payload.code}: ${payload.message}`,
        );
        this.emit('reject', payload, envelope);
        if (FATAL_CODES.has(payload.code)) {
          this.fatal = payload.code;
          // 让当前连接自然关闭，但不再重连。
          this.stopped = true;
          this._clearTimers();
        }
        return;
      }
      default:
        this.emit('message', envelope);
    }
  }

  _onClose(code, reason) {
    this._clearHeartbeat();
    this.ws = null;
    this.emit('close', { code, reason: reason?.toString?.() ?? '' });
    if (this.stopped) {
      this.emit('stopped');
      return;
    }
    const delay = this.backoff.nextDelay();
    this.logger.warn?.(
      `[link] disconnected (code=${code}); reconnecting in ${delay}ms`
      + ` (attempt ${this.backoff.attempt})`,
    );
    this.emit('reconnecting', { delayMs: delay, attempt: this.backoff.attempt });
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this._connect();
    }, delay);
    // 注意：不能 unref。断线期间如果进程只剩这个计时器，Node 会直接退出，
    // 表现为"运行时在服务端重启时悄悄消失"（G-106 实测问题）。
  }

  _clearHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  _clearTimers() {
    this._clearHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
