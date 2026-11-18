/**
 * 极简 RCON 客户端（Source RCON），只用于读人类玩家遥测。
 *
 * <p>为什么运行时需要它：mineflayer 拿不到**其他玩家**的血量（1.8 的玩家血量不在
 * 公开实体元数据里稳定暴露），而策略的 `target_health` 是必需特征。块 B 提供了
 * `/gladiator stats <player>` 单行输出，运行时可选用 `--rcon` 轮询它补齐。
 *
 * <p>只在回环地址上使用，与 Dashboard / 训练层用的是同一套协议实现。
 */

import net from 'node:net';

export const SERVERDATA_AUTH = 3;
export const SERVERDATA_AUTH_RESPONSE = 2;
export const SERVERDATA_EXECCOMMAND = 2;
export const SERVERDATA_RESPONSE_VALUE = 0;

/** 去掉 ANSI 转义与 Minecraft 的 § 颜色码。 */
export function stripFormatting(text) {
  return String(text ?? '')
    .replace(/\x1b\[[0-9;]*m/g, '')
    .replace(/[§\u00a7].?/g, '');
}

/**
 * 解析 `/gladiator stats <player>` 的单行输出。
 *
 * 形如：
 * `name=Steve health=20.0 food=20 saturation=5.0 armor=16.0 x=0.5 y=4.0 z=0.5
 *  yaw=0.0 pitch=0.0 sprinting=false blocking=false onGround=true hurt=0.0`
 *
 * @returns {Record<string, string|number|boolean>|null} 解析失败返回 null
 */
export function parseStatsLine(raw) {
  const text = stripFormatting(raw).trim();
  if (!text || text.includes('stats 失败')) {
    return null;
  }
  const out = {};
  for (const token of text.split(/\s+/)) {
    const index = token.indexOf('=');
    if (index <= 0) {
      continue;
    }
    const key = token.slice(0, index);
    const value = token.slice(index + 1);
    if (value === 'true' || value === 'false') {
      out[key] = value === 'true';
      continue;
    }
    const numeric = Number(value);
    out[key] = Number.isFinite(numeric) ? numeric : value;
  }
  return typeof out.name === 'string' || Number.isFinite(out.health) ? out : null;
}

export class RconClient {
  constructor({ host = '127.0.0.1', port = 25575, password = 'devpass', timeout = 3000 } = {}) {
    this.host = host;
    this.port = Number(port);
    this.password = password;
    this.timeout = timeout;
    this.socket = null;
    this.buffer = Buffer.alloc(0);
    this.pending = [];
    this.requestId = 0;
    this.ready = false;
  }

  connect() {
    return new Promise((resolve, reject) => {
      const socket = net.createConnection({ host: this.host, port: this.port });
      this.socket = socket;
      socket.setTimeout(this.timeout);
      const onError = (error) => {
        cleanup();
        reject(error);
      };
      const onTimeout = () => {
        cleanup();
        socket.destroy();
        reject(new Error(`RCON 连接超时（${this.host}:${this.port}）`));
      };
      const cleanup = () => {
        socket.removeListener('error', onError);
        socket.removeListener('timeout', onTimeout);
        socket.removeListener('data', onData);
      };
      const onData = (chunk) => {
        this.buffer = Buffer.concat([this.buffer, chunk]);
        this._drain();
      };
      socket.on('data', onData);
      socket.on('error', onError);
      socket.on('timeout', onTimeout);
      socket.on('close', () => {
        this.ready = false;
        const waiting = this.pending.splice(0);
        for (const entry of waiting) {
          entry.resolve(null);
        }
      });
      socket.once('connect', () => {
        this.requestId += 1;
        this._write(SERVERDATA_AUTH, this.requestId, this.password);
        this.authId = this.requestId;
      });
      // 认证成功由 _drain 里的 AUTH_RESPONSE 帧驱动（id 与 authId 相同视为成功）。
      this.onAuthenticated = () => {
        cleanup();
        this.ready = true;
        resolve(this);
      };
    });
  }

  _write(kind, id, body) {
    const payload = Buffer.from(String(body), 'utf8');
    const buf = Buffer.alloc(14 + payload.length);
    buf.writeInt32LE(10 + payload.length, 0);
    buf.writeInt32LE(id, 4);
    buf.writeInt32LE(kind, 8);
    payload.copy(buf, 12);
    buf.writeInt16LE(0, 12 + payload.length);
    this.socket.write(buf);
  }

  _drain() {
    while (this.buffer.length >= 4) {
      const length = this.buffer.readInt32LE(0);
      if (this.buffer.length < length + 4) {
        return;
      }
      const frame = this.buffer.subarray(4, length + 4);
      this.buffer = this.buffer.subarray(length + 4);
      if (frame.length < 10) {
        continue;
      }
      const id = frame.readInt32LE(0);
      const kind = frame.readInt32LE(4);
      const body = frame.subarray(8, frame.length - 2).toString('utf8');
      if (kind === SERVERDATA_AUTH_RESPONSE) {
        if (id === -1) {
          this.ready = false;
          this.socket.destroy();
          return;
        }
        if (typeof this.onAuthenticated === 'function') {
          const callback = this.onAuthenticated;
          this.onAuthenticated = null;
          callback();
        }
        continue;
      }
      const entry = this.pending.shift();
      if (entry) {
        entry.resolve(body);
      }
    }
  }

  command(text) {
    if (!this.ready) {
      return Promise.reject(new Error('RCON 尚未连接'));
    }
    this.requestId += 1;
    const id = this.requestId;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        const index = this.pending.findIndex((item) => item.id === id);
        if (index >= 0) {
          this.pending.splice(index, 1);
        }
        resolve('');
      }, this.timeout);
      if (timer.unref) {
        timer.unref();
      }
      this.pending.push({
        id,
        resolve: (body) => {
          clearTimeout(timer);
          resolve(body ?? '');
        },
      });
      this._write(SERVERDATA_EXECCOMMAND, id, text);
    });
  }

  close() {
    if (this.socket) {
      try {
        this.socket.end();
      } catch {
        // ignore
      }
      this.socket = null;
    }
    this.ready = false;
  }
}

/** `--rcon` 取值解析：`25575` / `127.0.0.1:25575` / `127.0.0.1:25575:pass`。 */
export function parseRconSpec(spec, fallbackPassword = 'devpass') {
  const raw = String(spec ?? '').trim();
  if (!raw) {
    return null;
  }
  const parts = raw.split(':');
  if (parts.length === 1) {
    const port = Number(parts[0]);
    return Number.isInteger(port) && port > 0
      ? { host: '127.0.0.1', port, password: fallbackPassword }
      : null;
  }
  const host = parts[0] || '127.0.0.1';
  const port = Number(parts[1]);
  const password = parts.length > 2 ? parts.slice(2).join(':') : fallbackPassword;
  return Number.isInteger(port) && port > 0 ? { host, port, password } : null;
}
