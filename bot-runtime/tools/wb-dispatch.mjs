#!/usr/bin/env node
/**
 * wb-dispatch —— 一条命令完成「把任务发给 WorkBuddy + 等它交付 + 打印简报」。
 *
 * 为什么要有它：以前我（dsh）用一个独立的 `Start-Process` 轮询当 watcher，界面里
 * 看不到、还会被清场命令顺手杀掉，结果就是「忘了挂 watcher」。现在把发送与等待
 * 收进同一个进程，**由 DSH 的 run_in_background 托管**，任务一结束就通知我。
 *
 * 用法：
 *   node bot-runtime/tools/wb-dispatch.mjs --file task.md [--timeout 90] [--poll 5]
 *   node bot-runtime/tools/wb-dispatch.mjs --text "任务正文..."
 *
 * 输出（stdout 最后一段，机器可读）：
 *   [wb-dispatch] sent ... / waiting ...
 *   ----BRIEFING-BEGIN----
 *   ## 简报
 *   ...
 *   ----BRIEFING-END----
 * 退出码：0 = 拿到简报；2 = 超时（打印最后一次助手消息尾部）；1 = CDP/发送失败。
 */

import fs from 'node:fs';
import WebSocket from 'ws';

const CDP_HTTP = process.env.WB_CDP_HTTP || 'http://127.0.0.1:9223';
const args = process.argv.slice(2);

function arg(name, def) {
  const i = args.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = args[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

const timeoutMin = Number(arg('timeout', 90));
const pollSec = Number(arg('poll', 5));
const minChars = Number(arg('min-chars', 60));
const requireBriefing = arg('require-briefing', true) !== 'false';
const settlePolls = Number(arg('settle-polls', 3));

function readTask() {
  const file = arg('file');
  if (typeof file === 'string') return fs.readFileSync(file, 'utf8');
  const text = arg('text');
  if (typeof text === 'string') return text;
  throw new Error('必须给 --file <path> 或 --text <内容>');
}

async function pageTarget() {
  const res = await fetch(`${CDP_HTTP}/json`);
  const list = await res.json();
  const page = list.find((t) => t.type === 'page' && /workbuddy/i.test(t.title || ''));
  if (!page) throw new Error('找不到 WorkBuddy 页面（CDP 里没有 title 匹配的 page）');
  return page;
}

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url, { maxPayload: 64 * 1024 * 1024 });
    this.seq = 0;
    this.pending = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.ws.once('open', resolve);
      this.ws.once('error', reject);
    });
    this.ws.on('message', (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      else p.resolve(msg.result);
    });
  }

  send(method, params = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const r = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || 'evaluate 失败');
    return r.result && r.result.value;
  }

  close() {
    try {
      this.ws.close();
    } catch {
      /* ignore */
    }
  }
}

const FOCUS_COMPOSER = `(() => {
  const el = document.querySelector('[contenteditable="true"]');
  if (!el) return { ok: false, reason: 'no-composer' };
  el.focus();
  const r = el.getBoundingClientRect();
  return { ok: true, x: r.left + r.width / 2, y: r.top + r.height / 2 };
})()`;

const SEND_BUTTON = `(() => {
  const btns = [...document.querySelectorAll('button')];
  const hit = btns.find((b) => /发送|send/i.test(b.textContent || '') || b.getAttribute('aria-label') === '发送');
  if (!hit) return { ok: false, reason: 'no-send' };
  const r = hit.getBoundingClientRect();
  return { ok: true, x: r.left + r.width / 2, y: r.top + r.height / 2, disabled: hit.disabled === true };
})()`;

const LAST_ASSISTANT = `(() => {
  const nodes = [...document.querySelectorAll('.markdown-body')];
  if (!nodes.length) return '';
  return nodes[nodes.length - 1].innerText || '';
})()`;

// 兜底：窗口被最小化/隐藏时 Chromium 可能不回渲染，.markdown-body 的 innerText 为空；
// 这时用 body.innerText（虚拟化未渲染的部分仍会缺，但至少能命中「## 简报」）。
const BODY_TEXT = `(() => (document.body ? document.body.innerText || '' : ''))()`;

async function click(cdp, x, y) {
  const base = { x: Math.round(x), y: Math.round(y), button: 'left', clickCount: 1 };
  await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', ...base });
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', ...base });
}

function extractBriefing(text) {
  const idx = text.indexOf('## 简报');
  if (idx < 0) return null;
  return text.slice(idx).trim();
}

async function main() {
  const task = readTask().trim();
  if (!task) throw new Error('任务正文为空');
  const page = await pageTarget();
  const cdp = new Cdp(page.webSocketDebuggerUrl);
  await cdp.ready;
  await cdp.send('Runtime.enable');

  await cdp.send('Page.enable').catch(() => {});
  await cdp.send('Page.bringToFront').catch(() => {});
  const before = String((await cdp.evaluate(LAST_ASSISTANT)) || '');
  const focus = await cdp.evaluate(FOCUS_COMPOSER);
  if (!focus || !focus.ok) throw new Error(`composer 不可用：${focus && focus.reason}`);
  await click(cdp, focus.x, focus.y);
  await cdp.send('Input.insertText', { text: task });
  await new Promise((r) => setTimeout(r, 400));

  const btn = await cdp.evaluate(SEND_BUTTON);
  if (!btn || !btn.ok) throw new Error(`发送按钮不可用：${btn && btn.reason}`);
  if (btn.disabled) throw new Error('发送按钮是 disabled（composer 可能没拿到文本）');
  await click(cdp, btn.x, btn.y);
  console.log(`[wb-dispatch] sent chars=${task.length} at=${new Date().toISOString()}`);
  console.log(`[wb-dispatch] waiting up to ${timeoutMin} min (poll ${pollSec}s)`);

  const deadline = Date.now() + timeoutMin * 60000;
  let last = before;
  let stable = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollSec * 1000));
    let text = String((await cdp.evaluate(LAST_ASSISTANT)) || '');
    if (text.length < 40) {
      const body = String((await cdp.evaluate(BODY_TEXT)) || '');
      if (body.length > text.length) text = body;
    }
    if (text === last) {
      stable += 1;
    } else {
      last = text;
      stable = 0;
    }
    const brief = extractBriefing(text);
    const fresh = text !== before && text.length >= minChars;
    const done = requireBriefing
      ? Boolean(brief) && stable >= settlePolls
      : fresh && stable >= settlePolls;
    if (done) {
      console.log(`----BRIEFING-BEGIN----`);
      console.log(brief || text.slice(-2000));
      console.log(`----BRIEFING-END----`);
      cdp.close();
      return 0;
    }
  }

  console.log(`[wb-dispatch] timeout after ${timeoutMin} min`);
  if (!last) {
    console.log('[wb-dispatch] 拿不到回复文本：WorkBuddy 窗口可能被最小化/隐藏（Chromium 不回渲染）。');
    console.log('[wb-dispatch] 处理：把窗口还原后重跑，或用 wb_gui_read 直接读。');
  }
  console.log('----TAIL-BEGIN----');
  console.log(last.slice(-2000));
  console.log('----TAIL-END----');
  cdp.close();
  return 2;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(`[wb-dispatch] failed: ${err && err.message ? err.message : err}`);
    process.exit(1);
  });
