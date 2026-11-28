// 临时工具：从 WorkBuddy 最后一条回复里**按重点取**，而不是固定截尾巴。
//
// 用法：
//   node tmp-wb-tail.mjs                     # 优先取「## 简报」段（我要求 WB 每次结尾必写），取不到退化为尾 800 字
//   node tmp-wb-tail.mjs --section 结论      # 取指定标题下的内容
//   node tmp-wb-tail.mjs --lines 8           # 该段最多取几行（默认 14）
//   node tmp-wb-tail.mjs --chars 600         # 无该段时尾部保留多少字
//   node tmp-wb-tail.mjs --full              # 全文（调试）
//
// 约定：给 WorkBuddy 的任务末尾要求它写一个简短结论段，形如
//   ## 简报
//   - 结论：……
//   - 改了：文件 1、文件 2
//   - 验证：命令 → 结果
//   - 风险/待办：……
import WebSocket from 'ws';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  if (i < 0) return def;
  const v = args[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
};
const full = args.includes('--full');
const lines = Math.max(3, Number(opt('--lines', 14)) || 14);
const chars = Math.max(120, Number(opt('--chars', 800)) || 800);
const section = String(opt('--section', '简报'));
const CDP = 'http://127.0.0.1:9223';

const expr = `(() => {
  const nodes = Array.from(document.querySelectorAll('.markdown-body, [class*="markdown"], [class*="message-content"]'));
  const texts = nodes.map(n => (n.innerText || '')).filter(t => t.trim().length > 0);
  return texts.length ? texts[texts.length - 1] : '';
})()`;

async function pageWs() {
  const res = await fetch(`${CDP}/json`);
  const pages = await res.json();
  const page = pages.find((p) => p.type === 'page' && !p.url.includes('windowKind=settings'));
  if (!page) throw new Error('找不到 WorkBuddy 页面（CDP 9223）');
  return page.webSocketDebuggerUrl;
}

let seq = 0;
function evaluate(ws, expression) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => reject(new Error('CDP evaluate 超时')), 15000);
    const onMessage = (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.id !== id) return;
      clearTimeout(timer);
      ws.off('message', onMessage);
      const value = msg.result && msg.result.result && msg.result.result.value;
      resolve(typeof value === 'string' ? value : '');
    };
    ws.on('message', onMessage);
    ws.send(JSON.stringify({ id, method: 'Runtime.evaluate', params: { expression, returnByValue: true } }));
  });
}

/** 取标题 `## <section>` 之后到下一个同级标题前的行；找不到返回 null。 */
function sliceSection(text, name) {
  const all = text.split('\n');
  const idx = all.findIndex((line) => /^#{1,6}\s/.test(line.trim()) && line.includes(name));
  if (idx < 0) return null;
  const level = (all[idx].trim().match(/^#+/) || ['#'])[0].length;
  const out = [];
  for (let i = idx + 1; i < all.length; i += 1) {
    const line = all[i];
    const m = line.trim().match(/^(#{1,6})\s/);
    if (m && m[1].length <= level) break;
    out.push(line);
    if (out.length >= lines) break;
  }
  const body = out.join('\n').trim();
  return body || null;
}

/** 没有简报段时：先挑含关键词的行（结论/失败/报错/待办/风险/测试），不足再补尾部。 */
function pickKeyLines(text, want) {
  const keywords = ['结论', '失败', '报错', '错误', '风险', '待办', '未做', '测试', '验证', '不支持', 'TODO'];
  const picked = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && keywords.some((k) => l.includes(k)))
    .slice(0, want);
  return picked;
}

const ws = new WebSocket(await pageWs());
await new Promise((resolve, reject) => {
  ws.once('open', resolve);
  ws.once('error', reject);
});
const raw = await evaluate(ws, expr);
ws.close();

const text = raw.replace(/\r/g, '').trimEnd();
if (!text) {
  console.log('[wb] 没有抓到回复文本（页面可能不在对话视图）');
  process.exit(1);
}
const len = text.length;

if (full) {
  console.log(`[wb] len=${len}（全文）`);
  console.log(text);
  process.exit(0);
}

const sec = sliceSection(text, section);
if (sec) {
  console.log(`[wb] len=${len} section="${section}" lines=${lines}`);
  console.log(sec);
  process.exit(0);
}

const keyed = pickKeyLines(text, lines);
console.log(`[wb] len=${len} 没有「${section}」段 → 关键词摘录 ${keyed.length} 行 + 尾 ${chars} 字`);
if (keyed.length) {
  keyed.forEach((l) => console.log(l.length > 160 ? `${l.slice(0, 160)}…` : l));
  console.log('---- 结尾 ----');
}
console.log(text.slice(-chars));
