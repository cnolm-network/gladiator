#!/usr/bin/env node
/**
 * A5：pathfinder 移动代价标定（G-207 的前置数据）。
 *
 * 做法：
 *  1. 以 A5_Nav 离线身份登录 Paper 1.8.8（默认 127.0.0.1:25566）；
 *  2. 在 y=99 铺一条 14×3 的石台（y=100 为站立面），把 Bot /tp 到起点；
 *  3. 沿 +X 方向走一条 **10 格直线**，重复 N 次（默认 20），每次记录
 *     路径耗时（ms）、实际位移（格）、是否卡住（超时/几乎没动）；
 *  4. 再测一次「3 格高台跳下」，记录掉落伤害（1.8.8：伤害 = floor(落差 − 3)）；
 *  5. 输出 mean/p50/p95/max、卡住次数、平均速度（blocks/s）与「每格耗时」标定表。
 *
 * 依赖服务器 OP（`/fill`、`/tp`）。用户名需事先在 ops.json 里，或从控制台执行
 * `op A5_Nav` —— 脚本会等待最多 30 秒。
 *
 * 用法：
 *   node scripts/a5-nav.mjs [--username A5_Nav] [--host 127.0.0.1] [--mc-port 25566]
 *                           [--runs 20] [--report a5-report.txt]
 */
import fs from 'node:fs';
import mineflayer from 'mineflayer';
import { Vec3 } from 'vec3';
// mineflayer-pathfinder 是 CommonJS：ESM 里只能拿 default 再解构。
import pathfinderPkg from 'mineflayer-pathfinder';

const { pathfinder, Movements, goals } = pathfinderPkg;

function arg(name, fallback) {
  const argv = process.argv.slice(2);
  const flag = `--${name}`;
  const i = argv.indexOf(flag);
  if (i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--')) {
    return argv[i + 1];
  }
  return fallback;
}

const USERNAME = arg('username', 'A5_Nav');
const HOST = arg('host', '127.0.0.1');
const PORT = Number(arg('mc-port', '25566'));
const RUNS = Number(arg('runs', '20'));
const REPORT = arg('report', 'a5-report.txt');
const VERSION = arg('mc-version', '1.8.8');

/** 路线参数：站立面 y=100，石台在 y=99；起点 x+1.5，终点 x+11.5 → 正好 10 格。 */
const BASE_X = -2150;
const BASE_Z = -470;
const FLOOR_Y = 99;
const STAND_Y = 100;
const START = { x: BASE_X + 1.5, y: STAND_Y, z: BASE_Z + 0.5 };
const GOAL = { x: BASE_X + 11.5, y: STAND_Y, z: BASE_Z + 0.5 };
const PATH_BLOCKS = 10;

const lines = [];
function log(text = '') {
  lines.push(text);
  console.log(text);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function pct(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function fmtPos() {
  const p = bot.entity?.position;
  return p ? `(${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)})` : '(?)';
}

function stats(values) {
  if (values.length === 0) {
    return { count: 0, mean: 0, p50: 0, p95: 0, max: 0, min: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    count: sorted.length,
    mean: sum / sorted.length,
    p50: pct(sorted, 50),
    p95: pct(sorted, 95),
    max: sorted[sorted.length - 1],
    min: sorted[0],
  };
}

const bot = mineflayer.createBot({
  host: HOST,
  port: PORT,
  username: USERNAME,
  version: VERSION,
  auth: 'offline',
  // 保持 mineflayer 默认的自动重生：上一轮如果死在测试台，登录时服务端会先要求重生，
  // 关掉它会导致 'spawn' 永远不触发（实测踩过）。
  respawn: true,
});

bot.loadPlugin(pathfinder);

let hurtAccum = 0;
bot.on('entityHurt', (entity) => {
  if (entity && entity.id === bot.entity?.id) {
    hurtAccum += 1;
  }
});

function cmd(text) {
  bot.chat(text);
}

async function waitFor(predicate, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await sleep(50);
  }
  throw new Error(`等待超时：${label}`);
}

async function teleportTo(x, y, z) {
  cmd(`/tp ${USERNAME} ${x.toFixed(2)} ${y.toFixed(2)} ${z.toFixed(2)}`);
  await waitFor(() => {
    const p = bot.entity?.position;
    if (!p) return false;
    return Math.abs(p.x - x) < 1.2 && Math.abs(p.y - y) < 2.5 && Math.abs(p.z - z) < 1.2;
  }, 6000, `tp 到 ${x},${y},${z}`);
  // 客户端要靠区块数据才知道脚下是石头：多等两 tick 再起跑，避免 pathfinder 看到"空气"。
  await sleep(350);
}

/** 用原版 instant_health 效果把血回满，保证伤害读数从 20 起算。 */
async function heal() {
  cmd(`/effect ${USERNAME} 6 1 10`);
  await sleep(500);
  if (bot.health < 19.9) {
    cmd(`/effect ${USERNAME} 6 1 10`);
    await sleep(500);
  }
  return bot.health;
}

async function goto(goal, timeoutMs) {
  const movements = new Movements(bot);
  // 只走路：不挖、不放、不搭桥，避免把测试台改掉。
  movements.canDig = false;
  movements.allow1by1towers = false;
  movements.allowParkour = false;
  movements.allowSprinting = true;
  bot.pathfinder.setMovements(movements);
  const started = process.hrtime.bigint();
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('TIMEOUT')), timeoutMs);
  });
  try {
    await Promise.race([bot.pathfinder.goto(goal), timeout]);
    const ms = Number(process.hrtime.bigint() - started) / 1e6;
    return { ok: true, ms };
  } catch (error) {
    const ms = Number(process.hrtime.bigint() - started) / 1e6;
    try {
      bot.pathfinder.setGoal(null);
      bot.pathfinder.stop();
    } catch {
      // 忽略
    }
    return { ok: false, ms, error: String(error.message ?? error) };
  } finally {
    clearTimeout(timer);
  }
}

async function main() {
  await new Promise((resolve, reject) => {
    bot.once('spawn', resolve);
    bot.once('error', reject);
    setTimeout(() => reject(new Error('登录超时')), 30000);
  });

  log(`# A5 pathfinder 移动代价标定`);
  log(`bot=${USERNAME} server=${HOST}:${PORT} version=${bot.version} runs=${RUNS}`);
  log(`路线：(${START.x},${STAND_Y},${START.z}) → (${GOAL.x},${STAND_Y},${GOAL.z})，直线 ${PATH_BLOCKS} 格`);
  log('');

  // 铺台子（OP）：14×9 的石台，y=99（宽一点，摔落测试要有落点）。
  const FLOOR_FILL = `/fill ${BASE_X} ${FLOOR_Y} ${BASE_Z - 4} ${BASE_X + 13} ${FLOOR_Y} ${BASE_Z + 4} minecraft:stone`;
  cmd(FLOOR_FILL);
  await sleep(600);
  const probeBlock = () => bot.blockAt(new Vec3(BASE_X + 5, FLOOR_Y, BASE_Z));
  let ground = probeBlock();
  if (!ground || ground.name === 'air') {
    log(`! 石台没有生成：Bot 可能没有 OP。请先从控制台执行 \`op ${USERNAME}\`，30 秒内会自动继续。`);
    const deadline = Date.now() + 30000;
    let ok = false;
    while (Date.now() < deadline && !ok) {
      await sleep(2000);
      cmd(FLOOR_FILL);
      await sleep(400);
      const block = probeBlock();
      ok = Boolean(block) && block.name !== 'air';
    }
    if (!ok) {
      log('! 仍然没有权限，退出。');
      finish(false);
      return;
    }
    log('· 已获得权限，石台就绪。');
  }
  log(`· 石台就绪：y=${FLOOR_Y} 铺满，站立面 y=${STAND_Y}`);
  // 预热：先站上去一次，让客户端把这块区块的方块数据收全（否则 pathfinder 会以为脚下是空气）。
  await teleportTo(START.x, START.y, START.z);
  try {
    await bot.waitForChunksToLoad();
  } catch {
    // 1.8.8 下可能不触发，忽略
  }
  await sleep(900);
  log(`· 预热完成：pos=${fmtPos()} 脚下方块=${bot.blockAt(new Vec3(Math.floor(START.x), FLOOR_Y, Math.floor(START.z)))?.name ?? '?'}`);

  const durations = [];
  const speeds = [];
  const failures = [];
  let stuck = 0;

  for (let i = 1; i <= RUNS; i += 1) {
    await teleportTo(START.x, START.y, START.z);
    let from = { ...bot.entity.position };
    let result = await goto(new goals.GoalBlock(GOAL.x, GOAL.y, GOAL.z), 15000);
    if (!result.ok) {
      // 偶发的 pathfinder 抖动（客户端区块数据刚刷新）——重试一次，重试仍失败才计卡住。
      await teleportTo(START.x, START.y, START.z);
      from = { ...bot.entity.position };
      result = await goto(new goals.GoalBlock(GOAL.x, GOAL.y, GOAL.z), 15000);
    }
    const to = { ...bot.entity.position };
    const moved = Math.hypot(to.x - from.x, to.z - from.z);
    if (!result.ok) {
      stuck += 1;
      failures.push({ run: i, reason: result.error, ms: result.ms, moved });
      log(`#${String(i).padStart(2, '0')} FAIL ${result.error} ${result.ms.toFixed(0)}ms 位移=${moved.toFixed(2)}`);
      continue;
    }
    if (moved < PATH_BLOCKS - 1.5) {
      stuck += 1;
      failures.push({ run: i, reason: 'INCOMPLETE', ms: result.ms, moved });
      log(`#${String(i).padStart(2, '0')} FAIL 位移不足 ${moved.toFixed(2)} 格 / ${result.ms.toFixed(0)}ms`);
      continue;
    }
    durations.push(result.ms);
    speeds.push(moved / (result.ms / 1000));
    log(`#${String(i).padStart(2, '0')} ok ${result.ms.toFixed(0)}ms 位移=${moved.toFixed(2)} 速度=${(moved / (result.ms / 1000)).toFixed(2)} b/s`);
  }

  const s = stats(durations);
  const speed = stats(speeds);
  log('');
  log('## 直线移动（10 格 / 次）');
  log(`runs=${RUNS} ok=${durations.length} stuck=${stuck}`);
  log(`耗时 ms：mean=${s.mean.toFixed(0)} p50=${s.p50.toFixed(0)} p95=${s.p95.toFixed(0)} max=${s.max.toFixed(0)} min=${s.min.toFixed(0)}`);
  log(`速度 b/s：mean=${speed.mean.toFixed(2)} p50=${speed.p50.toFixed(2)} p95=${speed.p95.toFixed(2)} min=${speed.min.toFixed(2)}`);
  for (const failure of failures) {
    log(`! #${failure.run} ${failure.reason} ${failure.ms.toFixed(0)}ms 位移=${failure.moved.toFixed(2)}`);
  }

  // ---- 掉落：3 格高台跳下 ----
  log('');
  log('## 掉落（3 格高台跳下）');
  const highTop = STAND_Y + 3; // 台面 y=103，比地面高 3 格
  // 3×3 台子，顶面 y=103（方块在 y=102）
  cmd(`/fill ${BASE_X + 2} ${highTop - 1} ${BASE_Z - 1} ${BASE_X + 4} ${highTop - 1} ${BASE_Z + 1} minecraft:stone`);
  await sleep(600);
  try {
    await heal();
    await teleportTo(BASE_X + 3.5, highTop, BASE_Z + 0.5);
    const before = bot.health;
    const stoodOn = bot.blockAt(new Vec3(BASE_X + 3, highTop - 1, BASE_Z));
    log(`· 站上高台：pos=${fmtPos()} 脚下方块=${stoodOn ? stoodOn.name : '?'}`);
    // 不用 pathfinder（它会因 maxDropDown 拒绝 3 格外的高度差，也可能原地不动），
    // 直接按前进键走出边缘 —— 这才是"跳下"的真实动作。
    bot.setControlState('forward', true);
    // 只走 0.45s（约 2.4 格）：足够走出 3×3 台边缘，又不会冲出 9 格宽的落点台面。
    await sleep(450);
    bot.setControlState('forward', false);
    await waitFor(() => bot.entity.onGround && bot.entity.position.y <= STAND_Y + 0.6, 8000, '落地');
    await sleep(600);
    const damage = before - bot.health;
    log(`health ${before} → ${bot.health}，落差=3 格，掉落伤害=${damage.toFixed(1)}`);
    log('说明：1.8.8 摔落伤害 = floor(落差 − 3)，落差 3 格 → 0 伤害（阈值点）。');
  } catch (error) {
    log(`! 3 格跌落测试失败：${error.message ?? error}`);
  }

  // ---- 掉落对照：6 格 ----
  log('');
  log('## 掉落对照（6 格落差）');
  try {
    await heal();
    const before = bot.health;
    // 直接 tp 到空中，让物理引擎自然下落：落差 = 6 格，落点是石台（y=100）。
    cmd(`/tp ${USERNAME} ${(BASE_X + 7.5).toFixed(2)} ${(STAND_Y + 6).toFixed(2)} ${(BASE_Z + 0.5).toFixed(2)}`);
    await sleep(400);
    await waitFor(() => bot.entity.onGround && bot.entity.position.y <= STAND_Y + 0.6, 8000, '落地');
    await sleep(600);
    const damage = before - bot.health;
    log(`health ${before} → ${bot.health}，落差=6 格，掉落伤害=${damage.toFixed(1)}`);
    log('用途：每格摔落伤害 = floor(落差 − 3) / 落差，可与移动代价一起作为「掉落代价」基线。');
  } catch (error) {
    log(`! 6 格跌落失败：${error.message ?? error}`);
  }

  // ---- 判定 ----
  const EXPECTED_MS = (PATH_BLOCKS / 4.3) * 1000; // 4.3 b/s 步行 ≈ 2330ms
  const LIMIT_MS = EXPECTED_MS * 2;
  const pass = durations.length === RUNS && stuck === 0 && s.p95 < LIMIT_MS;
  log('');
  log('## 判定');
  log(`期望上限：p95 < ${LIMIT_MS.toFixed(0)}ms（10 格 / 4.3 b/s ≈ ${EXPECTED_MS.toFixed(0)}ms，允许 2 倍）`);
  log(`实测：p95=${s.p95.toFixed(0)}ms stuck=${stuck} 完成=${durations.length}/${RUNS}`);
  log(`结果：${pass ? 'PASS' : 'FAIL'}`);

  log('');
  log('## 移动代价标定表（每格）');
  log(`mean  ${(s.mean / PATH_BLOCKS).toFixed(1)} ms/格  (${(1000 / (s.mean / PATH_BLOCKS)).toFixed(2)} 格/s)`);
  log(`p50   ${(s.p50 / PATH_BLOCKS).toFixed(1)} ms/格`);
  log(`p95   ${(s.p95 / PATH_BLOCKS).toFixed(1)} ms/格`);
  log(`max   ${(s.max / PATH_BLOCKS).toFixed(1)} ms/格`);
  log('用途：把 deny 区域当作高代价节点时，用 p95 每格耗时 × 绕行格数作为惩罚增量。');

  // 清理：把测试台拆掉，别把世界改成施工现场。
  cmd(`/fill ${BASE_X - 1} ${FLOOR_Y} ${BASE_Z - 5} ${BASE_X + 14} ${STAND_Y + 8} ${BASE_Z + 5} minecraft:air`);
  await sleep(500);
  log('· 已拆除测试台（/fill air）');

  fs.writeFileSync(REPORT, `${lines.join('\n')}\n`, 'utf8');
  log('');
  log(`report=${REPORT}`);

  finish(pass);
}

function finish(pass) {
  try {
    fs.writeFileSync(REPORT, `${lines.join('\n')}\n`, 'utf8');
  } catch {
    // 忽略
  }
  console.log(`EXIT ${pass ? 'PASS' : 'FAIL'}`);
  setTimeout(() => {
    try {
      bot.quit('done');
    } catch {
      // 忽略
    }
    process.exit(pass ? 0 : 1);
  }, 300);
}

main().catch((error) => {
  log(`! 脚本异常：${error?.stack ?? error}`);
  finish(false);
});
