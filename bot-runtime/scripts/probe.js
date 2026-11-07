#!/usr/bin/env node
/**
 * 环境探针：验证 mineflayer 在 1.8.8/1.8.9（协议 47）目标上的可用性。
 * 默认 1.8.8：目标服务端为 Spigot 1.8.8，且 1.8.8 在 mineflayer testedVersions 中；
 * 1.8.9 同协议 47 也可用，用 --version 1.8.9 切换。
 *
 * 用法：
 *   node scripts/probe.js                 # 只做离线能力检查
 *   node scripts/probe.js --host 127.0.0.1 --port 25565 --user ProbeBot
 *
 * 输出为「事实采集」：不修改任何本地文件，不影响服务端。
 */
import mineflayer from 'mineflayer';
import mcData from 'minecraft-data';

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const MC_VERSION = getArg('version', '1.8.8');

console.log('=== mineflayer 版本支持 ===');
console.log('mineflayer      :', (await import('mineflayer/package.json', { with: { type: 'json' } })).default.version);
console.log('oldestSupported :', mineflayer.oldestSupportedVersion);
console.log('latestSupported :', mineflayer.latestSupportedVersion);
console.log(`testedVersions 包含 ${MC_VERSION}:`, mineflayer.testedVersions.includes(MC_VERSION));

console.log('\n=== minecraft-data 协议数据 ===');
const data = mcData(MC_VERSION);
console.log(`${MC_VERSION} -> protocol`, data.version.version, '| major', data.version.majorVersion);
console.log('blocks:', Object.keys(data.blocksByName).length,
  '| items:', Object.keys(data.itemsByName).length,
  '| entities:', Object.keys(data.entitiesByName).length);

console.log('\n=== 动作插件模块（决定能力是否存在）===');
const { readdir } = await import('node:fs/promises');
const { dirname, join } = await import('node:path');
const { fileURLToPath } = await import('node:url');
const mfRoot = dirname(fileURLToPath(import.meta.resolve('mineflayer')));
const plugins = (await readdir(join(mfRoot, 'lib/plugins'))).filter((f) => f.endsWith('.js')).map((f) => f.replace('.js', ''));
const required = ['physics', 'digging', 'place_block', 'chat', 'entities', 'health', 'inventory', 'game'];
for (const name of required) {
  console.log(`${name.padEnd(14)}: ${plugins.includes(name) ? 'present' : 'MISSING'}`);
}
console.log('全部插件:', plugins.join(', '));

const host = getArg('host', null);
if (!host) {
  console.log('\n[离线模式] 未提供 --host，跳过真实连接。');
  process.exit(0);
}

const port = Number(getArg('port', 25565));
const username = getArg('user', 'ProbeBot');
console.log(`\n=== 连接 ${host}:${port} as ${username} (version=${MC_VERSION}) ===`);

const bot = mineflayer.createBot({
  host, port, username, version: MC_VERSION, auth: 'offline'
});

bot.once('login', () => console.log('login    : OK'));
bot.once('spawn', () => {
  console.log('spawn    : OK');
  console.log('position :', bot.entity.position);
  console.log('health   :', bot.health, '| food:', bot.food);
  console.log('game     :', bot.game.version, '| dimension:', bot.game.dimension);
  console.log('players  :', Object.keys(bot.players).join(', '));
});
bot.on('error', (e) => console.log('error    :', e.message));
bot.on('kicked', (r) => console.log('kicked   :', JSON.stringify(r)));
bot.once('end', (r) => {
  console.log('end      :', r);
  process.exit(0);
});

setTimeout(() => {
  console.log('\n[超时] 30s 内未完成 spawn，主动断开。');
  bot.end();
}, 30000);
