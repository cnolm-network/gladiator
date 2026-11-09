/**
 * 观战冒烟：以真实客户端身份进服，执行 /gladiator spectate，验证只读旁观。
 *
 *   # 先启动服务端 + 至少一个已就绪的 Bot，然后：
 *   node scripts/spectate-smoke.mjs --target Glad_Bot_1
 *
 * 脚本无法自己给自己 op：请在 --op-wait 秒内用控制台/RCON 执行 `op <username>`。
 * 退出码 0 表示「进入旁观 → 还原」两项断言都通过。
 */
import mineflayer from 'mineflayer';

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const username = arg('username', 'Watcher');
const host = arg('host', '127.0.0.1');
const port = Number(arg('port', '25565'));
const version = arg('mc-version', '1.8.8');
const target = arg('target', 'Glad_Bot_1');
const opWaitMs = Number(arg('op-wait', '12')) * 1000;

const log = (...parts) => console.log(`[spectate]`, ...parts);

const bot = mineflayer.createBot({ host, port, username, version, auth: 'offline' });

let failed = false;
const check = (label, ok, detail = '') => {
  log(ok ? `PASS ${label}` : `FAIL ${label}`, detail);
  if (!ok) failed = true;
};

bot.once('spawn', () => {
  log(`${username} 已进服，等待 ${opWaitMs / 1000}s 让管理员授权…`);
  setTimeout(() => {
    log(`发送 /gladiator spectate ${target}`);
    bot.chat(`/gladiator spectate ${target}`);
    setTimeout(() => {
      check('进入旁观者模式', bot.game.gameMode === 'spectator', `gameMode=${bot.game.gameMode}`);
      const pos = bot.entity?.position;
      if (pos) log(`位置 x=${pos.x.toFixed(1)} y=${pos.y.toFixed(1)} z=${pos.z.toFixed(1)}`);
      const targetEntity = Object.values(bot.entities).find(
        (entity) => entity.username?.toLowerCase() === target.toLowerCase(),
      );
      if (targetEntity) {
        const distance = targetEntity.position.distanceTo(bot.entity.position);
        check('已传送到目标附近', distance <= 8, `distance=${distance.toFixed(2)}`);
      } else {
        log(`未在实体列表里找到 ${target}（可能超出视距）`);
      }

      bot.chat('/gladiator unspectate');
      setTimeout(() => {
        check('退出后还原游戏模式', bot.game.gameMode !== 'spectator', `gameMode=${bot.game.gameMode}`);
        bot.quit(`spectate-smoke ${failed ? 'failed' : 'ok'}`);
        setTimeout(() => process.exit(failed ? 1 : 0), 300);
      }, 2500);
    }, 2500);
  }, opWaitMs);
});

bot.on('kicked', (reason) => {
  log('被踢出:', reason);
  process.exit(1);
});
bot.on('error', (error) => {
  log('错误:', error.message);
  process.exit(1);
});
bot.on('end', () => log('连接结束'));

setTimeout(() => {
  log('超时退出');
  process.exit(1);
}, 120000).unref();
