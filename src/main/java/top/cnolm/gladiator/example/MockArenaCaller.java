package top.cnolm.gladiator.example;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.GladiatorProvider;
import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.Users;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;

/**
 * 参考调用方（G-306）：只通过 {@link GladiatorProvider#get()} 与事件接入，
 * 不 import 任何竞技场插件类型，也不碰 Gladiator 内部类。
 *
 * <p>典型顺序（api-contract §4）：spawnBot → 等 Ready 事件 → 传送/发套件 →
 * pushGame。Gladiator 自身不做传送与套件。
 */
public final class MockArenaCaller implements Listener {

    private final World world;
    private final Location spawn;

    public MockArenaCaller(World world, Location spawn) {
        this.world = world;
        this.spawn = spawn;
    }

    /** 入口：发起一个 Bot 的登录；返回的票据用于日志/超时对账。 */
    public void spawnOne(String username) {
        GladiatorAPI api = GladiatorProvider.get();
        if (api == null || !api.isReady()) {
            return;
        }
        api.spawnBot(top.cnolm.gladiator.api.BotSpec.of(username, "default"));
    }

    /**
     * Bot 登录 + 实体生成完成：这里才是调用方做传送/发套件的时机。
     * 传送完成后立刻 pushGame，Gladiator 只记录会话与状态。
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onBotReady(GladiatorBotReadyEvent event) {
        Bot bot = event.getBot();
        Player player = event.getPlayer();
        if (player == null) {
            return;
        }
        GladiatorAPI api = GladiatorProvider.get();
        if (api == null) {
            return;
        }
        // 1) 调用方负责传送（示例直接传到固定点；生产代码里是选图/出生点逻辑）
        player.teleport(spawn);

        // 2) 调用方发套件（示例省略，真实实现由竞技场插件清背包/发装备）

        // 3) 会话登记：Mode 由调用方提供，Gladiator 只读
        Mode mode = new DeathmatchMode();
        GameSession session = api.pushGame(mode, bot, Users.of(player));
        if (api.getApiVersion() != null) {
            player.sendMessage("对局已开始: " + session.getSessionId());
        }
    }

    /** 无限制训练场：不传送、不发套件，仅登记 Bot 会话。 */
    public void placeForTraining(Bot bot) {
        GladiatorAPI api = GladiatorProvider.get();
        if (api != null && api.isReady() && bot.isReady()) {
            api.pushPlace(bot);
        }
    }

    public World getWorld() {
        return world;
    }
}
