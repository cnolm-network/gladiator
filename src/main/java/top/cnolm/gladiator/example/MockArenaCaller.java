package top.cnolm.gladiator.example;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.java.JavaPlugin;
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
        // G-727：Ready ≠ 客户端已就绪，单发一次 tp 会被位置预测吃掉——改为立即发 + 之后补发。
        teleportWithReassert(schedulerOwner(), player, spawn);

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

    // --- 位置补发（G-727）-----------------------------------------------------
    //: 补发节拍（tick）。Bot 刚 Ready 时客户端还在位置预测窗口内，服务端下发的 tp
    //: 会被客户端按「自己的位置」纠正掉：服务端坐标已经对了，客户端却停在原地，
    //: 双方互相看不见（观测里 entities 只剩自己）。arena 的 FlatBoard.reassertPositions
    //: 用同一套策略，这里保持一致。
    private static final int PLACE_REASSERT_TICKS = 10;
    //: 补发次数（不含立即那一次）。10 tick × 3 = 1.5 秒，覆盖客户端完成区块同步的窗口。
    private static final int PLACE_REASSERT_TIMES = 3;

    /**
     * 摆位：立即传送一次，随后每 {@link #PLACE_REASSERT_TICKS} tick 再补发，
     * 最多 {@link #PLACE_REASSERT_TIMES} 次（总共 1 + 3 = 4 次）。
     *
     * <p>为什么不能只发一次：Ready 事件只代表**服务端**完成了登录与实体生成，1.8 客户端
     * 此时还在用自己的位置预测覆盖服务端下发的 {@code S08PacketPlayerPosLook}。单发一次
     * 的结果是「服务端坐标正确、客户端停在旧位置」，mineflayer 侧表现为实体表只剩自己。
     *
     * <p>补发前判 {@code isOnline()}：玩家中途退出/被踢时立刻停，避免对着离线实体刷 tp。
     */
    /**
     * 调度用的插件实例：本类被打进插件 jar，取「提供本类的插件」即可；
     * 不在插件 classloader 里（单元测试/独立调用）时返回 {@code null}，调用方退回单发。
     */
    private static Plugin schedulerOwner() {
        try {
            return JavaPlugin.getProvidingPlugin(MockArenaCaller.class);
        } catch (RuntimeException error) {
            return null;
        }
    }

    private static void teleportWithReassert(final Plugin plugin, final Player player,
                                             final Location target) {
        if (player == null || target == null) {
            return;
        }
        player.teleport(target);
        if (plugin == null) {
            // 没有插件实例（纯单元测试/独立调用）时退回单发：至少不抛异常。
            return;
        }
        for (int index = 1; index <= PLACE_REASSERT_TIMES; index += 1) {
            final int attempt = index;
            plugin.getServer().getScheduler().runTaskLater(plugin, new Runnable() {
                @Override
                public void run() {
                    if (!player.isOnline()) {
                        return;
                    }
                    player.teleport(target);
                    if (attempt == PLACE_REASSERT_TIMES) {
                        plugin.getLogger().fine("[place] 摆位已补发 "
                                + PLACE_REASSERT_TIMES + " 次: " + player.getName());
                    }
                }
            }, PLACE_REASSERT_TICKS * (long) attempt);
        }
    }
    public World getWorld() {
        return world;
    }
}
