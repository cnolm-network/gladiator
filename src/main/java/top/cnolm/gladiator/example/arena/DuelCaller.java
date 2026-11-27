package top.cnolm.gladiator.example.arena;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.plugin.Plugin;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.Users;
import top.cnolm.gladiator.api.event.GladiatorBotFailedEvent;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 示例调用方 B（G-506）：一个"决斗"编排器，只通过 {@code api} 包与事件与 Gladiator 交互。
 *
 * <p>它和 {@code example.MockArenaCaller} 属于不同包、不同 {@link top.cnolm.gladiator.api.Mode} 实现，
 * 但两者跑在同一个 Gladiator 上时，Gladiator 源码不需要任何改动——这正是"换插件只改调用方"的验证点。</p>
 *
 * <p>本类只做示例：注册事件、spawn 两个 Bot、在两者都 Ready 后 pushGame，死亡时 eliminate。
 * 传送、套件、回血、计分都属于调用方职责。</p>
 */
public final class DuelCaller implements Listener {

    private final Plugin plugin;
    private final GladiatorAPI api;
    private final Map<String, Bot> ready = new LinkedHashMap<String, Bot>();
    private final Location spawnA;
    private final Location spawnB;
    private String sessionHint;

    public DuelCaller(Plugin plugin, GladiatorAPI api, Location spawnA, Location spawnB) {
        this.plugin = plugin;
        this.api = api;
        this.spawnA = spawnA;
        this.spawnB = spawnB;
    }

    /** 发起一场决斗；返回两个 BotRequest（PENDING），登录完成后走 Ready 事件。 */
    public BotRequest[] start(String userA, String userB) {
        ready.clear();
        sessionHint = null;
        BotRequest first = api.spawnBot(BotSpec.of(userA, "duel-a"));
        BotRequest second = api.spawnBot(BotSpec.of(userB, "duel-b"));
        plugin.getLogger().info("[duel] 已受理 " + first.getRequestId() + " / " + second.getRequestId());
        return new BotRequest[]{first, second};
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onBotReady(GladiatorBotReadyEvent event) {
        Bot bot = event.getBot();
        Player player = event.getPlayer();
        if (player == null) {
            return;
        }
        // 位置由调用方决定：这里把两个 Bot 摆到对角。
        // G-727：Ready ≠ 客户端已就绪，单发一次 tp 会被位置预测吃掉——改为立即发 + 之后补发。
        teleportWithReassert(plugin, player, ready.isEmpty() ? spawnA : spawnB);
        player.setHealth(player.getMaxHealth());
        ready.put(bot.getUniqueId().toString(), bot);
        plugin.getLogger().info("[duel] Bot 就绪 " + bot.getName() + "（" + ready.size() + "/2）");

        if (ready.size() == 2) {
            Bot[] pair = ready.values().toArray(new Bot[0]);
            sessionHint = api.pushGame(new DuelMode(), pair[0], pair[1]).getSessionId();
            plugin.getLogger().info("[duel] 对局开始 session=" + sessionHint);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onBotFailed(GladiatorBotFailedEvent event) {
        plugin.getLogger().warning("[duel] Bot 登录失败 " + event.getBotId() + ": " + event.getReason());
    }

    /** 由调用方在判定落败时调用；Gladiator 只做标记，不做规则判断。 */
    public void onDefeated(Player loser) {
        api.eliminate(Users.of(loser));
        plugin.getLogger().info("[duel] 淘汰 " + loser.getName() + " session=" + sessionHint);
    }

    /** 示例：把"谁赢了"写进调用方自己的统计（Bot 会被 StatsBoard 过滤掉）。 */
    public void recordWinner(Player winner, top.cnolm.gladiator.example.StatsBoard board) {
        if (!board.recordWin(Users.of(winner))) {
            plugin.getLogger().info("[duel] 跳过 Bot 战绩: " + winner.getName());
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

    public void register() {
        Bukkit.getPluginManager().registerEvents(this, plugin);
    }
}
