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
        player.teleport(ready.isEmpty() ? spawnA : spawnB);
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

    public void register() {
        Bukkit.getPluginManager().registerEvents(this, plugin);
    }
}
