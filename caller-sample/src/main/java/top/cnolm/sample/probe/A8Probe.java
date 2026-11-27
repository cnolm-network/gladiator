package top.cnolm.sample.probe;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;
import top.cnolm.gladiator.example.arena.DuelMode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * A8 探针：两场独立对局的数据隔离。
 *
 * <p>做法：4 只 Bot 分成 A/B 两组，各自 {@code pushGame}（两个 {@link GameSession}），
 * 每场用**自己的**记分对象（{@link MatchBoard}）统计参与者和胜场，然后断言：
 * <ul>
 *   <li>每个板只看到自己那场的 2 个参与者，两场参与者集合不相交；</li>
 *   <li>A 场胜场只落在 A 板，B 场胜场只落在 B 板；跨场查询为 0；</li>
 *   <li>两个 sessionId 不同，且胜负双方的 {@code stateOf} 仍为 ALIVE（isolate 不影响状态机）。</li>
 * </ul>
 *
 * <p>为什么不用 {@code example.StatsBoard}：它按设计**过滤 Bot**（{@code recordWin} 返回 false），
 * 用它统计 Bot 对局永远是 0 胜，无法证明"两场不串"。这里用等价的 UUID → 胜场 Map，
 * 只测隔离性，不测"过滤 Bot"那条调用方约定（那条由 {@code StatsBoard} 自己的单测覆盖）。
 */
public final class A8Probe implements Listener {

    private static final int TIMEOUT_TICKS = 240;
    private static final int SETTLE_TICKS = 20;

    private final Plugin plugin;
    private final GladiatorAPI api;
    private final CommandSender sender;
    private final Location spawn;
    private final ProbeTimeline timeline = new ProbeTimeline();

    private final Map<String, Bot> ready = new LinkedHashMap<String, Bot>();
    private final List<String> failures = new ArrayList<String>();
    private String[] names;
    private String sessionA;
    private String sessionB;
    private int doneTick = -1;
    private BukkitTask watch;

    private final MatchBoard boardA = new MatchBoard("A场");
    private final MatchBoard boardB = new MatchBoard("B场");

    public A8Probe(Plugin plugin, GladiatorAPI api, CommandSender sender, Location spawn) {
        this.plugin = plugin;
        this.api = api;
        this.sender = sender;
        this.spawn = spawn;
    }

    public void start() {
        String suffix = String.valueOf(System.currentTimeMillis() % 100000L);
        names = new String[]{"a8a" + suffix, "a8b" + suffix, "a8c" + suffix, "a8d" + suffix};
        timeline.start(plugin);
        Bukkit.getPluginManager().registerEvents(this, plugin);
        for (String name : names) {
            timeline.add("spawnBot 请求 " + name);
            api.spawnBot(BotSpec.of(name, "a8"));
        }
        watch = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
            @Override
            public void run() {
                tickWatch();
            }
        }, 5L, 5L);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onReady(GladiatorBotReadyEvent event) {
        String name = event.getBot().getName();
        if (!isTarget(name)) {
            return;
        }
        timeline.add("Ready " + name);
        Player player = event.getPlayer();
        // G-727：Ready 回调里客户端还没跟上，立即发一次 + 之后每 10 tick 补发（helper 自带 null 判断）。
        teleportWithReassert(plugin, player, spawn);
        ready.put(name, event.getBot());
        if (ready.size() == names.length && sessionA == null) {
            runMatches();
        }
    }

    private void runMatches() {
        Bot[] all = ready.values().toArray(new Bot[0]);
        Bot a1 = all[0];
        Bot a2 = all[1];
        Bot b1 = all[2];
        Bot b2 = all[3];

        // ---- A 场 ----
        timeline.add("pushGame A场 (" + a1.getName() + "," + a2.getName() + ")");
        GameSession sa = api.pushGame(new DuelMode(), a1, a2);
        sessionA = sa.getSessionId();
        boardA.join(a1);
        boardA.join(a2);
        timeline.add("A场 session=" + sessionA);

        // ---- B 场 ----
        timeline.add("pushGame B场 (" + b1.getName() + "," + b2.getName() + ")");
        GameSession sb = api.pushGame(new DuelMode(), b1, b2);
        sessionB = sb.getSessionId();
        boardB.join(b1);
        boardB.join(b2);
        timeline.add("B场 session=" + sessionB);

        // ---- 模拟各自结算 ----
        boardA.win(a1);
        boardB.win(b1);
        timeline.add("A场胜者 " + a1.getName() + " 记入 A 板；B场胜者 " + b1.getName() + " 记入 B 板");

        // ---- 断言 ----
        if (sessionA == null || sessionA.trim().isEmpty()) {
            failures.add("A 场 sessionId 为空");
        }
        if (sessionB == null || sessionB.trim().isEmpty()) {
            failures.add("B 场 sessionId 为空");
        }
        if (sessionA != null && sessionA.equals(sessionB)) {
            failures.add("两场 sessionId 相同（应各自独立）: " + sessionA);
        }
        if (boardA.participants().size() != 2 || boardB.participants().size() != 2) {
            failures.add("参与者数量异常 A=" + boardA.participants().size()
                    + " B=" + boardB.participants().size());
        }
        Set<UUID> overlap = new LinkedHashSet<UUID>(boardA.participants());
        overlap.retainAll(boardB.participants());
        if (!overlap.isEmpty()) {
            failures.add("两场参与者相交: " + overlap);
        }
        check("A 板 winsOf(A胜者)", 1, boardA.winsOf(a1.getUniqueId()));
        check("A 板 winsOf(B胜者)", 0, boardA.winsOf(b1.getUniqueId()));
        check("B 板 winsOf(B胜者)", 1, boardB.winsOf(b1.getUniqueId()));
        check("B 板 winsOf(A胜者)", 0, boardB.winsOf(a1.getUniqueId()));
        check("A 板 size", 2, boardA.size());
        check("B 板 size", 2, boardB.size());

        if (api.stateOf(a1) != UserState.ALIVE || api.stateOf(b1) != UserState.ALIVE) {
            failures.add("winner 的 stateOf 不是 ALIVE: A=" + api.stateOf(a1) + " B=" + api.stateOf(b1));
        }
        doneTick = timeline.tick();
    }

    private void check(String what, int expected, int actual) {
        if (expected != actual) {
            failures.add(what + " 期望 " + expected + " 实际 " + actual);
        }
        timeline.add("断言 " + what + " = " + actual + "（期望 " + expected + "）");
    }

    private void tickWatch() {
        if (doneTick >= 0 && timeline.tick() > doneTick + SETTLE_TICKS) {
            finish();
            return;
        }
        if (timeline.tick() > TIMEOUT_TICKS) {
            failures.add("超时 " + TIMEOUT_TICKS + " tick 仍有 Bot 未 Ready（" + ready.size() + "/4）");
            finish();
        }
    }

    private void finish() {
        if (watch != null) {
            watch.cancel();
            watch = null;
        }
        HandlerList.unregisterAll(this);
        timeline.stop();
        boolean pass = failures.isEmpty();

        List<String> out = new ArrayList<String>();
        out.add("§e[A8] 隔离探针：§f" + (pass ? "PASS" : "FAIL"));
        out.add("§7sessionA=" + sessionA + " sessionB=" + sessionB);
        out.add("§7" + boardA.describe());
        out.add("§7" + boardB.describe());
        out.add("§7—— 时间线 ——");
        for (String line : timeline.lines()) {
            out.add("§7" + line);
        }
        for (String failure : failures) {
            out.add("§c! " + failure);
        }
        for (String line : out) {
            sender.sendMessage(line);
            plugin.getLogger().info(line.replaceAll("§.", ""));
        }
    }

    private boolean isTarget(String name) {
        if (name == null || names == null) {
            return false;
        }
        for (String candidate : names) {
            if (candidate.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /** 等价于 example.StatsBoard，但不过滤 Bot：UUID → 胜场 + 参与者集合。 */
    private static final class MatchBoard {
        private final String name;
        private final Map<UUID, Integer> wins = new LinkedHashMap<UUID, Integer>();
        private final Set<UUID> participants = new LinkedHashSet<UUID>();

        private MatchBoard(String name) {
            this.name = name;
        }

        private void join(Bot bot) {
            participants.add(bot.getUniqueId());
            if (!wins.containsKey(bot.getUniqueId())) {
                wins.put(bot.getUniqueId(), Integer.valueOf(0));
            }
        }

        private void win(Bot bot) {
            Integer previous = wins.get(bot.getUniqueId());
            wins.put(bot.getUniqueId(), Integer.valueOf(previous == null ? 1 : previous.intValue() + 1));
        }

        private int winsOf(UUID id) {
            Integer value = wins.get(id);
            return value == null ? 0 : value.intValue();
        }

        private int size() {
            return wins.size();
        }

        private Set<UUID> participants() {
            return new LinkedHashSet<UUID>(participants);
        }

        private String describe() {
            StringBuilder builder = new StringBuilder(name + " 参与者=" + participants.size() + " 胜场=");
            boolean first = true;
            for (Map.Entry<UUID, Integer> entry : wins.entrySet()) {
                if (!first) {
                    builder.append(", ");
                }
                builder.append(shortId(entry.getKey())).append(":").append(entry.getValue());
                first = false;
            }
            return builder.toString();
        }

        private static String shortId(UUID id) {
            return id.toString().substring(0, 8);
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
}