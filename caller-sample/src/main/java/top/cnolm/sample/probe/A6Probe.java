package top.cnolm.sample.probe;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.event.GladiatorBotFailedEvent;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;
import top.cnolm.gladiator.api.event.GladiatorStateChangedEvent;
import top.cnolm.gladiator.example.arena.DuelMode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A6 探针：{@code Ready → 调用方传送 → 套件 → pushGame → StateChanged} 的顺序与结果。
 *
 * <p>为什么要自带一个调用方而不是直接复用 {@code MockArenaCaller}/{@code DuelCaller}：
 * 那两个示例把 {@code pushGame} 的 sessionId 只写进日志，没有 getter，无法断言"返回非空"。
 * 本类是**等价调用方**（链路与 {@code DuelCaller#onBotReady} 完全一致：Ready 里传送、发套件、
 * 两只都 Ready 后 pushGame），只是把每一步都记进时间线。
 *
 * <p>断言：
 * <ol>
 *   <li>每只 Bot 的 Ready 早于它的 pushGame（用同 tick 内可比的序号）；</li>
 *   <li>两只都 Ready 之后才 pushGame；</li>
 *   <li>sessionId 非空，且两只 Bot 的 {@code stateOf} 为 ALIVE。</li>
 * </ol>
 */
public final class A6Probe implements Listener {

    private static final int TIMEOUT_TICKS = 240;
    private static final int SETTLE_TICKS = 20;

    private final Plugin plugin;
    private final GladiatorAPI api;
    private final CommandSender sender;
    private final Location spawn;
    private final ProbeTimeline timeline = new ProbeTimeline();

    private final Map<String, int[]> readyMarks = new LinkedHashMap<String, int[]>(); // name -> {tick, seq}
    private final Map<String, Bot> readyBots = new LinkedHashMap<String, Bot>();
    private final List<String> stateLines = new ArrayList<String>();
    private final List<String> failures = new ArrayList<String>();

    private String[] names;
    private String sessionId;
    private int pushTick = -1;
    private int pushSeq = -1;
    private BukkitTask watch;

    public A6Probe(Plugin plugin, GladiatorAPI api, CommandSender sender, Location spawn) {
        this.plugin = plugin;
        this.api = api;
        this.sender = sender;
        this.spawn = spawn;
    }

    public void start() {
        String suffix = String.valueOf(System.currentTimeMillis() % 100000L);
        names = new String[]{"a6a" + suffix, "a6b" + suffix};
        timeline.start(plugin);
        Bukkit.getPluginManager().registerEvents(this, plugin);
        for (String name : names) {
            timeline.add("spawnBot 请求 " + name);
            api.spawnBot(BotSpec.of(name, "a6"));
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
        int seq = timeline.add("GladiatorBotReadyEvent " + name + " botId=" + event.getBot().getBotId());
        readyMarks.put(name, new int[]{timeline.tick(), seq});

        Player player = event.getPlayer();
        if (player == null) {
            failures.add(name + " 的 Ready 事件 getPlayer() 为 null");
            return;
        }
        // G-727：Ready 回调里客户端还没跟上，立即发一次 + 之后每 10 tick 补发。
        teleportWithReassert(plugin, player, spawn);
        timeline.add("调用方 teleport " + name + " → " + fmt(spawn));
        player.getInventory().addItem(new ItemStack(Material.DIAMOND_SWORD));
        timeline.add("调用方发套件 " + name + "（+1 diamond_sword）");

        readyBots.put(name, event.getBot());
        if (readyBots.size() == names.length && sessionId == null) {
            pushGame();
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onState(GladiatorStateChangedEvent event) {
        User user = event.getUser();
        if (!isTarget(user.getName())) {
            return;
        }
        timeline.add("GladiatorStateChangedEvent " + user.getName() + " "
                + event.getFrom() + "→" + event.getTo());
        stateLines.add(user.getName() + " " + event.getFrom() + "→" + event.getTo());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onFailed(GladiatorBotFailedEvent event) {
        if (isTarget(event.getBotId()) || isTargetByBotId(event.getBotId())) {
            timeline.add("GladiatorBotFailedEvent " + event.getBotId() + " reason=" + event.getReason());
            failures.add("Bot 登录失败: " + event.getBotId() + " " + event.getReason());
        }
    }

    private void pushGame() {
        pushTick = timeline.tick();
        pushSeq = timeline.add("pushGame(DuelMode, " + readyBots.keySet() + ") 调用");
        Bot[] bots = readyBots.values().toArray(new Bot[0]);
        GameSession session;
        try {
            session = api.pushGame(new DuelMode(), bots);
        } catch (RuntimeException ex) {
            failures.add("pushGame 抛异常: " + ex);
            finish();
            return;
        }
        sessionId = session == null ? null : session.getSessionId();
        timeline.add("pushGame 返回 sessionId=" + sessionId + " users=" + session.getUsers().size());
        for (Bot bot : bots) {
            UserState state = api.stateOf(bot);
            timeline.add("stateOf(" + bot.getName() + ")=" + state);
            if (state != UserState.ALIVE) {
                failures.add(bot.getName() + " 的 stateOf 不是 ALIVE: " + state);
            }
        }
    }

    private void tickWatch() {
        if (sessionId != null && timeline.tick() > pushTick + SETTLE_TICKS) {
            finish();
            return;
        }
        if (timeline.tick() > TIMEOUT_TICKS) {
            failures.add("超时 " + TIMEOUT_TICKS + " tick 仍未完成流程");
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

        // ---- 断言 ----
        boolean pass = true;
        if (readyMarks.size() != names.length) {
            failures.add("Ready 数量 " + readyMarks.size() + " != " + names.length);
            pass = false;
        }
        if (sessionId == null || sessionId.trim().isEmpty()) {
            failures.add("pushGame 返回的 sessionId 为空");
            pass = false;
        }
        for (String name : names) {
            int[] mark = readyMarks.get(name);
            if (mark == null) {
                pass = false;
                continue;
            }
            if (pushSeq < 0) {
                pass = false;
                continue;
            }
            if (!(mark[1] < pushSeq)) {
                failures.add(name + " 的 Ready 序号 " + mark[1] + " 不早于 pushGame 序号 " + pushSeq);
                pass = false;
            }
            if (!(mark[0] <= pushTick)) {
                failures.add(name + " 的 Ready tick " + mark[0] + " 晚于 pushGame tick " + pushTick);
                pass = false;
            }
        }
        // 状态事件不是 A6 的硬断言：pushGame 前两者已是 ALIVE，pushGame 不产生"变化"，
        // 因此通常没有 StateChanged。这里只如实说明，不判失败。
        for (String failure : failures) {
            pass = false;
        }

        List<String> out = new ArrayList<String>();
        out.add("§e[A6] 顺序探针：§f" + (pass ? "PASS" : "FAIL"));
        out.add("§7session=" + sessionId + "  bots=" + String.join(",", names));
        out.add("§7—— 时间线 ——");
        for (String line : timeline.lines()) {
            out.add("§7" + line);
        }
        if (!stateLines.isEmpty()) {
            out.add("§7状态迁移：" + String.join(" | ", stateLines));
        } else {
            out.add("§7状态迁移：（无）pushGame 前两者已是 ALIVE，没有 ALIVE→ALIVE 的迁移事件");
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

    private boolean isTargetByBotId(String botId) {
        if (botId == null || names == null) {
            return false;
        }
        for (String candidate : names) {
            if (botId.toLowerCase().contains(candidate.toLowerCase())) {
                return true;
            }
        }
        return false;
    }

    private static String fmt(Location location) {
        return String.format("%s %.2f,%.2f,%.2f", location.getWorld().getName(),
                location.getX(), location.getY(), location.getZ());
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