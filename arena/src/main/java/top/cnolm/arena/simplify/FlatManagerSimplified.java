package top.cnolm.arena.simplify;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.event.Event;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.scheduler.BukkitRunnable;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.arena.ArenaPlugin;
import top.cnolm.arena.FlatMatchManager;
import top.cnolm.arena.event.*;
import top.cnolm.arena.status.ArenaMatch;
import top.cnolm.arena.status.ArenaTeam;
import top.cnolm.arena.status.MatchManager;
import top.cnolm.arena.status.PlayerMatchData;
import top.cnolm.arena.status.PlayerResolver;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * 每个对局使用一个 {@link FlatManagerSimplified}。
 *
 * <p>对 {@link MatchManager} 驱动的 {@link FlatMatchManager} 的简化 API：
 * 用 {@link StartParams} 构建启动参数，无需 build()。
 *
 * <p>本类同时是录像采集器的一个实现：监听所有 arena 事件写入 {@link Recorder} 的四个 recordXXX 入口——
 * {@code recordEvent}（逐事件）、{@code recordPosition}（周期位置）、
 * {@code recordMatchStats}（击杀/淘汰/结算快照）、{@code record}（启停等自定义节点）。
 */
public class FlatManagerSimplified implements Listener {
    private FlatMatchManager flatMatchManager = null;
    private Recorder recorder = null;
    private boolean registered = false;
    public Map<UUID, BukkitTask> recordingTaskMap = new HashMap<>();

    public FlatManagerSimplified() {
        register();
    }

    private void register() {
        if (registered) {
            return;
        }
        Bukkit.getPluginManager().registerEvents(this, ArenaPlugin.getInstance());
        registered = true;
    }

    public void start(Player p1, Player p2) {
        start(new StartParams(p1, p2));
    }

    public void start(Player p1, Player p2, String kitId, int shieldSeconds) {
        start(new StartParams(p1, p2), kitId, shieldSeconds);
    }

    public void start(StartParams params) {
        flatMatchManager = params.build().effect(this);
    }

    public void start(StartParams params, String kitId, int shieldSeconds) {
        flatMatchManager = params.build().effect(this, kitId, shieldSeconds);
    }

    // ------------------------------------------------------------ 采集入口

    /** 周期位置采集：由 StartParams 在 effect 时启动。 */
    private void startRecording(Player player, int period) {
        cancelRecording(player.getUniqueId());
        BukkitTask task = new BukkitRunnable() {
            @Override
            public void run() {
                recordPosition(player);
            }
        }.runTaskTimer(ArenaPlugin.getInstance(), 0, period);
        recordingTaskMap.put(player.getUniqueId(), task);
    }

    private void cancelRecording(UUID uuid) {
        BukkitTask task = recordingTaskMap.remove(uuid);
        if (task != null) {
            task.cancel();
        }
    }

    private void stopAllRecording() {
        for (UUID uuid : new java.util.ArrayList<>(recordingTaskMap.keySet())) {
            cancelRecording(uuid);
        }
    }

    // recordXXX 的四个统一入口：都做空值保护，未开赛/无 Recorder 时静默跳过。

    private void recordPosition(Player player) {
        if (recorder != null && player != null) {
            recorder.recordPosition(FlatManagerSimplified.class, player);
        }
    }

    private void recordEvent(Event event, String detail) {
        if (recorder != null && isOurMatch(event)) {
            recorder.recordEvent(FlatManagerSimplified.class, event, detail);
        }
    }

    private void recordStats(String category, String content) {
        if (recorder != null) {
            recorder.recordMatchStats(FlatManagerSimplified.class, category, content);
        }
    }

    private void note(String category, String content) {
        if (recorder != null) {
            recorder.record(FlatManagerSimplified.class, category, content);
        }
    }

    /** 事件是否属于本包装类当前持有的对局；非 arena 事件（如 PlayerQuitEvent）返回 false。 */
    private boolean isOurMatch(Event event) {
        if (flatMatchManager == null) {
            return false;
        }
        UUID matchId;
        if (event instanceof ArenaEvent) {
            matchId = ((ArenaEvent) event).getMatchId();
        } else if (event instanceof ArenaMatchEvent) {
            matchId = ((ArenaMatchEvent) event).getMatchId();
        } else {
            return false;
        }
        return matchId != null && matchId.equals(flatMatchManager.getMatchId());
    }

    /** 当前对局的玩家数据快照，作为统计记录的内容。 */
    private String statsSnapshot() {
        if (flatMatchManager == null) {
            return "{}";
        }
        ArenaMatch<?, ?> match = MatchManager.getMatch(flatMatchManager.getMatchId());
        if (match == null) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        for (PlayerMatchData<?> data : match.getPlayerDataMap().values()) {
            if (sb.length() > 1) {
                sb.append(", ");
            }
            sb.append(name(PlayerResolver.get(data.getUuid())))
                    .append(": k=").append(data.getKills())
                    .append(" d=").append(data.getDeaths())
                    .append(" a=").append(data.getAssists())
                    .append(" score=").append(data.getScore())
                    .append(" status=").append(data.getStatus().getName());
        }
        return sb.append('}').toString();
    }

    private static String name(Player player) {
        return player == null ? "未知玩家" : player.getName();
    }

    // ------------------------------------------------------------ 生命周期事件

    @EventHandler(priority = EventPriority.MONITOR)
    public void onMatchStart(ArenaMatchStartEvent e) {
        recordEvent(e, "round=" + e.getRound());
        note("match-start", "round=" + e.getRound());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onKill(ArenaKillEvent e) {
        recordEvent(e, "victim=" + name(e.getVictim()) + " killer=" + name(e.getKiller())
                + " cause=" + e.getCause());
        recordStats("kill", name(e.getKiller()) + " -> " + name(e.getVictim()) + " " + statsSnapshot());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onEliminate(ArenaEliminateEvent e) {
        recordEvent(e, "victim=" + name(e.getVictim()) + " reason=" + e.getReason());
        recordStats("eliminate", name(e.getVictim()) + " reason=" + e.getReason() + " " + statsSnapshot());
        Player victim = e.getVictim();
        if (victim != null) {
            cancelRecording(victim.getUniqueId());
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onMatchEnd(ArenaMatchEndEvent e) {
        recordEvent(e, "reason=" + e.getReason() + " outcome=" + e.getOutcome());
        recordStats("match-end", "reason=" + e.getReason() + " outcome=" + e.getOutcome()
                + " " + statsSnapshot());
        note("match-end", "recording stopped");
        stopAllRecording();
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onQuit(PlayerQuitEvent e) {
        // 非 arena 事件，用 record 记自定义节点
        note("player-quit", name(e.getPlayer()) + " 退出服务器");
        cancelRecording(e.getPlayer().getUniqueId());
    }

    // ------------------------------------------------------------ 行为事件（逐事件接入 recordEvent）

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDamage(ArenaDamageEvent e) {
        recordEvent(e, "victim=" + name(e.getVictim()) + " attacker=" + name(e.getAttacker())
                + " damage=" + e.getDamage() + " cause=" + e.getCause());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onArrowHit(ArenaArrowHitEvent e) {
        recordEvent(e, "victim=" + name(e.getVictim()) + " shooter=" + name(e.getShooter())
                + " damage=" + e.getDamage());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onProjectileHit(ArenaProjectileHitEvent e) {
        recordEvent(e, "projectile=" + e.getProjectile().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onUseSplashPotion(ArenaUseSplashPotionEvent e) {
        recordEvent(e, "potion=" + e.getPotion().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onThrow(ArenaThrowEvent e) {
        recordEvent(e, "item=" + e.getItem().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onEat(ArenaEatEvent e) {
        recordEvent(e, "food=" + e.getFood().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDrink(ArenaDrinkEvent e) {
        recordEvent(e, "drink=" + e.getDrink().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onSwordBlock(ArenaSwordBlockEvent e) {
        recordEvent(e, "sword=" + e.getSword().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onPlaceFluid(ArenaPlaceFluidEvent e) {
        recordEvent(e, "fluid=" + e.getFluid() + " block=" + e.getBlock().getLocation());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onBlockPlace(ArenaBlockPlaceEvent e) {
        recordEvent(e, "block=" + e.getBlock().getLocation());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onBlockBreak(ArenaBlockBreakEvent e) {
        recordEvent(e, "block=" + e.getBlock().getLocation());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onItemDrop(ArenaItemDropEvent e) {
        recordEvent(e, "item=" + e.getItem().getItemStack().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onItemPickup(ArenaItemPickupEvent e) {
        recordEvent(e, "item=" + e.getItem().getItemStack().getType());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onWorldChange(ArenaChangedWorldEvent e) {
        recordEvent(e, "from=" + e.getFrom().getName() + " to=" + e.getTo().getName());
    }

    public FlatMatchManager getFlatMatchManager() {
        return flatMatchManager;
    }

    /** 当前 Recorder，供调用方导出 {@link Recorder#getContents()}。 */
    public Recorder getRecorder() {
        return recorder;
    }

    void setRecorder(Recorder recorder) {
        this.recorder = recorder;
    }

    public interface AsTeam {
        ArenaTeam run(Player p);
    }

    public static final class StartParams {
        public Location mid = null;
        public Integer spawnVecX = null;
        public Integer spawnVecZ = null;
        public ArenaTeam t1 = null;
        public ArenaTeam t2 = null;
        public Recorder recorder = null;
        public Integer recordPeriod = null;

        private final Player p1;
        private final Player p2;
        private boolean isBuilt = false;

        public StartParams(Player p1, Player p2) {
            this.p1 = p1;
            this.p2 = p2;
        }

        public StartParams mid(Location mid) {
            this.mid = mid;
            return this;
        }

        public StartParams spawnVec(int x, int z) {
            this.spawnVecX = x;
            this.spawnVecZ = z;
            return this;
        }

        public StartParams team(AsTeam maker) {
            this.t1 = maker.run(p1);
            this.t2 = maker.run(p2);
            return this;
        }

        public StartParams team(ArenaTeam t1, ArenaTeam t2) {
            this.t1 = t1;
            this.t2 = t2;
            return this;
        }

        public StartParams recorder(Recorder recorder) {
            this.recorder = recorder;
            return this;
        }

        public StartParams recordPeriod(int period) {
            this.recordPeriod = period;
            return this;
        }

        StartParams build() {
            this.isBuilt = true;
            if (mid == null) mid = new Location(p1.getWorld(), 0, 4, 0);
            if (spawnVecX == null) spawnVecX = 10;
            if (spawnVecZ == null) spawnVecZ = 10;
            if (recordPeriod == null) recordPeriod = 1;
            if (recorder == null) recorder = new Recorder(System::currentTimeMillis);
            // clone：Location.add 是原地修改，直接 add 会让两队共用同一个出生点
            if (t1 == null) t1 = new ArenaTeam("team1", ChatColor.WHITE,
                    mid.clone().add(spawnVecX, 0, spawnVecZ), Collections.singletonList(p1.getUniqueId()));
            if (t2 == null) t2 = new ArenaTeam("team2", ChatColor.WHITE,
                    mid.clone().add(-spawnVecX, 0, -spawnVecZ), Collections.singletonList(p2.getUniqueId()));
            return this;
        }

        FlatMatchManager effect(FlatManagerSimplified owner, String kitId, int shieldSeconds) {
            if (!isBuilt) throw new IllegalStateException("not built");
            owner.setRecorder(recorder);
            owner.startRecording(p1, recordPeriod);
            owner.startRecording(p2, recordPeriod);
            owner.note("setup", "kit=" + kitId + " shield=" + shieldSeconds + " period=" + recordPeriod);
            return new FlatMatchManager(t1, t2, kitId, shieldSeconds, recorder);
        }

        FlatMatchManager effect(FlatManagerSimplified owner) {
            if (!isBuilt) throw new IllegalStateException("not built");
            owner.setRecorder(recorder);
            owner.startRecording(p1, recordPeriod);
            owner.startRecording(p2, recordPeriod);
            owner.note("setup", "period=" + recordPeriod);
            return new FlatMatchManager(t1, t2);
        }
    }
}
