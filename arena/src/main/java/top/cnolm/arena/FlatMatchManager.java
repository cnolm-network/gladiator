package top.cnolm.arena;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.potion.PotionEffect;
import org.bukkit.scheduler.BukkitRunnable;
import top.cnolm.arena.event.ArenaEliminateEvent;
import top.cnolm.arena.event.ArenaKillEvent;
import top.cnolm.arena.event.ArenaMatchEndEvent;
import top.cnolm.arena.simplify.Recorder;
import top.cnolm.arena.status.*;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 每局游戏产生一个，结束即销毁 </br>
 * 我不知道你是每一局算作一个 Match 还是一整轮算作一个 Match，我这边认为是前者
 */
public class FlatMatchManager implements Listener {
    private final ArenaMatch<GamingStatusImpl, MatchStatusImpl> match;
    private final String kitId;
    private final int shieldSeconds;
    private final Recorder recorder;
    private static final int COUNTDOWN_TIME = 5;
    private static final boolean ALLOW_MOVE_IN_COUNTDOWN = false;
    private static final Map<UUID, FlatMatchManager> active = new ConcurrentHashMap<>();

    public FlatMatchManager(ArenaTeam t1, ArenaTeam t2) {
        this(t1, t2, ArenaKits.CLASSIC, 0, new Recorder(System::currentTimeMillis));
    }

    /**
     * @param kitId         套件（未知值回退 classic 并告警）
     * @param shieldSeconds 倒计时免伤秒数（0 = 不免伤）
     * @param recorder      录像记录器
     */
    public FlatMatchManager(ArenaTeam t1, ArenaTeam t2, String kitId, int shieldSeconds, Recorder recorder) {
        this.match = new ArenaMatch<>(GamingStatusImpl.class, MatchStatusImpl.class);
        String normalized = ArenaKits.normalize(kitId);
        if (normalized == null) {
            ArenaPlugin.getInstance().getLogger().warning("[flat] 未知套件 " + kitId + "，按 classic 发");
            normalized = ArenaKits.CLASSIC;
        }
        this.kitId = normalized;
        this.shieldSeconds = Math.max(0, shieldSeconds);
        this.recorder = recorder;

        startMatch(t1, t2);
    }

    /** 对局入口：注册监听 → 布置场地 → 登记对局 → 开始倒计时。 */
    private void startMatch(ArenaTeam t1, ArenaTeam t2) {
        Bukkit.getPluginManager().registerEvents(this, ArenaPlugin.getInstance());
        match.setStatus(MatchStatusImpl.PREPARING);
        match.addTeam(t1);
        match.addTeam(t2);
        t1.toSpawn();
        t2.toSpawn();
        // 先重置状态（会清掉全部药水效果），再发套件，最后上免伤
        match.forEachOnlinePlayer(this::resetStatus);
        match.forEachOnlinePlayer(this::giveKit);
        match.forEachOnlinePlayer(this::applyShield);
        active.put(match.getId(), this);
        MatchManager.startMatch(match);
        startCountdown();
    }

    /** Bot 晚于 startMatch Ready 时补做单人布场（套件/满状态/传送到所属队伍出生点）。 */
    public static void onParticipantReady(UUID playerId) {
        for (FlatMatchManager manager : active.values()) {
            ArenaMatch<GamingStatusImpl, MatchStatusImpl> m = manager.match;
            if (m.getPlayerData(playerId) == null) {
                continue;
            }
            Player player = PlayerResolver.get(playerId);
            if (player == null) {
                continue;
            }
            manager.resetStatus(player);
            manager.giveKit(player);
            if (m.getStatus() == MatchStatusImpl.COUNTDOWN) {
                manager.applyShield(player);
            }
            for (ArenaTeam team : m.getTeams()) {
                if (team.getPlayers().contains(playerId)) {
                    player.teleport(team.getSpawn());
                    break;
                }
            }
        }
    }

    /** Bot 被回收（golem despawn 无 Bukkit 事件）时按 DISCONNECT 淘汰。 */
    public static void onParticipantRemoved(UUID playerId) {
        for (FlatMatchManager manager : active.values()) {
            ArenaMatch<GamingStatusImpl, MatchStatusImpl> m = manager.match;
            if (m.getPlayerData(playerId) == null || m.getStatus() == MatchStatusImpl.END) {
                continue;
            }
            // 文字由 onEliminate 统一广播，避免同一淘汰发两条消息
            m.eliminatePlayer(playerId, null, EliminateReason.DISCONNECT);
        }
    }

    private void giveKit(Player target) {
        ArenaKits.give(target, kitId);
    }

    private void applyShield(Player target) {
        ArenaKits.applyShield(target, shieldSeconds > 0 ? shieldSeconds : COUNTDOWN_TIME);
    }

    private void resetStatus(Player target) {
        target.spigot().respawn();
        target.setMaxHealth(20);
        target.setHealth(20);
        target.setFoodLevel(20);
        target.setSaturation(5.0F);
        target.setFireTicks(0);
        target.setFallDistance(0.0F);
        target.setAllowFlight(false);
        target.setFlying(false);
        target.setGameMode(GameMode.SURVIVAL);
        target.setLevel(0);
        target.setExp(0);
        for (PotionEffect effect : target.getActivePotionEffects()) {
            target.removePotionEffect(effect.getType());
        }
    }

    private void startCountdown() {
        match.setStatus(MatchStatusImpl.COUNTDOWN);
        match.forEachOnlinePlayer(match::startCountdown);
        match.forEachOnlinePlayer(player -> player.sendMessage("§a比赛将在 5 秒后开始"));
        new BukkitRunnable() {
            int time = COUNTDOWN_TIME;
            @Override
            public void run() {
                if (time == 0) {
                    cancel();
                    match.setStatus(MatchStatusImpl.GAMING);
                    // 开打瞬间解免伤：抗性 V 留着会让前几秒的伤害全部无效
                    match.forEachOnlinePlayer(ArenaKits::clearShield);
                    match.forEachOnlinePlayer(player -> player.sendMessage("§a比赛开始"));
                    match.forEachOnlinePlayer(match::startFighting);
                    this.cancel();
                    return;
                } else {
                    match.forEachOnlinePlayer(player -> player.sendMessage("§a比赛将在 " + time + " 秒后开始"));
                    time--;
                }
            }
        }.runTaskTimer(ArenaPlugin.getInstance(), 0, 20);
    }

    public UUID getMatchId() {
        return match.getId();
    }

    /** 对局结束回调：由 Board 适配层注入，用来把 arena 的结算转成 RoundOutcome。 */
    public interface ResultListener {
        /**
         * @param manager 结束的那个 manager
         * @param winner  获胜队伍，平局为 null
         * @param reason  arena 侧的结束原因（one-team-left / no-team-left）
         */
        void onResult(FlatMatchManager manager, ArenaTeam winner, String reason);
    }

    private volatile ResultListener resultListener;

    public void setResultListener(ResultListener listener) {
        this.resultListener = listener;
    }

    /** 是否已进入「战斗中」（倒计时结束）。超时判定只在这种状态下开始计时。 */
    public boolean isFighting() {
        return match.getStatus() == MatchStatusImpl.GAMING;
    }

    /** 本局实际使用的套件（未知值在构造时已回退成 classic）。 */
    public String kitId() {
        return kitId;
    }

    /** 是否已结束（避免对已结束的局重复结算）。 */
    public boolean isEnded() {
        return match.getStatus() == MatchStatusImpl.END;
    }

    /**
     * 强制结束本局：把所有仍存活的成员都标记出局，结算成平局。
     *
     * <p>不能逐个 {@code eliminatePlayer}——第一次淘汰就会触发 checkEnd 并把胜利判给另一方。
     * 这里走 {@link ArenaMatch#eliminateAllWithoutEvents} 一次性清空，再由 arena 自己的
     * 结束路径（事件、文字、清理）收敛。
     */
    public void forceEnd() {
        if (isEnded()) {
            return;
        }
        broadcast("§e本局超时，判平局");
        match.eliminateAllWithoutEvents(EliminateReason.ABANDON);
    }

    /** 倒计时阶段禁止位移：把目标点拉回原点，保留视角转动，避免生硬 cancel 导致回弹抖动。 */
    @EventHandler(priority = EventPriority.NORMAL)
    public void onMove(PlayerMoveEvent event) {
        if (ALLOW_MOVE_IN_COUNTDOWN || match.getStatus() != MatchStatusImpl.COUNTDOWN) {
            return;
        }
        Player player = event.getPlayer();
        if (match.getPlayerData(player.getUniqueId()) == null) {
            return;
        }
        event.setTo(event.getFrom());
    }

    /** 真人掉线：按 DISCONNECT 淘汰；PlayerQuitEvent 时 Player 仍可用于文字提示。 */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onQuit(PlayerQuitEvent event) {
        Player player = event.getPlayer();
        if (match.getPlayerData(player.getUniqueId()) == null) {
            return;
        }
        match.eliminatePlayer(player.getUniqueId(), null, EliminateReason.DISCONNECT);
    }

    /** 击杀即淘汰：只处理本对局，killer 为 null 时按非击杀原因淘汰。 */
    @EventHandler(priority = org.bukkit.event.EventPriority.MONITOR)
    public void onKill(ArenaKillEvent event) {
        if (!match.getId().equals(event.getMatchId()) || event.getVictim() == null) {
            return;
        }
        Player victim = event.getVictim();
        Player killer = event.getKiller();
        EliminateReason reason = killer == null ? EliminateReason.FORCE : EliminateReason.KILL;
        ArenaPlugin.getInstance().getLogger().info("[flat] 击杀事件 victim=" + victim.getName()
                + " killer=" + (killer == null ? "null" : killer.getName())
                + " reason=" + reason);
        match.eliminatePlayer(victim.getUniqueId(), killer == null ? null : killer.getUniqueId(), reason);
        broadcast("§c" + name(victim) + " §7被 §c" + name(killer) + " §7击杀");
    }

    /** 每次淘汰后查询对局状态；非击杀淘汰在此补文字（击杀文字已由 onKill 发出）。 */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onEliminate(ArenaEliminateEvent event) {
        if (!match.getId().equals(event.getMatchId())) {
            return;
        }
        // 诊断用：淘汰来源与原因直接进服务器日志，便于事后判断是谁、为什么出局。
        ArenaPlugin.getInstance().getLogger().info("[flat] 淘汰 " + name(event.getVictim())
                + " reason=" + event.getReason()
                + " status=" + match.getStatus().getName());
        if (event.getReason() != EliminateReason.KILL) {
            broadcast(reasonText(event.getReason(), event.getVictim()));
        }
        match.checkEnd();
    }

    private String reasonText(EliminateReason reason, Player victim) {
        String who = "§c" + name(victim) + "§7";
        switch (reason) {
            case DISCONNECT:
                return who + " 掉线，判定出局";
            case ABANDON:
                return who + " 主动退出对战";
            case FORCE:
                return who + " 非击杀死亡，判定出局";
            default:
                return who + " 出局";
        }
    }

    private static String name(Player player) {
        return player == null ? "未知玩家" : player.getName();
    }

    private void broadcast(String message) {
        match.forEachOnlinePlayer(p -> p.sendMessage(message));
    }

    /** 对局结束后注销监听，避免 Manager 随对局累积。 */
    @EventHandler(priority = org.bukkit.event.EventPriority.MONITOR)
    public void onMatchEnd(ArenaMatchEndEvent event) {
        if (!match.getId().equals(event.getMatchId())) {
            return;
        }
        Object outcome = event.getOutcome();
        ArenaTeam winner;
        if (outcome instanceof ArenaTeam) {
            winner = (ArenaTeam) outcome;
            broadcast("§a" + winner.getName() + " §7获胜");
        } else {
            winner = null;
            broadcast("§e本局无人存活，平局");
        }
        MatchManager.endMatch(match);
        active.remove(match.getId());
        HandlerList.unregisterAll(this);
        ResultListener listener = this.resultListener;
        this.resultListener = null;
        if (listener != null) {
            listener.onResult(this, winner, event.getReason());
        }
    }
}
