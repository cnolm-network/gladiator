package top.cnolm.gladiator.core.board;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.GameEvent;
import top.cnolm.gladiator.api.board.GameEventType;
import top.cnolm.gladiator.api.board.MatchOutcome;
import top.cnolm.gladiator.api.board.ParticipantResolver;
import top.cnolm.gladiator.api.board.PlayerSnapshot;
import top.cnolm.gladiator.api.board.RoundOutcome;
import top.cnolm.gladiator.api.board.RoundSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

/**
 * 一次对局会话：{@link BoardContext} 的实现，同时负责把事件与 tick 转给板子。
 *
 * <p>线程模型：所有入口都断言主线程；板子写错线程会立刻炸掉，而不是让判定错帧。
 */
final class BoardSessionImpl implements BoardContext {

    /**
     * {@code countdown} 阶段的默认时长（秒）。
     *
     * <p>板子只说「现在是倒计时」，没说多久；外面要的是「还剩几秒」。默认对齐 arena 的
     * 5 秒倒计时；板子想要别的时长，目前唯一的口径是它自己的 phase 时序（这边只做兜底显示，
     * 不用来做任何判定——判定永远在板子手里）。
     */
    private static final int DEFAULT_COUNTDOWN_SECONDS = 5;
    /** 同一阶段内 {@code MATCH_PHASE} 的最小重推间隔（毫秒）：每秒一帧。 */
    private static final long PHASE_PUSH_INTERVAL_MS = 1000L;

    private final BoardRegistryImpl registry;
    private final Plugin plugin;
    private final Board board;
    private final GladiatorAPI api;
       private final ParticipantResolver resolver;
    private final RoundSpec spec;
    private final String sessionId;
    private final long startedAt;
    private final List<MatchOutcome.RoundRecord> records = new ArrayList<MatchOutcome.RoundRecord>();

    private int round = 1;
    private long tick;
    private long roundStartedAt;
    private int playerWins;
    private int botWins;
    private int draws;
    private boolean finished;
    private String phase = "starting";
    private String lastReason = "";
    //: G-719：最近一局的 reason 与时长（dashboard 逐局表格用；无局时 none / -1）。
    private String roundReason = "none";
    private double lastSeconds = -1.0D;
    // --- MATCH_PHASE（对局阶段下发给运行时）------------------------------------
    //: 当前阶段窗口的起点、总秒数与结束墙钟；seconds<0 / endsAt=0 表示板子没声明时长。
    private long phaseStartedAt;
    private int phaseSeconds = -1;
    private long phaseEndsAt;
    //: 最近一次下发的时间与秒数：同一阶段内「每 1 秒或秒数变化」才再推一帧。
    private long lastPhasePushAt;
    private int lastPhasePushSeconds = Integer.MIN_VALUE;

    BoardSessionImpl(BoardRegistryImpl registry, Plugin plugin, Board board, GladiatorAPI api,
                     ParticipantResolver resolver, RoundSpec spec, String sessionId) {
        this.registry = registry;
        this.plugin = plugin;
        this.board = board;
        this.api = api;
        this.resolver = resolver;
        this.spec = spec;
        this.sessionId = sessionId;
        this.startedAt = System.currentTimeMillis();
        this.roundStartedAt = this.startedAt;
        this.phaseStartedAt = this.startedAt;
    }

    // ---------------------------------------------------------------- context

    @Override
    public String sessionId() {
        return sessionId;
    }

    @Override
    public String boardId() {
        return board.id();
    }

    @Override
    public RoundSpec spec() {
        return spec;
    }

    @Override
    public int round() {
        return round;
    }

    @Override
    public long tick() {
        return tick;
    }

    @Override
    public long startedAt() {
        return startedAt;
    }

    @Override
    public GladiatorAPI api() {
        return api;
    }

    @Override
    public Player player(String name) {
        requireMainThread();
        if (name == null || name.isEmpty()) {
            return null;
        }
        Player direct = Bukkit.getPlayerExact(name);
        if (direct != null) {
            return direct;
        }
        return resolver == null ? null : resolver.resolve(name);
    }
    @Override
    public PlayerSnapshot snapshot(String player) {
        requireMainThread();
        Player target = player(player);
        if (target == null) {
            return PlayerSnapshot.offline(player);
        }
        Location loc = target.getLocation();
        return new PlayerSnapshot(
                target.getName(),
                target.getUniqueId().toString(),
                true,
                target.getHealth(),
                target.getFoodLevel(),
                target.getSaturation(),
                armorValue(target),
                loc.getX(), loc.getY(), loc.getZ(),
                loc.getYaw(), loc.getPitch(),
                loc.subtract(0.0D, 0.1D, 0.0D).getBlock().getType().isSolid(),
                target.isBlocking(),
                target.isSprinting(),
                target.getMaximumNoDamageTicks() - target.getNoDamageTicks());
    }

    @Override
    public boolean isOnline(String player) {
        requireMainThread();
        return player(player) != null;
    }

    @Override
    public void log(String message) {
        plugin.getLogger().info("[board/" + board.id() + "] " + message);
    }

    @Override
    public void endRound(RoundOutcome outcome) {
        // 旧入口：板子没报时长 → -1，等价于「用墙钟估算」（与 G-719 之前完全一致）。
        endRound(outcome, -1.0D);
    }

    @Override
    public void endRound(RoundOutcome outcome, double seconds) {
        requireMainThread();
        if (finished) {
            return;
        }
        RoundOutcome value = outcome == null ? RoundOutcome.draw("no-outcome") : outcome;
        long now = System.currentTimeMillis();
        // seconds < 0 / NaN = 板子不知道 → 用墙钟（这是 G-719 之前的行为，保持不变）。
        double elapsed = BoardContext.resolveRoundSeconds(seconds, roundStartedAt, now);
        int recordSeconds = BoardContext.recordSeconds(elapsed);
        records.add(new MatchOutcome.RoundRecord(round, value.getWinner(), value.getReason(),
                recordSeconds));
        // 逐局原因与时长：dashboard 的 roundsDetail 直接读这两个（会话级 reason 在 lastReason）。
        roundReason = value.getReason();
        lastSeconds = elapsed;
        if (RoundOutcome.WINNER_PLAYER.equals(value.getWinner())) {
            playerWins++;
        } else if (RoundOutcome.WINNER_BOT.equals(value.getWinner())) {
            botWins++;
        } else {
            draws++;
        }
        log("第 " + round + " 局结束: " + value.getWinner() + " (" + value.getReason() + ", "
                + recordSeconds + "s) 比分 " + playerWins + ":" + botWins);
        round++;
        roundStartedAt = now;
        if (round > spec.getRounds()) {
            endMatch(new MatchOutcome(playerWins, botWins, draws, "rounds-finished", records, null));
        }
    }

    @Override
    public String roundReason() {
        return roundReason;
    }

    @Override
    public double lastSeconds() {
        return lastSeconds;
    }

    @Override
    public void endMatch(MatchOutcome outcome) {
        requireMainThread();
        if (finished) {
            return;
        }
        finished = true;
        MatchOutcome value = outcome == null
                ? new MatchOutcome(playerWins, botWins, draws, "finished", records, null)
                : outcome;
        lastReason = value.getReason();
        // 结束也用同一条迁移路径：运行时要收到 phase=ended 才能全停（不然最后几帧还在出手）。
        transitionPhase("ended", System.currentTimeMillis());
        log("对局结束: " + value + " 会话 " + sessionId);
        registry.onSessionEnded(this);
    }

    @Override
    public void phase(String value) {
        if (finished || value == null || value.isEmpty()) {
            return;
        }
        if (value.equals(phase)) {
            // 同阶段重复声明：不重开时间窗、也不重推（倒计时递减由 ticker 的 pumpPhase 负责）。
            return;
        }
        transitionPhase(value, System.currentTimeMillis());
    }

    @Override
    public String phase() {
        return finished ? "ended" : phase;
    }

    // ------------------------------------------------------------ MATCH_PHASE

    /**
     * 阶段迁移：开新的时间窗（{@code seconds} / {@code endsAt}）并立刻下发一帧。
     *
     * <p>为什么在阶段**变化时**立刻推而不是等下一秒：运行时要靠它决定「这一帧能不能出手」，
     * 晚 1 秒就是倒计时结束后还在待命、或者开局瞬间先挨一刀。
     */
    /**
     * 会话刚创建时补推一帧 {@code starting}。
     *
     * <p>运行时的阶段闸门以「收到第一帧」为开关；板子通常要等自己的 {@code onStart} 里
     * 才调 {@link #phase(String)}，中间那一段会让 Bot 在倒计时前就出手（真机 2026-11-28 03:24）。
     */
    void declareStartingPhase() {
        requireMainThread();
        pushMatchPhase(System.currentTimeMillis(), true);
    }

    private void transitionPhase(String value, long nowMs) {        this.phase = value;
        this.phaseStartedAt = nowMs;
        openPhaseWindow(nowMs);
        pushMatchPhase(nowMs, true);
    }

    /**
     * 按阶段算时间窗。
     *
     * <p>{@code fighting} 用 {@link RoundSpec#getSecondsPerRound()}：倒计时结束后每局的硬上限，
     * 板子超时判定与这里同源（板子自己也可能提前结束，运行时不据此做判定，只用来显示）。
     */
    private void openPhaseWindow(long nowMs) {
        if ("countdown".equals(phase)) {
            phaseSeconds = DEFAULT_COUNTDOWN_SECONDS;
            phaseEndsAt = nowMs + phaseSeconds * 1000L;
        } else if ("fighting".equals(phase)) {
            phaseSeconds = spec.getSecondsPerRound();
            phaseEndsAt = nowMs + phaseSeconds * 1000L;
        } else if ("ended".equals(phase)) {
            phaseSeconds = 0;
            phaseEndsAt = nowMs;
        } else {
            // starting（或板子自造的阶段名）：不知道多久，只报阶段本身。
            phaseSeconds = -1;
            phaseEndsAt = 0L;
        }
    }

    /** 本阶段剩余秒数（向上取整，夹到 {@code [0, phaseSeconds]}）；{@code -1} = 板子没声明时长。 */
    int remainingPhaseSeconds(long nowMs) {
        if (phaseSeconds < 0) {
            return -1;
        }
        if (phaseEndsAt <= 0L) {
            return phaseSeconds;
        }
        long left = phaseEndsAt - nowMs;
        if (left <= 0L) {
            return 0;
        }
        int seconds = (int) Math.ceil(left / 1000.0D);
        return Math.min(seconds, phaseSeconds);
    }

    /**
     * 推一帧 {@code MATCH_PHASE}。
     *
     * @param force {@code true} = 阶段刚变化，无论距上次多久都推；{@code false} = 每秒节流
     */
    private void pushMatchPhase(long nowMs, boolean force) {
        int seconds = remainingPhaseSeconds(nowMs);
        if (!force
                && nowMs - lastPhasePushAt < PHASE_PUSH_INTERVAL_MS
                && seconds == lastPhasePushSeconds) {
            return;
        }
        lastPhasePushAt = nowMs;
        lastPhasePushSeconds = seconds;
        registry.publishMatchPhase(this, phase, round, seconds, phaseEndsAt);
    }

    /**
     * 10Hz ticker 每帧调用：倒计时/战斗中每秒补一帧（秒数递减）。
     *
     * <p>由注册表的 ticker 驱动而不是自己起 {@code BukkitRunnable}：会话在单测里也会被
     * 直接构造（没有 Bukkit 调度器），把节流放在这里就不会因为「没有服务端实例」而崩。
     */
    void pumpPhase(long nowMs) {
        if (finished) {
            return;
        }
        if (!"countdown".equals(phase) && !"fighting".equals(phase)) {
            return;
        }
        pushMatchPhase(nowMs, false);
    }

    /**
     * 本场参与者用户名：取自 {@link RoundSpec#extras} 的 {@code player} / {@code bot}。
     *
     * <p>这是板子的既有约定（见 {@code docs/board-api.md} §3）。真实玩家与 golem 假玩家
     * 也会出现在这里，下发方按 {@code BotRegistry} 过滤，找不到就丢掉。
     */
    List<String> participants() {
        List<String> names = new ArrayList<String>(2);
        addParticipant(names, spec.extra("player", ""));
        addParticipant(names, spec.extra("bot", ""));
        return names;
    }

    private static void addParticipant(List<String> names, String name) {
        if (name == null) {
            return;
        }
        String trimmed = name.trim();
        // 板子没填 extras 时 FlatBoard 会用 "player"/"bot" 当占位名，那不是真的参与者。
        if (trimmed.isEmpty() || "player".equals(trimmed) || "bot".equals(trimmed)) {
            return;
        }
        for (String existing : names) {
            if (existing.equalsIgnoreCase(trimmed)) {
                return;
            }
        }
        names.add(trimmed);
    }

    @Override
    public MatchOutcome summary() {
        return new MatchOutcome(playerWins, botWins, draws,
                lastReason.isEmpty() ? "running" : lastReason, records, null);
    }

    @Override
    public String lastReason() {
        return lastReason;
    }

    // ---------------------------------------------------------------- callbacks

    void onEvent(GameEvent event) {
        if (finished) {
            return;
        }
        try {
            board.onEvent(event);
        } catch (RuntimeException e) {
            plugin.getLogger().log(Level.WARNING,
                    "[board/" + board.id() + "] onEvent 抛异常，忽略本条事件: " + event.getType(), e);
        }
    }

    // ------------------------------------------------------------ Bot 生命周期事件

    /**
     * Bot 会话就绪 → {@link GameEventType#BOT_READY}。
     *
     * <p>事件形状与 Bukkit 翻译出来的其余事件一致（seq/ts/round 都齐），板子不必区分来源：
     * {@code actor} = 用户名，{@code data.botId} = 插件的 {@code glad-<小写用户名>}。
     * {@code data.reason} 固定写入空串，保证两种 Bot 事件的 data 键集相同（板子可以少判空）。
     */
    void onBotReady(String username, String botId) {
        emitBotEvent(GameEventType.BOT_READY, username, botId, "");
    }

    /** Bot 失效/未起来 → {@link GameEventType#BOT_OFFLINE}；{@code reason} 原样带出。 */
    void onBotOffline(String username, String botId, String reason) {
        emitBotEvent(GameEventType.BOT_OFFLINE, username, botId, reason == null ? "" : reason);
    }

    private void emitBotEvent(GameEventType type, String username, String botId, String reason) {
        if (finished) {
            return;
        }
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("botId", botId == null ? "" : botId);
        data.put("reason", reason);
        GameEvent event = GameEvent.of(type, registry.nextSeq(), System.currentTimeMillis(),
                round, username, "", data);
        onEvent(event);
    }

    void onTick() {
        if (finished) {
            return;
        }
        tick++;
        try {
            board.onTick(tick);
        } catch (RuntimeException e) {
            plugin.getLogger().log(Level.WARNING,
                    "[board/" + board.id() + "] onTick 抛异常，终止对局", e);
            finish("tick-error");
        }
    }

    void finish(String reason) {
        if (finished) {
            return;
        }
        finished = true;
        lastReason = reason == null ? "stopped" : reason;
        transitionPhase("ended", System.currentTimeMillis());
        try {
            board.onStop(reason);
        } catch (RuntimeException e) {
            plugin.getLogger().log(Level.WARNING, "[board/" + board.id() + "] onStop 抛异常", e);
        }
        registry.onSessionEnded(this);
    }

    boolean isFinished() {
        return finished;
    }

    private static double armorValue(Player player) {
        double total = 0.0D;
        for (org.bukkit.inventory.ItemStack item : player.getInventory().getArmorContents()) {
            if (item == null) {
                continue;
            }
            String type = item.getType().name();
            if (type.startsWith("DIAMOND_")) {
                total += 8.0D;
            } else if (type.startsWith("IRON_")) {
                total += 6.0D;
            } else if (type.startsWith("GOLD_")) {
                total += 5.0D;
            } else if (type.startsWith("CHAINMAIL_")) {
                total += 5.0D;
            } else if (type.startsWith("LEATHER_")) {
                total += 3.0D;
            }
        }
        return total;
    }

    private void requireMainThread() {
        if (!Bukkit.isPrimaryThread()) {
            throw new IllegalStateException(
                    "BoardContext 只能在主线程调用（板子 " + board.id() + " 在异步线程调用了上下文方法）");
        }
    }
}
