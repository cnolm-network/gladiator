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
        phase = "ended";
        log("对局结束: " + value + " 会话 " + sessionId);
        registry.onSessionEnded(this);
    }

    @Override
    public void phase(String value) {
        if (finished || value == null || value.isEmpty()) {
            return;
        }
        this.phase = value;
    }

    @Override
    public String phase() {
        return finished ? "ended" : phase;
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
        phase = "ended";
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
