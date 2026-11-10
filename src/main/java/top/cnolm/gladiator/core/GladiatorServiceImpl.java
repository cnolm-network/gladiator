package top.cnolm.gladiator.core;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import top.cnolm.gladiator.api.Analysis;
import top.cnolm.gladiator.api.AnalysisQuery;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.InformationKind;
import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.TrainingSession;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.UserStateRegistry;
import top.cnolm.gladiator.api.Users;
import top.cnolm.gladiator.api.event.GladiatorBotRemovedEvent;
import top.cnolm.gladiator.api.event.GladiatorSessionEndedEvent;
import top.cnolm.gladiator.api.event.GladiatorStateChangedEvent;
import top.cnolm.gladiator.core.guard.GuardBridge;
import top.cnolm.gladiator.core.guard.NoopGuardBridge;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link GladiatorAPI} 的 M1 实现（G-008 / G-301 / G-303）。
 *
 * <p>能力：服务注册、{@code isReady()} / {@code getApiVersion()}、Bot spawn 票据、
 * 对局与训练会话登记、{@code notice()} 投递、{@code analyze()} 快照、
 * {@code eliminate()} / {@code stateOf()} 状态机、WorldGuard PVP 规则查询。
 *
 * <p>{@code spawnBot()} 委托 {@link BotSpawner}（G-107 / B1）：本类只签发请求票据，
 * 真实登录（运行时）或身体生成（Golem）由具体 spawner 在主线程完成。未接线（M0 独立测试）时
 * 显式抛 {@link IllegalStateException}，绝不伪造 Bot。
 *
 * <p>本类同时实现 {@link UserStateRegistry.Lookup}，让 {@code Users.of(Player).getState()}
 * 能读到同一份状态，避免双状态源。所有状态迁移只发生在本类。
 */
public final class GladiatorServiceImpl implements GladiatorAPI, UserStateRegistry.Lookup {

    /** 语义化版本；只增不改。1.1.0 新增 {@code isPvpAllowed} / {@code queryPvp}。 */
    public static final String API_VERSION = "1.1.0";

    /** 单个 Bot 在 WS 建立前允许积压的 notice 数量，超出丢弃最旧的并计数。 */
    private static final int MAX_PENDING_NOTICES = 256;

    private final Plugin plugin;
    private final AtomicLong sequence = new AtomicLong();
    private final GameRegistry gameRegistry = new GameRegistry();
    private final UserStateMachine stateMachine = new UserStateMachine();
    private volatile BotSpawner spawner;
    private volatile GuardBridge guard = new NoopGuardBridge();
    private volatile boolean ready;
    private volatile boolean degradeWarned;

    private final Map<String, Bot> bots = new ConcurrentHashMap<String, Bot>();
    /** Bot 就绪时间：无会话时用于估算存活时长，避免 analyze 恒报 0。 */
    private final Map<String, Long> botReadyAt = new ConcurrentHashMap<String, Long>();
    private final Map<String, Deque<Information>> pendingNotices =
            new ConcurrentHashMap<String, Deque<Information>>();
    private final Map<String, SessionCounters> counters =
            new ConcurrentHashMap<String, SessionCounters>();

    public GladiatorServiceImpl(Plugin plugin) {
        this.plugin = Objects.requireNonNull(plugin, "plugin");
    }

    // ------------------------------------------------------------------
    // 生命周期（由 GladiatorMain 调用）
    // ------------------------------------------------------------------

    /** 注册完成后置为就绪；G-101 建立 WS 后这里会同时要求链路已监听。 */
    public void markReady() {
        ready = true;
    }

    /** onDisable 时回收，防止热重载后残留旧状态。 */
    public void shutdown() {
        ready = false;
        spawner = null;
        stateMachine.clear();
        gameRegistry.clear();
        bots.clear();
        pendingNotices.clear();
        counters.clear();
    }

    /** G-107：由 GladiatorMain 在链路起来后接上 spawn 服务。 */
    public void attachSpawner(BotSpawner spawner) {
        this.spawner = Objects.requireNonNull(spawner, "spawner");
    }

    public BotSpawner getSpawner() {
        return spawner;
    }

    /** G-203：安装规则桥接；未安装时为 allow 降级实现。 */
    public void setGuardBridge(GuardBridge bridge) {
        this.guard = Objects.requireNonNull(bridge, "bridge");
    }

    public GuardBridge getGuardBridge() {
        return guard;
    }

    public UserStateMachine getStateMachine() {
        return stateMachine;
    }

    /** 已就绪 Bot 的只读查找；未登记返回 null。 */
    public Bot findBot(String botId) {
        return botId == null ? null : bots.get(botId);
    }

    /** 按玩家 UUID 找到代理它的 BotHandle；不是 Bot 返回 null。 */
    public Bot findBotByUniqueId(UUID uuid) {
        if (uuid == null) {
            return null;
        }
        for (Bot bot : bots.values()) {
            if (uuid.equals(bot.getUniqueId())) {
                return bot;
            }
        }
        return null;
    }

    public int getPendingSpawnCount() {
        BotSpawner current = spawner;
        return current == null ? 0 : current.getPendingCount();
    }

    public BotRegistry getBotRegistry() {
        BotSpawner current = spawner;
        return current == null ? null : current.getRegistry();
    }

    public String sessionIdOf(UUID uuid) {
        return gameRegistry.sessionIdOf(uuid);
    }

    public Collection<GameSession> activeGames() {
        return gameRegistry.games();
    }

    public Collection<TrainingSession> activeTrainings() {
        return gameRegistry.trainingSessions();
    }

    // ------------------------------------------------------------------
    // GladiatorAPI
    // ------------------------------------------------------------------

    @Override
    public String getApiVersion() {
        requirePrimaryThread();
        return API_VERSION;
    }

    @Override
    public boolean isReady() {
        requirePrimaryThread();
        return ready && plugin.isEnabled();
    }

    @Override
    public BotRequest spawnBot(BotSpec spec) {
        requirePrimaryThread();
        Objects.requireNonNull(spec, "spec");
        if (!isReady()) {
            throw new IllegalStateException("Gladiator 尚未就绪，不能 spawnBot");
        }
        BotSpawner current = spawner;
        if (current == null) {
            // 未接线（M0 独立测试场景）：绝不返回"假成功"的票据，也不伪造 Bukkit Player。
            throw new IllegalStateException(
                    "spawnBot 尚未接线：需要 Bot 运行时与 WebSocket 通道（G-101 / G-107）");
        }
        return current.spawn(spec);
    }

    @Override
    public GameSession pushGame(Mode mode, User... users) {
        requirePrimaryThread();
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(users, "users");
        if (users.length == 0) {
            throw new IllegalArgumentException("pushGame 至少需要 1 个参与者");
        }

        List<User> participants = new ArrayList<User>(users.length);
        for (User user : users) {
            if (user == null) {
                throw new IllegalArgumentException("users 不能包含 null");
            }
            if (gameRegistry.hasSessionFor(user.getUniqueId())) {
                throw new IllegalStateException("参与者已在活动会话中: " + user.getName());
            }
            participants.add(user);
        }

        String sessionId = "game-" + sequence.incrementAndGet();
        GameSession session = GameSession.of(sessionId, mode, participants);
        gameRegistry.registerGame(session);
        for (User user : participants) {
            ensureCounters(counterKey(user));
            transition(user, UserState.ALIVE);
        }
        plugin.getLogger().info("pushGame " + sessionId
                + " mode=" + mode.getId() + " users=" + participants.size());
        return session;
    }

    @Override
    public TrainingSession pushPlace(Bot bot) {
        requirePrimaryThread();
        Objects.requireNonNull(bot, "bot");
        // G-105：只做状态跟踪与登记，不传送、不发套件、不做对局规则。
        if (!bot.isReady()) {
            throw new IllegalStateException("Bot 尚未就绪（未收到 GladiatorBotReadyEvent）: "
                    + bot.getBotId());
        }
        if (gameRegistry.hasSessionFor(bot.getUniqueId())) {
            throw new IllegalStateException("Bot 已在活动会话中: " + bot.getName());
        }

        String sessionId = "train-" + sequence.incrementAndGet();
        TrainingSession session = TrainingSession.of(sessionId, bot);
        gameRegistry.registerTraining(session);
        bots.put(bot.getBotId(), bot);
        ensureCounters(bot.getBotId());
        transition(bot, UserState.ALIVE);
        return session;
    }

    @Override
    public void notice(Bot bot, Information information) {
        requirePrimaryThread();
        Objects.requireNonNull(bot, "bot");
        Objects.requireNonNull(information, "information");

        String key = bot.getBotId();
        BotSpawner current = spawner;
        if (current != null && current.sendNotice(key, information)) {
            ensureCounters(key).recordAction(false);
            return;
        }

        // 链路未建立时先入队，WS 起来后由通道侧 drain。
        Deque<Information> queue = pendingNotices.get(key);
        if (queue == null) {
            Deque<Information> created = new ArrayDeque<Information>();
            Deque<Information> existing = pendingNotices.putIfAbsent(key, created);
            queue = existing == null ? created : existing;
        }
        boolean dropped = false;
        synchronized (queue) {
            while (queue.size() >= MAX_PENDING_NOTICES) {
                queue.removeFirst();
                dropped = true;
            }
            queue.addLast(information);
        }
        ensureCounters(key).recordAction(dropped);
    }

    @Override
    public Analysis analyze(AnalysisQuery query) {
        requirePrimaryThread();
        Objects.requireNonNull(query, "query");
        // 快照构建集中在 AnalyzeService（G-304）；当前同步执行，M4 可复用做异步采样。
        return new AnalyzeService(gameRegistry, bots, counters, botReadyAt)
                .sample(query, System.currentTimeMillis());
    }

    @Override
    public void eliminate(User user) {
        requirePrimaryThread();
        Objects.requireNonNull(user, "user");
        transition(user, UserState.ELIMINATED, "eliminate");
    }

    @Override
    public UserState stateOf(User user) {
        requirePrimaryThread();
        return user == null ? UserState.ALIVE : stateOfUuid(user.getUniqueId());
    }

    // ------------------------------------------------------------------
    // G-204：PVP 规则查询（只读；不取消任何事件）
    // ------------------------------------------------------------------

    @Override
    public boolean isPvpAllowed(Location location, User user) {
        return queryPvp(location, user).isAllowed();
    }

    @Override
    public PvpVerdict queryPvp(Location location, User user) {
        requirePrimaryThread();
        Objects.requireNonNull(location, "location");
        GuardBridge current = guard;
        if (!current.isAvailable() && !degradeWarned) {
            degradeWarned = true;
            plugin.getLogger().info("未检测到 WorldGuard，PVP 规则查询降级为 allow"
                    + "（source=degraded）；如需真实判定请由调用方安装并初始化 WorldGuard");
        }
        return current.query(location, user);
    }

    // ------------------------------------------------------------------
    // UserStateRegistry.Lookup：不带主线程约束，供 api 包只读查询
    // ------------------------------------------------------------------

    @Override
    public UserState stateOf(UUID uuid) {
        return stateOfUuid(uuid);
    }

    // ------------------------------------------------------------------
    // 内部钩子：供 BotHandle / WS / 事件监听器使用
    // ------------------------------------------------------------------

    /** G-107 完成登录后登记 Bot；不派发事件，事件由登录链在 spawn 完成后派发。 */
    public void registerBot(Bot bot) {
        Objects.requireNonNull(bot, "bot");
        bots.put(bot.getBotId(), bot);
        botReadyAt.put(bot.getBotId(), Long.valueOf(System.currentTimeMillis()));
        stateMachine.reset(bot.getUniqueId(), UserState.ALIVE);
        ensureCounters(bot.getBotId());
    }

    /** Bot 退出时清理并通知调用方。 */
    public void unregisterBot(Bot bot, String reason) {
        Objects.requireNonNull(bot, "bot");
        bots.remove(bot.getBotId());
        botReadyAt.remove(bot.getBotId());
        stateMachine.forget(bot.getUniqueId());
        pendingNotices.remove(bot.getBotId());
        counters.remove(bot.getBotId());
        String sessionId = gameRegistry.sessionIdOf(bot.getUniqueId());
        if (sessionId != null) {
            endSession(sessionId, "bot-removed");
        }
        Bukkit.getPluginManager().callEvent(new GladiatorBotRemovedEvent(bot, reason));
    }

    /** 结束对局/训练会话并派发标准事件。 */
    public boolean endSession(String sessionId, String reason) {
        Objects.requireNonNull(sessionId, "sessionId");
        if (!gameRegistry.remove(sessionId)) {
            return false;
        }
        Bukkit.getPluginManager().callEvent(new GladiatorSessionEndedEvent(sessionId, reason));
        return true;
    }

    /** G-101 建立 WS 后由通道侧取走积压的 notice；无主线程约束。 */
    public List<Information> drainNotices(String botId) {
        Deque<Information> queue = pendingNotices.get(botId);
        if (queue == null) {
            return Collections.emptyList();
        }
        synchronized (queue) {
            if (queue.isEmpty()) {
                return Collections.emptyList();
            }
            List<Information> drained = new ArrayList<Information>(queue);
            queue.clear();
            return drained;
        }
    }

    /** 状态迁移唯一写入点；from==to 时不派发事件。 */
    public void transition(User user, UserState to) {
        transition(user, to, null);
    }

    /**
     * 状态迁移唯一写入点（G-301）。
     *
     * <p>状态先写入 {@link UserStateMachine}，再按需下发 SPAWN 通道的 STATE 帧，
     * 最后派发 Bukkit 事件；from==to 时不派发事件、也不下发帧。
     */
    public void transition(User user, UserState to, String reason) {
        Objects.requireNonNull(user, "user");
        Objects.requireNonNull(to, "to");
        UUID uuid = user.getUniqueId();
        UserState from = stateMachine.stateOf(uuid);
        boolean changed = stateMachine.moveTo(uuid, to);
        if (!changed) {
            return;
        }
        BotSpawner current = spawner;
        if (current != null) {
            current.sendState(user, from, to, reason);
        }
        Bukkit.getPluginManager().callEvent(new GladiatorStateChangedEvent(user, from, to));
    }

    /** PlayerDeathEvent → DEAD（G-301）。 */
    public void handleDeath(Player player) {
        if (player == null) {
            return;
        }
        transition(Users.of(player), UserState.DEAD, "player-death");
    }

    /**
     * PlayerRespawnEvent → RESPAWNING → ALIVE；Mode 不允许重生时直接 ELIMINATED。
     *
     * <p>先写 RESPAWNING 再在下一 tick 写 ALIVE，让观察方能看到重生过程。
     */
    public void handleRespawn(final Player player) {
        if (player == null) {
            return;
        }
        UUID uuid = player.getUniqueId();
        if (stateMachine.stateOf(uuid) == UserState.ELIMINATED) {
            return;
        }
        final User user = Users.of(player);
        Mode mode = modeOf(uuid);
        if (mode != null && !mode.allowsRespawn(user)) {
            transition(user, UserState.ELIMINATED, "mode-denies-respawn");
            return;
        }
        transition(user, UserState.RESPAWNING, "player-respawn");
        Bukkit.getScheduler().runTask(plugin, new Runnable() {
            @Override
            public void run() {
                if (stateMachine.stateOf(player.getUniqueId()) == UserState.RESPAWNING) {
                    transition(Users.of(player), UserState.ALIVE, "respawn-complete");
                }
            }
        });
    }

    /** 伤害统计（G-304）：只累计计数，不影响伤害事件本身。 */
    public void recordDamage(Player victim, Player attacker, double amount) {
        // 虚空/指令等伤害可能是 Float.MAX_VALUE 之类的哨兵值，直接计入会让指标失真。
        if (victim == null || !Double.isFinite(amount) || amount <= 0.0d || amount > 1000.0d) {
            return;
        }
        SessionCounters taken = ensureCounters(counterKeyFor(victim));
        taken.addDamageTaken(amount);
        if (attacker != null) {
            ensureCounters(counterKeyFor(attacker)).addDamageDealt(amount);
        }
    }

    /** 方块统计（G-304）：placed=true 记放置，否则记破坏。 */
    public void recordBlockChange(Player player, boolean placed) {
        if (player == null) {
            return;
        }
        SessionCounters current = ensureCounters(counterKeyFor(player));
        if (placed) {
            current.addBlockPlaced();
        } else {
            current.addBlockBroken();
        }
    }

    /**
     * WorldGuard 的 DisallowedPVPEvent 回调（G-203）。
     *
     * <p>只做两件事：给双方中的 Bot 发 RULE_DENIED 通知、累计规则拒绝计数。
     * <strong>不取消事件、不修改事件</strong>——PVP 是否放行由调用方与区域配置决定。
     */
    public void handlePvpDenied(Player attacker, Player defender, String cause) {
        if (attacker == null || defender == null) {
            return;
        }
        String reason = cause == null || cause.isEmpty() ? "pvp-denied" : cause;
        notifyDenied(attacker, defender, reason);
        notifyDenied(defender, attacker, reason);
    }

    private void notifyDenied(Player botPlayer, Player opponent, String reason) {
        Bot bot = botByPlayer(botPlayer);
        if (bot == null) {
            return;
        }
        Location location = botPlayer.getLocation();
        PvpVerdict verdict = guard.query(location, bot);
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("action", "pvp");
        payload.put("reason", reason);
        payload.put("opponent", opponent == null ? "" : opponent.getName());
        payload.put("regionIds", new ArrayList<String>(verdict.getRegionIds()));
        payload.put("source", verdict.getSource().name().toLowerCase(java.util.Locale.ROOT));
        notice(bot, Information.of(InformationKind.RULE_DENIED, payload));
        ensureCounters(bot.getBotId()).addRuleDenial();
    }

    public int getKnownBotCount() {
        return bots.size();
    }

    public int getActiveGameCount() {
        return gameRegistry.gameCount();
    }

    public int getActiveTrainingCount() {
        return gameRegistry.trainingCount();
    }

    // ------------------------------------------------------------------
    // 私有工具
    // ------------------------------------------------------------------

    private Bot botByPlayer(Player player) {
        if (player == null) {
            return null;
        }
        Bot bot = findBotByUniqueId(player.getUniqueId());
        if (bot != null) {
            return bot;
        }
        BotSpawner current = spawner;
        if (current == null) {
            return null;
        }
        BotRegistry registry = current.getRegistry();
        return registry == null ? null : registry.findByUniqueId(player.getUniqueId());
    }

    private Mode modeOf(UUID uuid) {
        String sessionId = gameRegistry.sessionIdOf(uuid);
        if (sessionId == null) {
            return null;
        }
        for (GameSession session : gameRegistry.games()) {
            if (sessionId.equals(session.getSessionId())) {
                return session.getMode();
            }
        }
        return null;
    }

    private UserState stateOfUuid(UUID uuid) {
        return stateMachine.stateOf(uuid);
    }

    private String counterKeyFor(Player player) {
        Bot bot = botByPlayer(player);
        return bot != null ? bot.getBotId() : player.getUniqueId().toString();
    }

    private SessionCounters ensureCounters(String key) {
        SessionCounters current = counters.get(key);
        if (current == null) {
            SessionCounters created = new SessionCounters();
            SessionCounters existing = counters.putIfAbsent(key, created);
            current = existing == null ? created : existing;
        }
        return current;
    }

    private static String counterKey(User user) {
        return user.isBot() && user instanceof Bot
                ? ((Bot) user).getBotId()
                : user.getUniqueId().toString();
    }

    private static void requirePrimaryThread() {
        if (Bukkit.getServer() != null && !Bukkit.isPrimaryThread()) {
            throw new IllegalStateException(
                    "GladiatorAPI 必须在服务端主线程调用（当前: "
                            + Thread.currentThread().getName() + "）");
        }
    }
}
