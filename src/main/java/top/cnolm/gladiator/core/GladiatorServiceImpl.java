package top.cnolm.gladiator.core;

import org.bukkit.Bukkit;
import org.bukkit.plugin.Plugin;
import top.cnolm.gladiator.api.Analysis;
import top.cnolm.gladiator.api.AnalysisQuery;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.TrainingSession;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.UserStateRegistry;
import top.cnolm.gladiator.api.event.GladiatorBotRemovedEvent;
import top.cnolm.gladiator.api.event.GladiatorSessionEndedEvent;
import top.cnolm.gladiator.api.event.GladiatorStateChangedEvent;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link GladiatorAPI} 的 M0 最小可用实现（G-008）。
 *
 * <p>已经可用的能力：服务注册、{@code isReady()} / {@code getApiVersion()}、
 * 对局与训练会话登记、{@code notice()} 投递队列、{@code analyze()} 快照、
 * {@code eliminate()} / {@code stateOf()} 状态镜像。
 *
 * <p>尚未实现：{@code spawnBot()} 的真实登录（G-107）与 WebSocket 通道（G-101）。
 * 在此之前 {@code spawnBot} 会显式抛 {@link UnsupportedOperationException}，不伪造 Bot。
 *
 * <p>本类同时实现 {@link UserStateRegistry.Lookup}，让 {@code Users.of(Player).getState()}
 * 能读到同一份状态，避免双状态源。所有状态迁移只发生在本类。
 */
public final class GladiatorServiceImpl implements GladiatorAPI, UserStateRegistry.Lookup {

    /** 语义化版本；只增不改。 */
    public static final String API_VERSION = "1.0.0";

    /** 单个 Bot 在 WS 建立前允许积压的 notice 数量，超出丢弃最旧的并计数。 */
    private static final int MAX_PENDING_NOTICES = 256;

    private final Plugin plugin;
    private final AtomicLong sequence = new AtomicLong();
    private volatile boolean ready;

    private final Map<UUID, UserState> states = new ConcurrentHashMap<UUID, UserState>();
    private final Map<String, GameSession> games = new ConcurrentHashMap<String, GameSession>();
    private final Map<String, TrainingSession> training = new ConcurrentHashMap<String, TrainingSession>();
    private final Map<UUID, String> sessionByUser = new ConcurrentHashMap<UUID, String>();
    private final Map<String, Long> sessionStartedAt = new ConcurrentHashMap<String, Long>();
    private final Map<String, Bot> bots = new ConcurrentHashMap<String, Bot>();
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
        states.clear();
        games.clear();
        training.clear();
        sessionByUser.clear();
        sessionStartedAt.clear();
        bots.clear();
        pendingNotices.clear();
        counters.clear();
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
        // 真实登录由 Layer 2 的 mineflayer 运行时完成，属于 G-107。
        // 这里绝不返回"假成功"的票据，也不伪造 Bukkit Player。
        throw new UnsupportedOperationException(
                "spawnBot 尚未实现：需要 Bot 运行时与 WebSocket 通道（G-101 / G-107）");
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
            if (sessionByUser.containsKey(user.getUniqueId())) {
                throw new IllegalStateException("参与者已在活动会话中: " + user.getName());
            }
            participants.add(user);
        }

        String sessionId = "game-" + sequence.incrementAndGet();
        GameSession session = GameSession.of(sessionId, mode, participants);
        games.put(sessionId, session);
        sessionStartedAt.put(sessionId, session.getStartedAt());
        for (User user : participants) {
            sessionByUser.put(user.getUniqueId(), sessionId);
            ensureCounters(counterKey(user));
            transition(user, UserState.ALIVE);
        }
        plugin.getLogger().info("[Gladiator] pushGame " + sessionId
                + " mode=" + mode.getId() + " users=" + participants.size());
        return session;
    }

    @Override
    public TrainingSession pushPlace(Bot bot) {
        requirePrimaryThread();
        Objects.requireNonNull(bot, "bot");
        if (sessionByUser.containsKey(bot.getUniqueId())) {
            throw new IllegalStateException("Bot 已在活动会话中: " + bot.getName());
        }

        String sessionId = "train-" + sequence.incrementAndGet();
        TrainingSession session = TrainingSession.of(sessionId, bot);
        training.put(sessionId, session);
        sessionStartedAt.put(sessionId, session.getStartedAt());
        sessionByUser.put(bot.getUniqueId(), sessionId);
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
        String requested = query.getSessionId();
        long now = System.currentTimeMillis();
        long windowMs = query.getWindowSeconds() > 0L
                ? query.getWindowSeconds() * 1000L
                : Long.MAX_VALUE;

        List<Analysis.BotMetrics> perBot = new ArrayList<Analysis.BotMetrics>();
        for (Map.Entry<String, Bot> entry : bots.entrySet()) {
            String botId = entry.getKey();
            String sessionId = sessionByUser.get(entry.getValue().getUniqueId());
            if (requested != null && !requested.equals(sessionId)) {
                continue;
            }
            Long startedAt = sessionId == null ? null : sessionStartedAt.get(sessionId);
            double aliveSeconds = startedAt == null
                    ? 0.0d
                    : Math.max(0.0d, Math.min(now - startedAt, windowMs)) / 1000.0d;
            perBot.add(ensureCounters(botId).snapshot(botId, aliveSeconds));
        }
        return Analysis.of(requested, perBot);
    }

    @Override
    public void eliminate(User user) {
        requirePrimaryThread();
        Objects.requireNonNull(user, "user");
        transition(user, UserState.ELIMINATED);
    }

    @Override
    public UserState stateOf(User user) {
        requirePrimaryThread();
        return user == null ? UserState.ALIVE : stateOfUuid(user.getUniqueId());
    }

    // ------------------------------------------------------------------
    // UserStateRegistry.Lookup：不带主线程约束，供 api 包只读查询
    // ------------------------------------------------------------------

    @Override
    public UserState stateOf(UUID uuid) {
        return stateOfUuid(uuid);
    }

    // ------------------------------------------------------------------
    // 内部钩子：供 M1 的 BotHandle / WS / 事件监听器使用
    // ------------------------------------------------------------------

    /** G-107 完成登录后登记 Bot；不派发事件，事件由登录链在 spawn 完成后派发。 */
    public void registerBot(Bot bot) {
        Objects.requireNonNull(bot, "bot");
        bots.put(bot.getBotId(), bot);
        states.put(bot.getUniqueId(), UserState.ALIVE);
        ensureCounters(bot.getBotId());
    }

    /** Bot 退出时清理并通知调用方。 */
    public void unregisterBot(Bot bot, String reason) {
        Objects.requireNonNull(bot, "bot");
        bots.remove(bot.getBotId());
        states.remove(bot.getUniqueId());
        pendingNotices.remove(bot.getBotId());
        counters.remove(bot.getBotId());
        String sessionId = sessionByUser.remove(bot.getUniqueId());
        if (sessionId != null) {
            endSession(sessionId, "bot-removed");
        }
        Bukkit.getPluginManager().callEvent(new GladiatorBotRemovedEvent(bot, reason));
    }

    /** 结束对局/训练会话并派发标准事件。 */
    public boolean endSession(String sessionId, String reason) {
        Objects.requireNonNull(sessionId, "sessionId");
        GameSession game = games.remove(sessionId);
        TrainingSession place = game == null ? training.remove(sessionId) : null;
        if (game == null && place == null) {
            return false;
        }
        sessionStartedAt.remove(sessionId);
        if (game != null) {
            for (User user : game.getUsers()) {
                sessionByUser.remove(user.getUniqueId(), sessionId);
            }
        } else {
            sessionByUser.remove(place.getBot().getUniqueId(), sessionId);
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
        Objects.requireNonNull(user, "user");
        Objects.requireNonNull(to, "to");
        UUID uuid = user.getUniqueId();
        UserState from = states.containsKey(uuid) ? states.get(uuid) : UserState.ALIVE;
        states.put(uuid, to);
        if (from != to) {
            Bukkit.getPluginManager().callEvent(new GladiatorStateChangedEvent(user, from, to));
        }
    }

    public int getKnownBotCount() {
        return bots.size();
    }

    public int getActiveGameCount() {
        return games.size();
    }

    public int getActiveTrainingCount() {
        return training.size();
    }

    // ------------------------------------------------------------------
    // 私有工具
    // ------------------------------------------------------------------

    private UserState stateOfUuid(UUID uuid) {
        if (uuid == null) {
            return UserState.ALIVE;
        }
        UserState state = states.get(uuid);
        return state == null ? UserState.ALIVE : state;
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
