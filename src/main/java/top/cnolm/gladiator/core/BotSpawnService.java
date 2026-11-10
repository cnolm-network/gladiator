package top.cnolm.gladiator.core;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.gladiator.api.BotProfile;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.event.GladiatorBotFailedEvent;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;
import top.cnolm.gladiator.bridge.Envelope;
import top.cnolm.gladiator.bridge.LinkServer;
import top.cnolm.gladiator.bridge.SessionRegistry;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@code spawnBot} 的异步编排（G-107）。
 *
 * <p>登录由外部 mineflayer 运行时完成，本类只负责三段状态机：
 * <ol>
 *   <li>调用方 {@code spawnBot(spec)} → 生成 pending 请求并返回 {@link BotRequest}（PENDING）；</li>
 *   <li>运行时 WebSocket HELLO 到达 → 下发 {@code SPAWN}（携带请求与服务器地址）；</li>
 *   <li>同名真实玩家 join → 绑定 {@link BotHandle}、注册、在主线程派发
 *       {@link GladiatorBotReadyEvent}；超时或运行时上报失败 → {@link GladiatorBotFailedEvent}。</li>
 * </ol>
 *
 * <p>链路回调运行在 WebSocket 线程，所有 Bukkit 操作都通过 {@link #runOnMain(Runnable)} 切回主线程。
 * 已绑定成功的 handle 不会因为链路断开而被回收：只要 mineflayer 进程还活着，玩家实体就仍在服务器里；
 * 进程真正死亡时 Minecraft 连接会断，{@link PlayerQuitEvent} 负责最终清理。
 */
public final class BotSpawnService implements BotSpawner, Listener {

    private static final Gson GSON = new Gson();

    private final Plugin plugin;
    private final GladiatorServiceImpl service;
    private final BotRegistry registry = new BotRegistry();
    private final long spawnTimeoutSeconds;

    private final Map<String, PendingSpawn> byRequestId =
            new ConcurrentHashMap<String, PendingSpawn>();
    private final Map<String, PendingSpawn> byUsername =
            new ConcurrentHashMap<String, PendingSpawn>();
    private final Map<String, SessionRegistry.Session> sessionByUsername =
            new ConcurrentHashMap<String, SessionRegistry.Session>();
    private final AtomicLong requestSeq = new AtomicLong();
    private final AtomicLong frameSeq = new AtomicLong();

    private volatile LinkServer linkServer;
    private volatile boolean shuttingDown;

    public BotSpawnService(Plugin plugin, GladiatorServiceImpl service, long spawnTimeoutSeconds) {
        this.plugin = Objects.requireNonNull(plugin, "plugin");
        this.service = Objects.requireNonNull(service, "service");
        this.spawnTimeoutSeconds = Math.max(1L, spawnTimeoutSeconds);
    }

    /** 链路监听成功后才可下发 SPAWN；失败时 spawnBot 仍能拿票据但会等待超时。 */
    public void attachLinkServer(LinkServer linkServer) {
        this.linkServer = linkServer;
    }

    @Override
    public BotRegistry getRegistry() {
        return registry;
    }

    @Override
    public int getPendingCount() {
        return byRequestId.size();
    }

    // ------------------------------------------------------------------
    // 调用方入口（主线程）
    // ------------------------------------------------------------------

    @Override
    public BotRequest spawn(BotSpec spec) {
        Objects.requireNonNull(spec, "spec");
        String username = spec.getUsername();
        String usernameKey = key(username);

        BotHandle existing = registry.findByUsername(username);
        if (existing != null) {
            throw new IllegalStateException("Bot 已就绪，不能重复 spawn: " + username);
        }
        if (byUsername.containsKey(usernameKey)) {
            throw new IllegalStateException("Bot 正在发起中: " + username);
        }

        String requestId = "req-" + requestSeq.incrementAndGet();
        String botId = BotRegistry.botIdFor(username);
        final PendingSpawn pending = new PendingSpawn(requestId, botId, spec);
        byRequestId.put(requestId, pending);
        byUsername.put(usernameKey, pending);

        long timeoutTicks = spawnTimeoutSeconds * 20L;
        pending.timeoutTask = Bukkit.getScheduler().runTaskLater(plugin, new Runnable() {
            @Override
            public void run() {
                fail(pending.requestId, "TIMEOUT: 等待 Bot 登录超过 "
                        + spawnTimeoutSeconds + "s");
            }
        }, timeoutTicks);

        SessionRegistry.Session session = sessionByUsername.get(usernameKey);
        if (session != null) {
            attach(pending, session);
        }
        plugin.getLogger().info("spawnBot " + requestId + " bot=" + botId
                + " user=" + username
                + (session != null ? " (runtime 已连接，已下发 SPAWN)" : " (等待 runtime HELLO)"));
        return BotRequest.pending(requestId, botId);
    }

    // ------------------------------------------------------------------
    // 链路回调（WebSocket 线程）
    // ------------------------------------------------------------------

    public void onHello(final LinkServer server, final SessionRegistry.Session session,
                        final Envelope hello) {
        final String username = session.getUsername();
        runOnMain(new Runnable() {
            @Override
            public void run() {
                String usernameKey = key(username);
                SessionRegistry.Session previous = sessionByUsername.put(usernameKey, session);
                if (previous != null && previous != session) {
                    plugin.getLogger().info("Bot 会话重连: " + username
                            + " old=" + previous.getSessionId() + " new=" + session.getSessionId());
                }
                PendingSpawn pending = byUsername.get(usernameKey);
                if (pending != null) {
                    attach(pending, session);
                } else if (registry.findByUsername(username) != null) {
                    // 链路重连（Bot 实体还在）：补发 BOT_BOUND，让运行时拿到新的 sessionId。
                    // 这里不能再派发 Ready——调用方可能已经在上一轮 pushGame 了。
                    rebind(registry.findByUsername(username), session);
                } else {
                    plugin.getLogger().info("未通过 spawnBot 发起的 Bot 会话（可由调用方稍后 spawnBot）: "
                            + session.getBotId() + " username=" + username
                            + "（未派发 Ready 事件）");
                }
            }
        });
    }

    public void onMessage(final LinkServer server, final SessionRegistry.Session session,
                          final Envelope message) {
        if (!"SPAWN_FAILED".equals(message.type())) {
            return;
        }
        final String requestId = message.payloadString("requestId", "");
        String code = message.payloadString("code", "INTERNAL");
        String detail = message.payloadString("message", "");
        final String reason = detail.length() == 0 ? code : code + ": " + detail;
        runOnMain(new Runnable() {
            @Override
            public void run() {
                fail(requestId, reason);
            }
        });
    }

    public void onDisconnect(final LinkServer server, final SessionRegistry.Session session,
                             final String reason) {
        final String username = session.getUsername();
        runOnMain(new Runnable() {
            @Override
            public void run() {
                String usernameKey = key(username);
                sessionByUsername.remove(usernameKey, session);
                if ("replaced-by-reconnect".equals(reason)) {
                    return;
                }
                PendingSpawn pending = byUsername.get(usernameKey);
                if (pending != null && pending.session == session) {
                    fail(pending.requestId, "LINK_LOST: " + reason);
                }
                if (registry.findByUsername(username) != null) {
                    plugin.getLogger().info("Bot 链路断开但不回收实体（等待重连或退出）: "
                            + username + " reason=" + reason);
                }
            }
        });
    }

    /**
     * WELCOME 的 {@code playerUuid}（G-103 收尾）：已绑定则返回真实 UUID，
     * 已登录但尚未绑定则按用户名解析 Bukkit Player，未知返回 {@code null}（链路层写空串）。
     */
    @Override
    public String playerUuid(String username) {
        BotHandle handle = registry.findByUsername(username);
        if (handle != null) {
            return handle.getUniqueId().toString();
        }
        Player player = Bukkit.getServer() == null ? null : Bukkit.getServer().getPlayerExact(username);
        return player == null ? null : player.getUniqueId().toString();
    }

    // ------------------------------------------------------------------
    // 标准 Bukkit 事件（主线程）
    // ------------------------------------------------------------------

    @EventHandler(priority = EventPriority.MONITOR)
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        String usernameKey = key(player.getName());
        PendingSpawn pending = byUsername.get(usernameKey);
        if (pending == null) {
            return;
        }
        SessionRegistry.Session session = pending.session != null
                ? pending.session : sessionByUsername.get(usernameKey);
        if (session == null) {
            // 运行时还没连上链路：等 HELLO 时 attach() 再补绑。
            plugin.getLogger().info("Bot 玩家先于链路登录，等待 HELLO: "
                    + player.getName());
            return;
        }
        bind(pending, session, player);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onPlayerQuit(PlayerQuitEvent event) {
        Player player = event.getPlayer();
        BotHandle handle = registry.findByUniqueId(player.getUniqueId());
        if (handle == null) {
            return;
        }
        registry.unregister(handle);
        service.unregisterBot(handle, "player-quit");
        handle.unbind();
        plugin.getLogger().info("Bot 退出: " + player.getName());
    }

    // ------------------------------------------------------------------
    // 生命周期
    // ------------------------------------------------------------------

    /** onDisable：取消超时任务、静默丢弃 pending，不派发任何事件。 */
    public void shutdown() {
        shuttingDown = true;
        for (PendingSpawn pending : byRequestId.values()) {
            cancel(pending.timeoutTask);
            pending.done = true;
        }
        byRequestId.clear();
        byUsername.clear();
        sessionByUsername.clear();
        registry.clear();
        linkServer = null;
    }

    // ------------------------------------------------------------------
    // 内部
    // ------------------------------------------------------------------

    private void attach(final PendingSpawn pending, final SessionRegistry.Session session) {
        if (pending.done) {
            return;
        }
        pending.session = session;
        sendSpawn(pending, session);

        Player player = Bukkit.getPlayerExact(pending.username);
        if (player != null && player.isOnline()) {
            bind(pending, session, player);
        }
    }

    private void bind(PendingSpawn pending, SessionRegistry.Session session, Player player) {
        if (pending.done) {
            return;
        }
        pending.done = true;
        cancel(pending.timeoutTask);
        byRequestId.remove(pending.requestId, pending);
        byUsername.remove(key(pending.username), pending);

        BotHandle handle = new BotHandle(pending.botId, player.getName());
        handle.bind(player);
        registry.register(handle);
        service.registerBot(handle);

        JsonObject payload = new JsonObject();
        payload.addProperty("sessionId", session == null ? "" : session.getSessionId());
        payload.addProperty("playerUuid", player.getUniqueId().toString());
        payload.addProperty("username", player.getName());
        payload.addProperty("state", "Alive");
        payload.addProperty("ready", Boolean.TRUE);
        if (session != null) {
            send(session, "BOT_BOUND", payload, null);
        }

        plugin.getLogger().info("Bot ready: " + handle.getBotId()
                + " user=" + player.getName() + " uuid=" + player.getUniqueId());
        // 主线程 + MONITOR 阶段：调用方可以在这里安全传送/给套件，然后 pushGame。
        Bukkit.getPluginManager().callEvent(new GladiatorBotReadyEvent(handle));
    }

    /** 链路重连后用新会话补发 BOT_BOUND（不重复派发 Ready）。 */
    private void rebind(BotHandle handle, SessionRegistry.Session session) {
        if (handle == null || session == null) {
            return;
        }
        JsonObject payload = new JsonObject();
        payload.addProperty("sessionId", session.getSessionId());
        payload.addProperty("playerUuid", handle.getUniqueId().toString());
        payload.addProperty("username", handle.getName());
        payload.addProperty("state", protocolState(service.stateOf(handle)));
        payload.addProperty("ready", Boolean.valueOf(handle.isReady()));
        send(session, "BOT_BOUND", payload, null);
        plugin.getLogger().info("链路重连，已重新绑定 Bot: " + handle.getBotId()
                + " session=" + session.getSessionId());
    }

    private void fail(String requestId, String reason) {        if (requestId == null || requestId.length() == 0) {
            return;
        }
        PendingSpawn pending = byRequestId.remove(requestId);
        if (pending == null || pending.done) {
            return;
        }
        pending.done = true;
        cancel(pending.timeoutTask);
        byUsername.remove(key(pending.username), pending);
        plugin.getLogger().warning("Bot 登录失败: " + pending.botId
                + " request=" + requestId + " reason=" + reason);
        Bukkit.getPluginManager().callEvent(
                new GladiatorBotFailedEvent(pending.requestId, pending.botId, reason));
    }

    /**
     * 直接向该 Bot 的会话推送 NOTICE（G-203）。
     *
     * @return false 表示链路/会话不可用，调用方应回落到本地队列
     */
    @Override
    public boolean sendNotice(String botId, Information information) {
        SessionRegistry.Session session = sessionFor(botId);
        if (session == null || information == null) {
            return false;
        }
        JsonObject payload = new JsonObject();
        payload.addProperty("kind", information.getKind().name());
        payload.addProperty("at", Long.valueOf(information.getTimestamp()));
        payload.add("data", GSON.toJsonTree(information.getPayload()));
        send(session, "NOTICE", payload, null);
        return true;
    }

    /** 向该 Bot 的会话推送 STATE（协议 §2.4）；Bot 不在或未连链路时静默跳过。 */
    @Override
    public void sendState(User user, UserState previous, UserState state, String reason) {
        if (user == null || state == null) {
            return;
        }
        BotHandle handle = registry.findByUniqueId(user.getUniqueId());
        if (handle == null) {
            return;
        }
        SessionRegistry.Session session = sessionFor(handle.getBotId());
        if (session == null) {
            return;
        }
        JsonObject payload = new JsonObject();
        payload.addProperty("state", protocolState(state));
        payload.addProperty("previous", protocolState(previous));
        String sessionId = service.sessionIdOf(user.getUniqueId());
        payload.addProperty("gameId", sessionId == null ? "" : sessionId);
        if (reason != null) {
            payload.addProperty("reason", reason);
        }
        payload.addProperty("at", Long.valueOf(System.currentTimeMillis()));
        send(session, "STATE", payload, null);
    }

    /**
     * 主动规则查询（G-204）：链路线程收到 QUERY_PVP 后切主线程判定并回 QUERY_RESULT。
     */
    public void onQueryPvp(final LinkServer server, final SessionRegistry.Session session,
                           final Envelope query) {
        runOnMain(new Runnable() {
            @Override
            public void run() {
                JsonObject payload = new JsonObject();
                try {
                    BotHandle handle = registry.findByBotId(session.getBotId());
                    org.bukkit.Location location = parseLocation(query, handle);
                    if (location == null) {
                        payload.addProperty("allowed", Boolean.TRUE);
                        payload.add("regionIds", new com.google.gson.JsonArray());
                        payload.addProperty("source", "unknown");
                    } else {
                        top.cnolm.gladiator.api.PvpVerdict verdict =
                                service.queryPvp(location, handle);
                        payload.addProperty("allowed", Boolean.valueOf(verdict.isAllowed()));
                        com.google.gson.JsonArray ids = new com.google.gson.JsonArray();
                        for (String id : verdict.getRegionIds()) {
                            ids.add(new com.google.gson.JsonPrimitive(id));
                        }
                        payload.add("regionIds", ids);
                        payload.addProperty("source",
                                verdict.getSource().name().toLowerCase(Locale.ROOT));
                        if (verdict.getBox() != null) {
                            JsonObject box = new JsonObject();
                            box.addProperty("minX", Double.valueOf(verdict.getBox().getMinX()));
                            box.addProperty("minY", Double.valueOf(verdict.getBox().getMinY()));
                            box.addProperty("minZ", Double.valueOf(verdict.getBox().getMinZ()));
                            box.addProperty("maxX", Double.valueOf(verdict.getBox().getMaxX()));
                            box.addProperty("maxY", Double.valueOf(verdict.getBox().getMaxY()));
                            box.addProperty("maxZ", Double.valueOf(verdict.getBox().getMaxZ()));
                            payload.add("box", box);
                        }
                    }
                } catch (RuntimeException e) {
                    plugin.getLogger().warning("QUERY_PVP 处理失败: " + e);
                    payload.addProperty("allowed", Boolean.TRUE);
                    payload.add("regionIds", new com.google.gson.JsonArray());
                    payload.addProperty("source", "unknown");
                }
                send(session, "QUERY_RESULT", payload, Long.valueOf(query.seq()));
            }
        });
    }

    /** 把 BotProfile 编成 SPAWN.profile（G-507）；运行时未识别时回落到自己的默认值。 */
    private static JsonObject profileJson(BotProfile profile) {
        JsonObject json = new JsonObject();
        json.addProperty("cps", Double.valueOf(profile.getCps()));
        json.addProperty("cpsJitter", Double.valueOf(profile.getCpsJitter()));
        json.addProperty("reactionMs", Double.valueOf(profile.getReactionMs()));
        json.addProperty("reactionJitterMs", Double.valueOf(profile.getReactionJitterMs()));
        json.addProperty("reach", Double.valueOf(profile.getReach()));
        json.addProperty("model", profile.getModel());
        JsonObject assist = new JsonObject();
        assist.addProperty("autoRod", profile.isAutoRod());
        assist.addProperty("autoBlock", profile.isAutoBlock());
        assist.addProperty("wtap", profile.isWtap());
        assist.addProperty("aimLock", profile.isAimLock());
        json.add("assist", assist);
        return json;
    }

    private org.bukkit.Location parseLocation(Envelope query, BotHandle handle) {
        JsonObject payload = query.payload();
        Player player = handle == null ? null : handle.asPlayer();
        if (payload == null || !payload.has("location") || !payload.get("location").isJsonObject()) {
            return player == null ? null : player.getLocation();
        }
        JsonObject location = payload.getAsJsonObject("location");
        String worldName = location.has("world") ? location.get("world").getAsString() : null;
        org.bukkit.World world = worldName == null ? null : Bukkit.getWorld(worldName);
        if (world == null && player != null) {
            world = player.getWorld();
        }
        if (world == null) {
            return null;
        }
        double x = location.has("x") ? location.get("x").getAsDouble() : 0.0d;
        double y = location.has("y") ? location.get("y").getAsDouble() : 0.0d;
        double z = location.has("z") ? location.get("z").getAsDouble() : 0.0d;
        return new org.bukkit.Location(world, x, y, z);
    }

    /** 按 Bot 找到其当前链路会话；Bot 未连接返回 null。 */
    public SessionRegistry.Session sessionFor(String botId) {        LinkServer link = linkServer;
        if (link == null || botId == null) {
            return null;
        }
        for (SessionRegistry.Session session : link.getSessions().all()) {
            if (botId.equals(session.getBotId())) {
                return session;
            }
        }
        return null;
    }

    private static String protocolState(UserState state) {
        if (state == null) {
            return "Alive";
        }
        switch (state) {
            case DEAD:
                return "Dead";
            case RESPAWNING:
                return "Respawning";
            case ELIMINATED:
                return "Eliminated";
            case ALIVE:
            default:
                return "Alive";
        }
    }

    private void sendSpawn(PendingSpawn pending, SessionRegistry.Session session) {        JsonObject payload = new JsonObject();
        payload.addProperty("requestId", pending.requestId);
        payload.addProperty("botId", pending.botId);
        payload.addProperty("username", pending.username);
        payload.addProperty("group", pending.spec.getGroup());
        JsonObject server = new JsonObject();
        server.addProperty("host", serverHost());
        server.addProperty("port", Integer.valueOf(Bukkit.getPort()));
        payload.add("server", server);
        JsonObject metadata = new JsonObject();
        for (Map.Entry<String, String> entry : pending.spec.getMetadata().entrySet()) {
            metadata.addProperty(entry.getKey(), entry.getValue());
        }
        payload.add("metadata", metadata);
        payload.add("profile", profileJson(pending.spec.getProfile()));
        send(session, "SPAWN", payload, null);
        plugin.getLogger().info("→ SPAWN " + pending.botId + " @ "
                + serverHost() + ":" + Bukkit.getPort());
    }

    private void send(SessionRegistry.Session session, String type, JsonObject payload,
                      Long replyTo) {
        LinkServer link = linkServer;
        if (link == null) {
            plugin.getLogger().warning("链路未就绪，丢弃出站帧 " + type);
            return;
        }
        link.send(session, type, payload, replyTo);
    }

    private static String serverHost() {
        String ip = Bukkit.getIp();
        return ip == null || ip.length() == 0 ? "127.0.0.1" : ip;
    }

    private void runOnMain(Runnable task) {
        if (shuttingDown) {
            return;
        }
        if (Bukkit.isPrimaryThread()) {
            task.run();
            return;
        }
        if (!plugin.isEnabled()) {
            return;
        }
        Bukkit.getScheduler().runTask(plugin, task);
    }

    private static void cancel(BukkitTask task) {
        if (task != null) {
            task.cancel();
        }
    }

    private static String key(String username) {
        return username.toLowerCase(Locale.ROOT);
    }

    /** 一次待完成的 spawn；字段只在主线程改写。 */
    private static final class PendingSpawn {

        private final String requestId;
        private final String botId;
        private final String username;
        private final BotSpec spec;

        private volatile SessionRegistry.Session session;
        private volatile boolean done;
        private volatile BukkitTask timeoutTask;

        private PendingSpawn(String requestId, String botId, BotSpec spec) {
            this.requestId = requestId;
            this.botId = botId;
            this.username = spec.getUsername();
            this.spec = spec;
        }
    }
}
