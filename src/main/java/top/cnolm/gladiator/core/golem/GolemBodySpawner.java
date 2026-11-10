package top.cnolm.gladiator.core.golem;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
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
import top.cnolm.gladiator.core.BotHandle;
import top.cnolm.gladiator.core.BotRegistry;
import top.cnolm.gladiator.core.BotSpawner;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.golem.api.BotBody;
import top.cnolm.golem.api.BotId;
import top.cnolm.golem.api.BotSpawnSpec;
import top.cnolm.golem.api.DespawnReason;
import top.cnolm.golem.api.GolemService;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;

/**
 * golem 模式（{@code spawn.mode=golem}，B1）：{@code spawnBot} 直接换成 Golem 插件在服务端进程内
 * 生成的 NMS 身体，不登录、不依赖 bot-runtime、不占用协议连接。
 *
 * <p>与 {@code BotSpawnService} 的差异只在「Bot 从哪来」：
 * <ul>
 *   <li>没有等待登录的握手：{@code golem.spawn(spec)} 同步返回 {@link BotBody}，
 *       同一 tick 即可拿到 {@link BotBody#bukkitPlayer()}；个别实现晚一 tick 就绪，
 *       因此按 tick 重试到 {@code spawnTimeoutSeconds} 为止。</li>
 *   <li>没有通道 B：{@link #sendNotice} 返回 false（NOTICE 落到本地队列），{@link #sendState} 空实现，
 *       状态变化仍由 {@link GladiatorServiceImpl} 派发标准 Bukkit 事件。</li>
 *   <li>名字映射仍是两条线：golem 的 {@link BotId} 用用户名本身，gladiator 的 botId 用
 *       {@link BotRegistry#botIdFor(String)}（{@code glad-<小写用户名>}），两者不混用。</li>
 * </ul>
 *
 * <p>线程：{@link #spawn} 要求主线程；链路回调不存在，因此本类只由命令/调用方在主线程驱动。
 * 所有 Bukkit 事件（Ready/Failed/Removed）都在主线程派发。
 */
public final class GolemBodySpawner implements BotSpawner, Listener {

    /** 与 {@link BotSpec} 一致的用户名规则（3–16 位字母/数字/下划线）。 */
    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9_]{3,16}");

    /** 对账周期（tick）：清理被 golem 侧移除的身体，避免 gladiator 侧残留注册。 */
    private static final long RECONCILE_PERIOD_TICKS = 40L;

    private final Plugin plugin;
    private final GladiatorServiceImpl service;
    private final GolemService golem;
    private final long spawnTimeoutSeconds;
    private final BotRegistry registry = new BotRegistry();

    private final Map<String, PendingGolemSpawn> byRequestId =
            new ConcurrentHashMap<String, PendingGolemSpawn>();
    private final Map<String, PendingGolemSpawn> byUsername =
            new ConcurrentHashMap<String, PendingGolemSpawn>();
    private final AtomicLong requestSeq = new AtomicLong();
    private final AtomicInteger pendingCount = new AtomicInteger();

    private volatile BukkitTask reconcileTask;
    private volatile boolean shuttingDown;

    public GolemBodySpawner(Plugin plugin, GladiatorServiceImpl service, GolemService golem,
                            long spawnTimeoutSeconds) {
        this.plugin = Objects.requireNonNull(plugin, "plugin");
        this.service = Objects.requireNonNull(service, "service");
        this.golem = Objects.requireNonNull(golem, "golem");
        this.spawnTimeoutSeconds = Math.max(1L, spawnTimeoutSeconds);
    }

    /** 启动身体对账（由 {@link GolemBootstrap} 在主线程调用）。 */
    public void start() {
        if (reconcileTask != null) {
            return;
        }
        reconcileTask = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
            @Override
            public void run() {
                reconcile();
            }
        }, RECONCILE_PERIOD_TICKS, RECONCILE_PERIOD_TICKS);
    }

    // ------------------------------------------------------------------
    // BotSpawner
    // ------------------------------------------------------------------

    @Override
    public BotRequest spawn(BotSpec spec) {
        Objects.requireNonNull(spec, "spec");
        requirePrimaryThread();
        if (!service.isReady()) {
            throw new IllegalStateException("Gladiator 尚未就绪，不能 spawnBot");
        }

        String username = spec.getUsername();
        if (!NAME.matcher(username).matches()) {
            throw new IllegalArgumentException(
                    "非法 Bot 用户名（3-16 位字母/数字/下划线）: " + username);
        }
        if (registry.findByUsername(username) != null) {
            throw new IllegalStateException("Bot 已就绪，不能重复 spawn: " + username);
        }
        if (byUsername.containsKey(key(username))) {
            throw new IllegalStateException("Bot 正在发起中: " + username);
        }

        final String requestId = "req-" + requestSeq.incrementAndGet();
        final String botId = BotRegistry.botIdFor(username);
        final PendingGolemSpawn pending = new PendingGolemSpawn(requestId, botId, username,
                System.currentTimeMillis() + spawnTimeoutSeconds * 1000L);
        byRequestId.put(requestId, pending);
        byUsername.put(key(username), pending);
        pendingCount.incrementAndGet();

        BotBody body = null;
        try {
            body = golem.spawn(golemSpec(spec));
            if (body == null) {
                throw new IllegalStateException("Golem 未返回 BotBody（golem.spawn 返回 null）");
            }
            pending.body = body;
            Player player = body.bukkitPlayer();
            if (player != null) {
                return complete(pending, player);
            }
            // 同一 tick 拿不到 Player 视图：按 tick 重试到超时（多数实现下一 tick 即可就绪）。
            pending.waitTask = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
                @Override
                public void run() {
                    awaitPlayer(pending);
                }
            }, 1L, 1L);
            plugin.getLogger().info("spawnBot " + requestId + " bot=" + botId + " user=" + username
                    + " (golem body 已创建，等待 Player 视图)");
            return BotRequest.pending(requestId, botId);
        } catch (RuntimeException e) {
            return fail(pending, "GOLEM_SPAWN_ERROR: " + message(e), e, body);
        }
    }

    /** 没有通道 B：NOTICE 由 {@link GladiatorServiceImpl} 落到本地队列。 */
    @Override
    public boolean sendNotice(String botId, Information information) {
        return false;
    }

    /** 没有通道 B：状态事件照常走 Bukkit 事件。 */
    @Override
    public void sendState(User user, UserState previous, UserState state, String reason) {
        // no-op
    }

    @Override
    public String playerUuid(String username) {
        BotHandle handle = registry.findByUsername(username);
        if (handle != null) {
            return handle.getUniqueId().toString();
        }
        Player player = Bukkit.getServer() == null ? null : Bukkit.getServer().getPlayerExact(username);
        return player == null ? null : player.getUniqueId().toString();
    }

    @Override
    public int getPendingCount() {
        return pendingCount.get();
    }

    @Override
    public BotRegistry getRegistry() {
        return registry;
    }

    // ------------------------------------------------------------------
    // 生命周期
    // ------------------------------------------------------------------

    /** onDisable：取消计时任务与对账任务，静默丢弃 pending（不派发任何事件）。 */
    public void shutdown() {
        shuttingDown = true;
        BukkitTask reconcile = reconcileTask;
        if (reconcile != null) {
            reconcile.cancel();
            reconcileTask = null;
        }
        for (PendingGolemSpawn pending : byRequestId.values()) {
            cancel(pending.waitTask);
            pending.done = true;
        }
        byRequestId.clear();
        byUsername.clear();
        pendingCount.set(0);
        registry.clear();
    }

    // ------------------------------------------------------------------
    // 标准 Bukkit 事件
    // ------------------------------------------------------------------

    /** 身体玩家退出（若 golem 走了原版登出路径）：回收注册，避免残留就绪 Bot。 */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onPlayerQuit(PlayerQuitEvent event) {
        Player player = event.getPlayer();
        BotHandle handle = registry.findByUniqueId(player.getUniqueId());
        if (handle == null) {
            return;
        }
        removeHandle(handle, "player-quit");
    }

    // ------------------------------------------------------------------
    // 内部
    // ------------------------------------------------------------------

    /**
     * 组 golem 的生成参数。
     *
     * <p>位置优先级：{@code spawn.golem.location}（{@code x,y,z}）> 世界出生点；
     * 世界取 {@code spawn.golem.world}，留空则取第一个世界。
     */
    private BotSpawnSpec golemSpec(BotSpec spec) {
        BotProfile profile = spec.getProfile();
        return BotSpawnSpec.builder(BotId.of(spec.getUsername()), spawnLocation())
                .reach(profile.getReach())
                .cps((int) Math.round(profile.getCps()))
                .build();
    }

    private Location spawnLocation() {
        String worldName = plugin.getConfig().getString("spawn.golem.world", "");
        World world = null;
        if (worldName != null && !worldName.trim().isEmpty()) {
            world = Bukkit.getWorld(worldName.trim());
            if (world == null) {
                plugin.getLogger().warning("spawn.golem.world 不存在，回落到第一个世界: " + worldName);
            }
        }
        if (world == null) {
            if (Bukkit.getWorlds().isEmpty()) {
                throw new IllegalStateException("服务端没有可用世界，无法为 golem 提供生成位置");
            }
            world = Bukkit.getWorlds().get(0);
        }

        String raw = plugin.getConfig().getString("spawn.golem.location", "");
        if (raw == null || raw.trim().isEmpty()) {
            return world.getSpawnLocation();
        }
        String[] parts = raw.split(",");
        if (parts.length != 3) {
            throw new IllegalStateException(
                    "spawn.golem.location 格式应为 x,y,z，实际: " + raw);
        }
        double x;
        double y;
        double z;
        try {
            x = Double.parseDouble(parts[0].trim());
            y = Double.parseDouble(parts[1].trim());
            z = Double.parseDouble(parts[2].trim());
        } catch (NumberFormatException e) {
            throw new IllegalStateException("spawn.golem.location 含非数字: " + raw);
        }
        return new Location(world, x, y, z);
    }

    /** 每个 tick 检查一次身体是否已有 Bukkit Player 视图。 */
    private void awaitPlayer(PendingGolemSpawn pending) {
        if (shuttingDown || pending.done) {
            cancel(pending.waitTask);
            return;
        }
        try {
            Player player = pending.body == null ? null : pending.body.bukkitPlayer();
            if (player != null) {
                complete(pending, player);
                return;
            }
            if (System.currentTimeMillis() >= pending.deadline) {
                cancel(pending.waitTask);
                fail(pending, "GOLEM_TIMEOUT: 等待 Golem 身体就绪超过 "
                        + spawnTimeoutSeconds + "s", null, pending.body);
            }
        } catch (RuntimeException e) {
            cancel(pending.waitTask);
            fail(pending, "GOLEM_SPAWN_ERROR: " + message(e), e, pending.body);
        }
    }

    /** 成功路径：绑定 handle → 注册表 → service → 主线程派发 Ready。 */
    private BotRequest complete(PendingGolemSpawn pending, Player player) {
        finish(pending);
        BotHandle handle = new BotHandle(pending.botId, player.getName());
        handle.bind(player);
        registry.register(handle);
        service.registerBot(handle);
        plugin.getLogger().info("Bot ready: " + handle.getBotId()
                + " user=" + player.getName() + " uuid=" + player.getUniqueId()
                + " source=golem");
        Bukkit.getPluginManager().callEvent(new GladiatorBotReadyEvent(handle));
        return BotRequest.pending(pending.requestId, pending.botId).ready(handle);
    }

    /**
     * 失败路径：清理 golem 侧残留身体、派发 {@link GladiatorBotFailedEvent}，并把失败票据交回调用方。
     *
     * <p>异步（等待超时）时调用方已拿到 PENDING 票据，结果只能经事件传递，返回值仅作日志来源。
     */
    private BotRequest fail(PendingGolemSpawn pending, String reason, RuntimeException error,
                            BotBody body) {
        finish(pending);
        if (body != null) {
            try {
                golem.despawn(body.id(), DespawnReason.REQUESTED);
            } catch (RuntimeException e) {
                plugin.getLogger().warning("清理 golem 残留身体失败: " + body.id() + " (" + e + ")");
            }
        }
        if (error == null) {
            plugin.getLogger().warning("Bot 生成失败: " + pending.botId
                    + " request=" + pending.requestId + " reason=" + reason);
        } else {
            plugin.getLogger().log(java.util.logging.Level.WARNING,
                    "Bot 生成失败: " + pending.botId + " request=" + pending.requestId
                            + " reason=" + reason, error);
        }
        Bukkit.getPluginManager().callEvent(
                new GladiatorBotFailedEvent(pending.requestId, pending.botId, reason));
        return BotRequest.pending(pending.requestId, pending.botId).failed(reason);
    }

    /** 移除 pending：取消计时、清索引、减计数（幂等）。 */
    private void finish(PendingGolemSpawn pending) {
        if (pending.done) {
            return;
        }
        pending.done = true;
        cancel(pending.waitTask);
        if (byRequestId.remove(pending.requestId, pending)) {
            pendingCount.decrementAndGet();
        }
        byUsername.remove(key(pending.username), pending);
    }

    /**
     * 对账：golem 去掉身体时不会发 Bukkit 事件，因此定期核对两边注册表。
     *
     * <p>只处理「身体彻底不存在」的情况；身体死亡但仍在 golem 注册表里（{@code DeathPolicy.KEEP}）
     * 交给状态机走 DEAD / RESPAWNING，不在这里回收。
     */
    private void reconcile() {
        if (shuttingDown) {
            return;
        }
        for (BotHandle handle : registry.all()) {
            try {
                BotBody body = golem.get(BotId.of(handle.getUsername()));
                if (body == null || body.bukkitPlayer() == null) {
                    removeHandle(handle, "golem-body-gone");
                }
            } catch (RuntimeException e) {
                plugin.getLogger().warning("golem 对账失败: " + handle.getBotId() + " (" + e + ")");
            }
        }
    }

    private void removeHandle(BotHandle handle, String reason) {
        registry.unregister(handle);
        service.unregisterBot(handle, reason);
        handle.unbind();
        plugin.getLogger().info("Bot 退出: " + handle.getName() + " reason=" + reason);
    }

    private static void requirePrimaryThread() {
        if (Bukkit.getServer() != null && !Bukkit.isPrimaryThread()) {
            throw new IllegalStateException("spawnBot 必须在服务端主线程调用（当前: "
                    + Thread.currentThread().getName() + "）");
        }
    }

    private static String message(RuntimeException e) {
        String text = e.getMessage();
        return text == null || text.isEmpty() ? e.getClass().getSimpleName() : text;
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
    private static final class PendingGolemSpawn {

        private final String requestId;
        private final String botId;
        private final String username;
        private final long deadline;

        private volatile BotBody body;
        private volatile BukkitTask waitTask;
        private volatile boolean done;

        private PendingGolemSpawn(String requestId, String botId, String username, long deadline) {
            this.requestId = requestId;
            this.botId = botId;
            this.username = username;
            this.deadline = deadline;
        }
    }
}
