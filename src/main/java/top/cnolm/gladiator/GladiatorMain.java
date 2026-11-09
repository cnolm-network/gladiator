package top.cnolm.gladiator;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import org.bukkit.entity.Player;
import org.bukkit.plugin.ServicePriority;
import org.bukkit.plugin.java.JavaPlugin;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.GladiatorProvider;
import top.cnolm.gladiator.api.UserStateRegistry;
import top.cnolm.gladiator.bridge.Envelope;
import top.cnolm.gladiator.bridge.LinkHandler;
import top.cnolm.gladiator.bridge.LinkServer;
import top.cnolm.gladiator.bridge.SessionRegistry;
import top.cnolm.gladiator.command.GladiatorCommand;
import top.cnolm.gladiator.core.BotSpawnService;
import top.cnolm.gladiator.core.SpectatorService;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.gladiator.core.PlayerStateListener;
import top.cnolm.gladiator.core.guard.GuardBridge;
import top.cnolm.gladiator.core.guard.NoopGuardBridge;
import top.cnolm.gladiator.core.guard.WorldGuardBridge;
import top.cnolm.gladiator.core.guard.WorldGuardPvpListener;

import java.io.IOException;
import java.util.logging.Level;

/**
 * Layer 1 入口：服务端编排插件。
 *
 * <p>职责边界：只做编排与规则感知，不做运动控制，不实现具体 Mode 的对局规则。
 * Bot 由外部 mineflayer 运行时以真实协议登录，服务端侧通过 BotHandle 代理。
 *
 * <p>本类负责对外 API 服务的注册/注销（G-008）与通道 A 链路的生命周期（G-101）。
 * 调用方可用 {@link GladiatorProvider#get()} 或 Bukkit {@code ServicesManager} 取得同一实例。
 */
public final class GladiatorMain extends JavaPlugin {

    private boolean worldGuardPresent;
    private GladiatorServiceImpl service;
    private BotSpawnService spawnService;
    private SpectatorService spectatorService;
    private LinkServer linkServer;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        worldGuardPresent = detectWorldGuard();

        service = new GladiatorServiceImpl(this);
        GuardBridge guardBridge = worldGuardPresent
                ? new WorldGuardBridge(getLogger()) : new NoopGuardBridge();
        service.setGuardBridge(guardBridge);
        long spawnTimeout = getConfig().getLong("bridge.spawn-timeout-seconds", 30L);
        spawnService = new BotSpawnService(this, service, spawnTimeout);
        service.attachSpawner(spawnService);
        getServer().getPluginManager().registerEvents(spawnService, this);
        getServer().getPluginManager().registerEvents(new PlayerStateListener(service), this);
        registerPvpListener();

        // 只读观战：玩家进服后可用 /gladiator spectate 旁观 Bot 训练。
        spectatorService = new SpectatorService(this, getConfig().getBoolean("spectate.enabled", true));
        getServer().getPluginManager().registerEvents(
                new SpectatorService.QuitListener(spectatorService), this);

        getServer().getServicesManager()
                .register(GladiatorAPI.class, service, this, ServicePriority.Normal);
        GladiatorProvider.install(service);
        UserStateRegistry.install(service);

        getCommand("gladiator").setExecutor(new GladiatorCommand(this));
        startLink();
        service.markReady();

        getLogger().info("Gladiator " + getDescription().getVersion() + " enabled"
                + " | api=" + service.getApiVersion()
                + " | ready=" + service.isReady()
                + " | link=" + (linkServer != null && linkServer.isRunning()
                        ? "127.0.0.1:" + linkServer.getPort() : "down")
                + " | server=" + getServer().getVersion()
                + " | worldguard=" + (worldGuardPresent ? "present" : "absent(degraded)"));
        if (worldGuardPresent && !getConfig().getBoolean("worldguard.allow-degraded", true)) {
            getLogger().info("worldguard.allow-degraded=false，规则缺失将直接拒绝动作");
        }
    }

    @Override
    public void onDisable() {
        // 先还原观战者，避免插件卸载后玩家卡在旁观者模式。
        if (spectatorService != null) {
            spectatorService.shutdown();
            spectatorService = null;
        }
        // 再停 spawn：取消超时任务并静默丢弃 pending，避免停服过程中派发 Ready/Failed。
        if (spawnService != null) {
            spawnService.shutdown();
            spawnService = null;
        }
        if (linkServer != null) {
            linkServer.stop();
            linkServer = null;
        }
        if (service != null) {
            service.shutdown();
            getServer().getServicesManager().unregister(GladiatorAPI.class, service);
        }
        UserStateRegistry.uninstall();
        GladiatorProvider.uninstall();
        getLogger().info("Gladiator disabled (链路与 API 服务已注销)");
        service = null;
    }

    /** 从 config.yml 读取链路参数；绑定失败只降级，不阻止插件启用。 */
    private void startLink() {
        String host = getConfig().getString("bridge.host", "127.0.0.1");
        int port = getConfig().getInt("bridge.port", 8765);
        String token = getConfig().getString("bridge.token", "");
        long timeout = getConfig().getLong("bridge.heartbeat-timeout-seconds", 30L);
        LinkServer server = new LinkServer(host, port, token, timeout, new PluginLinkHandler());
        try {
            server.start();
            linkServer = server;
            if (spawnService != null) {
                spawnService.attachLinkServer(server);
            }
        } catch (IllegalArgumentException e) {
            getLogger().log(Level.SEVERE, "链路配置非法（仅允许回环地址）: " + e.getMessage(), e);
        } catch (IOException e) {
            getLogger().log(Level.WARNING, "链路监听失败，插件以降级模式运行: " + e.getMessage(), e);
        }
    }

    /**
     * 被动 PVP 拒绝通知（G-203）。
     *
     * <p>监听 {@code DisallowedPVPEvent} 只用于感知规则，<strong>绝不 setCancelled</strong>：
     * 该事件可能被其他插件用于放行 PVP，取消会改变外部行为。未装 WorldGuard 时静默跳过。
     */
    private void registerPvpListener() {
        if (!worldGuardPresent) {
            return;
        }
        try {
            WorldGuardPvpListener listener = new WorldGuardPvpListener(
                    new WorldGuardPvpListener.Sink() {
                        @Override
                        public void onPvpDenied(Player attacker, Player defender, String cause) {
                            GladiatorServiceImpl current = service;
                            if (current != null) {
                                current.handlePvpDenied(attacker, defender, cause);
                            }
                        }
                    });
            getServer().getPluginManager().registerEvents(listener, this);
            getLogger().info("已注册 DisallowedPVPEvent 监听（只观察，不取消）");
        } catch (Throwable t) {
            getLogger().log(Level.WARNING,
                    "DisallowedPVPEvent 监听注册失败，规则感知降级为主动查询: " + t.getMessage(), t);
        }
    }

    /** WorldGuard 以 softdepend 引入，运行时按类存在性判断，避免硬耦合。 */
    private boolean detectWorldGuard() {        try {
            Class.forName("com.sk89q.worldguard.bukkit.WorldGuardPlugin");
            return getServer().getPluginManager().getPlugin("WorldGuard") != null;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    public boolean isWorldGuardPresent() {
        return worldGuardPresent;
    }

    /** 对外 API 实例；插件未 enable 时为 null。 */
    public GladiatorAPI getApi() {
        return service;
    }

    /** 仅供本插件命令/内部模块使用。 */
    public GladiatorServiceImpl getService() {
        return service;
    }

    /** 通道 A 链路；未成功监听时为 null。 */
    public SpectatorService getSpectatorService() {
        return spectatorService;
    }

    public LinkServer getLinkServer() {
        return linkServer;
    }

    /**
     * 链路回调的默认实现。当前只做能力声明与日志；M1 后续任务在此把消息切到
     * Bukkit 主线程后调用 core（G-104 BotHandle / G-107 spawnBot）。
     */
    private final class PluginLinkHandler extends LinkHandler.Adapter {

        @Override
        public JsonObject serverInfo() {
            JsonObject info = new JsonObject();
            info.addProperty("serverVersion", getServer().getVersion());
            JsonArray features = new JsonArray();
            features.add(new JsonPrimitive("state.machine"));
            features.add(new JsonPrimitive("analyze"));
            if (worldGuardPresent) {
                features.add(new JsonPrimitive("guard.worldguard"));
            }
            info.add("features", features);
            JsonObject limits = new JsonObject();
            limits.addProperty("maxActionsPerSecond",
                    Integer.valueOf(getConfig().getInt("training.max-actions-per-second", 10)));
            limits.addProperty("observationHz",
                    Integer.valueOf(getConfig().getInt("training.observation-hz", 5)));
            limits.addProperty("idleAfterSeconds",
                    Integer.valueOf(getConfig().getInt("training.idle-after-seconds", 15)));
            info.add("limits", limits);
            return info;
        }

        @Override
        public void onHello(LinkServer server, SessionRegistry.Session session, Envelope hello) {
            getLogger().info("[Link] HELLO bot=" + session.getBotId()
                    + " username=" + session.getUsername()
                    + " session=" + session.getSessionId());
            BotSpawnService spawn = spawnService;
            if (spawn != null) {
                spawn.onHello(server, session, hello);
            }
        }

        @Override
        public void onMessage(LinkServer server, SessionRegistry.Session session, Envelope message) {
            // 链路层线程，禁止直接调用 Bukkit API 或 GladiatorAPI（后者要求主线程）。
            getLogger().info("[Link] " + message.type() + " from " + session.getBotId());
            BotSpawnService spawn = spawnService;
            if (spawn != null) {
                spawn.onMessage(server, session, message);
            }
        }

        @Override
        public void onDisconnect(LinkServer server, SessionRegistry.Session session, String reason) {
            getLogger().info("[Link] disconnected bot=" + session.getBotId() + " reason=" + reason);
            BotSpawnService spawn = spawnService;
            if (spawn != null) {
                spawn.onDisconnect(server, session, reason);
            }
        }

        @Override
        public String playerUuid(String username) {
            BotSpawnService spawn = spawnService;
            return spawn == null ? null : spawn.playerUuid(username);
        }

        @Override
        public void onQueryPvp(LinkServer server, SessionRegistry.Session session, Envelope query) {
            BotSpawnService spawn = spawnService;
            if (spawn == null) {
                super.onQueryPvp(server, session, query);
                return;
            }
            spawn.onQueryPvp(server, session, query);
        }

        @Override
        public void log(String message) {
            getLogger().info("[Link] " + message);
        }
    }
}
