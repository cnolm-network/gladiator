package top.cnolm.gladiator;

import org.bukkit.plugin.ServicePriority;
import org.bukkit.plugin.java.JavaPlugin;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.GladiatorProvider;
import top.cnolm.gladiator.api.UserStateRegistry;
import top.cnolm.gladiator.command.GladiatorCommand;
import top.cnolm.gladiator.core.GladiatorServiceImpl;

/**
 * Layer 1 入口：服务端编排插件。
 *
 * <p>职责边界：只做编排与规则感知，不做运动控制，不实现具体 Mode 的对局规则。
 * Bot 由外部 mineflayer 运行时以真实协议登录，服务端侧通过 BotHandle 代理。
 *
 * <p>本类负责对外 API 服务的注册/注销（G-008）：调用方可用
 * {@link GladiatorProvider#get()} 或 Bukkit {@code ServicesManager} 取得同一实例。
 */
public final class GladiatorMain extends JavaPlugin {

    private boolean worldGuardPresent;
    private GladiatorServiceImpl service;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        worldGuardPresent = detectWorldGuard();

        service = new GladiatorServiceImpl(this);
        getServer().getServicesManager()
                .register(GladiatorAPI.class, service, this, ServicePriority.Normal);
        GladiatorProvider.install(service);
        UserStateRegistry.install(service);

        getCommand("gladiator").setExecutor(new GladiatorCommand(this));
        service.markReady();

        getLogger().info("Gladiator " + getDescription().getVersion() + " enabled"
                + " | api=" + service.getApiVersion()
                + " | ready=" + service.isReady()
                + " | server=" + getServer().getVersion()
                + " | worldguard=" + (worldGuardPresent ? "present" : "absent(degraded)"));
        if (worldGuardPresent && !getConfig().getBoolean("worldguard.allow-degraded", true)) {
            getLogger().info("worldguard.allow-degraded=false，规则缺失将直接拒绝动作");
        }
    }

    @Override
    public void onDisable() {
        if (service != null) {
            service.shutdown();
            getServer().getServicesManager().unregister(GladiatorAPI.class, service);
        }
        UserStateRegistry.uninstall();
        GladiatorProvider.uninstall();
        getLogger().info("Gladiator disabled (API 服务已注销)");
        service = null;
    }

    /** WorldGuard 以 softdepend 引入，运行时按类存在性判断，避免硬耦合。 */
    private boolean detectWorldGuard() {
        try {
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
}
