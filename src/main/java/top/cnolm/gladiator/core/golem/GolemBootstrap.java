package top.cnolm.gladiator.core.golem;

import org.bukkit.plugin.Plugin;
import top.cnolm.gladiator.core.BotSpawner;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.golem.api.GolemService;

/**
 * golem 模式装配的唯一入口（B1）。
 *
 * <p><strong>为什么要有这一层</strong>：{@code golem-api} 是 {@code compileOnly} 依赖，
 * 服务端没装 Golem 时这些类在运行期根本不存在。如果 {@code GladiatorMain} 直接引用
 * {@link GolemService}，{@code spawn.mode=runtime} 的部署也会因类解析失败而无法启用插件。
 * 因此 {@code GladiatorMain} 只调用本类的静态方法，并整体包在 {@code catch (Throwable)} 里：
 * 只有真正走 golem 分支时才会加载 golem-api，缺失时退化为 {@code UnavailableSpawner}。
 */
public final class GolemBootstrap {

    private GolemBootstrap() {
    }

    /**
     * 尝试装配 golem spawner。
     *
     * @return 已接线的 spawner；未检测到可用的 Golem 服务（未装插件 / NMS 装配失败）返回 {@code null}
     */
    public static BotSpawner install(Plugin plugin, GladiatorServiceImpl service,
                                     long spawnTimeoutSeconds) {
        Object raw = plugin.getServer().getServicesManager().load(GolemService.class);
        if (!(raw instanceof GolemService)) {
            return null;
        }
        GolemService golem = (GolemService) raw;
        if (!golem.available()) {
            plugin.getLogger().warning("检测到 Golem 服务但 available()=false（NMS 未装配成功）");
            return null;
        }
        GolemBodySpawner spawner = new GolemBodySpawner(plugin, service, golem, spawnTimeoutSeconds);
        plugin.getServer().getPluginManager().registerEvents(spawner, plugin);
        spawner.start();
        plugin.getLogger().info("spawn mode=golem · Golem " + golem.version()
                + " | 身体由 golem 在进程内生成，无需 bot-runtime 登录");
        return spawner;
    }

    /** onDisable：取消对账任务并清空注册表；非 golem 来源时静默返回。 */
    public static void shutdown(BotSpawner spawner) {
        if (spawner instanceof GolemBodySpawner) {
            ((GolemBodySpawner) spawner).shutdown();
        }
    }
}
