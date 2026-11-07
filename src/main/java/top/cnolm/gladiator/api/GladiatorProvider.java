package top.cnolm.gladiator.api;

import org.bukkit.Bukkit;
import org.bukkit.plugin.RegisteredServiceProvider;

/**
 * 获取 {@link GladiatorAPI} 的静态门面。
 *
 * <pre>{@code
 * GladiatorAPI api = GladiatorProvider.get();
 * if (api == null || !api.isReady()) return;
 * }</pre>
 *
 * <p>未 enable 或插件未安装时返回 null，调用方必须判空。
 */
public final class GladiatorProvider {

    private static volatile GladiatorAPI cached;

    private GladiatorProvider() {
    }

    public static GladiatorAPI get() {
        GladiatorAPI current = cached;
        if (current != null) {
            return current;
        }
        if (Bukkit.getServer() == null) {
            return null;
        }
        RegisteredServiceProvider<GladiatorAPI> registration =
                Bukkit.getServicesManager().getRegistration(GladiatorAPI.class);
        if (registration == null) {
            return null;
        }
        cached = registration.getProvider();
        return cached;
    }

    /** 仅供 Gladiator 插件本体在 onEnable 调用；调用方请勿使用。 */
    public static void install(GladiatorAPI api) {
        cached = api;
    }

    /** 仅供 Gladiator 插件本体在 onDisable 调用；调用方请勿使用。 */
    public static void uninstall() {
        cached = null;
    }
}
