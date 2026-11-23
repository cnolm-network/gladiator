package top.cnolm.arena.status;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 参与者 Player 解析器：先走 Bukkit PlayerList，未命中再查显式登记的缓存。
 *
 * <p>golem 身体是进程内 NMS 实体、不进 PlayerList，{@code Bukkit.getPlayer(uuid)} 恒为 null；
 * 但它的 {@link Player} 引用本身有效，可由 {@code GladiatorBotReadyEvent} 或原生事件入口登记进来。
 * 本类让 arena 内部统一走一个入口，不依赖具体 Bot 来源。
 */
public final class PlayerResolver {
    private static final Map<UUID, Player> cache = new ConcurrentHashMap<>();

    private PlayerResolver() {
    }

    /** 解析参与者：Bukkit 优先，未命中查缓存；仍拿不到返回 null。 */
    public static Player get(UUID uuid) {
        if (uuid == null) {
            return null;
        }
        Player player = Bukkit.getPlayer(uuid);
        if (player != null) {
            return player;
        }
        Player cached = cache.get(uuid);
        if (cached == null) {
            return null;
        }
        if (cached.isValid()) {
            return cached;
        }
        cache.remove(uuid, cached);
        return null;
    }

    /** 登记播放器引用，供 Bukkit 反查不到时使用。 */
    public static void remember(Player player) {
        if (player != null) {
            cache.put(player.getUniqueId(), player);
        }
    }

    /** 移除登记，Bot 退出或对局结束调用。 */
    public static void forget(UUID uuid) {
        if (uuid != null) {
            cache.remove(uuid);
        }
    }

    public static void forgetAll(Iterable<UUID> uuids) {
        if (uuids == null) {
            return;
        }
        for (UUID uuid : uuids) {
            forget(uuid);
        }
    }
}
