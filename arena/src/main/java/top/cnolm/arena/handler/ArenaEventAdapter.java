package top.cnolm.arena.handler;

import org.bukkit.event.Cancellable;
import org.bukkit.event.Event;
import org.bukkit.plugin.java.JavaPlugin;
import top.cnolm.arena.status.ArenaMatch;
import top.cnolm.arena.status.MatchManager;

import java.util.Objects;
import java.util.UUID;

/**
 * 基础适配器：只收集对局数据并把包装事件转发到 Bukkit 事件总线。
 *
 * <p>Cancellable 在此层只读取标志，不回传给原始事件；回传由 {@link ArenaEventHandler} 负责。
 */
public abstract class ArenaEventAdapter {
    protected final JavaPlugin plugin;

    protected ArenaEventAdapter(JavaPlugin plugin) {
        this.plugin = Objects.requireNonNull(plugin, "plugin");
    }

    /**
     * 玩家当前所在对局 id，不在任何对局时返回 null。
     * golem 可能在 startMatch 之后才 Ready，此时 playerMatch 尚未登记：按参与者 UUID 反查并补登。
     */
    protected UUID matchIdOf(UUID playerId) {
        if (playerId == null) {
            return null;
        }
        ArenaMatch<?, ?> match = MatchManager.getPlayerMatch(playerId);
        if (match != null) {
            return match.getId();
        }
        for (ArenaMatch<?, ?> candidate : MatchManager.getMatches().values()) {
            if (candidate.getPlayerData(playerId) != null) {
                MatchManager.registerPlayer(playerId, candidate);
                return candidate.getId();
            }
        }
        return null;
    }

    /** 玩家是否处于指定对局，空值安全。 */
    protected boolean isInMatch(UUID playerId, UUID matchId) {
        if (playerId == null || matchId == null) {
            return false;
        }
        ArenaMatch<?, ?> match = MatchManager.getPlayerMatch(playerId);
        return match != null && matchId.equals(match.getId());
    }

    /** 转发任意事件，返回该事件本身便于后续读取结果。 */
    protected <E extends Event> E call(E event) {
        plugin.getServer().getPluginManager().callEvent(event);
        return event;
    }

    /** 转发可取消事件并读取取消标志；此层不触碰原始 Bukkit 事件。 */
    protected <E extends Event & Cancellable> boolean callCancellable(E event) {
        call(event);
        return event.isCancelled();
    }
}
