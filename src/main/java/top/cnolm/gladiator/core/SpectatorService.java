package top.cnolm.gladiator.core;

import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;
import org.bukkit.util.Vector;
import top.cnolm.gladiator.api.Bot;

import java.util.Iterator;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 只读观战：把玩家切到 {@link GameMode#SPECTATOR} 并送到 Bot 附近，退出时还原。
 *
 * <p>职责边界：这里只做「视角与游戏模式」的切换，不参与对局、不改 Bot 状态、不判定胜负。
 * 观战者在旁观者模式下无法破坏方块/攻击，因此不会干扰训练；训练环境侧还会把
 * 目标实体限定为参赛 Bot（见 {@code training/gladiator_env}），避免 Bot 追着观战者跑。</p>
 *
 * <p>状态保存在内存里：玩家退出服务器时会立即恢复游戏模式（避免下次登录还在旁观），
 * 插件卸载时对所有在线观战者做同样的还原。</p>
 */
public final class SpectatorService {

    private final Plugin plugin;
    private final Map<UUID, Snapshot> previous = new ConcurrentHashMap<>();
    private volatile boolean enabled;

    public SpectatorService(Plugin plugin, boolean enabled) {
        this.plugin = plugin;
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isSpectating(Player player) {
        return player != null && previous.containsKey(player.getUniqueId());
    }

    public int size() {
        return previous.size();
    }

    /**
     * 进入观战：切旁观者模式并传送到 Bot 上方，视线朝向 Bot。
     *
     * @return 是否成功；Bot 没有在线实体时返回 false（调用方负责提示）
     */
    public boolean enter(Player player, Bot bot) {
        if (!enabled || player == null) {
            return false;
        }
        Player target = bot == null ? null : bot.asPlayer();
        if (target == null || !target.isOnline()) {
            return false;
        }
        previous.putIfAbsent(player.getUniqueId(), new Snapshot(player.getGameMode(), player.getLocation().clone()));
        player.setGameMode(GameMode.SPECTATOR);

        Location destination = target.getLocation().clone().add(0.0D, 3.0D, 0.0D);
        Vector direction = target.getLocation().clone().add(0.0D, 1.4D, 0.0D).toVector()
                .subtract(destination.toVector());
        if (direction.lengthSquared() > 1.0E-6D) {
            destination.setDirection(direction);
        }
        player.teleport(destination);
        return true;
    }

    /** 退出观战：恢复进入前的游戏模式与位置。 */
    public boolean leave(Player player) {
        if (player == null) {
            return false;
        }
        Snapshot snapshot = previous.remove(player.getUniqueId());
        if (snapshot == null) {
            return false;
        }
        player.setGameMode(snapshot.mode);
        if (snapshot.location != null && snapshot.location.getWorld() != null) {
            player.teleport(snapshot.location);
        }
        return true;
    }

    /** 玩家退出服务器：只恢复游戏模式，不做传送（此时传送没有意义）。 */
    public void forget(Player player) {
        if (player == null) {
            return;
        }
        Snapshot snapshot = previous.remove(player.getUniqueId());
        if (snapshot == null) {
            return;
        }
        if (player.getGameMode() == GameMode.SPECTATOR) {
            player.setGameMode(snapshot.mode);
        }
    }

    /** 插件卸载：把所有在线观战者还原，避免残留旁观者模式。 */
    public void shutdown() {
        Iterator<Map.Entry<UUID, Snapshot>> it = previous.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<UUID, Snapshot> entry = it.next();
            it.remove();
            Player player = plugin.getServer().getPlayer(entry.getKey());
            if (player != null && player.isOnline() && player.getGameMode() == GameMode.SPECTATOR) {
                player.setGameMode(entry.getValue().mode);
            }
        }
    }

    /** 玩家退出时恢复游戏模式（不传送）。 */
    public static final class QuitListener implements Listener {

        private final SpectatorService service;

        public QuitListener(SpectatorService service) {
            this.service = service;
        }

        @EventHandler(priority = EventPriority.MONITOR)
        public void onQuit(PlayerQuitEvent event) {
            service.forget(event.getPlayer());
        }
    }

    /** 进入观战前的状态快照。 */
    private static final class Snapshot {
        final GameMode mode;
        final Location location;

        Snapshot(GameMode mode, Location location) {
            this.mode = mode;
            this.location = location;
        }
    }
}
