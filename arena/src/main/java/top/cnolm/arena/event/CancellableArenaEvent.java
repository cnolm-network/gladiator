package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.Cancellable;

import java.util.UUID;

/**
 * 可取消的玩家事件基类：只维护自身 cancelled 标志，不触碰原始 Bukkit 事件。
 * 取消的回传统一由 {@code ArenaEventHandler} 完成。
 */
public abstract class CancellableArenaEvent extends ArenaEvent implements Cancellable {
    private boolean cancelled;

    public CancellableArenaEvent(Player player, UUID matchId) {
        super(player, matchId);
    }

    @Override
    public boolean isCancelled() {
        return cancelled;
    }

    @Override
    public void setCancelled(boolean cancel) {
        this.cancelled = cancel;
    }
}
