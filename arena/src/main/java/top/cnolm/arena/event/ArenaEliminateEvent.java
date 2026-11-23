package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import top.cnolm.arena.EliminateReason;

import java.util.UUID;

/** 淘汰出局：某玩家被判定出局时触发，携带淘汰者（可为 null），不可取消。 */
public class ArenaEliminateEvent extends ArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Player killer;
    private final EliminateReason reason;

    public ArenaEliminateEvent(Player victim, Player killer, UUID matchId, EliminateReason cause) {
        super(victim, matchId);
        this.killer = killer;
        this.reason = cause;
    }

    public Player getVictim() {
        return getPlayer();
    }

    public Player getKiller() {
        return killer;
    }

    public boolean hasKiller() {
        return killer != null;
    }

    public EliminateReason getReason() {
        return reason;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
