package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import top.cnolm.arena.DeathReason;

import java.util.UUID;

/**
 * 玩家击杀事件：由死亡与伤害归因合并而来，不可取消。
 * 非击杀死亡时 killer 为 null，事件主体始终是 victim。
 */
public class ArenaKillEvent extends ArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Player killer;
    private final DeathReason cause;

    public ArenaKillEvent(Player victim, Player killer, UUID matchId, DeathReason cause) {
        super(victim, matchId);
        this.killer = killer;
        this.cause = cause;
    }

    public boolean hasKiller() {
        return killer != null;
    }

    public Player getKiller() {
        return killer;
    }

    public Player getVictim() {
        return getPlayer();
    }

    public DeathReason getCause() {
        return cause;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
