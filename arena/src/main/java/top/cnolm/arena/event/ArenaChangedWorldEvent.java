package top.cnolm.arena.event;

import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 改变世界：玩家切换世界后触发，记录来源与目标世界，不可取消。 */
public class ArenaChangedWorldEvent extends ArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final World from;
    private final World to;

    public ArenaChangedWorldEvent(Player player, UUID matchId, World from, World to) {
        super(player, matchId);
        this.from = from;
        this.to = to;
    }

    public World getFrom() {
        return from;
    }

    public World getTo() {
        return to;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
