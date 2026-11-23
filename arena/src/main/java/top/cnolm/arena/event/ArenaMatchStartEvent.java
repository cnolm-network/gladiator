package top.cnolm.arena.event;

import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 对局开始：由对局板子在开始新一局时触发，不可取消。 */
public class ArenaMatchStartEvent extends ArenaMatchEvent {
    private static final HandlerList handlers = new HandlerList();
    private final int round;

    public ArenaMatchStartEvent(UUID matchId, int round) {
        super(matchId);
        this.round = round;
    }

    public int getRound() {
        return round;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
