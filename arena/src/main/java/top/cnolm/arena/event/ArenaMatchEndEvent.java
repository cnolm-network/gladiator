package top.cnolm.arena.event;

import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 对局结束：一局分出胜负或超时后触发，不可取消。 */
public class ArenaMatchEndEvent extends ArenaMatchEvent {
    private static final HandlerList handlers = new HandlerList();
    private final int round;
    private final String reason;
    private final Object outcome;

    public ArenaMatchEndEvent(UUID matchId, int round, String reason, Object outcome) {
        super(matchId);
        this.round = round;
        this.reason = reason;
        this.outcome = outcome;
    }

    public int getRound() {
        return round;
    }

    public String getReason() {
        return reason;
    }

    /** 原始结算结果，由具体板子决定类型，避免 event 依赖板子实现。 */
    public Object getOutcome() {
        return outcome;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
