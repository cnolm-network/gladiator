package top.cnolm.gladiator.api.event;

import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;

import java.util.Objects;

/** 对局会话终止时发出，供调用方释放地图、写战绩。 */
public final class GladiatorSessionEndedEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final String sessionId;
    private final String reason;

    public GladiatorSessionEndedEvent(String sessionId, String reason) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.reason = reason;
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getReason() {
        return reason;
    }

    @Override
    public HandlerList getHandlers() {
        return HANDLERS;
    }

    public static HandlerList getHandlerList() {
        return HANDLERS;
    }
}
