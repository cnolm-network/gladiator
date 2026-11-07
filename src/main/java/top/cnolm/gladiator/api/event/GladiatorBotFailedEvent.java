package top.cnolm.gladiator.api.event;

import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;

import java.util.Objects;

/** Bot 登录失败、超时或被踢时发出。 */
public final class GladiatorBotFailedEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final String requestId;
    private final String botId;
    private final String reason;

    public GladiatorBotFailedEvent(String requestId, String botId, String reason) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.botId = Objects.requireNonNull(botId, "botId");
        this.reason = reason;
    }

    public String getRequestId() {
        return requestId;
    }

    public String getBotId() {
        return botId;
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
