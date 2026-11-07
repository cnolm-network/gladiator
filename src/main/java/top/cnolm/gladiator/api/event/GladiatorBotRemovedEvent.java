package top.cnolm.gladiator.api.event;

import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;
import top.cnolm.gladiator.api.Bot;

import java.util.Objects;

/** Bot 退出或被回收时发出，供调用方清理自身状态。 */
public final class GladiatorBotRemovedEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final Bot bot;
    private final String reason;

    public GladiatorBotRemovedEvent(Bot bot, String reason) {
        this.bot = Objects.requireNonNull(bot, "bot");
        this.reason = reason;
    }

    public Bot getBot() {
        return bot;
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
