package top.cnolm.gladiator.api.event;

import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

import java.util.Objects;

/** {@link UserState} 发生迁移时发出，供调用方做结算与播报。 */
public final class GladiatorStateChangedEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final User user;
    private final UserState from;
    private final UserState to;

    public GladiatorStateChangedEvent(User user, UserState from, UserState to) {
        this.user = Objects.requireNonNull(user, "user");
        this.from = Objects.requireNonNull(from, "from");
        this.to = Objects.requireNonNull(to, "to");
    }

    public User getUser() {
        return user;
    }

    public UserState getFrom() {
        return from;
    }

    public UserState getTo() {
        return to;
    }

    @Override
    public HandlerList getHandlers() {
        return HANDLERS;
    }

    public static HandlerList getHandlerList() {
        return HANDLERS;
    }
}
