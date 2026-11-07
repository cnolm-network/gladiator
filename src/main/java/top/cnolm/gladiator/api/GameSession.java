package top.cnolm.gladiator.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** 一次对局会话的不可变快照，由 {@link GladiatorAPI#pushGame(Mode, User...)} 返回。 */
public final class GameSession {

    private final String sessionId;
    private final Mode mode;
    private final List<User> users;
    private final long startedAt;

    private GameSession(String sessionId, Mode mode, List<User> users, long startedAt) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.mode = Objects.requireNonNull(mode, "mode");
        this.users = Collections.unmodifiableList(new ArrayList<User>(users));
        this.startedAt = startedAt;
    }

    public static GameSession of(String sessionId, Mode mode, List<User> users) {
        return of(sessionId, mode, users, System.currentTimeMillis());
    }

    public static GameSession of(String sessionId, Mode mode, List<User> users, long startedAt) {
        Objects.requireNonNull(users, "users");
        return new GameSession(sessionId, mode, users, startedAt);
    }

    public String getSessionId() {
        return sessionId;
    }

    public Mode getMode() {
        return mode;
    }

    public List<User> getUsers() {
        return users;
    }

    public long getStartedAt() {
        return startedAt;
    }
}
