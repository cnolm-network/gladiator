package top.cnolm.gladiator.api;

import java.util.Objects;

/** 无限制训练场会话，由 {@link GladiatorAPI#pushPlace(Bot)} 返回；无对局规则、无硬终止。 */
public final class TrainingSession {

    private final String sessionId;
    private final Bot bot;
    private final long startedAt;

    private TrainingSession(String sessionId, Bot bot, long startedAt) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.bot = Objects.requireNonNull(bot, "bot");
        this.startedAt = startedAt;
    }

    public static TrainingSession of(String sessionId, Bot bot) {
        return new TrainingSession(sessionId, bot, System.currentTimeMillis());
    }

    public String getSessionId() {
        return sessionId;
    }

    public Bot getBot() {
        return bot;
    }

    public long getStartedAt() {
        return startedAt;
    }
}
