package top.cnolm.gladiator.api;

import java.util.Objects;

/** {@link GladiatorAPI#analyze(AnalysisQuery)} 的查询条件。 */
public final class AnalysisQuery {

    private final String sessionId;
    private final long windowSeconds;

    private AnalysisQuery(String sessionId, long windowSeconds) {
        this.sessionId = sessionId;
        this.windowSeconds = Math.max(0L, windowSeconds);
    }

    /** 查询全部当前会话。 */
    public static AnalysisQuery all() {
        return new AnalysisQuery(null, 0L);
    }

    public static AnalysisQuery of(String sessionId) {
        return new AnalysisQuery(Objects.requireNonNull(sessionId, "sessionId"), 0L);
    }

    public static AnalysisQuery of(String sessionId, long windowSeconds) {
        return new AnalysisQuery(Objects.requireNonNull(sessionId, "sessionId"), windowSeconds);
    }

    /** null 表示不限定会话。 */
    public String getSessionId() {
        return sessionId;
    }

    /** 0 表示全量。 */
    public long getWindowSeconds() {
        return windowSeconds;
    }
}
