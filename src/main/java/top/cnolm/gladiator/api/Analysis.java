package top.cnolm.gladiator.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** {@link GladiatorAPI#analyze(AnalysisQuery)} 返回的不可变指标快照。 */
public final class Analysis {

    /** 单个 Bot 的聚合指标。 */
    public static final class BotMetrics {

        private final String botId;
        private final double aliveSeconds;
        private final double damageDealt;
        private final double damageTaken;
        private final int blocksPlaced;
        private final int blocksBroken;
        private final double pathMeters;
        private final int ruleDenials;
        private final int actionsAccepted;
        private final int actionsDropped;

        public BotMetrics(String botId, double aliveSeconds, double damageDealt,
                          double damageTaken, int blocksPlaced, int blocksBroken,
                          double pathMeters, int ruleDenials,
                          int actionsAccepted, int actionsDropped) {
            this.botId = Objects.requireNonNull(botId, "botId");
            this.aliveSeconds = aliveSeconds;
            this.damageDealt = damageDealt;
            this.damageTaken = damageTaken;
            this.blocksPlaced = blocksPlaced;
            this.blocksBroken = blocksBroken;
            this.pathMeters = pathMeters;
            this.ruleDenials = ruleDenials;
            this.actionsAccepted = actionsAccepted;
            this.actionsDropped = actionsDropped;
        }

        public String getBotId() { return botId; }
        public double getAliveSeconds() { return aliveSeconds; }
        public double getDamageDealt() { return damageDealt; }
        public double getDamageTaken() { return damageTaken; }
        public int getBlocksPlaced() { return blocksPlaced; }
        public int getBlocksBroken() { return blocksBroken; }
        public double getPathMeters() { return pathMeters; }
        public int getRuleDenials() { return ruleDenials; }
        public int getActionsAccepted() { return actionsAccepted; }
        public int getActionsDropped() { return actionsDropped; }
    }

    private final String sessionId;
    private final long generatedAt;
    private final List<BotMetrics> perBot;

    private Analysis(String sessionId, long generatedAt, List<BotMetrics> perBot) {
        this.sessionId = sessionId;
        this.generatedAt = generatedAt;
        this.perBot = Collections.unmodifiableList(new ArrayList<BotMetrics>(perBot));
    }

    public static Analysis of(String sessionId, List<BotMetrics> perBot) {
        Objects.requireNonNull(perBot, "perBot");
        return new Analysis(sessionId, System.currentTimeMillis(), perBot);
    }

    /** null 表示跨会话聚合。 */
    public String getSessionId() {
        return sessionId;
    }

    public long getGeneratedAt() {
        return generatedAt;
    }

    public List<BotMetrics> getPerBot() {
        return perBot;
    }
}
