package top.cnolm.gladiator.core;

import top.cnolm.gladiator.api.Analysis;
import top.cnolm.gladiator.api.AnalysisQuery;
import top.cnolm.gladiator.api.Bot;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 指标快照服务（G-304）。
 *
 * <p>把「存活时长 + 计数累加器」组合成不可变的 {@link Analysis}。计数器本身用
 * {@link SessionCounters} 的内部锁保护，因此采样可以在任意线程执行；当前由
 * {@code GladiatorAPI.analyze()} 在主线程同步调用，M4 的训练循环可以复用同一入口
 * 做异步采样，不必改动数据结构。
 *
 * <p>本类不写状态、不发事件：只读。
 */
public final class AnalyzeService {

    private final GameRegistry registry;
    private final Map<String, Bot> bots;
    private final Map<String, SessionCounters> counters;
    private final Map<String, Long> botReadyAt;

    public AnalyzeService(GameRegistry registry,
                          Map<String, Bot> bots,
                          Map<String, SessionCounters> counters,
                          Map<String, Long> botReadyAt) {
        this.registry = registry;
        this.bots = bots;
        this.counters = counters;
        this.botReadyAt = botReadyAt;
    }

    /**
     * 生成快照。
     *
     * @param query 过滤条件；{@code sessionId} 为 null 表示跨会话全量
     * @param now   采样时刻（毫秒），便于测试注入
     */
    public Analysis sample(AnalysisQuery query, long now) {
        String requested = query.getSessionId();
        long windowMs = query.getWindowSeconds() > 0L
                ? query.getWindowSeconds() * 1000L
                : Long.MAX_VALUE;

        List<Analysis.BotMetrics> perBot = new ArrayList<Analysis.BotMetrics>();
        for (Map.Entry<String, Bot> entry : bots.entrySet()) {
            String botId = entry.getKey();
            String sessionId = registry.sessionIdOf(entry.getValue().getUniqueId());
            if (requested != null && !requested.equals(sessionId)) {
                continue;
            }
            Long startedAt = sessionId == null ? null : registry.startedAt(sessionId);
            if (startedAt == null) {
                startedAt = botReadyAt.get(botId);
            }
            double aliveSeconds = startedAt == null
                    ? 0.0d
                    : Math.max(0.0d, Math.min(now - startedAt.longValue(), windowMs)) / 1000.0d;
            SessionCounters current = counters.get(botId);
            if (current == null) {
                current = new SessionCounters();
            }
            perBot.add(current.snapshot(botId, aliveSeconds));
        }
        return Analysis.of(requested, perBot);
    }
}
