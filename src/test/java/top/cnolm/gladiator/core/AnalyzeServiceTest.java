package top.cnolm.gladiator.core;

import org.junit.Test;
import top.cnolm.gladiator.api.Analysis;
import top.cnolm.gladiator.api.AnalysisQuery;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * G-304 快照语义：窗口裁剪、会话过滤、无会话时退化为 Bot 就绪时长。
 */
public class AnalyzeServiceTest {

    private static final long NOW = 1_700_000_000_000L;

    @Test
    public void aliveSecondsComeFromSessionStart() {
        FakeBot bot = new FakeBot("bot-1", "Glad_Bot_1");
        Map<String, Bot> bots = new HashMap<String, Bot>();
        bots.put("bot-1", bot);
        GameRegistry registry = new GameRegistry();
        FakeMode mode = new FakeMode();
        registry.registerGame(GameSession.of("game-1", mode,
                Collections.<User>singletonList(bot)));

        SessionCounters counters = new SessionCounters();
        counters.addBlockPlaced();
        counters.addRuleDenial();
        Map<String, SessionCounters> counterMap = new HashMap<String, SessionCounters>();
        counterMap.put("bot-1", counters);

        AnalyzeService service = new AnalyzeService(registry, bots, counterMap,
                new HashMap<String, Long>());
        Analysis analysis = service.sample(AnalysisQuery.all(), NOW);

        assertEquals(1, analysis.getPerBot().size());
        Analysis.BotMetrics metrics = analysis.getPerBot().get(0);
        assertEquals("bot-1", metrics.getBotId());
        assertEquals(1, metrics.getBlocksPlaced());
        assertEquals(1, metrics.getRuleDenials());
        assertTrue("aliveSeconds 应为正数", metrics.getAliveSeconds() >= 0.0d);
    }

    @Test
    public void windowClampsAliveSeconds() {
        FakeBot bot = new FakeBot("bot-1", "Glad_Bot_1");
        Map<String, Bot> bots = Collections.<String, Bot>singletonMap("bot-1", bot);
        GameRegistry registry = new GameRegistry();
        registry.registerGame(GameSession.of("game-1", new FakeMode(),
                Collections.<User>singletonList(bot), NOW - 60_000L));

        AnalyzeService service = new AnalyzeService(registry, bots,
                new HashMap<String, SessionCounters>(), new HashMap<String, Long>());
        Analysis analysis = service.sample(AnalysisQuery.of("game-1", 10L), NOW);

        assertEquals(10.0d, analysis.getPerBot().get(0).getAliveSeconds(), 0.001d);
    }

    @Test
    public void sessionFilterExcludesOtherSessions() {
        FakeBot bot = new FakeBot("bot-1", "Glad_Bot_1");
        Map<String, Bot> bots = Collections.<String, Bot>singletonMap("bot-1", bot);
        AnalyzeService service = new AnalyzeService(new GameRegistry(), bots,
                new HashMap<String, SessionCounters>(), new HashMap<String, Long>());

        Analysis analysis = service.sample(AnalysisQuery.of("game-42"), NOW);

        assertTrue(analysis.getPerBot().isEmpty());
    }

    private static final class FakeMode implements Mode {
        @Override
        public String getId() {
            return "fake";
        }

        @Override
        public boolean allowsRespawn(User user) {
            return true;
        }

        @Override
        public long getTimeLimitMs() {
            return 0L;
        }
    }

    private static final class FakeBot implements Bot {
        private final String botId;
        private final String name;
        private final UUID uuid = UUID.randomUUID();

        FakeBot(String botId, String name) {
            this.botId = botId;
            this.name = name;
        }

        @Override
        public String getBotId() {
            return botId;
        }

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public UUID getUniqueId() {
            return uuid;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public UserState getState() {
            return UserState.ALIVE;
        }

        @Override
        public boolean isBot() {
            return true;
        }

        @Override
        public org.bukkit.entity.Player asPlayer() {
            return null;
        }
    }
}
