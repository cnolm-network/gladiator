package top.cnolm.gladiator.core;

import org.bukkit.entity.Player;
import org.junit.Test;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.TrainingSession;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

import java.util.Arrays;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * G-105：GameRegistry 只做会话登记，不实现任何对局规则。
 */
public class GameRegistryTest {

    @Test
    public void registeringGameIndexesAllParticipants() {
        GameRegistry registry = new GameRegistry();
        User alice = new FakeUser("alice");
        User bob = new FakeUser("bob");
        GameSession game = GameSession.of("game-1", new FakeMode(), Arrays.asList(alice, bob), 1000L);

        registry.registerGame(game);

        assertEquals(1, registry.gameCount());
        assertEquals(0, registry.trainingCount());
        assertTrue(registry.hasSessionFor(alice.getUniqueId()));
        assertTrue(registry.hasSessionFor(bob.getUniqueId()));
        assertEquals("game-1", registry.sessionIdOf(bob.getUniqueId()));
        assertEquals(Long.valueOf(1000L), registry.startedAt("game-1"));
        assertTrue(registry.isGame("game-1"));
        assertFalse(registry.isTraining("game-1"));
        assertEquals(1, registry.games().size());
    }

    @Test
    public void registeringTrainingIndexesBot() {
        GameRegistry registry = new GameRegistry();
        BotHandle bot = new BotHandle(BotRegistry.botIdFor("Glad_Train_1"), "Glad_Train_1");
        TrainingSession training = TrainingSession.of("train-7", bot);

        registry.registerTraining(training);

        assertEquals(1, registry.trainingCount());
        assertEquals(0, registry.gameCount());
        assertEquals("train-7", registry.sessionIdOf(bot.getUniqueId()));
        assertNotNull(registry.startedAt("train-7"));
        assertTrue(registry.isTraining("train-7"));
        assertFalse(registry.isGame("train-7"));
        assertEquals(1, registry.trainingSessions().size());
    }

    @Test
    public void removeCleansParticipantIndexAndIsIdempotent() {
        GameRegistry registry = new GameRegistry();
        User carol = new FakeUser("carol");
        registry.registerGame(GameSession.of("game-2", new FakeMode(),
                Arrays.asList(carol), 2000L));

        assertTrue(registry.remove("game-2"));
        assertFalse(registry.hasSessionFor(carol.getUniqueId()));
        assertNull(registry.startedAt("game-2"));
        assertEquals(0, registry.gameCount());
        assertFalse("重复 remove 必须返回 false", registry.remove("game-2"));
        assertFalse(registry.remove(null));
    }

    @Test
    public void clearDropsAllSessions() {
        GameRegistry registry = new GameRegistry();
        registry.registerGame(GameSession.of("game-3", new FakeMode(),
                Arrays.asList(new FakeUser("dave")), 1L));
        registry.registerTraining(TrainingSession.of("train-3",
                new BotHandle(BotRegistry.botIdFor("Glad_Train_3"), "Glad_Train_3")));

        registry.clear();

        assertEquals(0, registry.gameCount());
        assertEquals(0, registry.trainingCount());
        assertEquals(0, registry.games().size());
    }

    private static final class FakeMode implements Mode {

        @Override
        public String getId() {
            return "fake";
        }

        @Override
        public boolean allowsRespawn(User user) {
            return false;
        }

        @Override
        public long getTimeLimitMs() {
            return 0L;
        }
    }

    private static final class FakeUser implements User {

        private final UUID uuid = UUID.randomUUID();
        private final String name;

        private FakeUser(String name) {
            this.name = name;
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
            return false;
        }

        @Override
        public Player asPlayer() {
            return null;
        }
    }
}
