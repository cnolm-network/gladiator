package top.cnolm.gladiator.core;

import org.bukkit.plugin.Plugin;
import org.junit.Test;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

import java.lang.reflect.Proxy;
import java.util.UUID;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * B1：{@code BotSpawnService} 的对外面抽成 {@link BotSpawner} 后，{@link GladiatorServiceImpl}
 * 必须对任何实现都一视同仁（runtime / golem / 占位）。
 *
 * <p>不启动服务端：用假的 {@link BotSpawner} 返回现成票据，只验证接线与委托。
 */
public class GladiatorServiceImplSpawnerTest {

    @Test
    public void spawnBotDelegatesToAttachedSpawner() {
        GladiatorServiceImpl service = readyService();
        FakeSpawner spawner = new FakeSpawner();
        service.attachSpawner(spawner);

        assertSame(spawner, service.getSpawner());
        BotRequest request = service.spawnBot(BotSpec.of("Goblin_A", "default"));

        assertEquals(BotRequest.State.READY, request.getState());
        assertEquals("glad-goblin_a", request.getBotId());
        assertEquals(1, spawner.spawnCalls);
        assertSame(spawner.registry, service.getBotRegistry());
        assertEquals(1, service.getPendingSpawnCount());
    }

    @Test
    public void unwiredSpawnerStillFailsLoudly() {
        GladiatorServiceImpl service = readyService();
        try {
            service.spawnBot(BotSpec.of("Goblin_A", "default"));
            fail("未接线时必须拒绝，不能返回假成功票据");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("尚未接线"));
        }
    }

    private static GladiatorServiceImpl readyService() {
        GladiatorServiceImpl service = new GladiatorServiceImpl(fakePlugin());
        service.markReady();
        return service;
    }

    private static Plugin fakePlugin() {
        return (Plugin) Proxy.newProxyInstance(Plugin.class.getClassLoader(),
                new Class<?>[]{Plugin.class}, (proxy, method, args) -> {
                    String name = method.getName();
                    if ("isEnabled".equals(name)) {
                        return Boolean.TRUE;
                    }
                    if ("getName".equals(name)) {
                        return "Gladiator";
                    }
                    if ("getLogger".equals(name)) {
                        return Logger.getLogger("GladiatorTest");
                    }
                    if ("hashCode".equals(name)) {
                        return Integer.valueOf(1);
                    }
                    if ("equals".equals(name)) {
                        return Boolean.valueOf(proxy == args[0]);
                    }
                    if ("toString".equals(name)) {
                        return "FakePlugin";
                    }
                    return null;
                });
    }

    private static final class FakeSpawner implements BotSpawner {

        private final BotRegistry registry = new BotRegistry();
        private int spawnCalls;

        @Override
        public BotRequest spawn(BotSpec spec) {
            spawnCalls++;
            FakeBot bot = new FakeBot(BotRegistry.botIdFor(spec.getUsername()), spec.getUsername());
            return BotRequest.pending("req-1", bot.getBotId()).ready(bot);
        }

        @Override
        public String playerUuid(String username) {
            return null;
        }

        @Override
        public int getPendingCount() {
            return 1;
        }

        @Override
        public BotRegistry getRegistry() {
            return registry;
        }

        @Override
        public boolean sendNotice(String botId, Information information) {
            return false;
        }

        @Override
        public void sendState(User user, UserState previous, UserState state, String reason) {
            // no-op
        }
    }

    private static final class FakeBot implements Bot {

        private final String botId;
        private final String name;

        private FakeBot(String botId, String name) {
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
            return UUID.nameUUIDFromBytes(("OfflinePlayer:" + name).getBytes());
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
