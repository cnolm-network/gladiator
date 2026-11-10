package top.cnolm.gladiator.core;

import org.junit.Test;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.InformationKind;
import top.cnolm.gladiator.api.User;

import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * B1：golem 缺失时的降级实现。这里是「配置要求 golem 但服务不可用」的唯一出口，
 * 报错文案必须让服主一眼看出原因，且不能抛出 NPE。
 */
public class UnavailableSpawnerTest {

    private static final String REASON = "spawn.mode=golem 但未检测到可用的 Golem 插件";

    @Test
    public void spawnFailsWithChineseExplanation() {
        UnavailableSpawner spawner = new UnavailableSpawner(REASON);
        try {
            spawner.spawn(BotSpec.of("Goblin_A", "default"));
            fail("不可用来源必须拒绝 spawn");
        } catch (IllegalStateException expected) {
            assertTrue("应带中文原因，实际: " + expected.getMessage(),
                    expected.getMessage().contains("spawn.mode=golem"));
            assertTrue("应包含 Golem 提示，实际: " + expected.getMessage(),
                    expected.getMessage().contains("Golem"));
        }
    }

    @Test
    public void readOnlySurfaceStaysSafe() {
        UnavailableSpawner spawner = new UnavailableSpawner(null);
        assertEquals(0, spawner.getPendingCount());
        assertNotNull(spawner.getRegistry());
        assertEquals(0, spawner.getRegistry().size());
        assertNull(spawner.playerUuid("Goblin_A"));
        assertFalse(spawner.sendNotice("glad-goblin_a",
                Information.empty(InformationKind.CHAT)));
        // 状态推送是空实现，不得抛异常
        spawner.sendState(new DummyUser(), null, null, "test");
    }

    /** 空实现状态下 sendState 不能因参数为 null 而抛异常。 */
    private static final class DummyUser implements User {
        @Override
        public UUID getUniqueId() {
            return UUID.nameUUIDFromBytes("OfflinePlayer:Dummy".getBytes());
        }

        @Override
        public String getName() {
            return "Dummy";
        }

        @Override
        public top.cnolm.gladiator.api.UserState getState() {
            return top.cnolm.gladiator.api.UserState.ALIVE;
        }

        @Override
        public boolean isBot() {
            return false;
        }

        @Override
        public org.bukkit.entity.Player asPlayer() {
            return null;
        }
    }
}
