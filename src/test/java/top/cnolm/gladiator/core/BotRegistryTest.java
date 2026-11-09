package top.cnolm.gladiator.core;

import org.junit.Test;

import java.nio.charset.Charset;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * G-104：BotRegistry 三向索引与 botId 约定。
 *
 * <p>只覆盖不依赖 Bukkit 的部分（未绑定的 BotHandle 退化为离线 UUID）。
 */
public class BotRegistryTest {

    private static final Charset UTF_8 = Charset.forName("UTF-8");

    @Test
    public void botIdIsDerivedFromLowercaseUsername() {
        assertEquals("glad-glad_bot_1", BotRegistry.botIdFor("Glad_Bot_1"));
        assertEquals("glad-glad_bot_1", BotRegistry.botIdFor("glad_bot_1"));
    }

    @Test
    public void botIdRejectsNull() {
        try {
            BotRegistry.botIdFor(null);
            fail("null username 必须被拒绝");
        } catch (NullPointerException expected) {
            // ok
        }
    }

    @Test
    public void unboundHandleUsesOfflineUuidAndIsNotReady() {
        String username = "Glad_Bot_1";
        BotHandle handle = new BotHandle(BotRegistry.botIdFor(username), username);

        UUID expected = UUID.nameUUIDFromBytes(
                ("OfflinePlayer:" + username).getBytes(UTF_8));
        assertEquals(expected, handle.getUniqueId());
        assertEquals(handle.getOfflineUuid(), handle.getUniqueId());
        assertEquals(username, handle.getName());
        assertEquals("glad-glad_bot_1", handle.getBotId());
        assertTrue(handle.isBot());
        assertFalse(handle.isReady());
        assertNull(handle.asPlayer());
    }

    @Test
    public void registryIndexesByBotIdUuidAndUsername() {
        BotHandle handle = new BotHandle(BotRegistry.botIdFor("Glad_Bot_2"), "Glad_Bot_2");
        BotRegistry registry = new BotRegistry();

        registry.register(handle);

        assertEquals(1, registry.size());
        assertSame(handle, registry.findByBotId("glad-glad_bot_2"));
        assertSame(handle, registry.findByUniqueId(handle.getUniqueId()));
        assertSame(handle, registry.findByUsername("Glad_Bot_2"));
        assertSame(handle, registry.findByUsername("glad_bot_2"));
        assertNull(registry.findByUsername("Glad_Bot_3"));
        assertEquals(1, registry.all().size());
        assertEquals(0, registry.readyBots().size());
    }

    @Test
    public void reRegisterWithSameBotIdKeepsSingleEntry() {
        BotRegistry registry = new BotRegistry();
        BotHandle first = new BotHandle(BotRegistry.botIdFor("Glad_Bot_4"), "Glad_Bot_4");
        BotHandle second = new BotHandle(BotRegistry.botIdFor("Glad_Bot_4"), "Glad_Bot_4");

        registry.register(first);
        registry.register(second);

        assertEquals(1, registry.size());
        assertSame(second, registry.findByBotId("glad-glad_bot_4"));
        assertSame(second, registry.findByUsername("Glad_Bot_4"));
    }

    @Test
    public void unregisterRemovesAllIndexes() {
        BotRegistry registry = new BotRegistry();
        BotHandle handle = new BotHandle(BotRegistry.botIdFor("Glad_Bot_5"), "Glad_Bot_5");
        registry.register(handle);

        registry.unregister(handle);

        assertEquals(0, registry.size());
        assertNull(registry.findByBotId("glad-glad_bot_5"));
        assertNull(registry.findByUniqueId(handle.getUniqueId()));
        assertNull(registry.findByUsername("Glad_Bot_5"));
    }

    @Test
    public void clearDropsEverything() {
        BotRegistry registry = new BotRegistry();
        registry.register(new BotHandle(BotRegistry.botIdFor("Glad_Bot_6"), "Glad_Bot_6"));
        registry.register(new BotHandle(BotRegistry.botIdFor("Glad_Bot_7"), "Glad_Bot_7"));

        registry.clear();

        assertEquals(0, registry.size());
    }
}
