package top.cnolm.gladiator.core.golem;

import org.bukkit.plugin.Plugin;
import org.junit.Before;
import org.junit.Test;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.core.BotHandle;
import top.cnolm.gladiator.core.BotRegistry;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.golem.api.BotId;
import top.cnolm.golem.api.GolemService;

import java.lang.reflect.Proxy;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * B1：golem 模式 spawner 的纯 Java 部分（不启动服务端）。
 *
 * <p>只覆盖不需要 Bukkit 实例的路径：重复 spawn 校验、botId 映射、注册表与计数接口。
 * Ready/Failed 事件与 Player 视图必须有真实服务器，放在真机验收里做。
 */
public class GolemBodySpawnerTest {

    private GladiatorServiceImpl service;
    private GolemBodySpawner spawner;

    @Before
    public void setUp() {
        Plugin plugin = fakePlugin();
        service = new GladiatorServiceImpl(plugin);
        service.markReady();
        spawner = new GolemBodySpawner(plugin, service, fakeGolem(), 5L);
        service.attachSpawner(spawner);
    }

    @Test
    public void duplicateSpawnIsRejectedWithRuntimeMessage() {
        BotHandle existing = new BotHandle(BotRegistry.botIdFor("Goblin_A"), "Goblin_A");
        spawner.getRegistry().register(existing);

        try {
            spawner.spawn(BotSpec.of("Goblin_A", "default"));
            fail("已就绪的 Bot 重复 spawn 必须被拒绝");
        } catch (IllegalStateException expected) {
            assertTrue("措辞应与运行时模式一致，实际: " + expected.getMessage(),
                    expected.getMessage().startsWith("Bot 已就绪，不能重复 spawn"));
        }
    }

    @Test
    public void botIdUsesLowercasePrefixWhileGolemUsesRawName() {
        String botId = BotRegistry.botIdFor("Goblin_A");
        assertEquals("glad-goblin_a", botId);
        // golem 侧的名字是原样用户名，两者不混用（golem 内部再用 BotId.of）
        assertEquals("Goblin_A", BotId.of("Goblin_A").name());
        assertTrue("gladiator botId 与 golem 名字必须不同", !botId.equals(BotId.of("Goblin_A").name()));
    }

    @Test
    public void registryIsExposedAndPendingStartsAtZero() {
        assertNotNull(spawner.getRegistry());
        assertSame(spawner.getRegistry(), service.getBotRegistry());
        assertEquals(0, spawner.getPendingCount());
        // golem 模式没有通道 B
        assertTrue(!spawner.sendNotice("glad-goblin_a", null));
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

    private static GolemService fakeGolem() {
        return (GolemService) Proxy.newProxyInstance(GolemService.class.getClassLoader(),
                new Class<?>[]{GolemService.class}, (proxy, method, args) -> {
                    String name = method.getName();
                    if ("available".equals(name)) {
                        return Boolean.TRUE;
                    }
                    if ("version".equals(name)) {
                        return "fake-0.1.0";
                    }
                    if ("hashCode".equals(name)) {
                        return Integer.valueOf(2);
                    }
                    if ("equals".equals(name)) {
                        return Boolean.valueOf(proxy == args[0]);
                    }
                    if ("toString".equals(name)) {
                        return "FakeGolem";
                    }
                    return null;
                });
    }
}
