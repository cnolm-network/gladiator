package top.cnolm.gladiator.core;

import org.bukkit.Location;
import org.bukkit.plugin.Plugin;
import org.junit.Test;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.core.guard.GuardBridge;
import top.cnolm.gladiator.core.guard.ObservedDenials;

import java.lang.reflect.Proxy;
import java.util.Collections;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * A3：观测式 deny 反馈在服务层的接线。
 *
 * <p>不启动服务端：{@code Bukkit.getServer()} 为 null 时 {@code requirePrimaryThread} 直接放行，
 * 通知走 pendingNotices 队列（不碰 Bukkit 调度）。
 */
public class GladiatorServiceImplDenialTest {

    @Test
    public void handleActionDeniedRecordsObservedEntry() {
        GladiatorServiceImpl service = readyService();
        Bot bot = fakeBot("glad-goblin_a", "Goblin_A");
        service.registerBot(bot);
        Location at = new Location(null, 10D, 64D, -5D);

        service.handleActionDenied(bot, "attack", at, "damage-cancelled");

        ObservedDenials.Entry hit = service.getObservedDenials().match(
                "glad-goblin_a", 10D, 64D, -5D, System.currentTimeMillis());
        assertNotNull(hit);
        assertEquals("attack", hit.getAction());
        assertEquals(8.0D, hit.getRadius(), 1e-9); // 默认 guard.observed-radius
        assertTrue(hit.getExpiresAt() > System.currentTimeMillis());
    }

    @Test
    public void handleActionDeniedIgnoresNullBot() {
        GladiatorServiceImpl service = readyService();
        service.handleActionDenied(null, "attack", new Location(null, 0, 0, 0), "x");
        assertEquals(0, service.getObservedDenials().totalSize());
    }

    @Test
    public void payloadHasFixedFields() {
        Location at = new Location(null, 10D, 64D, -5.5D);
        Map<String, Object> payload = GladiatorServiceImpl.deniedPayload(
                "attack", "damage-cancelled", at, 1_700_000_060_000L);

        assertEquals("attack", payload.get("action"));
        assertEquals("damage-cancelled", payload.get("reason"));
        assertEquals(Collections.emptyList(), payload.get("regionIds"));
        assertEquals("observed", payload.get("source"));
        assertEquals(Long.valueOf(1_700_000_060_000L), payload.get("expiresAt"));
        assertEquals("?,10.0,64.0,-5.5", payload.get("location"));
        assertEquals(6, payload.size());
    }

    @Test
    public void payloadWithoutLocationIsEmptyString() {
        Map<String, Object> payload = GladiatorServiceImpl.deniedPayload("use", "interact-cancelled", null, 1L);
        assertEquals("", payload.get("location"));
    }

    @Test
    public void queryPvpReturnsObservedDenyInsideRadius() {
        GladiatorServiceImpl service = readyService();
        service.setGuardBridge(new AllowAllBridge());
        Bot bot = fakeBot("glad-goblin_a", "Goblin_A");
        service.registerBot(bot);

        service.handleActionDenied(bot, "attack", new Location(null, 0D, 64D, 0D), "damage-cancelled");

        PvpVerdict inside = service.queryPvp(new Location(null, 3D, 64D, 3D), bot);
        assertFalse(inside.isAllowed());
        assertSame(PvpVerdict.Source.OBSERVED, inside.getSource());
        assertNotNull(inside.getBox());
        assertTrue(inside.getRegionIds().isEmpty());
    }

    @Test
    public void queryPvpFallsThroughOutsideRadius() {
        GladiatorServiceImpl service = readyService();
        service.setGuardBridge(new AllowAllBridge());
        Bot bot = fakeBot("glad-goblin_a", "Goblin_A");
        service.registerBot(bot);

        service.handleActionDenied(bot, "attack", new Location(null, 0D, 64D, 0D), "damage-cancelled");

        PvpVerdict outside = service.queryPvp(new Location(null, 20D, 64D, 0D), bot);
        assertTrue(outside.isAllowed());
        assertSame(PvpVerdict.Source.DEGRADED, outside.getSource());
    }

    @Test
    public void observedDenyIsPerBot() {
        GladiatorServiceImpl service = readyService();
        service.setGuardBridge(new AllowAllBridge());
        Bot attacker = fakeBot("glad-goblin_a", "Goblin_A");
        Bot other = fakeBot("glad-goblin_b", "Goblin_B");
        service.registerBot(attacker);
        service.registerBot(other);

        service.handleActionDenied(attacker, "attack", new Location(null, 0D, 64D, 0D), "damage-cancelled");

        assertFalse(service.queryPvp(new Location(null, 0D, 64D, 0D), attacker).isAllowed());
        assertTrue(service.queryPvp(new Location(null, 0D, 64D, 0D), other).isAllowed());
    }

    @Test
    public void clearsWhenObservationsExpire() {
        GladiatorServiceImpl service = readyService();
        service.setGuardBridge(new AllowAllBridge());
        Bot bot = fakeBot("glad-goblin_a", "Goblin_A");
        service.registerBot(bot);
        service.handleActionDenied(bot, "dig", new Location(null, 0D, 64D, 0D), "break-cancelled");

        ObservedDenials denials = service.getObservedDenials();
        assertNotNull(denials.match("glad-goblin_a", 0D, 64D, 0D, System.currentTimeMillis()));
        // 直接清空模拟 TTL 到期
        denials.clear("glad-goblin_a");
        assertNull(denials.match("glad-goblin_a", 0D, 64D, 0D, System.currentTimeMillis()));
        assertTrue(service.queryPvp(new Location(null, 0D, 64D, 0D), bot).isAllowed());
    }

    // ------------------------------------------------------------------

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
                        return Logger.getLogger("GladiatorDenialTest");
                    }
                    if ("getConfig".equals(name)) {
                        return null; // 走默认值：ttl=60s，radius=8
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

    private static Bot fakeBot(String botId, String name) {
        final UUID uuid = UUID.nameUUIDFromBytes(botId.getBytes());
        return (Bot) Proxy.newProxyInstance(Bot.class.getClassLoader(),
                new Class<?>[]{Bot.class}, (proxy, method, args) -> {
                    String called = method.getName();
                    if ("getBotId".equals(called)) {
                        return botId;
                    }
                    if ("getName".equals(called)) {
                        return name;
                    }
                    if ("getUniqueId".equals(called)) {
                        return uuid;
                    }
                    if ("isBot".equals(called)) {
                        return Boolean.TRUE;
                    }
                    if ("isReady".equals(called)) {
                        return Boolean.TRUE;
                    }
                    if ("asPlayer".equals(called)) {
                        return null;
                    }
                    if ("getState".equals(called)) {
                        return top.cnolm.gladiator.api.UserState.ALIVE;
                    }
                    if ("hashCode".equals(called)) {
                        return Integer.valueOf(botId.hashCode());
                    }
                    if ("equals".equals(called)) {
                        return Boolean.valueOf(proxy == args[0]);
                    }
                    if ("toString".equals(called)) {
                        return "FakeBot(" + botId + ")";
                    }
                    return null;
                });
    }

    /** 永远 allow 的桥接：用来证明 deny 只可能来自观测缓存。 */
    private static final class AllowAllBridge implements GuardBridge {
        @Override
        public PvpVerdict query(Location location, User user) {
            return PvpVerdict.degraded();
        }

        @Override
        public boolean isAvailable() {
            return false;
        }

        @Override
        public String describe() {
            return "test-allow-all";
        }
    }
}
