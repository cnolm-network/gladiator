package top.cnolm.gladiator.core.guard;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * A3：观测式拒绝缓存的命中口径。
 *
 * <p>纯 Java，不需要服务端：时间由测试显式传入。
 */
public class ObservedDenialsTest {

    private static final long NOW = 1_700_000_000_000L;

    private static ObservedDenials.Entry entry(String botId, double x, double y, double z,
                                              double radius, long expiresAt) {
        return new ObservedDenials.Entry(botId, x, y, z, radius, "attack", expiresAt);
    }

    @Test
    public void hitsInsideRadius() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW + 60_000L));

        assertNotNull(denials.match("glad-a", 3, 64, 3, NOW));
        assertNotNull(denials.match("glad-a", 0, 64, 0, NOW));
    }

    @Test
    public void missesOutsideRadius() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW + 60_000L));

        assertNull(denials.match("glad-a", 5.5, 64, 0, NOW));
        assertNull(denials.match("glad-a", 0, 70, 0, NOW));
    }

    @Test
    public void missesAfterExpiry() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW + 1_000L));

        assertNotNull(denials.match("glad-a", 0, 64, 0, NOW + 999L));
        assertNull(denials.match("glad-a", 0, 64, 0, NOW + 1_000L));
        assertNull(denials.match("glad-a", 0, 64, 0, NOW + 5_000L));
    }

    @Test
    public void isolatedPerBot() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW + 60_000L));

        assertNull(denials.match("glad-b", 0, 64, 0, NOW));
        assertEquals(1, denials.size("glad-a"));
        assertEquals(0, denials.size("glad-b"));
    }

    @Test
    public void capacityDropsOldest() {
        ObservedDenials denials = new ObservedDenials(2);
        denials.record(entry("glad-a", 0, 64, 0, 1, NOW + 60_000L));
        denials.record(entry("glad-a", 50, 64, 0, 1, NOW + 60_000L));
        denials.record(entry("glad-a", 100, 64, 0, 1, NOW + 60_000L));

        assertEquals(2, denials.size("glad-a"));
        // 最旧的一条（x=0）已被挤掉
        assertNull(denials.match("glad-a", 0, 64, 0, NOW));
        assertNotNull(denials.match("glad-a", 100, 64, 0, NOW));
    }

    @Test
    public void newestWinsWhenOverlapping() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 10, NOW + 60_000L));
        ObservedDenials.Entry newer = new ObservedDenials.Entry(
                "glad-a", 1, 64, 1, 10, "dig", NOW + 60_000L);
        denials.record(newer);

        assertEquals(newer, denials.match("glad-a", 0, 64, 0, NOW));
        assertEquals("dig", denials.match("glad-a", 0, 64, 0, NOW).getAction());
    }

    @Test
    public void boxMatchesRadius() {
        ObservedDenials.Entry entry = entry("glad-a", 10, 64, -5, 4, NOW + 1_000L);
        top.cnolm.gladiator.api.PvpVerdict.Box box = entry.box();

        assertEquals(6.0D, box.getMinX(), 1e-9);
        assertEquals(14.0D, box.getMaxX(), 1e-9);
        assertEquals(60.0D, box.getMinY(), 1e-9);
        assertEquals(-9.0D, box.getMinZ(), 1e-9);
        assertEquals(-1.0D, box.getMaxZ(), 1e-9);
    }

    @Test
    public void clearRemovesEverything() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW + 60_000L));
        denials.record(entry("glad-b", 0, 64, 0, 5, NOW + 60_000L));
        assertEquals(2, denials.totalSize());

        denials.clear("glad-a");
        assertEquals(1, denials.totalSize());
        assertNull(denials.match("glad-a", 0, 64, 0, NOW));

        denials.clear();
        assertEquals(0, denials.totalSize());
    }

    @Test
    public void expiredEntriesArePrunedOnMatch() {
        ObservedDenials denials = new ObservedDenials();
        denials.record(entry("glad-a", 0, 64, 0, 5, NOW));
        assertEquals(1, denials.size("glad-a"));

        assertNull(denials.match("glad-a", 0, 64, 0, NOW + 1));
        assertTrue(denials.size("glad-a") == 0);
    }
}
