package top.cnolm.gladiator.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * G-507：单 Bot 行为配置的默认值、校验与传播。
 *
 * <p>G-708 起 reach 默认 **3.0**（1.8 原版玩家水平）：3.99 这种超限值只能通过
 * {@code cheat.reach} 显式打开，否则一律拒绝——否则陪练的距离感是假的。
 */
public class BotProfileTest {

    @Test
    public void defaultsMatchDocumentedValues() {
        BotProfile p = BotProfile.defaults();
        assertEquals(15.0, p.getCps(), 1e-9);
        assertEquals(2.0, p.getCpsJitter(), 1e-9);
        assertEquals(220.0, p.getReactionMs(), 1e-9);
        assertEquals(40.0, p.getReactionJitterMs(), 1e-9);
        assertEquals(3.0, p.getReach(), 1e-9);
        assertEquals("default", p.getModel());
        assertEquals(1000.0 / 15.0, p.clickIntervalMs(), 1e-9);
        assertFalse("默认必须关着作弊开关", p.isCheatReach());
        assertEquals(BotProfile.DEFAULT_REACH, BotProfile.LEGIT_REACH_MAX, 1e-9);
    }

    @Test
    public void builderOverridesOnlyGivenFields() {
        BotProfile p = BotProfile.builder()
                .cps(17.0, 3.0)
                .reaction(180.0, 25.0)
                .reach(2.8)
                .model("aggressive")
                .build();
        assertEquals(17.0, p.getCps(), 1e-9);
        assertEquals(3.0, p.getCpsJitter(), 1e-9);
        assertEquals(180.0, p.getReactionMs(), 1e-9);
        assertEquals(25.0, p.getReactionJitterMs(), 1e-9);
        assertEquals(2.8, p.getReach(), 1e-9);
        assertEquals("aggressive", p.getModel());
        assertFalse(p.isCheatReach());
    }

    @Test
    public void rejectsOutOfRangeValues() {
        double[][] bad = {
                {-1.0, 2.0, 220.0, 40.0, 3.0},
                {15.0, 20.0, 220.0, 40.0, 3.0},
                {15.0, 2.0, -1.0, 40.0, 3.0},
                {15.0, 2.0, 220.0, -5.0, 3.0},
                {15.0, 2.0, 220.0, 40.0, 0.0},      // <= 0 → 低于下限
                {15.0, 2.0, 220.0, 40.0, 0.9},      // < MIN_REACH
                {15.0, 2.0, 220.0, 40.0, 6.5},      // > MAX_REACH
                {Double.NaN, 2.0, 220.0, 40.0, 3.0},
        };
        for (double[] row : bad) {
            try {
                BotProfile.of(row[0], row[1], row[2], row[3], row[4], "default");
                fail("应当拒绝 cps=" + row[0] + " jitter=" + row[1] + " reach=" + row[4]);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage() != null);
            }
        }
    }

    @Test
    public void reachAboveLegitLimitNeedsCheatSwitch() {
        // 不带开关：3.99 必须被拒，且错误信息里要能看出「该开 cheat.reach」
        try {
            BotProfile.builder().reach(3.99).build();
            fail("3.99 不应在默认（非作弊）模式下被接受");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("cheat.reach")
                    || expected.getMessage().contains("超限"));
        }
        // 开了开关：允许，并标记 isCheatReach
        BotProfile cheat = BotProfile.builder().reach(3.99).cheatReach(true).build();
        assertEquals(3.99, cheat.getReach(), 1e-9);
        assertTrue(cheat.isCheatReach());
        assertTrue(cheat.toString().contains("cheat"));
        // 开了开关但超过硬上限 6 依然拒绝
        try {
            BotProfile.builder().reach(7.0).cheatReach(true).build();
            fail("即便开了作弊开关，reach 也不能超过 6");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("reach"));
        }
    }

    @Test
    public void blankModelFallsBackToDefault() {
        assertEquals("default", BotProfile.of(15, 2, 220, 40, 3.0, "  ").getModel());
        assertEquals("default", BotProfile.of(15, 2, 220, 40, 3.0, null).getModel());
    }

    @Test
    public void botSpecCarriesProfileAndDefaultsWhenOmitted() {
        BotProfile custom = BotProfile.builder().cps(20, 1).reach(2.6).build();
        BotSpec spec = BotSpec.of("Glad_Bot_1", "default", null, custom);
        assertSame(custom, spec.getProfile());

        BotSpec plain = BotSpec.of("Glad_Bot_1", "default");
        assertSame(BotProfile.defaults(), plain.getProfile());
        assertEquals(3.0, plain.getProfile().getReach(), 1e-9);
    }
}
