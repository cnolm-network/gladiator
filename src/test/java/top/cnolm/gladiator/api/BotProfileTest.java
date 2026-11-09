package top.cnolm.gladiator.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

/** G-507：单 Bot 行为配置的默认值、校验与传播。 */
public class BotProfileTest {

    @Test
    public void defaultsMatchDocumentedValues() {
        BotProfile p = BotProfile.defaults();
        assertEquals(15.0, p.getCps(), 1e-9);
        assertEquals(2.0, p.getCpsJitter(), 1e-9);
        assertEquals(220.0, p.getReactionMs(), 1e-9);
        assertEquals(40.0, p.getReactionJitterMs(), 1e-9);
        assertEquals(3.99, p.getReach(), 1e-9);
        assertEquals("default", p.getModel());
        assertEquals(1000.0 / 15.0, p.clickIntervalMs(), 1e-9);
    }

    @Test
    public void builderOverridesOnlyGivenFields() {
        BotProfile p = BotProfile.builder()
                .cps(17.0, 3.0)
                .reaction(180.0, 25.0)
                .reach(4.5)
                .model("aggressive")
                .build();
        assertEquals(17.0, p.getCps(), 1e-9);
        assertEquals(3.0, p.getCpsJitter(), 1e-9);
        assertEquals(180.0, p.getReactionMs(), 1e-9);
        assertEquals(25.0, p.getReactionJitterMs(), 1e-9);
        assertEquals(4.5, p.getReach(), 1e-9);
        assertEquals("aggressive", p.getModel());
    }

    @Test
    public void rejectsOutOfRangeValues() {
        double[][] bad = {
                {-1.0, 2.0, 220.0, 40.0, 3.99},
                {15.0, 20.0, 220.0, 40.0, 3.99},
                {15.0, 2.0, -1.0, 40.0, 3.99},
                {15.0, 2.0, 220.0, -5.0, 3.99},
                {15.0, 2.0, 220.0, 40.0, 0.0},
                {15.0, 2.0, 220.0, 40.0, 6.5},
                {Double.NaN, 2.0, 220.0, 40.0, 3.99},
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
    public void blankModelFallsBackToDefault() {
        assertEquals("default", BotProfile.of(15, 2, 220, 40, 3.99, "  ").getModel());
        assertEquals("default", BotProfile.of(15, 2, 220, 40, 3.99, null).getModel());
    }

    @Test
    public void botSpecCarriesProfileAndDefaultsWhenOmitted() {
        BotProfile custom = BotProfile.builder().cps(20, 1).reach(4.2).build();
        BotSpec spec = BotSpec.of("Glad_Bot_1", "default", null, custom);
        assertSame(custom, spec.getProfile());

        BotSpec plain = BotSpec.of("Glad_Bot_1", "default");
        assertSame(BotProfile.defaults(), plain.getProfile());
        assertEquals(3.99, plain.getProfile().getReach(), 1e-9);
    }
}
