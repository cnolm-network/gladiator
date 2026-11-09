package top.cnolm.gladiator.api;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class PvpVerdictTest {

    @Test
    public void degradedAllowsWithDegradedSource() {
        PvpVerdict verdict = PvpVerdict.degraded();
        assertTrue(verdict.isAllowed());
        assertEquals(PvpVerdict.Source.DEGRADED, verdict.getSource());
        assertTrue(verdict.getRegionIds().isEmpty());
        assertNull(verdict.getBox());
    }

    @Test
    public void unknownAllowsWithUnknownSource() {
        PvpVerdict verdict = PvpVerdict.unknown();
        assertTrue(verdict.isAllowed());
        assertEquals(PvpVerdict.Source.UNKNOWN, verdict.getSource());
    }

    @Test
    public void flagVerdictCarriesRegionIdsAndBox() {
        PvpVerdict.Box box = PvpVerdict.Box.of(10, 60, 10, 20, 70, 20);
        PvpVerdict verdict = PvpVerdict.of(false, Arrays.asList("arena", "spawn"), PvpVerdict.Source.FLAG, box);
        assertFalse(verdict.isAllowed());
        assertEquals(PvpVerdict.Source.FLAG, verdict.getSource());
        assertEquals(Arrays.asList("arena", "spawn"), verdict.getRegionIds());
        assertEquals(10.0, box.getMinX(), 0.0);
        assertEquals(20.0, box.getMaxZ(), 0.0);
    }

    @Test
    public void boxNormalizesInvertedCorners() {
        PvpVerdict.Box box = PvpVerdict.Box.of(20, 70, 20, 10, 60, 10);
        assertEquals(10.0, box.getMinX(), 0.0);
        assertEquals(60.0, box.getMinY(), 0.0);
        assertEquals(10.0, box.getMinZ(), 0.0);
        assertEquals(20.0, box.getMaxX(), 0.0);
        assertEquals(70.0, box.getMaxY(), 0.0);
        assertEquals(20.0, box.getMaxZ(), 0.0);
    }

    @Test
    public void regionIdsAreDefensivelyCopied() {
        List<String> source = new ArrayList<>();
        source.add("arena");
        PvpVerdict verdict = PvpVerdict.of(true, source, PvpVerdict.Source.GLOBAL, null);
        source.add("mutated");
        assertEquals(1, verdict.getRegionIds().size());
    }
}
