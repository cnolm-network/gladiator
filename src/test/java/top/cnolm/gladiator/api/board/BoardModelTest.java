package top.cnolm.gladiator.api.board;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** 板子 API 的值对象：构造、裁剪、不可变性与事件取值助手。 */
public class BoardModelTest {

    @Test
    public void roundSpecDefaultsAreSane() {
        RoundSpec spec = RoundSpec.builder().build();
        assertEquals(1, spec.getRounds());
        assertEquals(60, spec.getSecondsPerRound());
        assertEquals("default", spec.getKitId());
        assertEquals(3, spec.getShieldSeconds());
        assertNull(spec.getPlayerSpawn());
        assertNull(spec.getBotSpawn());
    }

    @Test
    public void roundSpecClampsAndCopiesArrays() {
        double[] spawn = new double[]{1.0D, 4.0D, 2.0D};
        RoundSpec.Builder builder = RoundSpec.builder()
                .rounds(0)
                .secondsPerRound(-5)
                .shieldSeconds(-1)
                .kitId("  ")
                .playerSpawn(spawn[0], spawn[1], spawn[2]);
        RoundSpec spec = builder.build();
        assertEquals(1, spec.getRounds());
        assertEquals(1, spec.getSecondsPerRound());
        assertEquals(0, spec.getShieldSeconds());
        // 空白 kitId 不覆盖默认值。
        assertEquals("default", spec.getKitId());
        spawn[0] = 99.0D;
        assertEquals(1.0D, spec.getPlayerSpawn()[0], 1e-9);
        // 每次取都返回副本，改它不会污染 spec。
        spec.getPlayerSpawn()[0] = 42.0D;
        assertEquals(1.0D, spec.getPlayerSpawn()[0], 1e-9);
    }

    @Test
    public void roundSpecExtrasAreImmutable() {
        Map<String, String> probe = new HashMap<String, String>();
        RoundSpec spec = RoundSpec.builder().extra("player", "A").build();
        assertEquals("A", spec.extra("player", "?"));
        assertEquals("?", spec.extra("missing", "?"));
        try {
            spec.getExtras().put("x", "y");
            throw new AssertionError("extras 应该是不可变的");
        } catch (UnsupportedOperationException expected) {
            probe.put("ok", "1");
        }
        assertEquals(1, probe.size());
    }

    @Test
    public void gameEventExposesTypedAccessors() {
        Map<String, Object> data = new HashMap<String, Object>();
        data.put("damage", 3.5D);
        data.put("killer", "Bot_A");
        data.put("drops", 4);
        GameEvent event = GameEvent.of(GameEventType.PLAYER_DEATH, 7L, 1000L, 2, "Bot_A", "", data);
        assertEquals(GameEventType.PLAYER_DEATH, event.getType());
        assertEquals(7L, event.getSeq());
        assertEquals(2, event.getRound());
        assertEquals(3.5D, event.number("damage", 0.0D), 1e-9);
        assertEquals(4, event.integer("drops", 0));
        assertEquals("Bot_A", event.string("killer", ""));
        assertEquals("?", event.string("missing", "?"));
        // 空 actor/target 归一为空串，调用方不用判 null。
        assertEquals("", event.getTarget());
        try {
            event.getData().put("x", "y");
            throw new AssertionError("data 应该是不可变的");
        } catch (UnsupportedOperationException expected) {
            assertTrue(true);
        }
    }

    @Test
    public void roundOutcomeNormalizesNulls() {
        RoundOutcome outcome = new RoundOutcome(null, null, null);
        assertEquals(RoundOutcome.WINNER_DRAW, outcome.getWinner());
        assertEquals("unknown", outcome.getReason());
        assertTrue(outcome.getDetail().isEmpty());
        assertEquals(RoundOutcome.WINNER_BOT, RoundOutcome.bot("kill").getWinner());
        assertEquals("kill", RoundOutcome.bot("kill").getReason());
    }

    @Test
    public void matchOutcomeKeepsRoundOrder() {
        List<MatchOutcome.RoundRecord> rounds = new ArrayList<MatchOutcome.RoundRecord>();
        rounds.add(new MatchOutcome.RoundRecord(1, RoundOutcome.WINNER_PLAYER, "kill", 12));
        rounds.add(new MatchOutcome.RoundRecord(2, RoundOutcome.WINNER_BOT, "kill", 8));
        MatchOutcome outcome = new MatchOutcome(1, 1, 0, "rounds-finished", rounds, null);
        assertEquals(2, outcome.getRounds().size());
        assertEquals(1, outcome.getRounds().get(0).getIndex());
        assertEquals(RoundOutcome.WINNER_BOT, outcome.getRounds().get(1).getWinner());
        assertEquals(8, outcome.getRounds().get(1).getSeconds());
        assertEquals("rounds-finished", outcome.getReason());
        rounds.clear();
        assertEquals(2, outcome.getRounds().size());
    }

    @Test
    public void playerSnapshotOfflineIsInert() {
        PlayerSnapshot snapshot = PlayerSnapshot.offline("Ghost");
        assertFalse(snapshot.isOnline());
        assertEquals("Ghost", snapshot.getName());
        assertEquals(0.0D, snapshot.getHealth(), 1e-9);
    }

    @Test
    public void boardsRegistryStartsNull() {
        // 静态入口未安装时必须是 null，板子插件据此判断依赖顺序。
        assertNull(Boards.registry());
    }
}
