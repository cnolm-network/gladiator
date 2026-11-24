package top.cnolm.gladiator.command;

import org.bukkit.entity.Player;
import org.junit.Test;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.MatchOutcome;
import top.cnolm.gladiator.api.board.PlayerSnapshot;
import top.cnolm.gladiator.api.board.RoundOutcome;
import top.cnolm.gladiator.api.board.RoundSpec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code /gladiator board} 的纯逻辑测试：参数解析与状态行拼接。
 *
 * <p>不启动服务端：这两个方法被刻意抽成静态，就是为了让 dashboard 依赖的命令契约能有单测，
 * 而不是靠「起服 + 手敲命令」验证（那种验证没法进 CI，也不会在改坏时立刻报错）。
 *
 * <p>G-719 追加：{@code board frame}（一帧一次往返）与逐局 {@code roundReason}/{@code lastSeconds}。
 */
public class BoardCommandTest {

    private static String[] argv(String... rest) {
        String[] args = new String[rest.length + 2];
        args[0] = "board";
        args[1] = "start";
        System.arraycopy(rest, 0, args, 2, rest.length);
        return args;
    }

    @Test
    public void parsesMinimalArguments() {
        RoundSpec spec = GladiatorCommand.boardSpec(
                argv("flat-1v1", "Glad_P1", "Glad_B1", "3", "45"));
        assertEquals(3, spec.getRounds());
        assertEquals(45, spec.getSecondsPerRound());
        assertEquals("Glad_P1", spec.extra("player", ""));
        assertEquals("Glad_B1", spec.extra("bot", ""));
        assertNull(spec.getPlayerSpawn());
        assertNull(spec.getBotSpawn());
    }

    @Test
    public void parsesSpawnsAndExtras() {
        RoundSpec spec = GladiatorCommand.boardSpec(argv(
                "flat-1v1", "Glad_P1", "Glad_B1", "2", "30",
                "-6", "4", "0", "6", "4", "0", "kit=gapple", "note=test"));
        double[] player = spec.getPlayerSpawn();
        double[] bot = spec.getBotSpawn();
        assertEquals(-6.0D, player[0], 1e-9);
        assertEquals(4.0D, player[1], 1e-9);
        assertEquals(0.0D, player[2], 1e-9);
        assertEquals(6.0D, bot[0], 1e-9);
        assertEquals("gapple", spec.getKitId());
        assertEquals("test", spec.extra("note", ""));
    }

    @Test
    public void rejectsMissingArguments() {
        try {
            GladiatorCommand.boardSpec(argv("flat-1v1", "Glad_P1"));
            fail("参数不足应当抛 IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("缺少参数"));
        }
    }

    @Test
    public void rejectsBadNumber() {
        try {
            GladiatorCommand.boardSpec(argv("flat-1v1", "Glad_P1", "Glad_B1", "三轮", "30"));
            fail("非数字局数应当抛 IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("局数"));
        }
    }

    @Test
    public void rejectsDanglingToken() {
        try {
            GladiatorCommand.boardSpec(argv("flat-1v1", "Glad_P1", "Glad_B1", "1", "30", "oops"));
            fail("裸露参数应当抛 IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("key=value"));
        }
    }

    @Test
    public void statusLineReportsIdleWhenNoSession() {
        String line = GladiatorCommand.boardStatusLine(null, null);
        assertEquals("session=none board=- round=0 rounds=0 phase=idle score=0:0 draws=0"
                + " reason=none player=- bot=- roundReason=none lastSeconds=-1.0", line);
    }

    // ------------------------------------------------------- G-719：逐局原因/时长

    @Test
    public void statusLineAppendsRoundReasonAndSecondsAtTheEnd() {
        RoundSpec spec = GladiatorCommand.boardSpec(
                argv("flat-1v1", "Glad_P1", "Glad_B1", "3", "45"));
        BoardContext session = new StubContext(spec, "fighting", 2, "kill", 42.34D);
        String line = GladiatorCommand.boardStatusLine(session, null);
        assertEquals("session=s-1 board=flat-1v1 round=2 rounds=3 phase=fighting score=1:0 draws=0"
                + " reason=running player=Glad_P1 bot=Glad_B1"
                + " roundReason=kill lastSeconds=42.3", line);
        // 只能追加：新字段必须在既有字段之后，旧解析器按前缀取键才不会错位。
        assertTrue(line.indexOf("roundReason=") > line.indexOf(" bot=Glad_B1"));
        assertTrue(line.indexOf("lastSeconds=") > line.indexOf("roundReason="));
    }

    @Test
    public void statusLineShowsUnknownSecondsAsMinusOne() {
        RoundSpec spec = GladiatorCommand.boardSpec(
                argv("flat-1v1", "Glad_P1", "Glad_B1", "1", "30"));
        BoardContext session = new StubContext(spec, "countdown", 1, "none", -1.0D);
        String line = GladiatorCommand.boardStatusLine(session, null);
        assertTrue(line.endsWith(" roundReason=none lastSeconds=-1.0"));
    }

    @Test
    public void formatSecondsUsesDotDecimal() {
        // dashboard 按 "." 解析：不能被系统区域设置换成逗号。
        assertEquals("42.3", GladiatorCommand.formatSeconds(42.34D));
        assertEquals("-1.0", GladiatorCommand.formatSeconds(-1.0D));
        assertEquals("0.0", GladiatorCommand.formatSeconds(0.0D));
    }

    @Test
    public void resolveRoundSecondsFallsBackToWallClock() {
        // 板子报了就用板子的（含小数）
        assertEquals(42.3D, BoardContext.resolveRoundSeconds(42.3D, 0L, 999_999L), 1e-9);
        // 板子报未知（<0 / NaN）→ 墙钟估算，与 G-719 之前一致
        assertEquals(12.5D, BoardContext.resolveRoundSeconds(-1.0D, 1_000L, 13_500L), 1e-9);
        assertEquals(3.0D, BoardContext.resolveRoundSeconds(Double.NaN, 0L, 3_000L), 1e-9);
        // 时钟回拨也不会得到负数
        assertEquals(0.0D, BoardContext.resolveRoundSeconds(-1.0D, 5_000L, 1_000L), 1e-9);
    }

    @Test
    public void recordSecondsRoundsToInt() {
        assertEquals(42, BoardContext.recordSeconds(42.3D));
        assertEquals(43, BoardContext.recordSeconds(42.6D));
        assertEquals(0, BoardContext.recordSeconds(-1.0D));
        assertEquals(0, BoardContext.recordSeconds(Double.NaN));
    }

    // ------------------------------------------------------------ G-719：board frame

    @Test
    public void frameWithoutSessionKeepsThreeSegments() {
        String line = GladiatorCommand.boardFrameLine(null, null);
        assertEquals("session=none board=- round=0 rounds=0 phase=idle score=0:0 draws=0"
                + " reason=none player=- bot=- roundReason=none lastSeconds=-1.0"
                + " | stats - offline | stats - offline", line);
        assertEquals("固定三段：dashboard 按 \" | \" 切", 3, line.split(" \\| ", -1).length);
    }

    @Test
    public void frameMarksOfflineParticipants() {
        // 测试环境没有服务端实例 → findOnlinePlayer 返回 null → 两段都写 offline，且不报错、不省段。
        RoundSpec spec = GladiatorCommand.boardSpec(
                argv("flat-1v1", "Glad_P1", "Glad_B1", "2", "20"));
        BoardContext session = new StubContext(spec, "fighting", 1, "none", -1.0D);
        String line = GladiatorCommand.boardFrameLine(session, null);
        String[] segments = line.split(" \\| ", -1);
        assertEquals(3, segments.length);
        // 状态段必须与 board status 完全一致（复用同一个函数）
        assertEquals(GladiatorCommand.boardStatusLine(session, null), segments[0]);
        assertEquals("stats Glad_P1 offline", segments[1]);
        assertEquals("stats Glad_B1 offline", segments[2]);
        assertTrue(segments[0].contains("roundReason="));
        assertTrue(segments[0].contains("lastSeconds="));
    }

    @Test
    public void frameComposesStatsSegmentsWithExistingFormat() {
        // 纯拼接：stats 段就是 "stats <名字> " + stats 子命令那一行。
        String line = GladiatorCommand.boardFrameLine("session=s1 phase=fighting",
                "Glad_P1", "name=Glad_P1 health=20.0 food=20",
                "Glad_B1", "name=Glad_B1 health=8.0 food=18");
        assertEquals("session=s1 phase=fighting"
                + " | stats Glad_P1 name=Glad_P1 health=20.0 food=20"
                + " | stats Glad_B1 name=Glad_B1 health=8.0 food=18", line);
    }

    @Test
    public void frameUsesDashWhenParticipantUnknown() {
        // 没有会话时名字未知：用 "-" 占位，仍然是三段（dashboard 不用判空）。
        String line = GladiatorCommand.boardFrameLine("status", "", null, null, null);
        assertEquals("status | stats - offline | stats - offline", line);
    }

    // ------------------------------------------------------------------ 测试桩

    /** 最小 {@link BoardContext} 桩：只喂状态行需要的字段，其余空实现。 */
    private static final class StubContext implements BoardContext {

        private final RoundSpec spec;
        private final String phase;
        private final int round;
        private final String roundReason;
        private final double lastSeconds;

        StubContext(RoundSpec spec, String phase, int round, String roundReason, double lastSeconds) {
            this.spec = spec;
            this.phase = phase;
            this.round = round;
            this.roundReason = roundReason;
            this.lastSeconds = lastSeconds;
        }

        @Override
        public String sessionId() {
            return "s-1";
        }

        @Override
        public String boardId() {
            return "flat-1v1";
        }

        @Override
        public RoundSpec spec() {
            return spec;
        }

        @Override
        public int round() {
            return round;
        }

        @Override
        public long tick() {
            return 0L;
        }

        @Override
        public long startedAt() {
            return 0L;
        }

        @Override
        public GladiatorAPI api() {
            return null;
        }

        @Override
        public Player player(String name) {
            return null;
        }

        @Override
        public PlayerSnapshot snapshot(String player) {
            return PlayerSnapshot.offline(player);
        }

        @Override
        public boolean isOnline(String player) {
            return false;
        }

        @Override
        public void log(String message) {
            // 测试桩：不打日志。
        }

        @Override
        public void endRound(RoundOutcome outcome) {
            // 测试桩：不记账。
        }

        @Override
        public void endMatch(MatchOutcome outcome) {
            // 测试桩：不记账。
        }

        @Override
        public void phase(String value) {
            // 测试桩：只读。
        }

        @Override
        public String phase() {
            return phase;
        }

        @Override
        public MatchOutcome summary() {
            return new MatchOutcome(1, 0, 0, "running", null, null);
        }

        @Override
        public String lastReason() {
            return "";
        }

        @Override
        public String roundReason() {
            return roundReason;
        }

        @Override
        public double lastSeconds() {
            return lastSeconds;
        }
    }
}
