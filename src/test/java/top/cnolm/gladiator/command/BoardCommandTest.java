package top.cnolm.gladiator.command;

import org.junit.Test;
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
                + " reason=none player=- bot=-", line);
    }
}
