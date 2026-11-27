package top.cnolm.gladiator.core.board;

import org.bukkit.plugin.Plugin;
import org.junit.Test;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.GameEvent;
import top.cnolm.gladiator.api.board.RoundSpec;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@code MATCH_PHASE} 下发链路（块 B 插件侧）：板子声明阶段 → 会话算时间窗 → 每秒重推 → 结束通知。
 *
 * <p>判据（对应 docs/protocol.md §2.11）：
 * <ul>
 *   <li>阶段**变化**时立刻推一帧（不等到下一秒）；</li>
 *   <li>倒计时/战斗中每秒重推，`seconds` 递减；同一秒内重复调用不重推；</li>
 *   <li>同阶段重复声明不重开时间窗；</li>
 *   <li>结束（{@code finish}；{@code endMatch} 走同一条迁移路径）推 {@code ended} + {@code seconds=0}；</li>
 *   <li>没人注入 sink、或 extras 里没有参与者时静默跳过，不 NPE。</li>
 * </ul>
 *
 * <p>不启动服务端：{@code Bukkit.getServer()} 为 null 时主线程断言放行（同 BoardBotEventsTest）。
 */
public class BoardMatchPhaseTest {

    private static final Logger LOG = Logger.getLogger("BoardMatchPhaseTest");

    // ------------------------------------------------------------ 测试替身

    private static Plugin fakePlugin() {
        InvocationHandler handler = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) {
                String name = method.getName();
                if ("getLogger".equals(name)) {
                    return LOG;
                }
                if ("getName".equals(name)) {
                    return "Gladiator";
                }
                if ("isEnabled".equals(name)) {
                    return Boolean.TRUE;
                }
                if ("toString".equals(name)) {
                    return "FakePlugin";
                }
                if ("hashCode".equals(name)) {
                    return Integer.valueOf(1);
                }
                if ("equals".equals(name)) {
                    return Boolean.valueOf(proxy == args[0]);
                }
                return null;
            }
        };
        return (Plugin) Proxy.newProxyInstance(
                BoardMatchPhaseTest.class.getClassLoader(),
                new Class<?>[]{Plugin.class}, handler);
    }

    /** 记录每一帧；username 一起记，用来验证「只发给 extras 里的参与者」。 */
    private static final class RecordingSink implements MatchPhaseSink {
        final List<String> usernames = new ArrayList<String>();
        final List<String> phases = new ArrayList<String>();
        final List<Integer> seconds = new ArrayList<Integer>();
        final List<Long> endsAt = new ArrayList<Long>();

        @Override
        public void sendMatchPhase(String username, String phase, int round, int seconds, long endsAtMs) {
            usernames.add(username);
            phases.add(phase);
            this.seconds.add(Integer.valueOf(seconds));
            endsAt.add(Long.valueOf(endsAtMs));
        }

        int frames() {
            return phases.size();
        }

        String lastPhase() {
            return phases.get(phases.size() - 1);
        }

        int lastSeconds() {
            return seconds.get(seconds.size() - 1).intValue();
        }

        void clear() {
            usernames.clear();
            phases.clear();
            seconds.clear();
            endsAt.clear();
        }
    }

    private static final class IdleBoard implements Board {
        @Override
        public String id() {
            return "test-board";
        }

        @Override
        public void onStart(BoardContext context) {
        }

        @Override
        public void onEvent(GameEvent event) {
        }

        @Override
        public void onTick(long tick) {
        }

        @Override
        public void onStop(String reason) {
        }
    }

    private static RoundSpec spec(int secondsPerRound, boolean withParticipants) {
        RoundSpec.Builder builder = RoundSpec.builder()
                .rounds(3)
                .secondsPerRound(secondsPerRound)
                .kitId("classic");
        if (withParticipants) {
            builder.extra("player", "lvsm_").extra("bot", "Glad_B1");
        }
        return builder.build();
    }

    /** 直接注入会话（生产路径 start() 要 Bukkit 调度器）。 */
    private static BoardSessionImpl session(BoardRegistryImpl registry, RoundSpec spec) {
        return new BoardSessionImpl(registry, fakePlugin(), new IdleBoard(), null, null, spec, "phase-test");
    }

    // ------------------------------------------------------------ 用例

    @Test
    public void phaseChangePushesFrameImmediatelyToBothParticipants() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(60, true));

        session.phase("countdown");

        assertEquals("两个参与者各一帧", 2, sink.frames());
        assertEquals("lvsm_", sink.usernames.get(0));
        assertEquals("Glad_B1", sink.usernames.get(1));
        assertEquals("countdown", sink.lastPhase());
        assertEquals("倒计时默认 5 秒", 5, sink.lastSeconds());
        assertTrue("endsAt 必须是未来时刻", sink.endsAt.get(0).longValue() > System.currentTimeMillis());
    }

    @Test
    public void sameSecondDoesNotRepushButNextSecondDecrements() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(60, true));
        session.phase("countdown");
        sink.clear();

        long now = System.currentTimeMillis();
        // 同一秒内：节流住，不重推（5Hz ticker 下这是绝大多数帧）。
        session.pumpPhase(now + 100L);
        assertEquals(0, sink.frames());

        // 过了 1 秒：推一帧，秒数递减。
        session.pumpPhase(now + 1500L);
        assertEquals(2, sink.frames());
        int decremented = sink.lastSeconds();
        assertTrue("倒计时必须递减（实际 " + decremented + "）", decremented < 5 && decremented >= 3);

        // 再往后 1 秒再推，继续递减。
        int previous = decremented;
        session.pumpPhase(now + 2600L);
        assertTrue("继续递减（" + previous + " → " + sink.lastSeconds() + "）", sink.lastSeconds() < previous);
    }

    @Test
    public void repeatedSamePhaseDeclarationKeepsWindow() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(60, true));

        session.phase("countdown");
        long firstEndsAt = sink.endsAt.get(0).longValue();
        sink.clear();
        session.phase("countdown");

        assertEquals("同阶段重复声明不重推", 0, sink.frames());
        assertEquals("countdown", session.phase());
        assertEquals("时间窗不被重置", 5, session.remainingPhaseSeconds(firstEndsAt - 5000L));
    }

    @Test
    public void fightingUsesRoundSeconds() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(90, true));

        session.phase("fighting");
        assertEquals("fighting", sink.lastPhase());
        assertEquals("战斗阶段秒数 = 每局秒数", 90, sink.lastSeconds());

        // 战斗中也每秒补帧（剩余秒数递减），运行时不据此判定，只用来显示。
        long now = System.currentTimeMillis();
        session.pumpPhase(now + 1500L);
        assertEquals(4, sink.frames());
        assertTrue("战斗中剩余秒数递减", sink.lastSeconds() <= 89 && sink.lastSeconds() >= 88);
    }

    @Test
    public void finishAlsoPushesEnded() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(60, true));

        session.phase("countdown");
        sink.clear();
        session.finish("command");

        assertEquals(2, sink.frames());
        assertEquals("ended", sink.lastPhase());
        assertEquals(0, sink.lastSeconds());
    }

    @Test
    public void missingSinkOrParticipantsIsSilent() {
        // 没有注入 sink：phase() 不能抛（链路没起来也要能开对局）。
        BoardRegistryImpl bare = new BoardRegistryImpl(fakePlugin(), null, null);
        BoardSessionImpl session = session(bare, spec(60, true));
        session.phase("countdown");
        assertEquals("countdown", session.phase());

        // 注入 sink 但 extras 里没有参与者：一帧都不该发。
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl lone = session(registry, spec(60, false));
        lone.phase("countdown");
        assertEquals(0, sink.frames());
    }

    @Test
    public void unknownPhaseReportsUnknownSecondsAndDoesNotTick() {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        RecordingSink sink = new RecordingSink();
        registry.attachMatchPhaseSink(sink);
        BoardSessionImpl session = session(registry, spec(60, true));

        // 初始阶段就是 starting（板子不声明时外部看到的就是它）：重复声明是空操作。
        session.phase("starting");
        assertEquals("初始阶段不重推", 0, sink.frames());

        // 板子自造的阶段名：板子知道怎么玩，Gladiator 不知道时长 → 只报阶段本身。
        session.phase("suspense");
        assertEquals(2, sink.frames());
        assertEquals("suspense", sink.lastPhase());
        assertEquals("板子没声明时长时报 -1", -1, sink.lastSeconds());
        assertEquals("endsAt 未知记 0", 0L, sink.endsAt.get(0).longValue());

        // 非计时阶段：ticker 不该补帧（只有 countdown/fighting 有每秒重推）。
        sink.clear();
        session.pumpPhase(System.currentTimeMillis() + 5000L);
        assertEquals(0, sink.frames());
    }
}
