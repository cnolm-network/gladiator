package top.cnolm.gladiator.core.board;

import org.bukkit.plugin.Plugin;
import org.junit.Test;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.GameEvent;
import top.cnolm.gladiator.api.board.GameEventType;
import top.cnolm.gladiator.api.board.RoundSpec;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * BOT_READY / BOT_OFFLINE 进入板子事件流（两块之前：只声明不发送）。
 *
 * <p>判据（与 {@code BotSpawnService} 的两条链路一一对应）：
 * <ul>
 *   <li>就绪 → 板子收到 {@code BOT_READY}，{@code actor}=用户名、{@code data.botId} 正确；</li>
 *   <li>退出/spawn 失败 → 板子收到 {@code BOT_OFFLINE}，{@code data.reason} 原样带出；</li>
 *   <li>没有活动会话时静默丢弃，不 NPE、不缓存到下一场（避免「上一场的 ready 漏到这一场」）。</li>
 * </ul>
 *
 * <p>不启动服务端：{@code Bukkit.getServer()} 为 null 时主线程断言直接放行
 * （与 {@code GladiatorServiceImpl#requirePrimaryThread} 同口径）。
 */
public class BoardBotEventsTest {

    private static final Logger LOG = Logger.getLogger("BoardBotEventsTest");

    // ------------------------------------------------------------ 测试替身

    /** 只实现用到的方法：getLogger() 返回真 Logger，其余返回 null。 */
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
                    return Boolean.valueOf(proxy == (args == null ? null : args[0]));
                }
                return null;
            }
        };
        return (Plugin) Proxy.newProxyInstance(
                BoardBotEventsTest.class.getClassLoader(), new Class[]{Plugin.class}, handler);
    }

    /** 收集事件的假板子（非 final：异常路径的用例要覆写 onEvent）。 */
    private static class CollectingBoard implements Board {
        final List<GameEvent> events = new ArrayList<GameEvent>();
        String stopped;

        @Override
        public String id() {
            return "test-board";
        }

        @Override
        public void onStart(BoardContext context) {
        }

        @Override
        public void onEvent(GameEvent event) {
            events.add(event);
        }

        @Override
        public void onTick(long tick) {
        }

        @Override
        public void onStop(String reason) {
            stopped = reason;
        }
    }

    /** 直接注入一个会话（生产路径是 {@code start()}，它要 Bukkit 调度器，单测里不碰）。 */
    private static BoardRegistryImpl registryWithSession(CollectingBoard board) {
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        registry.active = new BoardSessionImpl(registry, fakePlugin(), board, null, null,
                RoundSpec.builder().build(), "test-session");
        return registry;
    }

    // ------------------------------------------------------------ 用例

    @Test
    public void readyEmitsBotReadyEvent() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);

        registry.onBotReady("Bot_V0A", "glad-bot_v0a");

        assertEquals(1, board.events.size());
        GameEvent event = board.events.get(0);
        assertEquals(GameEventType.BOT_READY, event.getType());
        assertEquals("Bot_V0A", event.getActor());
        assertEquals("", event.getTarget());
        assertEquals("glad-bot_v0a", event.string("botId", "?"));
        assertEquals("", event.string("reason", "?"));
        assertTrue("seq 必须被分配到（板子按 seq 去重）", event.getSeq() > 0L);
        assertEquals(1, event.getRound());
    }

    @Test
    public void offlineEmitsBotOfflineWithReason() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);

        registry.onBotOffline("Bot_V0A", "glad-bot_v0a", "TIMEOUT: 等待 Bot 登录超过 20s");

        assertEquals(1, board.events.size());
        GameEvent event = board.events.get(0);
        assertEquals(GameEventType.BOT_OFFLINE, event.getType());
        assertEquals("Bot_V0A", event.getActor());
        assertEquals("glad-bot_v0a", event.string("botId", "?"));
        assertEquals("TIMEOUT: 等待 Bot 登录超过 20s", event.string("reason", "?"));
    }

    @Test
    public void seqIncreasesAcrossBotEvents() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);

        registry.onBotReady("Bot_A", "glad-bot_a");
        registry.onBotOffline("Bot_A", "glad-bot_a", "player-quit");

        assertEquals(2, board.events.size());
        assertTrue("seq 必须严格递增（BOT_READY 先于 BOT_OFFLINE）",
                board.events.get(1).getSeq() > board.events.get(0).getSeq());
        assertEquals(GameEventType.BOT_OFFLINE, board.events.get(1).getType());
        assertEquals("player-quit", board.events.get(1).string("reason", "?"));
    }

    @Test
    public void noActiveSessionDropsSilently() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        // 没有 start()：active 为 null。两条链路都必须静默丢弃，而不是 NPE/缓存。
        registry.onBotReady("Bot_A", "glad-bot_a");
        registry.onBotOffline("Bot_A", "glad-bot_a", "spawn-failed");
        assertTrue(board.events.isEmpty());
    }

    @Test
    public void staticEntryIsNoOpWithoutEnabledRegistry() {
        // 未 enable（或已 disable）时静态入口也必须安全：BotSpawnService 不判断状态就调用。
        // 先保证没有残留（其它用例不起 enable，这里只是把语义钉死）。
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);
        registry.active = null;
        BoardRegistryImpl.notifyBotReady("Bot_A", "glad-bot_a");
        BoardRegistryImpl.notifyBotOffline("Bot_A", "glad-bot_a", "x");
        assertTrue(board.events.isEmpty());
    }

    @Test
    public void finishedSessionDoesNotForwardBotEvents() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);
        registry.active.finish("test-over");

        registry.onBotReady("Bot_A", "glad-bot_a");
        registry.onBotOffline("Bot_A", "glad-bot_a", "late");

        assertTrue("对局结束后不该再往板子塞事件", board.events.isEmpty());
    }

    @Test
    public void nullBotIdAndReasonAreNormalized() {
        CollectingBoard board = new CollectingBoard();
        BoardRegistryImpl registry = registryWithSession(board);

        registry.onBotReady("Bot_A", null);
        registry.onBotOffline("Bot_A", null, null);

        assertEquals(2, board.events.size());
        assertEquals("", board.events.get(0).string("botId", "?"));
        assertEquals("", board.events.get(1).string("botId", "?"));
        assertEquals("", board.events.get(1).string("reason", "?"));
    }

    @Test
    public void boardExceptionDoesNotBreakDispatch() {
        // 板子抛异常不该把 BotSpawnService 的绑定链路一起带崩（与 Bukkit 事件同口径）。
        Board throwing = new CollectingBoard() {
            @Override
            public void onEvent(GameEvent event) {
                throw new IllegalStateException("board-boom");
            }

            @Override
            public String id() {
                return "throwing-board";
            }
        };
        BoardRegistryImpl registry = new BoardRegistryImpl(fakePlugin(), null, null);
        registry.active = new BoardSessionImpl(registry, fakePlugin(), throwing, null, null,
                RoundSpec.builder().build(), "test-session-2");
        try {
            registry.onBotReady("Bot_A", "glad-bot_a");
        } catch (RuntimeException e) {
            fail("板子异常被吞掉才对（只记 warning）: " + e);
        }
    }
}
