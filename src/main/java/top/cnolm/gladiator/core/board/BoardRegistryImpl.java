package top.cnolm.gladiator.core.board;

import org.bukkit.Bukkit;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.ParticipantResolver;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.BoardRegistry;
import top.cnolm.gladiator.api.board.Boards;
import top.cnolm.gladiator.api.board.RoundSpec;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 板子注册表实现：持有已注册板子与「当前唯一一场」对局，负责起停调度。
 *
 * <p>同一时刻只允许一场对局：训练/对战的端口、玩家与记分都是全局唯一的，
 * 允许多场并行会让事件无法归属。
 *
 * <p>Bot 生命周期事件（{@link top.cnolm.gladiator.api.board.GameEventType#BOT_READY} /
 * {@link top.cnolm.gladiator.api.board.GameEventType#BOT_OFFLINE}）不来自 Bukkit，
 * 而来自 {@code core.BotSpawnService} 的绑定/失败链路：那边在另一个包里，只能走
 * {@link #notifyBotReady} / {@link #notifyBotOffline} 这两个静态入口；真正转发给板子的
 * {@link #onBotReady} / {@link #onBotOffline} 保持包内可见，避免外部代码绕过会话。
 */
public final class BoardRegistryImpl implements BoardRegistry {

    /**
     * 当前已启用的注册表（与 {@link Boards#install} 同生命周期）。
     *
     * <p>存在意义：{@code BotSpawnService} 拿不到本实例（它只持有 Plugin 与服务），
     * 而 Bot 就绪/失效必须从那条链路进来。未启用时这些静态入口静默丢弃。
     */
    private static volatile BoardRegistryImpl current;

    private final Plugin plugin;
    private final GladiatorAPI api;
    private final ParticipantResolver resolver;
    private final Map<String, Board> boards = new LinkedHashMap<String, Board>();
    private final AtomicLong seq = new AtomicLong();

    /** 当前对局；包内可见是为了单测能直接注入假会话（生产代码只经 {@link #start}/{@link #stop} 改写）。 */
    BoardSessionImpl active;
    /** 最近一场已结束的会话：{@link #active()} 清空后外部仍能读到最终比分与结束原因。 */
    BoardSessionImpl lastFinished;
    private BukkitTask ticker;
    private BoardEventBridge bridge;

    public BoardRegistryImpl(Plugin plugin, GladiatorAPI api, ParticipantResolver resolver) {
        this.plugin = plugin;
        this.api = api;
        this.resolver = resolver;
    }

    /** 插件 enable 时调用：装好静态入口、事件桥与 10Hz ticker。 */
    public void enable() {
        current = this;
        Boards.install(this);
        bridge = new BoardEventBridge(this);
        Bukkit.getPluginManager().registerEvents(bridge, plugin);
        ticker = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
            @Override
            public void run() {
                BoardSessionImpl session = active;
                if (session != null) {
                    session.onTick();
                }
            }
        }, 2L, 2L);
    }

    /** 插件 disable 时调用：停对局、停 ticker、清空注册表。 */
    public void disable() {
        stop("plugin-disable");
        if (ticker != null) {
            ticker.cancel();
            ticker = null;
        }
        boards.clear();
        Boards.install(null);
        current = null;
    }

    @Override
    public void register(Board board) {
        requireMainThread();
        if (board == null || board.id() == null || board.id().isEmpty()) {
            throw new IllegalArgumentException("板子 id 不能为空");
        }
        if (boards.containsKey(board.id())) {
            throw new IllegalArgumentException("板子 id 已注册: " + board.id());
        }
        boards.put(board.id(), board);
        plugin.getLogger().info("[board] 已注册板子: " + board.id());
    }

    @Override
    public void unregister(String boardId) {
        requireMainThread();
        Board board = boards.remove(boardId);
        if (board == null) {
            return;
        }
        if (active != null && active.boardId().equals(boardId)) {
            stop("board-unregistered");
        }
        plugin.getLogger().info("[board] 已注销板子: " + boardId);
    }

    @Override
    public Collection<String> ids() {
        return Collections.unmodifiableList(new ArrayList<String>(boards.keySet()));
    }

    @Override
    public Board get(String boardId) {
        return boards.get(boardId);
    }

    @Override
    public BoardContext active() {
        return active;
    }

    @Override
    public BoardContext start(String boardId, RoundSpec spec) {
        requireMainThread();
        Board board = boards.get(boardId);
        if (board == null) {
            throw new IllegalArgumentException("板子不存在: " + boardId);
        }
        if (active != null && !active.isFinished()) {
            throw new IllegalStateException("已有一场对局在进行中: " + active.sessionId());
        }
        String sessionId = boardId + "-" + System.currentTimeMillis();
        final BoardSessionImpl session = new BoardSessionImpl(this, plugin, board, api, resolver, spec, sessionId);
        active = session;
        bridge.attach(session);
        plugin.getLogger().info("[board] 对局开始: " + sessionId
                + " 局数=" + spec.getRounds() + " 每局=" + spec.getSecondsPerRound() + "s"
                + " 套件=" + spec.getKitId());
        try {
            board.onStart(session);
        } catch (RuntimeException e) {
            plugin.getLogger().warning("[board] onStart 抛异常，终止对局: " + e);
            session.finish("start-error");
        }
        return session;
    }

    @Override
    public void stop(String reason) {
        requireMainThread();
        BoardSessionImpl session = active;
        if (session == null) {
            return;
        }
        session.finish(reason);
        bridge.detach();
        active = null;
        plugin.getLogger().info("[board] 对局结束: " + reason);
    }

    /**
     * 会话自己结束（局数打满 / 板子调 {@code endMatch} / tick 抛异常）时由会话回调。
     *
     * <p>不清空 {@link #lastFinished}：外部（命令、dashboard）要在对局结束后仍能读到最终比分，
     * 所以这里只把会话搬进「最近一场」，并让下一场 {@link #start} 重新占用 {@link #active}。
     */
    void onSessionEnded(BoardSessionImpl session) {
        if (session == null) {
            return;
        }
        lastFinished = session;
        if (active == session) {
            active = null;
            if (bridge != null) {
                bridge.detach();
            }
        }
    }

    /** 最近一场已结束的会话；没有则 {@code null}。 */
    BoardSessionImpl lastFinished() {
        return lastFinished;
    }

    /** 最近一场已结束的会话（对外只读视图，供命令拼状态行）。 */
    public BoardContext lastFinishedSession() {
        return lastFinished;
    }

    // ------------------------------------------------------------ Bot 生命周期
    //
    // 与 Bukkit 事件（BoardEventBridge）分开：Bot 的「就绪/失效」不是服务器事件，
    // 而是 BotSpawnService 绑定（bind）与失败（fail）两条链路上的事实。
    // 两者都只在**有活动会话**时转发；没有会话就静默丢弃（对局没开，板子不该被打扰）。

    /**
     * Bot 就绪（跨包入口，供 {@code BotSpawnService#bind} 调用）。
     *
     * <p>为什么是静态：{@code BotSpawnService} 在 {@code core} 包，只持有 Plugin 与服务，
     * 拿不到本实例；而转发本身必须留在包内（见 {@link #onBotReady}）。
     */
    public static void notifyBotReady(String username, String botId) {
        BoardRegistryImpl impl = current;
        if (impl != null) {
            impl.onBotReady(username, botId);
        }
    }

    /** Bot 失效/未起来（跨包入口，供 {@code BotSpawnService} 的 quit 与 fail 两条路径调用）。 */
    public static void notifyBotOffline(String username, String botId, String reason) {
        BoardRegistryImpl impl = current;
        if (impl != null) {
            impl.onBotOffline(username, botId, reason);
        }
    }

    /** 把「Bot 就绪」转发给当前对局。 */
    void onBotReady(String username, String botId) {
        requireMainThread();
        BoardSessionImpl session = active;
        if (session == null) {
            return;
        }
        session.onBotReady(username, botId);
    }

    /** 把「Bot 失效/未起来」转发给当前对局；{@code reason} 会原样进事件 data。 */
    void onBotOffline(String username, String botId, String reason) {
        requireMainThread();
        BoardSessionImpl session = active;
        if (session == null) {
            return;
        }
        session.onBotOffline(username, botId, reason);
    }

    /** 事件计数：给事件桥用（seq 全局单调，便于外部按序去重）。 */
    long nextSeq() {
        return seq.incrementAndGet();
    }

    int roundOf(BoardSessionImpl session) {
        return session == null ? 0 : session.round();
    }

    /**
     * 主线程断言。
     *
     * <p>与 {@code GladiatorServiceImpl#requirePrimaryThread} 同口径：**没有服务端实例时放行**
     * （单测环境），只有真的跑在服务端上才要求主线程——否则任何离线单测都建不出会话。
     */
    private void requireMainThread() {
        if (Bukkit.getServer() != null && !Bukkit.isPrimaryThread()) {
            throw new IllegalStateException("BoardRegistry 只能在主线程调用");
        }
    }
}
