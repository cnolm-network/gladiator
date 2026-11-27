package top.cnolm.arena;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import top.cnolm.arena.simplify.FlatManagerSimplified;
import top.cnolm.arena.simplify.Recorder;
import top.cnolm.arena.status.ArenaTeam;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.GameEvent;
import top.cnolm.gladiator.api.board.RoundOutcome;
import top.cnolm.gladiator.api.board.RoundSpec;

/**
 * 平地 1v1 板子：把 arena 的回合引擎接到 Gladiator 的 {@link Board} 上。
 *
 * <p>分工很明确：
 * <ul>
 *   <li><b>本类</b>负责「一局对局的生命周期」——什么时候开一局、什么时候算打完、局数打满后收尾。
 *       它只认 {@link BoardContext}（轮次、tick、结束回调），不自己判死。</li>
 *   <li><b>{@link FlatMatchManager}</b> 负责「一局之内的一切」——套件、摆位、倒计时、受伤/击杀事件、
 *       淘汰、胜负、文字、清理。判死走 {@code PlayerDeathEvent → ArenaKillEvent}，不靠轮询血量。</li>
 * </ul>
 *
 * <p>这样做的原因：判死必须用事件（1.8 死亡到重生只有 2 tick，采样必漏），而事件已经在 arena 里
 * 包装好了；板子再自己做一套判定就会和 arena 的事件流打架。
 *
 * <p>除了「谁参加、在哪出生」要问 {@link BoardContext}（golem 假玩家不在 Bukkit PlayerList 里），
 * 其余一律直接用 Bukkit API——板子本来就是普通插件。
 */
final class FlatBoard implements Board {

    /** 局间缓冲：等客户端死亡界面/重生动画走完，再开下一局。 */
    private static final long ROUND_GAP_TICKS = 40L;
    /** G-721：开局后补发摆位的间隔与次数（bot 登录窗口内的 tp 会被客户端位置预测吃掉）。 */
    private static final int REASSERT_TICKS = 10;
    private static final int REASSERT_TIMES = 6;
    /** G-725：单次 teleport 允许的最大位移（格）；超过就分段传送，见 {@link #placeGradually}。 */
    private static final double STAGED_TELEPORT_DISTANCE = 48.0D;
    /** G-723：收尾回位的补发次数与间隔（同开局摆位）。 */
    private static final int RESTORE_TICKS = 10;
    private static final int RESTORE_TIMES = 3;
    /** 没给出生点时，双方沿 X 轴相对站位；单位格。 */
    private static final double DEFAULT_OFFSET = 4.0D;
    /** 开下一局前等待双方复活：每次重试间隔与最大次数（10 秒）。 */
    private static final long START_RETRY_TICKS = 10L;
    private static final int MAX_START_ATTEMPTS = 20;
    /** G-739：倒计时/开局卡死多久之后强制收尾（tick，20 tick = 1 秒）。 */
    private static final int STUCK_TIMEOUT_TICKS = 600;
    private int stuckTicks;
    /** 内置 Recorder 的位置采样周期（tick）；2 tick = 10Hz，和 dashboard 的录制同频。 */
    private static final int RECORD_PERIOD_TICKS = 2;
    /** 没给出生点时两队的默认半距（x,z）：水平间距 = 2·√(x²+z²) ≈ 7.2 格，进战斗距离。 */
    private static final int DEFAULT_SPAWN_VEC_X = 3;
    private static final int DEFAULT_SPAWN_VEC_Z = 2;

    private final Plugin plugin;
    private final String id;

    private BoardContext ctx;
    private String player = "";
    private String bot = "";
    private int roundSeconds = 120;

    /** 每场对局一个；跨局复用它，避免每局都注册一个新的 Listener。 */
    private FlatManagerSimplified simplified;
    private FlatMatchManager manager;
    private ArenaTeam playerTeam;
    private ArenaTeam botTeam;
    private boolean roundActive;
    private boolean finished;
    private int startAttempts;
    private long roundStartTick;
    /** 真正开打（phase=fighting）的墙钟时刻：本局时长按它算，不含倒计时与等复活。 */
    private long fightStartMs;
    /** 开局前的位置：对局结束后把双方送回去（否则会留在出生点，下一次对局距离越拉越远）。 */
    private Location playerStart;
    private Location botStart;

    FlatBoard(Plugin plugin, String id) {
        this.plugin = plugin;
        this.id = id;
    }

    @Override
    public String id() {
        return id;
    }

    /** 本场对局是否已收尾（供 /arena status 判断「进行中」还是「上一场已结束」）。 */
    boolean isFinished() {
        return ctx == null || finished;
    }

    /** 当前（或最近一局）的录像记录器；本场还没开过局时为 {@code null}。 */
    Recorder recorder() {
        return simplified == null ? null : simplified.getRecorder();
    }

    /** 最近一局的 arena matchId（导出文件名要用），没有则 {@code null}。 */
    String recordMatchId() {
        if (simplified == null || simplified.getLastMatchId() == null) {
            return null;
        }
        return simplified.getLastMatchId().toString();
    }

    @Override
    public void onStart(BoardContext context) {
        this.ctx = context;
        // 同一个 Board 实例会被下一场对局复用：所有「上一场」的状态必须在这里清掉，
        // 否则 finished=true 会让新对局一局都不开（表现为「开始了但没有任何局」）。
        this.finished = false;
        this.roundActive = false;
        this.startAttempts = 0;
        this.stuckTicks = 0;
        this.manager = null;
        this.playerTeam = null;
        this.botTeam = null;
        // 简化 API 每个对局一个实例（构造时注册 Listener），跨局复用、不重复注册。
        this.simplified = new FlatManagerSimplified();
        RoundSpec spec = context.spec();
        this.player = spec.extra("player", "player");
        this.bot = spec.extra("bot", "bot");
        this.roundSeconds = spec.getSecondsPerRound();
        Player startP = context.player(this.player);
        Player startB = context.player(this.bot);
        this.playerStart = startP == null ? null : startP.getLocation().clone();
        this.botStart = startB == null ? null : startB.getLocation().clone();
        startRound();
    }

    /**
     * 板子不自己判死：死亡/掉线/淘汰都由 {@link FlatMatchManager} 处理，结算经
     * {@link FlatMatchManager.ResultListener} 回流。这里只保留一条兜底——
     * 参与者在本局进行中从 Bukkit 视角彻底消失（既不在 PlayerList 也没有缓存引用）。
     */
    @Override
    public void onEvent(GameEvent event) {
        if (ctx == null || !roundActive) {
            return;
        }
        if (event.getType() == top.cnolm.gladiator.api.board.GameEventType.PLAYER_QUIT) {
            String name = event.getActor();
            if (player.equalsIgnoreCase(name)) {
                finishRound(RoundOutcome.bot("player-quit"));
            } else if (bot.equalsIgnoreCase(name)) {
                finishRound(RoundOutcome.player("bot-quit"));
            }
        }
    }

    @Override
    public void onTick(long tick) {
        if (ctx == null || !roundActive || finished) {
            return;
        }
        // G-728：开局阶段参与者掉线（运行时被杀/被判离线）时，对局必须收尾。
        // 否则会话永远停在 phase=starting，下一次 /gladiator board start 会被
        // 「已有一场对局在进行中」挡住（真机 23:50 复现）。
        if (manager == null && "starting".equals(ctx.phase())) {
            Player alivePlayer = ctx.player(player);
            Player aliveBot = ctx.player(bot);
            if (alivePlayer == null || aliveBot == null) {
                finishRound(alivePlayer == null
                        ? RoundOutcome.bot("player-offline")
                        : RoundOutcome.player("bot-offline"));
                return;
            }
        }
        if (manager != null) {
            // 阶段由板子声明：倒计时 → 战斗；外部（命令/dashboard）靠它判断现在能不能算伤害。
            if (manager.isFighting()) {
                stuckTicks = 0;
                if (!"fighting".equals(ctx.phase())) {
                    ctx.phase("fighting");
                    roundStartTick = tick;
                    fightStartMs = System.currentTimeMillis();
                }
                if ((tick - roundStartTick) / 10L >= roundSeconds) {
                    ctx.log("第 " + ctx.round() + " 局超时，按平局结算");
                    // 走 arena 自己的结算路径：全部淘汰 → 无人存活 → 平局。
                    manager.forceEnd();
                }
            } else if ("starting".equals(ctx.phase())) {
                ctx.phase("countdown");
            } else {
                // G-739：倒计时卡死看门狗。真机出现过「phase=countdown 挂了 3 分钟、
                // board status 一直显示 round=2」——arena 的倒计时任务被某个死亡事件
                // 吃掉后既不 fighting 也不结算。超过 30 秒仍未进入战斗就强制收尾。
                stuckTicks++;
                if (stuckTicks > STUCK_TIMEOUT_TICKS) {
                    ctx.log("第 " + ctx.round() + " 局卡在 " + ctx.phase() + "，强制判平局");
                    stuckTicks = 0;
                    finishRound(RoundOutcome.draw("stuck-" + ctx.phase()));
                }
            }
        } else if (roundActive && stuckTicks >= 0) {
            // 开局一直在等复活/等玩家，超过 60 秒说明流程断了，直接收尾避免整场挂住。
            stuckTicks++;
            if (stuckTicks > STUCK_TIMEOUT_TICKS * 2) {
                stuckTicks = 0;
                ctx.log("第 " + ctx.round() + " 局迟迟开不起来，判平局（no-manager）");
                finishRound(RoundOutcome.draw("no-manager"));
            }
        }
    }

    @Override
    public void onStop(String reason) {
        if (ctx != null) {
            ctx.log("板子清理: " + reason);
            // 对局结束后把双方送回开局位置：板子没有清理这一步时，玩家会一直留在出生点，
            // 下一场对局「距离开局位置越远、开局距离越大」，看起来像 Bot 不会靠近。
            restorePosition(ctx.player(player), playerStart, player);
            restorePosition(ctx.player(bot), botStart, bot);
        }
        roundActive = false;
        finished = true;
        manager = null;
        simplified = null;
        playerTeam = null;
        botTeam = null;
        ctx = null;
        player = "";
        bot = "";
        playerStart = null;
        botStart = null;
    }

    /** 对局结束后的归位；玩家已离线就跳过（不清空记录，交给下一场覆盖）。 */
    private void restorePosition(Player target, Location home, String label) {
        restorePosition(target, home, label, true);
    }

    /**
     * 归位（可静默）。
     *
     * <p>`announce=false` 用于「补发」的重复调用——真机上补发 3 次是对的（客户端重生流程会
     * 吃掉第一次 teleport），但日志没必要打三遍，否则一场对局结束刷 6 行同样的消息。
     */
    private void restorePosition(Player target, Location home, String label, boolean announce) {
        if (target == null || home == null || !target.isOnline()) {
            return;
        }
        target.teleport(home);
        target.setBedSpawnLocation(home, true);
        if (announce && ctx != null) {
            ctx.log("已把 " + label + " 送回开局位置");
        }
    }

    /**
     * 打满局数时归位。
     *
     * <p>延后 20 tick：最后一局往往以「某人被打死」结束，1.8 里死者还停在死亡界面，
     * 立刻 teleport 会被客户端重生流程覆盖回重生点。会话走 {@code stop} 时另有
     * {@link #onStop(String)} 兜底，两条路径都归位一次是幂等的。
     */
    private void scheduleRestore() {
        final BoardContext session = ctx;
        if (session == null) {
            return;
        }
        final Location homeP = playerStart;
        final Location homeB = botStart;
        // G-723：收尾回位同样会撞上「客户端还没走出重生流程」的窗口，单发一次会被吃掉，
        // 于是双方在客户端眼里继续停在原地、互相看不见（真机表现为 raw=1 / seen=0）。
        // 与开局摆位同一套办法：隔 10 tick 补发 3 次。
        for (int index = 1; index <= RESTORE_TIMES; index += 1) {
            final int attempt = index;
            plugin.getServer().getScheduler().runTaskLater(plugin, new Runnable() {
                @Override
                public void run() {
                    if (ctx != session) {
                        return;
                    }
                    restorePosition(ctx.player(player), homeP, player, attempt == 1);
                    restorePosition(ctx.player(bot), homeB, bot, attempt == 1);
                    if (attempt == RESTORE_TIMES) {
                        plugin.getLogger().info("[flat] 收尾回位已补发 " + attempt + " 次");
                    }
                }
            }, 20L + RESTORE_TICKS * attempt);
        }
    }

    /** 清掉中心周围 48 格内的地面掉落物（死亡掉落的套件）。 */
    private void clearGroundItems(Location center) {
        if (center == null || center.getWorld() == null) {
            return;
        }
        int removed = 0;
        for (Entity entity : center.getWorld().getNearbyEntities(center, 48.0D, 48.0D, 48.0D)) {
            if (entity instanceof Item) {
                entity.remove();
                removed++;
            }
        }
        if (removed > 0 && ctx != null) {
            ctx.log("清理地面掉落物 " + removed + " 个");
        }
    }

    // ------------------------------------------------------------------ 回合

    private void startRound() {
        try {
            doStartRound();
        } catch (RuntimeException error) {
            // G-727：之前这里没有任何兜底，一旦抛异常（例如 1.8 某些 Location 操作在
            // 非主线程/未加载区块上抛），对局就永远停在 phase=starting，外部只能看到
            // 「开始了但一局都不打」。现在至少把栈打出来并把这一局判成平局收尾。
            plugin.getLogger().log(java.util.logging.Level.SEVERE,
                    "[flat] 开局异常，判本局平局（draw/start-error）", error);
            finishRound(RoundOutcome.draw("start-error"));
        }
    }

    private void doStartRound() {
        if (ctx == null || finished) {
            return;
        }
        Player p = ctx.player(player);
        Player b = ctx.player(bot);
        if (p == null || b == null) {
            finishRound(p == null ? RoundOutcome.bot("player-offline") : RoundOutcome.player("bot-offline"));
            return;
        }
        // 上一局刚被打死的一方可能还停在死亡界面（1.8 的 respawn 是下一 tick 生效）；
        // 此时直接开下一局会在倒计时里被判死。等双方都活过来再开局，最多等 10 秒。
        boolean waiting = p.isDead() || p.getHealth() <= 0.0D
                || b.isDead() || b.getHealth() <= 0.0D;
        if (waiting && startAttempts < MAX_START_ATTEMPTS) {
            startAttempts++;
            final BoardContext session = ctx;
            plugin.getServer().getScheduler().runTaskLater(plugin, new Runnable() {
                @Override
                public void run() {
                    if (ctx == session && !finished) {
                        startRound();
                    }
                }
            }, START_RETRY_TICKS);
            return;
        }
        if (waiting) {
            // G-739：重试用尽仍有一方没活过来（死亡界面没人点重生 / 假玩家卡在死亡态）。
            // 之前这里会「硬开一局」，结果倒计时结束瞬间被判 one-team-left，甚至整场卡在
            // countdown 再也不推进（真机 06:29 复现：round=2 phase=countdown 挂了 3 分钟）。
            startAttempts = 0;
            ctx.log("第 " + ctx.round() + " 局无法开局：参与者 10 秒内未复活，判平局");
            finishRound(RoundOutcome.draw("not-ready"));
            return;
        }
        startAttempts = 0;

        String kitId = ctx.spec().getKitId();
        if (kitId == null || kitId.trim().isEmpty()) {
            kitId = ArenaKits.CLASSIC;
        }

        Location playerSpawn = spawnFor(p, ctx.spec().getPlayerSpawn(), -DEFAULT_OFFSET);
        Location botSpawn = spawnFor(b, ctx.spec().getBotSpawn(), DEFAULT_OFFSET);
        // 上一局死亡掉落的物品（护甲/剑/苹果）不清理会一直躺在地上，录像和复盘都会看到。
        // 只看一个中心（两侧出生点中点，半径 48）避免同一批掉落物被两侧各数一遍。
        clearGroundItems(playerSpawn.clone().add(botSpawn).multiply(0.5D));
        anchorRespawn(p, playerSpawn);
        anchorRespawn(b, botSpawn);

        playerTeam = new ArenaTeam("玩家", playerSpawn, ChatColor.RED);
        playerTeam.addPlayer(p.getUniqueId());
        botTeam = new ArenaTeam("Bot", botSpawn, ChatColor.BLUE);
        botTeam.addPlayer(b.getUniqueId());

        roundActive = true;
        roundStartTick = ctx.tick();
        ctx.phase("countdown");
        ctx.log("第 " + ctx.round() + " 局开始：" + player + " vs " + bot + " 套件=" + kitId
                + " 免伤=" + ctx.spec().getShieldSeconds() + "s");

        // 走用户的简化 API：它负责套件/免伤/摆位/录制/事件，本类只管「一局何时开始与结束」。
        manager = startSimplifiedMatch(p, b, playerSpawn, botSpawn, kitId);
        manager.setResultListener(this::onRoundResult);
        // 朝向必须在**简化 API 摆位之后**校正：`startMatch` 内部会用队伍出生点
        // 重新 teleport 一次，之前的朝向会被覆盖（2026-11-25 真机：`Glad_T1 yaw=0`）。
        // 动作空间 v2 的 `look_yaw` 是「每帧转多少度」，BC 策略几乎不主动转头，
        // 而真机出生方位随机 → 选 `move=forward` 会朝反方向跑开（G-721）。
        // 只修正出生朝向，不替模型做动作选择（ADR-020）。
        faceEachOther(p, b);
        ctx.log("第 " + ctx.round() + " 局套件已发：" + manager.kitId());
    }

    /**
     * 用 {@link FlatManagerSimplified} 起一局。
     *
     * <p>简化 API 只接受「中心点 + 半距」这种对称摆位（内部 {@code mid ± spawnVec}），
     * 所以把两侧出生点换算成中心与半距；换算后位置与上面算出的 playerSpawn/botSpawn 完全一致。
     */
    private FlatMatchManager startSimplifiedMatch(Player p, Player b,
                                                  Location playerSpawn, Location botSpawn, String kitId) {
        double dx = playerSpawn.getX() - botSpawn.getX();
        double dz = playerSpawn.getZ() - botSpawn.getZ();
        int halfX = (int) Math.round(Math.abs(dx) / 2.0D);
        int halfZ = (int) Math.round(Math.abs(dz) / 2.0D);
        if (halfX == 0 && halfZ == 0) {
            halfX = DEFAULT_SPAWN_VEC_X;
            halfZ = DEFAULT_SPAWN_VEC_Z;
        }
        Location mid = new Location(playerSpawn.getWorld(),
                (playerSpawn.getX() + botSpawn.getX()) / 2.0D,
                (playerSpawn.getY() + botSpawn.getY()) / 2.0D,
                (playerSpawn.getZ() + botSpawn.getZ()) / 2.0D);

        FlatManagerSimplified.StartParams params = new FlatManagerSimplified.StartParams(p, b)
                .mid(mid)
                .spawnVec(halfX, halfZ)
                .team(playerTeam, botTeam)
                .recordPeriod(RECORD_PERIOD_TICKS);
        simplified.start(params, kitId, ctx.spec().getShieldSeconds());
        // G-721：Bot 刚登录时客户端还没完成世界同步，此时下发的 tp 会被客户端位置预测吃掉
        // ——服务端坐标已经是出生点，客户端仍停在原地，双方互相看不见（观测里 entities 为空，
        // 策略在「盲」状态决策）。这里在开局后再补发几次（与真人链路 RESPAWN_PLACE_REFRESH 同思路）。
        reassertPositions(p, b, playerSpawn, botSpawn);
        return simplified.getFlatMatchManager();
    }

    /**
     * 重复下发摆位，直到客户端跟上（每 {@link #REASSERT_TICKS} tick 一次、最多
     * {@link #REASSERT_TIMES} 次）。只发 tp 与朝向，不碰装备与状态。
     */
    private void reassertPositions(final Player p, final Player b,
                                   final Location playerSpawn, final Location botSpawn) {
        for (int index = 1; index <= REASSERT_TIMES; index += 1) {
            final int attempt = index;
            plugin.getServer().getScheduler().runTaskLater(plugin, new Runnable() {
                @Override
                public void run() {
                    if (finished || !roundActive) {
                        return;
                    }
                    // G-725：一次 tp 跨上千格时，客户端来不及流式加载区块，mineflayer 的
                    // `entities` 会只剩自己（真机 22:11 的 Glad_D5 就是登录点离场地 1000+ 格）。
                    // 这里按「分段逼近」：先落到距目标不超过 STAGED_TELEPORT_DISTANCE 的位置，
                    // 最后一段才精确落点，给客户端时间把区块与实体表建起来。
                    placeGradually(p, playerSpawn, attempt);
                    placeGradually(b, botSpawn, attempt);
                    // G-723：朝向必须在**最后一次**摆位之后再校正——`teleport(Location)` 会用
                    // 传入 Location 的 yaw 覆盖角色朝向，先摆位后校朝向才是有效的。
                    if (attempt == REASSERT_TIMES) {
                        faceEachOther(p, b);
                        plugin.getLogger().info("[flat] 摆位已补发 " + attempt + " 次并校正朝向（客户端可能仍在同步世界）");
                    }
                }
            }, REASSERT_TICKS * attempt);
        }
    }

    /**
     * 分段摆位：距离超过 {@link #STAGED_TELEPORT_DISTANCE} 时，先传送到「朝目标前进
     * {@link #STAGED_TELEPORT_DISTANCE} 格」的位置；只有最后一次尝试才精确落点。
     *
     * <p>为什么要这样：1.8 客户端收到跨区块大跳的 `S08PacketPlayerPosLook` 后，会以自己
     * 的位置为基准重建实体表，期间服务端推来的 spawn 包会被丢弃，mineflayer 侧表现为
     * 「`bot.entities` 只剩自己、目标永远看不见」（真机 G-721/G-723 的间歇失明）。
     * 分两段（甚至多段）传送把单次位移压到实体追踪范围内，客户端每次都能正常加载。
     */
    private void placeGradually(Player target, Location spawn, int attempt) {
        if (target == null || spawn == null || !target.isOnline()) {
            return;
        }
        Location from = target.getLocation();
        if (from.getWorld() == null || !from.getWorld().equals(spawn.getWorld())) {
            target.teleport(spawn);
            return;
        }
        double distance = from.distance(spawn);
        if (attempt >= REASSERT_TIMES || distance <= STAGED_TELEPORT_DISTANCE) {
            target.teleport(spawn);
            return;
        }
        double ratio = STAGED_TELEPORT_DISTANCE / distance;
        Location step = new Location(
                spawn.getWorld(),
                from.getX() + (spawn.getX() - from.getX()) * (1.0 - ratio),
                spawn.getY(),
                from.getZ() + (spawn.getZ() - from.getZ()) * (1.0 - ratio),
                spawn.getYaw(),
                spawn.getPitch());
        target.teleport(step);
    }

    /** arena 的结算回流：赢家队伍 → Gladiator 的 RoundOutcome。 */
    private void onRoundResult(FlatMatchManager source, ArenaTeam winner, String reason) {
        if (!roundActive || source != manager) {
            return;
        }
        if (winner == playerTeam) {
            finishRound(RoundOutcome.player(reason));
        } else if (winner == botTeam) {
            finishRound(RoundOutcome.bot(reason));
        } else {
            finishRound(RoundOutcome.draw(reason));
        }
    }

    private void finishRound(RoundOutcome outcome) {
        if (!roundActive || ctx == null) {
            return;
        }
        roundActive = false;
        stuckTicks = 0;
        manager = null;
        playerTeam = null;
        botTeam = null;

        final BoardContext session = ctx;
        // 板子自己知道开打时刻，把真实局时长一起报上去（dashboard 逐局表格用，G-719）。
        double seconds = fightStartMs > 0L
                ? Math.max(0L, System.currentTimeMillis() - fightStartMs) / 1000.0D
                : -1.0D;
        fightStartMs = 0L;
        ctx.endRound(outcome, seconds);
        // endRound 会自增轮次；打满后会话已经 endMatch，不能再准备下一局。
        if (session.round() > session.spec().getRounds()) {
            finished = true;
            scheduleRestore();
            return;
        }
        plugin.getServer().getScheduler().runTaskLater(plugin, new Runnable() {
            @Override
            public void run() {
                if (ctx == session && !finished) {
                    startRound();
                }
            }
        }, ROUND_GAP_TICKS);
    }

    /**
     * 出生点：优先用 RoundSpec 给的坐标；否则以**世界出生点**为场地中心左右分开。
     *
     * <p>G-725（真机 22:11 / 23:40 两次）：原先用「双方各自当前位置 ±offset」，当两个 Bot
     * 的登录点相隔上千格时（1.8 的登录点由世界出生点决定，机器人重启一次就换一个地方），
     * 两侧出生点会各自留在原地 → 双方相距 1000+ 格，一局打不到人、只能超时平局。
     * 改用世界出生点做锚点后，无论从哪里登录，双方都会被摆到同一块场地；
     * 配合 {@link #placeGradually} 的分段传送，客户端也能正常建起实体表。
     */
    private Location spawnFor(Player target, double[] spawn, double offset) {
        if (spawn != null) {
            return new Location(target.getWorld(), spawn[0], spawn[1], spawn[2],
                    target.getLocation().getYaw(), target.getLocation().getPitch());
        }
        Location base = target.getWorld().getSpawnLocation().clone();
        base.setX(base.getX() + offset);
        return base;
    }

    /** 1.8 没有「服务端设置重生点」的 API，只能让客户端下一次重生落在原地。 */
    private void anchorRespawn(Player target, Location spawn) {
        target.setBedSpawnLocation(spawn, true);
    }

    /**
     * 把双方摆成面对面（只改朝向，不动位置）。
     *
     * <p>1.8 的 yaw 定义：0 = 朝 +Z，90 = 朝 -X，180 = 朝 -Z，-90 = 朝 +X。
     * pitch 取 0（平视），避免策略的 `look_pitch` 头被出生俯仰带偏。
     */
    private void faceEachOther(Player p, Player b) {
        Location pLoc = p.getLocation();
        Location bLoc = b.getLocation();
        if (pLoc.getWorld() == null || bLoc.getWorld() == null
                || !pLoc.getWorld().equals(bLoc.getWorld())) {
            return;
        }
        float playerYaw = yawTowards(pLoc, bLoc);
        float botYaw = yawTowards(bLoc, pLoc);
        Location playerTo = pLoc.clone();
        playerTo.setYaw(playerYaw);
        playerTo.setPitch(0.0F);
        p.teleport(playerTo);
        Location botTo = bLoc.clone();
        botTo.setYaw(botYaw);
        botTo.setPitch(0.0F);
        b.teleport(botTo);
    }

    /** 从 {@code from} 看向 {@code to} 的 yaw（1.8 约定：0=+Z、90=-X、180=-Z、-90=+X）。 */
    private float yawTowards(Location from, Location to) {
        double dx = to.getX() - from.getX();
        double dz = to.getZ() - from.getZ();
        return (float) Math.toDegrees(Math.atan2(-dx, dz));
    }
}
