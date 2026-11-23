package top.cnolm.arena;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
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
    /** 没给出生点时，双方沿 X 轴相对站位；单位格。 */
    private static final double DEFAULT_OFFSET = 4.0D;
    /** 开下一局前等待双方复活：每次重试间隔与最大次数（10 秒）。 */
    private static final long START_RETRY_TICKS = 10L;
    private static final int MAX_START_ATTEMPTS = 20;

    private final Plugin plugin;
    private final String id;

    private BoardContext ctx;
    private String player = "";
    private String bot = "";
    private int roundSeconds = 120;

    private FlatMatchManager manager;
    private ArenaTeam playerTeam;
    private ArenaTeam botTeam;
    private boolean roundActive;
    private boolean finished;
    private int startAttempts;
    private long roundStartTick;

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

    @Override
    public void onStart(BoardContext context) {
        this.ctx = context;
        // 同一个 Board 实例会被下一场对局复用：所有「上一场」的状态必须在这里清掉，
        // 否则 finished=true 会让新对局一局都不开（表现为「开始了但没有任何局」）。
        this.finished = false;
        this.roundActive = false;
        this.startAttempts = 0;
        this.manager = null;
        this.playerTeam = null;
        this.botTeam = null;
        RoundSpec spec = context.spec();
        this.player = spec.extra("player", "player");
        this.bot = spec.extra("bot", "bot");
        this.roundSeconds = spec.getSecondsPerRound();
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
        if (manager != null) {
            // 阶段由板子声明：倒计时 → 战斗；外部（命令/dashboard）靠它判断现在能不能算伤害。
            if (manager.isFighting()) {
                if (!"fighting".equals(ctx.phase())) {
                    ctx.phase("fighting");
                    roundStartTick = tick;
                }
                if ((tick - roundStartTick) / 10L >= roundSeconds) {
                    ctx.log("第 " + ctx.round() + " 局超时，按平局结算");
                    // 走 arena 自己的结算路径：全部淘汰 → 无人存活 → 平局。
                    manager.forceEnd();
                }
            } else if ("starting".equals(ctx.phase())) {
                ctx.phase("countdown");
            }
        }
    }

    @Override
    public void onStop(String reason) {
        if (ctx != null) {
            ctx.log("板子清理: " + reason);
        }
        roundActive = false;
        finished = true;
        manager = null;
        playerTeam = null;
        botTeam = null;
        ctx = null;
        player = "";
        bot = "";
    }

    // ------------------------------------------------------------------ 回合

    private void startRound() {
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
        if ((p.isDead() || p.getHealth() <= 0.0D || b.isDead() || b.getHealth() <= 0.0D)
                && startAttempts < MAX_START_ATTEMPTS) {
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
        startAttempts = 0;

        String kitId = ctx.spec().getKitId();
        if (kitId == null || kitId.trim().isEmpty()) {
            kitId = ArenaKits.CLASSIC;
        }

        Location playerSpawn = spawnFor(p, ctx.spec().getPlayerSpawn(), -DEFAULT_OFFSET);
        Location botSpawn = spawnFor(b, ctx.spec().getBotSpawn(), DEFAULT_OFFSET);
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

        manager = new FlatMatchManager(playerTeam, botTeam, kitId, ctx.spec().getShieldSeconds());
        manager.setResultListener(this::onRoundResult);
        ctx.log("第 " + ctx.round() + " 局套件已发：" + manager.kitId());
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
        manager = null;
        playerTeam = null;
        botTeam = null;

        final BoardContext session = ctx;
        ctx.endRound(outcome);
        // endRound 会自增轮次；打满后会话已经 endMatch，不能再准备下一局。
        if (session.round() > session.spec().getRounds()) {
            finished = true;
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

    /** 出生点：优先用 RoundSpec 给的坐标，否则以双方当前位置为基准左右分开。 */
    private Location spawnFor(Player target, double[] spawn, double offset) {
        if (spawn != null) {
            return new Location(target.getWorld(), spawn[0], spawn[1], spawn[2],
                    target.getLocation().getYaw(), target.getLocation().getPitch());
        }
        Location base = target.getLocation().clone();
        base.setX(base.getX() + offset);
        return base;
    }

    /** 1.8 没有「服务端设置重生点」的 API，只能让客户端下一次重生落在原地。 */
    private void anchorRespawn(Player target, Location spawn) {
        target.setBedSpawnLocation(spawn, true);
    }
}
