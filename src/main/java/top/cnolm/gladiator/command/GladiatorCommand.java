package top.cnolm.gladiator.command;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import top.cnolm.gladiator.GladiatorMain;
import top.cnolm.gladiator.api.Analysis;
import top.cnolm.gladiator.api.AnalysisQuery;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotProfile;
import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GameSession;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.TrainingSession;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.Users;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.MatchOutcome;
import top.cnolm.gladiator.api.board.RoundSpec;
import top.cnolm.gladiator.bridge.LinkServer;
import top.cnolm.gladiator.core.BotHandle;
import top.cnolm.gladiator.core.BotRegistry;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.gladiator.core.PlayerRespawn;
import top.cnolm.gladiator.core.PlayerTelemetry;
import top.cnolm.gladiator.core.SpectatorService;

import java.util.Locale;

/** 管理命令：只读取 core 状态，不承载游戏逻辑。 */
public final class GladiatorCommand implements CommandExecutor {

    /** {@code board frame} 的三段分隔符（一行一帧，dashboard 按它切开，G-719）。 */
    static final String FRAME_SEPARATOR = " | ";

    private final GladiatorMain plugin;

    public GladiatorCommand(GladiatorMain plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        String sub = args.length == 0 ? "status" : args[0].toLowerCase(Locale.ROOT);
        switch (sub) {
            case "status":
                GladiatorAPI api = plugin.getApi();
                GladiatorServiceImpl service = plugin.getService();
                sender.sendMessage("[Gladiator] version=" + plugin.getDescription().getVersion()
                        + " api=" + (api == null ? "unavailable" : api.getApiVersion())
                        + " ready=" + (api != null && api.isReady())
                        + " worldguard=" + (plugin.isWorldGuardPresent() ? "present" : "absent"));
                if (service != null) {
                    sender.sendMessage("[Gladiator] bots=" + service.getKnownBotCount()
                            + " games=" + service.getActiveGameCount()
                            + " training=" + service.getActiveTrainingCount()
                            + " pending=" + service.getPendingSpawnCount()
                            + " spawn=" + plugin.getSpawnMode());
                }
                LinkServer link = plugin.getLinkServer();
                sender.sendMessage("[Gladiator] link="
                        + (link == null ? "down" : link.getHost() + ":" + link.getPort())
                        + " sessions=" + (link == null ? 0 : link.getSessions().size()));
                Runtime runtime = Runtime.getRuntime();
                long usedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024L * 1024L);
                long maxMb = runtime.maxMemory() / (1024L * 1024L);
                sender.sendMessage("[Gladiator] heap used=" + usedMb + "MB max=" + maxMb + "MB");
                return true;
            case "spawn":
                handleSpawn(sender, label, args);
                return true;
            case "train":
                handleTrain(sender, label, args);
                return true;
            case "list":
                handleList(sender);
                return true;
            case "game":
                handleGame(sender);
                return true;
            case "analyze":
                handleAnalyze(sender, label, args);
                return true;
            case "reload":
                handleReload(sender);
                return true;
            case "pvp":
                handlePvp(sender, label, args);
                return true;
            case "stats":
                handleStats(sender, label, args);
                return true;
            case "respawn":
                handleRespawn(sender, label, args);
                return true;
            case "spectate":
                handleSpectate(sender, label, args);
                return true;
            case "unspectate":
                handleUnspectate(sender, label);
                return true;
            case "board":
                handleBoard(sender, label, args);
                return true;
            default:
                sender.sendMessage("[Gladiator] 用法: /" + label
                        + " <status|spawn <username>|train <username>|list|game|analyze [session] [windowSec]"
                        + "|pvp <x> <y> <z>|stats <player>|respawn <player>|spectate [bot]|unspectate"
                        + "|board <status|frame|start <boardId> <player> <bot> <rounds> <seconds> [6 个坐标] [kit=<id>]|stop>|reload>");
                return true;
        }
    }

    /**
     * `/gladiator respawn <player>` 的路由（块 B）：{@code target == null} → 失败文案，
     * 否则按是否死亡返回「已请求 / 跳过」。抽成静态方法便于单测，不需要真实服务端。
     */
    static String respawnResponse(Player target) {
        return PlayerRespawn.request(target);
    }

    /**
     * `/gladiator stats <player>` 的路由（块 B）：{@code target == null} → 失败文案，
     * 否则单行遥测。抽成静态方法，单测可以直接断言两个分支，不需要真实服务端。
     */
    static String statsResponse(Player target) {
        return target == null ? PlayerTelemetry.OFFLINE_MESSAGE : PlayerTelemetry.format(target);
    }

    /** 按名字找在线玩家；没有服务端实例（单测/早期初始化）时按离线处理，不抛给调用方。 */
    private static Player findOnlinePlayer(String name) {
        try {
            return org.bukkit.Bukkit.getPlayerExact(name);
        } catch (RuntimeException ex) {
            return null;
        }
    }

    /**
     * 只读遥测：给 Dashboard 的每局采样与 Bot 运行时的 {@code target_health} 用。
     *
     * <p>输出刻意不带 {@code [Gladiator]} 前缀与颜色码：RCON 侧按 {@code key=value} 直接解析。
     * 不传送、不修改任何状态，也不含对局规则（谁赢由编排层判断）。
     */
    private void handleStats(CommandSender sender, String label, String[] args) {
        if (args.length < 2) {
            sender.sendMessage("[Gladiator] 用法: /" + label + " stats <player>");
            return;
        }
        sender.sendMessage(statsResponse(findOnlinePlayer(args[1])));
    }

    /**
     * 强制重生：把人机对战里停在死亡界面的玩家（或 Bot）拉回战场。
     *
     * <p>1.8 没有原版命令能代替客户端点「重生」，这里调 {@code Player#spigot().respawn()}。
     * 未死亡时返回「跳过」而不是报错——编排层每局都会对双方各发一次，活着的那个必然走到这个分支。
     * 输出同样保持纯净单行，方便 RCON 解析。
     */
    private void handleRespawn(CommandSender sender, String label, String[] args) {
        if (args.length < 2) {
            sender.sendMessage("[Gladiator] 用法: /" + label + " respawn <player>");
            return;
        }
        sender.sendMessage(respawnResponse(findOnlinePlayer(args[1])));
    }

    private static boolean flag(String value) {
        String v = value.toLowerCase(java.util.Locale.ROOT);
        if (v.equals("on") || v.equals("true") || v.equals("1") || v.equals("yes")) {
            return true;
        }
        if (v.equals("off") || v.equals("false") || v.equals("0") || v.equals("no")) {
            return false;
        }
        throw new IllegalArgumentException("期望 on/off，实际: " + value);
    }

    /** 诊断入口：以调用方身份走公开 API 发起一次 spawn，用于联调链路与 Ready 事件。 */
    private void handleSpawn(CommandSender sender, String label, String[] args) {
        GladiatorAPI api = plugin.getApi();
        if (api == null || !api.isReady()) {
            sender.sendMessage("[Gladiator] spawn 失败：API 尚未就绪");
            return;
        }
        if (args.length < 2) {
            sender.sendMessage("[Gladiator] 用法: /" + label
                    + " spawn <username> [cps=<n>] [jitter=<n>] [reaction=<ms>]"
                    + " [rjitter=<ms>] [reach=<n>] [model=<name>]"
                    + " [wtap=on|off] [autorod=on|off] [autoblock=on|off] [aimlock=on|off]"
                    + " [cheat=on|off]（reach>3 必须 cheat=on）");
            return;
        }
        String username = args[1];
        if (!username.matches("[A-Za-z0-9_]{3,16}")) {
            sender.sendMessage("[Gladiator] 非法用户名（3-16 位字母/数字/下划线）: " + username);
            return;
        }
        BotProfile.Builder profile = BotProfile.builder();
        for (int i = 2; i < args.length; i++) {
            String raw = args[i];
            int eq = raw.indexOf('=');
            if (eq <= 0) {
                sender.sendMessage("[Gladiator] 无法解析配置项: " + raw + "（期望 key=value）");
                return;
            }
            String key = raw.substring(0, eq).toLowerCase(java.util.Locale.ROOT);
            String value = raw.substring(eq + 1);
            try {
                switch (key) {
                    case "cps":
                        profile.cps(Double.parseDouble(value), 2.0);
                        break;
                    case "jitter":
                        profile.cps(15.0, Double.parseDouble(value));
                        break;
                    case "reaction":
                        profile.reaction(Double.parseDouble(value), 40.0);
                        break;
                    case "rjitter":
                        profile.reaction(220.0, Double.parseDouble(value));
                        break;
                    case "reach":
                        profile.reach(Double.parseDouble(value));
                        break;
                    case "model":
                        profile.model(value);
                        break;
                    case "wtap":
                        profile.wtap(flag(value));
                        break;
                    case "autorod":
                        profile.autoRod(flag(value));
                        break;
                    case "autoblock":
                        profile.autoBlock(flag(value));
                        break;
                    case "aimlock":
                        profile.aimLock(flag(value));
                        break;
                    case "cheat":
                    case "cheat.reach":
                        profile.cheatReach(flag(value));
                        break;
                    default:
                        sender.sendMessage("[Gladiator] 未知配置项: " + key);
                        return;
                }
            } catch (IllegalArgumentException ex) {
                sender.sendMessage("[Gladiator] 配置项非法: " + raw + "（" + ex.getMessage() + "）");
                return;
            }
        }
        BotProfile built;
        try {
            built = profile.build();
        } catch (IllegalArgumentException ex) {
            sender.sendMessage("[Gladiator] 配置非法: " + ex.getMessage());
            return;
        }
        if (built.isCheatReach()) {
            // 超限 reach 必须留痕：日志里出现这一行，任何“这个 Bot 真强”的结论都得打折。
            plugin.getLogger().warning("[cheat] reach 超限=" + built.getReach()
                    + "（仅对照实验）user=" + username);
        }
        try {
            BotRequest request = api.spawnBot(
                    BotSpec.of(username, "default", null, built));
            sender.sendMessage("[Gladiator] spawn 已受理 request=" + request.getRequestId()
                    + " botId=" + request.getBotId() + " state=" + request.getState()
                    + " mode=" + plugin.getSpawnMode()
                    + ("golem".equals(plugin.getSpawnMode())
                            ? "（golem 进程内身体，无需登录）"
                            : "（等待运行时登录 + PlayerJoin）"));
        } catch (RuntimeException ex) {
            sender.sendMessage("[Gladiator] spawn 失败: " + ex.getMessage());
        }
    }

    /**
     * 诊断入口：把一个已就绪的 Bot 放进训练会话（调用方 API {@code pushPlace} 的命令行等价物）。
     * 训练环境（Layer 3）用它把 Bot 与 {@code TrainingSession} 对齐，便于 analyze 与会话生命周期。
     */
    private void handleTrain(CommandSender sender, String label, String[] args) {
        GladiatorAPI api = plugin.getApi();
        GladiatorServiceImpl service = plugin.getService();
        if (api == null || !api.isReady() || service == null) {
            sender.sendMessage("[Gladiator] train 失败：API 尚未就绪");
            return;
        }
        if (args.length < 2) {
            sender.sendMessage("[Gladiator] 用法: /" + label + " train <username>");
            return;
        }
        BotRegistry registry = service.getBotRegistry();
        Bot bot = registry == null ? null : registry.findByUsername(args[1]);
        if (bot == null) {
            sender.sendMessage("[Gladiator] train 失败：未找到 Bot（先 spawn 并等待 Ready）: " + args[1]);
            return;
        }
        try {
            api.pushPlace(bot);
            sender.sendMessage("[Gladiator] train 已开始 bot=" + bot.getBotId()
                    + " session=" + service.sessionIdOf(bot.getUniqueId()));
        } catch (RuntimeException ex) {
            sender.sendMessage("[Gladiator] train 失败: " + ex.getMessage());
        }
    }

    /** G-305：列出已登记 Bot（含未绑定完成的离线 UUID 占位）。 */    private void handleList(CommandSender sender) {
        GladiatorServiceImpl service = plugin.getService();
        if (service == null) {
            sender.sendMessage("[Gladiator] list 失败：服务未启用");
            return;
        }
        BotRegistry registry = service.getBotRegistry();
        if (registry == null) {
            sender.sendMessage("[Gladiator] list 失败：Bot 注册表未接线");
            return;
        }
        sender.sendMessage("[Gladiator] bots=" + registry.size()
                + " pending=" + service.getPendingSpawnCount());
        for (BotHandle bot : registry.all()) {
            sender.sendMessage("  - " + bot.getBotId()
                    + " name=" + bot.getName()
                    + " uuid=" + bot.getUniqueId()
                    + " ready=" + bot.isReady()
                    + " state=" + bot.getState());
        }
    }

    /** G-305：活动对局与训练会话。 */
    private void handleGame(CommandSender sender) {
        GladiatorServiceImpl service = plugin.getService();
        if (service == null) {
            sender.sendMessage("[Gladiator] game 失败：服务未启用");
            return;
        }
        sender.sendMessage("[Gladiator] games=" + service.getActiveGameCount()
                + " training=" + service.getActiveTrainingCount());
        for (GameSession game : service.activeGames()) {
            StringBuilder names = new StringBuilder();
            for (User user : game.getUsers()) {
                names.append(names.length() == 0 ? "" : ",")
                        .append(user.getName()).append(user.isBot() ? "(bot)" : "");
            }
            sender.sendMessage("  [game] " + game.getSessionId()
                    + " mode=" + game.getMode().getId() + " users=" + names);
        }
        for (TrainingSession training : service.activeTrainings()) {
            sender.sendMessage("  [train] " + training.getSessionId()
                    + " bot=" + training.getBot().getBotId());
        }
    }

    /** G-305：指标快照（异步采样，返回不可变对象）。 */
    private void handleAnalyze(CommandSender sender, String label, String[] args) {
        GladiatorAPI api = plugin.getApi();
        if (api == null || !api.isReady()) {
            sender.sendMessage("[Gladiator] analyze 失败：API 尚未就绪");
            return;
        }
        String sessionId = args.length > 1 ? args[1] : null;
        int window = 0;
        if (args.length > 2) {
            try {
                window = Integer.parseInt(args[2]);
            } catch (NumberFormatException e) {
                sender.sendMessage("[Gladiator] 用法: /" + label + " analyze [sessionId] [windowSeconds]");
                return;
            }
        }
        AnalysisQuery query = sessionId == null
                ? AnalysisQuery.all()
                : AnalysisQuery.of(sessionId, window > 0 ? window : 0);
        Analysis analysis = api.analyze(query);
        sender.sendMessage("[Gladiator] analyze session=" + (analysis.getSessionId() == null
                ? "all" : analysis.getSessionId()) + " bots=" + analysis.getPerBot().size());
        for (Analysis.BotMetrics m : analysis.getPerBot()) {
            sender.sendMessage("  - " + m.getBotId()
                    + " alive=" + String.format(Locale.ROOT, "%.1fs", m.getAliveSeconds())
                    + " dmg=" + String.format(Locale.ROOT, "%.1f/%.1f",
                    m.getDamageDealt(), m.getDamageTaken())
                    + " blocks=" + m.getBlocksPlaced() + "/" + m.getBlocksBroken()
                    + " path=" + String.format(Locale.ROOT, "%.1fm", m.getPathMeters())
                    + " denied=" + m.getRuleDenials()
                    + " actions=" + m.getActionsAccepted() + "/" + m.getActionsDropped());
        }
    }

    /** G-305：重载 config.yml。链路/超时等启动期参数需重启生效。 */
    private void handleReload(CommandSender sender) {
        plugin.reloadConfig();
        sender.sendMessage("[Gladiator] config.yml 已重载；bridge.*/spawn-timeout 等启动期参数需重启生效");
        sender.sendMessage("[Gladiator] 当前 maxActionsPerSecond="
                + plugin.getConfig().getInt("training.max-actions-per-second", 10)
                + " spawnTimeout=" + plugin.getConfig().getLong("bridge.spawn-timeout-seconds", 30L));
    }

    /**
     * 只读观战：把玩家切到旁观者模式并送到 Bot 附近。
     *
     * <p>无参数时优先选训练会话里的 Bot，其次选任意已就绪的 Bot。</p>
     */
    private void handleSpectate(CommandSender sender, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("[Gladiator] spectate 需要玩家执行（提供客户端视角）");
            return;
        }
        Player player = (Player) sender;
        if (!player.hasPermission("gladiator.spectate")) {
            sender.sendMessage("[Gladiator] spectate 失败：缺少权限 gladiator.spectate");
            return;
        }
        SpectatorService spectator = plugin.getSpectatorService();
        if (spectator == null || !spectator.isEnabled()) {
            sender.sendMessage("[Gladiator] spectate 失败：观战已禁用（config.yml: spectate.enabled）");
            return;
        }
        GladiatorServiceImpl service = plugin.getService();
        BotRegistry registry = service == null ? null : service.getBotRegistry();
        Bot target;
        if (args.length > 1) {
            String key = args[1];
            target = registry == null ? null : registry.findByUsername(key);
            if (target == null && registry != null) {
                target = registry.findByBotId(key.toLowerCase(Locale.ROOT));
            }
            if (target == null) {
                sender.sendMessage("[Gladiator] spectate 失败：未找到 Bot: " + key
                        + "（可用 /" + label + " list 查看）");
                return;
            }
        } else {
            target = firstSpectatable(service, registry);
        }
        if (target == null) {
            sender.sendMessage("[Gladiator] spectate 失败：当前没有可观的 Bot（先 spawn 并等待 Ready）");
            return;
        }
        if (!spectator.enter(player, target)) {
            sender.sendMessage("[Gladiator] spectate 失败：Bot 没有在线实体: " + target.getBotId());
            return;
        }
        sender.sendMessage("[Gladiator] 观战中 bot=" + target.getBotId()
                + "（旁观者模式，仅观察）；退出: /" + label + " unspectate");
    }

    /** 退出观战：还原进入前的游戏模式与位置。 */
    private void handleUnspectate(CommandSender sender, String label) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("[Gladiator] unspectate 需要玩家执行");
            return;
        }
        SpectatorService spectator = plugin.getSpectatorService();
        if (spectator == null || !spectator.leave((Player) sender)) {
            sender.sendMessage("[Gladiator] 当前不在观战状态");
            return;
        }
        sender.sendMessage("[Gladiator] 已退出观战并还原游戏模式与位置");
    }

    /**
     * `/gladiator board <status|frame|start|stop>`：板子对局的观察与控制。
     *
     * <p>为什么放在 Gladiator 而不是让每块板子自己开命令：dashboard 只认一条命令通道，
     * 而「当前是哪场对局」是注册表的状态；板子自己的命令只管自己的规则。
     *
     * <p>{@code frame}（G-719）是 {@code status} 的「一次往返版」：多带双方 stats，供 dashboard
     * 每帧只发一条 RCON。
     */
    private void handleBoard(CommandSender sender, String label, String[] args) {
        top.cnolm.gladiator.core.board.BoardRegistryImpl registry = plugin.getBoardRegistry();
        if (registry == null) {
            sender.sendMessage("[Gladiator] board 失败：板子注册表未启用");
            return;
        }
        String action = args.length >= 2 ? args[1].toLowerCase(Locale.ROOT) : "status";
        if ("status".equals(action)) {
            sender.sendMessage("[Gladiator] "
                    + boardStatusLine(registry.active(), registry.lastFinishedSession()));
            return;
        }
        if ("frame".equals(action)) {
            // G-719：状态 + 双方 stats 合成一行，dashboard 一帧只发一次 RCON。
            sender.sendMessage("[Gladiator] "
                    + boardFrameLine(registry.active(), registry.lastFinishedSession()));
            return;
        }
        if ("stop".equals(action)) {
            BoardContext session = registry.active();
            if (session == null) {
                sender.sendMessage("[Gladiator] board 无进行中对局");
                return;
            }
            String sessionId = session.sessionId();
            registry.stop("command");
            sender.sendMessage("[Gladiator] board 已停止 session=" + sessionId + " reason=command");
            return;
        }
        if (!"start".equals(action)) {
            sender.sendMessage("[Gladiator] 用法: /" + label + " board <status|frame|start|stop>");
            return;
        }
        RoundSpec spec;
        try {
            spec = boardSpec(args);
        } catch (IllegalArgumentException e) {
            sender.sendMessage("[Gladiator] board 启动失败：" + e.getMessage());
            sender.sendMessage("[Gladiator] 用法: /" + label
                    + " board start <boardId> <player> <bot> <rounds> <seconds> [px py pz bx by bz] [kit=<id>]");
            return;
        }
        try {
            BoardContext session = registry.start(args[2], spec);
            sender.sendMessage("[Gladiator] board 已开始 session=" + session.sessionId()
                    + " rounds=" + spec.getRounds() + " seconds=" + spec.getSecondsPerRound()
                    + " phase=" + session.phase());
        } catch (IllegalArgumentException e) {
            sender.sendMessage("[Gladiator] board 启动失败：" + e.getMessage());
        } catch (IllegalStateException e) {
            sender.sendMessage("[Gladiator] board 启动失败：" + e.getMessage());
        }
    }

    /**
     * 拼 `/gladiator board status` 的单行状态：{@code key=value} 空格分隔，dashboard 直接按空格切开。
     *
     * <p>优先报进行中的对局；没有时退回「最近一场已结束」，这样外部在结束瞬间仍能读到最终比分
     * （否则会读到 {@code session=none} 而丢掉胜负）。抽成静态便于单测。
     *
     * <p>G-719 在**末尾**追加 {@code roundReason=}（最近一局的判定原因）与
     * {@code lastSeconds=}（最近一局时长，一位小数；无局时 {@code -1.0}）——只追加，
     * 既有字段顺序与语义一字不动，旧解析器按前缀取键不受影响。
     */
    static String boardStatusLine(BoardContext active, BoardContext lastFinished) {
        BoardContext session = active != null ? active : lastFinished;
        if (session == null) {
            return "session=none board=- round=0 rounds=0 phase=idle score=0:0 draws=0"
                    + " reason=none player=- bot=-"
                    + " roundReason=none lastSeconds=" + formatSeconds(-1.0D);
        }
        MatchOutcome summary = session.summary();
        RoundSpec spec = session.spec();
        boolean ended = "ended".equals(session.phase());
        // 结束时 round 已经自增到 rounds+1；对外报「打了几局」比报「下一局的局号」更直观。
        int shownRound = ended ? Math.min(session.round() - 1, spec.getRounds()) : session.round();
        return "session=" + session.sessionId()
                + " board=" + session.boardId()
                + " round=" + shownRound
                + " rounds=" + spec.getRounds()
                + " phase=" + session.phase()
                + " score=" + summary.getPlayerRounds() + ":" + summary.getBotRounds()
                + " draws=" + summary.getDraws()
                + " reason=" + (ended ? session.lastReason() : "running")
                + " player=" + safe(spec.extra("player", "-"))
                + " bot=" + safe(spec.extra("bot", "-"))
                + " roundReason=" + safe(session.roundReason())
                + " lastSeconds=" + formatSeconds(session.lastSeconds());
    }

    /** 秒数统一一位小数（{@code Locale.ROOT}：不能用逗号小数点，dashboard 按 "." 解析）。 */
    static String formatSeconds(double seconds) {
        return String.format(Locale.ROOT, "%.1f", seconds);
    }

    /**
     * `/gladiator board frame`：**一帧一次往返**（G-719）。
     *
     * <p>为什么要有：dashboard 每帧原本要发三条 RCON（{@code board status} + 两条
     * {@code stats}），每条往返约 56ms，实测只跑到 5–6Hz。这条命令把三段合成一行，
     * 一次往返即可完成「同步比分 + 采样双方状态」。
     *
     * <p>输出（单行，{@code " | "} 分段）：
     * <pre>{@code <status 行> | stats <玩家名> <stats 行> | stats <Bot名> <stats 行>}</pre>
     * 状态段与 {@link #boardStatusLine} **逐字节一致**；stats 段复用 {@code stats} 子命令的行格式。
     * 会话不存在（名字未知）或参与者不在线时，该段写作 {@code stats <名字> offline}，
     * 不报错也不省略段（dashboard 可以按固定三段解析）。
     */
    static String boardFrameLine(BoardContext active, BoardContext lastFinished) {
        String statusLine = boardStatusLine(active, lastFinished);
        BoardContext session = active != null ? active : lastFinished;
        String playerName = participantName(session, "player");
        String botName = participantName(session, "bot");
        return boardFrameLine(statusLine, playerName, onlineStatsOrNull(playerName),
                botName, onlineStatsOrNull(botName));
    }

    /**
     * 纯拼接版本（不碰 Bukkit，单测直接用）：{@code statsLine} 传 {@code null} 表示离线。
     */
    static String boardFrameLine(String statusLine, String playerName, String playerStats,
                                 String botName, String botStats) {
        return statusLine
                + FRAME_SEPARATOR + frameStatsSegment(playerName, playerStats)
                + FRAME_SEPARATOR + frameStatsSegment(botName, botStats);
    }

    /** 从会话参数里取参与者名；无会话/没填 → 空串（段里显示 {@code -}）。 */
    private static String participantName(BoardContext session, String key) {
        if (session == null || session.spec() == null) {
            return "";
        }
        String name = session.spec().extra(key, "");
        return name == null ? "" : name.trim();
    }

    /** 在线就取 {@code stats} 子命令那一行，离线/没服务端返回 null（段里写 offline）。 */
    private static String onlineStatsOrNull(String name) {
        if (name.isEmpty()) {
            return null;
        }
        Player target = findOnlinePlayer(name);
        return target == null ? null : statsResponse(target);
    }

    private static String frameStatsSegment(String name, String statsLine) {
        String who = name == null || name.isEmpty() ? "-" : name;
        String body = statsLine == null || statsLine.isEmpty() ? "offline" : statsLine;
        return "stats " + who + " " + body;
    }

    /**
     * 解析 {@code board start} 的参数：
     * {@code args = [board, start, boardId, player, bot, rounds, seconds, (px py pz bx by bz)?, (key=value)*]}。
     *
     * <p>6 个坐标要么全给要么全不给（只给一半会被当成 {@code key=value} 报错），
     * 这样不会出现「玩家传了、Bot 没传」的半配置状态。抽成静态便于单测。
     */
    static RoundSpec boardSpec(String[] args) {
        if (args.length < 7) {
            throw new IllegalArgumentException("缺少参数（需要 boardId player bot rounds seconds）");
        }
        int rounds = parseInt(args[5], "局数");
        int seconds = parseInt(args[6], "每局秒数");
        RoundSpec.Builder builder = RoundSpec.builder()
                .rounds(rounds)
                .secondsPerRound(seconds)
                .extra("player", args[3])
                .extra("bot", args[4]);
        int index = 7;
        if (args.length >= index + 6) {
            builder.playerSpawn(parseDouble(args[index], "玩家 x"),
                    parseDouble(args[index + 1], "玩家 y"),
                    parseDouble(args[index + 2], "玩家 z"));
            builder.botSpawn(parseDouble(args[index + 3], "Bot x"),
                    parseDouble(args[index + 4], "Bot y"),
                    parseDouble(args[index + 5], "Bot z"));
            index += 6;
        }
        for (; index < args.length; index++) {
            String token = args[index];
            int eq = token.indexOf('=');
            if (eq <= 0 || eq == token.length() - 1) {
                throw new IllegalArgumentException("无法识别参数：" + token + "（要写成 key=value）");
            }
            String key = token.substring(0, eq);
            String value = token.substring(eq + 1);
            if ("kit".equalsIgnoreCase(key)) {
                builder.kitId(value);
            } else {
                builder.extra(key, value);
            }
        }
        return builder.build();
    }

    private static String safe(String value) {
        return value == null || value.isEmpty() || value.indexOf(' ') >= 0 ? "-" : value;
    }

    private static int parseInt(String raw, String what) {
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(what + " 不是整数：" + raw);
        }
    }

    private static double parseDouble(String raw, String what) {
        try {
            return Double.parseDouble(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(what + " 不是数字：" + raw);
        }
    }

    /** 优先返回训练会话里的 Bot，其次任意已就绪的 Bot。 */
    private Bot firstSpectatable(GladiatorServiceImpl service, BotRegistry registry) {
        if (service != null) {
            for (TrainingSession training : service.activeTrainings()) {
                Bot bot = training.getBot();
                if (bot != null && bot.isReady()) {
                    return bot;
                }
            }
        }
        if (registry != null) {
            for (BotHandle handle : registry.all()) {
                if (handle.isReady()) {
                    return handle;
                }
            }
        }
        return null;
    }

    /** G-204 诊断：查询指定坐标的 PVP 判定来源与区域。 */
    private void handlePvp(CommandSender sender, String label, String[] args) {
        GladiatorAPI api = plugin.getApi();
        if (api == null || !api.isReady()) {
            sender.sendMessage("[Gladiator] pvp 失败：API 尚未就绪");
            return;
        }
        if (args.length < 4 || args.length > 5) {
            sender.sendMessage("[Gladiator] 用法: /" + label + " pvp <x> <y> <z> [botName]");
            sender.sendMessage("[Gladiator] 省略 botName 时以执行者身份查询；"
                    + "给 botName 时以该 Bot 身份查询（控制台可用，能命中观测式 deny 缓存）");
            return;
        }
        // 带 botName：以 Bot 身份查（观测式 deny 按 Bot 隔离，控制台没有玩家上下文）
        org.bukkit.World world = null;
        User user = null;
        if (args.length == 5) {
            String botName = args[4];
            Bot bot = plugin.getService() == null
                    ? null : plugin.getService().findBot(BotRegistry.botIdFor(botName));
            if (bot == null) {
                sender.sendMessage("[Gladiator] 找不到这个 Bot：" + botName);
                return;
            }
            user = bot;
            Player asPlayer = bot.asPlayer();
            world = asPlayer != null ? asPlayer.getWorld() : null;
        } else {
            if (!(sender instanceof Player)) {
                sender.sendMessage("[Gladiator] pvp 需要玩家执行，或追加 botName 以 Bot 身份查询");
                return;
            }
            Player player = (Player) sender;
            user = Users.of(player);
            world = player.getWorld();
        }
        if (world == null) {
            world = org.bukkit.Bukkit.getWorlds().isEmpty()
                    ? null : org.bukkit.Bukkit.getWorlds().get(0);
        }
        if (world == null) {
            sender.sendMessage("[Gladiator] pvp 失败：服务端没有可用世界");
            return;
        }
        try {
            double x = Double.parseDouble(args[1]);
            double y = Double.parseDouble(args[2]);
            double z = Double.parseDouble(args[3]);
            PvpVerdict verdict = api.queryPvp(
                    new org.bukkit.Location(world, x, y, z), user);
            sender.sendMessage("[Gladiator] pvp allowed=" + verdict.isAllowed()
                    + " source=" + verdict.getSource().name().toLowerCase(Locale.ROOT)
                    + " regions=" + verdict.getRegionIds()
                    + " box=" + verdict.getBox());
        } catch (NumberFormatException e) {
            sender.sendMessage("[Gladiator] 坐标必须是数字: /" + label + " pvp <x> <y> <z> [botName]");
        }
    }
}

