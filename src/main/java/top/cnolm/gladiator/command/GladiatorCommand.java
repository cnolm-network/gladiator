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
import top.cnolm.gladiator.bridge.LinkServer;
import top.cnolm.gladiator.core.BotHandle;
import top.cnolm.gladiator.core.BotRegistry;
import top.cnolm.gladiator.core.GladiatorServiceImpl;
import top.cnolm.gladiator.core.SpectatorService;

import java.util.Locale;

/** 管理命令：只读取 core 状态，不承载游戏逻辑。 */
public final class GladiatorCommand implements CommandExecutor {

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
            case "spectate":
                handleSpectate(sender, label, args);
                return true;
            case "unspectate":
                handleUnspectate(sender, label);
                return true;
            default:
                sender.sendMessage("[Gladiator] 用法: /" + label
                        + " <status|spawn <username>|train <username>|list|game|analyze [session] [windowSec]"
                        + "|pvp <x> <y> <z>|spectate [bot]|unspectate|reload>");
                return true;
        }
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
                    + " [wtap=on|off] [autorod=on|off] [autoblock=on|off] [aimlock=on|off]");
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
                    default:
                        sender.sendMessage("[Gladiator] 未知配置项: " + key);
                        return;
                }
            } catch (IllegalArgumentException ex) {
                sender.sendMessage("[Gladiator] 配置项非法: " + raw + "（" + ex.getMessage() + "）");
                return;
            }
        }
        try {
            BotRequest request = api.spawnBot(
                    BotSpec.of(username, "default", null, profile.build()));
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

