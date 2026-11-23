package top.cnolm.arena;

import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import top.cnolm.arena.handler.ArenaEventHandler;
import top.cnolm.arena.status.ArenaMatch;
import top.cnolm.arena.status.MatchManager;
import top.cnolm.gladiator.api.board.Board;
import top.cnolm.gladiator.api.board.BoardContext;
import top.cnolm.gladiator.api.board.BoardRegistry;
import top.cnolm.gladiator.api.board.Boards;
import top.cnolm.gladiator.api.board.RoundSpec;

/**
 * 板子插件入口。
 *
 * <p>这个类很薄：只做「注册板子 + 一条管理命令」。真正的玩法在 {@link FlatBoard}。
 * 你可以整体替换 {@link FlatBoard}，只要它实现 {@link Board}。
 */
public final class ArenaPlugin extends JavaPlugin {

    private FlatBoard board;
    private static ArenaPlugin instance;
    private static ArenaEventHandler eventHandler;

    @Override
    public void onEnable() {
        instance = this;
        BoardRegistry registry = Boards.registry();
        if (registry == null) {
            getLogger().severe("Gladiator 尚未就绪（Boards.registry() 为 null），板子不注册");
            return;
        }
        board = new FlatBoard(this, getConfig().getString("board.id", "flat-1v1"));
        registry.register(board);
        eventHandler = new ArenaEventHandler(this);
        getServer().getPluginManager().registerEvents(eventHandler, this);
        getLogger().info("板子已注册: " + board.id() + "（已注册: " + registry.ids() + "）");
    }

    @Override
    public void onDisable() {
        BoardRegistry registry = Boards.registry();
        if (registry != null && board != null) {
            registry.unregister(board.id());
        }
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        BoardRegistry registry = Boards.registry();
        if (registry == null) {
            sender.sendMessage("§cGladiator 未就绪");
            return true;
        }
        String sub = args.length == 0 ? "status" : args[0].toLowerCase();
        if ("status".equals(sub)) {
            BoardContext active = registry.active();
            // 结束后的会话不会被自动清除（它的轮次已超过总局数），这里不能当成「进行中」显示，
            // 否则打完之后 status 会一直显示「第 N+1 局」。
            boolean running = active != null && !board.isFinished();
            sender.sendMessage("§7板子: §f" + registry.ids()
                    + " §7对局: §f" + (running
                    ? active.sessionId() + " 第 " + active.round() + "/" + active.spec().getRounds() + " 局"
                    : "无（上一场已结束）"));
            return true;
        }
        if ("stop".equals(sub)) {
            registry.stop("command");
            sender.sendMessage("§e对局已结束");
            return true;
        }
        if ("quit".equals(sub)) {
            if (!(sender instanceof Player)) {
                sender.sendMessage("§c该命令只能由玩家执行");
                return true;
            }
            Player player = (Player) sender;
            ArenaMatch<?, ?> match = MatchManager.getPlayerMatch(player.getUniqueId());
            if (match == null) {
                sender.sendMessage("§c你当前不在对战中");
                return true;
            }
            match.eliminatePlayer(player.getUniqueId(), null, EliminateReason.ABANDON);
            sender.sendMessage("§e你已退出对战");
            return true;
        }
        if ("start".equals(sub)) {
            if (args.length < 3) {
                sender.sendMessage("§c用法: /arena start <player> <bot> [局数] [每局秒数]");
                return true;
            }
            int rounds = args.length > 3 ? parseInt(args[3], 3) : 3;
            int seconds = args.length > 4 ? parseInt(args[4], 120) : 120;
            RoundSpec spec = RoundSpec.builder()
                    .rounds(rounds)
                    .secondsPerRound(seconds)
                    .kitId("classic")
                    .extra("player", args[1])
                    .extra("bot", args[2])
                    .build();
            try {
                registry.start(board.id(), spec);
                sender.sendMessage("§a对局开始: " + args[1] + " vs " + args[2]
                        + "（" + rounds + " 局 × " + seconds + "s）");
            } catch (RuntimeException e) {
                sender.sendMessage("§c开局失败: " + e.getMessage());
            }
            return true;
        }
        sender.sendMessage("§7/arena start <player> <bot> [局数] [每局秒数] | quit | stop | status");
        return true;
    }

    private static int parseInt(String raw, int def) {
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            return def;
        }
    }

    public static ArenaPlugin getInstance() {
        return instance;
    }

    public static ArenaEventHandler getEventHandler() {
        return eventHandler;
    }
}
