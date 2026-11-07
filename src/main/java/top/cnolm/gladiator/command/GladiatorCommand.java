package top.cnolm.gladiator.command;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import top.cnolm.gladiator.GladiatorMain;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.core.GladiatorServiceImpl;

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
                            + " training=" + service.getActiveTrainingCount());
                }
                return true;
            case "list":
            case "game":
            case "analyze":
            case "reload":
                sender.sendMessage("[Gladiator] '" + sub + "' 尚未实现（M3 交付）");
                return true;
            default:
                sender.sendMessage("[Gladiator] 用法: /" + label + " <status|list|game|analyze|reload>");
                return true;
        }
    }
}
