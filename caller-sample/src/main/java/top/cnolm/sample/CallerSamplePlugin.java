package top.cnolm.sample;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.plugin.java.JavaPlugin;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.GladiatorProvider;
import top.cnolm.gladiator.example.MockArenaCaller;
import top.cnolm.gladiator.example.arena.DuelCaller;
import top.cnolm.betterNameTag.api.BetterNameTagAPI;
import top.cnolm.betterNameTag.api.NameTagProvider;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * G-506 的验收载体：一个**独立**插件，只通过 Gladiator 的公开 API 与事件工作。
 *
 * <p>它和 Gladiator 的关系只有两条：编译期依赖 jar、运行期 {@code depend: [Gladiator]}。
 * Gladiator 的源码、plugin.yml、配置都不需要为它改动——这正是"换插件只改调用方"的验证点。</p>
 *
 * <p>两个示例调用方分属不同包、用不同的 {@code Mode}：</p>
 * <ul>
 *   <li>{@code arena} → {@link MockArenaCaller} + {@code DeathmatchMode}（死斗，不重生）</li>
 *   <li>{@code duel}  → {@link DuelCaller} + {@code DuelMode}（决斗，允许重生、5 分钟时限）</li>
 * </ul>
 *
 * <p>命令：</p>
 * <pre>
 * /caller arena &lt;username&gt; [x y z]   注册 A 调用方并 spawn 一个 Bot
 * /caller duel &lt;userA&gt; &lt;userB&gt;      注册 B 调用方并 spawn 两个 Bot
 * /caller reset                     注销当前调用方
 * /caller info                      打印 API 版本与 Gladiator 就绪状态
 * /caller tag &lt;player&gt; &lt;role&gt;     给玩家/ Bot 打上彩色斜体角色标签（trained/rusher/aimbot/...）
 * /caller tag &lt;player&gt; clear       移除角色标签
 * /caller tagcolor &lt;role&gt; &lt;color&gt; 改某个角色标签的颜色（&amp;0–&amp;f / §0–§f）并写回 config.yml
 * /caller tagcolor &lt;role&gt; reset    清除该角色的颜色覆盖（回落到内置默认）
 * </pre>
 */
public final class CallerSamplePlugin extends JavaPlugin {

    /** 角色 → 头顶标签前缀（都是斜体；训练方固定红色）。 */
    private static final Map<String, String> ROLE_TAGS = new LinkedHashMap<String, String>();
    /** 实际使用的标签表：内置默认值 + config.yml 的 tags 覆盖（可热重载）。 */
    private final Map<String, String> tags = new LinkedHashMap<String, String>();

    static {
        ROLE_TAGS.put("trained", "\u00a7c\u00a7o[训练] ");
        ROLE_TAGS.put("rusher", "\u00a76\u00a7o[Rush] ");
        ROLE_TAGS.put("aimbot", "\u00a7b\u00a7o[Aim] ");
        ROLE_TAGS.put("scripted", "\u00a7a\u00a7o[Script] ");
        ROLE_TAGS.put("wander", "\u00a7e\u00a7o[Wander] ");
        ROLE_TAGS.put("random", "\u00a7d\u00a7o[Random] ");
        ROLE_TAGS.put("idle", "\u00a77\u00a7o[Idle] ");
    }

    private GladiatorAPI api;
    private Listener active;
    private String activeName;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        loadTags();
        api = GladiatorProvider.get();
        if (api == null) {
            // depend 只保证加载顺序；服务本身仍可能注册失败。
            getLogger().warning("找不到 GladiatorAPI：请确认 Gladiator 已启用。");
        } else {
            getLogger().info("接入 Gladiator api=" + api.getApiVersion() + " ready=" + api.isReady());
        }
    }

    @Override
    public void onDisable() {
        unregisterActive();
        api = null;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sender.sendMessage("用法: /" + label + " <arena|duel|reset|info|tag> ...");
            return true;
        }
        String sub = args[0].toLowerCase();
        if ("info".equals(sub)) {
            sender.sendMessage("api=" + (api == null ? "null" : api.getApiVersion())
                    + " ready=" + (api != null && api.isReady())
                    + " active=" + (activeName == null ? "none" : activeName));
            return true;
        }
        if ("reset".equals(sub)) {
            String previous = activeName == null ? "none" : activeName;
            unregisterActive();
            sender.sendMessage("已注销调用方: " + previous);
            return true;
        }
        if ("tag".equals(sub)) {
            return handleTag(sender, label, args);
        }
        if ("tagcolor".equals(sub)) {
            return handleTagColor(sender, label, args);
        }
        if ("reloadtags".equals(sub)) {
            reloadConfig();
            loadTags();
            sender.sendMessage("已重载标签：" + String.join("|", tags.keySet()));
            return true;
        }
        if (api == null || !api.isReady()) {
            sender.sendMessage("Gladiator 尚未就绪。");
            return true;
        }

        if ("a6".equals(sub)) {
            Location spawn = parseLocation(args, 1);
            if (spawn == null) {
                sender.sendMessage("坐标不合法。");
                return true;
            }
            sender.sendMessage("§e[A6] 开始：Ready → 传送 → 套件 → pushGame 顺序探针（约 2–10 秒）");
            new top.cnolm.sample.probe.A6Probe(this, api, sender, spawn).start();
            return true;
        }
        if ("a8".equals(sub)) {
            Location spawn = parseLocation(args, 1);
            if (spawn == null) {
                sender.sendMessage("坐标不合法。");
                return true;
            }
            sender.sendMessage("§e[A8] 开始：两场独立对局的数据隔离探针（4 只 Bot，约 2–10 秒）");
            new top.cnolm.sample.probe.A8Probe(this, api, sender, spawn).start();
            return true;
        }
        if ("a9".equals(sub)) {
            if (args.length < 2) {
                sender.sendMessage("用法: /" + label + " a9 <normal|cancel-damage|manual-respawn> [botName] [x y z]");
                sender.sendMessage("§7不传 botName 时由探针自己 spawnBot（golem 后端）；"
                        + "传 botName 会接管已由 /gladiator spawn 登录的 runtime Bot。");
                return true;
            }
            top.cnolm.sample.probe.A9Probe.Scenario scenario;
            String mode = args[1].toLowerCase();
            if ("normal".equals(mode)) {
                scenario = top.cnolm.sample.probe.A9Probe.Scenario.NORMAL;
            } else if ("cancel-damage".equals(mode) || "cancel-death".equals(mode)) {
                // cancel-death 是旧名（golem 时代的「取消致死伤害」），语义相同，保留兼容。
                scenario = top.cnolm.sample.probe.A9Probe.Scenario.CANCEL_DAMAGE;
            } else if ("manual-respawn".equals(mode)) {
                scenario = top.cnolm.sample.probe.A9Probe.Scenario.MANUAL_RESPAWN;
            } else {
                sender.sendMessage("未知场景: " + mode + "（normal|cancel-damage|manual-respawn）");
                return true;
            }
            // /caller a9 normal A9_R1  → 接管已登录 Bot；多出来的三项才当坐标。
            String botName = null;
            int offset = 2;
            if (args.length >= 3 && !isNumber(args[2])) {
                botName = args[2];
                offset = 3;
            }
            Location spawn = parseLocation(args, offset);
            if (spawn == null) {
                sender.sendMessage("坐标不合法。");
                return true;
            }
            sender.sendMessage("§e[A9] 开始：" + scenario
                    + (botName == null ? "" : "，接管 " + botName) + "（约 3–12 秒）");
            new top.cnolm.sample.probe.A9Probe(this, api, sender, spawn, scenario, botName).start();
            return true;
        }
        if ("arena".equals(sub)) {
            if (args.length < 2) {
                sender.sendMessage("用法: /" + label + " arena <username> [x y z]");
                return true;
            }
            Location spawn = parseLocation(args, 2);
            if (spawn == null) {
                sender.sendMessage("坐标不合法。");
                return true;
            }
            unregisterActive();
            MockArenaCaller caller = new MockArenaCaller(spawn.getWorld(), spawn);
            Bukkit.getPluginManager().registerEvents(caller, this);
            active = caller;
            activeName = "arena(DeathmatchMode)";
            caller.spawnOne(args[1]);
            sender.sendMessage("已用 arena 调用方 spawn " + args[1]
                    + " @ " + fmt(spawn));
            return true;
        }

        if ("duel".equals(sub)) {
            if (args.length < 3) {
                sender.sendMessage("用法: /" + label + " duel <userA> <userB>");
                return true;
            }
            Location base = parseLocation(args, 3);
            if (base == null) {
                sender.sendMessage("坐标不合法。");
                return true;
            }
            Location spawnA = base.clone();
            Location spawnB = base.clone().add(4.0, 0.0, 0.0);
            unregisterActive();
            DuelCaller caller = new DuelCaller(this, api, spawnA, spawnB);
            caller.register();
            active = caller;
            activeName = "duel(DuelMode)";
            caller.start(args[1], args[2]);
            sender.sendMessage("已用 duel 调用方 spawn " + args[1] + " vs " + args[2]);
            return true;
        }

        sender.sendMessage("未知子命令: " + sub);
        return true;
    }

    /** /caller tag &lt;player&gt; &lt;role|clear&gt;：借 BetterNameTag 的 cover 前缀给实体上色。 */

    /** 内置默认值打底，再用 config.yml 的 tags 段覆盖；颜色用 & 转义。 */
    private void loadTags() {
        tags.clear();
        tags.putAll(ROLE_TAGS);
        org.bukkit.configuration.ConfigurationSection section = getConfig().getConfigurationSection("tags");
        if (section == null) {
            return;
        }
        for (String key : section.getKeys(false)) {
            String value = section.getString(key);
            if (value != null) {
                tags.put(key, org.bukkit.ChatColor.translateAlternateColorCodes('&', value));
            }
        }
    }

    private boolean handleTag(CommandSender sender, String label, String[] args) {
        if (args.length < 3) {
            sender.sendMessage("用法: /" + label + " tag <player> <" + String.join("|", tags.keySet()) + "|clear>");
            return true;
        }
        Player target = Bukkit.getPlayerExact(args[1]);
        if (target == null) {
            sender.sendMessage("找不到在线玩家: " + args[1]);
            return true;
        }
        BetterNameTagAPI nameTags = NameTagProvider.isAvailable() ? NameTagProvider.getAPI() : null;
        if (nameTags == null) {
            sender.sendMessage("BetterNameTag 未安装或未启用，无法设置头顶标签。");
            return true;
        }
        String role = args[2].toLowerCase();
        if ("clear".equals(role)) {
            nameTags.removeCover(target);
            nameTags.updatePlayer(target);
            sender.sendMessage("已清除 " + target.getName() + " 的角色标签。");
            return true;
        }
        String prefix = tags.get(role);
        if (prefix == null) {
            sender.sendMessage("未知角色: " + role + "（可用: " + String.join("|", tags.keySet()) + "|clear）");
            return true;
        }
        nameTags.setCoverPrefix(target, prefix);
        nameTags.updatePlayer(target);
        sender.sendMessage("已设置 " + target.getName() + " 的角色标签: " + role);
        return true;
    }


    /** /caller tagcolor &lt;role&gt; &lt;color|reset&gt;：Dashboard 训练前用它把脚本颜色同步到服务端标签。 */
    private boolean handleTagColor(CommandSender sender, String label, String[] args) {
        if (args.length < 3) {
            sender.sendMessage("用法: /" + label + " tagcolor <role> <&0-&f | §0-§f | reset>");
            return true;
        }
        String role = args[1].toLowerCase();
        if (!role.matches("[a-z0-9][a-z0-9_-]{0,23}")) {
            sender.sendMessage("角色名需为 1-24 位小写字母/数字/下划线/连字符： " + role);
            return true;
        }
        if ("reset".equalsIgnoreCase(args[2])) {
            getConfig().set("tags." + role, null);
            saveConfig();
            reloadConfig();
            loadTags();
            sender.sendMessage("已清除角色颜色覆盖: " + role);
            return true;
        }
        String spec = args[2].replace("&", "").replace("\u00a7", "");
        if (spec.length() != 1 || "0123456789abcdef".indexOf(Character.toLowerCase(spec.charAt(0))) < 0) {
            sender.sendMessage("颜色只允许 &0-&f / §0-§f 单色码，收到: " + args[2]);
            return true;
        }
        char code = Character.toLowerCase(spec.charAt(0));
        String text = labelText(role);
        String prefix = "\u00a7" + code + "\u00a7o" + text;
        // config.yml 用 & 转义，写回时把 § 换成 &，热重载后 loadTags() 会再翻译回来。
        getConfig().set("tags." + role, prefix.replace('\u00a7', '&'));
        saveConfig();
        reloadConfig();
        loadTags();
        sender.sendMessage("已设置角色 " + role + " 的颜色: §" + code
                + "（写回 config.yml 并已热重载）");
        return true;
    }

    /** 取某角色标签的文字部分（去掉颜色码）：优先内置默认文本，其次 config 现有值，最后用 id 拼。 */
    private String labelText(String role) {
        String raw = ROLE_TAGS.get(role);
        if (raw == null) {
            raw = tags.get(role);
        }
        if (raw == null) {
            return "[" + role + "] ";
        }
        String stripped = org.bukkit.ChatColor.stripColor(raw);
        if (stripped == null || stripped.trim().isEmpty()) {
            return "[" + role + "] ";
        }
        return stripped;
    }

    private void unregisterActive() {
        if (active != null) {
            HandlerList.unregisterAll(active);
            active = null;
            activeName = null;
        }
    }

    /** [x y z] 可省略；省略时用执行者（或世界出生点）当前位置。 */
    /** 用于把 `/caller a9 normal A9_R1` 与 `/caller a9 normal 0 4 0` 区分开。 */
    private static boolean isNumber(String text) {
        if (text == null || text.isEmpty()) {
            return false;
        }
        try {
            Double.parseDouble(text);
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    private Location parseLocation(String[] args, int offset) {
        if (args.length >= offset + 3) {
            try {
                double x = Double.parseDouble(args[offset]);
                double y = Double.parseDouble(args[offset + 1]);
                double z = Double.parseDouble(args[offset + 2]);
                return new Location(defaultWorld(), x, y, z);
            } catch (NumberFormatException ex) {
                return null;
            }
        }
        // 1.8.8 的 getOnlinePlayers() 返回 Collection<Player>（1.9+ 才改为数组）
        java.util.Iterator<? extends Player> it = Bukkit.getOnlinePlayers().iterator();
        Player player = it.hasNext() ? it.next() : null;
        if (player != null) {
            return player.getLocation();
        }
        World world = defaultWorld();
        return world == null ? null : world.getSpawnLocation();
    }

    private World defaultWorld() {
        World world = Bukkit.getWorlds().isEmpty() ? null : Bukkit.getWorlds().get(0);
        return world;
    }

    private static String fmt(Location loc) {
        return loc.getWorld().getName() + " " + loc.getX() + "," + loc.getY() + "," + loc.getZ();
    }
}
