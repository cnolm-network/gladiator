package top.cnolm.sample.probe;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerRespawnEvent;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.GladiatorAPI;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.Users;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;
import top.cnolm.gladiator.api.event.GladiatorStateChangedEvent;
import top.cnolm.gladiator.example.arena.DuelMode;

import java.util.ArrayList;
import java.util.List;

/**
 * A9 探针：竞技场「取消伤害 / 手动重生」时状态机是否仍迁移。
 *
 * <p>两种后端都能跑：
 * <ul>
 *   <li><b>runtime 后端</b>（推荐）：mineflayer 真实登录，Bot 在 {@code PlayerList} 里，
 *       会有真实的 {@code PlayerDeathEvent} / {@code PlayerRespawnEvent}。
 *       用 {@code /caller a9 normal A9_R1} 形式<b>接管</b>一只已由
 *       {@code /gladiator spawn A9_R1} 生成的 Bot（不再重复 spawn）。</li>
 *   <li><b>golem 后端</b>：不传名字时自己 {@code spawnBot}（名称自动生成）。</li>
 * </ul>
 *
 * <p>三种场景：
 * <ul>
 *   <li>{@link Scenario#NORMAL}：{@code player.damage(100)} 致死 → 等客户端/服务端自然重生，
 *       期望 {@code ALIVE→DEAD→RESPAWNING→ALIVE}；若 60 tick 内没有重生，调用方兜底调
 *       {@code spigot().respawn()} 并如实记录；</li>
 *   <li>{@link Scenario#CANCEL_DAMAGE}：LOWEST 取消 {@code EntityDamageEvent}，
 *       打 3 次致死量，期望玩家不死、状态机<b>不</b>迁到 DEAD（即跟着真实世界走）；</li>
 *   <li>{@link Scenario#MANUAL_RESPAWN}：致死当 tick 立刻 {@code spigot().respawn()}，
 *       记录是否产生 {@code PlayerRespawnEvent} 与 {@code RESPAWNING→ALIVE}。</li>
 * </ul>
 *
 * <p>监听器在结束时一定注销（{@link HandlerList#unregisterAll}），不污染后续测试。
 */
public final class A9Probe implements Listener {

    public enum Scenario {
        NORMAL,
        CANCEL_DAMAGE,
        MANUAL_RESPAWN
    }

    private static final int SPAWN_TIMEOUT = 400;
    private static final int ATTACH_TIMEOUT = 120;
    private static final float KILL_DAMAGE = 100.0F;
    /** 击杀后等待状态机走完的 tick 上限。 */
    private static final int RESPAWN_TIMEOUT = 120;
    /** 致命伤害后若仍存活，重试次数与间隔（CANCEL_DAMAGE 与兜底都用）。 */
    private static final int RETRY_MAX = 3;
    private static final int RETRY_GAP = 15;
    /** 首次可击杀的等待 tick：覆盖 EntityPlayer 构造里的 60 tick 无敌。 */
    private static final int INVULNERABLE_WAIT = 80;
    private static final int NATURAL_RESPAWN_WAIT = 60;

    private final Plugin plugin;
    private final GladiatorAPI api;
    private final CommandSender sender;
    private final Scenario scenario;
    private final Location spawn;
    private String requestedName;
    private final ProbeTimeline timeline = new ProbeTimeline();

    private final List<String> transitions = new ArrayList<String>();
    private final List<String> failures = new ArrayList<String>();
    private final List<String> notes = new ArrayList<String>();

    private String botName;
    private Bot bot;
    private User user;
    private Player player;
    private BukkitTask task;

    private boolean attached;
    private boolean started;
    private boolean killStarted;
    private boolean deadSeen;
    private boolean respawnEventSeen;
    private boolean callerRespawned;
    private boolean cancelDamaged;
    private boolean fallStarted;
    private boolean fallLanded;
    private int fallLandedTick = -1;
    private int damageAttempts;
    private int killTick = -1;
    private int deathTick = -1;
    private int killAt = -1;
    private int spawnRequestedAt = -1;

    public A9Probe(Plugin plugin, GladiatorAPI api, CommandSender sender, Location spawn,
                   Scenario scenario, String requestedName) {
        this.plugin = plugin;
        this.api = api;
        this.sender = sender;
        this.spawn = spawn;
        this.scenario = scenario;
        this.requestedName = requestedName;
    }

    public void start() {
        timeline.start(plugin);
        if (requestedName != null && !requestedName.isEmpty()) {
            botName = requestedName;
            Player online = Bukkit.getPlayerExact(botName);
            if (online != null) {
                attach(online);
            } else {
                notes.add("指定 Bot 尚未在线，等待上线：" + botName);
            }
        } else {
            String suffix = String.valueOf(System.currentTimeMillis() % 100000L);
            String tag = scenario.name().toLowerCase().replace("_", "");
            botName = "a9" + tag.substring(0, Math.min(3, tag.length())) + suffix;
            if (botName.length() > 16) {
                botName = botName.substring(0, 16);
            }
            spawnOwnBot();
        }
        Bukkit.getPluginManager().registerEvents(this, plugin);
        task = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
            @Override
            public void run() {
                step();
            }
        }, 1L, 1L);
    }

    private void spawnOwnBot() {
        spawnRequestedAt = timeline.tick();
        api.spawnBot(BotSpec.of(botName, "a9"));
        timeline.add("spawnBot " + botName + "（自动生成名）");
    }

    private void attach(Player online) {
        attached = true;
        player = online;
        user = Users.of(online);
        timeline.add("接管已在线 Bot " + botName + " uuid=" + online.getUniqueId());
        online.teleport(spawn);
        timeline.add("teleport → " + fmt(spawn));
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onReady(GladiatorBotReadyEvent event) {
        if (bot != null || !event.getBot().getName().equals(botName)) {
            return;
        }
        bot = event.getBot();
        user = bot;
        player = event.getPlayer();
        timeline.add("Ready " + botName + " player=" + (player == null ? "null" : player.getName()));
        if (player == null) {
            failures.add("Ready 事件 getPlayer() 为 null");
            finish();
            return;
        }
        player.teleport(spawn);
        timeline.add("teleport → " + fmt(spawn));
    }

    /**
     * CANCEL_DAMAGE 场景：LOWEST 取消该 Bot 的伤害。
     *
     * <p>1.8.8 的 {@code PlayerDeathEvent} 不是 {@code Cancellable}（1.9+ 才是），
     * "取消死亡"只能通过取消致死伤害实现；{@code Player#damage(double)} 经
     * {@code EntityLiving#damageEntity → d()} 派发 {@code EntityDamageEvent(CUSTOM)}，可取消。
     */
    @EventHandler(priority = EventPriority.LOWEST)
    public void onDamageForCancel(EntityDamageEvent event) {
        if (scenario != Scenario.CANCEL_DAMAGE || player == null) {
            return;
        }
        if (!(event.getEntity() instanceof Player)) {
            return;
        }
        Player victim = (Player) event.getEntity();
        if (!victim.getUniqueId().equals(player.getUniqueId())) {
            return;
        }
        event.setCancelled(true);
        cancelDamaged = true;
        timeline.add("EntityDamageEvent(LOWEST) setCancelled(true) cause=" + event.getCause()
                + " damage=" + event.getDamage());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        if (player == null || event.getEntity() == null) {
            return;
        }
        if (event.getEntity().getUniqueId().equals(player.getUniqueId())) {
            deathTick = timeline.tick();
            timeline.add("PlayerDeathEvent 实际触发（未被取消，" + scenario + "）");
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onRespawn(PlayerRespawnEvent event) {
        if (player == null || event.getPlayer() == null) {
            return;
        }
        if (event.getPlayer().getUniqueId().equals(player.getUniqueId())) {
            respawnEventSeen = true;
            timeline.add("PlayerRespawnEvent 触发 @ " + fmt(event.getRespawnLocation())
                    + (deathTick > 0 ? "（死亡后 " + (timeline.tick() - deathTick) + " tick）" : ""));
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onState(GladiatorStateChangedEvent event) {
        if (player == null || !event.getUser().getUniqueId().equals(player.getUniqueId())) {
            return;
        }
        String step = event.getFrom() + "→" + event.getTo();
        transitions.add(step);
        timeline.add("StateChanged " + step);
        if (event.getTo() == UserState.DEAD) {
            deadSeen = true;
        }
    }

    private void step() {
        try {
            if (player == null) {
                if (requestedName != null && !attached) {
                    Player online = Bukkit.getPlayerExact(requestedName);
                    if (online != null) {
                        attach(online);
                        return;
                    }
                    if (timeline.tick() > ATTACH_TIMEOUT) {
                        notes.add("指定 Bot " + requestedName + " 未上线，改为 spawnBot 兜底");
                        botName = requestedName;
                        spawnOwnBot();
                        requestedName = null;
                    }
                    return;
                }
                if (spawnRequestedAt >= 0 && timeline.tick() - spawnRequestedAt > SPAWN_TIMEOUT) {
                    failures.add("超时 " + SPAWN_TIMEOUT + " tick 没有 Ready");
                    finish();
                }
                return;
            }
            if (!started) {
                started = true;
                timeline.add("pushGame(DuelMode, " + botName + ")");
                try {
                    String session = api.pushGame(new DuelMode(), user).getSessionId();
                    timeline.add("session=" + session + " stateOf=" + api.stateOf(user)
                            + " hp=" + player.getHealth());
                } catch (RuntimeException ex) {
                    // 同一个 Bot 连着跑多个场景时，上一轮会话还在（Gladiator 没有"结束会话"的 API），
                    // 这里复用已有会话继续测状态机，并如实记录。
                    notes.add("pushGame 失败（复用已有会话）：" + ex.getMessage());
                    timeline.add("复用已有会话，stateOf=" + api.stateOf(user));
                }
                killAt = timeline.tick() + INVULNERABLE_WAIT;
                return;
            }
            int tick = timeline.tick();
            if (scenario == Scenario.CANCEL_DAMAGE) {
                // 关键事实（实测）：1.8.8 的 Player#damage(double) 走 Entity.damageEntity，
                // **不派发 EntityDamageEvent**，所以取消不了。要测「取消致死伤害」必须用
                // 真正会派发事件、且可取消的伤害源 —— 这里用原版摔落伤害（DamageSource.FALL）。
                if (!fallStarted && tick >= killAt) {
                    fallStarted = true;
                    killStarted = true;
                    killTick = tick;
                    final double fallY = spawn.getY() + 56.0D;
                    timeline.add("自由落体：y=" + fallY + " → 地面（落差约 56 格，落地伤害 > 20，必然致死）");
                    player.teleport(new Location(spawn.getWorld(),
                            spawn.getX() + 0.5D, fallY, spawn.getZ() + 0.5D));
                    timeline.add("teleport 完成，等待落地");
                    return;
                }
                if (fallStarted && !fallLanded) {
                    if (player.isOnGround() || player.isDead() || tick > killTick + 160) {
                        fallLanded = true;
                        fallLandedTick = tick;
                        timeline.add("已落地/结束：y=" + String.format("%.2f", player.getLocation().getY())
                                + " hp=" + player.getHealth() + " isDead=" + player.isDead()
                                + " cancelled=" + cancelDamaged + " stateOf=" + api.stateOf(user));
                    }
                    return;
                }
                if (fallLanded && tick >= fallLandedTick + 20) {
                    finish();
                }
                return;
            }
            if (!killStarted && tick >= killAt) {
                doDamage("首次");
                if (scenario == Scenario.MANUAL_RESPAWN) {
                    // 「死亡后立刻」：同一 tick 内调用，抢在客户端自己的 respawn 包之前。
                    doCallerRespawn(true);
                }
                return;
            }
            if (killStarted && !deadSeen && !player.isDead() && damageAttempts < RETRY_MAX
                    && tick >= killTick + RETRY_GAP) {
                notes.add("上一发未致死（hp=" + player.getHealth() + "），重试");
                doDamage("重试 " + (damageAttempts + 1));
                return;
            }
            if (scenario == Scenario.MANUAL_RESPAWN) {
                if (respawnEventSeen && hasTransition("RESPAWNING→ALIVE")) {
                    finish();
                    return;
                }
                if (killTick > 0 && tick > killTick + RESPAWN_TIMEOUT) {
                    failures.add("手动重生后 " + RESPAWN_TIMEOUT + " tick 未完成 RESPAWNING→ALIVE");
                    finish();
                }
                return;
            }
            // NORMAL：优先让真实客户端/服务端自己重生；超时才由调用方兜底。
            if (hasTransition("RESPAWNING→ALIVE")) {
                finish();
                return;
            }
            if ((deadSeen || player.isDead()) && !callerRespawned
                    && deathAnchor() > 0 && tick > deathAnchor() + NATURAL_RESPAWN_WAIT) {
                notes.add(NATURAL_RESPAWN_WAIT + " tick 内没有自然重生，调用方兜底 respawn（结果如实记录）");
                doCallerRespawn(false);
                return;
            }
            if (killTick > 0 && tick > killTick + RESPAWN_TIMEOUT) {
                failures.add("击杀后 " + RESPAWN_TIMEOUT + " tick 仍未走完状态迁移");
                finish();
            }
        } catch (RuntimeException ex) {
            failures.add("探针异常: " + ex);
            finish();
        }
    }

    private int deathAnchor() {
        return deathTick > 0 ? deathTick : killTick;
    }

    private void doDamage(String label) {
        killStarted = true;
        damageAttempts++;
        if (killTick < 0) {
            killTick = timeline.tick();
        }
        timeline.add("damage(" + (int) KILL_DAMAGE + ") " + label + "：hp=" + player.getHealth()
                + " stateOf=" + api.stateOf(user)
                + (scenario == Scenario.CANCEL_DAMAGE ? "（LOWEST 会取消）" : ""));
        try {
            player.damage(KILL_DAMAGE);
        } catch (RuntimeException ex) {
            failures.add("damage 抛异常: " + ex);
        }
        timeline.add("damage 返回：hp=" + player.getHealth() + " isDead=" + player.isDead()
                + " stateOf=" + api.stateOf(user) + " cancelled=" + cancelDamaged);
    }

    private void doCallerRespawn(boolean withPushPlace) {
        callerRespawned = true;
        if (withPushPlace) {
            timeline.add("调用方 pushPlace(bot) 试探");
            if (bot != null) {
                try {
                    api.pushPlace(bot);
                    notes.add("pushPlace 成功");
                } catch (RuntimeException ex) {
                    notes.add("pushPlace 抛异常（死后 Bot 不可用，符合预期）：" + ex.getClass().getSimpleName());
                }
            } else {
                notes.add("attach 模式没有 Bot 句柄，跳过 pushPlace");
            }
        }
        timeline.add("调用 spigot().respawn()（hp=" + player.getHealth()
                + " isDead=" + player.isDead() + "）");
        try {
            player.spigot().respawn();
        } catch (RuntimeException ex) {
            failures.add("spigot().respawn() 抛异常: " + ex);
        }
        timeline.add("respawn 返回：hp=" + player.getHealth() + " isDead=" + player.isDead()
                + " stateOf=" + api.stateOf(user));
    }

    private boolean hasTransition(String step) {
        return transitions.contains(step);
    }

    private boolean ordered(String first, String second) {
        int a = transitions.indexOf(first);
        int b = transitions.indexOf(second);
        return a >= 0 && b >= 0 && a < b;
    }

    private void finish() {
        if (!killStarted && failures.isEmpty()) {
            failures.add("探针在造成伤害前就结束了，本轮不构成有效验证");
        }
        if (task != null) {
            task.cancel();
            task = null;
        }
        HandlerList.unregisterAll(this);
        timeline.stop();

        boolean pass;
        List<String> expected = new ArrayList<String>();

        if (scenario == Scenario.CANCEL_DAMAGE) {
            expected.add("（假设）取消致死伤害后状态机仍迁移到 DEAD");
            boolean alive = player != null && !player.isDead() && player.getHealth() > 0.0D;
            boolean noPhantomDeath = !hasTransition("ALIVE→DEAD");
            boolean effective = fallStarted && cancelDamaged;
            if (!effective) {
                failures.add("没有走到「取消致死伤害」这一步（fallStarted=" + fallStarted
                        + " cancelDamaged=" + cancelDamaged + "），本轮不构成有效验证");
            }
            notes.add("玩家存活=" + alive + " hp=" + (player == null ? "?" : player.getHealth())
                    + " 出现 ALIVE→DEAD=" + !noPhantomDeath + " 实际取消过伤害=" + cancelDamaged);
            if (!effective) {
                pass = false;
            } else if (alive) {
                pass = noPhantomDeath;
                notes.add("结论：假设「取消死亡后状态机仍迁移」不成立；状态机跟随真实世界，保持 "
                        + (user == null ? "?" : api.stateOf(user)));
                if (!noPhantomDeath) {
                    failures.add("玩家未死亡但状态机已迁到 DEAD（幻影死亡）");
                }
            } else {
                pass = hasTransition("ALIVE→DEAD");
                notes.add("玩家确实死亡 —— LOWEST 取消没有生效（检查事件是否被其它插件提前放行）");
                if (!pass) {
                    failures.add("玩家已死亡但状态机没有迁移到 DEAD");
                }
            }
        } else {
            expected.add("ALIVE→DEAD");
            expected.add("DEAD→RESPAWNING");
            expected.add("RESPAWNING→ALIVE");
            boolean ok = ordered("ALIVE→DEAD", "DEAD→RESPAWNING")
                    && ordered("DEAD→RESPAWNING", "RESPAWNING→ALIVE");
            if (!ok) {
                failures.add("状态序列不满足 " + expected);
            }
            if (!respawnEventSeen) {
                notes.add("未观测到 PlayerRespawnEvent（状态机可能靠其它路径回到 ALIVE）");
            }
            pass = ok && failures.isEmpty();
        }

        List<String> out = new ArrayList<String>();
        out.add("§e[A9/" + scenario + "] 状态机探针：§f" + (pass ? "PASS" : "FAIL"));
        out.add("§7后端：" + (attached ? "runtime（接管已登录玩家）" : "由探针 spawnBot")
                + "  bot=" + botName);
        out.add("§7期望：" + String.join(" → ", expected));
        out.add("§7实际：" + (transitions.isEmpty() ? "（无状态迁移）" : String.join(" → ", transitions)));
        out.add("§7玩家：isDead=" + (player == null ? "?" : String.valueOf(player.isDead()))
                + " hp=" + (player == null ? "?" : String.valueOf(player.getHealth()))
                + " finalState=" + (user == null ? "?" : String.valueOf(api.stateOf(user)))
                + " PlayerRespawnEvent=" + respawnEventSeen);
        for (String note : notes) {
            out.add("§7· " + note);
        }
        out.add("§7—— 时间线 ——");
        for (String line : timeline.lines()) {
            out.add("§7" + line);
        }
        for (String failure : failures) {
            out.add("§c! " + failure);
        }
        for (String line : out) {
            sender.sendMessage(line);
            plugin.getLogger().info(line.replaceAll("§.", ""));
        }
    }

    private static String fmt(Location location) {
        if (location == null || location.getWorld() == null) {
            return "-";
        }
        return String.format("%s %.2f,%.2f,%.2f", location.getWorld().getName(),
                location.getX(), location.getY(), location.getZ());
    }
}
