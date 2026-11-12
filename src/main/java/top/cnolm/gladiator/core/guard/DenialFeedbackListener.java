package top.cnolm.gladiator.core.guard;

import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.projectiles.ProjectileSource;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.core.GladiatorServiceImpl;

/**
 * 观测式规则反馈（A3）：<b>不依赖任何其它插件的 API</b>，只从 Bukkit 事件里学习限制。
 *
 * <p>监听器一律 {@code priority = MONITOR, ignoreCancelled = false}——只有这样才能看到
 * 「别的插件把它取消了」这个事实。判定口径：
 * <ul>
 *   <li>{@link EntityDamageByEntityEvent} 被取消 且 攻击方是 Bot（含弹射物的 shooter）
 *       → {@code action=attack, reason=damage-cancelled}；</li>
 *   <li>{@link PlayerInteractEvent} 被取消 → {@code action=use}；</li>
 *   <li>{@link BlockPlaceEvent} 被取消 → {@code action=place}；</li>
 *   <li>{@link BlockBreakEvent} 被取消 → {@code action=dig}。</li>
 * </ul>
 *
 * <p>本监听器<strong>只读</strong>：不取消、不修改任何事件；也不判断「是不是领地」——
 * 谁取消的、为什么取消都与 Gladiator 无关，只要是「Bot 被拒了」就记一次 deny。
 */
public final class DenialFeedbackListener implements Listener {

    private final GladiatorServiceImpl service;

    public DenialFeedbackListener(GladiatorServiceImpl service) {
        this.service = service;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = false)
    public void onDamage(EntityDamageByEntityEvent event) {
        if (!event.isCancelled()) {
            return;
        }
        Player attacker = resolveAttacker(event.getDamager());
        if (attacker == null) {
            return;
        }
        Bot bot = service.findBotByUniqueId(attacker.getUniqueId());
        if (bot == null) {
            return;
        }
        Entity victim = event.getEntity();
        service.handleActionDenied(bot, "attack",
                victim == null ? attacker.getLocation() : victim.getLocation(),
                "damage-cancelled");
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = false)
    public void onInteract(PlayerInteractEvent event) {
        if (!event.isCancelled()) {
            return;
        }
        // 纯挥臂（LEFT_CLICK_AIR 且没点到方块）不算「使用被拒」：golem 的 swingArm 会主动发一个
        // 这样的 interact 事件，若照单全收，每次普通攻击都会顺带记一条假的 use deny。
        if (event.getAction() == org.bukkit.event.block.Action.LEFT_CLICK_AIR
                && event.getClickedBlock() == null) {
            return;
        }
        Bot bot = service.findBotByUniqueId(event.getPlayer().getUniqueId());
        if (bot == null) {
            return;
        }
        service.handleActionDenied(bot, "use", event.getPlayer().getLocation(), "interact-cancelled");
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = false)
    public void onPlace(BlockPlaceEvent event) {
        if (!event.isCancelled()) {
            return;
        }
        Bot bot = service.findBotByUniqueId(event.getPlayer().getUniqueId());
        if (bot == null) {
            return;
        }
        service.handleActionDenied(bot, "place", event.getBlock().getLocation(), "place-cancelled");
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = false)
    public void onBreak(BlockBreakEvent event) {
        if (!event.isCancelled()) {
            return;
        }
        Bot bot = service.findBotByUniqueId(event.getPlayer().getUniqueId());
        if (bot == null) {
            return;
        }
        service.handleActionDenied(bot, "dig", event.getBlock().getLocation(), "break-cancelled");
    }

    /** 直接攻击取攻击者；弹射物取 shooter（1.8 的 {@code Projectile#getShooter}）。 */
    private static Player resolveAttacker(Entity damager) {
        if (damager instanceof Player) {
            return (Player) damager;
        }
        if (damager instanceof Projectile) {
            ProjectileSource shooter = ((Projectile) damager).getShooter();
            if (shooter instanceof Player) {
                return (Player) shooter;
            }
        }
        return null;
    }
}
