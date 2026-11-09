package top.cnolm.gladiator.core;

import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerRespawnEvent;

/**
 * 标准 Bukkit 事件 → 状态机 / 指标（G-301 / G-304）。
 *
 * <p>全部为观察者：<strong>不取消事件、不修改伤害或方块结果</strong>，
 * 也不读取任何竞技场插件状态。死亡/重生在 MONITOR 阶段处理，保证其他插件已裁决完毕。
 */
public final class PlayerStateListener implements Listener {

    private final GladiatorServiceImpl service;

    public PlayerStateListener(GladiatorServiceImpl service) {
        this.service = service;
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Player player = event.getEntity();
        if (player != null) {
            service.handleDeath(player);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onRespawn(PlayerRespawnEvent event) {
        Player player = event.getPlayer();
        if (player != null) {
            service.handleRespawn(player);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player)) {
            return;
        }
        Player victim = (Player) event.getEntity();
        Player attacker = null;
        if (event instanceof EntityDamageByEntityEvent) {
            org.bukkit.entity.Entity damager = ((EntityDamageByEntityEvent) event).getDamager();
            if (damager instanceof Player) {
                attacker = (Player) damager;
            }
        }
        service.recordDamage(victim, attacker, event.getFinalDamage());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlockPlace(BlockPlaceEvent event) {
        service.recordBlockChange(event.getPlayer(), true);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlockBreak(BlockBreakEvent event) {
        service.recordBlockChange(event.getPlayer(), false);
    }
}
