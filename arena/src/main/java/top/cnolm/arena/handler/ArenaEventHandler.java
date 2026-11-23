package top.cnolm.arena.handler;

import org.bukkit.Material;
import org.bukkit.entity.Arrow;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerBucketEmptyEvent;
import org.bukkit.event.player.PlayerChangedWorldEvent;
import org.bukkit.event.player.PlayerDropItemEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerItemConsumeEvent;
import org.bukkit.event.player.PlayerPickupItemEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.projectiles.ProjectileSource;
import top.cnolm.arena.DeathReason;
import top.cnolm.arena.EliminateReason;
import top.cnolm.arena.FlatMatchManager;
import top.cnolm.arena.event.*;
import top.cnolm.arena.status.PlayerResolver;
import top.cnolm.gladiator.api.event.GladiatorBotReadyEvent;
import top.cnolm.gladiator.api.event.GladiatorBotRemovedEvent;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 完整事件处理器：监听原始 Bukkit 事件 → 转发包装事件 → 按需把取消回传给原始事件。
 *
 * <p>只对处于对局中的玩家生效；对局外的玩家完全不产生包装事件。
 */
public class ArenaEventHandler extends ArenaEventAdapter implements Listener {
    private static final long ATTRIBUTION_WINDOW_MS = 30_000L;

    private final Map<UUID, AttackRecord> lastAttacker = new ConcurrentHashMap<>();

    public ArenaEventHandler(JavaPlugin plugin) {
        super(plugin);
    }

    // ------------------------------------------------------------ 伤害 / 击杀

    /** 统一伤害入口；只监听父事件，避免与 EntityDamageByEntityEvent 重复触发。 */
    @EventHandler(priority = EventPriority.NORMAL)
    public void onDamage(EntityDamageEvent e) {
        if (!(e.getEntity() instanceof Player)) {
            return;
        }
        Player victim = (Player) e.getEntity();
        PlayerResolver.remember(victim);
        UUID matchId = matchIdOf(victim.getUniqueId());
        if (matchId == null) {
            return;
        }

        Player attacker = null;
        if (e instanceof EntityDamageByEntityEvent) {
            Entity damager = ((EntityDamageByEntityEvent) e).getDamager();
            if (damager instanceof Player) {
                attacker = (Player) damager;
                PlayerResolver.remember(attacker);
            } else if (damager instanceof Arrow) {
                ProjectileSource source = ((Arrow) damager).getShooter();
                Player shooter = source instanceof Player ? (Player) source : null;
                // 命中是既成结果，不可取消，直接转发即可
                call(new ArenaArrowHitEvent(victim, shooter, matchId, (Arrow) damager, e.getFinalDamage()));
            }
        }
        if (attacker != null) {
            lastAttacker.put(victim.getUniqueId(),
                    new AttackRecord(attacker.getUniqueId(), System.currentTimeMillis()));
        }

        ArenaDamageEvent wrapped = new ArenaDamageEvent(victim, attacker, matchId,
                e.getFinalDamage(), e.getCause());
        if (callCancellable(wrapped)) {
            e.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent e) {
        Player victim = e.getEntity();
        PlayerResolver.remember(victim);
        UUID matchId = matchIdOf(victim.getUniqueId());
        if (matchId == null) {
            return;
        }
        Player killer = resolveKiller(victim);
        DeathReason reason = killer == null ? mapCause(victim.getLastDamageCause()) : DeathReason.KILL;
        call(new ArenaKillEvent(victim, killer, matchId, reason));
        lastAttacker.remove(victim.getUniqueId());
    }

    // ------------------------------------------------------------ 物品使用

    /**
     * 右键交互：按物品拆成多种语义事件。
     *
     * <p>取消物品类事件后必须 {@code updateInventory()}：1.8 客户端会预测性扣减手中物品，
     * 服务端取消若不主动同步，物品数量会在 x / x-1 之间反复跳动。
     */
    @EventHandler(priority = EventPriority.NORMAL)
    public void onInteract(PlayerInteractEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        ItemStack item = e.getItem();
        if (matchId == null || item == null || e.isCancelled()) {
            return;
        }

        Material material = item.getType();
        boolean cancelled;
        if (isSplashPotion(item)) {
            cancelled = callCancellable(new ArenaUseSplashPotionEvent(player, matchId, item));
        } else if (isThrowable(material)) {
            cancelled = callCancellable(new ArenaThrowEvent(player, matchId, item));
        } else if (isSword(material)) {
            cancelled = callCancellable(new ArenaSwordBlockEvent(player, matchId, item));
        } else if (isDrink(material)) {
            cancelled = callCancellable(new ArenaDrinkEvent(player, matchId, item));
        } else if (material.isEdible()) {
            cancelled = callCancellable(new ArenaEatEvent(player, matchId, item));
        } else {
            return;
        }

        if (cancelled) {
            e.setCancelled(true);
            player.updateInventory();
        }
    }

    /** 消耗类事件同样按固体/液体重组，取消后同步物品栏避免跳动。 */
    @EventHandler(priority = EventPriority.NORMAL)
    public void onConsume(PlayerItemConsumeEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        ItemStack item = e.getItem();
        boolean cancelled = isDrink(item.getType())
                ? callCancellable(new ArenaDrinkEvent(player, matchId, item))
                : callCancellable(new ArenaEatEvent(player, matchId, item));
        if (cancelled) {
            e.setCancelled(true);
            player.updateInventory();
        }
    }

    /** 倒桶放置流体，取消后同步物品栏避免桶数量跳动。 */
    @EventHandler(priority = EventPriority.NORMAL)
    public void onBucketEmpty(PlayerBucketEmptyEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        ArenaPlaceFluidEvent wrapped = new ArenaPlaceFluidEvent(
                player, matchId, e.getBlockClicked(), e.getBucket());
        if (callCancellable(wrapped)) {
            e.setCancelled(true);
            player.updateInventory();
        }
    }

    // ------------------------------------------------------------ 方块 / 物品

    @EventHandler(priority = EventPriority.NORMAL)
    public void onBlockPlace(BlockPlaceEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        if (callCancellable(new ArenaBlockPlaceEvent(player, matchId, e.getBlock()))) {
            e.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.NORMAL)
    public void onBlockBreak(BlockBreakEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        if (callCancellable(new ArenaBlockBreakEvent(player, matchId, e.getBlock()))) {
            e.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.NORMAL)
    public void onDrop(PlayerDropItemEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        if (callCancellable(new ArenaItemDropEvent(player, matchId, e.getItemDrop()))) {
            e.setCancelled(true);
            player.updateInventory();
        }
    }

    @EventHandler(priority = EventPriority.NORMAL)
    public void onPickup(PlayerPickupItemEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null || e.isCancelled()) {
            return;
        }
        if (callCancellable(new ArenaItemPickupEvent(player, matchId, e.getItem()))) {
            e.setCancelled(true);
            player.updateInventory();
        }
    }

    // ------------------------------------------------------------ 世界 / 投掷物

    @EventHandler(priority = EventPriority.MONITOR)
    public void onWorldChange(PlayerChangedWorldEvent e) {
        Player player = e.getPlayer();
        UUID matchId = matchIdOf(player.getUniqueId());
        if (matchId == null) {
            return;
        }
        call(new ArenaChangedWorldEvent(player, matchId, e.getFrom(), player.getWorld()));
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onProjectileHit(org.bukkit.event.entity.ProjectileHitEvent e) {
        Projectile projectile = e.getEntity();
        ProjectileSource source = projectile.getShooter();
        if (!(source instanceof Player)) {
            return;
        }
        Player shooter = (Player) source;
        UUID matchId = matchIdOf(shooter.getUniqueId());
        if (matchId == null) {
            return;
        }
        // 1.8 的 ProjectileHitEvent 没有 getHitEntity()，命中实体由 ArenaArrowHitEvent 承载
        call(new ArenaProjectileHitEvent(shooter, matchId, projectile, null));
    }

    // ------------------------------------------------------------ 对局生命周期（由外部触发）

    public void fireMatchStart(UUID matchId, int round) {
        call(new ArenaMatchStartEvent(matchId, round));
    }

    public void fireMatchEnd(UUID matchId, int round, String reason, Object outcome) {
        call(new ArenaMatchEndEvent(matchId, round, reason, outcome));
    }

    public void fireEliminate(Player victim, Player killer, UUID matchId, EliminateReason reason) {
        call(new ArenaEliminateEvent(victim, killer, matchId, reason));
    }

    // ------------------------------------------------------------ golem 参与者登记

    /**
     * Bot 就绪时登记 Player 引用：golem 不在 PlayerList，必须在这里把真实 Player 交给解析器。
     * 同时通知感兴趣的 Manager 做补布场，覆盖 Ready 晚于 startMatch 的时序。
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onBotReady(GladiatorBotReadyEvent event) {
        Player player = event.getPlayer();
        PlayerResolver.remember(player);
        if (player != null) {
            FlatMatchManager.onParticipantReady(player.getUniqueId());
        }
    }

    /** Bot 退出时先淘汰再清缓存：golem despawn 无 Bukkit 事件，这是唯一的退出信号。 */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onBotRemoved(GladiatorBotRemovedEvent event) {
        UUID uuid = event.getBot().getUniqueId();
        FlatMatchManager.onParticipantRemoved(uuid);
        PlayerResolver.forget(uuid);
        lastAttacker.remove(uuid);
    }

    // ------------------------------------------------------------ 内部辅助

    private Player resolveKiller(Player victim) {
        Player direct = victim.getKiller();
        if (direct != null) {
            return direct;
        }
        AttackRecord record = lastAttacker.get(victim.getUniqueId());
        if (record == null || System.currentTimeMillis() - record.time > ATTRIBUTION_WINDOW_MS) {
            return null;
        }
        return PlayerResolver.get(record.attacker);
    }

    private static DeathReason mapCause(EntityDamageEvent lastDamage) {
        if (lastDamage == null) {
            return DeathReason.CUSTOM;
        }
        switch (lastDamage.getCause()) {
            case FALL: return DeathReason.FALLING;
            case SUFFOCATION: return DeathReason.SUFFOCATION;
            case DROWNING: return DeathReason.DROWNING;
            case STARVATION: return DeathReason.STARVATION;
            case POISON: return DeathReason.POISON;
            case WITHER: return DeathReason.WITHER;
            case FALLING_BLOCK: return DeathReason.FALLING_BLOCK;
            case FIRE: return DeathReason.FIRE;
            case FIRE_TICK: return DeathReason.FIRE_TICK;
            case LAVA: return DeathReason.LAVA;
            case SUICIDE: return DeathReason.SUICIDE;
            default: return DeathReason.CUSTOM;
        }
    }

    private static boolean isThrowable(Material material) {
        return material == Material.ENDER_PEARL || material == Material.SNOW_BALL
                || material == Material.EGG || material == Material.EXP_BOTTLE;
    }

    private static boolean isDrink(Material material) {
        return material == Material.POTION || material == Material.MILK_BUCKET
                || material == Material.WATER_BUCKET || material == Material.LAVA_BUCKET;
    }

    /** 1.8 中喷溅药水与普通药水共用 POTION，需读 Potion meta 区分。 */
    private static boolean isSplashPotion(ItemStack item) {
        if (item.getType() != Material.POTION) {
            return false;
        }
        org.bukkit.potion.Potion potion = org.bukkit.potion.Potion.fromItemStack(item);
        return potion != null && potion.isSplash();
    }

    private static boolean isSword(Material material) {
        return material.name().endsWith("_SWORD");
    }

    private static final class AttackRecord {
        private final UUID attacker;
        private final long time;

        private AttackRecord(UUID attacker, long time) {
            this.attacker = attacker;
            this.time = time;
        }
    }

}
