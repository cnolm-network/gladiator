package top.cnolm.gladiator.core.board;

import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityDeathEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.entity.ProjectileHitEvent;
import org.bukkit.event.player.PlayerDropItemEvent;
import org.bukkit.event.player.PlayerItemConsumeEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerPickupItemEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.event.player.PlayerRespawnEvent;
import org.bukkit.event.player.PlayerTeleportEvent;
import top.cnolm.gladiator.api.board.GameEvent;
import top.cnolm.gladiator.api.board.GameEventType;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bukkit 事件 → {@link GameEvent} 的翻译器。
 *
 * <p>这是「事件驱动判定」的入口：板子拿到的死亡/重生/伤害全部来自这里，
 * 而不是外部进程的 1Hz 轮询。只在有活动会话时转发，并把 seq 与局号补进事件。
 *
 * <p>优先级用 {@link EventPriority#MONITOR}：只观察，不改别的插件的处理结果。
 */
public final class BoardEventBridge implements Listener {

    private final BoardRegistryImpl registry;

    private volatile BoardSessionImpl session;

    BoardEventBridge(BoardRegistryImpl registry) {
        this.registry = registry;
    }

    void attach(BoardSessionImpl session) {
        this.session = session;
    }

    void detach() {
        this.session = null;
    }

    // ------------------------------------------------------------ 玩家事件

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onJoin(PlayerJoinEvent event) {
        emit(GameEventType.PLAYER_JOIN, event.getPlayer().getName(), "", null);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onQuit(PlayerQuitEvent event) {
        emit(GameEventType.PLAYER_QUIT, event.getPlayer().getName(), "", null);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Player player = event.getEntity();
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("deathMessage", event.getDeathMessage() == null ? "" : event.getDeathMessage());
        data.put("killer", player.getKiller() == null ? "" : player.getKiller().getName());
        data.put("drops", event.getDrops() == null ? 0 : event.getDrops().size());
        data.put("keepInventory", event.getKeepInventory());
        emit(GameEventType.PLAYER_DEATH, player.getName(), dataString(data, "killer"), data);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onRespawn(PlayerRespawnEvent event) {
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("x", event.getRespawnLocation().getX());
        data.put("y", event.getRespawnLocation().getY());
        data.put("z", event.getRespawnLocation().getZ());
        emit(GameEventType.PLAYER_RESPAWN, event.getPlayer().getName(), "", data);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onTeleport(PlayerTeleportEvent event) {
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("cause", event.getCause() == null ? "" : event.getCause().name());
        data.put("toX", event.getTo().getX());
        data.put("toY", event.getTo().getY());
        data.put("toZ", event.getTo().getZ());
        emit(GameEventType.PLAYER_TELEPORT, event.getPlayer().getName(), "", data);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPickup(PlayerPickupItemEvent event) {
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("item", event.getItem().getItemStack().getType().name());
        data.put("amount", event.getItem().getItemStack().getAmount());
        emit(GameEventType.ITEM_PICKUP, event.getPlayer().getName(), "", data);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDrop(PlayerDropItemEvent event) {
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("item", event.getItemDrop().getItemStack().getType().name());
        data.put("amount", event.getItemDrop().getItemStack().getAmount());
        emit(GameEventType.ITEM_DROP, event.getPlayer().getName(), "", data);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onConsume(PlayerItemConsumeEvent event) {
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("item", event.getItem().getType().name());
        emit(GameEventType.ITEM_CONSUME, event.getPlayer().getName(), "", data);
    }

    // ------------------------------------------------------------ 战斗事件

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDamage(EntityDamageEvent event) {
        String actor = "";
        if (event instanceof EntityDamageByEntityEvent) {
            EntityDamageByEntityEvent byEntity = (EntityDamageByEntityEvent) event;
            if (byEntity.getDamager() instanceof Player) {
                actor = ((Player) byEntity.getDamager()).getName();
            } else if (byEntity.getDamager() instanceof org.bukkit.entity.Projectile) {
                org.bukkit.projectiles.ProjectileSource shooter =
                        ((org.bukkit.entity.Projectile) byEntity.getDamager()).getShooter();
                if (shooter instanceof Player) {
                    actor = ((Player) shooter).getName();
                }
            }
        }
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("damage", event.getDamage());
        data.put("finalDamage", event.getFinalDamage());
        data.put("cause", event.getCause() == null ? "" : event.getCause().name());
        data.put("cancelled", event.isCancelled());
        emit(GameEventType.ENTITY_DAMAGE, actor, entityName(event), data);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onEntityDeath(EntityDeathEvent event) {
        Player killer = event.getEntity().getKiller();
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("drops", event.getDrops() == null ? 0 : event.getDrops().size());
        emit(GameEventType.ENTITY_KILL, killer == null ? "" : killer.getName(),
                entityName(event), data);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onProjectileHit(ProjectileHitEvent event) {
        org.bukkit.projectiles.ProjectileSource shooter = event.getEntity().getShooter();
        String actor = shooter instanceof Player ? ((Player) shooter).getName() : "";
        Map<String, Object> data = new LinkedHashMap<String, Object>();
        data.put("projectile", event.getEntity().getType().name());
        emit(GameEventType.PROJECTILE_HIT, actor, "", data);
    }

    // ------------------------------------------------------------ 内部

    private void emit(GameEventType type, String actor, String target, Map<String, Object> data) {
        BoardSessionImpl current = session;
        if (current == null) {
            return;
        }
        GameEvent event = GameEvent.of(type, registry.nextSeq(), System.currentTimeMillis(),
                registry.roundOf(current), actor, target, data);
        current.onEvent(event);
    }

    private static String entityName(org.bukkit.event.entity.EntityEvent event) {
        org.bukkit.entity.Entity entity = event.getEntity();
        return entity == null ? "" : entity.getType().name();
    }

    private static String dataString(Map<String, Object> data, String key) {
        Object value = data.get(key);
        return value == null ? "" : String.valueOf(value);
    }
}
