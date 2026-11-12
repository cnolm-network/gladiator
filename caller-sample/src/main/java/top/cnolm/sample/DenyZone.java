package top.cnolm.sample;

import org.bukkit.Location;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.HandlerList;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.plugin.Plugin;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A3 验收用的「假领地插件」——<b>不依赖 WorldGuard</b>。
 *
 * <p>在球体范围内以 {@code LOWEST} 取消所有由玩家发起的伤害事件，并把每次取消打到控制台。
 * 用来验证 Gladiator 的观测式 deny 反馈：它只知道「有人取消了」，不需要知道是谁、为什么。
 *
 * <p>只在 caller-sample 里用；生产环境应换成真正的领地插件。
 */
public final class DenyZone implements Listener {

    /** 一个球形保护区。 */
    public static final class Zone {
        private final Location center;
        private final double radius;

        public Zone(Location center, double radius) {
            this.center = center.clone();
            this.radius = radius;
        }

        public Location getCenter() {
            return center.clone();
        }

        public double getRadius() {
            return radius;
        }

        public boolean contains(Location location) {
            if (location == null || location.getWorld() == null
                    || center.getWorld() == null
                    || !location.getWorld().equals(center.getWorld())) {
                return false;
            }
            return location.distanceSquared(center) <= radius * radius;
        }

        @Override
        public String toString() {
            return String.format(java.util.Locale.ROOT, "%s %.1f,%.1f,%.1f r=%.1f",
                    center.getWorld().getName(), center.getX(), center.getY(), center.getZ(), radius);
        }
    }

    private final Plugin plugin;
    private final List<Zone> zones = new ArrayList<Zone>();
    private boolean registered;

    public DenyZone(Plugin plugin) {
        this.plugin = plugin;
    }

    public void add(Zone zone) {
        zones.add(zone);
        if (!registered) {
            plugin.getServer().getPluginManager().registerEvents(this, plugin);
            registered = true;
        }
    }

    public List<Zone> list() {
        return Collections.unmodifiableList(new ArrayList<Zone>(zones));
    }

    public void clear() {
        zones.clear();
        if (registered) {
            HandlerList.unregisterAll(this);
            registered = false;
        }
    }

    public boolean isEmpty() {
        return zones.isEmpty();
    }

    /** LOWEST：比所有真实插件更早，模拟「领地插件一票否决」。 */
    @EventHandler(priority = EventPriority.LOWEST)
    public void onDamage(EntityDamageByEntityEvent event) {
        if (!(event.getDamager() instanceof Player)) {
            return;
        }
        Entity victim = event.getEntity();
        Location at = victim == null ? null : victim.getLocation();
        for (Zone zone : zones) {
            if (zone.contains(at)) {
                event.setCancelled(true);
                Player attacker = (Player) event.getDamager();
                plugin.getLogger().info("[DenyZone] 取消 " + attacker.getName() + " → "
                        + (victim == null ? "?" : victim.getType() + "@" + fmt(at))
                        + "（zone " + zone + "，damage=" + event.getDamage() + "）");
                return;
            }
        }
    }

    private static String fmt(Location location) {
        if (location == null) {
            return "?";
        }
        return String.format(java.util.Locale.ROOT, "%.1f,%.1f,%.1f",
                location.getX(), location.getY(), location.getZ());
    }
}
