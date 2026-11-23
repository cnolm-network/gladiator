package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 投掷物命中：由 ProjectileHitEvent 包装，携带投掷物与命中目标，不可取消。 */
public class ArenaProjectileHitEvent extends ArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Projectile projectile;
    private final org.bukkit.entity.Entity hitEntity;

    public ArenaProjectileHitEvent(Player shooter, UUID matchId, Projectile projectile, org.bukkit.entity.Entity hitEntity) {
        super(shooter, matchId);
        this.projectile = projectile;
        this.hitEntity = hitEntity;
    }

    public Projectile getProjectile() {
        return projectile;
    }

    /** 命中的实体，打中方块时为 null。 */
    public org.bukkit.entity.Entity getHitEntity() {
        return hitEntity;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
