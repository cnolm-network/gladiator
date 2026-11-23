package top.cnolm.arena.event;

import org.bukkit.entity.Arrow;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 箭矢命中：由伤害事件拆出，射手可为 null（环境发射），不可取消。 */
public class ArenaArrowHitEvent extends ArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Player shooter;
    private final Arrow arrow;
    private final double damage;

    public ArenaArrowHitEvent(Player victim, Player shooter, UUID matchId, Arrow arrow, double damage) {
        super(victim, matchId);
        this.shooter = shooter;
        this.arrow = arrow;
        this.damage = damage;
    }

    public Player getVictim() {
        return getPlayer();
    }

    public Player getShooter() {
        return shooter;
    }

    public boolean hasShooter() {
        return shooter != null;
    }

    public Arrow getArrow() {
        return arrow;
    }

    public double getDamage() {
        return damage;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
