package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.event.entity.EntityDamageEvent.DamageCause;

import java.util.UUID;

/** 统一伤害事件：合并实体伤害与环境伤害，攻击者可为 null（摔落/火焰等）。 */
public class ArenaDamageEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Player attacker;
    private final double damage;
    private final DamageCause cause;

    public ArenaDamageEvent(Player victim, Player attacker, UUID matchId, double damage, DamageCause cause) {
        super(victim, matchId);
        this.attacker = attacker;
        this.damage = damage;
        this.cause = cause;
    }

    public Player getVictim() {
        return getPlayer();
    }

    public Player getAttacker() {
        return attacker;
    }

    public boolean hasAttacker() {
        return attacker != null;
    }

    public double getDamage() {
        return damage;
    }

    public DamageCause getCause() {
        return cause;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
