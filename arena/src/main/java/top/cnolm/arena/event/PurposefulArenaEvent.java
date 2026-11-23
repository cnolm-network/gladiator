package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import top.cnolm.arena.status.MatchManager;

import java.util.UUID;

public abstract class PurposefulArenaEvent extends ArenaEvent implements Purposeful {
    protected final Player target;

    public PurposefulArenaEvent(Player p1, Player p2, UUID matchId) {
        super(p1, matchId);
        this.target = p2;
    }

    @Override
    public Player purpose() {
        return target;
    }

    @Override
    public boolean isEffective() {
        return super.isEffective() && MatchManager.isPlayerInMatch(target.getUniqueId(), matchId);
    }
}
