package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.player.PlayerEvent;
import top.cnolm.arena.status.ArenaMatch;
import top.cnolm.arena.status.GamingStatus;
import top.cnolm.arena.status.MatchManager;
import top.cnolm.arena.status.MatchStatus;

import java.util.UUID;

public abstract class ArenaEvent extends PlayerEvent {
    protected final UUID matchId;

    public ArenaEvent(Player p, UUID matchId) {
        super(p);
        this.matchId = matchId;
    }

    public UUID getMatchId() {
        return matchId;
    }

    public ArenaMatch<?, ?> getMatch() {
        return MatchManager.getMatch(matchId);
    }

    public <T extends GamingStatus, V extends MatchStatus> ArenaMatch<T, V> getMatch(Class<T> statusClass, Class<V> matchStatusClass) {
        return MatchManager.getMatch(matchId, statusClass, matchStatusClass);
    }

    public boolean isEffective() {
        return MatchManager.isPlayerInMatch(player.getUniqueId(), matchId);
    }
}
