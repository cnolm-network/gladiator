package top.cnolm.arena.event;

import org.bukkit.event.Event;
import top.cnolm.arena.status.ArenaMatch;
import top.cnolm.arena.status.GamingStatus;
import top.cnolm.arena.status.MatchManager;
import top.cnolm.arena.status.MatchStatus;

import java.util.UUID;

/** 对局级事件基类：只绑定 matchId，不绑定具体玩家。 */
public abstract class ArenaMatchEvent extends Event {
    protected final UUID matchId;

    public ArenaMatchEvent(UUID matchId) {
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
}
