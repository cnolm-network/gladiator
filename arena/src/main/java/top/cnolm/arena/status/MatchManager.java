package top.cnolm.arena.status;

import org.bukkit.entity.Player;
import top.cnolm.arena.ArenaPlugin;
import top.cnolm.arena.handler.ArenaEventHandler;

import java.util.*;

/**
 * 静态类，用于处理对局的状态
 */
public class MatchManager {
    private static Map<UUID, ArenaMatch<?, ?>> matchMap = new HashMap<>();
    private static Map<UUID, ArenaMatch<?, ?>> playerMatch = new HashMap<>();
    private static Map<UUID, Set<ArenaMatch<?, ?>>> matchHistory = new HashMap<>();

    /**
     * 处理游戏开始后的外部逻辑，不处理对局内逻辑
     * @param match 开始的对局
     */
    public static void startMatch(ArenaMatch<?, ?> match) {
        matchMap.put(match.getId(), match);
        match.getPlayerDataMap().keySet().forEach(uuid -> registerPlayer(uuid, match));
        ArenaPlugin.getEventHandler().fireMatchStart(match.getId(), 0);
    }

    /**
     * 登记参与者到对局映射。相比 forEachOnlinePlayer 的写法，这里直接用 UUID，
     * 因此 golem 即使不在 PlayerList（晚于 startMatch Ready）也能被登记。
     */
    public static void registerPlayer(UUID playerId, ArenaMatch<?, ?> match) {
        if (playerId != null && match != null) {
            playerMatch.put(playerId, match);
        }
    }

    /**
     * 当对局结束后执行的外部逻辑，不处理对局内逻辑
     * @param match 结束的对局
     */
    public static void endMatch(ArenaMatch<?, ?> match) {
        if (match == null) {
            return;
        }
        matchMap.remove(match.getId());
        match.getPlayerDataMap().keySet().forEach(uuid -> playerMatch.remove(uuid, match));
        matchHistory.computeIfAbsent(match.getId(), k -> new HashSet<>()).add(match);
    }

    public static Map<UUID, ArenaMatch<?, ?>> getMatches() {
        return matchMap;
    }

    public static ArenaMatch<?, ?> getMatch(UUID matchId) {
        return matchMap.getOrDefault(matchId, null);
    }

    public static <T extends GamingStatus, V extends MatchStatus> ArenaMatch<T, V> getMatch(UUID matchId, Class<T> statusClass, Class<V> matchStatusClass) {
        @SuppressWarnings("unchecked")
        ArenaMatch<T, V> match = (ArenaMatch<T, V>) matchMap.getOrDefault(matchId, null);
        return match;
    }

    public static boolean isPlayerInMatch(Player player) {
        return isPlayerInMatch(player.getUniqueId());
    }

    public static boolean isPlayerInMatch(UUID playerId) {
        return playerMatch.containsKey(playerId);
    }

    public static boolean isPlayerInMatch(UUID playerId, UUID matchId) {
        if (playerId == null || matchId == null) {
            return false;
        }
        ArenaMatch<?, ?> inMatch = playerMatch.get(playerId);
        ArenaMatch<?, ?> byId = matchMap.get(matchId);
        return inMatch != null && byId != null && inMatch.getId().equals(byId.getId());
    }

    public static ArenaMatch<?, ?> getPlayerMatch(UUID playerId) {
        return playerMatch.getOrDefault(playerId, null);
    }

    public static ArenaMatch<?, ?> getPlayerMatch(Player player) {
        return getPlayerMatch(player.getUniqueId());
    }

    public static <T extends GamingStatus, V extends MatchStatus> ArenaMatch<T, V> getPlayerMatch(Player player, Class<T> statusClass, Class<V> matchStatusClass) {
        return getPlayerMatch(player.getUniqueId(), statusClass, matchStatusClass);
    }

    public static <T extends GamingStatus, V extends MatchStatus> ArenaMatch<T, V> getPlayerMatch(UUID playerId, Class<T> statusClass, Class<V> matchStatusClass) {
        @SuppressWarnings("unchecked")
        ArenaMatch<T, V> match = (ArenaMatch<T, V>) playerMatch.getOrDefault(playerId, null);
        return match;
    }
}
