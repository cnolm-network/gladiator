package top.cnolm.arena.status;

import org.bukkit.entity.Player;
import top.cnolm.arena.ArenaPlugin;
import top.cnolm.arena.EliminateReason;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 对局信息类 </br>
 * 仅作为数据结构，不处理游戏逻辑，游戏逻辑处理在 {@link }
 * @param <T>
 * @param <V>
 */
public class ArenaMatch<T extends GamingStatus, V extends MatchStatus> {
    private final UUID id;
    private V status;
    // 状态类型令牌：供 PlayerMatchData 创建默认状态，避免运行期泛型擦除问题
    private final Class<T> statusClass;
    private final Class<V> matchStatusClass;
    private final Set<ArenaTeam> teams = ConcurrentHashMap.newKeySet();
    private final Map<UUID, PlayerMatchData<T>> playerDataMap = new HashMap<>();
    private final Set<UUID> eliminated = ConcurrentHashMap.newKeySet();
    private final Map<String, Object> data = Collections.synchronizedMap(new HashMap<String, Object>());

    public ArenaMatch(Class<T> statusClass, Class<V> matchStatusClass) {
        this.id = UUID.randomUUID();
        this.matchStatusClass = Objects.requireNonNull(matchStatusClass, "matchStatusClass");
        this.status = resolveDefaultMatchStatus(matchStatusClass);
        this.statusClass = Objects.requireNonNull(statusClass, "statusClass");
    }

    /** 从类型令牌解析默认对局状态，兼容枚举实现与普通类实现 */
    private static <V extends MatchStatus> V resolveDefaultMatchStatus(Class<V> clazz) {
        Status sample;
        if (clazz.isEnum()) {
            V[] constants = clazz.getEnumConstants();
            if (constants.length == 0) {
                throw new IllegalStateException("No constants in enum " + clazz.getName());
            }
            sample = constants[0];
        } else {
            try {
                sample = clazz.getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Cannot instantiate match status type " + clazz.getName(), e);
            }
        }
        Status defaultStatus = sample.defaultStatus();
        if (defaultStatus == null) {
            throw new IllegalStateException("defaultStatus() returned null for " + clazz.getName());
        }
        return clazz.cast(defaultStatus);
    }

    public UUID getId() {
        return id;
    }

    public V getStatus() {
        return status;
    }

    public void setStatus(V status) {
        this.status = status;
    }

    public Set<ArenaTeam> getTeams() {
        return Collections.unmodifiableSet(teams);
    }

    public Map<UUID, PlayerMatchData<T>> getPlayerDataMap() {
        return Collections.unmodifiableMap(playerDataMap);
    }

    public PlayerMatchData<T> getPlayerData(UUID uuid) {
        return playerDataMap.get(uuid);
    }

    public PlayerMatchData<T> getPlayerData(Player player) {
        return playerDataMap.get(player.getUniqueId());
    }

    public Map<String, Object> getData() {
        return Collections.unmodifiableMap(data);
    }

    public void setData(String key, Object value) {
        data.put(key, value);
    }

    public Object getData(String key) {
        return data.get(key);
    }

    public Class<T> getStatusClass() {
        return statusClass;
    }

    public Class<V> getMatchStatusClass() {
        return matchStatusClass;
    }

    /**
     * 淘汰玩家并广播事件；killer 为 null 表示非击杀淘汰（摔落、弃权等）。
     * 淘汰状态写入 eliminated，供 checkEnd 判定存活队伍。
     */
    public void eliminatePlayer(UUID uuidP, UUID uuidK, EliminateReason cause) {
        // 幂等：golem 可能同时触发 PlayerQuitEvent 与 BotRemovedEvent，避免重复淘汰/重复文字
        if (!eliminated.add(uuidP)) {
            return;
        }
        // 先写淘汰状态再广播：事件监听是同步的，晚写会让 checkEnd 漏判本轮淘汰
        PlayerMatchData<T> data = playerDataMap.get(uuidP);
        if (data != null) {
            data.eliminate();
        }
        Player victim = PlayerResolver.get(uuidP);
        Player killer = uuidK == null ? null : PlayerResolver.get(uuidK);
        ArenaPlugin.getEventHandler().fireEliminate(victim, killer, getId(), cause);
    }

    public void eliminatePlayer(Player player, Player killer, EliminateReason cause) {
        eliminatePlayer(player.getUniqueId(), killer == null ? null : killer.getUniqueId(), cause);
    }

    /** @deprecated 建议使用 {@link #eliminatePlayer(UUID, UUID, EliminateReason)} **/
    @Deprecated
    public void eliminatePlayer(UUID uuidP, UUID uuidK) {
        eliminatePlayer(uuidP, uuidK, EliminateReason.KILL);
    }

    /** @deprecated 建议使用 {@link #eliminatePlayer(Player, Player, EliminateReason)} **/
    @Deprecated
    public void eliminatePlayer(Player player, Player killer) {
        eliminatePlayer(player.getUniqueId(), killer == null ? null : killer.getUniqueId());
    }

    /**
     * 把所有仍存活的参与者标记为出局，但<b>不</b>逐个广播淘汰事件。
     *
     * <p>用于超时/中止：若像 {@link #eliminatePlayer} 那样逐个淘汰，第一次淘汰就会触发
     * {@code checkEnd()}（存活队伍 ≤1）并把胜利判给另一方——超时应该判平局而不是送人头。
     * 这里先静默清空存活集合，再一次性收敛成「无人存活」。
     */
    public void eliminateAllWithoutEvents(EliminateReason cause) {
        boolean changed = false;
        for (UUID uuid : new ArrayList<>(playerDataMap.keySet())) {
            if (eliminated.add(uuid)) {
                PlayerMatchData<T> data = playerDataMap.get(uuid);
                if (data != null) {
                    data.eliminate();
                }
                changed = true;
            }
        }
        if (changed) {
            checkEnd();
        }
    }

    /** 查询某参与者是否已被淘汰。 */
    public boolean isEliminated(UUID uuid) {
        return uuid != null && eliminated.contains(uuid);
    }

    public void startCountdown(UUID uuid) {
        playerDataMap.get(uuid).countdown();
    }

    public void startCountdown(Player player) {
        startCountdown(player.getUniqueId());
    }

    public void startFighting(UUID uuid) {
        playerDataMap.get(uuid).fighting();
    }

    public void startFighting(Player player) {
        startFighting(player.getUniqueId());
    }

    // 使用 R 避免遮蔽类级别泛型 T/V
    public <R> R getData(String key, Class<R> clazz) {
        return clazz.cast(data.get(key));
    }

    /**
     * 查询对局是否应当结束：存活队伍数 ≤ 1 时结束。
     * 存活队伍定义为「至少有一名成员未被淘汰」；0 队表示该局无人存活。
     *
     * @return true 表示已触发结束处理，false 表示对局继续
     */
    public boolean checkEnd() {
        if (isEndStatus(status)) {
            return true;
        }
        int aliveTeams = 0;
        for (ArenaTeam team : teams) {
            for (UUID uuid : team.getPlayers()) {
                if (!eliminated.contains(uuid)) {
                    aliveTeams++;
                    break;
                }
            }
        }
        if (aliveTeams <= 1) {
            handleEnd(aliveTeams);
            return true;
        }
        return false;
    }

    private static boolean isEndStatus(MatchStatus current) {
        return current != null && ("END".equals(current.getName()) || "CLOSED".equals(current.getName()));
    }

    private void handleEnd(int aliveTeams) {
        V end = resolveMatchStatus("END");
        if (end != null) {
            setStatus(end);
        }
        String reason = aliveTeams == 1 ? "one-team-left" : "no-team-left";
        ArenaPlugin.getEventHandler().fireMatchEnd(getId(), 0, reason, aliveTeams == 1 ? winner() : null);
    }

    @SuppressWarnings("unchecked")
    private V resolveMatchStatus(String name) {
        if (status == null) {
            return null;
        }
        try {
            Status resolved = status.fromName(name);
            return resolved == null ? null : (V) resolved;
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** 返回唯一存活队伍；0 队时返回 null。 */
    private ArenaTeam winner() {
        for (ArenaTeam team : teams) {
            for (UUID uuid : team.getPlayers()) {
                if (!eliminated.contains(uuid)) {
                    return team;
                }
            }
        }
        return null;
    }

    public void addPlayer(UUID uuid) {
        if (playerDataMap.containsKey(uuid)) {
            return;
        }
        playerDataMap.put(uuid, new PlayerMatchData<>(uuid, this, statusClass));
    }

    public void addTeam(ArenaTeam team) {
        if (team != null) {
            teams.add(team);
            for (UUID uuid : team.getPlayers()) {
                addPlayer(uuid);
            }
        }
    }

    public void forEachOnlinePlayer(PRunnable runnable) {
        playerDataMap.forEach((uuid, data) -> runnable.call(PlayerResolver.get(uuid)));
    }

    public void forEachAlivePlayer(PRunnable runnable) {
        playerDataMap.forEach((uuid, data) -> {
            if (!eliminated.contains(uuid)) {
                runnable.call(PlayerResolver.get(uuid));
            }
        });
    }

    public void forEachPlayer(URunnable runnable) {
        playerDataMap.forEach((uuid, data) -> runnable.run(uuid));
    }

    @FunctionalInterface
    public interface PRunnable {
        default void call(Player player) {
            if (player != null) run(player);
        }
        void run(Player player);
    }

    @FunctionalInterface
    public interface URunnable {
        void run(UUID uuid);
    }
}
