package top.cnolm.arena.status;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public class PlayerMatchData<T extends GamingStatus> {
    private final UUID uuid;
    private final ArenaMatch<?, ?> arenaMatch;
    private final Map<String, Object> data = new HashMap<>();
    private final Class<T> statusClass;
    private T status;
    private int score;
    private int kills;
    private int deaths;
    private int assists;

    @SuppressWarnings("unchecked")
    public PlayerMatchData(UUID uuid, ArenaMatch<?, ?> arenaMatch, T status, int score, int kills, int deaths, int assists) {
        this(uuid, arenaMatch, status == null ? null : (Class<T>) status.getClass(), status, score, kills, deaths, assists);
    }

    /**
     * 通过状态类型令牌创建玩家数据。
     * 兼容枚举实现（如 GamingStatusImpl）与普通类实现，不使用已废弃的 Class#newInstance()。
     */
    public PlayerMatchData(UUID uuid, ArenaMatch<?, ?> arenaMatch, Class<T> clazz) {
        this(uuid, arenaMatch, Objects.requireNonNull(clazz, "clazz"), resolveDefaultStatus(clazz), 0, 0, 0, 0);
    }

    private PlayerMatchData(UUID uuid, ArenaMatch<?, ?> arenaMatch, Class<T> statusClass, T status,
                            int score, int kills, int deaths, int assists) {
        this.uuid = uuid;
        this.arenaMatch = arenaMatch;
        this.statusClass = statusClass;
        this.status = status;
        this.score = score;
        this.kills = kills;
        this.deaths = deaths;
        this.assists = assists;
    }

    private static <T extends GamingStatus> T resolveDesignatedStatus(Class<T> clazz, SRunnable runnable) {
        Objects.requireNonNull(clazz, "clazz");
        GamingStatus sample;
        if (clazz.isEnum()) {
            T[] constants = clazz.getEnumConstants();
            if (constants.length == 0) {
                throw new IllegalStateException("No constants in enum " + clazz.getName());
            }
            sample = constants[0];
        } else {
            try {
                sample = clazz.getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Cannot instantiate status type " + clazz.getName(), e);
            }
        }
        GamingStatus defaultStatus = runnable.run(sample);
        if (defaultStatus == null) {
            throw new IllegalStateException("defaultStatus() returned null for " + clazz.getName());
        }
        return clazz.cast(defaultStatus);
    }

    private static <T extends GamingStatus> T resolveDefaultStatus(Class<T> clazz) {
        return resolveDesignatedStatus(clazz, GamingStatus::defaultStatus);
    }

    private static <T extends GamingStatus> T resolveCountdownStatus(Class<T> clazz) {
        return resolveDesignatedStatus(clazz, GamingStatus::countdown);
    }

    private static <T extends GamingStatus> T resolveFightingStatus(Class<T> clazz) {
        return resolveDesignatedStatus(clazz, GamingStatus::fighting);
    }

    private static <T extends GamingStatus> T resolveEliminatedStatus(Class<T> clazz) {
        return resolveDesignatedStatus(clazz, GamingStatus::eliminate);
    }

    public UUID getUuid() {
        return uuid;
    }

    public ArenaMatch<?, ?> getMatch() {
        return arenaMatch;
    }

    public T getStatus() {
        return status;
    }

    public int getScore() {
        return score;
    }

    public int getKills() {
        return kills;
    }

    public int getDeaths() {
        return deaths;
    }

    public int getAssists() {
        return assists;
    }

    public void setStatus(T status) {
        this.status = status;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public void setKills(int kills) {
        this.kills = kills;
    }

    public void setDeaths(int deaths) {
        this.deaths = deaths;
    }

    public void setAssists(int assists) {
        this.assists = assists;
    }

    public void addScore(int score) {
        this.score += score;
    }

    public void addKills(int kills) {
        this.kills += kills;
    }

    public void addDeaths(int deaths) {
        this.deaths += deaths;
    }

    public void addAssists(int assists) {
        this.assists += assists;
    }

    public void setData(String key, Object value) {
        data.put(key, value);
    }

    public Object getData(String key) {
        return data.get(key);
    }

    public <V> V getData(String key, Class<V> clazz) {
        return clazz.cast(data.get(key));
    }

    public void removeData(String key) {
        data.remove(key);
    }

    public Map<String, Object> getData() {
        return data;
    }

    public Class<T> getStatusClass() {
        return statusClass;
    }

    public void switchStatus(StatusSwitch<T> statusSwitch) {
        if (statusClass == null) {
            throw new IllegalStateException("statusClass is unavailable; status was created without a type token");
        }
        this.status = statusSwitch.run(statusClass);
    }

    public void defaultStatus() {
        switchStatus(PlayerMatchData::resolveDefaultStatus);
    }

    public void eliminate() {
        switchStatus(PlayerMatchData::resolveEliminatedStatus);
    }

    public void countdown() {
        switchStatus(PlayerMatchData::resolveCountdownStatus);
    }

    public void fighting() {
        switchStatus(PlayerMatchData::resolveFightingStatus);
    }

    private interface SRunnable {
        GamingStatus run(GamingStatus status);
    }

    /** 返回类型声明为 S，调用方无需向下转型，从类型系统消除 unchecked cast */
    public interface StatusSwitch<S extends GamingStatus> {
        S run(Class<S> clazz);
    }
}
