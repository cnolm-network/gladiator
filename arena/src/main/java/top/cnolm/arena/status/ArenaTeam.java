package top.cnolm.arena.status;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.entity.Player;

import java.util.*;

public class ArenaTeam {
    private final UUID id;
    private final String name;
    private final ChatColor color;
    private final Location spawn;
    private final List<UUID> players;
    private final Map<String, Object> data = new HashMap<>();

    public ArenaTeam(String name, ChatColor color, Location spawn, List<UUID> players) {
        this.id = UUID.randomUUID();
        this.name = name;
        this.color = color;
        this.spawn = spawn;
        this.players = players;
    }

    public ArenaTeam(String name, Location spawn, ChatColor color) {
        this(name, color, spawn, new ArrayList<>());
    }

    public String getName() {
        return name;
    }

    public ChatColor getColor() {
        return color;
    }

    public List<UUID> getPlayers() {
        return players;
    }

    public UUID getId() {
        return id;
    }

    public Location getSpawn() {
        return spawn;
    }

    public Map<String, Object> getData() {
        return data;
    }

    public void setData(String key, Object value) {
        data.put(key, value);
    }

    public Object getData(String key) {
        return data.get(key);
    }

    public <T> T getData(String key, Class<T> clazz) {
        return clazz.cast(data.get(key));
    }

    public void addPlayer(UUID player) {
        players.add(player);
    }

    public void removePlayer(Player player) {
        players.remove(player.getUniqueId());
    }

    public boolean containsPlayer(Player player) {
        return players.contains(player.getUniqueId());
    }

    public void forEachOnlinePlayer(PRunnable runnable) {
        players.forEach(uuid -> runnable.call(PlayerResolver.get(uuid)));
    }

    public void forEachPlayer(URunnable runnable) {
        players.forEach(runnable::run);
    }

    public void toSpawn() {
        forEachOnlinePlayer(player -> player.teleport(spawn));
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
