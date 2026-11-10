package top.cnolm.gladiator.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * BotHandle 注册表（G-104）：botId / UUID / username 三向索引。
 *
 * <p>只在主线程写入（spawn 绑定与玩家退出）；读取允许来自链路线程，
 * 因此内部用 {@link ConcurrentHashMap}，不做复合原子操作。
 */
public final class BotRegistry {

    private final Map<String, BotHandle> byBotId = new ConcurrentHashMap<String, BotHandle>();
    private final Map<UUID, BotHandle> byUuid = new ConcurrentHashMap<UUID, BotHandle>();
    private final Map<String, BotHandle> byUsername = new ConcurrentHashMap<String, BotHandle>();

    /**
     * 本插件的 botId 约定：{@code glad-<小写用户名>}。
     *
     * <p>运行时先连链路后等 SPAWN 时也能用同一公式算出 botId；插件始终以该值作为
     * 对外 botId，避免两端各自生成导致对不上。
     */
    public static String botIdFor(String username) {
        Objects.requireNonNull(username, "username");
        return "glad-" + username.toLowerCase(Locale.ROOT);
    }

    /** 供 spawner（含 golem 模式）在主线程写入。 */
    public void register(BotHandle handle) {
        Objects.requireNonNull(handle, "handle");
        BotHandle previous = byBotId.put(handle.getBotId(), handle);
        if (previous != null && previous != handle) {
            detach(previous);
        }
        byUuid.put(handle.getUniqueId(), handle);
        byUsername.put(key(handle.getUsername()), handle);
    }

    /** 供 spawner（含 golem 模式）在主线程移除。 */
    public void unregister(BotHandle handle) {
        if (handle == null) {
            return;
        }
        byBotId.remove(handle.getBotId(), handle);
        byUuid.remove(handle.getUniqueId(), handle);
        byUsername.remove(key(handle.getUsername()), handle);
    }

    public BotHandle findByBotId(String botId) {
        return botId == null ? null : byBotId.get(botId);
    }

    public BotHandle findByUniqueId(UUID uuid) {
        return uuid == null ? null : byUuid.get(uuid);
    }

    public BotHandle findByUsername(String username) {
        return username == null ? null : byUsername.get(key(username));
    }

    public Collection<BotHandle> all() {
        return new ArrayList<BotHandle>(byBotId.values());
    }

    public List<BotHandle> readyBots() {
        List<BotHandle> ready = new ArrayList<BotHandle>();
        for (BotHandle handle : byBotId.values()) {
            if (handle.isReady()) {
                ready.add(handle);
            }
        }
        return ready;
    }

    public int size() {
        return byBotId.size();
    }

    /** 供 spawner 在 shutdown 时整体清空。 */
    public void clear() {
        byBotId.clear();
        byUuid.clear();
        byUsername.clear();
    }

    private void detach(BotHandle handle) {
        byUuid.remove(handle.getUniqueId(), handle);
        byUsername.remove(key(handle.getUsername()), handle);
    }

    private static String key(String username) {
        return username.toLowerCase(Locale.ROOT);
    }
}
