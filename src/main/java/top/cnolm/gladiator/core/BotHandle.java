package top.cnolm.gladiator.core;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import top.cnolm.gladiator.api.Bot;
import top.cnolm.gladiator.api.UserState;
import top.cnolm.gladiator.api.UserStateRegistry;

import java.nio.charset.Charset;
import java.util.Objects;
import java.util.UUID;

/**
 * 服务端侧的 Bot 代理（G-104）。
 *
 * <p>Bot 是外部 mineflayer 运行时以真实协议登录进来的普通 Bukkit 玩家，
 * 因此这里<strong>不伪造任何 PlayerConnection</strong>：所有 Player 相关操作都委托给
 * {@link #asPlayer()} 返回的真实 {@link Player}。绑定前 {@code asPlayer()} 为 {@code null}、
 * {@code getUniqueId()} 退化为离线 UUID（与 Minecraft 离线模式算法一致），
 * 绑定后立即切换为真实玩家 UUID。
 *
 * <p>生命周期：{@code spawnBot} 创建 pending 请求 → 玩家 join 时 {@link #bind(Player)}
 * → 注册进 {@link BotRegistry} 并派发 {@code GladiatorBotReadyEvent} → 退出时
 * {@link #unbind()}。状态读写统一走 {@link UserStateRegistry}，不在这里维护第二份状态。
 */
public final class BotHandle implements Bot {

    private static final Charset UTF_8 = Charset.forName("UTF-8");

    private final String botId;
    private final String username;
    private final UUID offlineUuid;

    private volatile UUID uuid;
    private volatile Player player;
    private volatile boolean ready;

    /** 由 spawner 创建；{@code bind} 前处于未就绪状态（UUID 退化为离线 UUID）。 */
    public BotHandle(String botId, String username) {
        this.botId = Objects.requireNonNull(botId, "botId");
        this.username = Objects.requireNonNull(username, "username");
        this.offlineUuid = UUID.nameUUIDFromBytes(
                ("OfflinePlayer:" + username).getBytes(UTF_8));
        this.uuid = offlineUuid;
    }

    /** 真实玩家实体已登录；此后 UUID/名称/在线状态全部以 Bukkit 为准。 */
    public void bind(Player player) {
        Objects.requireNonNull(player, "player");
        this.player = player;
        this.uuid = player.getUniqueId();
        this.ready = true;
    }

    /** 玩家退出/被踢；保留 handle 以便调用方读取历史身份，但不再当作就绪。 */
    public void unbind() {
        this.ready = false;
        this.player = null;
        this.uuid = offlineUuid;
    }

    @Override
    public String getBotId() {
        return botId;
    }

    @Override
    public boolean isReady() {
        Player current = player;
        return ready && current != null && isLive(current);
    }

    @Override
    public UUID getUniqueId() {
        return uuid;
    }

    @Override
    public String getName() {
        Player current = player;
        return current == null ? username : current.getName();
    }

    @Override
    public UserState getState() {
        return UserStateRegistry.stateOf(uuid);
    }

    @Override
    public boolean isBot() {
        return true;
    }

    /**
     * 真实 Bukkit 玩家。未绑定或已退出时为 {@code null}；已绑定时返回服务端当前实例
     * （若 Bukkit 已重建 Player 对象则重新解析一次）。
     */
    @Override
    public Player asPlayer() {
        Player current = player;
        if (current != null && isLive(current)) {
            return current;
        }
        if (!ready) {
            return null;
        }
        return Bukkit.getPlayer(uuid);
    }

    /**
     * 「这个 Player 还能用吗」。
     *
     * <p>不能只看 {@code Player#isOnline()}：1.8.8 的 {@code isOnline()} 实现是
     * {@code CraftServer.getPlayer(uuid) != null}，也就是<strong>必须登录进 PlayerList</strong>。
     * golem 身体是进程内 NMS 实体、从不进 PlayerList，因此 {@code isOnline()} 恒为 false，
     * 但实体在世界里真实存在（{@code isValid()} 为 true）。两者取并集：
     * <ul>
     *   <li>runtime 模式：真实登录的玩家 isOnline=true；退出时由 PlayerQuitEvent 解绑。</li>
     *   <li>golem 模式：实体有效即可用；despawn 后 golem 侧返回 null {@code bukkitPlayer()}，
     *       由 {@code GolemBodySpawner} 的对账任务解绑。</li>
     * </ul>
     */
    private static boolean isLive(Player player) {
        return player.isOnline() || player.isValid();
    }

    /** 请求时的原始用户名（不随 Player 改名变化，用于匹配 pending 请求与 golem 身体）。 */
    public String getUsername() {
        return username;
    }

    UUID getOfflineUuid() {
        return offlineUuid;
    }

    @Override
    public String toString() {
        return "BotHandle{" + botId + ", " + getName() + ", ready=" + ready + "}";
    }
}
