package top.cnolm.gladiator.core;

import org.bukkit.entity.Player;

/**
 * 强制重生（只替玩家把「死亡界面点重生」这一步做掉）。
 *
 * <p>为什么需要它：1.8 客户端死亡后会停在死亡界面，必须手动点「重生」；
 * 服务端没有原版命令能代替。人机对战里玩家被打死后不点重生，就永远停在死亡界面，
 * Bot 也追不到人，对局体验直接崩掉。{@code Player#spigot().respawn()} 是 Spigot
 * 提供的官方入口（非 NMS），只在玩家确实死亡时调用。
 *
 * <p>边界：
 * <ul>
 *   <li>只读判断 + 一次 respawn 调用，不改血量/背包/位置；</li>
 *   <li>不含对局规则：重生后回哪里由调用方（Dashboard 编排层的 {@code tp}/{@code spawnpoint}）决定；</li>
 *   <li>不引入任何反作弊/领地依赖（ADR-015）。</li>
 * </ul>
 *
 * <p>返回值是给 RCON 解析的固定单行文案：
 * <pre>
 * respawn 已请求：&lt;name&gt;
 * respawn 跳过：&lt;name&gt; 未死亡
 * respawn 失败：玩家不在线
 * </pre>
 */
public final class PlayerRespawn {

    public static final String REQUESTED_PREFIX = "respawn 已请求：";
    public static final String SKIPPED_PREFIX = "respawn 跳过：";
    public static final String OFFLINE_MESSAGE = "respawn 失败：玩家不在线";

    /** 真正触发重生的动作。抽成接口是为了单测能注入假实现（{@code Player$Spigot} 是抽象类，无法代理）。 */
    public interface Action {
        void respawn(Player player);
    }

    private static final Action DEFAULT_ACTION = player -> player.spigot().respawn();

    private PlayerRespawn() {
    }

    /** 对在线玩家请求重生；{@code null} 视为不在线。 */
    public static String request(Player player) {
        return request(player, DEFAULT_ACTION);
    }

    static String request(Player player, Action action) {
        if (player == null) {
            return OFFLINE_MESSAGE;
        }
        String name = player.getName();
        if (!player.isDead()) {
            return SKIPPED_PREFIX + name + " 未死亡";
        }
        if (action == null) {
            return "respawn 失败：" + name;
        }
        try {
            action.respawn(player);
        } catch (RuntimeException ex) {
            String reason = ex.getMessage();
            return "respawn 失败：" + (reason == null || reason.isEmpty() ? name : reason);
        }
        return REQUESTED_PREFIX + name;
    }

    /** 是否处于死亡状态（离线/未死亡都为 false）；给命令路由做防御性判断。 */
    public static boolean isDead(Player player) {
        return player != null && player.isDead();
    }
}
