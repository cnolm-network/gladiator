package top.cnolm.gladiator.api;

import org.bukkit.entity.Player;

import java.util.UUID;

/**
 * 统一参与者视图：真人与 Bot 在 API 层没有区别。
 *
 * <p>调用方可以混合把真人与 Bot 传入 {@link GladiatorAPI#pushGame(Mode, User...)}。
 */
public interface User {

    UUID getUniqueId();

    String getName();

    /** 只读；实际迁移由 Gladiator 内部状态机执行。 */
    UserState getState();

    boolean isBot();

    /** 在线时返回真实 Bukkit Player；离线返回 null。Bot 也是真实登录的 CraftPlayer。 */
    Player asPlayer();
}
