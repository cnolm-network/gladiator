package top.cnolm.gladiator.api;

import java.util.Objects;
import java.util.UUID;

/**
 * 真人 {@link User} 的状态查询入口。
 *
 * <p>Gladiator 插件本体在 onEnable 时安装一个 Lookup（由 core 的状态机实现），
 * api 包本身不依赖 core。未安装时，未纳入会话的在线玩家返回 {@link UserState#ALIVE}。
 * 调用方不需要、也不应该使用本类的 install/uninstall。
 */
public final class UserStateRegistry {

    public interface Lookup {
        UserState stateOf(UUID uuid);
    }

    private static volatile Lookup lookup;

    private UserStateRegistry() {
    }

    /** 仅供 Gladiator 插件本体调用。 */
    public static void install(Lookup value) {
        lookup = Objects.requireNonNull(value, "lookup");
    }

    /** 仅供 Gladiator 插件本体调用。 */
    public static void uninstall() {
        lookup = null;
    }

    public static UserState stateOf(UUID uuid) {
        Lookup current = lookup;
        if (current == null || uuid == null) {
            return UserState.ALIVE;
        }
        UserState state = current.stateOf(uuid);
        return state == null ? UserState.ALIVE : state;
    }
}
