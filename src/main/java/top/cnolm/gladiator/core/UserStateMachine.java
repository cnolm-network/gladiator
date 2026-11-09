package top.cnolm.gladiator.core;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import top.cnolm.gladiator.api.UserState;

/**
 * Gladiator 自有状态机（G-301）：UUID → {@link UserState} 的唯一存储。
 *
 * <p>迁移由标准 Bukkit 事件（死亡/重生）与调用方的 {@code eliminate()} 驱动，
 * <strong>不读取任何竞技场插件的状态字段</strong>。本类只负责存储与"是否变化"的判断，
 * 事件派发仍由 {@link GladiatorServiceImpl#transition} 统一完成，保证单一写入点。
 */
public final class UserStateMachine {

    private final Map<UUID, UserState> states = new ConcurrentHashMap<UUID, UserState>();

    /** 未登记的用户一律视为 ALIVE（与新玩家语义一致）。 */
    public UserState stateOf(UUID uuid) {
        if (uuid == null) {
            return UserState.ALIVE;
        }
        UserState state = states.get(uuid);
        return state == null ? UserState.ALIVE : state;
    }

    /**
     * 迁移到目标状态。
     *
     * @return true 表示状态确实发生变化（此前不存在也视为变化）
     */
    public boolean moveTo(UUID uuid, UserState to) {
        if (uuid == null || to == null) {
            return false;
        }
        UserState from = states.put(uuid, to);
        return from != to;
    }

    /** 以指定状态初始化（已存在时覆盖），用于 Bot 登录完成。 */
    public void reset(UUID uuid, UserState state) {
        if (uuid != null && state != null) {
            states.put(uuid, state);
        }
    }

    public void forget(UUID uuid) {
        if (uuid != null) {
            states.remove(uuid);
        }
    }

    public void clear() {
        states.clear();
    }

    public int size() {
        return states.size();
    }
}
