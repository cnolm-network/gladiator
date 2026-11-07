package top.cnolm.gladiator.api;

/**
 * 参与者在对局/训练会话中的存活状态。
 *
 * <p>迁移由 Gladiator 内部的状态机执行，调用方只能读取；
 * {@link GladiatorAPI#eliminate(User)} 是调用方唯一的状态写入入口（只打标记）。
 */
public enum UserState {
    ALIVE,
    DEAD,
    RESPAWNING,
    ELIMINATED
}
