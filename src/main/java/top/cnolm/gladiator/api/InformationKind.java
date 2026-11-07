package top.cnolm.gladiator.api;

/** {@link Information} 的类别，用于接收方做分类处理。 */
public enum InformationKind {
    /** 规则拒绝，例如 WorldGuard PVP deny。 */
    RULE_DENIED,
    /** 聊天栏摘要。 */
    CHAT,
    /** 对局事件（伤害、击杀、方块操作等）。 */
    MATCH_EVENT,
    /** 空闲窗口开启，允许策略后端发起外部检索。 */
    IDLE,
    /** 调用方自定义。 */
    CUSTOM
}
