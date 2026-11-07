package top.cnolm.gladiator.api;

/**
 * 由竞技场插件（调用方）实现的规则查询接口。
 *
 * <p>Gladiator 不实现任何对局规则，也不读取任何竞技场插件的状态。
 * 本接口只允许"只读查询"，不得写入状态；淘汰请调 {@link GladiatorAPI#eliminate(User)}。
 */
public interface Mode {

    /** 稳定标识，仅用于日志与观测，不参与规则判定。 */
    String getId();

    /** 该 User 死亡后是否允许重生。 */
    boolean allowsRespawn(User user);

    /** 会话超时（毫秒），0 表示不限时。 */
    long getTimeLimitMs();
}
