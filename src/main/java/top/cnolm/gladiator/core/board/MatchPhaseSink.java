package top.cnolm.gladiator.core.board;

/**
 * {@code MATCH_PHASE} 帧的下发通道。
 *
 * <p>为什么要有这一层：{@link BoardRegistryImpl} 只该知道「什么时候推」，不该知道
 * 「怎么发」——真正发帧要拿通道 A 的 {@code LinkServer} 与 {@code BotRegistry}，
 * 两者都在 {@code GladiatorMain} 手里，而注册表在 {@code startLink()} 之前就已经建好
 * （链路端口可能绑不上，此时推帧只能是空操作）。
 *
 * <p>接口只收**参与者用户名**，不收 botId：用户名 → botId 的换算（以及「这个名字是不是
 * 本插件的 Bot」）由实现方用 {@code BotRegistry} 判定，注册表不重复这套规则。
 *
 * <p>线程：只在主线程调用（会话的 {@code phase()} 与 10Hz ticker 都在主线程）。
 */
public interface MatchPhaseSink {

    /**
     * 给一个参与者推一帧 {@code MATCH_PHASE}；不是本插件 Bot 或链路未就绪时静默跳过。
     *
     * @param username 参与者用户名（真实玩家也会传进来，实现方应自行过滤）
     * @param phase    {@code starting|countdown|fighting|ended}
     * @param round    当前局号（1 起）
     * @param seconds  本阶段剩余秒数；{@code -1} 表示板子没声明时长（如 {@code starting}）
     * @param endsAt   本阶段结束的墙钟时间（epoch 毫秒）；{@code 0} 表示未知
     */
    void sendMatchPhase(String username, String phase, int round, int seconds, long endsAt);
}
