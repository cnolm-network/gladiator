package top.cnolm.gladiator.core;

import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

/**
 * {@code spawnBot} 的来源抽象（B1）。
 *
 * <p>Gladiator 只关心「Bot 从哪里来」，不关心它怎么被创建：
 * <ul>
 *   <li>{@link BotSpawnService} —— 外部 mineflayer 运行时经通道 A 真实登录（默认，{@code spawn.mode=runtime}）；</li>
 *   <li>{@code top.cnolm.gladiator.core.golem.GolemBodySpawner} —— Golem 插件在进程内生成的
 *       NMS 身体（{@code spawn.mode=golem}，不登录、不依赖 bot-runtime）；</li>
 *   <li>{@link UnavailableSpawner} —— 配置要求 golem 但服务不可用时的占位实现，只报错不伪造。</li>
 * </ul>
 *
 * <p>实现方负责：状态机迁移的唯一写入点在 {@link GladiatorServiceImpl}，
 * 本接口只签发票据、维护注册表、并把 NOTICE / STATE 推给自身来源的 Bot。
 *
 * <p>所有方法都要求主线程调用（链路回调需自行切主线程后再进来）。
 */
public interface BotSpawner {

    /**
     * 发起一次 spawn，立即返回票据。
     *
     * @throws IllegalStateException 参数/重复等同步错误；异步失败通过
     *         {@code GladiatorBotFailedEvent} 通知
     */
    BotRequest spawn(BotSpec spec);

    /** {@code WELCOME.payload.playerUuid}：未知名返回 {@code null}。 */
    String playerUuid(String username);

    /** 当前等待完成的 spawn 请求数。 */
    int getPendingCount();

    /** 与 {@link GladiatorServiceImpl} 共享的 Bot 注册表。 */
    BotRegistry getRegistry();

    /**
     * 直接向该 Bot 的运行时推送 NOTICE。
     *
     * @return false 表示该来源没有下发通道，调用方应回落到本地队列
     */
    boolean sendNotice(String botId, Information information);

    /** 推送 STATE 事件；该来源没有通道时为空实现（状态事件照常走 Bukkit 事件）。 */
    void sendState(User user, UserState previous, UserState state, String reason);
}
