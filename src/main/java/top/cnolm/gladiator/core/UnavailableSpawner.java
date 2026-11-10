package top.cnolm.gladiator.core;

import top.cnolm.gladiator.api.BotRequest;
import top.cnolm.gladiator.api.BotSpec;
import top.cnolm.gladiator.api.Information;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

/**
 * 配置要求 golem 模式但服务不可用时的占位 spawner。
 *
 * <p>存在的意义是<strong>让失败可解释</strong>：插件照常启用，{@code /gladiator status} 能显示
 * {@code spawn=golem}，{@code /gladiator spawn X} 返回一句中文错误而不是
 * {@code IllegalStateException: spawnBot 尚未接线} 或空指针。
 *
 * <p>不伪造任何 Bot、不返回假成功票据：{@link #spawn} 直接抛异常。
 */
public final class UnavailableSpawner implements BotSpawner {

    private final String reason;
    private final BotRegistry registry = new BotRegistry();

    public UnavailableSpawner(String reason) {
        this.reason = reason == null ? "spawn 来源不可用" : reason;
    }

    public String getReason() {
        return reason;
    }

    @Override
    public BotRequest spawn(BotSpec spec) {
        throw new IllegalStateException("spawnBot 不可用：" + reason);
    }

    /** 无 Bot 来源，永远解析不出 UUID。 */
    @Override
    public String playerUuid(String username) {
        return null;
    }

    @Override
    public int getPendingCount() {
        return 0;
    }

    /** 空注册表：{@code /gladiator list} 显示 0 而不是 NPE。 */
    @Override
    public BotRegistry getRegistry() {
        return registry;
    }

    /** 没有下发通道，调用方回落到本地队列。 */
    @Override
    public boolean sendNotice(String botId, Information information) {
        return false;
    }

    @Override
    public void sendState(User user, UserState previous, UserState state, String reason) {
        // 无通道：状态事件照常由 GladiatorServiceImpl 派发 Bukkit 事件。
    }
}
