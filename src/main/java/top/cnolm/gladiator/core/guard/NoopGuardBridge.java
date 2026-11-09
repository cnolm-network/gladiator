package top.cnolm.gladiator.core.guard;

import org.bukkit.Location;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.User;

/**
 * 未安装 WorldGuard 时的降级实现（G-205）：一律 allow，来源 {@code degraded}。
 *
 * <p>降级是显式契约的一部分：Gladiator 不因为缺少区域插件而拒绝训练，
 * 但每次查询都会把来源标成 degraded，Layer 2 可据此在日志里明示。
 */
public final class NoopGuardBridge implements GuardBridge {

    @Override
    public PvpVerdict query(Location location, User user) {
        return PvpVerdict.degraded();
    }

    @Override
    public boolean isAvailable() {
        return false;
    }

    @Override
    public String describe() {
        return "noop";
    }
}
