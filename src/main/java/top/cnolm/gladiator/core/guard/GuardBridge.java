package top.cnolm.gladiator.core.guard;

import org.bukkit.Location;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.User;

/**
 * 规则查询桥接：把 WorldGuard 的 PVP 判定收敛为一个可替换的小接口。
 *
 * <p>实现必须是只读的——不得写区域、不得取消任何 Bukkit 事件（ADR-015 / AGENTS.md）。
 * 调用方在主线程调用；实现内部不做缓存（缓存由 Layer 2 的 RestrictionCache 负责）。
 */
public interface GuardBridge {

    /** 查询坐标处的 PVP 判定；永不返回 null。 */
    PvpVerdict query(Location location, User user);

    /** 本桥接是否能给出真实判定（false 表示降级实现）。 */
    boolean isAvailable();

    /** 用于日志/features 的简短描述，例如 {@code worldguard-legacy} 或 {@code noop}。 */
    String describe();
}
