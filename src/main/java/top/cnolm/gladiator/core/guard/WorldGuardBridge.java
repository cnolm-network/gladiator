package top.cnolm.gladiator.core.guard;

import com.sk89q.worldguard.bukkit.WGBukkit;
import com.sk89q.worldguard.protection.ApplicableRegionSet;
import com.sk89q.worldguard.protection.flags.DefaultFlag;
import com.sk89q.worldguard.protection.managers.RegionManager;
import com.sk89q.worldguard.protection.regions.ProtectedRegion;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import org.bukkit.Location;
import top.cnolm.gladiator.api.PvpVerdict;
import top.cnolm.gladiator.api.User;

/**
 * WorldGuard（legacy 6.x）只读桥接（G-203 / G-204）。
 *
 * <p>只读取区域与 {@code pvp} flag：不写区域、不取消事件、不引用任何竞技场插件类型。
 * 区域显式声明 pvp 时来源为 {@code flag}，否则为 {@code global}；
 * 世界未加载或取不到 RegionManager 时返回 {@code unknown}（allow，调用方自行记录）。
 *
 * <p>区域边界通过反射读取：WorldGuard 的 {@code getMinimumPoint()} 返回 WorldEdit 的
 * {@code BlockVector}，而本项目只显式依赖 WorldGuard API 面（WorldEdit 仅运行时由 WG 提供），
 * 因此不在编译期绑定 WorldEdit 类型；反射失败时 {@code box} 为 null，其余判定不受影响。
 */
public final class WorldGuardBridge implements GuardBridge {

    private static final Method GET_MIN_POINT = lookup(ProtectedRegion.class, "getMinimumPoint");
    private static final Method GET_MAX_POINT = lookup(ProtectedRegion.class, "getMaximumPoint");
    /**
     * {@code RegionManager#getApplicableRegions(Location)} 的签名里带 WorldEdit 的 Vector，
     * 编译期不可见；用反射拿到方法后强转回 WorldGuard 的 {@link ApplicableRegionSet}。
     */
    private static final Method GET_APPLICABLE =
            lookup(RegionManager.class, "getApplicableRegions", Location.class);

    private final Logger logger;

    public WorldGuardBridge(Logger logger) {
        this.logger = logger;
    }

    @Override
    public PvpVerdict query(Location location, User user) {
        if (location == null || location.getWorld() == null) {
            return PvpVerdict.unknown();
        }
        RegionManager manager;
        try {
            manager = WGBukkit.getRegionManager(location.getWorld());
        } catch (RuntimeException e) {
            logger.warning("WorldGuard 区域表不可用（" + location.getWorld().getName() + "）：" + e.getMessage());
            return PvpVerdict.unknown();
        } catch (LinkageError e) {
            logger.warning("WorldGuard 类加载失败，降级放行：" + e);
            return PvpVerdict.degraded();
        }
        if (manager == null) {
            return PvpVerdict.unknown();
        }

        ApplicableRegionSet set;
        try {
            if (GET_APPLICABLE == null) {
                return PvpVerdict.degraded();
            }
            set = (ApplicableRegionSet) GET_APPLICABLE.invoke(manager, location);
        } catch (ReflectiveOperationException e) {
            logger.warning("WorldGuard 区域查询失败（" + location + "）：" + e);
            return PvpVerdict.unknown();
        } catch (RuntimeException e) {
            logger.warning("WorldGuard 区域查询失败（" + location + "）：" + e.getMessage());
            return PvpVerdict.unknown();
        }
        if (set == null) {
            return PvpVerdict.unknown();
        }

        List<String> regionIds = new ArrayList<String>();
        boolean explicitFlag = false;
        double[] union = null;

        for (ProtectedRegion region : set) {
            regionIds.add(region.getId());
            if (region.getFlag(DefaultFlag.PVP) != null) {
                explicitFlag = true;
            }
            double[] bounds = boundsOf(region);
            if (bounds != null) {
                union = union == null ? bounds : union(union, bounds);
            }
        }

        boolean allowed = set.allows(DefaultFlag.PVP);
        PvpVerdict.Source source = explicitFlag ? PvpVerdict.Source.FLAG : PvpVerdict.Source.GLOBAL;
        PvpVerdict.Box box = union == null ? null
                : PvpVerdict.Box.of(union[0], union[1], union[2], union[3], union[4], union[5]);
        return PvpVerdict.of(allowed, regionIds, source, box);
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public String describe() {
        return "worldguard-legacy";
    }

    // ------------------------------------------------------------------
    // 反射读取 BlockVector：不把 WorldEdit 类型带进编译期
    // ------------------------------------------------------------------

    private static Method lookup(Class<?> owner, String name, Class<?>... parameters) {
        try {
            return owner.getMethod(name, parameters);
        } catch (Throwable t) {
            return null;
        }
    }

    private static double[] boundsOf(ProtectedRegion region) {
        if (GET_MIN_POINT == null || GET_MAX_POINT == null) {
            return null;
        }
        try {
            Object min = GET_MIN_POINT.invoke(region);
            Object max = GET_MAX_POINT.invoke(region);
            if (min == null || max == null) {
                return null;
            }
            return new double[] {
                    coordinate(min, "getX"), coordinate(min, "getY"), coordinate(min, "getZ"),
                    coordinate(max, "getX"), coordinate(max, "getY"), coordinate(max, "getZ")
            };
        } catch (Throwable t) {
            return null;
        }
    }

    private static double coordinate(Object vector, String getter) throws Exception {
        return ((Number) vector.getClass().getMethod(getter).invoke(vector)).doubleValue();
    }

    private static double[] union(double[] a, double[] b) {
        return new double[] {
                Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2]),
                Math.max(a[3], b[3]), Math.max(a[4], b[4]), Math.max(a[5], b[5])
        };
    }
}
