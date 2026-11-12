package top.cnolm.gladiator.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 一次 PVP 规则查询的结果：是否允许、命中的区域、判定来源，以及可选的三维边界。
 *
 * <p>来源语义见 {@code docs/protocol.md} §2.6：
 * <ul>
 *   <li>{@link Source#FLAG}：命中的 WorldGuard 区域显式设置了 pvp flag；</li>
 *   <li>{@link Source#GLOBAL}：无区域覆盖或区域未设置该 flag，回落到全局默认；</li>
 *   <li>{@link Source#DEGRADED}：未安装 WorldGuard，按 allow 降级；</li>
 *   <li>{@link Source#UNKNOWN}：无法判定（世界未加载等），按 allow 放行并由调用方记录；</li>
 *   <li>{@link Source#OBSERVED}：来自<b>观测到的拒绝</b>（A3）——别的插件取消过该 Bot 的
 *       攻击/交互，Gladiator 据此在短期内直接判 deny，不再查询 WorldGuard。</li>
 * </ul>
 *
 * <p>不可变对象，可跨线程读取。
 */
public final class PvpVerdict {

    /** 判定来源。 */
    public enum Source {
        FLAG,
        GLOBAL,
        DEGRADED,
        UNKNOWN,
        /** 观测式判定：近期该 Bot 在同一区域被别的插件拒绝过（见 {@code ObservedDenials}）。 */
        OBSERVED
    }

    /** 轴对齐三维包围盒（世界坐标，含边界）。 */
    public static final class Box {

        private final double minX;
        private final double minY;
        private final double minZ;
        private final double maxX;
        private final double maxY;
        private final double maxZ;

        public Box(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) {
            this.minX = Math.min(minX, maxX);
            this.minY = Math.min(minY, maxY);
            this.minZ = Math.min(minZ, maxZ);
            this.maxX = Math.max(minX, maxX);
            this.maxY = Math.max(minY, maxY);
            this.maxZ = Math.max(minZ, maxZ);
        }

        public static Box of(double minX, double minY, double minZ,
                             double maxX, double maxY, double maxZ) {
            return new Box(minX, minY, minZ, maxX, maxY, maxZ);
        }

        public double getMinX() {
            return minX;
        }

        public double getMinY() {
            return minY;
        }

        public double getMinZ() {
            return minZ;
        }

        public double getMaxX() {
            return maxX;
        }

        public double getMaxY() {
            return maxY;
        }

        public double getMaxZ() {
            return maxZ;
        }

        @Override
        public String toString() {
            return "Box(" + minX + "," + minY + "," + minZ + ")-(" + maxX + "," + maxY + "," + maxZ + ")";
        }
    }

    private final boolean allowed;
    private final List<String> regionIds;
    private final Source source;
    private final Box box;

    private PvpVerdict(boolean allowed, List<String> regionIds, Source source, Box box) {
        this.allowed = allowed;
        this.regionIds = Collections.unmodifiableList(new ArrayList<>(regionIds == null
                ? Collections.<String>emptyList() : regionIds));
        this.source = Objects.requireNonNull(source, "source");
        this.box = box;
    }

    public static PvpVerdict of(boolean allowed, List<String> regionIds, Source source, Box box) {
        return new PvpVerdict(allowed, regionIds, source, box);
    }

    /** 未安装 WorldGuard 时的降级结果：允许，来源 DEGRADED。 */
    public static PvpVerdict degraded() {
        return new PvpVerdict(true, Collections.<String>emptyList(), Source.DEGRADED, null);
    }

    /** 无法判定时的结果：允许，来源 UNKNOWN。 */
    public static PvpVerdict unknown() {
        return new PvpVerdict(true, Collections.<String>emptyList(), Source.UNKNOWN, null);
    }

    public boolean isAllowed() {
        return allowed;
    }

    public List<String> getRegionIds() {
        return regionIds;
    }

    public Source getSource() {
        return source;
    }

    /** 命中的区域并集边界；无区域或无 WorldGuard 时为 null。 */
    public Box getBox() {
        return box;
    }

    @Override
    public String toString() {
        return "PvpVerdict(allowed=" + allowed + ", source=" + source
                + ", regions=" + regionIds + ", box=" + box + ")";
    }
}
