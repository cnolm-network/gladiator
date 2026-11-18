package top.cnolm.gladiator.core;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;

import java.util.Locale;

/**
 * 只读玩家遥测快照（块 B）。
 *
 * <p>用途：Dashboard「人机对战」每 1 秒用 RCON 调 {@code /gladiator stats <player>} 采样双方状态，
 * 判定死亡/超时并录制帧；Bot 运行时也用同一命令补 {@code target_health}（mineflayer 拿不到
 * 其他玩家血量）。因此**输出必须是稳定单行、键值可解析**，且不能带任何颜色/前缀。
 *
 * <p>边界：
 * <ul>
 *   <li>只读：不传送、不修改血量/背包/状态，不注册任何监听器；</li>
 *   <li>不依赖 NMS，也不引入反作弊/领地依赖（ADR-015）；</li>
 *   <li>不含对局规则：谁赢谁输由调用方（Dashboard 编排层）判断。</li>
 * </ul>
 *
 * <p>字段语义（顺序固定）：
 * <pre>
 * name=&lt;名字&gt; health=&lt;血量&gt; food=&lt;饱食度&gt; saturation=&lt;饱和度&gt; armor=&lt;护甲点数&gt;
 * x=&lt;X&gt; y=&lt;Y&gt; z=&lt;Z&gt; yaw=&lt;偏航角&gt; pitch=&lt;俯仰角&gt;
 * sprinting=&lt;bool&gt; blocking=&lt;bool&gt; onGround=&lt;bool&gt; hurt=&lt;受击剩余无敌秒数&gt;
 * </pre>
 *
 * <p>{@code hurt} 取 {@code Entity#getNoDamageTicks() / 10.0}：受击后原版会给 10 tick（0.5s）
 * 无敌帧，该值大于 0 表示"刚刚被打过"，0 表示没有受击记录。它是**只读**的派生量，
 * 不需要维护任何额外状态。
 */
public final class PlayerTelemetry {

    private static final double TICKS_PER_SECOND = 10.0D;

    /** 找不到玩家时的统一文案（Dashboard / 运行时按子串判定）。 */
    public static final String OFFLINE_MESSAGE = "stats 失败：玩家不在线";

    private PlayerTelemetry() {
    }

    /** 一次采样的不可变快照。 */
    public static final class Snapshot {
        private final String name;
        private final double health;
        private final int food;
        private final double saturation;
        private final double armor;
        private final double x;
        private final double y;
        private final double z;
        private final double yaw;
        private final double pitch;
        private final boolean sprinting;
        private final boolean blocking;
        private final boolean onGround;
        private final double hurt;

        Snapshot(Player player, Location location) {
            this.name = player.getName();
            this.health = player.getHealth();
            this.food = player.getFoodLevel();
            this.saturation = player.getSaturation();
            this.armor = armorPoints(player.getInventory());
            this.x = location.getX();
            this.y = location.getY();
            this.z = location.getZ();
            this.yaw = location.getYaw();
            this.pitch = location.getPitch();
            this.sprinting = player.isSprinting();
            this.blocking = player.isBlocking();
            this.onGround = player.isOnGround();
            this.hurt = player.getNoDamageTicks() / TICKS_PER_SECOND;
        }

        public String getName() {
            return name;
        }

        public double getHealth() {
            return health;
        }

        public int getFood() {
            return food;
        }

        public double getSaturation() {
            return saturation;
        }

        public double getArmor() {
            return armor;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        public double getZ() {
            return z;
        }

        public double getYaw() {
            return yaw;
        }

        public double getPitch() {
            return pitch;
        }

        public boolean isSprinting() {
            return sprinting;
        }

        public boolean isBlocking() {
            return blocking;
        }

        public boolean isOnGround() {
            return onGround;
        }

        public double getHurt() {
            return hurt;
        }

        /** 单行文本（字段顺序固定，供 RCON 解析）。 */
        public String toLine() {
            return "name=" + name
                    + " health=" + decimal(health)
                    + " food=" + food
                    + " saturation=" + decimal(saturation)
                    + " armor=" + decimal(armor)
                    + " x=" + decimal(x)
                    + " y=" + decimal(y)
                    + " z=" + decimal(z)
                    + " yaw=" + decimal(yaw)
                    + " pitch=" + decimal(pitch)
                    + " sprinting=" + sprinting
                    + " blocking=" + blocking
                    + " onGround=" + onGround
                    + " hurt=" + decimal(hurt);
        }

        @Override
        public String toString() {
            return toLine();
        }
    }

    /** 采样一个在线玩家（只读）。 */
    public static Snapshot of(Player player) {
        if (player == null) {
            throw new IllegalArgumentException("player 不能为空");
        }
        Location location = player.getLocation();
        return new Snapshot(player, location);
    }

    /** 直接返回单行文本；{@code player == null} 时返回 {@link #OFFLINE_MESSAGE}。 */
    public static String format(Player player) {
        return player == null ? OFFLINE_MESSAGE : of(player).toLine();
    }

    /**
     * 护甲点数（1.8 原版防御值：头盔/胸甲/护腿/靴子）。
     *
     * <p>用材质**名称**判断而不是枚举常量，避免不同服务端版本的枚举差异。
     * 总上限 20 点（钻石全套 3+8+6+3）。
     */
    static double armorPoints(PlayerInventory inventory) {
        if (inventory == null) {
            return 0.0D;
        }
        ItemStack[] armor = inventory.getArmorContents();
        if (armor == null || armor.length == 0) {
            return 0.0D;
        }
        double total = 0.0D;
        for (ItemStack piece : armor) {
            total += defensePoints(piece == null ? null : piece.getType());
        }
        return total;
    }

    /** 单件护甲防御值；非护甲返回 0。 */
    static double defensePoints(Material material) {
        if (material == null) {
            return 0.0D;
        }
        String name = material.name();
        double[] table;
        if (name.startsWith("DIAMOND_")) {
            table = new double[]{3, 8, 6, 3};
        } else if (name.startsWith("IRON_")) {
            table = new double[]{2, 6, 5, 2};
        } else if (name.startsWith("CHAINMAIL_")) {
            table = new double[]{2, 5, 4, 1};
        } else if (name.startsWith("GOLD_") || name.startsWith("GOLDEN_")) {
            table = new double[]{2, 5, 3, 1};
        } else if (name.startsWith("LEATHER_")) {
            table = new double[]{1, 3, 2, 1};
        } else {
            return 0.0D;
        }
        if (name.endsWith("_HELMET")) {
            return table[0];
        }
        if (name.endsWith("_CHESTPLATE")) {
            return table[1];
        }
        if (name.endsWith("_LEGGINGS")) {
            return table[2];
        }
        if (name.endsWith("_BOOTS")) {
            return table[3];
        }
        return 0.0D;
    }

    private static String decimal(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return "0.0";
        }
        return String.format(Locale.ROOT, "%.1f", value);
    }
}
