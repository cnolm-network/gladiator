package top.cnolm.gladiator.api.board;

/** 玩家状态快照（不可变，只含基础类型）。判定用事件，这里只做展示/兜底。 */
public final class PlayerSnapshot {

    private final String name;
    private final String uuid;
    private final boolean online;
    private final double health;
    private final int food;
    private final float saturation;
    private final double armor;
    private final double x;
    private final double y;
    private final double z;
    private final float yaw;
    private final float pitch;
    private final boolean onGround;
    private final boolean blocking;
    private final boolean sprinting;
    private final int hurtTicks;

    public PlayerSnapshot(String name, String uuid, boolean online, double health, int food,
                          float saturation, double armor, double x, double y, double z,
                          float yaw, float pitch, boolean onGround, boolean blocking,
                          boolean sprinting, int hurtTicks) {
        this.name = name;
        this.uuid = uuid;
        this.online = online;
        this.health = health;
        this.food = food;
        this.saturation = saturation;
        this.armor = armor;
        this.x = x;
        this.y = y;
        this.z = z;
        this.yaw = yaw;
        this.pitch = pitch;
        this.onGround = onGround;
        this.blocking = blocking;
        this.sprinting = sprinting;
        this.hurtTicks = hurtTicks;
    }

    /** 玩家不在线时的占位快照。 */
    public static PlayerSnapshot offline(String name) {
        return new PlayerSnapshot(name, "", false, 0.0D, 0, 0.0F, 0.0D,
                0.0D, 0.0D, 0.0D, 0.0F, 0.0F, false, false, false, 0);
    }

    public String getName() {
        return name;
    }

    public String getUuid() {
        return uuid;
    }

    public boolean isOnline() {
        return online;
    }

    public double getHealth() {
        return health;
    }

    public int getFood() {
        return food;
    }

    public float getSaturation() {
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

    public float getYaw() {
        return yaw;
    }

    public float getPitch() {
        return pitch;
    }

    public boolean isOnGround() {
        return onGround;
    }

    public boolean isBlocking() {
        return blocking;
    }

    public boolean isSprinting() {
        return sprinting;
    }

    public int getHurtTicks() {
        return hurtTicks;
    }

    /** 水平距离（只比 x/z，不含垂直分量）。 */
    public double horizontalDistance(PlayerSnapshot other) {
        double dx = x - other.x;
        double dz = z - other.z;
        return Math.sqrt(dx * dx + dz * dz);
    }

    @Override
    public String toString() {
        return "PlayerSnapshot{" + name + " online=" + online + " health=" + health
                + " armor=" + armor + " pos=" + x + "," + y + "," + z + '}';
    }
}
