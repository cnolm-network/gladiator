package top.cnolm.gladiator.core;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.junit.Test;

import java.lang.reflect.Proxy;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * 块 B：只读玩家遥测快照。
 *
 * <p>用 {@link Proxy} 伪造 Bukkit 接口，不启动服务端；只验证格式化、护甲点数与只读边界。
 */
public class PlayerTelemetryTest {

    private static ItemStack armor(Material material) {
        // ItemStack 是具体类，1.8.8 里 new ItemStack(Material) 不会触碰服务端。
        return material == null ? null : new ItemStack(material);
    }

    private static Player player(final String name, final double health, final int food,
                                 final float saturation, final ItemStack[] armor,
                                 final double x, final double y, final double z,
                                 final float yaw, final float pitch,
                                 final boolean sprinting, final boolean blocking,
                                 final boolean onGround, final int noDamageTicks) {
        final PlayerInventory inventory = (PlayerInventory) Proxy.newProxyInstance(
                PlayerInventory.class.getClassLoader(),
                new Class<?>[]{PlayerInventory.class},
                (proxy, method, args) -> {
                    if ("getArmorContents".equals(method.getName())) {
                        return armor;
                    }
                    if ("toString".equals(method.getName())) {
                        return "FakeInventory";
                    }
                    if ("hashCode".equals(method.getName())) {
                        return System.identityHashCode(proxy);
                    }
                    if ("equals".equals(method.getName())) {
                        return proxy == args[0];
                    }
                    return null;
                });
        final Location location = new Location(null, x, y, z, yaw, pitch);
        return (Player) Proxy.newProxyInstance(
                Player.class.getClassLoader(),
                new Class<?>[]{Player.class},
                (proxy, method, args) -> {
                    String called = method.getName();
                    if ("getName".equals(called)) {
                        return name;
                    }
                    if ("getHealth".equals(called)) {
                        return health;
                    }
                    if ("getFoodLevel".equals(called)) {
                        return food;
                    }
                    if ("getSaturation".equals(called)) {
                        return saturation;
                    }
                    if ("getInventory".equals(called)) {
                        return inventory;
                    }
                    if ("getLocation".equals(called)) {
                        return location;
                    }
                    if ("isSprinting".equals(called)) {
                        return sprinting;
                    }
                    if ("isBlocking".equals(called)) {
                        return blocking;
                    }
                    if ("isOnGround".equals(called)) {
                        return onGround;
                    }
                    if ("getNoDamageTicks".equals(called)) {
                        return noDamageTicks;
                    }
                    if ("isDead".equals(called)) {
                        return health <= 0.0D;
                    }
                    if ("getActivePotionEffects".equals(called)) {
                        return new java.util.ArrayList<org.bukkit.potion.PotionEffect>();
                    }
                    if ("toString".equals(called)) {
                        return "FakePlayer{" + name + "}";
                    }
                    if ("hashCode".equals(called)) {
                        return System.identityHashCode(proxy);
                    }
                    if ("equals".equals(called)) {
                        return proxy == args[0];
                    }
                    return null;
                });
    }

    @Test
    public void lineHasFixedFieldsAndOrder() {
        ItemStack[] armor = new ItemStack[]{
                armor(Material.DIAMOND_HELMET), armor(Material.DIAMOND_CHESTPLATE),
                armor(Material.DIAMOND_LEGGINGS), armor(Material.DIAMOND_BOOTS),
        };
        Player player = player("Steve", 20.0D, 20, 5.0F, armor,
                0.5D, 4.0D, 0.5D, 0.0F, 0.0F, false, false, true, 0);

        assertEquals(
                "name=Steve health=20.0 food=20 saturation=5.0 armor=20.0"
                        + " x=0.5 y=4.0 z=0.5 yaw=0.0 pitch=0.0"
                        + " sprinting=false blocking=false onGround=true hurt=0.0"
                        + " dead=false effects=-",
                PlayerTelemetry.format(player));
    }

    @Test
    public void hurtIsInvulnerabilitySecondsAndBooleansFollowPlayer() {
        Player player = player("Alex", 13.37D, 17, 2.25F, new ItemStack[0],
                -12.34D, 64.0D, 7.5D, 90.0F, -45.0F, true, true, false, 7);

        String line = PlayerTelemetry.format(player);
        assertTrue(line, line.contains("health=13.4"));
        assertTrue(line, line.contains("saturation=2.3"));
        assertTrue(line, line.contains("x=-12.3"));
        assertTrue(line, line.contains("yaw=90.0"));
        assertTrue(line, line.contains("pitch=-45.0"));
        assertTrue(line, line.contains("sprinting=true"));
        assertTrue(line, line.contains("blocking=true"));
        assertTrue(line, line.contains("onGround=false"));
        assertTrue(line, line.contains("hurt=0.7"));
        assertTrue(line, line.contains("armor=0.0"));
    }

    @Test
    public void armorPointsFollowVanillaTable() {
        assertEquals(20.0D, PlayerTelemetry.armorPoints(inventoryOf(
                Material.DIAMOND_HELMET, Material.DIAMOND_CHESTPLATE,
                Material.DIAMOND_LEGGINGS, Material.DIAMOND_BOOTS)), 1e-9);
        assertEquals(15.0D, PlayerTelemetry.armorPoints(inventoryOf(
                Material.IRON_HELMET, Material.IRON_CHESTPLATE,
                Material.IRON_LEGGINGS, Material.IRON_BOOTS)), 1e-9);
        assertEquals(7.0D, PlayerTelemetry.armorPoints(inventoryOf(
                Material.LEATHER_HELMET, Material.LEATHER_CHESTPLATE,
                Material.LEATHER_LEGGINGS, Material.LEATHER_BOOTS)), 1e-9);
        assertEquals(0.0D, PlayerTelemetry.armorPoints(inventoryOf(
                Material.DIAMOND_SWORD, null, null, null)), 1e-9);
        assertEquals(0.0D, PlayerTelemetry.armorPoints(null), 1e-9);
    }

    @Test
    public void defensePointsRejectNonArmor() {
        assertEquals(0.0D, PlayerTelemetry.defensePoints(null), 1e-9);
        assertEquals(0.0D, PlayerTelemetry.defensePoints(Material.AIR), 1e-9);
        assertEquals(0.0D, PlayerTelemetry.defensePoints(Material.STONE), 1e-9);
        assertEquals(8.0D, PlayerTelemetry.defensePoints(Material.DIAMOND_CHESTPLATE), 1e-9);
    }

    @Test
    public void offlinePlayerYieldsFailureText() {
        assertEquals("stats 失败：玩家不在线", PlayerTelemetry.format(null));
        assertEquals(PlayerTelemetry.OFFLINE_MESSAGE, PlayerTelemetry.format(null));
    }

    @Test(expected = IllegalArgumentException.class)
    public void ofRejectsNull() {
        PlayerTelemetry.of(null);
    }

    private static PlayerInventory inventoryOf(final Material... materials) {
        ItemStack[] armor = new ItemStack[materials.length];
        for (int i = 0; i < materials.length; i++) {
            armor[i] = armor(materials[i]);
        }
        return (PlayerInventory) Proxy.newProxyInstance(
                PlayerInventory.class.getClassLoader(),
                new Class<?>[]{PlayerInventory.class},
                (proxy, method, args) -> {
                    if ("getArmorContents".equals(method.getName())) {
                        return armor;
                    }
                    return null;
                });
    }
}
