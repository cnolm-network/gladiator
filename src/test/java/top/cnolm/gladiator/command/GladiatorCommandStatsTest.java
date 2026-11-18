package top.cnolm.gladiator.command;

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
 * 块 B：`/gladiator stats <player>` 的路由分支。
 *
 * <p>命令本身是 Bukkit 的 {@code CommandExecutor}，起服务端代价太高；路由逻辑被抽成
 * {@link GladiatorCommand#statsResponse(Player)}（在线 → 单行遥测，离线 → 失败文案），
 * 这里直接覆盖这两个分支，保证 RCON 侧拿到的文本形状稳定。
 */
public class GladiatorCommandStatsTest {

    @Test
    public void offlinePlayerReturnsFailureText() {
        assertEquals("stats 失败：玩家不在线", GladiatorCommand.statsResponse(null));
    }

    @Test
    public void onlinePlayerReturnsSingleParsableLine() {
        PlayerInventory inventory = (PlayerInventory) Proxy.newProxyInstance(
                PlayerInventory.class.getClassLoader(),
                new Class<?>[]{PlayerInventory.class},
                (proxy, method, args) -> "getArmorContents".equals(method.getName())
                        ? new ItemStack[]{new ItemStack(Material.IRON_HELMET), new ItemStack(Material.IRON_BOOTS)}
                        : null);
        Location location = new Location(null, 10.5D, 64.0D, -3.25D, 45.0F, 0.0F);
        Player player = (Player) Proxy.newProxyInstance(
                Player.class.getClassLoader(),
                new Class<?>[]{Player.class},
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "getName":
                            return "Steve";
                        case "getHealth":
                            return 18.0D;
                        case "getFoodLevel":
                            return 20;
                        case "getSaturation":
                            return 5.0F;
                        case "getInventory":
                            return inventory;
                        case "getLocation":
                            return location;
                        case "isSprinting":
                            return false;
                        case "isBlocking":
                            return false;
                        case "isOnGround":
                            return true;
                        case "getNoDamageTicks":
                            return 0;
                        default:
                            return null;
                    }
                });

        String line = GladiatorCommand.statsResponse(player);
        assertEquals(
                "name=Steve health=18.0 food=20 saturation=5.0 armor=4.0"
                        + " x=10.5 y=64.0 z=-3.3 yaw=45.0 pitch=0.0"
                        + " sprinting=false blocking=false onGround=true hurt=0.0",
                line);
        // 单行、无前缀、无颜色：RCON 侧直接 split 即可。
        assertTrue(line.indexOf('\n') < 0);
        assertTrue(line.indexOf("[Gladiator]") < 0);
        assertTrue(line.indexOf('\u00a7') < 0);
    }
}
