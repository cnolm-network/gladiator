package top.cnolm.gladiator.command;

import org.bukkit.entity.Player;
import org.junit.Test;

import java.lang.reflect.Proxy;

import static org.junit.Assert.assertEquals;

/**
 * `/gladiator respawn <player>` 的命令路由：离线 → 失败文案；在线未死亡 → 跳过。
 *
 * <p>「已请求」分支需要真实的 {@code Player$Spigot}（抽象类），由
 * {@code PlayerRespawnTest} 用注入动作覆盖。
 */
public class GladiatorCommandRespawnTest {

    private static Player player(final String name, final boolean dead) {
        return (Player) Proxy.newProxyInstance(
                Player.class.getClassLoader(),
                new Class<?>[]{Player.class},
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "getName":
                            return name;
                        case "isDead":
                            return dead;
                        case "toString":
                            return "FakePlayer{" + name + "}";
                        case "hashCode":
                            return System.identityHashCode(proxy);
                        case "equals":
                            return proxy == args[0];
                        default:
                            return null;
                    }
                });
    }

    @Test
    public void offlinePlayerReturnsFailureText() {
        assertEquals("respawn 失败：玩家不在线", GladiatorCommand.respawnResponse(null));
    }

    @Test
    public void alivePlayerReturnsSkipText() {
        assertEquals("respawn 跳过：Steve 未死亡",
                GladiatorCommand.respawnResponse(player("Steve", false)));
    }
}
