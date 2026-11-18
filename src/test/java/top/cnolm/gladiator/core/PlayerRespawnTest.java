package top.cnolm.gladiator.core;

import org.bukkit.entity.Player;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * 块 B 续：强制重生（`/gladiator respawn`）的判定与文案。
 *
 * <p>`Player$Spigot` 是抽象类没法用 Proxy 伪造，所以实现里把「真正调 respawn」抽成
 * {@link PlayerRespawn.Action}，这里注入假动作验证三条分支。
 */
public class PlayerRespawnTest {

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
    public void offlinePlayerIsReported() {
        assertEquals("respawn 失败：玩家不在线", PlayerRespawn.request(null, null));
        assertEquals("respawn 失败：玩家不在线", PlayerRespawn.request(null));
    }

    @Test
    public void alivePlayerIsSkippedWithoutCallingAction() {
        List<String> calls = new ArrayList<>();
        String result = PlayerRespawn.request(player("Steve", false), p -> calls.add(p.getName()));
        assertEquals("respawn 跳过：Steve 未死亡", result);
        assertTrue(calls.isEmpty());
        assertFalse(PlayerRespawn.isDead(player("Steve", false)));
    }

    @Test
    public void deadPlayerTriggersActionAndReportsRequested() {
        List<String> calls = new ArrayList<>();
        Player dead = player("Steve", true);
        String result = PlayerRespawn.request(dead, p -> calls.add(p.getName()));
        assertEquals("respawn 已请求：Steve", result);
        assertEquals(1, calls.size());
        assertEquals("Steve", calls.get(0));
        assertTrue(PlayerRespawn.isDead(dead));
    }

    @Test
    public void actionFailureDoesNotThrow() {
        Player dead = player("Steve", true);
        String result = PlayerRespawn.request(dead, p -> {
            throw new IllegalStateException("server busy");
        });
        assertEquals("respawn 失败：server busy", result);
    }

    @Test
    public void nullActionFallsBackToFailureText() {
        assertEquals("respawn 失败：Steve", PlayerRespawn.request(player("Steve", true), null));
    }
}
