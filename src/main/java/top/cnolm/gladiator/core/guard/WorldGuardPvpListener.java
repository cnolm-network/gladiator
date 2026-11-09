package top.cnolm.gladiator.core.guard;

import com.sk89q.worldguard.protection.events.DisallowedPVPEvent;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;

/**
 * 被动规则通知（G-203）：监听 WorldGuard 的 {@link DisallowedPVPEvent}，把拒绝事实转成 NOTICE。
 *
 * <p><strong>绝不调用 {@code setCancelled} / {@code setCancelled(false)}</strong>：
 * 该事件可被其他插件用于放行 PVP，Gladiator 只观察，不参与裁决（AGENTS.md 强制约束）。
 */
public final class WorldGuardPvpListener implements Listener {

    /** 拒绝事件的接收方，由 core 实现（转 NOTICE + 计数）。 */
    public interface Sink {
        void onPvpDenied(Player attacker, Player defender, String cause);
    }

    private final Sink sink;

    public WorldGuardPvpListener(Sink sink) {
        this.sink = sink;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = false)
    public void onDisallowedPvp(DisallowedPVPEvent event) {
        if (event == null) {
            return;
        }
        Player attacker = event.getAttacker();
        Player defender = event.getDefender();
        String cause = event.getCause() == null ? "unknown" : event.getCause().getEventName();
        sink.onPvpDenied(attacker, defender, cause);
        // 刻意不触碰 event.setCancelled(...)：本插件只观察。
    }
}
