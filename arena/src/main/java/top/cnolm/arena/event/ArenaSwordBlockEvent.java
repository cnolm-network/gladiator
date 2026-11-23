package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.inventory.ItemStack;

import java.util.UUID;

/** 防砍：玩家右键举剑进入格挡姿态时触发，可取消。 */
public class ArenaSwordBlockEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final ItemStack sword;

    public ArenaSwordBlockEvent(Player player, UUID matchId, ItemStack sword) {
        super(player, matchId);
        this.sword = sword;
    }

    public ItemStack getSword() {
        return sword;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
