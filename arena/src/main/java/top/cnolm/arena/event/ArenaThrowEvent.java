package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.inventory.ItemStack;

import java.util.UUID;

/** 投掷物品：从 PlayerInteractEvent 拆出的"扔出"语义（珍珠、雪球等）。 */
public class ArenaThrowEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final ItemStack item;

    public ArenaThrowEvent(Player player, UUID matchId, ItemStack item) {
        super(player, matchId);
        this.item = item;
    }

    public ItemStack getItem() {
        return item;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
