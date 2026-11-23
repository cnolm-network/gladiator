package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.inventory.ItemStack;

import java.util.UUID;

/** 进食：由 Consume + Interact 重组而来，只表达"吃固体食物"。 */
public class ArenaEatEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final ItemStack food;

    public ArenaEatEvent(Player player, UUID matchId, ItemStack food) {
        super(player, matchId);
        this.food = food;
    }

    public ItemStack getFood() {
        return food;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
