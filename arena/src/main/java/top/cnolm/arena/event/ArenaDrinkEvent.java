package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.inventory.ItemStack;

import java.util.UUID;

/** 饮用：由 Consume + Interact 重组而来，只表达"喝药水/桶装液体"。 */
public class ArenaDrinkEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final ItemStack drink;

    public ArenaDrinkEvent(Player player, UUID matchId, ItemStack drink) {
        super(player, matchId);
        this.drink = drink;
    }

    public ItemStack getDrink() {
        return drink;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
