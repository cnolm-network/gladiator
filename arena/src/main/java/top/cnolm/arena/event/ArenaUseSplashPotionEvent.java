package top.cnolm.arena.event;

import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;
import org.bukkit.inventory.ItemStack;

import java.util.UUID;

/** 使用喷溅药水：从 PlayerInteractEvent 拆出的"泼药水"语义。 */
public class ArenaUseSplashPotionEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final ItemStack potion;

    public ArenaUseSplashPotionEvent(Player player, UUID matchId, ItemStack potion) {
        super(player, matchId);
        this.potion = potion;
    }

    public ItemStack getPotion() {
        return potion;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
