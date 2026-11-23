package top.cnolm.arena.event;

import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 对局内丢弃物品，可取消。 */
public class ArenaItemDropEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Item item;

    public ArenaItemDropEvent(Player player, UUID matchId, Item item) {
        super(player, matchId);
        this.item = item;
    }

    public Item getItem() {
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
