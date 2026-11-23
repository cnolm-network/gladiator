package top.cnolm.arena.event;

import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 对局内破坏方块，可取消。 */
public class ArenaBlockBreakEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Block block;

    public ArenaBlockBreakEvent(Player player, UUID matchId, Block block) {
        super(player, matchId);
        this.block = block;
    }

    public Block getBlock() {
        return block;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
