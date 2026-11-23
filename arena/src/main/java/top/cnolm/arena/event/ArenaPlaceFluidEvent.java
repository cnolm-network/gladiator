package top.cnolm.arena.event;

import org.bukkit.Material;
import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.event.HandlerList;

import java.util.UUID;

/** 放置流体：由倒桶触发，携带流体类型，可取消。 */
public class ArenaPlaceFluidEvent extends CancellableArenaEvent {
    private static final HandlerList handlers = new HandlerList();
    private final Block block;
    private final Material fluid;

    public ArenaPlaceFluidEvent(Player player, UUID matchId, Block block, Material fluid) {
        super(player, matchId);
        this.block = block;
        this.fluid = fluid;
    }

    public Block getBlock() {
        return block;
    }

    public Material getFluid() {
        return fluid;
    }

    @Override
    public HandlerList getHandlers() {
        return handlers;
    }

    public static HandlerList getHandlerList() {
        return handlers;
    }
}
