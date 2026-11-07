package top.cnolm.gladiator.api.event;

import org.bukkit.entity.Player;
import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;
import top.cnolm.gladiator.api.Bot;

import java.util.Objects;

/**
 * Bot 真实登录、spawn 完成、BotHandle 注册完毕后在主线程发出。
 *
 * <p><strong>调用方在这个事件里完成传送与给予套件</strong>，然后才调用
 * {@code GladiatorAPI.pushGame(...)}。Gladiator 不碰传送与背包。
 */
public final class GladiatorBotReadyEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final Bot bot;

    public GladiatorBotReadyEvent(Bot bot) {
        this.bot = Objects.requireNonNull(bot, "bot");
    }

    public Bot getBot() {
        return bot;
    }

    /** 真实登录得到的 Bukkit Player，调用方可直接传送/给套件。 */
    public Player getPlayer() {
        return bot.asPlayer();
    }

    @Override
    public HandlerList getHandlers() {
        return HANDLERS;
    }

    public static HandlerList getHandlerList() {
        return HANDLERS;
    }
}
