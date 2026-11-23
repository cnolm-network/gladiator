package top.cnolm.gladiator.api.board;

import org.bukkit.entity.Player;

/**
 * 把「名字」解析成 {@link Player}。
 *
 * <p>存在的唯一理由：mineflayer Bot 与真实玩家都是 {@code CraftPlayer}，但 golem 的 Bot 身体
 * 是进程内 NMS 实体、不在 {@code PlayerList} 里，{@code Bukkit.getPlayerExact} 拿不到，
 * 只有 Gladiator 的 BotRegistry 能把它对应的 {@code Player} 交出来。
 *
 * <p>实现必须是主线程安全的读操作，不得触发世界写入。
 */
public interface ParticipantResolver {

    /** 解析不到返回 {@code null}。 */
    Player resolve(String username);
}
