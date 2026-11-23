package top.cnolm.gladiator.api.board;

/**
 * 板子（对局插件）能收到的服务器事件类型。
 *
 * <p>这里的每一项都由 Gladiator 在主线程从对应的 Bukkit 事件翻译而来，板子只读不改：
 * 板子用它们做「谁死了、谁重生了、谁打中了谁」，再自己判定回合胜负。
 * 板子<strong>不</strong>通过这里获得或下发 Bot 的动作——动作只走通道 B / 训练环境。
 */
public enum GameEventType {

    /** 玩家（真人与 Bot 一视同仁）进入服务器。 */
    PLAYER_JOIN,
    /** 玩家离开服务器；对局中 Bot 掉线按失败语义处理（该局判负）。 */
    PLAYER_QUIT,
    /** 玩家死亡；{@code data.deathMessage} 为死亡消息，{@code data.killer} 可为空。 */
    PLAYER_DEATH,
    /** 玩家重生完成（1.8 的死亡到重生只有约 2 tick，必须靠事件而不是轮询）。 */
    PLAYER_RESPAWN,
    /** 玩家被传送；{@code data.cause} 为原因字符串（板子自己触发的也会收到）。 */
    PLAYER_TELEPORT,
    /** 实体受伤；{@code actor} 为攻击者名（可能为空），{@code target} 为受伤者名。 */
    ENTITY_DAMAGE,
    /** 实体死亡（怪物/玩家统一）；玩家死亡请优先用 {@link #PLAYER_DEATH}。 */
    ENTITY_KILL,
    /** 投掷物命中（鱼竿、雪球、箭）。 */
    PROJECTILE_HIT,
    /** 拾取物品。 */
    ITEM_PICKUP,
    /** 丢弃物品。 */
    ITEM_DROP,
    /** 消耗物品（吃东西、喝药）。 */
    ITEM_CONSUME,
    /** Bot 会话就绪（WELCOME 完成），板子可以开始对它发指令。 */
    BOT_READY,
    /** Bot 会话断开。 */
    BOT_OFFLINE
}
