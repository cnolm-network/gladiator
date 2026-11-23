package top.cnolm.gladiator.api.board;

import java.util.Collection;

/** 板子注册表：板子在这里登记，外部（dashboard/命令）在这里发起对局。 */
public interface BoardRegistry {

    /** 注册板子；同一 id 已存在时抛 {@link IllegalArgumentException}。 */
    void register(Board board);

    /** 注销板子；正在进行的对局会被终止。 */
    void unregister(String boardId);

    Collection<String> ids();

    Board get(String boardId);

    /** 当前进行中的对局会话；没有则返回 {@code null}。 */
    BoardContext active();

    /**
     * 用指定板子开一场对局。
     *
     * @throws IllegalStateException    已有一场对局在进行
     * @throws IllegalArgumentException 板子不存在
     */
    BoardContext start(String boardId, RoundSpec spec);

    /** 终止当前对局；{@code reason} 会传给 {@link Board#onStop(String)}。 */
    void stop(String reason);
}
