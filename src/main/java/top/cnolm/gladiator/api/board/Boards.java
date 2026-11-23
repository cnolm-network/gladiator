package top.cnolm.gladiator.api.board;

/**
 * 板子注册表的静态入口。
 *
 * <p>放在静态类里而不是 {@code GladiatorAPI} 的方法上，是为了不破坏既有实现
 * （API 只增不改；新增一个接口方法会让所有外部实现类编译失败）。
 *
 * <p>用法（板子插件 onEnable）：
 * <pre>{@code
 * Boards.registry().register(new FlatArenaBoard(this));
 * }</pre>
 */
public final class Boards {

    private static volatile BoardRegistry registry;

    private Boards() {
    }

    /** 返回注册表；Gladiator 未启用时返回 {@code null}。 */
    public static BoardRegistry registry() {
        return registry;
    }

    /** 仅由 Gladiator 内部调用。 */
    public static void install(BoardRegistry value) {
        registry = value;
    }
}
