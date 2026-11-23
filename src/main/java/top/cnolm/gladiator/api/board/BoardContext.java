package top.cnolm.gladiator.api.board;

import org.bukkit.entity.Player;
import top.cnolm.gladiator.api.GladiatorAPI;

/**
 * 板子在一次对局里能用到的「Gladiator 专属」能力。
 *
 * <p>所有方法都必须在<strong>主线程</strong>调用；Gladiator 会直接抛
 * {@link IllegalStateException} 而不是静默排队——对局判定错一帧就会错一整局。
 * 其余一切（清场、套装、传送、药水、计分板）板子直接用 Bukkit API，不要包一层。
 * 谁什么时候该被清场、发什么套件、什么时候算赢，全部是板子自己的规则。
 */
public interface BoardContext {

    /** 本次对局会话 id（日志与外部录制用）。 */
    String sessionId();

    /** 板子 id（{@link Board#id()}）。 */
    String boardId();

    /** 开局参数。 */
    RoundSpec spec();

    /** 当前局号（1 起）。 */
    int round();

    /** 会话开始以来的 10Hz tick 计数（0 起，只在 {@link Board#onTick} 里递增）。 */
    long tick();

    /** 对局开始时的毫秒时间戳。 */
    long startedAt();

    /** 交给板子的 Gladiator 服务（生成 Bot、查询 PvP 判定、淘汰标记等）。 */
    GladiatorAPI api();

    /**
     * 取这一局的参与者（真实玩家 / mineflayer Bot / golem 假玩家），拿不到返回 {@code null}。
     *
     * <p>拿到就是普通的 {@link org.bukkit.entity.Player}：传送、给物品、加药水、改血量全部走
     * Bukkit API，不需要再问板子外面任何东西。
     */
    Player player(String name);

    /**
     * 读取参与者快照；不在线时返回 {@link PlayerSnapshot#offline(String)}。
     */
    PlayerSnapshot snapshot(String player);

    /** 参与者是否在线（判定口径同 {@link #snapshot(String)}，golem 假玩家也算在线）。 */
    boolean isOnline(String player);

    /** 记录一条板子日志（写进服务端日志，前缀 {@code [board/<id>]}）。 */
    void log(String message);

    /**
     * 结束当前局。Gladiator 记账（比分、时长）并把 {@link #round()} 加一；
     * 下一局由板子在 {@link Board#onTick} 或事件里自行开始。
     */
    void endRound(RoundOutcome outcome);

    /** 结束整场对局（调用后板子不应再发任何指令，直到下一次 {@link Board#onStart}）。 */
    void endMatch(MatchOutcome outcome);

    /**
     * 声明当前阶段，供外部（命令 / dashboard）观察；Gladiator 只透出，不解释也不校验。
     *
     * <p>约定用短横线小写词：{@code starting} / {@code countdown} / {@code fighting} /
     * {@code ended}。板子不调用时外部看到的是 {@code starting}；会话结束后一律是 {@code ended}。
     */
    void phase(String phase);

    /** 当前阶段（见 {@link #phase(String)}）。 */
    String phase();

    /** 到目前为止的累计比分（对局未结束也返回当前累计）。 */
    MatchOutcome summary();

    /** 会话结束原因；未结束返回空串。 */
    String lastReason();
}
