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

    /**
     * 结束当前局，并显式带上本局实际时长（秒）。{@code seconds < 0}（或 {@code NaN}）表示
     * 板子自己也不知道（例如整局靠事件驱动的板子），此时与 {@link #endRound(RoundOutcome)}
     * 完全等价：Gladiator 用墙钟估算。
     *
     * <p>为什么要这个重载：板子才知道「开打时刻」——它可能在倒计时、等待双方复活之后才真正
     * 开打，墙钟会把准备时间也算进局时长（G-719）。外部（dashboard）读
     * {@link #lastSeconds()} 拿到的就是这个数。
     *
     * <p>默认实现忽略秒数并退化为单参版本：旧板子一行都不用改。
     */
    default void endRound(RoundOutcome outcome, double seconds) {
        endRound(outcome);
    }

    /**
     * 最近一局的判定原因（板子传给 {@link RoundOutcome#getReason()} 的原文），
     * 还没有打过任何一局时返回 {@code "none"}。
     *
     * <p>为什么单独给一个 getter：{@link #summary()} 里的 reason 是**会话级**的（如
     * {@code rounds-finished}），只有这个才是「上一局为什么结束」（如 {@code kill}/{@code timeout}），
     * dashboard 的逐局表格要用它。
     */
    default String roundReason() {
        return "none";
    }

    /**
     * 最近一局的时长（秒，可能带小数）；还没有打过任何一局时返回 {@code -1}。
     *
     * @see #endRound(RoundOutcome, double)
     */
    default double lastSeconds() {
        return -1.0D;
    }

    /**
     * 秒数归一（G-719 纯函数，便于单测）：板子给负数/NaN 时退化成墙钟估算。
     *
     * @param seconds            板子上报的局时长；{@code <0}/{@code NaN} = 未知
     * @param roundStartedAtMs   本局开始时的毫秒时间戳（Gladiator 自己记的）
     * @param nowMs              当前毫秒时间戳
     */
    static double resolveRoundSeconds(double seconds, long roundStartedAtMs, long nowMs) {
        if (!Double.isNaN(seconds) && seconds >= 0.0D) {
            return seconds;
        }
        return Math.max(0L, nowMs - roundStartedAtMs) / 1000.0D;
    }

    /**
     * 写进 {@link MatchOutcome.RoundRecord} 的整数秒（该字段是 int，G-719 纯函数）。
     * 四舍五入；未知/负数记 0。
     */
    static int recordSeconds(double seconds) {
        if (Double.isNaN(seconds) || seconds <= 0.0D) {
            return 0;
        }
        return (int) Math.round(seconds);
    }

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
