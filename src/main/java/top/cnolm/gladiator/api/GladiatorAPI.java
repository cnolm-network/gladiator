package top.cnolm.gladiator.api;

/**
 * Gladiator 对外唯一的服务接口。
 *
 * <p>调用方（竞技场插件）通过 {@link GladiatorProvider#get()} 或 Bukkit
 * {@code ServicesManager} 取得实例。所有方法必须在服务端<strong>主线程</strong>调用。
 *
 * <p>Gladiator 不做传送、发套件、选图、匹配、结算，也不感知任何反作弊插件；
 * 环境准备属于调用方 init 阶段，见 {@code docs/api-contract.md} §8。
 */
public interface GladiatorAPI {

    /** 语义化版本；只增不改，调用方可据此做兼容判断。 */
    String getApiVersion();

    /** 插件 enable 完成、WS 已监听后为 true。 */
    boolean isReady();

    /** 发起 Bot 登录，立即返回票据；完成后由 BotReady/BotFailed 事件通知。 */
    BotRequest spawnBot(BotSpec spec);

    /** 把真人与 Bot 加入对局；调用方须已完成传送与套件。 */
    GameSession pushGame(Mode mode, User... users);

    /** 把 Bot 放入无限制训练场；Gladiator 不传送、不发套件。 */
    TrainingSession pushPlace(Bot bot);

    /** 向 Bot 推送结构化信息，只投递不执行 Bukkit 操作。 */
    void notice(Bot bot, Information information);

    /** 会话级训练指标快照；采样异步，返回不可变对象。 */
    Analysis analyze(AnalysisQuery query);

    /** 淘汰标记：调用方/Mode 唯一可写的状态入口。 */
    void eliminate(User user);

    UserState stateOf(User user);
}
