package top.cnolm.gladiator.api.board;

/**
 * 对局板子：一块板 = 一种服务器玩法（平地 1v1、Sumo、Nodebuff……）。
 *
 * <p>板子由<b>服务器插件</b>实现（例如仓库根目录的 {@code arena}），在自己的
 * {@code onEnable} 里调用 {@link BoardRegistry#register(Board)}。Gladiator 只提供：
 * 事件流、10Hz tick、执行命令/传送/读快照的原语、回合与场次的记账。
 *
 * <p>板子<b>不</b>处理 Bot 的动作：动作由 {@code bot-runtime}（Node）或 golem 下发，
 * 板子只观察结果（谁被打中、谁死了）。这就是「板子只处理 observe」的边界。
 *
 * <p>所有回调都在主线程。
 */
public interface Board {

    /** 板子唯一 id，例如 {@code "flat-1v1"}；同一 id 重复注册会被拒绝。 */
    String id();

    /** 一场对局开始。板子应在这里做第一局的清场/摆位/发套件/倒计时。 */
    void onStart(BoardContext context);

    /** 服务器事件。只读，不要在这里 sleep 或做长耗时操作。 */
    void onEvent(GameEvent event);

    /** 每 0.1 秒一次（2 tick）。板子用它做倒计时、超时判定、回合切换。 */
    void onTick(long tick);

    /** 对局结束或异常终止；板子应在这里清理自己留下的实体/计分板/BossBar。 */
    void onStop(String reason);
}
