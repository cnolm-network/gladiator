package top.cnolm.gladiator.api;

/** Bot 视角；不暴露 mineflayer session、WebSocket 等运行时细节。 */
public interface Bot extends User {

    String getBotId();

    /** 已登录、spawn 完成、BotHandle 注册完毕。 */
    boolean isReady();
}
