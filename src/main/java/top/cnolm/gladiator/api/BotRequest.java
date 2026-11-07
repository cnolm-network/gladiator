package top.cnolm.gladiator.api;

import java.util.Objects;

/**
 * {@link GladiatorAPI#spawnBot(BotSpec)} 的票据，不可变。
 *
 * <p>登录是异步的：先拿到 PENDING，随后由 {@code GladiatorBotReadyEvent} 或
 * {@code GladiatorBotFailedEvent} 通知结果；状态推进通过 ready()/failed() 产生新实例。
 */
public final class BotRequest {

    public enum State {
        PENDING,
        READY,
        FAILED
    }

    private final String requestId;
    private final String botId;
    private final State state;
    private final Bot bot;
    private final String failureReason;

    private BotRequest(String requestId, String botId, State state, Bot bot, String failureReason) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.botId = Objects.requireNonNull(botId, "botId");
        this.state = Objects.requireNonNull(state, "state");
        this.bot = bot;
        this.failureReason = failureReason;
    }

    public static BotRequest pending(String requestId, String botId) {
        return new BotRequest(requestId, botId, State.PENDING, null, null);
    }

    public BotRequest ready(Bot value) {
        return new BotRequest(requestId, botId, State.READY, Objects.requireNonNull(value, "bot"), null);
    }

    public BotRequest failed(String reason) {
        return new BotRequest(requestId, botId, State.FAILED, null, reason);
    }

    public String getRequestId() {
        return requestId;
    }

    public String getBotId() {
        return botId;
    }

    public State getState() {
        return state;
    }

    /** 仅 READY 时非 null。 */
    public Bot getBot() {
        return bot;
    }

    /** 仅 FAILED 时非 null。 */
    public String getFailureReason() {
        return failureReason;
    }
}
