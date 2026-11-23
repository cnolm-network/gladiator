package top.cnolm.gladiator.api.board;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** 一局的判定结果。由板子给出，Gladiator 只记账与转发。 */
public final class RoundOutcome {

    /** 玩家（真人）赢。 */
    public static final String WINNER_PLAYER = "player";
    /** Bot 赢。 */
    public static final String WINNER_BOT = "bot";
    /** 平局/超时无人获胜。 */
    public static final String WINNER_DRAW = "draw";

    private final String winner;
    private final String reason;
    private final Map<String, Object> detail;

    public RoundOutcome(String winner, String reason, Map<String, Object> detail) {
        this.winner = winner == null ? WINNER_DRAW : winner;
        this.reason = reason == null ? "unknown" : reason;
        this.detail = detail == null
                ? Collections.<String, Object>emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<String, Object>(detail));
    }

    public static RoundOutcome player(String reason) {
        return new RoundOutcome(WINNER_PLAYER, reason, null);
    }

    public static RoundOutcome bot(String reason) {
        return new RoundOutcome(WINNER_BOT, reason, null);
    }

    public static RoundOutcome draw(String reason) {
        return new RoundOutcome(WINNER_DRAW, reason, null);
    }

    public String getWinner() {
        return winner;
    }

    public String getReason() {
        return reason;
    }

    public Map<String, Object> getDetail() {
        return detail;
    }

    @Override
    public String toString() {
        return "RoundOutcome{" + winner + " reason=" + reason + '}';
    }
}
