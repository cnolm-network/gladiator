package top.cnolm.gladiator.api.board;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 一整场对局的结果（按局累计）。 */
public final class MatchOutcome {

    private final int playerRounds;
    private final int botRounds;
    private final int draws;
    private final String reason;
    private final List<RoundRecord> rounds;
    private final Map<String, Object> detail;

    public MatchOutcome(int playerRounds, int botRounds, int draws, String reason,
                        List<RoundRecord> rounds, Map<String, Object> detail) {
        this.playerRounds = playerRounds;
        this.botRounds = botRounds;
        this.draws = draws;
        this.reason = reason == null ? "finished" : reason;
        this.rounds = rounds == null
                ? Collections.<RoundRecord>emptyList()
                : Collections.unmodifiableList(new ArrayList<RoundRecord>(rounds));
        this.detail = detail == null
                ? Collections.<String, Object>emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<String, Object>(detail));
    }

    public int getPlayerRounds() {
        return playerRounds;
    }

    public int getBotRounds() {
        return botRounds;
    }

    public int getDraws() {
        return draws;
    }

    public String getReason() {
        return reason;
    }

    public List<RoundRecord> getRounds() {
        return rounds;
    }

    public Map<String, Object> getDetail() {
        return detail;
    }

    @Override
    public String toString() {
        return "MatchOutcome{" + playerRounds + ":" + botRounds + " draws=" + draws
                + " reason=" + reason + '}';
    }

    /** 单局记录。 */
    public static final class RoundRecord {
        private final int index;
        private final String winner;
        private final String reason;
        private final int seconds;

        public RoundRecord(int index, String winner, String reason, int seconds) {
            this.index = index;
            this.winner = winner;
            this.reason = reason;
            this.seconds = seconds;
        }

        public int getIndex() {
            return index;
        }

        public String getWinner() {
            return winner;
        }

        public String getReason() {
            return reason;
        }

        public int getSeconds() {
            return seconds;
        }

        @Override
        public String toString() {
            return "#" + index + " " + winner + "(" + reason + "," + seconds + "s)";
        }
    }
}
