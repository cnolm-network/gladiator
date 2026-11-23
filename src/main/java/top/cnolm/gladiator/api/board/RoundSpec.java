package top.cnolm.gladiator.api.board;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 一场对局的参数（不可变）。
 *
 * <p>Gladiator 只负责把这套参数<b>原样</b>交给板子：它不理解 kitId 是什么、也不理解
 * 出生点意味着什么——这些全部是板子的规则。{@code extras} 留给板子放自己的配置
 * （例如 "kit=diamond" / "allowPearl=true"），Gladiator 不读也不校验。
 */
public final class RoundSpec {

    private final int rounds;
    private final int secondsPerRound;
    private final String kitId;
    private final double[] playerSpawn;
    private final double[] botSpawn;
    private final int shieldSeconds;
    private final Map<String, String> extras;

    private RoundSpec(Builder b) {
        this.rounds = b.rounds;
        this.secondsPerRound = b.secondsPerRound;
        this.kitId = b.kitId;
        this.playerSpawn = b.playerSpawn;
        this.botSpawn = b.botSpawn;
        this.shieldSeconds = b.shieldSeconds;
        this.extras = Collections.unmodifiableMap(new LinkedHashMap<String, String>(b.extras));
    }

    public static Builder builder() {
        return new Builder();
    }

    /** 局数（>=1）。 */
    public int getRounds() {
        return rounds;
    }

    /** 每局时长（秒，>=1）；超时由板子自己判。 */
    public int getSecondsPerRound() {
        return secondsPerRound;
    }

    /** 套件标识，Gladiator 视为不透明字符串。 */
    public String getKitId() {
        return kitId;
    }

    /** 玩家出生点 {x,y,z}，未配置时为 {@code null}。 */
    public double[] getPlayerSpawn() {
        return playerSpawn == null ? null : playerSpawn.clone();
    }

    /** Bot 出生点 {x,y,z}，未配置时为 {@code null}。 */
    public double[] getBotSpawn() {
        return botSpawn == null ? null : botSpawn.clone();
    }

    /** 准备阶段的免伤秒数（板子可在倒计时里给抗性，Gladiator 不代劳）。 */
    public int getShieldSeconds() {
        return shieldSeconds;
    }

    public Map<String, String> getExtras() {
        return extras;
    }

    public String extra(String key, String def) {
        String v = extras.get(key);
        return v == null ? def : v;
    }

    /** 构造器。 */
    public static final class Builder {
        private int rounds = 1;
        private int secondsPerRound = 60;
        private String kitId = "default";
        private double[] playerSpawn;
        private double[] botSpawn;
        private int shieldSeconds = 3;
        private final Map<String, String> extras = new LinkedHashMap<String, String>();

        public Builder rounds(int v) {
            this.rounds = Math.max(1, v);
            return this;
        }

        public Builder secondsPerRound(int v) {
            this.secondsPerRound = Math.max(1, v);
            return this;
        }

        public Builder kitId(String v) {
            if (v != null && !v.trim().isEmpty()) {
                this.kitId = v.trim();
            }
            return this;
        }

        public Builder playerSpawn(double x, double y, double z) {
            this.playerSpawn = new double[]{x, y, z};
            return this;
        }

        public Builder botSpawn(double x, double y, double z) {
            this.botSpawn = new double[]{x, y, z};
            return this;
        }

        public Builder shieldSeconds(int v) {
            this.shieldSeconds = Math.max(0, v);
            return this;
        }

        public Builder extra(String key, String value) {
            if (key != null && value != null) {
                this.extras.put(key, value);
            }
            return this;
        }

        public RoundSpec build() {
            return new RoundSpec(this);
        }
    }
}
