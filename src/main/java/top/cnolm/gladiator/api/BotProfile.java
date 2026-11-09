package top.cnolm.gladiator.api;

import java.util.Objects;

/**
 * 单个 Bot 的行为配置。
 *
 * <p>这些参数描述的是“操作风格”，不是对局规则：CPS 与反应延时决定动作发出的节奏，
 * 攻击距离决定客户端认为多远的实体可以打（服务端仍会自行校验），模型名只作为
 * 训练侧的选择标签，Gladiator 不会据此加载任何算法。
 *
 * <p>默认值对应“原版手感”：15±2 CPS、220±40ms 反应、3.99 格攻击距离、默认模型。
 */
public final class BotProfile {

    public static final double DEFAULT_CPS = 15.0;
    public static final double DEFAULT_CPS_JITTER = 2.0;
    public static final double DEFAULT_REACTION_MS = 220.0;
    public static final double DEFAULT_REACTION_JITTER_MS = 40.0;
    public static final double DEFAULT_REACH = 3.99;
    public static final String DEFAULT_MODEL = "default";

    private static final BotProfile DEFAULTS = new BotProfile(
            DEFAULT_CPS, DEFAULT_CPS_JITTER, DEFAULT_REACTION_MS,
            DEFAULT_REACTION_JITTER_MS, DEFAULT_REACH, DEFAULT_MODEL,
            false, false, false, false);

    private final double cps;
    private final double cpsJitter;
    private final double reactionMs;
    private final double reactionJitterMs;
    private final double reach;
    private final String model;
    private final boolean autoRod;
    private final boolean autoBlock;
    private final boolean wtap;
    private final boolean aimLock;

    private BotProfile(double cps, double cpsJitter, double reactionMs,
                       double reactionJitterMs, double reach, String model,
                       boolean autoRod, boolean autoBlock, boolean wtap, boolean aimLock) {
        checkFinite(cps, "cps");
        checkFinite(cpsJitter, "cpsJitter");
        checkFinite(reactionMs, "reactionMs");
        checkFinite(reactionJitterMs, "reactionJitterMs");
        checkFinite(reach, "reach");
        if (cps <= 0.0) {
            throw new IllegalArgumentException("cps 必须 > 0");
        }
        if (cpsJitter < 0.0 || cpsJitter >= cps) {
            throw new IllegalArgumentException("cpsJitter 必须在 [0, cps) 内");
        }
        if (reactionMs < 0.0 || reactionJitterMs < 0.0) {
            throw new IllegalArgumentException("反应延时与抖动不能为负");
        }
        if (reach <= 0.0 || reach > 6.0) {
            throw new IllegalArgumentException("reach 必须在 (0, 6] 内");
        }
        this.cps = cps;
        this.cpsJitter = cpsJitter;
        this.reactionMs = reactionMs;
        this.reactionJitterMs = reactionJitterMs;
        this.reach = reach;
        this.model = model == null || model.trim().isEmpty() ? DEFAULT_MODEL : model.trim();
        this.autoRod = autoRod;
        this.autoBlock = autoBlock;
        this.wtap = wtap;
        this.aimLock = aimLock;
    }

    private static void checkFinite(double value, String name) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException(name + " 必须是有限数");
        }
    }

    /** 默认配置：15±2 CPS、220±40ms、3.99 格、default 模型。 */
    public static BotProfile defaults() {
        return DEFAULTS;
    }

    /**
     * 以默认值为底，覆盖任意字段。
     *
     * @param cps             每秒点击数，&gt; 0
     * @param cpsJitter       CPS 抖动（±），[0, cps)
     * @param reactionMs      基础反应延时（毫秒），&gt;= 0
     * @param reactionJitterMs 反应延时抖动（±，毫秒），&gt;= 0
     * @param reach           攻击距离（格），(0, 6]
     * @param model           训练模型标签，null/空串回落 "default"
     */
    public static BotProfile of(double cps, double cpsJitter, double reactionMs,
                                double reactionJitterMs, double reach, String model) {
        return new BotProfile(cps, cpsJitter, reactionMs, reactionJitterMs, reach, model,
                false, false, false, false);
    }

    /**
     * 带战斗辅助开关的完整构造（辅助默认关闭，训练里按课程逐步打开）。
     */
    public static BotProfile of(double cps, double cpsJitter, double reactionMs,
                                double reactionJitterMs, double reach, String model,
                                boolean autoRod, boolean autoBlock, boolean wtap,
                                boolean aimLock) {
        return new BotProfile(cps, cpsJitter, reactionMs, reactionJitterMs, reach, model,
                autoRod, autoBlock, wtap, aimLock);
    }

    public static Builder builder() {
        return new Builder();
    }

    public double getCps() {
        return cps;
    }

    public double getCpsJitter() {
        return cpsJitter;
    }

    public double getReactionMs() {
        return reactionMs;
    }

    public double getReactionJitterMs() {
        return reactionJitterMs;
    }

    public double getReach() {
        return reach;
    }

    public String getModel() {
        return model;
    }

    /** 中距离是否自动切鱼竿击退。 */
    public boolean isAutoRod() {
        return autoRod;
    }

    /** 空闲时是否自动举剑格挡。 */
    public boolean isAutoBlock() {
        return autoBlock;
    }

    /** 命中后是否做 sprint reset（W-tap）。 */
    public boolean isWtap() {
        return wtap;
    }

    /** 攻击前是否 3D 对准目标眼睛。 */
    public boolean isAimLock() {
        return aimLock;
    }

    /** 两次点击之间的基础间隔（毫秒）。 */
    public double clickIntervalMs() {
        return 1000.0 / cps;
    }

    @Override
    public String toString() {
        return "BotProfile{cps=" + cps + "±" + cpsJitter
                + ", reaction=" + reactionMs + "±" + reactionJitterMs + "ms"
                + ", reach=" + reach + ", model=" + model
                + ", assist(rod=" + autoRod + ", block=" + autoBlock
                + ", wtap=" + wtap + ", aim=" + aimLock + ")}";
    }

    /** 便捷构造器，未设置的字段取默认值。 */
    public static final class Builder {
        private double cps = DEFAULT_CPS;
        private double cpsJitter = DEFAULT_CPS_JITTER;
        private double reactionMs = DEFAULT_REACTION_MS;
        private double reactionJitterMs = DEFAULT_REACTION_JITTER_MS;
        private double reach = DEFAULT_REACH;
        private String model = DEFAULT_MODEL;
        private boolean autoRod = false;
        private boolean autoBlock = false;
        private boolean wtap = false;
        private boolean aimLock = false;

        public Builder cps(double cps, double jitter) {
            this.cps = cps;
            this.cpsJitter = jitter;
            return this;
        }

        public Builder reaction(double ms, double jitterMs) {
            this.reactionMs = ms;
            this.reactionJitterMs = jitterMs;
            return this;
        }

        public Builder reach(double reach) {
            this.reach = reach;
            return this;
        }

        public Builder model(String model) {
            this.model = Objects.requireNonNull(model, "model");
            return this;
        }

        public Builder autoRod(boolean value) {
            this.autoRod = value;
            return this;
        }

        public Builder autoBlock(boolean value) {
            this.autoBlock = value;
            return this;
        }

        public Builder wtap(boolean value) {
            this.wtap = value;
            return this;
        }

        public Builder aimLock(boolean value) {
            this.aimLock = value;
            return this;
        }

        public BotProfile build() {
            return new BotProfile(cps, cpsJitter, reactionMs, reactionJitterMs, reach, model,
                    autoRod, autoBlock, wtap, aimLock);
        }
    }
}
