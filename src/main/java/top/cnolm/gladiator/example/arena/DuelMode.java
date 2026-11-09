package top.cnolm.gladiator.example.arena;

import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.User;

/**
 * 示例调用方 B 的 Mode 实现（G-506 跨调用方兼容性）。
 *
 * <p>与 {@code example.DeathmatchMode} 的区别：这里是"决斗"，允许重生、有 5 分钟时限。
 * 两个 Mode 都属于调用方，Gladiator 源码里不出现任何一个。</p>
 */
public final class DuelMode implements Mode {

    public static final String ID = "duel";

    private final long timeLimitMs;

    public DuelMode() {
        this(5 * 60 * 1000L);
    }

    public DuelMode(long timeLimitMs) {
        this.timeLimitMs = timeLimitMs;
    }

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public boolean allowsRespawn(User user) {
        return true;
    }

    @Override
    public long getTimeLimitMs() {
        return timeLimitMs;
    }
}
