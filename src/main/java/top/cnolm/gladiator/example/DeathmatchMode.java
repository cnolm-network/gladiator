package top.cnolm.gladiator.example;

import top.cnolm.gladiator.api.Mode;
import top.cnolm.gladiator.api.User;

/**
 * 参考 Mode：纯规则查询，不持有任何竞技场插件类型（G-306）。
 *
 * <p>Gladiator 只调用这里的方法读取规则，不会写状态；淘汰由调用方调
 * {@link top.cnolm.gladiator.api.GladiatorAPI#eliminate(User)}。
 *
 * <p>本类仅作集成示例，生产实现由调用方自己提供（例如从竞技场插件的配置/对局对象里读）。
 */
public final class DeathmatchMode implements Mode {

    private final long timeLimitMs;

    public DeathmatchMode() {
        this(0L);
    }

    public DeathmatchMode(long timeLimitMs) {
        this.timeLimitMs = Math.max(0L, timeLimitMs);
    }

    @Override
    public String getId() {
        return "deathmatch";
    }

    @Override
    public boolean allowsRespawn(User user) {
        // 死斗：不允许重生，死亡即淘汰（也可返回 true 让 Gladiator 走 DEAD→RESPAWNING→ALIVE）
        return false;
    }

    @Override
    public long getTimeLimitMs() {
        return timeLimitMs;
    }
}
