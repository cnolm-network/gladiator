package top.cnolm.gladiator.api.board;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** 一条对局事件（不可变）。只包含 Java 基础类型，不泄漏 Bukkit 类型。 */
public final class GameEvent {

    private final GameEventType type;
    private final long seq;
    private final long ts;
    private final int round;
    private final String actor;
    private final String target;
    private final Map<String, Object> data;

    private GameEvent(GameEventType type, long seq, long ts, int round,
                      String actor, String target, Map<String, Object> data) {
        this.type = type;
        this.seq = seq;
        this.ts = ts;
        this.round = round;
        this.actor = actor == null ? "" : actor;
        this.target = target == null ? "" : target;
        this.data = data == null
                ? Collections.<String, Object>emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<String, Object>(data));
    }

    public static GameEvent of(GameEventType type, long seq, long ts, int round,
                               String actor, String target, Map<String, Object> data) {
        return new GameEvent(type, seq, ts, round, actor, target, data);
    }

    public GameEventType getType() {
        return type;
    }

    public long getSeq() {
        return seq;
    }

    public long getTs() {
        return ts;
    }

    public int getRound() {
        return round;
    }

    public String getActor() {
        return actor;
    }

    public String getTarget() {
        return target;
    }

    public Map<String, Object> getData() {
        return data;
    }

    public String string(String key, String def) {
        Object v = data.get(key);
        return v == null ? def : String.valueOf(v);
    }

    public double number(String key, double def) {
        Object v = data.get(key);
        if (v instanceof Number) {
            return ((Number) v).doubleValue();
        }
        return def;
    }

    public int integer(String key, int def) {
        Object v = data.get(key);
        if (v instanceof Number) {
            return ((Number) v).intValue();
        }
        return def;
    }

    @Override
    public String toString() {
        return "GameEvent{" + type + " seq=" + seq + " round=" + round
                + " actor=" + actor + " target=" + target + " data=" + data + '}';
    }
}
