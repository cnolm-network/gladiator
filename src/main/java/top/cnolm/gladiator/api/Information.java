package top.cnolm.gladiator.api;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 推送给 Bot 的结构化信息载荷。
 *
 * <p>不可变；payload 只允许原语、字符串、枚举与扁平集合，
 * 不得携带 Bukkit / NMS / mineflayer 对象。
 */
public final class Information {

    private final InformationKind kind;
    private final long timestamp;
    private final Map<String, Object> payload;

    private Information(InformationKind kind, long timestamp, Map<String, Object> payload) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.timestamp = timestamp;
        Map<String, Object> copy = new LinkedHashMap<String, Object>();
        if (payload != null) {
            copy.putAll(payload);
        }
        this.payload = Collections.unmodifiableMap(copy);
    }

    public static Information of(InformationKind kind, Map<String, Object> payload) {
        return new Information(kind, System.currentTimeMillis(), payload);
    }

    public static Information of(InformationKind kind, long timestamp, Map<String, Object> payload) {
        return new Information(kind, timestamp, payload);
    }

    public static Information empty(InformationKind kind) {
        return new Information(kind, System.currentTimeMillis(), null);
    }

    public InformationKind getKind() {
        return kind;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public Map<String, Object> getPayload() {
        return payload;
    }
}
