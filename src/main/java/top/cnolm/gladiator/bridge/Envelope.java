package top.cnolm.gladiator.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * 协议 v1 统一信封（见 docs/protocol.md §1）。
 *
 * <p>只做解析与序列化，不含游戏逻辑、不引用 Bukkit。JSON 由服务端自带的 Gson 2.2.4
 * 处理（spigot-api 的传递依赖），因此本插件不需要 shade 任何第三方库。
 * 注意 Gson 2.2.4 没有 {@code JsonParser.parseString}，必须使用实例方法。
 */
public final class Envelope {

    public static final int VERSION = 1;

    private static final Gson GSON = new Gson();

    private final JsonObject raw;

    private Envelope(JsonObject raw) {
        this.raw = raw;
    }

    /** 报文格式错误（非对象、缺 v/type、类型不符）。 */
    public static final class MalformedException extends Exception {
        private static final long serialVersionUID = 1L;

        public MalformedException(String message) {
            super(message);
        }
    }

    public static Envelope parse(String text) throws MalformedException {
        if (text == null || text.length() == 0) {
            throw new MalformedException("empty frame");
        }
        JsonElement element;
        try {
            element = new JsonParser().parse(text);
        } catch (RuntimeException e) {
            throw new MalformedException("malformed JSON");
        }
        if (element == null || !element.isJsonObject()) {
            throw new MalformedException("envelope must be a JSON object");
        }
        JsonObject obj = element.getAsJsonObject();
        if (!isPrimitive(obj, "v") || !obj.get("v").getAsJsonPrimitive().isNumber()) {
            throw new MalformedException("missing or invalid field: v");
        }
        if (!isPrimitive(obj, "type") || obj.get("type").getAsString().length() == 0) {
            throw new MalformedException("missing or invalid field: type");
        }
        if (obj.has("payload") && !obj.get("payload").isJsonObject() && !obj.get("payload").isJsonNull()) {
            throw new MalformedException("payload must be an object");
        }
        return new Envelope(obj);
    }

    /** 构造一条出站报文；seq 由调用方按单方向单调递增维护。 */
    public static Envelope of(String type, long seq, String botId, Long replyTo, JsonObject payload) {
        if (type == null || type.length() == 0) {
            throw new IllegalArgumentException("type");
        }
        JsonObject obj = new JsonObject();
        obj.addProperty("v", Integer.valueOf(VERSION));
        obj.addProperty("type", type);
        obj.addProperty("seq", Long.valueOf(seq));
        obj.addProperty("ts", Long.valueOf(System.currentTimeMillis()));
        obj.addProperty("botId", botId == null ? "" : botId);
        if (replyTo == null) {
            obj.add("replyTo", com.google.gson.JsonNull.INSTANCE);
        } else {
            obj.addProperty("replyTo", replyTo);
        }
        obj.add("payload", payload == null ? new JsonObject() : payload);
        return new Envelope(obj);
    }

    public int version() {
        return raw.get("v").getAsInt();
    }

    public String type() {
        return raw.get("type").getAsString();
    }

    public long seq() {
        return hasNumber("seq") ? raw.get("seq").getAsLong() : 0L;
    }

    public long ts() {
        return hasNumber("ts") ? raw.get("ts").getAsLong() : 0L;
    }

    public String botId() {
        return isPrimitive(raw, "botId") ? raw.get("botId").getAsString() : "";
    }

    /** 未设置时返回 null。 */
    public Long replyTo() {
        return hasNumber("replyTo") ? Long.valueOf(raw.get("replyTo").getAsLong()) : null;
    }

    public JsonObject payload() {
        JsonElement p = raw.get("payload");
        return (p == null || !p.isJsonObject()) ? new JsonObject() : p.getAsJsonObject();
    }

    /** 读取 payload 中的字符串字段，缺失/类型不符返回 defaultValue。 */
    public String payloadString(String key, String defaultValue) {
        JsonObject p = payload();
        return isPrimitive(p, key) ? p.get(key).getAsString() : defaultValue;
    }

    public String toJson() {
        return GSON.toJson(raw);
    }

    private boolean hasNumber(String key) {
        return isPrimitive(raw, key) && raw.get(key).getAsJsonPrimitive().isNumber();
    }

    private static boolean isPrimitive(JsonObject obj, String key) {
        return obj.has(key) && obj.get(key).isJsonPrimitive();
    }
}
