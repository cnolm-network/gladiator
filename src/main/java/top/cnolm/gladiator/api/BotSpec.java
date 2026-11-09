package top.cnolm.gladiator.api;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * 发起一个 Bot 所需的规格。
 *
 * <p>{@code group} 是<strong>不透明元数据</strong>：调用方自行解释（例如权限组名），
 * Gladiator 不会据此设置任何权限，也不感知任何反作弊插件。
 */
public final class BotSpec {

    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9_]{3,16}");

    private final String username;
    private final String group;
    private final Map<String, String> metadata;
    private final BotProfile profile;

    private BotSpec(String username, String group, Map<String, String> metadata, BotProfile profile) {
        Objects.requireNonNull(username, "username");
        if (!NAME.matcher(username).matches()) {
            throw new IllegalArgumentException("Invalid bot username: " + username);
        }
        this.username = username;
        this.group = group == null ? "" : group;
        Map<String, String> copy = new LinkedHashMap<String, String>();
        if (metadata != null) {
            copy.putAll(metadata);
        }
        this.metadata = Collections.unmodifiableMap(copy);
        this.profile = profile == null ? BotProfile.defaults() : profile;
    }

    public static BotSpec of(String username, String group) {
        return new BotSpec(username, group, null, null);
    }

    public static BotSpec of(String username, String group, Map<String, String> metadata) {
        return new BotSpec(username, group, metadata, null);
    }

    /**
     * 带行为配置的规格（G-507）。
     *
     * @param profile 每个 Bot 独立的 CPS / 反应延时 / 攻击距离 / 模型标签；null 取默认值
     */
    public static BotSpec of(String username, String group, Map<String, String> metadata,
                             BotProfile profile) {
        return new BotSpec(username, group, metadata, profile);
    }

    public String getUsername() {
        return username;
    }

    public String getGroup() {
        return group;
    }

    public Map<String, String> getMetadata() {
        return metadata;
    }

    /** 该 Bot 的行为配置，永不为 null。 */
    public BotProfile getProfile() {
        return profile;
    }
}
