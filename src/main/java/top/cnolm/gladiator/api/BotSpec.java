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

    private BotSpec(String username, String group, Map<String, String> metadata) {
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
    }

    public static BotSpec of(String username, String group) {
        return new BotSpec(username, group, null);
    }

    public static BotSpec of(String username, String group, Map<String, String> metadata) {
        return new BotSpec(username, group, metadata);
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
}
