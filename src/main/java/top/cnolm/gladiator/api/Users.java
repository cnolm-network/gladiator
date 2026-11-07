package top.cnolm.gladiator.api;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;

import java.util.Objects;
import java.util.UUID;

/** 把 Bukkit Player 包装成 {@link User}，供 pushGame 混合传参。 */
public final class Users {

    private Users() {
    }

    public static User of(Player player) {
        Objects.requireNonNull(player, "player");
        return new PlayerUser(player.getUniqueId(), player.getName());
    }

    private static final class PlayerUser implements User {

        private final UUID uuid;
        private final String name;

        private PlayerUser(UUID uuid, String name) {
            this.uuid = uuid;
            this.name = name;
        }

        @Override
        public UUID getUniqueId() {
            return uuid;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public UserState getState() {
            return UserStateRegistry.stateOf(uuid);
        }

        @Override
        public boolean isBot() {
            return false;
        }

        @Override
        public Player asPlayer() {
            return Bukkit.getPlayer(uuid);
        }
    }
}
