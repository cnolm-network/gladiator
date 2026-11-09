package top.cnolm.gladiator.example;

import org.bukkit.entity.Player;
import org.junit.Test;
import top.cnolm.gladiator.api.User;
import top.cnolm.gladiator.api.UserState;

import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** G-505：调用方统计必须过滤 Bot，榜单里不能出现 Bot 脏数据。 */
public class StatsBoardTest {

    @Test
    public void botWinsAreIgnored() {
        StatsBoard board = new StatsBoard();
        assertFalse("Bot 的战绩不应进榜", board.recordWin(new FakeUser(UUID.randomUUID(), "Glad_Bot_1", true)));
        assertEquals(0, board.size());
    }

    @Test
    public void nullIsIgnored() {
        StatsBoard board = new StatsBoard();
        assertFalse(board.recordWin(null));
        assertEquals(0, board.size());
    }

    @Test
    public void humanWinsAreCountedAndRanked() {
        StatsBoard board = new StatsBoard();
        UUID alice = UUID.randomUUID();
        UUID bob = UUID.randomUUID();
        assertTrue(board.recordWin(new FakeUser(alice, "Alice", false)));
        assertTrue(board.recordWin(new FakeUser(alice, "Alice", false)));
        assertTrue(board.recordWin(new FakeUser(bob, "Bob", false)));
        assertEquals(2, board.winsOf(alice));
        assertEquals(1, board.winsOf(bob));
        assertEquals(alice, board.ranking().get(0));
    }

    private static final class FakeUser implements User {
        private final UUID id;
        private final String name;
        private final boolean bot;

        FakeUser(UUID id, String name, boolean bot) {
            this.id = id;
            this.name = name;
            this.bot = bot;
        }

        @Override
        public UUID getUniqueId() {
            return id;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public UserState getState() {
            return UserState.ALIVE;
        }

        @Override
        public boolean isBot() {
            return bot;
        }

        @Override
        public Player asPlayer() {
            return null;
        }
    }
}
