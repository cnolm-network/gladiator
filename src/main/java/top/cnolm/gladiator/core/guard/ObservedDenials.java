package top.cnolm.gladiator.core.guard;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 「观测到的拒绝」短期缓存（A3：用实时反馈替代 WorldGuard API 查询）。
 *
 * <p>任何插件取消了 Bot 的伤害/交互，都会被 {@link DenialFeedbackListener} 记成一条 deny：
 * 之后 {@code queryPvp} 命中同 Bot、同球体范围、且未过期的记录就直接判 deny
 * （{@code source=observed}），不必每次去问 WorldGuard。
 *
 * <p>纯 Java、无 Bukkit 依赖：时间由调用方以毫秒传入，便于单测。
 * key 是 Gladiator 的 botId（{@code glad-<username>}），因此天然按 Bot 隔离。
 */
public final class ObservedDenials {

    /** 每个 Bot 最多保留的记录数；超出后丢最旧的。 */
    public static final int DEFAULT_CAPACITY = 16;

    /** 一条观测到的拒绝。 */
    public static final class Entry {

        private final String botId;
        private final double x;
        private final double y;
        private final double z;
        private final double radius;
        private final String action;
        private final long expiresAt;

        public Entry(String botId, double x, double y, double z,
                     double radius, String action, long expiresAt) {
            this.botId = botId;
            this.x = x;
            this.y = y;
            this.z = z;
            this.radius = Math.max(0D, radius);
            this.action = action;
            this.expiresAt = expiresAt;
        }

        public String getBotId() {
            return botId;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        public double getZ() {
            return z;
        }

        public double getRadius() {
            return radius;
        }

        public String getAction() {
            return action;
        }

        public long getExpiresAt() {
            return expiresAt;
        }

        public boolean expired(long nowMillis) {
            return nowMillis >= expiresAt;
        }

        /** 点是否落在以观测点为中心、半径 radius 的立方体内。 */
        public boolean contains(double px, double py, double pz) {
            return Math.abs(px - x) <= radius
                    && Math.abs(py - y) <= radius
                    && Math.abs(pz - z) <= radius;
        }

        /** 该记录对应的三维边界，供 {@code PvpVerdict} 返回。 */
        public top.cnolm.gladiator.api.PvpVerdict.Box box() {
            return top.cnolm.gladiator.api.PvpVerdict.Box.of(
                    x - radius, y - radius, z - radius,
                    x + radius, y + radius, z + radius);
        }

        @Override
        public String toString() {
            return "Entry(" + botId + " @" + x + "," + y + "," + z
                    + " r=" + radius + " action=" + action + " exp=" + expiresAt + ")";
        }
    }

    private final ConcurrentHashMap<String, Deque<Entry>> byBot =
            new ConcurrentHashMap<String, Deque<Entry>>();
    private final int capacity;

    public ObservedDenials() {
        this(DEFAULT_CAPACITY);
    }

    public ObservedDenials(int capacity) {
        this.capacity = Math.max(1, capacity);
    }

    /** 追加一条记录；同一 Bot 超过容量时丢最旧的。 */
    public void record(Entry entry) {
        if (entry == null || entry.getBotId() == null) {
            return;
        }
        Deque<Entry> queue = byBot.get(entry.getBotId());
        if (queue == null) {
            Deque<Entry> created = new ArrayDeque<Entry>();
            Deque<Entry> existing = byBot.putIfAbsent(entry.getBotId(), created);
            queue = existing == null ? created : existing;
        }
        synchronized (queue) {
            while (queue.size() >= capacity) {
                queue.pollFirst();
            }
            queue.addLast(entry);
        }
    }

    /**
     * 找一条命中的记录：同 Bot、未过期、点在半径内。多条命中时取最新一条。
     *
     * @return 命中的记录；没有则 {@code null}
     */
    public Entry match(String botId, double x, double y, double z, long nowMillis) {
        if (botId == null) {
            return null;
        }
        Deque<Entry> queue = byBot.get(botId);
        if (queue == null) {
            return null;
        }
        List<Entry> snapshot;
        synchronized (queue) {
            prune(queue, nowMillis);
            snapshot = new ArrayList<Entry>(queue);
        }
        for (int i = snapshot.size() - 1; i >= 0; i--) {
            Entry entry = snapshot.get(i);
            if (!entry.expired(nowMillis) && entry.contains(x, y, z)) {
                return entry;
            }
        }
        return null;
    }

    /** 某 Bot 当前保留的记录数（含已过期但尚未 prune 的）。 */
    public int size(String botId) {
        Deque<Entry> queue = byBot.get(botId);
        if (queue == null) {
            return 0;
        }
        synchronized (queue) {
            return queue.size();
        }
    }

    /** 全部 Bot 的记录总数。 */
    public int totalSize() {
        int total = 0;
        for (Deque<Entry> queue : byBot.values()) {
            synchronized (queue) {
                total += queue.size();
            }
        }
        return total;
    }

    public void clear() {
        byBot.clear();
    }

    public void clear(String botId) {
        if (botId != null) {
            byBot.remove(botId);
        }
    }

    private static void prune(Deque<Entry> queue, long nowMillis) {
        while (!queue.isEmpty() && queue.peekFirst().expired(nowMillis)) {
            queue.pollFirst();
        }
    }
}
