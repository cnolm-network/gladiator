package top.cnolm.gladiator.example;

import top.cnolm.gladiator.api.User;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 示例：调用方的战绩榜（G-505 数据隔离）。
 *
 * <p>竞技场插件通常有自己的战绩/榜单统计；Bot 参战时如果不过滤，练出来的
 * 战绩会混进真人榜单。这里演示调用方侧的过滤约定：<b>写统计前先判 {@link User#isBot()}</b>。</p>
 *
 * <p>Gladiator 本身不提供任何全局统计（那是调用方职责），也不会替调用方过滤。</p>
 */
public final class StatsBoard {

    private final Map<UUID, AtomicInteger> wins = new ConcurrentHashMap<UUID, AtomicInteger>();

    /**
     * 记录一次胜利。
     *
     * @return false 表示这条记录被忽略（null 或 Bot）
     */
    public boolean recordWin(User winner) {
        if (winner == null || winner.isBot()) {
            return false;
        }
        AtomicInteger counter = wins.get(winner.getUniqueId());
        if (counter == null) {
            AtomicInteger created = new AtomicInteger();
            counter = wins.putIfAbsent(winner.getUniqueId(), created);
            if (counter == null) {
                counter = created;
            }
        }
        counter.incrementAndGet();
        return true;
    }

    public int winsOf(UUID playerId) {
        AtomicInteger counter = playerId == null ? null : wins.get(playerId);
        return counter == null ? 0 : counter.get();
    }

    /** 按胜场降序；同分按 UUID 稳定排序，避免每次顺序不同。 */
    public List<UUID> ranking() {
        List<UUID> ranked = new ArrayList<UUID>(wins.keySet());
        Collections.sort(ranked, new Comparator<UUID>() {
            @Override
            public int compare(UUID left, UUID right) {
                int byWins = Integer.compare(wins.get(right).get(), wins.get(left).get());
                return byWins != 0 ? byWins : left.toString().compareTo(right.toString());
            }
        });
        return ranked;
    }

    public int size() {
        return wins.size();
    }

    public void clear() {
        wins.clear();
    }
}
