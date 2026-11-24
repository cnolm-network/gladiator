package top.cnolm.arena.simplify;

import org.bukkit.entity.Player;
import org.bukkit.event.Event;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 对局记录器：按时间顺序收集对局事件、位置与统计。
 *
 * <p>所有 recordXXX 都在服务端主线程调用（Bukkit 事件与调度任务均在主线程），
 * 因此内部 List 无需并发保护；导出时返回只读视图，避免调用方改动内部记录。
 */
public class Recorder {
    private final List<Content> contents = new ArrayList<>();
    private final TimeGetter getter;

    public Recorder(TimeGetter getter) {
        this.getter = Objects.requireNonNull(getter, "getter");
    }

    /** 记录一次 Bukkit/Arena 事件，category 取事件类名。 */
    public void recordEvent(Class<?> fromClass, Event e, String content) {
        add("EventHandler:" + fromClass.getName(), e.getClass().getName(), content);
    }

    /** 记录玩家某一时刻的位置。 */
    public void recordPosition(Class<?> fromClass, Player player) {
        add("PositionTracker:" + fromClass.getName(), "moveto", "{loc=" + player.getLocation() + "}");
    }

    /** 记录对局统计（击杀、淘汰、结算等）。 */
    public void recordMatchStats(Class<?> fromClass, String category, String content) {
        add("MatchStats:" + fromClass.getName(), category, content);
    }

    /** 自定义记录。 */
    public void record(Class<?> fromClass, String category, String content) {
        add("Custom:" + fromClass.getName(), category, content);
    }

    /** 统一写入：时间戳只取一次，避免 List 索引与 Content 内部时间不一致。 */
    private void add(String from, String category, String line) {
        contents.add(new Content(getter.getTime(), from, category, line));
    }

    /** 当前记录条数。 */
    public int size() {
        return contents.size();
    }

    /** 只读导出，按写入顺序（近似时间顺序）排列。 */
    public List<Content> getContents() {
        return Collections.unmodifiableList(contents);
    }

    /** 清空记录，供复用同一 Recorder 的场景。 */
    public void clear() {
        contents.clear();
    }

    public static final class Content {
        public final String timeStamp;
        public final String from;
        public final String category;
        public final String line;

        public Content(String timeStamp, String from, String category, String line) {
            this.timeStamp = timeStamp;
            this.from = from;
            this.category = category;
            this.line = line;
        }

        @Override
        public String toString() {
            return timeStamp + " [" + from + "] " + category + " " + line;
        }
    }

    public interface TimeGetter {
        default String getTime() {
            return String.valueOf(input());
        }

        Object input();
    }
}
