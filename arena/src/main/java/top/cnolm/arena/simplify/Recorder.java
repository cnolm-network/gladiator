package top.cnolm.arena.simplify;

import org.bukkit.entity.Player;
import org.bukkit.event.Event;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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

    // ------------------------------------------------------------ JSONL 导出

    /**
     * 把全部记录导出为 JSONL 行（一行一条，UTF-8 无 BOM，按写入顺序）。
     *
     * <p>字段固定为 {@code seq, at, category, from, content}：
     * <ul>
     *   <li>{@code seq}：1 起的序号，便于外部按序去重；</li>
     *   <li>{@code at}：{@link TimeGetter} 的时间戳——能解析成整数就写数字（epoch 毫秒），
     *       否则写字符串（自定义时钟）；</li>
     *   <li>{@code category}：{@code event|position|stats}，由记录来源归并
     *       （{@code Custom} 归到 {@code event}，原文保留在 {@code from}）；</li>
     *   <li>{@code from}：记录者，形如 {@code EventHandler:<类名>}；</li>
     *   <li>{@code content}：记录正文（原样，未结构化）。</li>
     * </ul>
     *
     * <p>注意 {@code position} 记的是<strong>服务端观测到的位置</strong>，不含 Bot 的意图；
     * 想知道 Bot 当时想干什么请看运行时 console（`[intent]` / `[brain]` 日志）。
     */
    public List<String> dumpJsonLines() {
        List<String> lines = new ArrayList<String>(contents.size());
        int seq = 1;
        for (Content content : contents) {
            lines.add(toJsonLine(seq, content));
            seq++;
        }
        return lines;
    }

    /**
     * 写成 JSONL 文件（覆盖写，父目录不存在则创建），UTF-8 无 BOM。
     *
     * @throws IOException 目录不可建或写入失败
     */
    public void exportTo(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        Path parent = file.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        List<String> lines = dumpJsonLines();
        StringBuilder sb = new StringBuilder(lines.size() * 96);
        for (String line : lines) {
            sb.append(line).append('\n');
        }
        Files.write(file, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static String toJsonLine(int seq, Content content) {
        StringBuilder sb = new StringBuilder(96);
        sb.append('{');
        sb.append("\"seq\":").append(seq);
        sb.append(",\"at\":").append(atJson(content.timeStamp));
        sb.append(",\"category\":").append(quote(categoryOf(content.from)));
        sb.append(",\"from\":").append(quote(content.from));
        sb.append(",\"content\":").append(quote(content.line));
        return sb.append('}').toString();
    }

    /** {@code at} 只在能解析成整数时写数字，否则退回字符串，避免把自定义时钟写坏。 */
    private static String atJson(String timeStamp) {
        String raw = timeStamp == null ? "" : timeStamp.trim();
        if (raw.isEmpty()) {
            return "null";
        }
        try {
            return String.valueOf(Long.parseLong(raw));
        } catch (NumberFormatException e) {
            return quote(timeStamp);
        }
    }

    /** {@code from} 前缀 → 对外 category（只允许 event/position/stats 三种）。 */
    static String categoryOf(String from) {
        if (from == null) {
            return "event";
        }
        if (from.startsWith("PositionTracker:")) {
            return "position";
        }
        if (from.startsWith("MatchStats:")) {
            return "stats";
        }
        return "event";
    }

    /** 最小 JSON 字符串转义（不引 gson：板子只依赖 Spigot API）。 */
    static String quote(String value) {
        if (value == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder(value.length() + 16);
        sb.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", Integer.valueOf(c)));
                    } else {
                        sb.append(c);
                    }
                    break;
            }
        }
        return sb.append('"').toString();
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
