package top.cnolm.arena.simplify;

import org.bukkit.plugin.Plugin;
import top.cnolm.arena.ArenaPlugin;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

/**
 * 把 {@link Recorder} 落盘成 JSONL：{@code plugins/Arena/records/<matchId>-<yyyyMMdd-HHmmss>.jsonl}。
 *
 * <p>为什么单独一个类：{@link Recorder} 只懂「记录」，不该知道服务器目录与命名规则；
 * 命令（{@code /arena record dump}）与对局结束的自动导出都要用同一套命名，避免出现两份。
 *
 * <p>文件名按 {@code yyyyMMdd-HHmmss} 排序即时间序，便于直接 {@code ls} 找最近一局；
 * 同一秒内重复导出由调用方自己区分（内容会被覆盖，命令会给绝对路径提示）。
 */
public final class RecorderExport {

    /** 相对插件数据目录的录像目录名（即 {@code plugins/Arena/records}）。 */
    public static final String RECORDS_DIR = "records";
    /** 扩展名（JSON Lines）。 */
    public static final String EXTENSION = ".jsonl";

    private static final DateTimeFormatter STAMP =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.ROOT);
    /** 文件名里不允许出现的字符（Windows 更严格，这里按 Windows 口径统一）。 */
    private static final String ILLEGAL = "[\\\\/:*?\"<>|\\s]";

    private RecorderExport() {
    }

    /** 录像目录（不创建）；Arena 未启用时抛 {@link IllegalStateException}。 */
    public static Path recordsDirectory() {
        ArenaPlugin plugin = ArenaPlugin.getInstance();
        if (plugin == null) {
            throw new IllegalStateException("Arena 未启用，无法定位录像目录");
        }
        return directoryOf(plugin);
    }

    /** 插件数据目录下的录像目录（不创建）。 */
    public static Path directoryOf(Plugin plugin) {
        if (plugin == null) {
            throw new IllegalArgumentException("plugin");
        }
        return plugin.getDataFolder().toPath().resolve(RECORDS_DIR);
    }

    /**
     * 用当前时间戳导出到默认目录。
     *
     * @return 实际写入的文件路径
     */
    public static Path export(Recorder recorder, String matchId) throws IOException {
        return export(recorder, matchId, recordsDirectory(), null, System.currentTimeMillis());
    }

    /**
     * 导出到默认目录，文件名可指定（不含扩展名也可以，会自动补 {@code .jsonl}）。
     *
     * @return 实际写入的文件路径
     */
    public static Path export(Recorder recorder, String matchId, String fileName) throws IOException {
        return export(recorder, matchId, recordsDirectory(), fileName, System.currentTimeMillis());
    }

    /**
     * 全参导出：目录、文件名、时间戳都由调用方给（便于单测与命令复用）。
     *
     * @param fileName {@code null}/空 = {@code <matchId>-<yyyyMMdd-HHmmss>.jsonl}
     * @return 实际写入的文件路径
     */
    public static Path export(Recorder recorder, String matchId, Path directory,
                              String fileName, long nowMs) throws IOException {
        if (recorder == null) {
            throw new IllegalArgumentException("recorder");
        }
        if (directory == null) {
            throw new IllegalArgumentException("directory");
        }
        String name = isBlank(fileName)
                ? defaultFileName(matchId, nowMs)
                : normalizeFileName(fileName);
        Path file = directory.resolve(name);
        recorder.exportTo(file);
        return file;
    }

    /** 默认文件名：{@code <matchId>-<yyyyMMdd-HHmmss>.jsonl}（matchId 为空时退化成 {@code match}）。 */
    public static String defaultFileName(String matchId, long nowMs) {
        String id = sanitize(isBlank(matchId) ? "match" : matchId.trim());
        return id + "-" + timestamp(nowMs) + EXTENSION;
    }

    /** {@code yyyyMMdd-HHmmss}（本地时区，与人看日志的习惯一致）。 */
    public static String timestamp(long nowMs) {
        return STAMP.format(Instant.ofEpochMilli(nowMs).atZone(ZoneId.systemDefault()));
    }

    /** 用户给的文件名：去掉路径分隔与非法字符，缺扩展名时补上。 */
    static String normalizeFileName(String raw) {
        String name = sanitize(raw.trim());
        if (name.isEmpty()) {
            name = "record";
        }
        if (!name.toLowerCase(Locale.ROOT).endsWith(EXTENSION)) {
            name = name + EXTENSION;
        }
        return name;
    }

    private static String sanitize(String raw) {
        return raw == null ? "" : raw.replaceAll(ILLEGAL, "_");
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
