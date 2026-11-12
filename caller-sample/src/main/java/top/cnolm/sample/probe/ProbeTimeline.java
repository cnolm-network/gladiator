package top.cnolm.sample.probe;

import org.bukkit.Bukkit;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;

import java.util.ArrayList;
import java.util.List;

/**
 * 探针用的时间线：tick 时钟 + 全局序号 + 文本行。
 *
 * <p>为什么要"序号"：同一 tick 内可能发生多个事件（Ready 与 pushGame 常常同 tick），
 * 只比较 tick 无法判断先后。序号在同一 tick 内单调递增，用于断言"Ready 早于 pushGame"。
 */
public final class ProbeTimeline {

    private final List<String> lines = new ArrayList<String>();
    private BukkitTask clock;
    private int tick;
    private int seq;

    public void start(Plugin plugin) {
        clock = Bukkit.getScheduler().runTaskTimer(plugin, new Runnable() {
            @Override
            public void run() {
                tick++;
            }
        }, 1L, 1L);
    }

    public void stop() {
        if (clock != null) {
            clock.cancel();
            clock = null;
        }
    }

    public int tick() {
        return tick;
    }

    public int seq() {
        return seq;
    }

    /**
     * 追加一行并返回它的序号（同一 tick 内可比先后）。
     */
    public int add(String text) {
        int current = ++seq;
        lines.add(String.format("t+%-3d #%02d %s", tick, current, text));
        return current;
    }

    public List<String> lines() {
        return new ArrayList<String>(lines);
    }
}
