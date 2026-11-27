package top.cnolm.gladiator.bot;

import org.junit.Test;
import top.cnolm.gladiator.core.PlayerTelemetry;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * 「对手血量没读到」不能变成「对手血量 = 0」的回归钉子（G-727 配套）。
 *
 * <p>链路（三段，任何一段把「没有」写成 0，大脑就会以为对手已经被打死）：
 *
 * <ol>
 *   <li><b>Java 侧</b>：Bot 运行时用 RCON 调 {@code /gladiator stats <对手>}；
 *       对手不在线/不存在时 {@link PlayerTelemetry#format} 返回
 *       {@link PlayerTelemetry#OFFLINE_MESSAGE}，<b>整行不含 health 字段</b>；</li>
 *   <li><b>JS 解析</b>：{@code bot-runtime/src/rcon.js::parseStatsLine} 见
 *       {@code "stats 失败"} 子串直接返回 {@code null}（已有测试：
 *       {@code bot-runtime/test/rcon-client.test.mjs:162}）；</li>
 *   <li><b>JS 记账</b>：{@code bot-runtime/src/index.js::recordOpponentHealth(null)} 早退，
 *       既不写 {@code opponentHealth = 0}，也不触发击杀计数。</li>
 * </ol>
 *
 * <p>第 3 段是纯运行时内部函数（未导出、`index.js` 有 import 时副作用），本模块单测覆盖不到；
 * 所以这里钉住**最上游的契约**：拿不到玩家时绝不产出 {@code health=0}，且文案保持
 * {@code "stats 失败"} 这个可被子串识别的形态。真机事故：2026-11-18 23:32 无 RCON 时
 * 每 tick 都记一次击杀、并把 {@code target_health} 压成 0。
 */
public class OpponentHealthTelemetryTest {

    @Test
    public void offlineTargetDoesNotEmitHealthZero() {
        String line = PlayerTelemetry.format(null);
        assertEquals("离线文案被改过：JS 侧按子串判「没有数据」，改动会破坏契约",
                PlayerTelemetry.OFFLINE_MESSAGE, line);
        assertFalse("离线时必须整行没有 health 字段；一旦写出 health=0.0，"
                + "recordOpponentHealth 会把它当成「对手血量归零」并记一次击杀",
                line.contains("health="));
    }

    @Test
    public void offlineMessageKeepsTheParserContract() {
        // parseStatsLine 的判据是 "stats 失败" 子串 → 返回 null（而不是 {health: 0}）。
        assertTrue("JS 解析器靠这个子串判「没有数据」：" + PlayerTelemetry.OFFLINE_MESSAGE,
                PlayerTelemetry.OFFLINE_MESSAGE.contains("stats 失败"));
        // 反过来：这条文案不能长得像正常的 stats 行（否则解析器会把它当数据）。
        assertFalse("离线文案不能带 '=' 键值，看起来像数据就一定被误读",
                PlayerTelemetry.OFFLINE_MESSAGE.contains("="));
    }
}
