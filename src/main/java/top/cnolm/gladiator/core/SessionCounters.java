package top.cnolm.gladiator.core;

import top.cnolm.gladiator.api.Analysis;

/**
 * 会话级指标累加器。
 *
 * <p>M2/M3 的标准 Bukkit 事件监听器负责写入，{@code analyze()} 只读取不可变快照。
 * 所有字段用同一把锁保护，写入可以在采样线程进行，快照在主线程生成。
 */
public final class SessionCounters {

    private double damageDealt;
    private double damageTaken;
    private double pathMeters;
    private int blocksPlaced;
    private int blocksBroken;
    private int ruleDenials;
    private int actionsAccepted;
    private int actionsDropped;

    public synchronized void addDamageDealt(double value) {
        damageDealt += value;
    }

    public synchronized void addDamageTaken(double value) {
        damageTaken += value;
    }

    public synchronized void addPathMeters(double value) {
        pathMeters += value;
    }

    public synchronized void addBlockPlaced() {
        blocksPlaced++;
    }

    public synchronized void addBlockBroken() {
        blocksBroken++;
    }

    public synchronized void addRuleDenial() {
        ruleDenials++;
    }

    /** {@code notice()} 等通道的投递结果；G-101 建立 WS 后由动作通道继续累加。 */
    public synchronized void recordAction(boolean dropped) {
        if (dropped) {
            actionsDropped++;
        } else {
            actionsAccepted++;
        }
    }

    public synchronized Analysis.BotMetrics snapshot(String botId, double aliveSeconds) {
        return new Analysis.BotMetrics(
                botId,
                aliveSeconds,
                damageDealt,
                damageTaken,
                blocksPlaced,
                blocksBroken,
                pathMeters,
                ruleDenials,
                actionsAccepted,
                actionsDropped);
    }
}
