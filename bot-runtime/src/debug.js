/**
 * 调试日志（G-717）。
 *
 * <p>默认**静默**：这些是排查用、频率高的细节（例如远距离追击时的 `OUT_OF_REACH_CHASED`），
 * 平时留在 info 里会把真机日志刷满，反而淹掉「决策/失败」这类该一眼看见的行。
 *
 * <p>打开方式（任一）：
 * - CLI `--debug`；
 * - 环境变量 `GLADIATOR_DEBUG=1`。
 *
 * <p>输出前缀固定 `[debug]`，grep 时可以和普通日志分开。
 */

let enabled = false;

/** 打开/关闭调试输出（index.js 在解析完 CLI 后调用一次；测试里也用）。 */
export function setDebugEnabled(value) {
  enabled = Boolean(value);
}

export function isDebugEnabled() {
  return enabled;
}

/** 调试日志：关闭时是纯 no-op（不拼字符串、不进 stdout）。 */
export function debugLog(...parts) {
  if (!enabled) {
    return;
  }
  console.log('[debug]', ...parts);
}
