/**
 * G-735：`hz=` / `applyMsMean=` 的统计口径 + 心跳新增字段。
 *
 * 这两个数是本轮的**验收指标**（决策率 ≥18Hz、执行耗时可见），必须能离线验，
 * 否则又回到「只能看真机日志」的老路。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { DecisionRate, RollingMean } from '../src/brain-metrics.js';
import { formatHeartbeat } from '../src/brain-telemetry.js';

test('DecisionRate：20Hz 的采样应算出约 20Hz（不是「样本数/窗口」）', () => {
  const rate = new DecisionRate({ windowMs: 10000 });
  for (let index = 0; index < 20; index += 1) {
    rate.mark(1000 + index * 50);          // 0.95 秒内 20 帧
  }
  // 首末间隔口径：(20-1)/0.95 ≈ 20.0
  assert.ok(Math.abs(rate.hz(2000) - 20) < 0.6, `实际 ${rate.hz(2000)}`);
  assert.equal(rate.count(2000), 20);
});

test('DecisionRate：样本不足 2 个时返回 0（开局不显示假频率）', () => {
  const rate = new DecisionRate({ windowMs: 10000 });
  assert.equal(rate.hz(1000), 0);
  rate.mark(1000);
  assert.equal(rate.hz(1000), 0);
});

test('DecisionRate：窗口外的样本被清掉（长时间静默后归零）', () => {
  const rate = new DecisionRate({ windowMs: 10000 });
  for (let index = 0; index < 10; index += 1) {
    rate.mark(1000 + index * 50);
  }
  assert.ok(rate.hz(1500) > 0);
  assert.equal(rate.hz(60_000), 0, '60 秒后的窗口里已经没有样本');
  assert.equal(rate.count(60_000), 0);
});

test('DecisionRate：reset 清空（换局/停脑时用）', () => {
  const rate = new DecisionRate({ windowMs: 10000 });
  rate.mark(1000).mark(1050);
  assert.ok(rate.hz(1100) > 0);
  rate.reset();
  assert.equal(rate.hz(1100), 0);
});

test('RollingMean：滑动窗口只保留最近 N 个样本', () => {
  const mean = new RollingMean({ size: 3 });
  mean.push(10);
  mean.push(20);
  assert.equal(mean.mean(), 15);
  mean.push(30);
  assert.equal(mean.mean(), 20);
  mean.push(60);                            // 挤掉 10 → (20+30+60)/3 = 36.67
  assert.ok(Math.abs(mean.mean() - 110 / 3) < 1e-9);
  assert.equal(mean.count, 3);
});

test('RollingMean：非数被忽略、空窗口为 0', () => {
  const mean = new RollingMean({ size: 3 });
  assert.equal(mean.mean(), 0);
  mean.push(Number.NaN);
  mean.push(undefined);
  assert.equal(mean.mean(), 0);
  mean.push(40);
  assert.equal(mean.mean(), 40);
  mean.reset();
  assert.equal(mean.mean(), 0);
});

test('心跳带 G-735 字段：hz / superseded / applyMsMean，且 lastTick 是整帧耗时', () => {
  const line = formatHeartbeat({
    decisions: 400,
    attacks: 19,
    busy: false,
    busySkips: 2,
    hz: 19.6,
    superseded: 320,
    obsMs: 1,
    inferMs: 2,
    applyMs: 95,
    applyMsMean: 96.34,
    lastTick: 4,
    lastAction: null,
    target: 'Glad_T9',
  }, 10_000);

  assert.match(line, /decisions=400/);
  assert.match(line, /hz=19\.6/);
  assert.match(line, /superseded=320/);
  assert.match(line, /obsMs=1 /);
  assert.match(line, /inferMs=2 /);
  assert.match(line, /applyMs=95 /);
  assert.match(line, /applyMsMean=96\.3/);
  assert.match(line, /lastTick=4ms/, 'lastTick 是整帧耗时，不是「距上次」');
  assert.match(line, /busy=0/);
});

test('心跳保持向后兼容：没有 G-735 字段时不出现它们，lastTick 退回「距上次」', () => {
  const line = formatHeartbeat({ decisions: 1, lastTickAt: 0, target: null }, 500);
  assert.doesNotMatch(line, /hz=/);
  assert.doesNotMatch(line, /superseded=/);
  assert.doesNotMatch(line, /applyMsMean=/);
  assert.match(line, /lastTick=never/);
});
