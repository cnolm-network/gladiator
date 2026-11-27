/**
 * G-735：计划队列与串行执行器。
 *
 * 这两个类把「决策 20Hz」和「执行 ~5Hz」解耦——是本轮把有效决策率从 4Hz 拉回 20Hz 的关键。
 * 单测覆盖的是**语义**而不是时序：
 * - 队列容量 1：同步连提两个 → 只有最后一个执行，旧的计入 `superseded`；
 * - 正在执行的计划**不打断**（否则会留下半截 move），执行期间到的计划只排队；
 * - 执行抛错只记 `failed`，不影响后续计划。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { PlanExecutor, PlanQueue, makePlan } from '../src/plan-queue.js';

const sleep = (ms) => new Promise((resolve) => {
  setTimeout(resolve, ms);
});

test('makePlan 复制 intents，避免调用方后续改动影响已入队的计划', () => {
  const intents = [{ type: 'idle' }];
  const plan = makePlan({ planId: 7, kind: 'action', intents });
  intents.push({ type: 'attack' });
  assert.equal(plan.planId, 7);
  assert.equal(plan.kind, 'action');
  assert.equal(plan.intents.length, 1);
  assert.ok(plan.createdAt > 0);
});

test('PlanQueue 容量 1：新计划顶掉还没执行的旧计划', () => {
  const queue = new PlanQueue({ size: 1 });
  assert.deepEqual(queue.push(makePlan({ planId: 1 })), { accepted: true, superseded: 0 });
  assert.deepEqual(queue.push(makePlan({ planId: 2 })), { accepted: true, superseded: 1 });
  assert.equal(queue.length, 1);
  assert.equal(queue.shift().planId, 2, '应保留最新计划');
  assert.equal(queue.shift(), null);
  assert.equal(queue.stats.superseded, 1);
  assert.equal(queue.stats.taken, 1);
});

test('PlanQueue 容量 2：保留最近两个，丢最旧', () => {
  const queue = new PlanQueue({ size: 2 });
  queue.push(makePlan({ planId: 1 }));
  queue.push(makePlan({ planId: 2 }));
  assert.deepEqual(queue.push(makePlan({ planId: 3 })), { accepted: true, superseded: 1 });
  assert.deepEqual([queue.shift().planId, queue.shift().planId], [2, 3]);
});

test('PlanQueue.clear 丢掉全部待执行计划', () => {
  const queue = new PlanQueue({ size: 2 });
  queue.push(makePlan({ planId: 1 }));
  queue.push(makePlan({ planId: 2 }));
  assert.equal(queue.clear(), 2);
  assert.equal(queue.empty, true);
  assert.equal(queue.stats.dropped, 2);
});

test('PlanExecutor 串行：同一时刻只执行一个计划，且不打断正在执行的那个', async () => {
  const events = [];
  const executor = new PlanExecutor({
    execute: async (plan) => {
      events.push(`start${plan.planId}`);
      await sleep(20);
      events.push(`end${plan.planId}`);
    },
  });

  executor.submit(makePlan({ planId: 1 }));
  await sleep(5);                       // 让 1 真正开始执行
  assert.equal(executor.busy, true);
  executor.submit(makePlan({ planId: 2 }));
  executor.submit(makePlan({ planId: 3 }));   // 顶掉 2（容量 1）
  await executor.idle();

  assert.deepEqual(events, ['start1', 'end1', 'start3', 'end3'],
    '1 必须完整跑完；2 被 3 顶掉，不执行');
  assert.equal(executor.stats.superseded, 1);
  assert.equal(executor.stats.executed, 2);
  assert.equal(executor.busy, false);
});

test('PlanExecutor 同步连提两个：只执行最后一个（不能执行两帧前的画面）', async () => {
  const executed = [];
  const executor = new PlanExecutor({
    execute: async (plan) => {
      executed.push(plan.planId);
      await sleep(1);
    },
  });

  executor.submit(makePlan({ planId: 'a' }));
  executor.submit(makePlan({ planId: 'b' }));
  await executor.idle();

  assert.deepEqual(executed, ['b']);
  assert.equal(executor.stats.submitted, 2);
  assert.equal(executor.stats.superseded, 1);
});

test('PlanExecutor 执行抛错：记 failed，后续计划照常执行', async () => {
  const executed = [];
  const errors = [];
  const executor = new PlanExecutor({
    execute: async (plan) => {
      if (plan.planId === 'boom') {
        throw new Error('动作炸了');
      }
      executed.push(plan.planId);
    },
    onError: (error, plan) => errors.push([plan.planId, error.message]),
  });

  executor.submit(makePlan({ planId: 'boom' }));
  await sleep(5);
  executor.submit(makePlan({ planId: 'ok' }));
  await executor.idle();

  assert.deepEqual(executed, ['ok']);
  assert.equal(executor.stats.failed, 1);
  assert.deepEqual(errors, [['boom', '动作炸了']]);
});

test('PlanExecutor.clear 只丢待执行的，不影响正在执行的那个', async () => {
  const events = [];
  const executor = new PlanExecutor({
    execute: async (plan) => {
      events.push(`start${plan.planId}`);
      await sleep(15);
      events.push(`end${plan.planId}`);
    },
  });

  executor.submit(makePlan({ planId: 1 }));
  await sleep(5);
  executor.submit(makePlan({ planId: 2 }));
  const dropped = executor.clear();
  await executor.idle();

  assert.equal(dropped, 1, '只丢掉排队中的 2');
  assert.deepEqual(events, ['start1', 'end1']);
  assert.equal(executor.stats.cleared, 1);
});

test('PlanExecutor 空队列时 submit 会重新拉起执行循环（无丢帧窗口）', async () => {
  const executed = [];
  const executor = new PlanExecutor({
    execute: async (plan) => {
      executed.push(plan.planId);
    },
  });
  // 连续 5 轮「提交 → 等空」，每轮都必须执行到（覆盖 _kick/_drain 的关窗补踢）。
  for (let index = 1; index <= 5; index += 1) {
    executor.submit(makePlan({ planId: index }));
    await executor.idle();
  }
  assert.deepEqual(executed, [1, 2, 3, 4, 5]);
});
