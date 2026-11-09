import assert from 'node:assert/strict';
import test from 'node:test';

import { describeLoginFailure, SPAWN_FAILURE_CODES } from '../src/minecraft.js';

test('online-mode 鉴权失败归类为 ONLINE_MODE', () => {
  const failure = describeLoginFailure(new Error('Failed to verify username!'));
  assert.equal(failure.code, SPAWN_FAILURE_CODES.ONLINE_MODE);
  assert.match(failure.message, /online-mode/);
});

test('白名单拒绝归类为 NOT_WHITELISTED', () => {
  const failure = describeLoginFailure('You are not white-listed on this server!');
  assert.equal(failure.code, SPAWN_FAILURE_CODES.NOT_WHITELISTED);
  assert.match(failure.message, /whitelist\.json/);
});

test('协议版本不符归类为 VERSION_MISMATCH', () => {
  const failure = describeLoginFailure(new Error('Unsupported version 1.20.4'));
  assert.equal(failure.code, SPAWN_FAILURE_CODES.VERSION_MISMATCH);
  assert.match(failure.message, /1\.8\.8/);
});

test('连接被拒绝归类为 SERVER_UNREACHABLE', () => {
  const failure = describeLoginFailure(new Error('connect ECONNREFUSED 127.0.0.1:25565'));
  assert.equal(failure.code, SPAWN_FAILURE_CODES.SERVER_UNREACHABLE);
});

test('未知错误归入 LOGIN_FAILED 且保留原始信息', () => {
  const failure = describeLoginFailure(new Error('something odd happened'));
  assert.equal(failure.code, SPAWN_FAILURE_CODES.LOGIN_FAILED);
  assert.match(failure.message, /something odd happened/);
});
