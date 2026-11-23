/**
 * 37 维特征编码 + 离散动作解码（`spaces.py` 的 JS 镜像）。
 *
 * <p>**必须与 `training/gladiator_env/spaces.py::encode_observation` 逐项一致**：
 * 字段顺序、默认值、归一化尺度、clip 全部照抄。任何一侧改了观测定义，另一侧必须同步，
 * 否则模型看到的输入和训练时不同（实测会导致策略直接失效）。
 * 每个赋值后面标注了 spaces.py 的行号，便于逐行对照。
 *
 * <p>同时提供 `decodeAction()`：把 `ACTION_NAMES` 的离散动作编号翻译成协议动作列表，
 * 逻辑与 `spaces.py::decode_action` 一致（含「够不到就先追一步」与「没目标就待机」）。
 * 运行时需要它才能在本地闭环决策，而不依赖 Layer 3 下发动作。
 */

import { HEAD_SIZES_V2 } from './action-space-v2.js';

export const OBS_DIM = 37;
export const NUM_ACTIONS = 11;

/** G-711：跨帧特征在观测里的起始下标与分母（与 spaces.py 的 PREV_ACTION_OFFSET / FRAMES_NORMALIZER 一致）。 */
export const PREV_ACTION_OFFSET = 25;
export const FRAMES_SINCE_ATTACK_INDEX = 35;
export const FRAMES_SINCE_HIT_INDEX = 36;
export const CROSS_FRAME_NORMALIZER = 50.0;

/** 观测裁剪范围（spaces.py:51 OBS_CLIP）。 */
export const OBS_CLIP = 5.0;

/** 默认攻击距离（= BotProfile.LEGIT_REACH_MAX；1.8 原版玩家约 3 格，不是 3.99）。 */
export const DEFAULT_REACH = 3.0;

/**
 * G-711 帧计数的取值语义：`undefined` 退回去读帧里的值，`null` = 没有信息 → 饱和值。
 * （Python 侧 `None` 也是这个意思：`frames / 50` 取 1.0。）
 */
function framesSince(explicit, fromFrame) {
  const value = explicit !== undefined ? explicit : fromFrame;
  if (value === null || value === undefined) {
    return CROSS_FRAME_NORMALIZER;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : CROSS_FRAME_NORMALIZER;
}

/** 1.8 受击无敌帧（秒），`self_attack_cooldown` 的归一化分母（spaces.py:56）。 */
export const ATTACK_COOLDOWN_S = 0.5;

/** 复合动作的移动时长（spaces.py:64 MOVE_MS）。 */
export const MOVE_MS = 180;

/** 观测距离与运行时 eyeDistance 的偏差余量（spaces.py:223 REACH_TOLERANCE）。 */
export const REACH_TOLERANCE = 0.15;

/** 与 `spaces.py:19 OBS_FIELDS` 完全同序。 */
export const OBS_FIELDS_JS = Object.freeze([
  'self_health',            // 0
  'self_food',              // 1
  'self_saturation',        // 2
  'self_on_ground',         // 3
  'self_yaw_sin',           // 4
  'self_yaw_cos',           // 5
  'self_pitch',             // 6
  'target_dx',              // 7
  'target_dy',              // 8
  'target_dz',              // 9
  'target_distance',        // 10
  'target_health',          // 11
  'target_on_ground',       // 12
  'has_target',             // 13
  'step_fraction',          // 14
  'state_alive',            // 15
  'self_held_durability',   // 16
  'self_armor_remaining',   // 17
  'self_gapples',           // 18
  'opponent_speed',         // 19
  'opponent_vertical_speed',// 20
  'self_since_attack',      // 21
  'in_reach',               // 22
  'self_vertical_speed',    // 23 G-702：自身 vy，clip(-5,5) 后 /5
  'self_attack_cooldown',   // 24 G-702：距上次 attack 还差多少到 0.5s，clip(0,0.5) 后 /0.5
  // --- G-711：跨帧信息（25..36），顺序 = ACTION_HEADS_V2 ---
  'prev_move',              // 25 上一帧 move 头 / 5
  'prev_look_yaw',          // 26 上一帧 look_yaw 头 / 7
  'prev_look_pitch',        // 27 上一帧 look_pitch 头 / 5
  'prev_attack',            // 28 上一帧 attack 头 / 2
  'prev_use',               // 29 上一帧 use 头 / 3
  'prev_jump',              // 30 上一帧 jump 头 / 2
  'prev_sneak',             // 31 上一帧 sneak 头 / 2
  'prev_sprint',            // 32 上一帧 sprint 头 / 2
  'prev_slot',              // 33 上一帧 slot 头 / 9
  'prev_sprint_reset',      // 34 上一帧 sprint_reset 头 / 3
  'frames_since_attack',    // 35 距上次 attack 的帧数，clip(0,50) 后 /50
  'frames_since_hit',       // 36 距上次受伤的帧数，clip(0,50) 后 /50
]);

/** 动作名（spaces.py:49 ACTION_NAMES），下标即离散动作编号。 */
export const ACTION_NAMES = Object.freeze([
  'idle',
  'approach',
  'attack',
  'strafe_left',
  'strafe_right',
  'retreat',
  'jump',
  'use_rod',
  'eat_gapple',
  'block',
  'switch_sword',
]);

/** `spaces.py:71 _f`：非数字 / NaN / Inf 一律回落到默认值。 */
function num(value, fallback = 0.0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return parsed;
}

function clamp(value, low, high) {
  if (value < low) {
    return low;
  }
  return value > high ? high : value;
}

/**
 * 找最近的其它玩家（spaces.py:81 nearest_player）。
 *
 * @param {object} observation 观测帧
 * @param {string} username    自己
 * @param {Iterable<string>|null} allowed 只接受名单内的玩家（大小写不敏感）
 * @returns {object|null}
 */
export function nearestPlayer(observation, username, allowed = null) {
  const allow = allowed ? new Set(Array.from(allowed, (name) => String(name).toLowerCase())) : null;
  const self = String(username || '').toLowerCase();
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const entity of (observation && observation.entities) || []) {
    if (!entity || typeof entity !== 'object') {
      continue;
    }
    if (entity.kind !== 'player') {
      continue;
    }
    const name = String(entity.name ?? '');
    if (name.toLowerCase() === self) {
      continue;
    }
    if (allow && !allow.has(name.toLowerCase())) {
      continue;
    }
    const distance = num(entity.distance, Number.POSITIVE_INFINITY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = entity;
    }
  }
  return best;
}

/**
 * 从实体里读对手血量（协议实体列表不带血量，但 mineflayer 有时能从 metadata 拿到）。
 *
 * <p>返回 `null` 表示读不到，调用方应回落到 RCON `/gladiator stats`（块 B 提供）。
 *
 * <p><b>玩家实体的 0 不是「死了」，而是「不知道」</b>：1.8 不通过实体元数据下发其他玩家的血量，
 * mineflayer 对 player 实体的 `health` 默认是 0。早期实现把 0 当成真实血量，导致 brain 在对手
 * 满血时记「击杀」、并把 `target_health=0` 喂给策略（真机 2026-11-18 23:32 复现：
 * 开局就打印「对手血量归零」）。所以 player 只接受 > 0 的值，其余一律当未知。
 */
export function entityHealth(entity) {
  if (!entity) {
    return null;
  }
  const isPlayer = entity.kind === 'player' || entity.type === 'player';
  const accept = (value) => {
    const health = Number(value);
    if (!Number.isFinite(health)) {
      return null;
    }
    if (isPlayer && health <= 0) {
      return null;
    }
    return health;
  };
  const direct = accept(entity.health);
  if (direct !== null) {
    return direct;
  }
  const meta = entity.metadata;
  if (Array.isArray(meta)) {
    // 1.8 里 LivingEntity 的血量在 metadata index 6（float）。
    for (const entry of meta) {
      if (!entry || typeof entry !== 'object') {
        continue;
      }
      if (entry.key === 6 || entry.key === 'health') {
        const health = accept(entry.value);
        if (health !== null) {
          return health;
        }
      }
    }
  } else if (meta && typeof meta === 'object') {
    for (const key of ['6', 'health']) {
      const health = accept(meta[key]);
      if (health !== null) {
        return health;
      }
    }
  }
  return null;
}

/**
 * 编码一帧观测为 37 维特征（`spaces.py encode_observation`）。
 *
 * @param {object} observation `buildObservation()` 的输出
 * @param {object} options
 * @param {string} options.username       自己
 * @param {number} [options.stepFraction] 当前步数 / 最大步数
 * @param {number|null} [options.opponentHealth] 对手血量（优先于实体字段）
 * @param {Iterable<string>|null} [options.allowed] 只把这些玩家当目标
 * @param {number} [options.reach]        自己的攻击距离（默认 3.0，1.8 原版水平）
 * @param {number} [options.sinceAttack]  距上次 attack 的秒数（默认 5）
 * @param {number|null} [options.selfVelocityY] 自身 vy（G-702；缺省回落读 `observation.self.velocity.y`）
 * @param {number|null} [options.attackCooldownRemaining] 距冷却结束还有多少秒（G-702；缺省 0）
 * @param {ArrayLike<number>|null} [options.prevAction] 上一帧十头下标（G-711；缺省读
 *   `observation.crossFrame.prevAction`，都没有则 10 维全 0 = 「不知道」）
 * @param {number|null} [options.framesSinceAttack] 距上次出手的帧数（G-711；缺省 1.0 = 很久以前）
 * @param {number|null} [options.framesSinceHit] 距上次受伤的帧数（G-711；同上）
 * @returns {Float64Array} 长度 37
 *
 * <p>返回 float64（numpy 那边是 float32，逐项差 ≤1e-7，属 float32 舍入；
 * 策略前向一律按 float64 算，所以对最终概率无可见影响）。
 */
export function encodeObservation(observation, {
  username = '',
  stepFraction = 0.0,
  opponentHealth = null,
  allowed = null,
  reach = DEFAULT_REACH,
  sinceAttack = 5.0,
  selfVelocityY = null,
  attackCooldownRemaining = null,
  // 注意：这三项**没有默认值**（= undefined），因为 `undefined`（没传，可回落到帧）
  // 与 `null`（明确「没有这个信息」）语义不同，默认成 null 会把回落路径关掉。
  prevAction,
  framesSinceAttack,
  framesSinceHit,
} = {}) {
  const selfPayload = (observation && observation.self) || {};
  const pos = selfPayload.pos || {};
  const sx = num(pos.x);
  const sy = num(pos.y);
  const sz = num(pos.z);
  const yaw = num(selfPayload.yaw);
  const pitch = num(selfPayload.pitch);
  const target = nearestPlayer(observation, username, allowed);
  const state = String((observation && observation.state) ?? 'Alive');

  const vector = new Float64Array(OBS_DIM);
  vector[0] = num(selfPayload.health, 20.0) / 20.0;          // spaces.py:137
  vector[1] = num(selfPayload.food, 20.0) / 20.0;            // spaces.py:138
  vector[2] = num(selfPayload.saturation) / 5.0;             // spaces.py:139
  vector[3] = selfPayload.onGround ? 1.0 : 0.0;              // spaces.py:140
  vector[4] = Math.sin(yaw);                                 // spaces.py:141
  vector[5] = Math.cos(yaw);                                 // spaces.py:142
  vector[6] = clamp(pitch / (Math.PI / 2.0), -1.0, 1.0);     // spaces.py:143
  vector[13] = 0.0;                                          // spaces.py:144
  vector[14] = clamp(num(stepFraction), 0.0, 1.0);           // spaces.py:145
  vector[15] = state === 'Alive' ? 1.0 : 0.0;                // spaces.py:146
  const held = selfPayload.heldItem || {};
  const heldDurability = (held.durability || {}).remaining;
  vector[16] = clamp(num(heldDurability, 1.0), 0.0, 1.0);    // spaces.py:149
  vector[17] = clamp(num(selfPayload.armorRemaining), 0.0, 1.0); // spaces.py:150
  const inventory = selfPayload.inventory || [];
  let gapples = 0;
  for (const item of inventory) {
    if (item && item.name === 'golden_apple') {
      gapples += num(item.count);
    }
  }
  vector[18] = clamp(gapples / 5.0, 0.0, 1.0);               // spaces.py:153
  vector[21] = clamp(num(sinceAttack, 5.0), 0.0, 5.0) / 5.0; // spaces.py:170

  // G-702（index 23/24）：跳劈时序 + 出手冷却。与 spaces.py 同序同尺度。
  const selfVelocity = selfPayload.velocity || {};
  const vy = selfVelocityY !== null && selfVelocityY !== undefined
    ? num(selfVelocityY)
    : num(selfVelocity.y);
  vector[23] = clamp(vy, -5.0, 5.0) / 5.0;                   // spaces.py:177
  const cooldown = attackCooldownRemaining === null || attackCooldownRemaining === undefined
    ? 0.0
    : num(attackCooldownRemaining);
  vector[24] = clamp(cooldown, 0.0, ATTACK_COOLDOWN_S) / ATTACK_COOLDOWN_S; // spaces.py:183

  // G-711（index 25..36）：跨帧信息。与 spaces.encode_observation 同序同尺度。
  //
  // 三种取值语义（**不要混**，混了就会出现「明明没有上一帧动作却编码成上一帧选了 0 档」）：
  // - `undefined`：调用方没提供 → 退回去读 `observation.crossFrame`；
  // - `null`：明确「没有这个信息」→ prevAction 全 0 / 帧计数取饱和 1.0（= 与 Python 的 None 同义）；
  // - 具体值：直接用。
  const crossFrame = (observation && observation.crossFrame) || {};
  const prev = prevAction !== undefined ? prevAction : crossFrame.prevAction;
  if (prev && typeof prev.length === 'number') {
    for (let offset = 0; offset < HEAD_SIZES_V2.length && offset < prev.length; offset += 1) {
      const size = HEAD_SIZES_V2[offset];
      const value = Math.trunc(num(prev[offset]));
      vector[PREV_ACTION_OFFSET + offset] = clamp(size > 0 ? value / size : 0.0, 0.0, 1.0);
    }
  }
  vector[FRAMES_SINCE_ATTACK_INDEX] =
    clamp(framesSince(framesSinceAttack, crossFrame.framesSinceAttack),
      0.0, CROSS_FRAME_NORMALIZER) / CROSS_FRAME_NORMALIZER;
  vector[FRAMES_SINCE_HIT_INDEX] =
    clamp(framesSince(framesSinceHit, crossFrame.framesSinceHit),
      0.0, CROSS_FRAME_NORMALIZER) / CROSS_FRAME_NORMALIZER;

  if (target !== null) {
    const tpos = target.pos || {};
    const dx = num(tpos.x) - sx;                             // spaces.py:160
    const dy = num(tpos.y) - sy;                             // spaces.py:161
    const dz = num(tpos.z) - sz;                             // spaces.py:162
    const horizontal = Math.sqrt(dx * dx + dz * dz);         // spaces.py:163
    const distance = num(target.distance, Math.sqrt(dx * dx + dy * dy + dz * dz)); // spaces.py:164
    vector[7] = dx / 16.0;                                   // spaces.py:165
    vector[8] = dy / 8.0;                                    // spaces.py:166
    vector[9] = dz / 16.0;                                   // spaces.py:167
    vector[10] = distance / 32.0;                            // spaces.py:168
    const health = opponentHealth !== null && opponentHealth !== undefined
      ? num(opponentHealth)
      : num(target.health, 20.0);                            // spaces.py:170
    vector[11] = clamp(health / 20.0, 0.0, 1.0);             // spaces.py:169-173
    vector[12] = target.onGround ? 1.0 : 0.0;                // spaces.py:174
    vector[13] = 1.0;                                        // spaces.py:175
    const velocity = target.velocity || {};
    const vx = num(velocity.x);
    const vy = num(velocity.y);
    const vz = num(velocity.z);
    vector[19] = clamp(Math.sqrt(vx * vx + vz * vz), 0.0, 10.0) / 10.0; // spaces.py:179
    vector[20] = clamp(vy, -5.0, 5.0) / 5.0;                 // spaces.py:180
    vector[22] = horizontal <= num(reach, DEFAULT_REACH) ? 1.0 : 0.0;  // spaces.py:183
  }

  for (let index = 0; index < OBS_DIM; index += 1) {
    vector[index] = clamp(vector[index], -OBS_CLIP, OBS_CLIP); // spaces.py:212
  }
  return vector;
}

/** yaw 指向目标（spaces.py:195 `yaw_to_target`，YAW_FORMULA="neg"）。 */
export function yawToTarget(dx, dz) {
  return Math.atan2(-dx, -dz);
}

/** pitch 指向目标（spaces.py:202 `pitch_to_target`，mineflayer 正值向下）。 */
export function pitchToTarget(dy, horizontal) {
  if (horizontal < 1e-6) {
    return 0.0;
  }
  return Math.atan2(-dy, horizontal);
}

/** 构造让 Bot 面朝目标的 `look` 动作（spaces.py:209 `look_at`）。 */
export function lookAt(target, selfPos) {
  const tpos = target.pos || {};
  const dx = num(tpos.x) - num(selfPos.x);
  const dy = num(tpos.y) + 1.4 - (num(selfPos.y) + 1.62);
  const dz = num(tpos.z) - num(selfPos.z);
  const horizontal = Math.sqrt(dx * dx + dz * dz);
  return {
    type: 'look',
    params: { yaw: yawToTarget(dx, dz), pitch: pitchToTarget(dy, horizontal) },
  };
}

const IDLE_ACTION = Object.freeze({ type: 'idle', params: { durationMs: MOVE_MS } });

/**
 * 离散动作 → 协议动作列表（spaces.py:226 `decode_action`）。
 *
 * @param {number} index 0..10
 * @param {object} observation 与编码用的是同一帧
 * @param {string} username 自己
 * @param {object} [options] `{allowed, reach}`
 * @returns {Array<{type:string, params:object}>}
 */
export function decodeAction(index, observation, username, { allowed = null, reach = DEFAULT_REACH } = {}) {
  const actionIndex = Number(index);
  if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= NUM_ACTIONS) {
    throw new Error(`动作编号越界: ${index}（合法范围 0..${NUM_ACTIONS - 1}）`);
  }
  const name = ACTION_NAMES[actionIndex];
  const selfPayload = (observation && observation.self) || {};
  const selfPos = selfPayload.pos || {};
  const target = nearestPlayer(observation, username, allowed);
  const distance = target ? num(target.distance, Number.POSITIVE_INFINITY) : Number.POSITIVE_INFINITY;

  if (name === 'idle') {
    return [{ type: 'idle', params: { durationMs: MOVE_MS } }];                    // spaces.py:246-247
  }
  if (name === 'jump') {
    return [{ type: 'jump', params: {} }];                                         // spaces.py:248-249
  }
  if (name === 'eat_gapple') {
    const health = num(selfPayload.health, 20.0);                                  // spaces.py:252
    if (health >= 19.5) {
      return [{ type: 'idle', params: { durationMs: MOVE_MS } }];
    }
    return [{ type: 'use', params: { item: 'gapple', holdMs: 1600 } }];            // spaces.py:255
  }
  if (name === 'block') {
    return [{ type: 'use', params: { item: 'sword', holdMs: 300 } }];              // spaces.py:257
  }
  if (name === 'switch_sword') {
    return [{ type: 'use', params: { item: 'sword', holdMs: 50 } }];               // spaces.py:259
  }
  if (target === null) {
    return [{ type: 'idle', params: { durationMs: MOVE_MS } }];                    // spaces.py:261-263
  }

  const look = lookAt(target, selfPos);                                            // spaces.py:265
  const effectiveReach = Math.max(0.0, num(reach, DEFAULT_REACH) - REACH_TOLERANCE);        // spaces.py:269
  if (name === 'approach') {
    const sprint = distance > 2.5;                                                 // spaces.py:272
    const duration = distance <= 4.0 ? MOVE_MS : MOVE_MS * 2;                      // spaces.py:273
    return [look, { type: 'move', params: { direction: 'forward', durationMs: duration, sprint } }];
  }
  if (name === 'retreat') {
    return [look, { type: 'move', params: { direction: 'back', durationMs: MOVE_MS } }];
  }
  if (name === 'strafe_left') {
    return [look, { type: 'move', params: { direction: 'left', durationMs: MOVE_MS } }];
  }
  if (name === 'strafe_right') {
    return [look, { type: 'move', params: { direction: 'right', durationMs: MOVE_MS } }];
  }
  if (name === 'use_rod') {
    if (distance > 6.0 || distance < 2.0) {                                        // spaces.py:284
      return [look, { type: 'move', params: { direction: 'forward', durationMs: MOVE_MS, sprint: true } }];
    }
    return [look, { type: 'use', params: { item: 'rod' } }];                       // spaces.py:287
  }
  if (name === 'attack') {
    const entityId = target.id;
    if (entityId === undefined || entityId === null || distance > effectiveReach) { // spaces.py:290
      return [look, {
        type: 'move',
        params: { direction: 'forward', durationMs: MOVE_MS, sprint: distance > 2.5 },
      }];
    }
    return [look, { type: 'attack', params: { targetEntityId: Number(entityId) } }, look]; // spaces.py:296
  }
  throw new Error(`未处理的动作: ${name}`);
}

export { IDLE_ACTION };
