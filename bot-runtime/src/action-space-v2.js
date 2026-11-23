/**
 * 动作空间 v2（十头 MultiDiscrete）的运行时侧定义。
 *
 * <p>顺序与档位必须与 `training/gladiator_env/spaces.py::ACTION_HEADS_V2` **逐位一致**：
 * `.npz` 里的 `W2_<head>` 是按这个名字挂上去的，顺序错了权重会「挂到错的头」而不会抛错，
 * 所以这里既给 `policy.js` 校验用，也给 `intent.js` 的下标→意图解码用。
 *
 * <p>只放**纯常量**、不 import 任何模块：`intent.js` 与 `policy.js` 都要用它，
 * 让它依赖别人会形成环。`move` / `use` / `sprintReset` 的取值表仍然只在
 * `intent.js` 里（那是「怎么执行」的知识），这里只描述「每头几档 + 顺序」。
 */

/** 十个头的档位数（键顺序 = 头顺序）。 */
export const ACTION_HEADS_V2 = Object.freeze({
  move: 5,
  look_yaw: 7,
  look_pitch: 5,
  attack: 2,
  use: 3,
  jump: 2,
  sneak: 2,
  sprint: 2,
  slot: 9,
  sprint_reset: 3,
});

/** 头名（下标即动作向量的下标）。 */
export const HEAD_ORDER_V2 = Object.freeze(Object.keys(ACTION_HEADS_V2));

/** 每头档位数（与 `HEAD_ORDER_V2` 同序）。 */
export const HEAD_SIZES_V2 = Object.freeze(HEAD_ORDER_V2.map((head) => ACTION_HEADS_V2[head]));

/** 十头展开后的总维数（`_meta.n_actions`，当前 40）。 */
export const ACTION_DIM_V2 = HEAD_SIZES_V2.reduce((sum, size) => sum + size, 0);

/** `look_yaw` 头：每 tick 的偏航增量（度），与 spaces.py 同序。 */
export const LOOK_YAW_VALUES = Object.freeze([-30, -10, -3, 0, 3, 10, 30]);

/** `look_pitch` 头：每 tick 的俯仰增量（度），mineflayer 正值向下。 */
export const LOOK_PITCH_VALUES = Object.freeze([-15, -5, 0, 5, 15]);

/** 热键栏格数（`slot` 头的档位数）。 */
export const SLOT_COUNT = 9;
