/**
 * 规则缓存（G-206）：把 NOTICE(RESTRICTED) 与 QUERY_RESULT 的结论落成带 TTL 的区域矩形。
 *
 * <p>判定时"离开即失效"：查询点不在任何已缓存矩形内时返回 allowed，
 * 避免把 A 区域的 deny 误用到 B 区域。过期条目在 sweep()/查询时清理。
 *
 * <p>本模块不依赖 mineflayer，可独立单测。
 */

function normalizeBox(box) {
  if (!box) {
    return null;
  }
  const minX = Math.min(box.minX, box.maxX);
  const minY = Math.min(box.minY, box.maxY);
  const minZ = Math.min(box.minZ, box.maxZ);
  const maxX = Math.max(box.minX, box.maxX);
  const maxY = Math.max(box.minY, box.maxY);
  const maxZ = Math.max(box.minZ, box.maxZ);
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

export class RestrictionCache {
  constructor({ defaultTtlMs = 15000, now = () => Date.now(), maxEntries = 256 } = {}) {
    this.defaultTtlMs = defaultTtlMs;
    this.now = now;
    this.maxEntries = maxEntries;
    this.entries = new Map();
  }

  /** 写入/覆盖一条限制；regionId 为空时用矩形作 key。 */
  put({ regionId, box, allowPvp = false, expiresAt, action = 'pvp', reason = 'restricted' }) {
    const normalized = normalizeBox(box);
    const key = regionId || (normalized
      ? `${normalized.minX},${normalized.minY},${normalized.minZ}`
      : null);
    if (!key) {
      return null;
    }
    const entry = {
      regionId: regionId || key,
      box: normalized,
      allowPvp: Boolean(allowPvp),
      action,
      reason,
      expiresAt: Number.isFinite(expiresAt) ? expiresAt : this.now() + this.defaultTtlMs,
    };
    this.entries.set(key, entry);
    if (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      this.entries.delete(oldest);
    }
    return entry;
  }

  /** 用 OBSERVATION/QUERY_RESULT 的完整限制列表重建缓存（服务端是唯一真源）。 */
  replaceAll(list = []) {
    this.entries.clear();
    for (const item of list) {
      if (item && item.allowPvp === false) {
        this.put(item);
      }
    }
  }

  /** 删除已过期条目，返回删除数量。 */
  sweep() {
    const t = this.now();
    let removed = 0;
    for (const [key, entry] of this.entries) {
      if (entry.expiresAt <= t) {
        this.entries.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  /** 查询坐标是否被缓存中的 deny 区域覆盖。 */
  check(x, y, z) {
    this.sweep();
    for (const entry of this.entries.values()) {
      if (entry.allowPvp) {
        continue;
      }
      if (!entry.box) {
        continue;
      }
      const b = entry.box;
      if (x >= b.minX && x <= b.maxX && y >= b.minY && y <= b.maxY && z >= b.minZ && z <= b.maxZ) {
        return { denied: true, entry };
      }
    }
    return { denied: false, entry: null };
  }

  /** 供 OBSERVATION.restrictions 使用的快照。 */
  snapshot() {
    this.sweep();
    return [...this.entries.values()].map((entry) => ({
      regionId: entry.regionId,
      box: entry.box,
      allowPvp: entry.allowPvp,
      expiresAt: entry.expiresAt,
    }));
  }

  get size() {
    return this.entries.size;
  }

  clear() {
    this.entries.clear();
  }
}
