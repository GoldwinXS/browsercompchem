/**
 * Pure geometry-measurement helpers for the click-to-measure tool.
 *
 * These operate on a flat [x0,y0,z0, x1,...] Angstrom buffer (the same layout
 * the rest of the demo uses) and take atom indices. They are deliberately
 * dependency-free so they can be unit-tested in Node without three.js or a DOM
 * (see measure.test.ts).
 */

/** Euclidean distance (Angstrom) between atoms i and j. */
export function distance(positions: ArrayLike<number>, i: number, j: number): number {
  const dx = positions[3 * i]! - positions[3 * j]!;
  const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
  const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Bond/vertex angle in DEGREES at the middle atom `j`, i.e. the angle
 * subtended by the vectors (i - j) and (k - j). `j` is the vertex.
 */
export function angle(
  positions: ArrayLike<number>,
  i: number,
  j: number,
  k: number,
): number {
  const ax = positions[3 * i]! - positions[3 * j]!;
  const ay = positions[3 * i + 1]! - positions[3 * j + 1]!;
  const az = positions[3 * i + 2]! - positions[3 * j + 2]!;
  const bx = positions[3 * k]! - positions[3 * j]!;
  const by = positions[3 * k + 1]! - positions[3 * j + 1]!;
  const bz = positions[3 * k + 2]! - positions[3 * j + 2]!;
  const dot = ax * bx + ay * by + az * bz;
  const na = Math.sqrt(ax * ax + ay * ay + az * az);
  const nb = Math.sqrt(bx * bx + by * by + bz * bz);
  if (na === 0 || nb === 0) return 0;
  // clamp for numerical safety before acos
  const c = Math.max(-1, Math.min(1, dot / (na * nb)));
  return (Math.acos(c) * 180) / Math.PI;
}
