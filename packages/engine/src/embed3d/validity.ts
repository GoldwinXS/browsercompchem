import { COVALENT_RADII, FALLBACK_RADIUS } from "./covalentRadii.js";

/**
 * Geometry-validity gate shared by the classical embedder (embed3d.ts) and the
 * demo worker's multi-seed ANI-2x search (apps/demo/src/optimizer.worker.ts).
 * Originally implemented only in the worker (as `geometryViolations`); pulled
 * into the engine so both call sites use IDENTICAL logic and constants -- a
 * conformer that is "valid" by the classical seed's gate must mean the same
 * thing as "valid" by the ANI-2x polish's gate.
 *
 * Counts, for a given geometry against the KNOWN bond graph:
 *   - every bonded (1-2) pair stretched further than its covalent-radii sum
 *     plus a slack allowance (a real bond that "broke" during relaxation);
 *   - every NON-bonded pair (any pair not itself a bond) sitting closer than a
 *     flat clash distance (two atoms occupying the same space -- a fold-
 *     through-itself or a chain threaded through a ring).
 * Returns the violation count (0 = geometrically sane).
 */
export interface ValidityOptions {
  /** Non-bonded pairs closer than this (Angstrom) count as a clash. */
  clashDistance?: number;
  /** A bonded pair may stretch this far past its covalent-radii sum before it counts as "broken". */
  bondStretchSlack?: number;
  /** Override the covalent-radii table (defaults to Cordero et al. 2008, see covalentRadii.ts). */
  covalentRadii?: Record<string, number>;
}

export const DEFAULT_CLASH_DISTANCE = 1.15;
export const DEFAULT_BOND_STRETCH_SLACK = 0.7;

export function countGeometryViolations(
  symbols: string[],
  positions: Float64Array,
  bonds: { i: number; j: number }[],
  opts: ValidityOptions = {},
): number {
  const clash = opts.clashDistance ?? DEFAULT_CLASH_DISTANCE;
  const slack = opts.bondStretchSlack ?? DEFAULT_BOND_STRETCH_SLACK;
  const radii = opts.covalentRadii ?? COVALENT_RADII;
  const radiusOf = (sym: string): number => radii[sym] ?? FALLBACK_RADIUS;
  const n = symbols.length;

  const dist = (i: number, j: number): number =>
    Math.hypot(
      positions[3 * i]! - positions[3 * j]!,
      positions[3 * i + 1]! - positions[3 * j + 1]!,
      positions[3 * i + 2]! - positions[3 * j + 2]!,
    );

  let violations = 0;
  const bonded = new Set<number>();
  for (const b of bonds) {
    bonded.add(b.i * n + b.j);
    bonded.add(b.j * n + b.i);
    const cutoff = radiusOf(symbols[b.i]!) + radiusOf(symbols[b.j]!) + slack;
    if (dist(b.i, b.j) > cutoff) violations++;
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (bonded.has(i * n + j)) continue;
      if (dist(i, j) < clash) violations++;
    }
  }
  return violations;
}

/**
 * Count of non-bonded pairs closer than the sum of their covalent radii -- the
 * "tangle metric" used to compare a topology-aware embed3d start against a
 * naive flattened/jittered 2D start (see embed3d.test.ts). Unlike
 * countGeometryViolations' flat clash distance (tuned to gate ANI-2x
 * relaxations), this scales per element pair, so it is a fair size-independent
 * measure of "how folded through itself is this geometry" for molecules
 * spanning very different element compositions.
 */
export function tangleMetric(
  symbols: string[],
  positions: ArrayLike<number>,
  bonds: { i: number; j: number }[],
  covalentRadii: Record<string, number> = COVALENT_RADII,
): number {
  const radiusOf = (sym: string): number => covalentRadii[sym] ?? FALLBACK_RADIUS;
  const n = symbols.length;
  const dist = (i: number, j: number): number =>
    Math.hypot(
      positions[3 * i]! - positions[3 * j]!,
      positions[3 * i + 1]! - positions[3 * j + 1]!,
      positions[3 * i + 2]! - positions[3 * j + 2]!,
    );
  const bonded = new Set<number>();
  for (const b of bonds) {
    bonded.add(b.i * n + b.j);
    bonded.add(b.j * n + b.i);
  }
  let tangled = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (bonded.has(i * n + j)) continue;
      if (dist(i, j) < radiusOf(symbols[i]!) + radiusOf(symbols[j]!)) tangled++;
    }
  }
  return tangled;
}
