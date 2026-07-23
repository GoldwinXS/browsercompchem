import type { Molecule } from "../geometry/molecule.js";
import { FireOptimizer, type FireOptions } from "../optimize/fire.js";
import { ClassicalForceField, type FFBond, type ForceFieldOptions } from "./forceField.js";
import { countGeometryViolations, type ValidityOptions } from "./validity.js";

export type { FFBond } from "./forceField.js";

/**
 * Topology-aware multi-start 3D embedder ("mini-ETKDG"): relax the classical
 * ClassicalForceField from several random 3D starting placements with the
 * SAME FireOptimizer used for the ANI-2x polish, score each result by FF
 * energy + the geometry-validity gate, and return the best VALID conformer.
 * This replaces "RDKit 2D coords + small jitter" as the seed for ALL molecule
 * sizes -- the random-start + classical-relax approach costs nothing (no
 * neural network evaluation) and, unlike a 2D-derived seed, starts from a
 * geometry the classical FF has already pulled apart into a sane 3D shape
 * before ANI-2x ever sees it.
 *
 * Deterministic: uses a seeded PRNG (mulberry32, matching apps/demo/src/main.ts's
 * implementation bit-for-bit) rather than Math.random, so the same
 * (atoms, bonds, opts) always produces the same result -- required for the
 * embed3d.test.ts gates and handy for reproducing a bad case.
 */

/** mulberry32 -- identical algorithm to apps/demo/src/main.ts's seeded RNG (kept as an
 * independent copy so the engine package has no dependency on the demo app). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Per-atom radius (Angstrom) the random-start box/ball scales with, before the cbrt(N) growth. */
const BASE_RADIUS_PER_ATOM = 1.35;
/** Floor so a 1-2 atom "molecule" still gets a non-degenerate box. */
const MIN_RADIUS = 2.0;

/** Scatter n atoms uniformly at random inside a ball whose radius grows with cbrt(n) (~constant atomic density in 3D). */
function randomStart(n: number, rng: () => number): Float64Array {
  const radius = Math.max(MIN_RADIUS, BASE_RADIUS_PER_ATOM * Math.cbrt(Math.max(n, 1)));
  const pos = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    // Rejection-sample a point in the ball from a bounding cube; a handful of
    // tries is enough (~52% acceptance) and a cube fallback on the rare miss
    // is harmless -- this only needs to be a reasonable, not exact, scatter.
    let x = 0;
    let y = 0;
    let z = 0;
    for (let tries = 0; tries < 20; tries++) {
      x = (rng() * 2 - 1) * radius;
      y = (rng() * 2 - 1) * radius;
      z = (rng() * 2 - 1) * radius;
      if (x * x + y * y + z * z <= radius * radius) break;
    }
    pos[3 * i] = x;
    pos[3 * i + 1] = y;
    pos[3 * i + 2] = z;
  }
  return pos;
}

export interface Embed3dOptions {
  /** Number of random restarts to try (default 6). */
  attempts?: number;
  /** PRNG seed (default a fixed constant -- override for a different but still reproducible draw). */
  seed?: number;
  /** Override the classical FF's force constants. */
  ff?: ForceFieldOptions;
  /** Override the FIRE optimizer's convergence knobs for the classical relax. */
  fire?: FireOptions;
  /** Override the validity gate's thresholds (defaults match the demo worker's gate). */
  validity?: ValidityOptions;
}

export interface Embed3dAttempt {
  energy: number;
  violations: number;
}

export interface Embed3dResult {
  /** Best conformer's flat Angstrom coordinates. */
  positions: Float64Array;
  /** Its classical FF energy (arbitrary units -- see ClassicalForceField). */
  energy: number;
  /** True if the winner passed the validity gate (0 violations); false means every attempt was flagged. */
  valid: boolean;
  /** Violation count for the returned winner. */
  violations: number;
  /** Per-attempt (energy, violations), in attempt order -- useful for diagnostics/tests. */
  attempts: Embed3dAttempt[];
}

const DEFAULT_ATTEMPTS = 6;
/** Arbitrary fixed default seed (digits of an early mulberry32 constant) -- any caller wanting
 * a different draw passes its own `seed`; the default just needs to be fixed for determinism. */
const DEFAULT_SEED = 0x2836a2ec;

const DEFAULT_FIRE: FireOptions = {
  forceTolerance: 5e-2,
  maxSteps: 500,
  dtStart: 0.05,
  dtMax: 0.25,
  maxStep: 0.15,
};

/**
 * Embed `atoms` (element symbols) into a 3D geometry from `bonds` (known
 * connectivity) alone, via multi-start classical-FF relaxation. No ANI-2x, no
 * RDKit -- pure topology in, untangled 3D coordinates out.
 */
export async function embed3d(
  atoms: string[],
  bonds: FFBond[],
  opts: Embed3dOptions = {},
): Promise<Embed3dResult> {
  const n = atoms.length;
  const attempts = Math.max(1, opts.attempts ?? DEFAULT_ATTEMPTS);
  const rng = mulberry32(opts.seed ?? DEFAULT_SEED);
  const ff = new ClassicalForceField(atoms, bonds, opts.ff);
  const fire = new FireOptimizer({ ...DEFAULT_FIRE, ...opts.fire });

  const log: Embed3dAttempt[] = [];
  let best: { positions: Float64Array; energy: number; violations: number } | undefined;
  let bestValid: { positions: Float64Array; energy: number; violations: number } | undefined;

  for (let a = 0; a < attempts; a++) {
    const start = randomStart(n, rng);
    const mol: Molecule = { symbols: atoms, positions: start, charge: 0, multiplicity: 1 };
    const result = await fire.optimize(mol, ff);
    const violations = countGeometryViolations(atoms, result.molecule.positions, bonds, opts.validity);
    const cand = { positions: result.molecule.positions, energy: result.energy, violations };
    log.push({ energy: cand.energy, violations: cand.violations });

    if (violations === 0 && (!bestValid || cand.energy < bestValid.energy)) bestValid = cand;
    const better = (x: typeof cand, y: typeof cand): boolean =>
      x.violations !== y.violations ? x.violations < y.violations : x.energy < y.energy;
    if (!best || better(cand, best)) best = cand;
  }

  const winner = bestValid ?? best;
  if (!winner) throw new Error("embed3d: no attempts produced a result");
  return {
    positions: winner.positions,
    energy: winner.energy,
    valid: !!bestValid,
    violations: winner.violations,
    attempts: log,
  };
}
