/**
 * embed3d correctness gates.
 *
 * embed3d() is a topology-aware classical force field (harmonic bonds +
 * harmonic angles + soft non-bonded repulsion) relaxed with FIRE from several
 * random 3D starts (seeded, deterministic), replacing "RDKit 2D coords + tiny
 * jitter" as the seed handed to the ANI-2x polish. These tests encode the
 * geometric correctness a good embedder must have: benzene comes out planar
 * with 120 degree ring angles, acetylene/CO2 come out linear, small molecules
 * still reach their correct final geometry once ANI-2x polishes the classical
 * seed, and -- the key large-molecule gate -- a real large molecule (a
 * distance-perceived cholesterol topology, plus a long flexible alkane chain)
 * embeds with NO severe non-bonded clashes, where a naive flattened/jittered
 * 2D-style start is measurably more tangled.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { embed3d, type FFBond } from "../src/embed3d/embed3d.js";
import { ClassicalForceField } from "../src/embed3d/forceField.js";
import { countGeometryViolations, tangleMetric } from "../src/embed3d/validity.js";
import { COVALENT_RADII, FALLBACK_RADIUS } from "../src/embed3d/covalentRadii.js";
import { jacobiEigen } from "../src/vibrations/jacobi.js";
import { Ani2xProvider } from "../src/potentials/ani2x/provider.js";
import { FireOptimizer } from "../src/optimize/fire.js";
import type { Molecule } from "../src/geometry/molecule.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const modelDir = here + "../../../models/ani2x";

// ---------------------------------------------------------------------------
// Small geometry helpers shared by the tests below.
// ---------------------------------------------------------------------------
function angleDeg(pos: ArrayLike<number>, a: number, center: number, b: number): number {
  const ax = pos[3 * a]! - pos[3 * center]!;
  const ay = pos[3 * a + 1]! - pos[3 * center + 1]!;
  const az = pos[3 * a + 2]! - pos[3 * center + 2]!;
  const bx = pos[3 * b]! - pos[3 * center]!;
  const by = pos[3 * b + 1]! - pos[3 * center + 1]!;
  const bz = pos[3 * b + 2]! - pos[3 * center + 2]!;
  const lenA = Math.hypot(ax, ay, az);
  const lenB = Math.hypot(bx, by, bz);
  const cosT = (ax * bx + ay * by + az * bz) / (lenA * lenB);
  return (Math.acos(Math.min(1, Math.max(-1, cosT))) * 180) / Math.PI;
}

function dist(pos: ArrayLike<number>, i: number, j: number): number {
  return Math.hypot(pos[3 * i]! - pos[3 * j]!, pos[3 * i + 1]! - pos[3 * j + 1]!, pos[3 * i + 2]! - pos[3 * j + 2]!);
}

/** RMS perpendicular deviation of a set of atoms from their own best-fit plane
 * (via the smallest-eigenvalue eigenvector of the position covariance, reusing
 * the engine's own Jacobi eigensolver -- no new linear-algebra dependency). */
function planarityRms(pos: ArrayLike<number>, atomIndices: number[]): number {
  const m = atomIndices.length;
  let cx = 0,
    cy = 0,
    cz = 0;
  for (const idx of atomIndices) {
    cx += pos[3 * idx]!;
    cy += pos[3 * idx + 1]!;
    cz += pos[3 * idx + 2]!;
  }
  cx /= m;
  cy /= m;
  cz /= m;
  const cov = new Float64Array(9);
  for (const idx of atomIndices) {
    const dx = pos[3 * idx]! - cx;
    const dy = pos[3 * idx + 1]! - cy;
    const dz = pos[3 * idx + 2]! - cz;
    cov[0] += dx * dx;
    cov[1] += dx * dy;
    cov[2] += dx * dz;
    cov[4] += dy * dy;
    cov[5] += dy * dz;
    cov[8] += dz * dz;
  }
  cov[3] = cov[1]!;
  cov[6] = cov[2]!;
  cov[7] = cov[5]!;
  const { vectors } = jacobiEigen(cov, 3);
  const normal = vectors[0]!; // smallest-eigenvalue eigenvector = best-fit plane normal
  let ss = 0;
  for (const idx of atomIndices) {
    const dx = pos[3 * idx]! - cx;
    const dy = pos[3 * idx + 1]! - cy;
    const dz = pos[3 * idx + 2]! - cz;
    const d = dx * normal[0]! + dy * normal[1]! + dz * normal[2]!;
    ss += d * d;
  }
  return Math.sqrt(ss / m);
}

// ---------------------------------------------------------------------------
// Benzene: aromatic ring, required to come out planar with 120 degree angles.
// ---------------------------------------------------------------------------
function benzeneTopology(): { symbols: string[]; bonds: FFBond[] } {
  const symbols = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"];
  const bonds: FFBond[] = [];
  for (let k = 0; k < 6; k++) bonds.push({ i: k, j: (k + 1) % 6, order: 4 }); // aromatic ring
  for (let k = 0; k < 6; k++) bonds.push({ i: k, j: k + 6, order: 1 }); // ring C-H
  return { symbols, bonds };
}

describe("embed3d: benzene (aromatic ring)", () => {
  it("comes out planar, 120 degree ring angles, C-C bonds within a few % of 1.40 A", async () => {
    const { symbols, bonds } = benzeneTopology();
    const res = await embed3d(symbols, bonds, { seed: 42 });

    expect(res.valid).toBe(true);
    expect(res.violations).toBe(0);

    for (let k = 0; k < 6; k++) {
      const d = dist(res.positions, k, (k + 1) % 6);
      expect(Math.abs(d - 1.40) / 1.40).toBeLessThan(0.03); // within 3%
    }
    for (let k = 0; k < 6; k++) {
      const a = angleDeg(res.positions, (k + 5) % 6, k, (k + 1) % 6);
      expect(Math.abs(a - 120)).toBeLessThan(1.0);
    }

    const ringAtoms = [0, 1, 2, 3, 4, 5];
    const rms = planarityRms(res.positions, ringAtoms);
    // The classical relax's own FIRE tolerance is loose (5e-2, tuned for speed
    // across 6 restarts, not final precision -- the ANI-2x polish tightens
    // further), so a few percent of the ring diameter is expected here.
    expect(rms).toBeLessThan(0.05); // Angstrom -- the ring is essentially flat
  }, 30000);

  it("is deterministic: same seed -> bit-identical result", async () => {
    const { symbols, bonds } = benzeneTopology();
    const a = await embed3d(symbols, bonds, { seed: 42 });
    const b = await embed3d(symbols, bonds, { seed: 42 });
    expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
    expect(a.energy).toBe(b.energy);
  }, 30000);
});

// ---------------------------------------------------------------------------
// Linear topologies: acetylene (a true triple bond) and CO2 (cumulated
// doubles) both hit the "2 neighbors, sp" branch of the angle heuristic.
// ---------------------------------------------------------------------------
describe("embed3d: linear molecules", () => {
  it("acetylene (H-C#C-H) embeds linear (~180 degrees)", async () => {
    const symbols = ["H", "C", "C", "H"];
    const bonds: FFBond[] = [
      { i: 0, j: 1, order: 1 },
      { i: 1, j: 2, order: 3 },
      { i: 2, j: 3, order: 1 },
    ];
    const res = await embed3d(symbols, bonds, { seed: 7 });
    expect(res.violations).toBe(0);
    expect(angleDeg(res.positions, 0, 1, 2)).toBeGreaterThan(178);
    expect(angleDeg(res.positions, 1, 2, 3)).toBeGreaterThan(178);
  }, 30000);

  it("CO2 (O=C=O, cumulated doubles) embeds linear (~180 degrees)", async () => {
    const symbols = ["O", "C", "O"];
    const bonds: FFBond[] = [
      { i: 0, j: 1, order: 2 },
      { i: 1, j: 2, order: 2 },
    ];
    const res = await embed3d(symbols, bonds, { seed: 3 });
    expect(res.violations).toBe(0);
    expect(angleDeg(res.positions, 0, 1, 2)).toBeGreaterThan(178);
  }, 30000);
});

// ---------------------------------------------------------------------------
// Small molecules: the classical seed only needs to be "reasonable" -- the
// real geometric target (104.5 / 109.5 degrees) is reached after the ANI-2x
// FIRE polish, exactly as it will be in the real pipeline (embed3d seeds,
// ANI-2x relaxes further).
// ---------------------------------------------------------------------------
describe("embed3d + ANI-2x polish: small molecules reach their correct geometry", () => {
  let provider: Ani2xProvider;
  beforeAll(async () => {
    provider = await Ani2xProvider.create({ modelDir, variant: "single-f32" });
  });

  async function polish(symbols: string[], positions: Float64Array): Promise<Molecule> {
    const fire = new FireOptimizer({ forceTolerance: 1e-4, maxSteps: 2000, dtMax: 0.2, maxStep: 0.1 });
    const r = await fire.optimize({ symbols, positions, charge: 0, multiplicity: 1 }, provider);
    expect(r.converged).toBe(true);
    return r.molecule;
  }

  it("water: classical seed is reasonable, ANI-2x polish reaches ~104.5 degrees", async () => {
    const symbols = ["O", "H", "H"];
    const bonds: FFBond[] = [
      { i: 0, j: 1, order: 1 },
      { i: 0, j: 2, order: 1 },
    ];
    const seed = await embed3d(symbols, bonds, { seed: 1 });
    expect(seed.violations).toBe(0);
    const relaxed = await polish(symbols, seed.positions);
    const finalAngle = angleDeg(relaxed.positions, 1, 0, 2);
    expect(Math.abs(finalAngle - 104.5)).toBeLessThan(3);
  }, 30000);

  it("methane: classical seed is reasonable, ANI-2x polish reaches tetrahedral ~109.5 degrees", async () => {
    const symbols = ["C", "H", "H", "H", "H"];
    const bonds: FFBond[] = [
      { i: 0, j: 1, order: 1 },
      { i: 0, j: 2, order: 1 },
      { i: 0, j: 3, order: 1 },
      { i: 0, j: 4, order: 1 },
    ];
    const seed = await embed3d(symbols, bonds, { seed: 2 });
    expect(seed.violations).toBe(0);
    const relaxed = await polish(symbols, seed.positions);
    for (let i = 1; i <= 4; i++) {
      for (let j = i + 1; j <= 4; j++) {
        expect(Math.abs(angleDeg(relaxed.positions, i, 0, j) - 109.47)).toBeLessThan(3);
      }
    }
  }, 30000);
});

// ---------------------------------------------------------------------------
// Large molecules: the actual bug being fixed. A raw 2D+jitter start folds a
// large molecule through itself; embed3d must not.
// ---------------------------------------------------------------------------

/** Build a straight-chain alkane C(n)H(2n+2) topology (a maximally floppy large molecule -- no ring to constrain it). */
function alkaneTopology(nCarbons: number): { symbols: string[]; bonds: FFBond[] } {
  const symbols: string[] = [];
  const bonds: FFBond[] = [];
  for (let k = 0; k < nCarbons; k++) symbols.push("C");
  for (let k = 0; k < nCarbons - 1; k++) bonds.push({ i: k, j: k + 1, order: 1 });
  let h = nCarbons;
  for (let k = 0; k < nCarbons; k++) {
    const nH = k === 0 || k === nCarbons - 1 ? 3 : 2;
    for (let x = 0; x < nH; x++) {
      symbols.push("H");
      bonds.push({ i: k, j: h, order: 1 });
      h++;
    }
  }
  return { symbols, bonds };
}

/** Mimic of the OLD (fixed) seeding strategy for a topology with no real 2D
 * layout available in an engine test: lay the chain out along x (a 2D-style
 * "flat" depiction) and apply the same style of small deterministic jitter
 * rdkit.ts's breakPlanarity() uses. This is NOT a claim that RDKit's actual
 * depiction algorithm looks like this -- it is a stand-in that reproduces the
 * one property that matters here: a flattened, non-3D-aware starting layout,
 * which is exactly what makes a floppy/large molecule fold through itself. */
function naiveFlatSeed(symbols: string[], nCarbons: number): Float64Array {
  const pos = new Float64Array(symbols.length * 3);
  for (let k = 0; k < nCarbons; k++) {
    pos[3 * k] = k * 1.5;
    pos[3 * k + 1] = 0;
    pos[3 * k + 2] = 0;
  }
  let h = nCarbons;
  for (let k = 0; k < nCarbons; k++) {
    const nH = k === 0 || k === nCarbons - 1 ? 3 : 2;
    for (let x = 0; x < nH; x++) {
      pos[3 * h] = k * 1.5 + 0.5;
      pos[3 * h + 1] = (x % 2 === 0 ? 1 : -1) * 0.9;
      pos[3 * h + 2] = 0;
      h++;
    }
  }
  let a = 0xc0ffee >>> 0;
  const rng = (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let k = 0; k < pos.length; k++) pos[k] = pos[k]! + (rng() * 2 - 1) * 0.6;
  return pos;
}

describe("embed3d: large molecule -- a long flexible alkane chain (C30H62, 92 atoms)", () => {
  it("embeds valid (0 gate violations) and untangled where a naive flat+jitter start is not", async () => {
    const nCarbons = 30;
    const { symbols, bonds } = alkaneTopology(nCarbons);
    expect(symbols.length).toBe(92);

    const res = await embed3d(symbols, bonds, { seed: 99 });
    expect(res.valid).toBe(true);
    expect(res.violations).toBe(0);
    expect(countGeometryViolations(symbols, res.positions, bonds)).toBe(0);

    // Every bond lands near its ideal length after the classical relax.
    const ff = new ClassicalForceField(symbols, bonds);
    for (const b of bonds) {
      const r0 = COVALENT_RADII[symbols[b.i]!]! + COVALENT_RADII[symbols[b.j]!]!;
      expect(Math.abs(dist(res.positions, b.i, b.j) - r0)).toBeLessThan(0.15);
    }
    void ff; // constructed only to assert it builds without throwing for this topology

    // KEY assertion: embed3d's tangle metric (non-bonded pairs closer than the
    // sum of covalent radii) beats a naive flattened/jittered start.
    const embedTangle = tangleMetric(symbols, res.positions, bonds);
    const naiveTangle = tangleMetric(symbols, naiveFlatSeed(symbols, nCarbons), bonds);
    expect(embedTangle).toBe(0);
    expect(embedTangle).toBeLessThan(naiveTangle);
  }, 60000);

  it("is deterministic: same seed -> bit-identical result", async () => {
    const { symbols, bonds } = alkaneTopology(12);
    const a = await embed3d(symbols, bonds, { seed: 555 });
    const b = await embed3d(symbols, bonds, { seed: 555 });
    expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
  }, 30000);
});

// ---------------------------------------------------------------------------
// Large molecule #2: a real molecule (cholesterol, 74 atoms), topology
// recovered from the ANI-2x reference fixture's equilibrium coordinates by
// distance perception (test-only -- production code always gets real bonds
// from RDKit; this is just how to get a real large-molecule TOPOLOGY into an
// engine-only test with no RDKit dependency). All perceived bonds are tagged
// "single" (order 1): distance alone cannot recover bond order, and the
// specific order only matters to the 3-neighbor sp2 rule below, which already
// treats sp2 centers correctly regardless of exactly which incident bond is
// the multiple one.
// ---------------------------------------------------------------------------
function perceiveBondsFromReference(symbols: string[], coords: number[][]): FFBond[] {
  const bonds: FFBond[] = [];
  const radius = (s: string): number => COVALENT_RADII[s] ?? FALLBACK_RADIUS;
  for (let i = 0; i < symbols.length; i++) {
    for (let j = i + 1; j < symbols.length; j++) {
      const dx = coords[i]![0]! - coords[j]![0]!;
      const dy = coords[i]![1]! - coords[j]![1]!;
      const dz = coords[i]![2]! - coords[j]![2]!;
      const d = Math.hypot(dx, dy, dz);
      const cutoff = (radius(symbols[i]!) + radius(symbols[j]!)) * 1.25;
      if (d < cutoff) bonds.push({ i, j, order: 1 });
    }
  }
  return bonds;
}

describe("embed3d: large molecule -- cholesterol (74 atoms, real polycyclic topology)", () => {
  const refs = JSON.parse(
    readFileSync(here + "fixtures/ani2x-references.json", "utf8"),
  ) as Record<string, { symbols: string[]; coords: number[][] }>;
  const symbols = refs.cholesterol!.symbols;
  const coords = refs.cholesterol!.coords;
  const bonds = perceiveBondsFromReference(symbols, coords);

  it("perceived topology is plausible (roughly one bond per non-H heavy-atom valence)", () => {
    // Cholesterol C27H46O: 74 atoms, a real molecular graph has on the order of
    // 76-80 bonds (rings + chain + the one C=C + the OH) -- just a sanity check
    // that distance perception did not badly over/under-connect the structure.
    expect(bonds.length).toBeGreaterThan(70);
    expect(bonds.length).toBeLessThan(85);
  });

  it("embeds valid (0 gate violations), untangled vs. a naive flattened+jittered start", async () => {
    const res = await embed3d(symbols, bonds, { seed: 555 });
    expect(res.valid).toBe(true);
    expect(res.violations).toBe(0);

    // Every perceived bond lands near its ideal (covalent-radii-sum) length.
    for (const b of bonds) {
      const r0 = COVALENT_RADII[symbols[b.i]!]! + COVALENT_RADII[symbols[b.j]!]!;
      expect(Math.abs(dist(res.positions, b.i, b.j) - r0)).toBeLessThan(0.2);
    }

    // Naive baseline: flatten the (real, non-planar) reference structure onto
    // a plane and jitter it, the same way rdkit.ts's breakPlanarity() turns a
    // 2D RDKit depiction into a seed -- z=0 plus small per-axis noise. Collapsing
    // a real 3D steroid skeleton onto a plane is exactly what invents the
    // pathological close contacts a genuine 2D depiction's stretched/overlapping
    // layout also produces for a fused-ring system.
    let a = 777 >>> 0;
    const rng = (): number => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const naive = new Float64Array(symbols.length * 3);
    for (let i = 0; i < symbols.length; i++) {
      naive[3 * i] = coords[i]![0]! + (rng() * 2 - 1) * 0.05;
      naive[3 * i + 1] = coords[i]![1]! + (rng() * 2 - 1) * 0.05;
      naive[3 * i + 2] = 0 + (rng() * 2 - 1) * 0.6; // flattened + z jitter
    }

    const embedTangle = tangleMetric(symbols, res.positions, bonds);
    const naiveTangle = tangleMetric(symbols, naive, bonds);
    expect(embedTangle).toBe(0);
    expect(embedTangle).toBeLessThan(naiveTangle);
    // Not just "better" -- the real bug is qualitative: the naive start is
    // severely tangled (dozens of overlapping atom pairs from the fused ring
    // system collapsing onto a plane), embed3d's is not.
    expect(naiveTangle).toBeGreaterThan(10);
  }, 60000);
});

// ---------------------------------------------------------------------------
// The classical force field's own analytic gradient, checked directly against
// central finite differences (independent of FIRE/embed3d -- if this is wrong,
// nothing above can be trusted).
// ---------------------------------------------------------------------------
describe("ClassicalForceField: analytic gradient matches finite difference", () => {
  it("propane topology (bonds + angles + 1-4 non-bonded repulsion all active)", () => {
    const symbols = ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"];
    const bonds: FFBond[] = [
      { i: 0, j: 1, order: 1 },
      { i: 1, j: 2, order: 1 },
      { i: 0, j: 3, order: 1 },
      { i: 0, j: 4, order: 1 },
      { i: 0, j: 5, order: 1 },
      { i: 1, j: 6, order: 1 },
      { i: 1, j: 7, order: 1 },
      { i: 2, j: 8, order: 1 },
      { i: 2, j: 9, order: 1 },
      { i: 2, j: 10, order: 1 },
    ];
    const ff = new ClassicalForceField(symbols, bonds);
    // A generic, non-degenerate geometry (no exactly-linear/zero angles or
    // exactly-coincident atoms, where the angle-gradient formula is singular).
    const positions = Float64Array.from([
      0, 0, 0, 1.5, 0.2, 0, 2.9, -0.3, 0.4, -0.5, 0.9, 0.3, -0.6, -0.8, -0.4, 0.2, 0.3, -1.1, 1.6, 1.2, 0.6, 1.4,
      -1.0, 0.7, 3.5, 0.6, 1.1, 3.3, -1.2, 0.2, 2.7, -0.5, 1.4,
    ]);
    const { forces } = ff.energyForcesSync(positions);
    const h = 1e-6;
    let maxErr = 0;
    for (let k = 0; k < positions.length; k++) {
      const plus = Float64Array.from(positions);
      plus[k] = plus[k]! + h;
      const minus = Float64Array.from(positions);
      minus[k] = minus[k]! - h;
      const fd = (ff.energyForcesSync(plus).energy - ff.energyForcesSync(minus).energy) / (2 * h);
      const analyticGrad = -forces[k]!;
      maxErr = Math.max(maxErr, Math.abs(fd - analyticGrad));
    }
    expect(maxErr).toBeLessThan(1e-4);
  });
});
