/**
 * Harmonic normal-mode analysis regression tests.
 *
 * Each molecule is relaxed to a tight stationary point with FIRE, then
 * computeNormalModes() builds the finite-difference Hessian of ANI-2x's analytic
 * forces, mass-weights, diagonalizes, and drops the trans/rot modes.
 *
 * The baked target wavenumbers are the values the accuracy bench already
 * established and validated against experiment (water raw harmonic
 * [1721, 3815, 3934] cm^-1 vs exp [1595, 3657, 3756]; CO2 handled as linear with
 * 4 modes [645.5, 645.5, 1413.5, 2489]). Tolerances are generous (50 cm^-1) so
 * the test guards the pipeline, not the last digit of a specific model build.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { fileURLToPath } from "node:url";
import { Ani2xProvider } from "../src/potentials/ani2x/provider.js";
import { FireOptimizer } from "../src/optimize/fire.js";
import { computeNormalModes } from "../src/vibrations/normalModes.js";
import type { Molecule } from "../src/geometry/molecule.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const modelDir = here + "../../../models/ani2x";

async function relaxTight(mol: Molecule, provider: Ani2xProvider): Promise<Molecule> {
  const fire = new FireOptimizer({
    forceTolerance: 1e-4,
    maxSteps: 3000,
    dtMax: 0.2,
    maxStep: 0.1,
  });
  const r = await fire.optimize(mol, provider);
  return r.molecule;
}

/** Match each target to its nearest computed frequency, asserting it is within tol. */
function expectNear(freqs: number[], targets: number[], tol: number): void {
  for (const t of targets) {
    const nearest = freqs.reduce(
      (best, f) => (Math.abs(f - t) < Math.abs(best - t) ? f : best),
      freqs[0]!,
    );
    expect(Math.abs(nearest - t), `target ${t}, nearest ${nearest.toFixed(1)}`).toBeLessThan(tol);
  }
}

describe("computeNormalModes: harmonic frequencies", () => {
  let provider: Ani2xProvider;
  beforeAll(async () => {
    provider = await Ani2xProvider.create({ modelDir, variant: "full-f32" });
  });

  it("water: 3 real modes near ~1721 / ~3815 / ~3934 cm^-1", async () => {
    const mol: Molecule = {
      symbols: ["O", "H", "H"],
      positions: Float64Array.from([0, 0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586]),
      charge: 0,
      multiplicity: 1,
    };
    const relaxed = await relaxTight(mol, provider);
    const nm = await computeNormalModes(relaxed, provider);

    expect(nm.isLinear).toBe(false);
    expect(nm.frequencies.length).toBe(3); // 3N - 6 = 3
    // all real (positive) at a genuine minimum
    expect(nm.frequencies.every((f) => f > 0)).toBe(true);
    expectNear(nm.frequencies, [1721, 3815, 3934], 50);
    // trans/rot residual should be small (well-behaved stationary point)
    expect(nm.maxResidualTransRot).toBeLessThan(60);
    // each mode is a unit-norm 3N vector
    for (const m of nm.modes) {
      expect(m.length).toBe(9);
      let ss = 0;
      for (const c of m) ss += c * c;
      expect(Math.abs(Math.sqrt(ss) - 1)).toBeLessThan(1e-9);
    }
  }, 60000);

  it("CO2: linear, 4 modes near ~646 (x2) / ~1414 / ~2489 cm^-1", async () => {
    const mol: Molecule = {
      symbols: ["C", "O", "O"],
      positions: Float64Array.from([0, 0, 0, 1.16, 0, 0, -1.16, 0, 0]),
      charge: 0,
      multiplicity: 1,
    };
    const relaxed = await relaxTight(mol, provider);
    const nm = await computeNormalModes(relaxed, provider);

    expect(nm.isLinear).toBe(true);
    expect(nm.frequencies.length).toBe(4); // 3N - 5 = 4
    expect(nm.frequencies.every((f) => f > 0)).toBe(true);
    // the two bends are degenerate (~646), then asymmetric ~1414, ~2489
    expectNear(nm.frequencies, [646, 646, 1414, 2489], 60);
  }, 60000);
});

describe("computeNormalModes: translation/rotation removal", () => {
  it("removes exactly 6 near-zero modes for a non-linear molecule (methane)", async () => {
    const provider = await Ani2xProvider.create({ modelDir, variant: "full-f32" });
    // Tetrahedral-ish methane seed.
    const c = 0.63;
    const mol: Molecule = {
      symbols: ["C", "H", "H", "H", "H"],
      positions: Float64Array.from([
        0, 0, 0,
        c, c, c,
        c, -c, -c,
        -c, c, -c,
        -c, -c, c,
      ]),
      charge: 0,
      multiplicity: 1,
    };
    const relaxed = await relaxTight(mol, provider);
    const nm = await computeNormalModes(relaxed, provider);

    expect(nm.isLinear).toBe(false);
    // 3N = 15; 15 - 6 = 9 vibrational modes
    expect(nm.frequencies.length).toBe(9);
    // the 6 removed trans/rot modes are near zero
    expect(nm.maxResidualTransRot).toBeLessThan(60);
    // all 15 wavenumbers accounted for
    expect(nm.allWavenumbers.length).toBe(15);
    // the 6 smallest-magnitude wavenumbers are the removed ones -> near zero
    const byAbs = [...nm.allWavenumbers].sort((a, b) => Math.abs(a) - Math.abs(b));
    for (let i = 0; i < 6; i++) expect(Math.abs(byAbs[i]!)).toBeLessThan(60);
  }, 60000);
});
