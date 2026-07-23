/**
 * Simulated IR spectrum + dipole correctness tests.
 *
 * The definitive proof this feature is doing real physics (not just plotting
 * uniform sticks) is the set of SELECTION RULES a harmonic IR intensity model
 * must reproduce from symmetry alone, independent of any absolute-intensity
 * prefactor:
 *
 *   - CO2 (D-inf-h, centrosymmetric): only UNGERADE normal modes can have a
 *     nonzero dipole derivative (the dipole operator is itself ungerade, so a
 *     gerade mode's d(mu)/dQ vanishes by parity to all orders -- a rigorous
 *     group-theory result, not a numerical coincidence). The symmetric stretch
 *     (sigma_g+) is GERADE -> must be IR-silent. The bend (pi_u) and the
 *     antisymmetric stretch (sigma_u+) are UNGERADE -> both IR-active, with
 *     the antisymmetric stretch by far the strongest (the classic ~2349 cm^-1
 *     greenhouse-gas band). See docs/LITERATURE_VALIDATION.md's CO2 section.
 *   - Methane (Td): translations span only the T2 irrep, so only T2 normal
 *     modes are IR-active; the totally-symmetric A1 breathing mode and the
 *     doubly-degenerate E bend are IR-SILENT (Raman-active only).
 *   - Water (C2v): translations span a1+b1+b2, and water's 3 modes are
 *     2 x a1 + 1 x b2 -- every mode sits in an IR-active irrep, so all three
 *     must show real intensity.
 *
 * Frequencies come from the SAME computeNormalModes() the vibrations feature
 * uses (no separate Hessian code here); this file only exercises the new
 * spectra/ modules built on top of it.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { fileURLToPath } from "node:url";
import { Ani2xProvider } from "../src/potentials/ani2x/provider.js";
import { FireOptimizer } from "../src/optimize/fire.js";
import { computeNormalModes } from "../src/vibrations/normalModes.js";
import { computeDipole } from "../src/spectra/dipole.js";
import { computeIRSpectrum } from "../src/spectra/irSpectrum.js";
import type { Molecule } from "../src/geometry/molecule.js";
import type { ModeIntensity } from "../src/spectra/irIntensities.js";

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

/**
 * Group ascending-sorted frequencies into degeneracy clusters: a harmonic
 * Hessian diagonalization reproduces an exact symmetry-required degeneracy to
 * near machine precision (adjacent gap ~1e-6 cm^-1), while distinct vibrational
 * bands of these small molecules sit tens to thousands of cm^-1 apart -- so a
 * modest adjacent-gap threshold cleanly separates "the same mode" from "a
 * different mode" without hardcoding any absolute frequency target.
 */
function clusterByDegeneracy(freqs: number[], gapThreshold = 5): number[][] {
  const order = freqs.map((_, i) => i).sort((a, b) => freqs[a]! - freqs[b]!);
  const clusters: number[][] = [[order[0]!]];
  for (let k = 1; k < order.length; k++) {
    const prev = freqs[order[k - 1]!]!;
    const cur = freqs[order[k]!]!;
    if (cur - prev > gapThreshold) clusters.push([]);
    clusters[clusters.length - 1]!.push(order[k]!);
  }
  return clusters;
}

function maxRelative(modes: ModeIntensity[]): number {
  return modes.reduce((m, x) => Math.max(m, x.relative), 0);
}

describe("computeDipole + computeIRSpectrum: selection rules", () => {
  let provider: Ani2xProvider;
  beforeAll(async () => {
    provider = await Ani2xProvider.create({ modelDir, variant: "full-f32" });
  }, 60000);

  it("water: dipole ~1-3 D and all three modes IR-active (C2v, 2a1+b2, every irrep IR-active)", async () => {
    const mol: Molecule = {
      symbols: ["O", "H", "H"],
      positions: Float64Array.from([0, 0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586]),
      charge: 0,
      multiplicity: 1,
    };
    const relaxed = await relaxTight(mol, provider);

    const dip = await computeDipole(relaxed);
    // EHT-Mulliken is a coarse point-charge model; it need not hit the
    // experimental 1.85 D exactly (this landed ~2.5 D), but it must be a real,
    // finite, non-crazy dipole -- the task's own documented tolerance window.
    expect(dip.magnitude).toBeGreaterThan(0.5);
    expect(dip.magnitude).toBeLessThan(4.0);

    const nm = await computeNormalModes(relaxed, provider);
    expect(nm.frequencies.length).toBe(3);
    // sanity: harmonic frequencies still land in the right region (bend region
    // is 800-2000 first here, both stretches above 3400) -- generous, this is
    // not a frequency-accuracy test (vibrations.test.ts already owns that).
    expectNear(nm.frequencies, [1721, 3815, 3934], 60);

    const spec = await computeIRSpectrum(relaxed, nm.modes, nm.frequencies);
    expect(spec.modes.length).toBe(3);

    // No accidental degeneracy in water -- three distinct clusters.
    const clusters = clusterByDegeneracy(nm.frequencies);
    expect(clusters.map((c) => c.length)).toEqual([1, 1, 1]);

    // Every mode is a genuine, non-vanishing IR band.
    for (const m of spec.modes) {
      expect(m.isActive).toBe(true);
      expect(m.relative).toBeGreaterThan(1);
    }
    // Bend (lowest freq) and antisymmetric stretch (highest freq) are the
    // strong bands; report their sizes for the record.
    const byFreq = [...spec.modes].sort((a, b) => Math.abs(a.frequency) - Math.abs(b.frequency));
    const [bend, symStretch, antiStretch] = byFreq;
    expect(bend!.relative).toBeGreaterThan(10);
    expect(antiStretch!.relative).toBeGreaterThan(10);
    expect(symStretch!.relative).toBeGreaterThan(1);

    // Curve is a real, non-empty broadened spectrum.
    expect(spec.curve.wavenumbers.length).toBeGreaterThan(100);
    expect(spec.curve.absorbance.some((a) => a > 1)).toBe(true);
  }, 90000);

  it("CO2: symmetric stretch (gerade) is IR-silent; antisymmetric stretch (ungerade) dominates; bend active; dipole ~0", async () => {
    const mol: Molecule = {
      symbols: ["C", "O", "O"],
      positions: Float64Array.from([0, 0, 0, 1.16, 0, 0, -1.16, 0, 0]),
      charge: 0,
      multiplicity: 1,
    };
    const relaxed = await relaxTight(mol, provider);

    const dip = await computeDipole(relaxed);
    // Centrosymmetric geometry -> dipole vanishes by symmetry regardless of
    // the (nonzero) individual Mulliken atomic charges.
    expect(dip.magnitude).toBeLessThan(0.2);

    const nm = await computeNormalModes(relaxed, provider);
    expect(nm.isLinear).toBe(true);
    expect(nm.frequencies.length).toBe(4);
    expectNear(nm.frequencies, [646, 646, 1414, 2489], 60);

    const spec = await computeIRSpectrum(relaxed, nm.modes, nm.frequencies);

    // Clusters: degenerate bend pair, then the two non-degenerate stretches.
    const clusters = clusterByDegeneracy(nm.frequencies);
    expect(clusters.map((c) => c.length)).toEqual([2, 1, 1]);
    const [bendIdx, symIdx, antiIdx] = clusters;
    const byIdx = (i: number) => spec.modes[i]!;

    const maxRel = maxRelative(spec.modes);

    // *** THE definitive gate ***: the deperturbed symmetric stretch (~1333
    // exp / ~1414 computed harmonic) is centrosymmetric-GERADE and MUST be
    // (near-)silent -- less than 1% of the strongest band.
    const symStretch = byIdx(symIdx![0]!);
    expect(symStretch.relative).toBeLessThan(1);

    // The antisymmetric stretch (~2349 exp / ~2489 computed) is ungerade and
    // must be the strongest band in the spectrum.
    const antiStretch = byIdx(antiIdx![0]!);
    expect(antiStretch.relative).toBeCloseTo(maxRel, 5);
    expect(antiStretch.isActive).toBe(true);
    expect(antiStretch.relative).toBeGreaterThan(symStretch.relative * 10);

    // The doubly-degenerate bend (~667 exp / ~646 computed) is ungerade (pi_u)
    // and must be active too.
    for (const i of bendIdx!) {
      expect(byIdx(i).isActive).toBe(true);
      expect(byIdx(i).relative).toBeGreaterThan(1);
    }

    // The plotted curve must show the ~1414 region essentially flat/absent
    // relative to the towering ~2489 band.
    const curveAt = (target: number) => {
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < spec.curve.wavenumbers.length; i++) {
        const d = Math.abs(spec.curve.wavenumbers[i]! - target);
        if (d < bestDist) {
          bestDist = d;
          best = spec.curve.absorbance[i]!;
        }
      }
      return best;
    };
    const at1414 = curveAt(Math.abs(symStretch.frequency));
    const at2489 = curveAt(Math.abs(antiStretch.frequency));
    expect(at2489).toBeGreaterThan(50); // strong peak, near the 0-100 relative scale
    expect(at1414).toBeLessThan(at2489 * 0.05); // sym-stretch region is essentially absent
  }, 90000);

  it("methane: A1 breathing + E bend IR-silent (Td, non-T2 irreps); T2 modes dominate; dipole ~0", async () => {
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

    const dip = await computeDipole(relaxed);
    // Td symmetry forces the dipole to vanish (the 4 equal-charge H's are
    // arranged so their position vectors sum to zero about the central C).
    expect(dip.magnitude).toBeLessThan(0.2);

    const nm = await computeNormalModes(relaxed, provider);
    expect(nm.frequencies.length).toBe(9); // 3*5 - 6

    const spec = await computeIRSpectrum(relaxed, nm.modes, nm.frequencies);

    // Td methane's 9 modes split as T2(3, lowest) + E(2) + A1(1) + T2(3,
    // highest) -- ANI-2x's own harmonic frequencies run somewhat high vs the
    // Shimanouchi fundamentals (nu4 1306, nu2 1534, nu1 2917, nu3 3019) but the
    // MULTIPLICITY structure is symmetry-exact and is what we assert on;
    // absolute-frequency accuracy for this potential is vibrations.test.ts's
    // job, not this feature's.
    const clusters = clusterByDegeneracy(nm.frequencies);
    expect(clusters.map((c) => c.length)).toEqual([3, 2, 1, 3]);
    const [t2LowIdx, eIdx, a1Idx, t2HighIdx] = clusters;
    const byIdx = (i: number) => spec.modes[i]!;

    // The A1 totally-symmetric "breathing" stretch: IR-silent BY SYMMETRY.
    // Measured relative intensity sits ~1e-22 (pure finite-difference/roundoff
    // noise floor, confirmed delta-independent from 0.002-0.04 Angstrom during
    // development) -- effectively exact zero, not "small".
    const a1 = byIdx(a1Idx![0]!);
    expect(a1.relative).toBeLessThan(1e-6);
    expect(a1.isActive).toBe(false);

    // The doubly-degenerate E bend: also IR-silent (Raman-only irrep in Td) --
    // same ~1e-22 noise-floor scale as A1.
    for (const i of eIdx!) {
      expect(byIdx(i).relative).toBeLessThan(1e-6);
    }

    // The two T2 triplets are the ONLY dipole-allowed fundamentals (Td
    // translations span exactly T2) and must dominate the forbidden A1/E
    // modes by many orders of magnitude -- true regardless of each T2 band's
    // own absolute size. In practice the high-frequency T2 (antisymmetric C-H
    // stretch, nu3) is the single strongest band in the whole spectrum
    // (relative ~100), while the low-frequency T2 (bend, nu4) is a real but
    // WEAK band here (~0.17% of the strongest -- this EHT-Mulliken model
    // underestimates the bending mode's dipole response relative to real
    // methane, where nu4 is moderate, not this faint; see the final report's
    // honesty note). Both are still ~10^16-10^17x the forbidden A1/E modes,
    // which is the actual, prefactor-independent selection-rule claim.
    const t2Relatives = [...t2LowIdx!, ...t2HighIdx!].map((i) => byIdx(i).relative);
    for (const r of t2Relatives) expect(r).toBeGreaterThan(1e-3);
    expect(Math.max(...t2Relatives)).toBeCloseTo(maxRelative(spec.modes), 5);
    // The strongest band overall belongs to a T2 mode, not a forbidden one.
    const maxIdx = spec.modes.reduce((best, m, i) => (m.relative > spec.modes[best]!.relative ? i : best), 0);
    expect([...t2LowIdx!, ...t2HighIdx!]).toContain(maxIdx);
  }, 90000);
});

/** Match each target to its nearest computed frequency, asserting it is within tol.
 * (Same helper vibrations.test.ts uses -- duplicated locally to keep this file
 * self-contained rather than reaching into another test file.) */
function expectNear(freqs: number[], targets: number[], tol: number): void {
  for (const t of targets) {
    const nearest = freqs.reduce(
      (best, f) => (Math.abs(f - t) < Math.abs(best - t) ? f : best),
      freqs[0]!,
    );
    expect(Math.abs(nearest - t), `target ${t}, nearest ${nearest.toFixed(1)}`).toBeLessThan(tol);
  }
}
