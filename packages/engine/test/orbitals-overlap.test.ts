/**
 * Analytic STO overlap-integral tests, anchored to textbook values.
 *
 * The classic reference point is the hydrogen 1s-1s overlap: with a hydrogenic
 * exponent zeta = 1.0 bohr^-1 at the H2 bond length R = 0.74 A, S ~= 0.75 (see
 * e.g. Mulliken, Rieke, Orloff & Orloff, J. Chem. Phys. 17, 1248 (1949), and
 * every quantum-chemistry text's worked H2 overlap). We also check the physical
 * limits (S -> 1 as R -> 0 for identical orbitals, S -> 0 as R -> infinity) and
 * the qualitative s-p / p-p sigma vs pi behaviour.
 */
import { describe, it, expect } from "vitest";
import {
  overlap1s1s,
  sigmaFundamental,
  piFundamental,
} from "../src/orbitals/overlap.js";

const BOHR_PER_A = 1 / 0.529177210903;

describe("STO overlap integrals", () => {
  it("H2 1s-1s overlap at 0.74 A (zeta=1.0) is ~0.75", () => {
    const s = overlap1s1s(1.0, 1.0, 0.74 * BOHR_PER_A);
    expect(s).toBeGreaterThan(0.74);
    expect(s).toBeLessThan(0.76);
    expect(s).toBeCloseTo(0.7534, 3);
  });

  it("S -> 1 as R -> 0 for identical orbitals", () => {
    expect(overlap1s1s(1.3, 1.3, 1e-3)).toBeCloseTo(1.0, 5);
    // 2s and 2p sigma of carbon likewise self-normalize toward 1 at contact.
    expect(sigmaFundamental(2, 1.625, 0, 2, 1.625, 0, 1e-3)).toBeCloseTo(1.0, 4);
  });

  it("S -> 0 as R -> infinity", () => {
    expect(Math.abs(overlap1s1s(1.3, 1.3, 12))).toBeLessThan(1e-3);
    expect(Math.abs(sigmaFundamental(2, 1.625, 1, 2, 1.625, 1, 14))).toBeLessThan(1e-3);
  });

  it("overlap decreases monotonically with distance (1s-1s)", () => {
    let prev = Infinity;
    for (const rA of [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]) {
      const s = overlap1s1s(1.3, 1.3, rA * BOHR_PER_A);
      expect(s).toBeLessThan(prev);
      prev = s;
    }
  });

  it("s-p sigma overlap is nonzero and vanishes only at R->0/infinity", () => {
    const s = sigmaFundamental(2, 1.625, 0, 2, 1.625, 1, 1.4 * BOHR_PER_A);
    expect(Math.abs(s)).toBeGreaterThan(0.1);
  });

  it("p-p sigma is larger in magnitude than p-p pi at a bonding distance", () => {
    // Carbon 2p at ~1.4 A: sigma overlaps head-on (bigger), pi sideways (smaller).
    const sig = sigmaFundamental(2, 1.625, 1, 2, 1.625, 1, 1.4 * BOHR_PER_A);
    const pi = piFundamental(2, 1.625, 2, 1.625, 1.4 * BOHR_PER_A);
    expect(Math.abs(sig)).toBeGreaterThan(Math.abs(pi));
    // pi overlap is positive (both lobes same sign sideways); the head-on sigma
    // is negative in the global-axis convention (both p_z point +z, so the near
    // lobes have opposite sign). Magnitudes are what carry physical meaning.
    expect(pi).toBeGreaterThan(0);
    expect(sig).toBeLessThan(0);
  });

  it("heteronuclear overlap (t != 0 branch) is finite and sane", () => {
    // C(2p) - H(1s) sigma at 1.09 A exercises the q != 0 B_k recurrence branch.
    const s = sigmaFundamental(2, 1.625, 1, 1, 1.3, 0, 1.09 * BOHR_PER_A);
    expect(Number.isFinite(s)).toBe(true);
    expect(Math.abs(s)).toBeGreaterThan(0.1);
    expect(Math.abs(s)).toBeLessThan(1.0);
  });
});
