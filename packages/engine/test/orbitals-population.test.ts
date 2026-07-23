/**
 * Mulliken population-analysis tests over the extended-Hueckel tier.
 *
 * Reference facts checked here:
 *   - Charges of a neutral molecule sum to zero (to machine precision), a direct
 *     consequence of S-normalized MOs (sum of gross populations = N_elec).
 *   - Water: oxygen carries a clear negative charge, the two hydrogens an equal
 *     positive one (by the molecule's C2v symmetry). The published EHT Mulliken
 *     water charges sit near O ~ -0.6, H ~ +0.3; we assert signs, H1==H2
 *     symmetry, and a sane magnitude range rather than exact values.
 *   - Benzene: all six carbons share one charge and all six hydrogens another,
 *     by D6h symmetry, and (neutral) they cancel.
 *   - The single-MO composition weights sum to 1 and, for the water HOMO (the
 *     oxygen lone pair), are overwhelmingly oxygen.
 */
import { describe, it, expect } from "vitest";
import { extendedHuckel } from "../src/orbitals/extendedHuckel.js";
import { mullikenCharges, orbitalComposition } from "../src/orbitals/population.js";
import type { Molecule } from "../src/geometry/molecule.js";

function mol(symbols: string[], pos: number[]): Molecule {
  return { symbols, positions: Float64Array.from(pos), charge: 0, multiplicity: 1 };
}

/** O at origin, H at (0, +-0.757, 0.586): C2v water, molecular plane x = 0. */
function water(): Molecule {
  return mol(["O", "H", "H"], [0, 0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586]);
}

/** D6h benzene, planar in xy. C-C 1.39 A, C-H 1.09 A. */
function benzene(): Molecule {
  const rC = 1.39, rH = rC + 1.09;
  const syms: string[] = [];
  const pos: number[] = [];
  for (let k = 0; k < 6; k++) {
    const a = (k * Math.PI) / 3;
    syms.push("C");
    pos.push(rC * Math.cos(a), rC * Math.sin(a), 0);
  }
  for (let k = 0; k < 6; k++) {
    const a = (k * Math.PI) / 3;
    syms.push("H");
    pos.push(rH * Math.cos(a), rH * Math.sin(a), 0);
  }
  return mol(syms, pos);
}

function sum(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]!;
  return s;
}

describe("Mulliken charges: neutrality and populations", () => {
  it("charges of neutral molecules sum to zero within 1e-8", async () => {
    for (const m of [water(), benzene(), mol(["C", "H", "H", "H", "H"], [0, 0, 0, 0.629, 0.629, 0.629, -0.629, -0.629, 0.629, -0.629, 0.629, -0.629, 0.629, -0.629, -0.629])]) {
      const r = await extendedHuckel(m);
      const { atomCharges, atomPopulations } = mullikenCharges(r, m.symbols);
      expect(Math.abs(sum(atomCharges))).toBeLessThan(1e-8);
      // Total population recovers the valence-electron count.
      expect(Math.abs(sum(atomPopulations) - r.nElectrons)).toBeLessThan(1e-8);
    }
  });
});

describe("Mulliken charges: water polarity and symmetry", () => {
  it("O clearly negative, H equal and positive, sane magnitude", async () => {
    const m = water();
    const r = await extendedHuckel(m);
    const { atomCharges } = mullikenCharges(r, m.symbols);
    const [qO, qH1, qH2] = [atomCharges[0]!, atomCharges[1]!, atomCharges[2]!];
    // Oxygen pulls density: negative. Hydrogens: positive.
    expect(qO).toBeLessThan(0);
    expect(qH1).toBeGreaterThan(0);
    expect(qH2).toBeGreaterThan(0);
    // C2v symmetry -> the two hydrogens are identical.
    expect(Math.abs(qH1 - qH2)).toBeLessThan(1e-6);
    // Neutral -> O charge balances the two H.
    expect(Math.abs(qO + qH1 + qH2)).toBeLessThan(1e-8);
    // Sane range (published EHT Mulliken water ~ O -0.6, H +0.3). Assert a band,
    // not an exact value: EHT charges are qualitative.
    expect(qO).toBeGreaterThan(-1.2);
    expect(qO).toBeLessThan(-0.15);
    expect(qH1).toBeGreaterThan(0.07);
    expect(qH1).toBeLessThan(0.6);
  });
});

describe("Mulliken charges: benzene symmetry", () => {
  it("six equal carbons, six equal hydrogens, canceling", async () => {
    const m = benzene();
    const r = await extendedHuckel(m);
    const { atomCharges } = mullikenCharges(r, m.symbols);
    const qC = atomCharges.slice(0, 6);
    const qH = atomCharges.slice(6, 12);
    for (let i = 1; i < 6; i++) {
      expect(Math.abs(qC[i]! - qC[0]!)).toBeLessThan(1e-6);
      expect(Math.abs(qH[i]! - qH[0]!)).toBeLessThan(1e-6);
    }
    // Carbon and hydrogen charges are opposite in sign and cancel per CH unit.
    expect(Math.abs(qC[0]! + qH[0]!)).toBeLessThan(1e-6);
    // Non-trivial polarity (the C-H bond is polarized), not an all-zero result.
    expect(Math.abs(qC[0]!)).toBeGreaterThan(1e-3);
  });
});

describe("orbital composition (single-MO Mulliken partition)", () => {
  it("weights sum to 1 and the water HOMO is oxygen-dominated", async () => {
    const m = water();
    const r = await extendedHuckel(m);
    const comp = orbitalComposition(r, r.homoIndex);
    // Partition of a normalized MO sums to 1.
    const total = comp.reduce((s, c) => s + c.weight, 0);
    expect(Math.abs(total - 1)).toBeLessThan(1e-8);
    // Sorted by descending magnitude: the top contributor is an oxygen AO.
    expect(comp[0]!.atomIndex).toBe(0);
    // The b1 lone pair is essentially pure O p -> the oxygen share dominates.
    const oShare = comp.filter((c) => c.atomIndex === 0).reduce((s, c) => s + c.weight, 0);
    expect(oShare).toBeGreaterThan(0.85);
  });

  it("rejects an out-of-range orbital index", async () => {
    const r = await extendedHuckel(water());
    expect(() => orbitalComposition(r, r.nMO)).toThrow(/out of range/);
  });
});
