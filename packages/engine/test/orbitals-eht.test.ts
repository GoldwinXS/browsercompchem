/**
 * Extended Hueckel Theory (EHT) molecular-orbital tests, anchored to the
 * qualitative electronic structure textbooks report for these molecules and to
 * the published EHT frontier-orbital energies (Hoffmann, J. Chem. Phys. 39,
 * 1397 (1963), with the standard parameter set in parameters.ts).
 *
 * Reference facts checked here:
 *   - H2: a bonding sigma below an antibonding sigma*.
 *   - Ethylene: a pi HOMO / pi* LUMO with a sane gap.
 *   - Water: the HOMO is the oxygen lone pair b1 (pure p perpendicular to the
 *     molecular plane).
 *   - Benzene: a doubly-degenerate pi HOMO (e1g, ~-12.8 eV with these params)
 *     and a doubly-degenerate pi* LUMO (e2u, ~-8.3 eV) -- the classic
 *     a2u < e1g(2) < e2u(2) < b2g pi manifold.
 *   - Orthonormality C^T S C = I, correct electron counts, and the sigma /
 *     sigma* nodal structure of the H2 orbital grid.
 */
import { describe, it, expect } from "vitest";
import { extendedHuckel } from "../src/orbitals/extendedHuckel.js";
import { evaluateOrbitalOnGrid, autoGridSpec } from "../src/orbitals/grid.js";
import type { Molecule } from "../src/geometry/molecule.js";

function mol(symbols: string[], pos: number[]): Molecule {
  return { symbols, positions: Float64Array.from(pos), charge: 0, multiplicity: 1 };
}

/** Trigonal-planar-ish ethylene, C=C along x in the xy plane. */
function ethylene(): Molecule {
  const d = 1.33, ch = 1.086, ang = (120 * Math.PI) / 180;
  const cx = d / 2;
  const hx = cx + ch * Math.cos(Math.PI - ang);
  const hy = ch * Math.sin(Math.PI - ang);
  return mol(
    ["C", "C", "H", "H", "H", "H"],
    [-cx, 0, 0, cx, 0, 0, -hx, hy, 0, -hx, -hy, 0, hx, hy, 0, hx, -hy, 0],
  );
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

/** Max |C^T S C - I| over the MO block. */
function orthonormalityError(
  coeffs: Float64Array,
  S: Float64Array,
  nb: number,
  nMO: number,
): number {
  let maxErr = 0;
  for (let a = 0; a < nMO; a++) {
    for (let b = 0; b < nMO; b++) {
      let s = 0;
      for (let i = 0; i < nb; i++) {
        let sj = 0;
        for (let j = 0; j < nb; j++) sj += S[i * nb + j]! * coeffs[j * nMO + b]!;
        s += coeffs[i * nMO + a]! * sj;
      }
      maxErr = Math.max(maxErr, Math.abs(s - (a === b ? 1 : 0)));
    }
  }
  return maxErr;
}

/** Fraction of an MO's density carried by pz AOs (pi character in a planar mol). */
function pzFraction(r: Awaited<ReturnType<typeof extendedHuckel>>, mo: number): number {
  let pz = 0, tot = 0;
  for (let i = 0; i < r.aos.length; i++) {
    const c = r.coefficients[i * r.nMO + mo]!;
    tot += c * c;
    if (r.aos[i]!.type === "pz") pz += c * c;
  }
  return tot > 0 ? pz / tot : 0;
}

describe("EHT: H2 sigma / sigma*", () => {
  it("bonding sigma lies below antibonding sigma*, 2 electrons", async () => {
    const r = await extendedHuckel(mol(["H", "H"], [0, 0, 0, 0, 0, 0.74]));
    expect(r.nBasisFunctions).toBe(2);
    expect(r.nElectrons).toBe(2);
    expect(r.homoIndex).toBe(0);
    expect(r.lumoIndex).toBe(1);
    // bonding (HOMO) energy is below antibonding (LUMO)
    expect(r.orbitalEnergies[0]!).toBeLessThan(r.orbitalEnergies[1]!);
    // bonding MO: same-sign combination of the two 1s; antibonding: opposite.
    const cB0 = r.coefficients[0 * r.nMO + 0]!;
    const cB1 = r.coefficients[1 * r.nMO + 0]!;
    const cA0 = r.coefficients[0 * r.nMO + 1]!;
    const cA1 = r.coefficients[1 * r.nMO + 1]!;
    expect(cB0 * cB1).toBeGreaterThan(0); // in phase
    expect(cA0 * cA1).toBeLessThan(0); // out of phase
  });
});

describe("EHT: ethylene pi / pi*", () => {
  it("pi HOMO and pi* LUMO with a sane gap", async () => {
    const r = await extendedHuckel(ethylene());
    expect(r.nElectrons).toBe(12); // 2*C(4) + 4*H(1)
    const eHomo = r.orbitalEnergies[r.homoIndex]!;
    const eLumo = r.orbitalEnergies[r.lumoIndex]!;
    expect(eLumo).toBeGreaterThan(eHomo);
    const gap = eLumo - eHomo;
    expect(gap).toBeGreaterThan(3);
    expect(gap).toBeLessThan(8); // EHT ethylene pi-pi* ~5 eV
    // HOMO and LUMO are the pi system (pz-dominated, molecule in xy plane).
    expect(pzFraction(r, r.homoIndex)).toBeGreaterThan(0.9);
    expect(pzFraction(r, r.lumoIndex)).toBeGreaterThan(0.9);
    // HOMO = in-phase pi (C pz same sign); LUMO = out-of-phase pi*.
    const findPz = (atom: number, mo: number): number => {
      const i = r.aos.findIndex((a) => a.atomIndex === atom && a.type === "pz");
      return r.coefficients[i * r.nMO + mo]!;
    };
    expect(findPz(0, r.homoIndex) * findPz(1, r.homoIndex)).toBeGreaterThan(0);
    expect(findPz(0, r.lumoIndex) * findPz(1, r.lumoIndex)).toBeLessThan(0);
  });
});

describe("EHT: water HOMO is the oxygen lone-pair b1", () => {
  it("HOMO is the pure O p perpendicular to the molecular plane", async () => {
    // O at origin, H at (0, +-0.757, 0.586): molecular plane is x = 0 (yz).
    const r = await extendedHuckel(mol(["O", "H", "H"], [0, 0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586]));
    expect(r.nElectrons).toBe(8); // O(6) + 2 H(1)
    // The perpendicular direction is x -> the b1 lone pair is O px.
    const iPx = r.aos.findIndex((a) => a.atomIndex === 0 && a.type === "px");
    const cPx = r.coefficients[iPx * r.nMO + r.homoIndex]!;
    expect(Math.abs(cPx)).toBeGreaterThan(0.95); // essentially pure O px
    // Lone pair: density localized on oxygen (atom 0).
    let oWeight = 0, tot = 0;
    for (let i = 0; i < r.aos.length; i++) {
      const c = r.coefficients[i * r.nMO + r.homoIndex]!;
      tot += c * c;
      if (r.aos[i]!.atomIndex === 0) oWeight += c * c;
    }
    expect(oWeight / tot).toBeGreaterThan(0.9);
  });
});

describe("EHT: benzene degenerate pi HOMO/LUMO", () => {
  it("doubly-degenerate e1g HOMO and e2u LUMO near published EHT energies", async () => {
    const r = await extendedHuckel(benzene());
    expect(r.nElectrons).toBe(30); // 6*C(4) + 6*H(1)
    // HOMO is a degenerate pair (e1g): HOMO and HOMO-1 share an energy.
    const eHomo = r.orbitalEnergies[r.homoIndex]!;
    const eHomoM1 = r.orbitalEnergies[r.homoIndex - 1]!;
    expect(Math.abs(eHomo - eHomoM1)).toBeLessThan(1e-6);
    // LUMO is a degenerate pair (e2u): LUMO and LUMO+1 share an energy.
    const eLumo = r.orbitalEnergies[r.lumoIndex]!;
    const eLumoP1 = r.orbitalEnergies[r.lumoIndex + 1]!;
    expect(Math.abs(eLumo - eLumoP1)).toBeLessThan(1e-6);
    // Both frontier pairs are pure pi (pz).
    expect(pzFraction(r, r.homoIndex)).toBeGreaterThan(0.95);
    expect(pzFraction(r, r.lumoIndex)).toBeGreaterThan(0.95);
    // Published-EHT frontier energies (this parameter set): e1g ~ -12.81 eV,
    // e2u ~ -8.27 eV, HOMO-LUMO gap ~ 4.5 eV.
    expect(eHomo).toBeGreaterThan(-13.3);
    expect(eHomo).toBeLessThan(-12.3);
    expect(eLumo).toBeGreaterThan(-8.8);
    expect(eLumo).toBeLessThan(-7.8);
    expect(eLumo - eHomo).toBeGreaterThan(4);
    expect(eLumo - eHomo).toBeLessThan(5);
  });
});

describe("EHT: orthonormality and electron counts", () => {
  it("C^T S C = identity within 1e-8 (water, ethylene, benzene)", async () => {
    for (const m of [
      mol(["O", "H", "H"], [0, 0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586]),
      ethylene(),
      benzene(),
    ]) {
      const r = await extendedHuckel(m);
      const err = orthonormalityError(r.coefficients, r.overlap, r.nBasisFunctions, r.nMO);
      expect(err).toBeLessThan(1e-8);
    }
  });

  it("rejects molecules with elements outside the EHT set", async () => {
    await expect(extendedHuckel(mol(["B", "H", "H", "H"], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]))).rejects.toThrow(
      /outside the EHT set/,
    );
  });
});

describe("EHT: H2 orbital grid nodal structure", () => {
  it("sigma bonding keeps one sign between the nuclei; sigma* changes sign at the midplane", async () => {
    const R = 0.74;
    const r = await extendedHuckel(mol(["H", "H"], [0, 0, 0, 0, 0, R]));
    const spec = autoGridSpec(Float64Array.from([0, 0, 0, 0, 0, R]), { spacing: 0.15, padding: 2 });
    const s = spec.spacing as number;
    const [nx, ny, nz] = spec.dims;
    const ix = Math.round((0 - spec.origin[0]) / s);
    const iy = Math.round((0 - spec.origin[1]) / s);
    const sampleAlongBond = (field: Float32Array): number[] =>
      [0.1, 0.3, 0.5, 0.7, 0.9].map((f) => {
        const iz = Math.round((f * R - spec.origin[2]) / s);
        return field[ix + nx * (iy + ny * iz)]!;
      });

    const bonding = evaluateOrbitalOnGrid(r, r.homoIndex, spec);
    const anti = evaluateOrbitalOnGrid(r, r.lumoIndex, spec);
    const bSamples = sampleAlongBond(bonding);
    const aSamples = sampleAlongBond(anti);

    // sigma: all samples between the nuclei share one sign (no node).
    const allSameSign = bSamples.every((v) => Math.sign(v) === Math.sign(bSamples[0]!));
    expect(allSameSign).toBe(true);
    bSamples.forEach((v) => expect(Math.abs(v)).toBeGreaterThan(1e-3));

    // sigma*: the endpoints straddle a nodal plane -> opposite signs, and the
    // value at the midpoint is near zero.
    expect(Math.sign(aSamples[0]!)).toBe(-Math.sign(aSamples[aSamples.length - 1]!));
    const midIz = Math.round((0.5 * R - spec.origin[2]) / s);
    const midVal = anti[ix + nx * (iy + ny * midIz)]!;
    expect(Math.abs(midVal)).toBeLessThan(0.15 * Math.max(...aSamples.map(Math.abs)));
  });
});
