/**
 * Molecular dipole moment from Extended-Hueckel Mulliken partial charges.
 *
 * mu = sum_i q_i * R_i, the point-charge approximation to the dipole. This is
 * only origin-independent for a NEUTRAL molecule (sum_i q_i = 0); every target
 * here is neutral, so the caller need not recenter on a center of charge/mass.
 *
 * q_i comes from the existing EHT Mulliken population analysis
 * (orbitals/population.ts) -- this module does not touch the overlap/coefficient
 * math at all, it only sums charges * positions and converts units.
 *
 * Units: EHT Mulliken charges are in units of e (elementary charge); Molecule
 * positions are Angstrom (see geometry/molecule.ts). So the raw sum is in
 * e*Angstrom, converted to Debye via
 *
 *   1 e * Angstrom = 4.80320 D
 *
 * Derivation: 1 D = 1e-18 esu*cm (Debye's original cgs-esu definition); the
 * elementary charge is e = 4.803204712...e-10 esu; 1 Angstrom = 1e-8 cm. So
 *   1 e*Angstrom = 4.803204712e-10 esu * 1e-8 cm = 4.803204712e-18 esu*cm
 *                = 4.803204712 D.
 * (Equivalently in SI: e = 1.602176634e-19 C, 1 D = 3.33564e-30 C*m, 1 Angstrom
 * = 1e-10 m -> 1 e*Angstrom = 1.602176634e-29 C*m / 3.33564e-30 (C*m/D) =
 * 4.80321 D -- same value from the SI side.)
 *
 * EHT-Mulliken dipoles are a qualitative estimate (water lands well within
 * [0.5, 4.0] D of the experimental 1.85 D; symmetry-nonpolar molecules like
 * CO2/CH4 land near zero from geometry alone, independent of the charge
 * magnitudes) -- see packages/engine/test/spectra.test.ts for the tolerances.
 */
import type { Molecule } from "../geometry/molecule.js";
import { extendedHuckel, type ExtendedHuckelOptions } from "../orbitals/extendedHuckel.js";
import { mullikenCharges } from "../orbitals/population.js";

/** 1 e*Angstrom in Debye (see module docstring for the derivation). */
export const DEBYE_PER_E_ANGSTROM = 4.80320;

export interface DipoleResult {
  /** mu_x, mu_y, mu_z in Debye (lab/molecule frame the input geometry is in). */
  vector: [number, number, number];
  /** |mu| in Debye. */
  magnitude: number;
}

/**
 * Molecular dipole moment mu = sum_i q_i * R_i, in Debye, from an EHT-Mulliken
 * charge analysis of `mol`. Runs its own (cheap, non-iterative) extended-Hueckel
 * solve -- callers do not need to have one already computed.
 */
export async function computeDipole(
  mol: Molecule,
  options: ExtendedHuckelOptions = {},
): Promise<DipoleResult> {
  const eht = await extendedHuckel(mol, options);
  const { atomCharges } = mullikenCharges(eht, mol.symbols);

  let mx = 0;
  let my = 0;
  let mz = 0;
  const n = mol.symbols.length;
  for (let i = 0; i < n; i++) {
    const q = atomCharges[i]!;
    mx += q * mol.positions[3 * i]!;
    my += q * mol.positions[3 * i + 1]!;
    mz += q * mol.positions[3 * i + 2]!;
  }
  const vector: [number, number, number] = [
    mx * DEBYE_PER_E_ANGSTROM,
    my * DEBYE_PER_E_ANGSTROM,
    mz * DEBYE_PER_E_ANGSTROM,
  ];
  const magnitude = Math.hypot(vector[0], vector[1], vector[2]);
  return { vector, magnitude };
}
