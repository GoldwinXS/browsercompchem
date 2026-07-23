import type { Molecule } from "../geometry/molecule.js";

/**
 * Stub for the Extended Hueckel Theory (EHT) orbital tier.
 *
 * NOT YET IMPLEMENTED. Intended design (Hoffmann, J. Chem. Phys. 39,
 * 1397 (1963)):
 *
 *  - Build a minimal valence Slater-type-orbital basis per atom (from a
 *    small built-in table of valence-shell ionization potentials (VSIP)
 *    and Slater exponents, keyed by element).
 *  - Overlap matrix S from analytic STO-STO overlap integrals (function
 *    of internuclear distance + orbital angular momentum).
 *  - Off-diagonal Hamiltonian via the Wolfsberg-Helmholz approximation:
 *      H_ij = K * S_ij * (H_ii + H_jj) / 2   (K ~ 1.75, diagonal H_ii = -VSIP_i)
 *  - Solve the generalized eigenproblem H C = S C E for molecular
 *    orbital energies E and coefficients C.
 *
 * This is the cheapest of the three compute tiers (RDKit-JS force
 * fields / ONNX ML potentials / this) and is meant to give fast,
 * qualitative frontier-orbital (HOMO/LUMO) pictures for teaching and
 * for a first-pass electronic-structure view, not quantitative
 * energetics -- a real Hartree-Fock or DFT tier is a separate, later
 * effort.
 */
export interface ExtendedHuckelOptions {
  /** Wolfsberg-Helmholz proportionality constant (typical value ~1.75). */
  wolfsbergHelmholzK?: number;
}

export interface ExtendedHuckelResult {
  /** Molecular orbital energies, ascending, length = number of basis functions. */
  orbitalEnergies: Float64Array;
  /** MO coefficients, column-major: coefficients[basisIndex * nMO + moIndex]. */
  coefficients: Float64Array;
  nBasisFunctions: number;
}

export async function extendedHuckel(
  _mol: Molecule,
  _options: ExtendedHuckelOptions = {},
): Promise<ExtendedHuckelResult> {
  throw new Error("extendedHuckel is not implemented yet (see doc comment in extendedHuckel.ts).");
}
