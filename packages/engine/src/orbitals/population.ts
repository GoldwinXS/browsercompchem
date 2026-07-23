/**
 * Mulliken population analysis over an extended-Hueckel result.
 *
 * Mulliken's partition (R. S. Mulliken, J. Chem. Phys. 23, 1833 (1955)) splits
 * the electron density among the atomic orbitals using the overlap matrix. For a
 * closed-shell set of MOs with coefficients C (AO x MO) and occupations occ, the
 * GROSS ORBITAL POPULATION of AO mu is
 *
 *   n_mu = sum_i occ_i * c_{mu,i} * (S c_i)_mu
 *        = sum_i occ_i * c_{mu,i} * sum_nu S_{mu,nu} c_{nu,i}.
 *
 * Summed over all AOs this recovers the electron count exactly, because each MO
 * is S-normalized (c_i^T S c_i = 1), so sum_mu n_mu = sum_i occ_i = N_elec.
 *
 * Grouping the AO populations by atom gives the atomic population N_A, and the
 * Mulliken partial charge is
 *
 *   q_A = Z_valence(A) - N_A.
 *
 * Because sum_A N_A = N_elec = sum_A Z_valence(A) - charge, the charges sum to
 * the molecular charge (zero for a neutral) to machine precision.
 *
 * The same c_{mu} (S c)_mu weights, taken for a SINGLE MO, give that orbital's
 * fractional composition per AO (summing to 1). Some weights can be slightly
 * negative -- the well-known Mulliken artefact of assigning half the overlap
 * population to each partner -- so a "percent" can dip below 0 or above 100; that
 * is expected and left intact rather than clamped.
 */
import type { ExtendedHuckelResult } from "./extendedHuckel.js";
import { EHT_PARAMS, type OrbitalType } from "./parameters.js";

/** Result of a Mulliken population analysis, index-aligned with the atoms. */
export interface MullikenResult {
  /** Gross Mulliken electron population on each atom. Length = nAtoms. */
  atomPopulations: Float64Array;
  /** Mulliken partial charge q_A = Z_valence(A) - population. Length = nAtoms. */
  atomCharges: Float64Array;
}

/** One atomic orbital's contribution to a single MO (weights sum to ~1). */
export interface AoContribution {
  /** Atom this AO sits on. */
  atomIndex: number;
  /** Real-Cartesian AO flavour ("s" | "px" | "py" | "pz"). */
  aoType: OrbitalType;
  /** Mulliken partition weight c_mu (S c)_mu for the MO (fraction, may be < 0). */
  weight: number;
}

/**
 * S c for a single MO column: (S c)_mu = sum_nu S_{mu,nu} c_{nu,mo}. Reused by
 * both the charge and composition passes.
 */
function overlapTimesCoeff(
  result: ExtendedHuckelResult,
  mo: number,
  out: Float64Array,
): void {
  const { overlap: S, coefficients: C, nMO, nBasisFunctions: nb } = result;
  for (let mu = 0; mu < nb; mu++) {
    let s = 0;
    const rowBase = mu * nb;
    for (let nu = 0; nu < nb; nu++) s += S[rowBase + nu]! * C[nu * nMO + mo]!;
    out[mu] = s;
  }
}

/**
 * Mulliken partial charges for an EHT result. `symbols` supplies the per-atom
 * valence-electron reference (from the EHT parameter table); it must match the
 * geometry the result was computed for.
 */
export function mullikenCharges(
  result: ExtendedHuckelResult,
  symbols: readonly string[],
): MullikenResult {
  const { coefficients: C, occupations, nMO, aos, nBasisFunctions: nb } = result;
  const gross = new Float64Array(nb); // gross orbital population per AO
  const Sc = new Float64Array(nb);
  for (let mo = 0; mo < nMO; mo++) {
    const occ = occupations[mo]!;
    if (occ <= 0) continue;
    overlapTimesCoeff(result, mo, Sc);
    for (let mu = 0; mu < nb; mu++) gross[mu] = gross[mu]! + occ * C[mu * nMO + mo]! * Sc[mu]!;
  }

  const nAtoms = symbols.length;
  const atomPopulations = new Float64Array(nAtoms);
  for (let mu = 0; mu < nb; mu++) {
    const a = aos[mu]!.atomIndex;
    atomPopulations[a] = atomPopulations[a]! + gross[mu]!;
  }

  const atomCharges = new Float64Array(nAtoms);
  for (let a = 0; a < nAtoms; a++) {
    const zVal = EHT_PARAMS[symbols[a]!]?.valenceElectrons ?? 0;
    atomCharges[a] = zVal - atomPopulations[a]!;
  }
  return { atomPopulations, atomCharges };
}

/**
 * Per-AO Mulliken composition of a single MO, sorted by descending |weight| so
 * the dominant contributors come first. Weights sum to ~1 (the MO is
 * S-normalized). `orbitalIndex` is an MO index into the result's energy list.
 */
export function orbitalComposition(
  result: ExtendedHuckelResult,
  orbitalIndex: number,
): AoContribution[] {
  const { coefficients: C, nMO, aos, nBasisFunctions: nb } = result;
  if (orbitalIndex < 0 || orbitalIndex >= nMO) {
    throw new Error(
      `orbitalComposition: orbital ${orbitalIndex} out of range (nMO=${nMO}).`,
    );
  }
  const Sc = new Float64Array(nb);
  overlapTimesCoeff(result, orbitalIndex, Sc);
  const out: AoContribution[] = [];
  for (let mu = 0; mu < nb; mu++) {
    out.push({
      atomIndex: aos[mu]!.atomIndex,
      aoType: aos[mu]!.type,
      weight: C[mu * nMO + orbitalIndex]! * Sc[mu]!,
    });
  }
  out.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  return out;
}
