/**
 * Harmonic vibrational analysis for the accuracy suite.
 *
 * The heavy lifting -- finite-difference Hessian of the analytic forces,
 * mass-weighting, Jacobi diagonalization, wavenumber conversion, and removal of
 * the 6/5 trans/rot modes -- now lives in the ENGINE
 * (@browser-comp-chem/engine, src/vibrations). This module is a thin wrapper
 * that adds the thermochemistry the bench needs (zero-point energy and 0->298 K
 * vibrational thermal energy, RRHO) on top of the engine's frequencies.
 *
 * Keeping a single implementation in the engine means the demo and the bench
 * compute identical frequencies; this file used to carry its own Hessian +
 * eigensolver copy (and packages/bench/src/accuracy/linalg.ts), both now gone.
 *
 * All energies from the provider are Hartree; positions/forces are Angstrom /
 * Hartree-per-Angstrom (ANI-2x convention). Frequencies come out in cm^-1.
 */
import type { EnergyForceProvider, Molecule } from "@browser-comp-chem/engine";
import { computeNormalModes, isLinearGeometry } from "@browser-comp-chem/engine";
import { CM1_TO_KJ_MOL, HC_OVER_K, R_KJ, T_STANDARD } from "./chem.js";

/**
 * Re-exported from the engine (same (symbols, positions) signature the bench
 * has always used) so existing imports in heats.ts keep working.
 */
export const isLinear = isLinearGeometry;

export interface VibResult {
  /** All 3N wavenumbers (cm^-1), ascending; includes the near-zero modes. */
  allWavenumbers: number[];
  /** The 3N-6 (or 3N-5) vibrational wavenumbers, ascending. Imaginary -> negative. */
  frequencies: number[];
  /** True if the molecule was detected linear (5 trans/rot modes removed). */
  linear: boolean;
  /** Largest magnitude of the six/five removed (should-be-zero) modes, cm^-1. */
  maxResidualTransRot: number;
  /** Zero-point vibrational energy from the harmonic frequencies (kJ/mol). */
  zpeKJ: number;
  /** Vibrational thermal energy at 298.15 K, sum h*nu/(exp-1) (kJ/mol, excludes ZPE). */
  vibThermalKJ: number;
}

/** Full harmonic analysis at a (presumed stationary) geometry. */
export async function vibrationalAnalysis(
  mol: Molecule,
  provider: EnergyForceProvider,
  step = 1e-3,
): Promise<VibResult> {
  const nm = await computeNormalModes(mol, provider, { stepSize: step });

  // ZPE and vibrational thermal energy from positive (real) frequencies.
  let zpeKJ = 0;
  let vibThermalKJ = 0;
  for (const nu of nm.frequencies) {
    if (nu <= 0) continue; // ignore imaginary modes in thermochemistry
    zpeKJ += 0.5 * nu * CM1_TO_KJ_MOL;
    const x = (HC_OVER_K * nu) / T_STANDARD;
    vibThermalKJ += R_KJ * T_STANDARD * (x / (Math.exp(x) - 1));
  }

  return {
    allWavenumbers: nm.allWavenumbers,
    frequencies: nm.frequencies,
    linear: nm.isLinear,
    maxResidualTransRot: nm.maxResidualTransRot,
    zpeKJ,
    vibThermalKJ,
  };
}
