import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";

/**
 * Stub for a numerical (finite-difference) Hessian evaluator.
 *
 * NOT YET IMPLEMENTED. Intended design: central-difference the analytic
 * forces returned by an EnergyForceProvider with respect to each of the
 * 3N Cartesian coordinates,
 *
 *   H_ij = -(F_i(x_j + h) - F_i(x_j - h)) / (2h)
 *
 * requiring 6N energy/force evaluations for an N-atom system (2 per
 * coordinate). Downstream uses once implemented: harmonic vibrational
 * frequencies (mass-weight H, diagonalize, convert eigenvalues to
 * wavenumbers), transition-state curvature checks, and validating that
 * an EnergyForceProvider's analytic forces are self-consistent with its
 * energy (i.e. that d(energy)/dx matches the returned forces to within
 * the finite-difference step's truncation error) -- useful as a sanity
 * test for any new provider (ONNX ML potential, Hueckel, etc.) before
 * trusting it in the optimizer.
 */
export interface HessianOptions {
  /** Finite-difference step size per coordinate (same units as Molecule.positions). */
  stepSize?: number;
}

export interface HessianResult {
  /** Flattened (3N x 3N) Hessian, row-major: H[i*3N + j]. */
  hessian: Float64Array;
  n3: number;
}

export async function finiteDifferenceHessian(
  _mol: Molecule,
  _provider: EnergyForceProvider,
  _options: HessianOptions = {},
): Promise<HessianResult> {
  throw new Error("finiteDifferenceHessian is not implemented yet (see doc comment in finiteDifference.ts).");
}
