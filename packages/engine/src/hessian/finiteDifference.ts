import { cloneMolecule, type Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";

/**
 * Numerical (finite-difference) Cartesian Hessian from an EnergyForceProvider's
 * ANALYTIC forces.
 *
 * We central-difference the analytic forces with respect to each of the 3N
 * Cartesian coordinates:
 *
 *   H_ic = d^2E/dx_i dx_c = -dF_i/dx_c
 *        = -(F_i(x_c + h) - F_i(x_c - h)) / (2h)
 *
 * This costs 2 force evaluations per coordinate, i.e. 6N evaluations for an
 * N-atom system. Differentiating the analytic forces (rather than double-
 * differencing the energy) keeps one order of accuracy and halves the noise.
 * The result is symmetrized to remove the small asymmetry from finite-precision
 * forces.
 *
 * Units follow the provider: for ANI-2x, energies are Hartree and positions are
 * Angstrom, so the Hessian is Hartree/Angstrom^2. Downstream (see
 * ../vibrations) this is mass-weighted and diagonalized to give harmonic
 * wavenumbers, and is also useful for transition-state curvature checks and for
 * validating that a new provider's analytic forces are self-consistent.
 */
export interface HessianOptions {
  /** Finite-difference step size per coordinate (same units as Molecule.positions). */
  stepSize?: number;
  /**
   * Optional progress callback, invoked after each of the 3N coordinate columns
   * is finished (done counts up to n3). Lets a worker stream progress without
   * the Hessian code knowing anything about messaging.
   */
  onProgress?: (done: number, total: number) => void;
}

export interface HessianResult {
  /** Flattened (3N x 3N) Hessian, row-major: H[i*3N + j]. */
  hessian: Float64Array;
  n3: number;
}

const DEFAULT_STEP = 1e-3;

export async function finiteDifferenceHessian(
  mol: Molecule,
  provider: EnergyForceProvider,
  options: HessianOptions = {},
): Promise<HessianResult> {
  const step = options.stepSize ?? DEFAULT_STEP;
  const n3 = mol.positions.length;
  const H = new Float64Array(n3 * n3);
  const work = cloneMolecule(mol);

  for (let c = 0; c < n3; c++) {
    const x0 = mol.positions[c]!;
    work.positions[c] = x0 + step;
    const fPlus = (await provider.energyForces(work)).forces;
    work.positions[c] = x0 - step;
    const fMinus = (await provider.energyForces(work)).forces;
    work.positions[c] = x0;
    // H_ic = -dF_i/dx_c
    for (let i = 0; i < n3; i++) {
      H[i * n3 + c] = -(fPlus[i]! - fMinus[i]!) / (2 * step);
    }
    if (options.onProgress) options.onProgress(c + 1, n3);
  }

  // Symmetrize (finite-precision forces make H_ij and H_ji differ slightly).
  for (let i = 0; i < n3; i++) {
    for (let j = i + 1; j < n3; j++) {
      const avg = 0.5 * (H[i * n3 + j]! + H[j * n3 + i]!);
      H[i * n3 + j] = avg;
      H[j * n3 + i] = avg;
    }
  }

  return { hessian: H, n3 };
}
