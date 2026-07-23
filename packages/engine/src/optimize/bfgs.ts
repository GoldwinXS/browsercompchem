import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";
import type { Optimizer, OptimizerOptions, OptimizeResult } from "./types.js";

/**
 * Skeleton for a quasi-Newton BFGS geometry optimizer.
 *
 * NOT YET IMPLEMENTED. Intended eventual design (standard damped/
 * bounded BFGS as used by ASE/most QM codes):
 *
 *  - Maintain an approximate inverse Hessian H (start as identity, or
 *    a cheap diagonal guess from e.g. a Lindh-style model Hessian).
 *  - Each step: proposed displacement = -H . g (g = -forces, i.e. the
 *    energy gradient), optionally trust-region/line-search clamped.
 *  - After moving, compute s (position step) and y (gradient change)
 *    and apply the BFGS inverse-Hessian update:
 *      H_{k+1} = (I - s y^T / (y^T s)) H_k (I - y s^T / (y^T s)) + s s^T / (y^T s)
 *  - Guard against y^T s <= 0 (non-convex region) by skipping the
 *    update or resetting H to identity that step.
 *
 * This is expected to converge in far fewer steps than FIRE near a
 * minimum (superlinear vs. FIRE's damped-MD linear convergence), at
 * the cost of an O(N^2) inverse-Hessian update/storage per step -- fine
 * for small/medium molecules, a real limitation for large ones (where
 * L-BFGS's limited-memory variant would be preferred instead).
 *
 * FIRE (fire.ts) is the fully working optimizer for now; this stub
 * exists so the Optimizer seam has a documented second implementation
 * to fill in without having to touch call sites later.
 */
export interface BfgsOptions extends OptimizerOptions {
  /** Initial guess for the inverse Hessian diagonal (identity * this). */
  initialInverseHessianScale?: number;
}

export class BfgsOptimizer implements Optimizer {
  readonly name = "bfgs";

  constructor(private readonly options: BfgsOptions = {}) {}

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async optimize(
    _mol: Molecule,
    _provider: EnergyForceProvider,
    _options?: OptimizerOptions,
  ): Promise<OptimizeResult> {
    throw new Error(
      "BfgsOptimizer is a skeleton (see doc comment in bfgs.ts) -- not implemented yet. Use FireOptimizer.",
    );
  }
}
