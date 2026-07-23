import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";

/** Convergence/behavior knobs shared by every geometry optimizer. */
export interface OptimizerOptions {
  /** Stop when max component of the force array drops below this. */
  forceTolerance?: number;
  /** Hard cap on the number of energy/force evaluations. */
  maxSteps?: number;
}

export interface OptimizeStep {
  step: number;
  energy: number;
  maxForce: number;
}

export interface OptimizeResult {
  /** Final relaxed geometry (a clone -- the input Molecule is not mutated in place). */
  molecule: Molecule;
  energy: number;
  maxForce: number;
  converged: boolean;
  steps: number;
  /** Per-step trace, useful for debugging/plotting convergence. */
  history: OptimizeStep[];
}

/** Common shape for every optimizer in this package (FIRE, BFGS, ...). */
export interface Optimizer {
  readonly name: string;
  optimize(
    mol: Molecule,
    provider: EnergyForceProvider,
    options?: OptimizerOptions,
  ): Promise<OptimizeResult>;
}
