import type { Molecule } from "../geometry/molecule.js";

/**
 * Result of a single energy+force evaluation.
 *
 * Units follow atomic-unit convention used across the engine's "accuracy
 * tier" (ONNX ML potentials, future Hueckel/HF): energy in Hartree,
 * forces in Hartree/Bohr, flattened the same way as Molecule.positions
 * (length 3*N, -dE/dx_i layout). Individual providers (e.g. a toy
 * Lennard-Jones potential used only for testing the optimizer) may use
 * their own arbitrary reduced units instead -- when they do, that must
 * be documented on the provider itself, since the optimizer is unit-
 * agnostic and just follows whatever the provider returns.
 */
export interface EnergyForces {
  energy: number;
  forces: Float64Array;
}

/**
 * Common interface every compute backend implements: RDKit-JS force
 * fields, ONNX ML interatomic potentials, and (eventually) ab initio
 * methods all reduce to "given a geometry, give me energy + forces".
 * This is the seam the optimizer, hessian, and bench packages are
 * written against, so any of them can be swapped in without touching
 * the rest of the stack.
 */
export interface EnergyForceProvider {
  /** Human-readable identifier, e.g. "lj-test", "ani2x-onnx-webgpu". */
  readonly name: string;

  /** Evaluate energy and forces at the given geometry. May be async
   * because real providers call into WASM/WebGPU/worker threads. */
  energyForces(mol: Molecule): Promise<EnergyForces>;
}
