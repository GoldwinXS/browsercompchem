import { cloneMolecule, type Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";
import type { Optimizer, OptimizerOptions, OptimizeResult, OptimizeStep } from "./types.js";

/**
 * FIRE (Fast Inertial Relaxation Engine) geometry optimizer.
 *
 * Reference: Bitzek, Koskinen, Gaehler, Moseler, Gumbsch,
 * "Structural Relaxation Made Simple", Phys. Rev. Lett. 97, 170201 (2006).
 * https://doi.org/10.1103/PhysRevLett.97.170201
 *
 * This is a from-scratch reimplementation that follows the widely-used
 * ASE (`ase.optimize.fire.FIRE`) formulation of the algorithm (semi-
 * implicit Euler MD step + adaptive velocity mixing/timestep), operating
 * on unit-mass point particles over a flat position buffer. It is the
 * one *fully working* optimizer in this package (BFGS is a skeleton --
 * see bfgs.ts); it is deliberately simple and dependency-free so that
 * the LJ-dimer / LJ7-cluster tests in test/fire.test.ts exercise real,
 * from-scratch code rather than a stub.
 */
export interface FireOptions extends OptimizerOptions {
  /** Initial (and minimum) integration timestep. */
  dtStart?: number;
  /** Upper bound the adaptive timestep is allowed to grow to. */
  dtMax?: number;
  /** Cap on per-atom displacement per step (prevents blow-up on steep repulsive walls). */
  maxStep?: number;
  /** Steps of "downhill" progress required before the timestep is allowed to grow. */
  nMin?: number;
  finc?: number;
  fdec?: number;
  alphaStart?: number;
  falpha?: number;
}

const DEFAULTS: Required<FireOptions> = {
  forceTolerance: 1e-6,
  maxSteps: 2000,
  dtStart: 0.1,
  dtMax: 1.0,
  maxStep: 0.2,
  nMin: 5,
  finc: 1.1,
  fdec: 0.5,
  alphaStart: 0.1,
  falpha: 0.99,
};

function maxAbs(arr: Float64Array): number {
  let m = 0;
  for (let k = 0; k < arr.length; k++) {
    const a = Math.abs(arr[k]!);
    if (a > m) m = a;
  }
  return m;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k]! * b[k]!;
  return s;
}

function norm(a: Float64Array): number {
  return Math.sqrt(dot(a, a));
}

export class FireOptimizer implements Optimizer {
  readonly name = "fire";
  private readonly opts: Required<FireOptions>;

  constructor(options: FireOptions = {}) {
    this.opts = { ...DEFAULTS, ...options };
  }

  async optimize(
    mol: Molecule,
    provider: EnergyForceProvider,
    options?: OptimizerOptions,
  ): Promise<OptimizeResult> {
    const opts: Required<FireOptions> = { ...this.opts, ...options };
    const working = cloneMolecule(mol);
    const n3 = working.positions.length;

    let v: Float64Array = new Float64Array(n3); // velocities, unit mass
    let dt = opts.dtStart;
    let alpha = opts.alphaStart;
    let nPos = 0;

    const history: OptimizeStep[] = [];
    let energy = NaN;
    let forces: Float64Array = new Float64Array(n3);
    let maxForce = Infinity;
    let converged = false;
    let step = 0;

    for (step = 0; step < opts.maxSteps; step++) {
      const result = await provider.energyForces(working);
      energy = result.energy;
      forces = result.forces;
      maxForce = maxAbs(forces);

      history.push({ step, energy, maxForce });

      if (maxForce < opts.forceTolerance) {
        converged = true;
        break;
      }

      // --- FIRE velocity mixing / adaptive dt (ASE formulation) ---
      const power = dot(forces, v);
      if (power > 0) {
        const fNorm = norm(forces);
        const vNorm = norm(v);
        if (fNorm > 0) {
          for (let k = 0; k < n3; k++) {
            v[k] = (1 - alpha) * v[k]! + alpha * (forces[k]! / fNorm) * vNorm;
          }
        }
        if (nPos > opts.nMin) {
          dt = Math.min(dt * opts.finc, opts.dtMax);
          alpha *= opts.falpha;
        }
        nPos += 1;
      } else {
        v = new Float64Array(n3);
        alpha = opts.alphaStart;
        dt *= opts.fdec;
        nPos = 0;
      }

      // --- semi-implicit Euler MD step ---
      for (let k = 0; k < n3; k++) {
        v[k] = v[k]! + dt * forces[k]!;
      }
      const dr = new Float64Array(n3);
      for (let k = 0; k < n3; k++) dr[k] = dt * v[k]!;

      // clip per-atom displacement to maxStep
      let maxAtomDisp = 0;
      const nAtoms = n3 / 3;
      for (let i = 0; i < nAtoms; i++) {
        const d = Math.sqrt(
          dr[3 * i]! ** 2 + dr[3 * i + 1]! ** 2 + dr[3 * i + 2]! ** 2,
        );
        if (d > maxAtomDisp) maxAtomDisp = d;
      }
      const scale = maxAtomDisp > opts.maxStep ? opts.maxStep / maxAtomDisp : 1;

      for (let k = 0; k < n3; k++) {
        working.positions[k] = working.positions[k]! + dr[k]! * scale;
      }
    }

    return {
      molecule: working,
      energy,
      maxForce,
      converged,
      steps: step,
      history,
    };
  }
}
