/** Shared ANI-2x provider + a tight FIRE relaxation helper for the accuracy suite. */
import {
  Ani2xProvider,
  FireOptimizer,
  type EnergyForceProvider,
  type Molecule,
} from "@browser-comp-chem/engine";
import { MODEL_DIR } from "./paths.js";

/** Force-tolerance (Ha/Angstrom) we require before trusting a Hessian / energy diff. */
export const TIGHT_FORCE_TOL = 1e-4;
/** Above this max force we treat a geometry as NOT a stationary point and flag it. */
export const STATIONARY_THRESHOLD = 1e-3;

let cached: Promise<Ani2xProvider> | undefined;

/** The most accurate variant (full-f32, 8-member ensemble), created once. */
export function getProvider(): Promise<Ani2xProvider> {
  if (!cached) {
    cached = Ani2xProvider.create({ modelDir: MODEL_DIR, variant: "full-f32" });
  }
  return cached;
}

export interface RelaxResult {
  molecule: Molecule;
  energy: number; // Hartree
  maxForce: number; // Ha/Angstrom
  converged: boolean;
  steps: number;
}

/** Relax to a tight stationary point; returns final energy + max force. */
export async function relax(
  mol: Molecule,
  provider: EnergyForceProvider,
  forceTolerance = TIGHT_FORCE_TOL,
  maxSteps = 3000,
): Promise<RelaxResult> {
  const fire = new FireOptimizer({ forceTolerance, maxSteps, dtMax: 0.2, maxStep: 0.1 });
  const r = await fire.optimize(mol, provider);
  return {
    molecule: r.molecule,
    energy: r.energy,
    maxForce: r.maxForce,
    converged: r.converged,
    steps: r.steps,
  };
}
