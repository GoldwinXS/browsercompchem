/**
 * Geometry-optimization Web Worker.
 *
 * Runs the pure-TS ANI-2x EnergyForceProvider and the FIRE optimizer OFF the
 * main thread. The main thread only renders; this worker does all the heavy
 * numerics and streams per-step progress back so the UI never blocks.
 *
 * Protocol (main -> worker):
 *   { type: "init", modelDir, variant, members? }
 *   { type: "optimize", symbols, positions: number[], options }
 *
 * Protocol (worker -> main):
 *   { type: "ready", variant, members }
 *   { type: "step", step, energy, maxForce, positions: Float32Array }  (transferable)
 *   { type: "done", converged, energy, maxForce, steps, elapsedMs, positions: Float32Array }
 *   { type: "error", message }
 *
 * The trick that lets us reuse the engine's real FireOptimizer unchanged: we
 * wrap the provider so every energy/force evaluation (one per FIRE step, at the
 * current geometry) posts a {step,...} message. FIRE evaluates E/F at the live
 * geometry, checks convergence, then moves -- so the posted positions/energy are
 * a consistent (geometry, energy) pair, exactly what the renderer wants.
 */
import {
  Ani2xProvider,
  FireOptimizer,
  computeNormalModes,
  type EnergyForceProvider,
  type EnergyForces,
  type Molecule,
} from "@browser-comp-chem/engine";

/** Above this max force we treat a geometry as NOT a stationary point and relax it first. */
const STATIONARY_THRESHOLD = 1e-3;

let provider: Ani2xProvider | undefined;

function maxAbs(a: Float64Array): number {
  let m = 0;
  for (let k = 0; k < a.length; k++) {
    const v = Math.abs(a[k]!);
    if (v > m) m = v;
  }
  return m;
}

/**
 * EnergyForceProvider that delegates to the real ANI-2x provider but posts a
 * per-step message on each evaluation. Counts evaluations as FIRE steps.
 */
class ReportingProvider implements EnergyForceProvider {
  readonly name = "ani2x-reporting";
  private step = 0;
  constructor(private readonly inner: Ani2xProvider) {}

  async energyForces(mol: Molecule): Promise<EnergyForces> {
    const res = await this.inner.energyForces(mol);
    const maxForce = maxAbs(res.forces);
    // Transferable snapshot of the geometry that produced this energy.
    const pos = new Float32Array(mol.positions.length);
    for (let i = 0; i < pos.length; i++) pos[i] = mol.positions[i]!;
    const msg = {
      type: "step" as const,
      step: this.step,
      energy: res.energy,
      maxForce,
      positions: pos,
    };
    (self as unknown as Worker).postMessage(msg, [pos.buffer]);
    this.step += 1;
    return res;
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const data = ev.data as
    | { type: "init"; modelDir: string; variant: string; members?: number }
    | {
        type: "optimize";
        symbols: string[];
        positions: number[];
        options: {
          forceTolerance: number;
          maxSteps: number;
          dtMax: number;
        };
      }
    | { type: "vibrations"; symbols: string[]; positions: number[] };

  try {
    if (data.type === "init") {
      provider = await Ani2xProvider.create(
        data.members === undefined
          ? { modelDir: data.modelDir, variant: data.variant }
          : { modelDir: data.modelDir, variant: data.variant, members: data.members },
      );
      (self as unknown as Worker).postMessage({
        type: "ready",
        variant: data.variant,
        members: provider.members,
      });
      return;
    }

    if (data.type === "optimize") {
      if (!provider) throw new Error("optimizer worker: model not initialized");
      const mol: Molecule = {
        symbols: data.symbols,
        positions: Float64Array.from(data.positions),
        charge: 0,
        multiplicity: 1,
      };
      const reporting = new ReportingProvider(provider);
      const fire = new FireOptimizer({
        forceTolerance: data.options.forceTolerance,
        maxSteps: data.options.maxSteps,
        dtMax: data.options.dtMax,
      });
      const t0 = performance.now();
      const result = await fire.optimize(mol, reporting);
      const elapsedMs = performance.now() - t0;
      const finalPos = new Float32Array(result.molecule.positions.length);
      for (let i = 0; i < finalPos.length; i++) {
        finalPos[i] = result.molecule.positions[i]!;
      }
      (self as unknown as Worker).postMessage(
        {
          type: "done",
          converged: result.converged,
          energy: result.energy,
          maxForce: result.maxForce,
          steps: result.steps,
          elapsedMs,
          positions: finalPos,
        },
        [finalPos.buffer],
      );
      return;
    }

    if (data.type === "vibrations") {
      if (!provider) throw new Error("optimizer worker: model not initialized");
      let mol: Molecule = {
        symbols: data.symbols,
        positions: Float64Array.from(data.positions),
        charge: 0,
        multiplicity: 1,
      };

      const t0 = performance.now();

      // Verify we're at (or near) a stationary point; the Hessian is only
      // meaningful there. If the forces are too large, relax first with FIRE.
      let maxForce = maxAbs((await provider.energyForces(mol)).forces);
      let relaxed = false;
      if (maxForce >= STATIONARY_THRESHOLD) {
        const fire = new FireOptimizer({
          forceTolerance: 1e-4,
          maxSteps: 2000,
          dtMax: 0.2,
          maxStep: 0.1,
        });
        const r = await fire.optimize(mol, provider);
        mol = r.molecule;
        maxForce = r.maxForce;
        relaxed = true;
      }

      // 6N analytic-force evaluations; stream Hessian-column progress so the UI
      // can show a bar. Never blocks the main thread (this is the worker).
      const n3 = mol.positions.length;
      const nm = await computeNormalModes(mol, provider, {
        onProgress: (done, total) => {
          (self as unknown as Worker).postMessage({
            type: "vib-progress",
            done,
            total,
          });
        },
      });

      // Pack the per-mode Cartesian displacement vectors into one flat buffer
      // (nModes * n3) for a single transferable, plus the equilibrium geometry
      // the animation oscillates around (adopts any further relaxation).
      const nModes = nm.modes.length;
      const flatModes = new Float64Array(nModes * n3);
      for (let m = 0; m < nModes; m++) flatModes.set(nm.modes[m]!, m * n3);
      const eqPos = new Float32Array(n3);
      for (let i = 0; i < n3; i++) eqPos[i] = mol.positions[i]!;

      (self as unknown as Worker).postMessage(
        {
          type: "vib-done",
          frequencies: nm.frequencies,
          modes: flatModes,
          nModes,
          n3,
          isLinear: nm.isLinear,
          maxResidualTransRot: nm.maxResidualTransRot,
          maxForce,
          relaxed,
          equilibrium: eqPos,
          elapsedMs: performance.now() - t0,
        },
        [flatModes.buffer, eqPos.buffer],
      );
      return;
    }
  } catch (err) {
    (self as unknown as Worker).postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
