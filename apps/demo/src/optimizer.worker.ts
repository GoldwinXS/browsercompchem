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
  type EnergyForceProvider,
  type EnergyForces,
  type Molecule,
} from "@browser-comp-chem/engine";

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
      };

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
  } catch (err) {
    (self as unknown as Worker).postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
