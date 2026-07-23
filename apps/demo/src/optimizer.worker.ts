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

/** Covalent radii (Angstrom) for the geometry-validity gate in the embed search. */
const COVALENT: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  S: 1.05,
  Cl: 1.02,
  F: 0.57,
};
/** A real bond can be stretched at most this far past its covalent-radius sum. */
const BOND_STRETCH_SLACK = 0.7;
/** No two non-bonded atoms may sit closer than this (a clash = broken geometry). */
const CLASH_DISTANCE = 1.15;

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
 * Is a relaxed conformer geometrically sane given the known connectivity?
 * ANI-2x can score a fragmented/rearranged structure as low-energy, so the
 * multi-seed search must reject any result where a real bond stretched absurdly
 * far or two non-bonded atoms overlap. Returns the count of violations (0 = ok).
 */
function geometryViolations(
  symbols: string[],
  pos: Float64Array,
  bonds: { i: number; j: number }[],
): number {
  const dist = (i: number, j: number): number =>
    Math.hypot(pos[3 * i]! - pos[3 * j]!, pos[3 * i + 1]! - pos[3 * j + 1]!, pos[3 * i + 2]! - pos[3 * j + 2]!);
  let violations = 0;
  const bonded = new Set<number>();
  const n = symbols.length;
  for (const b of bonds) {
    bonded.add(b.i * n + b.j);
    bonded.add(b.j * n + b.i);
    const cutoff = (COVALENT[symbols[b.i]!] ?? 0.77) + (COVALENT[symbols[b.j]!] ?? 0.77) + BOND_STRETCH_SLACK;
    if (dist(b.i, b.j) > cutoff) violations++;
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (bonded.has(i * n + j)) continue;
      if (dist(i, j) < CLASH_DISTANCE) violations++;
    }
  }
  return violations;
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
    | {
        type: "embed";
        symbols: string[];
        /** several 3D starting geometries (flat [x,y,z,...] each) */
        seeds: number[][];
        /** RDKit connectivity, used to gate out geometrically-broken conformers */
        bonds: { i: number; j: number; order: number }[];
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

    if (data.type === "embed") {
      if (!provider) throw new Error("optimizer worker: model not initialized");
      if (data.seeds.length === 0) throw new Error("embed: no seeds provided");
      const reporting = new ReportingProvider(provider);
      const t0 = performance.now();
      interface Cand {
        energy: number;
        positions: Float64Array;
        maxForce: number;
        steps: number;
        converged: boolean;
        violations: number;
      }
      // Best geometrically-VALID conformer (violations 0) by energy; plus a
      // fallback = fewest violations, then energy, in case every seed is broken.
      let best: Cand | undefined; // lowest-energy valid
      let fallback: Cand | undefined; // best-effort if none valid
      const better = (a: Cand, b: Cand): boolean =>
        a.violations !== b.violations ? a.violations < b.violations : a.energy < b.energy;
      for (let s = 0; s < data.seeds.length; s++) {
        (self as unknown as Worker).postMessage({
          type: "embed-progress",
          seed: s,
          total: data.seeds.length,
          bestEnergy: best ? best.energy : null,
        });
        const mol: Molecule = {
          symbols: data.symbols,
          positions: Float64Array.from(data.seeds[s]!),
          charge: 0,
          multiplicity: 1,
        };
        const fire = new FireOptimizer({
          forceTolerance: data.options.forceTolerance,
          maxSteps: data.options.maxSteps,
          dtMax: data.options.dtMax,
        });
        // Stream each seed's descent (nice live animation of the search); the
        // reporting provider posts a {step,...} per evaluation.
        const r = await fire.optimize(mol, reporting);
        const cand: Cand = {
          energy: r.energy,
          positions: r.molecule.positions,
          maxForce: r.maxForce,
          steps: r.steps,
          converged: r.converged,
          violations: geometryViolations(data.symbols, r.molecule.positions, data.bonds),
        };
        if (cand.violations === 0 && (!best || cand.energy < best.energy)) best = cand;
        if (!fallback || better(cand, fallback)) fallback = cand;
      }
      const winner = best ?? fallback;
      if (!winner) throw new Error("embed: search produced no result");
      const elapsedMs = performance.now() - t0;
      const finalPos = new Float32Array(winner.positions.length);
      for (let i = 0; i < finalPos.length; i++) finalPos[i] = winner.positions[i]!;
      // Reuse the "done" protocol: the main thread adopts the winner as the
      // geometry and Perturb origin, regardless of which seed the streamed steps
      // ended on.
      (self as unknown as Worker).postMessage(
        {
          type: "done",
          converged: winner.converged,
          energy: winner.energy,
          maxForce: winner.maxForce,
          steps: winner.steps,
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
