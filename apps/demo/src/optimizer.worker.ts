/**
 * Geometry-optimization Web Worker.
 *
 * Runs the pure-TS ANI-2x EnergyForceProvider and the FIRE optimizer OFF the
 * main thread. The main thread only renders; this worker does all the heavy
 * numerics and streams per-step progress back so the UI never blocks.
 *
 * Protocol (main -> worker):
 *   { type: "init", modelDir, variant, members? }                      (epoch-independent)
 *   { type: "optimize", symbols, positions: number[], options, epoch }
 *   { type: "embed", symbols, seed, bonds, options, epoch }
 *   { type: "vibrations", symbols, positions: number[], epoch }
 *   { type: "orbitals", symbols, positions: number[], epoch }
 *   { type: "orbital-grid", orbitalIndex, isovalue, epoch }
 *   { type: "irspectrum", symbols, positions: number[], epoch }
 *   { type: "abandon", epoch }                      (no reply; see COOPERATIVE
 *                                                    CANCELLATION below)
 *
 * Protocol (worker -> main):
 *   { type: "ready", variant, members }                                (epoch-independent)
 *   { type: "step", step, energy, maxForce, positions, epoch }  (transferable)
 *   { type: "done", converged, energy, maxForce, steps, elapsedMs, positions, epoch }
 *   { type: "vib-progress" | "vib-done", ..., epoch }
 *   { type: "orb-done" | "orb-progress" | "orb-grid-done", ..., epoch }
 *   { type: "ir-progress" | "ir-done", ..., epoch }
 *   { type: "error", message, epoch? }         (epoch omitted only for "init" failures)
 *
 * GEOMETRY EPOCH. Every geometry-bearing request carries the main thread's
 * current geometry epoch; the worker echoes it on every reply for that request.
 * The main thread bumps the epoch whenever the geometry it is showing changes
 * (new molecule, perturb) and drops any reply whose epoch is stale -- so an
 * in-flight water solve can never repaint itself over methane. The worker also
 * tracks the highest epoch it has seen (`latestEpoch`) so a slow reply from an
 * abandoned geometry cannot overwrite the cached orbital solve a newer geometry
 * depends on. "init"/"ready" model-loading messages are epoch-independent.
 *
 * COOPERATIVE CANCELLATION. Dropping a stale reply on the main thread is not
 * enough: the ANI-2x provider's energyForces() is `async` but does no real I/O
 * (synchronous math wrapped in a promise), so an optimize/Hessian resolves
 * entirely through MICROtasks -- and a worker only dispatches `message` events as
 * MACROtasks. Without a real yield, `self.onmessage` cannot run again until the
 * current op has finished, so a molecule the user selects mid-computation sits
 * frozen at its raw seed until the ABANDONED op finishes grinding (measured:
 * 1219 ms -> 12805 ms for the same water optimize). Two pieces fix that:
 *   - CooperativeProvider wraps the provider for every op that evaluates
 *     energies/forces, yielding a real macrotask every YIELD_EVERY evaluations
 *     (so queued messages are dispatched) and throwing AbandonedGeometryError as
 *     soon as its epoch falls behind `latestEpoch`. That error is swallowed at
 *     the top level: an abandoned op posts nothing at all.
 *   - The main thread posts { type: "abandon", epoch } from resetForNewGeometry,
 *     because a geometry change is not always accompanied by a new request --
 *     switching to a built-in preset bumps the epoch and posts no compute at all,
 *     and `latestEpoch` would otherwise never learn the old op is pointless.
 * Suppressing a reply is always safe: the worker only bails when its epoch is
 * behind `latestEpoch`, and `latestEpoch` only ever holds an epoch the main
 * thread itself sent, so a suppressed reply is exactly one the main thread's
 * stale-reply gate would have dropped.
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
  extendedHuckel,
  evaluateOrbitalOnGrid,
  autoGridSpec,
  marchingCubes,
  gradientNormals,
  mullikenCharges,
  orbitalComposition,
  embed3d,
  computeIRSpectrum,
  type EnergyForceProvider,
  type EnergyForces,
  type Molecule,
  type ExtendedHuckelResult,
  type FFBond,
} from "@browser-comp-chem/engine";

/** Above this max force we treat a geometry as NOT a stationary point and relax it first. */
const STATIONARY_THRESHOLD = 1e-3;

let provider: Ani2xProvider | undefined;

/**
 * Highest geometry epoch seen across all incoming requests. Monotonic. Used to
 * decide whether a just-finished orbital solve still describes the geometry the
 * main thread is showing: a solve that started while it was current but whose
 * geometry has since been abandoned (epoch < latestEpoch by the time it lands)
 * must NOT overwrite the cache a newer solve depends on.
 */
let latestEpoch = -1;

/**
 * Most-recent Extended-Hueckel result + the Angstrom geometry it was computed
 * for + the epoch it belongs to. Kept so a follow-up "orbital-grid" request
 * (when the user clicks an MO) can evaluate psi on a grid without recomputing
 * the (fast) EHT solve or re-sending the coefficients across the worker
 * boundary. The EHT tier needs no ANI model, so these paths do not touch
 * `provider`. `lastOrbEpoch` guards the grid path against evaluating a cached
 * solve that belongs to a different geometry than the request.
 */
let lastOrbitals: ExtendedHuckelResult | undefined;
let lastOrbPositions: number[] = [];
let lastOrbEpoch = -1;

function maxAbs(a: Float64Array): number {
  let m = 0;
  for (let k = 0; k < a.length; k++) {
    const v = Math.abs(a[k]!);
    if (v > m) m = v;
  }
  return m;
}

/**
 * Energy/force evaluations between macrotask yields.
 *
 * The tradeoff: a yield is what lets a queued message (a new request, or an
 * "abandon") be dispatched at all, so it bounds worst-case cancellation latency
 * at YIELD_EVERY evaluations; but each yield costs an event-loop turn on top of
 * the evaluation. An ANI-2x evaluation is expensive next to a MessageChannel
 * bounce -- measured here: 18 ms/evaluation for water (3 atoms, 65 evaluations
 * per perturbed optimize), 102 ms/evaluation for naphthalene (18 atoms) -- so
 * the overhead is unmeasurable either way. From-idle wall time for the same
 * perturbed-water optimize (apps/demo/tools/interleave.mjs, two runs each):
 *   no yielding (pre-fix)  1286 ms
 *   YIELD_EVERY=1          1253 / 1151 ms
 *   YIELD_EVERY=4          1185 / 1184 ms   <- chosen; interleaved ratio 0.97x
 * All three sit inside the same ~100 ms run-to-run noise band. 4 is therefore
 * picked for margin, not speed: it keeps the per-yield cost under a fraction of
 * a percent even if evaluations ever get much cheaper, while still noticing an
 * abandonment within 4 evaluations (~70 ms on water, ~0.4 s on the 18-atom
 * worst case) -- fast enough that the interleave bench cannot distinguish an
 * optimize requested mid-abandonment from one requested at idle.
 */
const YIELD_EVERY = 4;

/**
 * One module-level MessageChannel used as an unclamped macrotask trampoline, with
 * a FIFO of pending resolvers (a channel per yield would allocate thousands of
 * ports per optimization). setTimeout(0) will NOT do: nested timers are clamped
 * to ~4 ms, which on a 6N-evaluation Hessian would add hundreds of milliseconds;
 * a MessageChannel bounce is ~0.1 ms and unclamped. Each postMessage on port2
 * delivers exactly one message to port1, which wakes exactly one waiter, so the
 * queue and the resolver list stay in lockstep even with two ops interleaved.
 */
const yieldChannel = new MessageChannel();
const yieldWaiters: (() => void)[] = [];
yieldChannel.port1.onmessage = () => {
  yieldWaiters.shift()?.();
};
function yieldToEventLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    yieldWaiters.push(resolve);
    yieldChannel.port2.postMessage(0);
  });
}

/**
 * Thrown by CooperativeProvider when the geometry its op belongs to has been
 * abandoned. Unwinds whatever engine code is running (FIRE, the finite-difference
 * Hessian) and is swallowed at the top level -- an abandoned op posts nothing.
 */
class AbandonedGeometryError extends Error {}

/**
 * EnergyForceProvider wrapper used by EVERY op that evaluates energies/forces.
 * It does three things the bare provider cannot:
 *   - yields a real macrotask every YIELD_EVERY evaluations, so messages queued
 *     for this worker actually get dispatched mid-op;
 *   - aborts the op (AbandonedGeometryError) as soon as its geometry epoch falls
 *     behind the newest epoch the worker has seen;
 *   - optionally posts the {type:"step"} progress message the UI animates from
 *     (optimize/embed want it; the Hessian paths report progress per force
 *     column instead and would only fight the renderer with per-evaluation
 *     geometry updates).
 * Evaluations are counted as FIRE steps, exactly as before.
 */
class CooperativeProvider implements EnergyForceProvider {
  readonly name = "ani2x-cooperative";
  private step = 0;
  private evaluations = 0;
  private readonly reportSteps: boolean;

  constructor(
    private readonly inner: EnergyForceProvider,
    private readonly epoch: number,
    opts: { reportSteps?: boolean } = {},
  ) {
    this.reportSteps = opts.reportSteps ?? false;
  }

  async energyForces(mol: Molecule): Promise<EnergyForces> {
    this.evaluations += 1;
    if (this.evaluations % YIELD_EVERY === 0) await yieldToEventLoop();
    // Checked AFTER the yield: the yield is what allows a newer request (or an
    // "abandon") to be dispatched and raise `latestEpoch` in the first place.
    if (this.epoch < latestEpoch) {
      throw new AbandonedGeometryError(
        `geometry epoch ${this.epoch} abandoned (latest ${latestEpoch})`,
      );
    }
    const res = await this.inner.energyForces(mol);
    if (this.reportSteps) {
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
        epoch: this.epoch,
      };
      (self as unknown as Worker).postMessage(msg, [pos.buffer]);
      this.step += 1;
    }
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
        epoch: number;
      }
    | {
        type: "embed";
        symbols: string[];
        /** RDKit's planarity-broken 2D display seed (flat [x,y,z,...]) -- used ONLY as
         * a fallback when no bond connectivity is available (embed3d needs a bond
         * graph to build its classical force field; with none, there is nothing for
         * it to do and we fall back to relaxing this seed directly with ANI-2x). */
        seed: number[];
        /** RDKit's real connectivity: the topology embed3d's classical force field is
         * built from, and that the geometry-validity gate checks against. Empty when
         * unavailable (e.g. a preset with no explicit bonds). */
        bonds: FFBond[];
        options: {
          forceTolerance: number;
          maxSteps: number;
          dtMax: number;
        };
        epoch: number;
      }
    | { type: "vibrations"; symbols: string[]; positions: number[]; epoch: number }
    | { type: "orbitals"; symbols: string[]; positions: number[]; epoch: number }
    | {
        type: "orbital-grid";
        orbitalIndex: number;
        isovalue: number;
        epoch: number;
      }
    | { type: "irspectrum"; symbols: string[]; positions: number[]; epoch: number }
    /** Pure staleness signal: the main thread replaced the geometry without
     * requesting anything new (preset switch, perturb). Raises `latestEpoch` so
     * an in-flight op can notice it has been abandoned; produces no reply. */
    | { type: "abandon"; epoch: number };

  // The geometry epoch of this request (undefined only for "init"). Echoed on
  // every reply and used in the catch so even an error respects staleness.
  const epoch = "epoch" in data ? data.epoch : undefined;
  if (epoch !== undefined && epoch > latestEpoch) latestEpoch = epoch;
  // "abandon" carries no work: recording the epoch above was the whole point.
  if (data.type === "abandon") return;

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
      const reporting = new CooperativeProvider(provider, data.epoch, { reportSteps: true });
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
          epoch: data.epoch,
        },
        [finalPos.buffer],
      );
      return;
    }

    if (data.type === "embed") {
      if (!provider) throw new Error("optimizer worker: model not initialized");
      const reporting = new CooperativeProvider(provider, data.epoch, { reportSteps: true });
      const t0 = performance.now();

      // Topology-aware classical pre-relax (see @browser-comp-chem/engine's
      // embed3d): several cheap random-3D-start relaxations of a harmonic-
      // bond/angle + soft-repulsion force field built from the KNOWN bond
      // graph, gated for validity, keeping the best untangled conformer. This
      // replaces "RDKit 2D coords + jitter" as the ANI-2x FIRE polish's
      // starting point for every molecule size -- no more large-molecule
      // single-seed branch, since embed3d's multi-start search is cheap
      // enough (no neural-network evaluations) to run unconditionally.
      // embed3d needs bonds to build its force field; with none available
      // (e.g. a preset with no explicit connectivity, or a single free atom)
      // there is nothing for it to do, so fall back to relaxing the plain
      // 2D+jitter display seed directly with ANI-2x -- the pre-hardening
      // behavior for that case.
      let seedPositions: Float64Array;
      if (data.bonds.length > 0 && data.symbols.length > 1) {
        const embedded = await embed3d(data.symbols, data.bonds);
        // embed3d is a multi-start classical relax with no progress hook, so it
        // offers CooperativeProvider no chance to notice an abandonment while it
        // runs. Check on the way out: with the geometry already replaced, the
        // ANI-2x polish below would be pure waste.
        if (data.epoch < latestEpoch) {
          throw new AbandonedGeometryError(
            `geometry epoch ${data.epoch} abandoned during embed3d (latest ${latestEpoch})`,
          );
        }
        seedPositions = embedded.positions;
        if (!embedded.valid) {
          // Every classical-relax attempt was flagged by the validity gate --
          // still hand the ANI-2x polish the least-bad one (it may yet relax
          // out of it) but note this for anyone watching the worker console.
          console.warn(
            `embed3d: no attempt passed the validity gate for ${data.symbols.length} atoms ` +
              `(best had ${embedded.violations} violation(s)); polishing it with ANI-2x anyway.`,
          );
        }
      } else {
        seedPositions = Float64Array.from(data.seed);
      }

      const fire = new FireOptimizer({
        forceTolerance: data.options.forceTolerance,
        maxSteps: data.options.maxSteps,
        dtMax: data.options.dtMax,
      });
      const mol: Molecule = { symbols: data.symbols, positions: seedPositions, charge: 0, multiplicity: 1 };
      // Stream the ANI-2x polish's descent (the reporting provider posts a
      // {step,...} per evaluation) so the UI still animates the relaxation live.
      const result = await fire.optimize(mol, reporting);

      const elapsedMs = performance.now() - t0;
      const finalPos = new Float32Array(result.molecule.positions.length);
      for (let i = 0; i < finalPos.length; i++) finalPos[i] = result.molecule.positions[i]!;
      // Reuse the "done" protocol: the main thread adopts the polished
      // geometry as the current geometry and Perturb origin.
      (self as unknown as Worker).postMessage(
        {
          type: "done",
          converged: result.converged,
          energy: result.energy,
          maxForce: result.maxForce,
          steps: result.steps,
          elapsedMs,
          positions: finalPos,
          epoch: data.epoch,
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

      // The longest-blocking op in the worker (a 6N-evaluation Hessian on top of
      // a full relax), so it is the one that most needs the cooperative wrapper.
      // reportSteps is off: the UI shows force-column progress for this op, and
      // per-evaluation geometry updates would fight the mode animation.
      const coop = new CooperativeProvider(provider, data.epoch);

      // Verify we're at (or near) a stationary point; the Hessian is only
      // meaningful there. If the forces are too large, relax first with FIRE.
      let maxForce = maxAbs((await coop.energyForces(mol)).forces);
      let relaxed = false;
      if (maxForce >= STATIONARY_THRESHOLD) {
        const fire = new FireOptimizer({
          forceTolerance: 1e-4,
          maxSteps: 2000,
          dtMax: 0.2,
          maxStep: 0.1,
        });
        const r = await fire.optimize(mol, coop);
        mol = r.molecule;
        maxForce = r.maxForce;
        relaxed = true;
      }

      // 6N analytic-force evaluations; stream Hessian-column progress so the UI
      // can show a bar. Never blocks the main thread (this is the worker).
      const n3 = mol.positions.length;
      const nm = await computeNormalModes(mol, coop, {
        onProgress: (done, total) => {
          (self as unknown as Worker).postMessage({
            type: "vib-progress",
            done,
            total,
            epoch: data.epoch,
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
          epoch: data.epoch,
        },
        [flatModes.buffer, eqPos.buffer],
      );
      return;
    }

    if (data.type === "irspectrum") {
      // Simulated IR spectrum: reuses computeNormalModes (same Hessian/relax
      // path as "vibrations" above -- deliberately duplicated rather than
      // shared across requests, since either op can be started independently
      // and each must produce a self-consistent (geometry, modes) pair for
      // its own reply), then finite-differences EHT-Mulliken dipoles along
      // each mode to get intensities (packages/engine/src/spectra/*). The EHT
      // solves are cheap and model-free; only the Hessian needs `provider`.
      if (!provider) throw new Error("optimizer worker: model not initialized");
      let mol: Molecule = {
        symbols: data.symbols,
        positions: Float64Array.from(data.positions),
        charge: 0,
        multiplicity: 1,
      };

      const t0 = performance.now();

      // Same Hessian cost as "vibrations" -- same cooperative wrapper, same
      // reason (see there); reportSteps off for the same reason too.
      const coop = new CooperativeProvider(provider, data.epoch);

      let maxForce = maxAbs((await coop.energyForces(mol)).forces);
      let relaxed = false;
      if (maxForce >= STATIONARY_THRESHOLD) {
        const fire = new FireOptimizer({
          forceTolerance: 1e-4,
          maxSteps: 2000,
          dtMax: 0.2,
          maxStep: 0.1,
        });
        const r = await fire.optimize(mol, coop);
        mol = r.molecule;
        maxForce = r.maxForce;
        relaxed = true;
      }

      const nm = await computeNormalModes(mol, coop, {
        onProgress: (done, total) => {
          (self as unknown as Worker).postMessage({
            type: "ir-progress",
            done,
            total,
            epoch: data.epoch,
          });
        },
      });

      const spec = await computeIRSpectrum(mol, nm.modes, nm.frequencies);

      const n3 = mol.positions.length;
      const eqPos = new Float32Array(n3);
      for (let i = 0; i < n3; i++) eqPos[i] = mol.positions[i]!;
      // curve arrays are plain Float64Array already -- transfer them directly.
      const wavenumbers = spec.curve.wavenumbers;
      const absorbance = spec.curve.absorbance;

      (self as unknown as Worker).postMessage(
        {
          type: "ir-done",
          modes: spec.modes,
          wavenumbers,
          absorbance,
          dipoleVector: spec.dipole.vector,
          dipoleMagnitude: spec.dipole.magnitude,
          isLinear: nm.isLinear,
          maxResidualTransRot: nm.maxResidualTransRot,
          maxForce,
          relaxed,
          equilibrium: eqPos,
          elapsedMs: performance.now() - t0,
          epoch: data.epoch,
        },
        [wavenumbers.buffer, absorbance.buffer, eqPos.buffer],
      );
      return;
    }

    if (data.type === "orbitals") {
      // Extended Hueckel MO solve. Fast and model-free (no ANI provider needed);
      // the coefficients + AO metadata are cached for follow-up grid requests.
      const t0 = performance.now();
      const mol: Molecule = {
        symbols: data.symbols,
        positions: Float64Array.from(data.positions),
        charge: 0,
        multiplicity: 1,
      };
      const res = await extendedHuckel(mol);
      // Only adopt this solve into the shared cache if it still describes the
      // current geometry. Two solves can be in flight (the main thread starts a
      // new one as soon as a geometry change clears the busy flag); the older
      // one must not clobber the cache the newer geometry's grid requests read.
      if (data.epoch >= latestEpoch) {
        lastOrbitals = res;
        lastOrbPositions = data.positions.slice();
        lastOrbEpoch = data.epoch;
      }

      // Mulliken partial charges (per atom) and, per MO, its dominant AO
      // contributors -- both cheap from the coefficients + overlap we already
      // have, and worth precomputing once so the UI (charge readout, per-orbital
      // composition, diagram tooltips) needs no extra round-trips.
      const { atomCharges } = mullikenCharges(res, data.symbols);
      const composition: { atomIndex: number; aoType: string; weight: number }[][] = [];
      for (let mo = 0; mo < res.nMO; mo++) {
        const contribs = orbitalComposition(res, mo)
          .filter((c) => Math.abs(c.weight) >= 0.03) // drop <3% noise
          .slice(0, 4)
          .map((c) => ({ atomIndex: c.atomIndex, aoType: c.aoType, weight: c.weight }));
        composition.push(contribs);
      }

      (self as unknown as Worker).postMessage({
        type: "orb-done",
        epoch: data.epoch,
        energies: Array.from(res.orbitalEnergies),
        occupations: Array.from(res.occupations),
        homoIndex: res.homoIndex,
        lumoIndex: res.lumoIndex,
        nBasis: res.nBasisFunctions,
        nMO: res.nMO,
        nElectrons: res.nElectrons,
        singular: res.singular,
        droppedCount: res.droppedCount,
        charges: Array.from(atomCharges),
        composition,
        elapsedMs: performance.now() - t0,
      });
      return;
    }

    if (data.type === "orbital-grid") {
      // A grid request from an abandoned geometry: ignore silently (the main
      // thread would drop the reply anyway, and evaluating it would waste work).
      if (data.epoch < latestEpoch) return;
      if (!lastOrbitals || lastOrbEpoch !== data.epoch) {
        throw new Error("orbital-grid: compute orbitals first");
      }
      const t0 = performance.now();
      // A grid that comfortably encloses the molecule (Angstrom world coords, so
      // the isosurface vertices land directly on the ball-and-stick model).
      const spec = autoGridSpec(Float64Array.from(lastOrbPositions), {
        padding: 3.0,
        spacing: 0.3,
        maxDim: 72,
      });
      const field = evaluateOrbitalOnGrid(lastOrbitals, data.orbitalIndex, spec, (done, total) => {
        (self as unknown as Worker).postMessage({ type: "orb-progress", epoch: data.epoch, done, total });
      });
      const sp: [number, number, number] =
        typeof spec.spacing === "number" ? [spec.spacing, spec.spacing, spec.spacing] : spec.spacing;
      const mcGrid = { dims: spec.dims, origin: spec.origin, spacing: sp };

      const iso = Math.abs(data.isovalue);
      // Positive lobe at +iso (outward normal toward decreasing field, sign -1);
      // negative lobe at -iso (outward normal toward increasing field, sign +1).
      const posMesh = marchingCubes(field, mcGrid, iso);
      const posNormals = gradientNormals(field, mcGrid, posMesh.positions, -1);
      const negMesh = marchingCubes(field, mcGrid, -iso);
      const negNormals = gradientNormals(field, mcGrid, negMesh.positions, +1);

      // Peak |psi| on the grid -- lets the UI keep the isovalue slider sensible.
      let maxAbsPsi = 0;
      for (let i = 0; i < field.length; i++) {
        const v = Math.abs(field[i]!);
        if (v > maxAbsPsi) maxAbsPsi = v;
      }

      (self as unknown as Worker).postMessage(
        {
          type: "orb-grid-done",
          epoch: data.epoch,
          orbitalIndex: data.orbitalIndex,
          isovalue: iso,
          maxAbsPsi,
          posPositions: posMesh.positions,
          posNormals,
          posIndices: posMesh.indices,
          posTriangles: posMesh.triangleCount,
          negPositions: negMesh.positions,
          negNormals,
          negIndices: negMesh.indices,
          negTriangles: negMesh.triangleCount,
          elapsedMs: performance.now() - t0,
        },
        [
          posMesh.positions.buffer,
          posNormals.buffer,
          posMesh.indices.buffer,
          negMesh.positions.buffer,
          negNormals.buffer,
          negMesh.indices.buffer,
        ],
      );
      return;
    }
  } catch (err) {
    // An abandoned op is an intentional bail-out, not a failure: the geometry it
    // was computing no longer exists on the main thread, which has already
    // cleared its own busy flags for it (resetForNewGeometry). Post NOTHING --
    // an "error" reply for a stale epoch would be dropped by the main thread's
    // gate anyway, and one for a live epoch would be a lie.
    if (err instanceof AbandonedGeometryError) return;
    (self as unknown as Worker).postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
      epoch, // undefined for "init" (always shown); stale op errors get dropped
    });
  }
};
