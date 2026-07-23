# ANI-2x in the browser via onnxruntime-web — feasibility spike

Date: 2026-07-22. Machine: Windows 11, RTX 3060, Node 24, Python 3.11.
Stack: torch 2.13.0+cpu, torchani 2.8.4, onnx 1.22.0, onnxruntime 1.27 (Python),
onnxruntime-web 1.22.0 (browser).

## TL;DR recommendation

**AEV-in-JS is needed, and forces are the real problem, not the NN.** ANI-2x cannot
be exported to ONNX as a single end-to-end graph: the AEV featurizer's
neighbor/triple construction is a cascade of data-dependent ops
(`triu_indices`, `nonzero`, `unique_consecutive`, argumentless `repeat_interleave`,
data-dependent `tril_indices`) that the ONNX exporter cannot lower. The clean,
validated path is **Strategy (c): compute the AEV in JS and run only the
per-element neural networks in ONNX.** That pipeline is numerically exact
(energy vs TorchANI within 5e-8 Ha; forces within 3e-6 Ha/A).

The blocker for shipping ANI-2x "as-is" is **not** raw energy latency — a single
energy evaluation is 3–63 ms on the WASM backend across 3–129 atoms, which is
fine. The blocker is **forces**. The CELU derivative (`aten::elu_backward`)
prevents exporting an analytic gradient graph, so forces fall back to finite
differences = **6N energy calls per force evaluation**. That is 1.3 s for aspirin
and ~49 s for a 129-atom peptide, per force call — and a geometry optimization
(dozens of force calls) took **94 seconds for aspirin**. That does not scale.

**Verdict: not blocked, but "ship as-is" fails on forces.** To make this
production-viable, the single highest-value next step is **analytic forces**:
implement the AEV analytic gradient (dAEV/dr) in JS and contract it with dE/dAEV
(the latter obtainable either by a small custom ONNX symbolic for `elu_backward`,
or analytically in JS since the nets are just Linear+CELU). That converts a force
eval from 6N NN-inferences to ~1, i.e. a 100–800x speedup for the molecules here.
Separately, the 55 MB model should be shrunk (see below). WebGPU vs WASM could not
be measured in this environment (no GPU adapter, see caveat) and must be confirmed
on the real Chrome/RTX 3060 before finalizing the backend choice.

## Chosen export strategy and why

Attempts, in the order the plan prescribed:

- **(a) Full-model export (energy+forces, or energy-only), dynamic axes.**
  FAILED. Tracing energy-only reaches the AEV and dies on unsupported ops. After
  registering a custom symbolic for `aten::triu_indices` (constant-baked at the
  traced atom count), the next blocker is `aten::unique_consecutive`, and behind
  it `neighbors_to_triples()` uses argumentless `repeat_interleave`, a
  data-dependent `tril_indices(counts_max, ...)`, `nonzero`, and boolean masking.
  These are inherent to torchani's sparse neighbor/triple bookkeeping and do not
  lower to ONNX opset 17. Adding forces via an autograd wrapper failed even
  earlier ("Cannot insert a Tensor that requires grad as a constant") because the
  double-trace bakes AEV intermediates.
- **(b) Fixed-size padded export.** Not pursued once (c) proved clean; the same
  data-dependent triple ops would still need baking per-geometry, which is
  fragile and would specialize the graph to a fixed neighbor topology.
- **(c) Export only the per-element NNs; compute AEV in JS. CHOSEN.** The AEV is
  a fixed closed-form sum (radial + angular symmetry functions); the neighbor
  gymnastics exist only for sparse-compute efficiency and are unnecessary for a
  dense all-pairs implementation at these molecule sizes. The NNs are plain MLPs
  that export trivially and are WebGPU-friendly.

No maintained ANI-2x ONNX export exists (GitHub/PyPI searched; torchani's only
ONNX attempt, PR #329, was abandoned in 2020 on old-PyTorch limitations), so a
from-scratch export was required regardless.

### What was exported
- `ani2x_nn_energy.onnx` — inputs `aev[N,1008]` + `mask[N,7]` (one-hot species),
  output scalar = ensemble-mean NN energy (pre-self-energy). Runs all 7 element
  nets on all atoms and masks by species (dense, ONNX/WebGPU-friendly).
- `params.json` — AEV constants (radial eta=19.7, 16 shifts, Rc=5.1; angular
  eta=12.5, zeta=14.1, 8 shifts, 4 sections, Rc=3.5), species order
  (H,C,N,O,S,F,Cl), self-energies. Consumed by the JS AEV.
- `web/aev.js` — dense all-pairs AEV in plain JS, exact port of the validated
  numpy reference (`aev_ref.py`). Layout matches torchani: radial column =
  species(j)*16 + shift; angular column = triu_pair(si,sk)*32 + (shift*4+section).

The NN-gradient graph (`ani2x_nn_grad.onnx`) could NOT be exported:
`aten::elu_backward` (CELU derivative) is not in the opset-17 symbolic registry.
Hence finite-difference forces in the browser.

## Validation numbers

Reference = TorchANI ANI-2x, float64, autograd forces. All gates PASS.

### AEV (JS/numpy reference vs torchani AEVComputer)
| molecule | atoms | max abs dAEV |
|---|---|---|
| water | 3 | 1.1e-16 |
| methane | 5 | 4.4e-16 |
| aspirin | 21 | 6.7e-16 |
| caffeine | 24 | 8.9e-16 |
| cholesterol | 74 | 4.4e-15 |
| peptide | 129 | 2.2e-15 |

AEV is exact to machine precision. **Worst 4.4e-15.**

### Energy and forces (ONNX pipeline vs TorchANI)
| molecule | atoms | dE (ONNX f32 path) Ha | E_ref Ha | max dF (double FD) Ha/A |
|---|---|---|---|---|
| water | 3 | 2.1e-09 | -76.38822 | 1.5e-06 |
| methane | 5 | 1.7e-08 | -40.49942 | 2.1e-07 |
| aspirin | 21 | 6.3e-09 | -648.53128 | 2.8e-06 |
| caffeine | 24 | 1.1e-08 | -680.21285 | 2.5e-06 |
| cholesterol | 74 | 1.3e-09 | -1131.56004 | (n/a) |
| peptide | 129 | 4.6e-08 | -3013.90598 | (n/a) |

Energy gate (<1e-5 Ha): **PASS, worst 4.6e-8 Ha.**
Force gate (<1e-4 Ha/A): **PASS, worst 2.8e-6 Ha/A** (finite differences done in
double precision to isolate method + pipeline correctness from float32 noise;
browser forces use the same FD method in float32).
In-browser float32 WASM energies reproduce the reference to 5 decimals
(e.g. water -76.38822, aspirin -648.53128).

## Latency (browser)

Backend: **WASM only** (float32, onnxruntime-web 1.22.0, 4 threads,
cross-origin-isolated). Median of 60 energy calls after 10 warmup calls.
`aev_ms` is the JS featurizer portion of each energy call; the remainder is ONNX
NN inference. `force_eval` = one full gradient (finite differences, 6N energy
calls); measured (median of 3) where taken, else derived as 6N x energy.

| molecule | atoms | energy median (ms) | of which AEV-in-JS (ms) | FD force-eval (ms) |
|---|---|---|---|---|
| water | 3 | 2.76 | 0.005 | 46.6 (meas) / 49.7 (6N) |
| methane | 5 | 3.50 | 0.05 | 97.9 (meas) / 105 (6N) |
| aspirin | 21 | 11.27 | 0.78 | 1321 (meas) / 1420 (6N) |
| caffeine | 24 | 11.84 | 0.99 | 1720 (meas) / 1705 (6N) |
| cholesterol | 74 | 38.68 | 9.13 | 17172 (6N) |
| peptide | 129 | 63.11 | 11.96 | 48849 (6N) |

Notes:
- Measured force time tracks the 6N model closely (aspirin 1321 vs 1420 ms),
  confirming the per-force cost is dominated by NN inference calls, not the AEV.
- On WASM the **NN inference dominates** the energy call (aspirin: ~10.5 ms NN vs
  0.78 ms AEV; peptide: ~51 ms NN vs 12 ms AEV). This is because the exported
  graph runs all 7 element nets on every atom x 8 ensemble members = 56 MLP
  passes per call (about 7x more than necessary).
- The AEV-in-JS cost is small but grows O(N * neighbors^2) from the angular term
  (peptide 12 ms). It would become the bottleneck only if the NN were made ~10x
  cheaper (distillation / single ensemble member / per-atom gather).

### WebGPU — NOT MEASURED (environment limitation)
The benchmark requests a `webgpu` execution provider, but WebGPU latency could not
be obtained here: the preview browser (Electron 42 / Chromium 148) exposes
`navigator.gpu` yet `requestAdapter()` returns null (no GPU adapter surfaced by
this Electron build), and the browser extension had only a remote macOS Chrome
connected, not the local RTX 3060 machine. The harness auto-detects a working
adapter and will benchmark WebGPU automatically when run in a real Chrome on the
3060. **This must be done before choosing the backend.** Expectation: WebGPU
should accelerate the NN matmuls substantially for larger molecules, but the
workload is many tiny MLPs, so per-dispatch overhead may erode the gain for small
molecules; and it does not change the fundamental 6N-forces problem.

## Geometry optimization

Aspirin, perturbed by up to 0.15 A/atom, steepest descent (lr=0.02) with
finite-difference forces, WASM float32:

- **Converged in 66 steps** to max|F| = 0.0196 Ha/A (threshold 0.02).
- **Wall time: 94.0 s.** Energy -648.4137 -> -648.5318 Ha (dE = -0.118 Ha).
- ~8382 energy evaluations (66 force evals x 126 energy calls + overhead).

This is the clearest architectural signal: a trivial 21-atom relaxation takes a
minute and a half purely because each of 66 force evaluations costs 126 NN
inferences. Analytic forces would cut the ~8382 energy calls to ~66.

## Model file size
`ani2x_nn_energy.onnx` = **54,916,068 bytes (52.4 MiB)**, float32, all 8 ensemble
members x 7 element nets. Reduction options: single ensemble member (~7 MB, small
accuracy cost, loses QBC uncertainty), float16 weights (~26 MB), or per-atom
species-gather instead of running all 7 nets (compute, not size). For the web a
sub-10 MB model is desirable.

## Concrete next steps (priority order)
1. **Analytic forces.** Implement dAEV/dr in JS (closed-form radial/angular
   derivatives) and dE/dAEV either via a custom `elu_backward` ONNX symbolic or
   in JS. This is the make-or-break item; it removes the 6N penalty.
2. **Shrink the NN** (single member or fp16) and gather per-atom by species to
   drop the 7x waste, targeting <10 MB and ~1-2 ms/energy for drug-sized mols.
3. **Measure WebGPU on the real RTX 3060 Chrome** with the existing harness to
   settle backend choice.
4. If (1) proves too costly to maintain, consider distilling ANI-2x into a model
   with an ONNX-exportable featurizer, or an equivariant MLIP with a native web
   force path.

## Files in this spike
- `export_full.py` / `onnx_symbolics.py` — the (a) full-export attempts + triu symbolic.
- `export_nn.py` — the chosen (c) export: NN energy ONNX + params.json (+ validates NN split).
- `aev_ref.py` — dense numpy AEV reference. `validate_aev.py` — AEV vs torchani.
- `validate_energy.py` — full energy (ONNX) + FD forces vs torchani. `validation_output.txt` — its output.
- `make_molecules.py` / `molecules.json` — test geometries (RDKit).
- `web/aev.js`, `web/bench.js`, `web/index.html` — browser benchmark. `server.mjs` — COOP/COEP static server.
- `ani2x_nn_energy.onnx`, `params.json`, `bench_results.json`.

---

## Wave two addendum — analytic forces (2026-07-22)

Wave two implemented analytic forces and integrated ANI-2x into `packages/engine`.
The spike's finite-difference forces (6N energy calls) are gone.

### Chosen dE/dAEV strategy: pure-TypeScript MLP forward+backward (option b)

Rather than fight the ONNX symbolic registry over `aten::elu_backward`, the
per-element MLPs are run in **plain float64 TypeScript with a hand-written
backward pass**. `CELU'(z, alpha) = z>0 ? 1 : exp(z/alpha)` is elementary, so
`dE/dAEV` is exact. This choice also (a) removes the onnxruntime node-vs-web
split — one codepath runs in vitest and the browser; (b) makes per-atom species
gather trivial (each atom runs only its own element's net, not all 7); and
(c) makes single-member and fp16 variants fall out for free. There is **no ONNX
runtime and no WASM dependency** in the engine's ANI-2x path anymore.

Key numerical finding: **ANI-2x ships as float32**, and torchani's `.double()`
merely upcasts those exact f32 bits. The engine's `full-f32` variant stores the
same f32 weights and evaluates them in **float64**, so it reproduces the torchani
double-precision reference to machine precision — better than the spike's f32
ONNX-compute path.

`dAEV/dr` is the closed-form radial + angular derivative, contracted in reverse
mode against `g = dE/dAEV` (never materialising the [N,1008,N,3] Jacobian). The
0.95 cosine damping is carried through the theta derivative exactly, so it is
non-singular at the collinear limit.

### Validation (fresh TorchANI ANI-2x f64 references; gen_references.py)

Engine analytic pipeline vs TorchANI (energy gate <1e-5 Ha, force gate <1e-4 Ha/A):

| variant | members | worst dE (Ha) | worst maxAbs dF (Ha/A) | energy gate | force gate |
|---|---|---|---|---|---|
| full-f32  | 8 | ~0 (machine) | 1.1e-9 | PASS | PASS |
| full-f16  | 8 | 7.6e-5 | 2.2e-5 | miss (large mols) | PASS |
| single-f32| 1 | 4.1e-3 | 8.2e-3 | miss | miss |
| single-f16| 1 | 4.4e-3 | 8.2e-3 | miss | miss |

Only **full-f32** clears both strict gates (it is effectively exact). full-f16
passes forces but its energy drifts to ~7.6e-5 Ha on cholesterol/peptide.
Single-member deviations (~1e-3 Ha, ~8e-3 Ha/A) are the **ensemble spread**, not
error — they are the query-by-committee uncertainty and the natural hook for a
future uncertainty estimate (TODO: expose per-atom ensemble stddev). AEV adjoint
vs Romberg-extrapolated finite differences of the validated AEV: <1e-8 on water,
methane, aspirin, caffeine (water/methane/caffeine ~1e-11..1e-13; aspirin ~3e-10).

### Model sizes (models/ani2x/, compact binary + manifest.json)

| variant | file | bytes |
|---|---|---|
| full-f32   | weights-full-f32.bin   | 54,823,136 (52.3 MiB) |
| single-f32 | weights-single-f32.bin |  6,852,892 (6.5 MiB)  |
| full-f16   | weights-full-f16.bin   | 27,411,568 (26.1 MiB) |
| single-f16 | weights-single-f16.bin |  3,426,446 (3.3 MiB)  |

### Performance (pure-TS float64, warm V8 / Node 24; representative of browser V8)

Note: the NN path is now plain JS — there is no WASM/WebGPU backend to choose;
the same code runs in Chrome's V8. Median of 50 force-evals after 10 warmups.

| molecule | atoms | full-f32 force-eval (ms) | single-f32 force-eval (ms) | spike FD force-eval (ms) |
|---|---|---|---|---|
| water | 3 | 14.9 | 1.8 | 46.6 |
| methane | 5 | 26.9 | 3.4 | 97.9 |
| aspirin | 21 | 108 | 15.2 | 1321 |
| caffeine | 24 | 123 | 17.7 | 1720 |
| cholesterol | 74 | 425 | 77 | 17172 |
| peptide | 129 | 712 | 118 | 48849 |

Aspirin FIRE geometry optimization (17 steps, force tol 0.02 Ha/A):
- **full-f32: ~2.0 s** (E_final -648.5344 Ha) — vs the spike's **94 s** steepest
  descent (~47x). Meets the <3 s target.
- single-f32: ~0.29 s (E_final -648.5355 Ha).

FIRE reaches a marginally deeper minimum (~-648.5344) than the spike's fixed-lr
steepest-descent stop (-648.5318), as expected from a momentum optimizer.
The per-force-eval cost is dominated by the 1008-wide first matmul across the
8 ensemble members; single-member is ~8x cheaper at the documented accuracy cost.

### New / added files (wave two)
- `export_weights.py` — dumps the per-element MLP weights to models/ani2x/
  (manifest.json + weights-{full,single}-{f32,f16}.bin).
- `gen_references.py` — writes fresh TorchANI f64 energy+force references into
  packages/engine/test/fixtures/ani2x-references.json.
- Engine: `packages/engine/src/potentials/ani2x/{aev,mlp,model,provider,index}.ts`
  and tests `test/ani2x.test.ts`, `test/ani2x-aev-grad.test.ts`.
