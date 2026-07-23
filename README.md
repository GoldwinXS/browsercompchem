# BrowserCompChem

A fully client-side computational chemistry workbench -- a browser-native
successor to desktop tools like WebMO. There is no backend: sketching,
geometry optimization, force-field and ML-potential energy/force
evaluation, and orbital calculations all happen in the browser tab, using
WebAssembly (RDKit-JS) and WebGPU (ONNX Runtime Web) for the compute-heavy
parts, and a hybrid ray-traced three.js renderer for visualization. Nothing
here is a toy demo of "chemistry-flavored 3D graphics" -- the goal is real
energies and forces, checked against literature values at every stage,
with the checking apparatus (the bench package) treated as a first-class,
permanent part of the project rather than a one-off validation script.

The strategy is to build the compute engine as a standalone, dependency-light
npm package first, and let a demo UI and a public benchmark page consume it
as a library -- so the same code that proves out an optimizer or a new
potential in a unit test is exactly the code a user's browser tab runs.
Correctness work (does this method reproduce known dimer/cluster minima?
does this ML potential's energy error stay under 1 kcal/mol against a
CCSD(T) reference set?) is meant to stay visible and public indefinitely,
not be a milestone that gets deleted once the "real" product ships.

## Architecture: three compute tiers

```
                      +-------------------------------------------+
                      |                three.js scene              |
                      |     (three-realtime-rt hybrid ray tracer)  |
                      +-------------------------------------------+
                                     ^ molecule / trajectory
                                     |
      +------------------+  +------------------------+  +------------------------+
      |   TIER 1: INSTANT |  | TIER 2: ML ACCURACY    |  | TIER 3: ORBITALS       |
      |   RDKit-JS (WASM) |  | onnxruntime-web/WebGPU |  | extended Hueckel (->HF)|
      |                   |  |                        |  |                       |
      | sketch -> 3D      |  | ML interatomic         |  | frontier orbitals,    |
      | conformer embed   |  | potentials (energy +   |  | HOMO/LUMO, qualitative|
      | MMFF/UFF force    |  | forces), e.g. ANI-2x-  |  | electronic structure  |
      | field relax       |  | style networks         |  |                       |
      +--------+----------+  +-----------+------------+  +-----------+------------+
               |                          |                           |
               +--------------------------+---------------------------+
                                     |
                       +-------------------------------+
                       |      packages/engine           |
                       |  geometry (Molecule)            |
                       |  potentials (EnergyForceProvider)|
                       |  optimize   (FIRE, BFGS*)        |
                       |  hessian    (finite difference*) |
                       |  orbitals   (extended Hueckel*)  |
                       +-------------------------------+
                                     |
                       +-------------------------------+
                       |      packages/bench             |
                       |  dataset schema + citations      |
                       |  MAE / RMSE / max error stats     |
                       |  runs any tier against literature |
                       +-------------------------------+

  (* = documented skeleton in this scaffold, not yet implemented; FIRE +
       Lennard-Jones are the one real, tested, end-to-end path today)
```

Every tier reduces to the same seam: `EnergyForceProvider.energyForces(molecule)
-> {energy, forces}`. The optimizer, hessian, and bench packages are written
against that interface, not against RDKit, ONNX, or any particular method --
so a new potential (a different ML model, a real Hueckel-derived tier, a
future ab initio method) plugs in without touching the rest of the stack.

## Roadmap

The project is staged deliberately from "provably correct" outward, rather
than "impressive demo" inward:

1. **Test bench** (current stage) -- prove the engine's primitives are
   correct against known analytic/literature results (LJ cluster minima
   today; RDKit-JS force-field energies, then ML-potential energies against
   a curated CCSD(T)/experimental reference set, next). The bench UI is a
   permanent public artifact, not a phase-1-only checkbox.
2. **Teaching tool** -- an interactive UI good enough to use in a classroom:
   sketch a molecule, watch it relax, see orbitals populate, compare force
   fields vs. ML potentials vs. (eventually) real ab initio side by side.
3. **Useful tool** -- fast enough, accurate enough, and featureful enough
   (conformer search, IR/Raman-adjacent vibrational analysis, reaction
   coordinate scans) that someone doing real work reaches for it instead of
   a desktop package.
4. **Infrastructure** -- the engine package is stable and general enough
   that other browser-based chemistry tools build on it rather than
   reimplementing the same primitives.

## Repo layout

```
packages/
  engine/   core library: geometry, potentials, optimize, hessian, orbitals
  bench/    benchmark harness: EnergyForceProvider + dataset -> error stats
apps/
  demo/     Vite app; renders a hardcoded caffeine molecule via three-realtime-rt
models/     ONNX model weights land here (not covered by .gitignore, see models/README.md)
spike/      exploratory work by a different agent -- not touched by this scaffold
bench-data/ curated literature reference datasets, being assembled separately --
            not present/read yet; packages/bench/README.md documents the schema
            it should eventually conform to
```

## Dev setup

Requires Node >= 22 (npm workspaces).

```bash
npm install          # installs all three workspace packages
npm test             # runs vitest in packages/engine and packages/bench
npm run build        # builds packages/engine and packages/bench to dist/
npm run dev          # starts the Vite dev server for apps/demo
```

`packages/engine`'s tests are the thing to trust first: a from-scratch FIRE
optimizer relaxing a Lennard-Jones dimer to its analytic minimum
(r = 2^(1/6) sigma) and a 7-atom LJ cluster to within a small tolerance of
the published Cambridge Cluster Database global minimum energy
(E = -16.505384 epsilon). Everything else in the engine (BFGS, finite-
difference Hessian, extended Hueckel) is presently a documented interface
stub -- see the doc comments in each module under `packages/engine/src/` for
the intended design before implementing it.
