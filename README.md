# BrowserCompChem

**Live: https://goldwinxs.github.io/browsercompchem/** — works on desktop and phone, no install, no server, no account. Nothing leaves your browser tab.

**Validation: [computed vs. literature, side by side](docs/VALIDATION.md)** —
fresh engine numbers against cited experimental values (also linked in-app at
[/validation.html](https://goldwinxs.github.io/browsercompchem/validation.html)).

A fully client-side computational chemistry workbench — a browser-native
successor to tools like WebMO. Sketch or paste a molecule, optimize its
geometry with a machine-learned potential, animate its normal modes, simulate
its IR spectrum, and render its molecular orbitals as 3D isosurfaces — all of
it computed in the tab, most of it in a couple of seconds.

This is not chemistry-flavored 3D graphics. Every numerical claim in the app
is either validated against literature or explicitly labeled as qualitative,
and the checking apparatus is a permanent, versioned part of the repository.

## What it does

- **SMILES / structure sketching to 3D** — RDKit (WebAssembly) parses SMILES or
  a JSME-sketched structure and supplies the real bond topology. A built-in
  topology-aware 3D embedder (distance-geometry-style classical force field,
  multi-start with a validity gate) produces an untangled starting geometry
  for any size of molecule, which the ML potential then refines.
- **Geometry optimization** — a pure-TypeScript reimplementation of the
  **ANI-2x** neural network potential (elements H, C, N, O, F, S, Cl; neutral,
  closed-shell molecules) with analytic forces, relaxed by a from-scratch FIRE
  optimizer. Energies and forces are gated in CI against TorchANI, the
  reference implementation, at 1e-5 Ha / 1e-4 Ha/A on real molecules
  (cholesterol, a peptide, aspirin, caffeine).
- **Vibrations** — finite-difference Hessian on the ANI surface, harmonic
  frequencies with translation/rotation projected out, animated normal modes.
  Test-gated against gas-phase literature values for water and CO2.
- **IR spectra** — per-mode intensities from finite-differenced dipole
  derivatives, rendered as a Lorentzian-broadened spectrum. The symmetry
  selection rules come out exactly: CO2's symmetric stretch computes IR-silent
  while the antisymmetric stretch dominates, and that behavior is a hard test.
- **Molecular orbitals** — an extended Hückel tier (Hoffmann parameters,
  Wolfsberg–Helmholz, Löwdin orthogonalization) with marching-cubes isosurface
  rendering, full MO ladder, HOMO/LUMO frontier panel, Mulliken populations
  and partial charges. Benzene's doubly degenerate e1g HOMO and water's
  1b1 > 3a1 > 1b2 ordering reproduce as they should.
- **Measurement** — click through atoms for distances, angles, and dihedrals.
- **Rendering** — three.js with an optional real-time hybrid ray tracer
  ([three-realtime-rt](https://www.npmjs.com/package/three-realtime-rt)).
- **Phone-usable UI** — draggable bottom sheet, finger-sized controls.

## What it deliberately does not claim

The app labels its own accuracy tiers in the UI, and the same honesty applies
here:

- ANI-2x geometries and frequencies are **DFT-adjacent** (the network is
  trained to reproduce wB97X DFT; published benchmarks put it within roughly
  1–2 kcal/mol on its domain). They are not CCSD(T), and the domain is
  neutral closed-shell organics near equilibrium — no ions, radicals, metals,
  transition states, reactions, or solvent.
- Extended Hückel orbital **energies are qualitative**. Orderings,
  degeneracies, symmetries, and nodal structure are trustworthy; absolute eV
  are not (EHT systematically overbinds vs. photoelectron ionization energies
  by ~3.5–4 eV, and the UI says so).
- IR **intensities and the dipole are qualitative** (Mulliken point-charge
  model — the band pattern and symmetry zeros are right; absolute km/mol and
  Debye run high, and the UI says so).

`docs/LITERATURE_VALIDATION.md` collects the experimental reference values
(NIST CCCBDB / WebBook, primary PES and spectroscopy literature, with
citations) used to check the app, plus notes on how to validate this class of
method honestly.

## Architecture

```
                    +-------------------------------------------+
                    |            three.js scene                  |
                    |   (optional three-realtime-rt ray tracer)  |
                    +-------------------------------------------+
                                   ^
                                   |
    +--------------------+  +----------------------+  +----------------------+
    | TIER 1: TOPOLOGY   |  | TIER 2: ML ACCURACY  |  | TIER 3: ELECTRONS    |
    | RDKit-JS (WASM)    |  | ANI-2x, pure TS      |  | extended Hueckel     |
    | SMILES/sketch,     |  | energies + analytic  |  | MOs, isosurfaces,    |
    | bonds, embed3d     |  | forces, FIRE relax   |  | Mulliken, dipole, IR |
    +---------+----------+  +----------+-----------+  +----------+-----------+
              |                        |                         |
              +------------------------+-------------------------+
                                   |
                     +--------------------------------+
                     |        packages/engine          |
                     |  geometry, potentials (ANI-2x), |
                     |  optimize (FIRE), vibrations,   |
                     |  orbitals (EHT), spectra (IR),  |
                     |  embed3d, isosurface            |
                     +--------------------------------+
```

Everything reduces to one seam — `EnergyForceProvider.energyForces(molecule)`
— so the optimizer, Hessian, and benches are written against an interface,
not against any particular method. (An ONNX Runtime / WebGPU inference path
was prototyped and deliberately rejected; `spike/ani2x-onnx/RESULTS.md`
documents why the pure-TypeScript implementation won.)

## Verification

- `npm test -w packages/engine` — 52 tests: TorchANI parity gates, analytic
  gradients vs. Romberg finite differences, literature-anchored vibrational
  frequencies, EHT orbital orderings/degeneracies, IR selection rules,
  embedder geometry gates (benzene planar and hexagonal, acetylene linear,
  large molecules untangled), classical force-field gradient vs. finite
  difference.
- `npm test -w apps/demo` — UI-side unit tests.
- `npm run fuzz -w apps/demo` — a seeded state-fuzz bench that drives the
  running app through randomized interleaved user actions (switch molecules
  mid-computation, etc.) and asserts state invariants after every action.

## Repo layout

```
packages/
  engine/     core library: geometry, ANI-2x, FIRE, vibrations, spectra,
              orbitals, embed3d, isosurface
  bench/      benchmark harness: EnergyForceProvider + dataset -> error stats
apps/
  demo/       the Vite app deployed to GitHub Pages
models/       ANI-2x weights (git-tracked deliberately; see models/README.md)
bench-data/   curated literature reference data + citations
docs/         LITERATURE_VALIDATION.md — cited experimental reference values
spike/        archived ONNX-runtime exploration (kept for the negative result)
```

## Dev setup

Requires Node >= 22 (npm workspaces).

```bash
npm install
npm test             # engine + bench tests
npm run dev          # Vite dev server for apps/demo
```
