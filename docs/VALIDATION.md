# Validation: computed vs. literature

Every number in the "computed" columns below was produced by the code in this
repository (engine build of 2026-07-23, ANI-2x `full-f32` variant) in a single
fresh run — the same code paths the deployed app executes in your browser. The
"literature" columns are experimental gas-phase values with their sources;
`docs/LITERATURE_VALIDATION.md` carries the fuller citation list and notes.
The same comparisons run continuously as hard gates in the test suite
(`npm test -w packages/engine`, 52 tests).

The one-paragraph honest summary: **geometries and vibrational frequencies are
quantitative** (the ML potential reproduces its DFT training target closely);
**orbital energies, IR intensities, and dipoles are qualitative and labeled as
such in the app** — orderings, degeneracies, symmetry selection rules, and
band patterns are trustworthy; their absolute magnitudes are not.

## 1. Potential parity: this implementation vs. TorchANI

Reference energies/forces were generated independently with TorchANI ANI-2x
(PyTorch, float64, autograd forces — `spike/ani2x-onnx/gen_references.py`) and
baked into the test fixtures. Freshly computed values from this engine:

| Molecule | Atoms | ΔE vs TorchANI (Ha) | max Δforce (Ha/Å) |
|---|---|---|---|
| water | 3 | 0 (bit-identical) | 2.9e-10 |
| methane | 5 | 0 (bit-identical) | 9.2e-12 |
| aspirin | 21 | 0 (bit-identical) | 1.1e-9 |
| caffeine | 24 | 0 (bit-identical) | 6.3e-10 |
| cholesterol | 74 | 0 (bit-identical) | 7.1e-10 |
| peptide | 129 | 0 (bit-identical) | 8.5e-10 |

Energies agree to all 17 significant digits of float64; force components agree
to ~1e-9 (analytic adjoint here vs. autograd there — different differentiation
paths, so exact bit-agreement is not expected). This is what a faithful
reimplementation of the same architecture and weights looks like, as opposed
to an approximation of it. The CI gates are far looser (1e-5 Ha / 1e-4 Ha/Å);
measured agreement exceeds them by four-plus orders of magnitude.

ANI-2x itself is trained to reproduce ωB97X/6-31G(d) DFT (Devereux et al.,
*J. Chem. Theory Comput.* 2020, 16, 4192); published benchmarks put it within
roughly 1–2 kcal/mol of that target on its domain (neutral, closed-shell
H/C/N/O/F/S/Cl molecules near equilibrium). So "agrees with TorchANI" means
"delivers ANI-2x's published DFT-adjacent accuracy," no more, no less.

## 2. Equilibrium geometries (ANI-2x + FIRE, force tolerance 1e-4)

| Molecule | Quantity | Computed | Experiment | Source |
|---|---|---|---|---|
| water | r(O–H) | 0.9638 Å | 0.9578 Å | Hoy & Bunker, *J. Mol. Spectrosc.* 1979 |
| water | ∠(H–O–H) | 104.07° | 104.48° | same |
| methane | r(C–H) | 1.0924 Å | 1.087 Å | Hirota, *J. Mol. Spectrosc.* 1979 |
| methane | ∠(H–C–H), all six | 109.4712° | 109.471° (Td exact) | symmetry |
| acetylene | r(C≡C) | 1.2019 Å | 1.203 Å | Kuchitsu (ed.), Springer 1998 |
| acetylene | r(C–H) | 1.0683 Å | 1.063 Å | same |
| acetylene | ∠(H–C–C) | 179.99° | 180° (linear) | symmetry |
| CO2 | r(C=O) | 1.1650 Å | 1.162 Å | Herzberg 1966 |
| CO2 | ∠(O–C–O) | 180.00° | 180° (linear) | symmetry |
| benzene | r(C–C), all six | 1.3924 Å (spread 1.2e-5 Å) | 1.397 Å | Herzberg 1966 |
| benzene | r(C–H) | 1.0876 Å | 1.084 Å | same |

Bond lengths land within 0.003–0.006 Å of experiment; angles within ~0.4°;
symmetry-required linearity and Td/D6h equivalence emerge to numerical
precision rather than being imposed. (The benzene run starts from this
repository's own topology-based 3D embedder and relaxes to the hexagon.)

## 3. Harmonic vibrational frequencies

Computed values are **harmonic** wavenumbers; experimental band positions are
**anharmonic fundamentals**, which sit a few percent lower. That gap is
physics (anharmonicity), not error — DFT/ab initio harmonic frequencies carry
the same offset, which is why empirical scale factors exist.

Water:

| Mode | Computed (harmonic) | Exp. fundamental | Exp.-derived harmonic |
|---|---|---|---|
| bend | 1721 cm⁻¹ | 1595 cm⁻¹ | ~1649 cm⁻¹ |
| sym. stretch | 3815 cm⁻¹ | 3657 cm⁻¹ | ~3832 cm⁻¹ |
| antisym. stretch | 3935 cm⁻¹ | 3756 cm⁻¹ | ~3943 cm⁻¹ |

(Fundamentals: Shimanouchi, NSRDS-NBS 39. Against the experiment-derived
*harmonic* values the stretches agree to ~0.5%; the bend runs ~4% high.)

CO2 (computed linear, 4 modes, bend exactly doubly degenerate):

| Mode | Computed | Exp. fundamental | Note |
|---|---|---|---|
| bend (×2) | 645 cm⁻¹ | 667 cm⁻¹ | degenerate pair reproduced |
| sym. stretch | 1413 cm⁻¹ | ~1333 cm⁻¹ (deperturbed) | observed Raman is the 1285/1388 Fermi dyad |
| antisym. stretch | 2489 cm⁻¹ | 2349 cm⁻¹ | the strong IR band |

## 4. IR selection rules (the intensity model's hard test)

Intensities come from finite-differenced Mulliken-charge dipole derivatives —
a deliberately simple model whose *pattern* must still obey group theory:

| Molecule | Mode | Relative intensity | Group-theory requirement |
|---|---|---|---|
| CO2 | sym. stretch (1413) | **0.00 — inactive** | IR-forbidden (centrosymmetric, mutual exclusion) |
| CO2 | antisym. stretch (2489) | **100 — strongest** | strongly IR-allowed |
| CO2 | bend (645, ×2) | 5.6 each | IR-allowed |
| water | all three modes | 100 / 40.7 / 38.6 | all IR-allowed (C2v) |

The forbidden mode computes at the floating-point noise floor, not merely
"small." CO2's dipole computes 0.0004 D (zero by symmetry); water's computes
2.51 D vs. the experimental 1.85 D — Mulliken point charges overestimate
polarity, the app's UI says so, and the absolute km/mol scale inherits that
caveat. Trust the pattern and the zeros, not the absolute intensities.

## 5. Extended Hückel orbitals

| Check | Computed | Reference |
|---|---|---|
| benzene HOMO degeneracy (e1g pair) | split 1.6e-5 eV — degenerate | required by D6h symmetry |
| benzene EHT HOMO | −12.81 eV | PES vertical IE 9.244 eV → offset **3.56 eV** |
| caffeine EHT HOMO | −11.88 eV | PES vertical IE 7.95 eV (Dougherty et al. 1978) → offset **3.93 eV** |
| water frontier ordering | 1b1 above 3a1 (test-gated) | PES assignment (Turner et al.) |

The ~3.5–4 eV overbinding is the documented, *systematic* EHT offset — note
how consistent it is between two very different molecules. That consistency is
exactly why orderings, spacings, degeneracies, and orbital shapes are
trustworthy while absolute eV are not, and the app's UI states this rather
than presenting orbital energies as ionization energies. (The benzene EHT
HOMO–LUMO gap computes 4.52 eV; benzene's lowest singlet transition is
4.69 eV — suggestive, but EHT gaps should be treated as qualitative.)

## 6. What this validation does not cover

- Charged species, radicals, metals, transition states, reactions, solvent —
  outside ANI-2x's domain and not offered by the app.
- Absolute IR intensities and dipole magnitudes (see section 4).
- Absolute orbital energies (see section 5).
- Large-molecule *global* minima: the embedder + optimizer produce a good
  untangled local minimum, not a certified global conformer search.

## Reproducing these numbers

```bash
npm install
npm test -w packages/engine   # 52 tests, includes all gates above
```

The app itself (https://goldwinxs.github.io/browsercompchem/) runs the same
engine — load water, optimize, compute vibrations/IR/orbitals, and compare
against the tables above live.
