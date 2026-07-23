# Computational Chemistry Accuracy Bench - Reference Data

Curated, source-traced literature reference data for a permanent, public, reproducible
accuracy benchmark comparing our browser compchem tool (ML potentials such as ANI-2x;
later Huckel/HF) against experimental and high-level ab initio literature values.

**Guiding principle: every datum carries its source, and accuracy/traceability beat
coverage.** Values that could not be corroborated were dropped or flagged, never guessed.

Curated 2026-07-22. Elements are constrained to ANI-2x coverage: **H, C, N, O, F, S, Cl**.

---

## Contents

| File | What it holds | Count |
|---|---|---|
| `thermochemistry.json` | Gas-phase standard enthalpies of formation (delta_f H, 298.15 K, kJ/mol) | 26 molecules |
| `frequencies.json` | Experimental fundamental vibrational frequencies (cm^-1) | 9 molecules |
| `conformers.json` | Relative conformer energies from high-level ab initio + experiment (kcal/mol) | 4 systems / 5 data pairs |
| `geometries/*.xyz` | 3D geometries (Angstrom) for every molecule referenced above | 29 files |
| `geometries/PROVENANCE.md` | Per-file geometry origin (experimental vs computed-MMFF) | - |
| `geometries/_generate.py` | Deterministic regenerator for the geometries | - |

---

## JSON schema

All three datasets share a common per-entry shape (fields present when known;
fields are omitted rather than filled with a guess when unknown):

```
{
  "id":          "methanol",           // stable slug, matches geometries/<id>.xyz
  "name":        "Methanol",
  "formula":     "CH4O",
  "smiles":      "CO",
  "cas":         "67-56-1",            // thermochemistry entries
  "value":       -201.0,               // the reference number
  "uncertainty": 0.6,                  // OMITTED (or null) when the source gives none - never invented
  "unit":        "kJ/mol",             // or "cm^-1" (freq, per-mode) or "kcal/mol" (conformers)
  "verification":"cross-checked",      // "cross-checked" | "single-source"  (see legend below)
  "source": {
    "database":  "NIST Chemistry WebBook (...)",   // or "paper" + "doi" for conformers
    "url":       "https://...",
    "retrieved": "2026-07-22"
  },
  "note":        "free-text caveats, alternate values, discrepancies"
}
```

Dataset-specific fields:

- **frequencies.json**: each entry has `point_group` and a `modes` array of
  `{mode, symmetry, frequency, description}`. `verification` is per-molecule.
- **conformers.json**: `reference_conformer`, `conformer`, `energy_type`
  (electronic vs ZPE-corrected vs effective-0K), `level_of_theory`, and a
  `source` block with `paper` + `doi`. A `deferred` array documents systems
  intentionally left out (with the reason) rather than filled with a shaky number.

### verification legend

- **cross-checked** - two or more independent experimental determinations (or
  CODATA + JANAF, or benchmark + focal-point analysis) agree within combined
  uncertainty.
- **single-source** - only one reliable determination was reachable this session,
  or sources disagree beyond their stated uncertainty (the `note` says which and
  gives the competing value).

---

## CAVEAT 1 - Enthalpy of formation is NOT a raw energy (thermochemistry.json)

An ML potential (ANI-2x) or an ab initio method returns a **total electronic energy**
(or, after atomization, an atomization energy). The stored numbers are **standard
enthalpies of formation** delta_f H(298.15 K) relative to elemental reference states
(C graphite, H2, N2, O2, F2, S rhombic, Cl2). **You cannot compare these two things
directly.** Doing so is the single most common way to get a meaningless benchmark.

To compare like-with-like, convert the model output into a delta_f H, or convert the
reference into an atomization energy. Recommended recipe (model -> delta_f H):

1. **Atomization energy at 0 K.** With the same model, compute
   `sum(E_atom) - E_molecule` at the optimized geometry to get the equilibrium
   atomization energy `D_e`. Subtract the molecule's **zero-point vibrational energy**
   (ZPE) to get `D_0` (the 0 K atomization energy). ZPE from a harmonic Hessian is
   usually scaled (~0.98-0.99 for ZPE) or taken anharmonically.
2. **Thermal correction 0 -> 298.15 K.** Add translational, rotational, and
   vibrational enthalpy contributions (and the constituent atoms' `H(298)-H(0)`)
   using the standard rigid-rotor / harmonic-oscillator partition functions, or a
   published `H(298)-H(0)` table.
3. **Add experimental atomic enthalpies of formation.** Use the well-established
   gas-phase atomic delta_f H (e.g. C 716.87, H 217.998, N 472.44, O 249.229,
   F 79.38, S 277.17, Cl 121.301 kJ/mol at 298.15 K; CODATA/ATcT). Combine per the
   Hess-cycle to arrive at the molecule's delta_f H(298).

Pitfalls: (a) mixing 0 K and 298 K quantities; (b) forgetting ZPE entirely;
(c) using an inconsistent ZPE scale factor; (d) elemental reference-state confusion
(graphite vs gas-phase C). The **raw experimental values are what we store**; the
conversion machinery lives in the bench harness, not here.

---

## CAVEAT 2 - Harmonic vs fundamental frequencies (frequencies.json)

The stored frequencies are **experimental fundamentals** (anharmonic, observed band
origins). A normal-mode calculation from a Hessian gives **harmonic** frequencies
omega_e, which are systematically **higher** than observed fundamentals nu (roughly
3-6% for X-H stretches, less for bends). Comparing raw harmonic omega to these
fundamentals will show a fake ~5% "error". Compare like-with-like by either:

- applying a method-appropriate harmonic-to-fundamental **scaling factor**
  (see CCCBDB vibrational scaling tables, https://cccbdb.nist.gov/vibscalejust.asp), or
- computing **anharmonic** frequencies (e.g. VPT2) and comparing to these fundamentals.

Extra care: **Fermi resonances** (notably CO2 nu1 / 2nu2, stored as the deperturbed
fundamental 1333 cm^-1 but observed as a dyad ~1388/1285) and large-amplitude motions
(methanol OH torsion at ~200 cm^-1; ammonia umbrella inversion doubling) are not well
described by a single harmonic number - flag these modes when scoring.

---

## CAVEAT 3 - What each conformer energy includes (conformers.json)

Relative conformer energies mix conventions across the literature: some are pure
**electronic** (Born-Oppenheimer, no ZPE) CCSD(T)/CBS differences; others are
**ZPE-corrected** or effective 0 K / experimental values. Each entry states this in
`energy_type`. To benchmark: relax **both** conformers with the model and take the
**electronic** difference to compare against an electronic reference; add ZPE (Hessian)
before comparing against a ZPE-corrected or experimental reference. Sub-kcal orderings
(glycine II/III, cysteine II/III) are genuine stress tests and are level-sensitive.

---

## Geometries

One XYZ per molecule (Angstrom). Two provenance classes, recorded in each file's
comment line and in `geometries/PROVENANCE.md`:

- **[experimental]** - exact structures built from published spectroscopic parameters
  (CCCBDB experimental geometry section / Herzberg): water, ammonia, methane,
  hydrogen sulfide, CO, CO2, HCN, acetylene, ethylene, formaldehyde. These are the
  small rigid molecules (and the core of the frequency set), where geometry accuracy
  matters most.
- **[computed-mmff]** - RDKit ETKDGv3 + MMFF94 (UFF fallback) from SMILES. A reasonable
  low-energy structure, **NOT experimental** and **NOT the exact ab initio benchmark
  conformer**. Intended as a starting geometry; a production bench should re-optimize
  with the target method. For the conformer systems the specific rotamer is only
  approximate at the force-field level (n-butane is set to the anti/180-deg rotamer;
  glycine/cysteine are labeled conf I but are MMFF minima, not the CCSD(T) structures).

Regenerate deterministically: `cd geometries && python _generate.py` (fixed seed).

**Recommended upgrade:** replace the computed-MMFF files with CCCBDB experimental
equilibrium geometries (or your own high-level optimized structures) before treating
geometry RMSD as a bench metric.

---

## Verification summary (curator's honest accounting)

- **thermochemistry.json** - 26 molecules; **23 cross-checked**, **3 single-source**
  (acetaldehyde, HCN, fluoromethane - each flagged in its `note` with the competing
  value and a "verify before hard use" warning). One value deliberately overridden:
  formaldehyde uses the modern -108.6 (Fletcher & Pilcher / ATcT), not the superseded
  NIST-JANAF -115.9.
- **frequencies.json** - 9 molecules, all **cross-checked** against both NIST CCCBDB
  (fetched) and the Shimanouchi NSRDS-NBS 39 compilation.
- **conformers.json** - 4 systems / 5 data pairs (butane, ethanol, glycine, cysteine
  x2), all **cross-checked** (ab initio + experiment or benchmark + focal-point).
  Alanine dipeptide C5/C7eq was **deferred** (documented in the `deferred` array):
  no exact CCSD(T)/CBS value was reachable from a non-paywalled primary source this
  session, and it was not worth guessing.

### Sources actually reachable this session

- **NIST Chemistry WebBook** (webbook.nist.gov) - fully fetchable; primary source for
  thermochemistry. Used the gas-phase enthalpy pages (`Mask=1`).
- **NIST CCCBDB** (cccbdb.nist.gov `exp2x.asp`) - fully fetchable; primary source for
  frequencies, and independently corroborated the methanol delta_f H (-201.00 +/- 0.60).
- **ATcT species pages** (atct.anl.gov) - **returned HTTP 403** to the fetcher; ATcT
  values are cited from well-established published numbers, not fresh fetches.
- **Publisher sites** (ACS, RSC, ScienceDirect) - mostly paywalled/403; conformer
  values came from arXiv preprints, PMC open-access mirrors, and cross-checked search
  summaries, with DOIs recorded for the peer-reviewed version of record.

### Values dropped or overridden for failing verification

- **Alanine dipeptide C5/C7eq** - dropped (deferred); level-dependent, no corroborated
  primary value reachable.
- **Formaldehyde -115.9 (NIST-JANAF)** - rejected as superseded; used -108.6.
- **Acetaldehyde** - kept the fetched -170.7 (Wiberg) but flagged single-source because
  older compilations/ATcT give ~-166; a ~4-5 kJ/mol genuine literature discrepancy.
- **Indirect liquid+vaporization estimates** for furan (-27.7, -29.8) and thiophene
  (116-218) - rejected in favor of direct combustion calorimetry.

---

## Recommended NEXT datasets to add

Prioritized, with URLs:

1. **GMTKN55** - large, curated main-group thermochemistry/kinetics/noncovalent
   benchmark; mine the organic subsets (e.g. ACONF alkane conformers, BH76 barriers,
   G21IP/EA). https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn/gmtkn55
2. **ANI-2x held-out / COMP6v2 test data** - the natural in-distribution accuracy check
   for ANI-2x itself (energies + forces). https://github.com/aiqm/ANI1x_datasets and
   https://github.com/isayev/COMP6
3. **TorsionNet500 / torsion benchmarks** - high-quality QM torsion profiles; directly
   tests the conformational energetics that matter for our tool.
   https://github.com/PfizerRD/TorsionNet
4. **ATcT full release** (once fetchable) - replace/upgrade the single-source
   thermochemistry entries (acetaldehyde, HCN, fluoromethane) with ATcT provenanced
   values + covariance. https://atct.anl.gov
5. **CCCBDB experimental geometries** - swap the computed-MMFF geometries for
   experimental equilibrium structures. https://cccbdb.nist.gov
6. **Alanine dipeptide** high-level conformer set (Kang & Park CPL 2014
   doi:10.1016/j.cplett.2014.03.038; Beachy et al. JACS 1997 doi:10.1021/ja962310g) -
   finish the deferred conformer entry.
7. **W4-11 / W4-17 atomization energies** - sub-kJ/mol atomization-energy references,
   the cleanest possible test of the enthalpy-of-formation pathway.
   (Karton et al.; see the Weizmann thermochemistry benchmarks.)
8. **Sulfur/halogen extensions** - more S and Cl/F organics (thiols, sulfides,
   chloro/fluoro-organics) to stress the less-covered corners of ANI-2x element space.
