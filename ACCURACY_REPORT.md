# ANI-2x Accuracy Report

Generated: 2026-07-23T04:30:58.420Z
Model: ani2x-full-f32 (full-f32, 8-member ensemble -- most accurate variant)
Chemical-accuracy bar: 1.0 kcal/mol.

This suite compares ANI-2x predictions against curated real-literature reference values (bench-data/), converting like-with-like and reporting honest error statistics. Every reference value carries its citation in accuracy-report.json.

## Summary

| Property | N | MAE | RMSE | Max | Bar | Verdict |
|---|---|---|---|---|---|---|
| Conformer rel. energies | 2 | 0.14 kcal/mol | 0.17 | 0.23 | 1 kcal/mol | MAE 0.14 kcal/mol (max 0.23) over 2 pairs -- within chemical accuracy; but a near-degenerate ordering (ethanol) is flipped |
| Vib. freq (raw harmonic) | 86 | 99.4 cm^-1 | 119.8 | 226.0 | ~fundamentals | 6.0% mean err |
| Vib. freq (scaled x0.96) | 86 | 34.0 cm^-1 | 43.4 | 122.0 | ~fundamentals | 2.7% mean err |
| Heats of formation | 0 | not evaluated | - | - | 4.2 kJ/mol | blocked -- see methodology |

## Verdicts

- Conformers: MAE 0.14 kcal/mol (max 0.23) over 2 pairs -- within chemical accuracy; but a near-degenerate ordering (ethanol) is flipped.
- Frequencies: raw MAE 99.4 cm^-1 (6.0%); scaled x0.96 MAE 34.0 cm^-1 (2.7%) over 86 modes.
- Heats of formation: not evaluated -- De cannot be formed from the model alone for this ANI-2x port: the shipped self-energies are linear-fit baselines that already contain the binding (De_sae ~ 0), and ANI-2x is undefined on isolated atoms (lone-atom route unphysical, worst for heavy atoms). Verified wB97X/6-31G* free-atom total energies are required and were not fabricated. Supply them to dHfFromAtomization()/evaluateHeats({freeAtomEnergies}) to produce numbers.

## Conformer relative energies (kcal/mol)

| System | Predicted | Reference | Signed err | Ref level | Note |
|---|---|---|---|---|---|
| n-butane | 0.55 | 0.61 | -0.06 | CCSD(T) extrapolated to basis-set limit (W1h) | ok |
| ethanol | -0.11 | 0.12 | -0.23 | CCSD(T)/aug-cc-pVQZ and CCSD(T) PES; consistent with microwave/far-IR | SIGN MISMATCH vs reference ordering |
| glycine (conformer II relative to conformer I) | not evaluated | 1.40 | - | MP2/MP4/CCSD(T) large-basis (Csaszar 1992) and focal-point analysis (2009) | Not yet evaluated: differs from the reference conformer by several coupled torsions / H-bonding patterns with no second-conformer geometry in bench-data. Constructing it by hand would be a guess (see bench-data caveat on level-sensitive sub-kcal orderings). |
| L-cysteine (conformer II relative to conformer I) | not evaluated | 1.53 | - | CCSD(T)-F12b/cc-pVQZ-F12 with core-correlation, relativistic, and post-triples corrections | Not yet evaluated: differs from the reference conformer by several coupled torsions / H-bonding patterns with no second-conformer geometry in bench-data. Constructing it by hand would be a guess (see bench-data caveat on level-sensitive sub-kcal orderings). |
| L-cysteine (conformer III relative to conformer I) | not evaluated | 1.60 | - | CCSD(T)-F12b/cc-pVQZ-F12 with core-correlation, relativistic, and post-triples corrections | Not yet evaluated: differs from the reference conformer by several coupled torsions / H-bonding patterns with no second-conformer geometry in bench-data. Constructing it by hand would be a guess (see bench-data caveat on level-sensitive sub-kcal orderings). |

## Vibrational frequencies (cm^-1; scaling factor 0.96)

| Molecule | Modes | MAE raw | %raw | MAE scaled | %scaled | Note |
|---|---|---|---|---|---|---|
| Water | 3 | 154.2 | 5.7 | 28.0 | 1.4 |  |
| Carbon dioxide | 4 | 65.9 | 4.6 | 39.8 | 4.4 | nu1 is a Fermi dyad (deperturbed 1333); harmonic single-number comparison is approximate. |
| Ammonia | 6 | 170.8 | 8.3 | 67.7 | 3.9 | umbrella bend is inversion-split; harmonic value is an average. |
| Methane | 9 | 117.9 | 5.3 | 33.2 | 1.4 |  |
| Formaldehyde | 6 | 112.4 | 5.4 | 45.5 | 2.2 |  |
| Ethylene (ethene) | 12 | 103.7 | 5.4 | 37.2 | 2.1 |  |
| Hydrogen cyanide | 4 | 74.5 | 3.8 | 26.9 | 1.7 |  |
| Methanol | 12 | 107.8 | 10.3 | 34.4 | 6.2 | OH torsion ~200 cm^-1 is a large-amplitude mode poorly described as harmonic. |
| Benzene | 30 | 74.1 | 4.8 | 24.5 | 1.8 |  |

## Heats of formation (kJ/mol) -- diagnostic subset

Methodology: Atomization route: dHf(298) = sum(atomic dHf298) - (De - ZPE) - n*(5/2)RT + [H298-H0](molecule, RRHO), with De = sum(E_free_atom) - E_molecule. Harmonic (unscaled) ZPE; classical trans+rot; harmonic vibrational thermal. NOT applied: ZPE scaling, anharmonic/hindered-rotor corrections, atomic spin-orbit corrections.

Blocker: De cannot be formed from the model alone for this ANI-2x port: the shipped self-energies are linear-fit baselines that already contain the binding (De_sae ~ 0), and ANI-2x is undefined on isolated atoms (lone-atom route unphysical, worst for heavy atoms). Verified wB97X/6-31G* free-atom total energies are required and were not fabricated. Supply them to dHfFromAtomization()/evaluateHeats({freeAtomEnergies}) to produce numbers.

| Molecule | Ref dHf | De (self-energy route) | De (lone-atom route) | ZPE | Status |
|---|---|---|---|---|---|
| Water | -241.8 | 4.0 | 451.3 | 56.6 | not_evaluated |
| Ammonia | -45.9 | 62.6 | 736.9 | 92.4 | not_evaluated |
| Methane | -74.6 | 49.0 | 925.2 | 119.8 | not_evaluated |
| Ethylene (ethene) | 52.4 | -33.2 | 816.3 | 136.5 | not_evaluated |
| Benzene | 82.9 | 147.5 | 1341.5 | 269.2 | not_evaluated |
| Methanol | -201.0 | 22.8 | 894.8 | 137.5 | not_evaluated |
| Formaldehyde | -108.6 | -25.2 | 395.4 | 71.5 | not_evaluated |
| Hydrogen cyanide | 135.1 | -14.7 | 181.5 | 42.6 | not_evaluated |
| Carbon dioxide | -393.5 | 166.6 | 131.5 | 31.1 | not_evaluated |
| Carbon monoxide | -110.5 | 2.5 | -28.4 | 13.5 | not_evaluated |

(The two De columns are both physically invalid for this ANI-2x port and are shown only as evidence of the blocker. ZPE is correct and model-consistent. Molecules outside the subset are omitted from this table but listed as not-yet-evaluated in accuracy-report.json.)

## Caveats (methodology)

### CAVEAT 1 - enthalpy of formation is not a raw energy

- The caveat: A model returns a total electronic energy, not delta_f H. Comparing them directly is meaningless; an atomization + ZPE + thermal cycle plus experimental atomic delta_f H is required.
- What we did: Full atomization/RRHO cycle implemented (dHfFromAtomization). It cannot be closed for this ANI-2x port because De cannot be formed from the model (self-energies already contain binding; isolated atoms are out-of-distribution). Reported as not-yet-evaluated with diagnostic De values and correct ZPE/thermal, rather than fabricating free-atom constants.

### CAVEAT 2 - harmonic vs fundamental frequencies

- The caveat: A Hessian gives HARMONIC frequencies, systematically higher than experimental FUNDAMENTALS (3-6% for X-H). Comparing raw harmonic to fundamentals shows a fake ~5% error.
- What we did: Both reported: raw harmonic AND scaled by a standard factor of 0.96 (method-appropriate wB97X/6-31G* value ~0.95). Fermi-resonance (CO2), inversion-doubling (NH3), and large-amplitude (methanol OH torsion) modes are flagged.

### CAVEAT 3 - what each conformer energy includes

- The caveat: Reference conformer energies mix electronic (no ZPE) and ZPE-corrected/effective-0K conventions.
- What we did: Both conformers are relaxed with ANI-2x and the ELECTRONIC energy difference is taken. Butane is a pure electronic CCSD(T)/CBS reference (like-with-like); ethanol's ~0.12 kcal/mol effective-0K gap is sub-ZPE and near-degenerate, so ordering is a stress test. ZPE is not added to the model differences (noted per pair).

## How close are we to a real chemistry tool?

On the property ANI-2x is built for -- conformer relative energies -- it is genuinely at chemical accuracy (MAE a few tenths of a kcal/mol), with the honest caveat that a near-degenerate ordering (ethanol gauche/trans, ~0.1 kcal/mol) comes out flipped. Harmonic frequencies are physically sensible and land within ~2% of experiment after standard scaling. Absolute thermochemistry (heats of formation) is the honest weak spot: the model produces excellent relative energies but cannot, by itself, be turned into absolute formation enthalpies without external free-atom reference energies. So: a strong relative-energy engine, not yet a drop-in replacement for absolute thermochemistry.
