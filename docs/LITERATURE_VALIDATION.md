# Literature Validation Reference

> Cross-check reference for the BrowserCompChem workbench. Every value below is
> experimental (or critically-evaluated) gas-phase data with a primary or
> authoritative-compilation citation. Geometry + vibrations pulled from NIST
> CCCBDB; ionization energies cross-checked against NIST WebBook's compiled
> experimental-IE listings; orbital-ordering assignments verified against the
> original PES assignment literature (Turner, Lindholm/Asbrink, Kimura, Baker)
> where obtainable.
>
> Compiled 2026-07-23 by a research subagent; honesty flags preserved verbatim
> at the bottom. Treat third-decimal eV values flagged below as needing a direct
> primary-source check before quoting.

## Where to look (so you never have to rebuild this table from memory)

| Resource | URL | What it's good for |
|---|---|---|
| **NIST CCCBDB** (SRD 101) | https://cccbdb.nist.gov | Per-molecule experimental **geometry** and **vibrational frequencies**, each with a literature citation, plus side-by-side comparison against dozens of computational methods. Best single starting point for small molecules. |
| **NIST Chemistry WebBook** | https://webbook.nist.gov/chemistry | For **ionization energies**: aggregates essentially every published measurement (PE/PI/EI/TE) per molecule with year/author, plus a critically-evaluated recommended value (usually S.G. Lias). Fastest way to see consensus vs. disputed. Also IR/UV-Vis/thermochemistry. |
| **CRC Handbook of Chemistry and Physics** | (print/library) | Tabulated bond lengths, angles, dipole moments, some IEs. Tertiary cross-check; usually itself sourced from the primary literature below. |
| Turner, Baker, Baker, Brundle, *Molecular Photoelectron Spectroscopy* (Wiley-Interscience, 1970) | — | The classic PES atlas; vertical IEs + MO assignments for H2O, NH3, CO2, C2H4, C2H2, benzene, and more. |
| Kimura, Katsumata, Achiba, Yamazaki, Iwata, *Handbook of HeI Photoelectron Spectra of Fundamental Organic Molecules* (1981) | — | 200-molecule HeI PES compilation with ab initio MO assignments; the standard secondary source later papers cite for "the" vertical IE ladder. |
| Shimanouchi, *Tables of Molecular Vibrational Frequencies, Consolidated Vol. I* (NSRDS-NBS 39, 1972) | — | Standard compiled experimental fundamental wavenumbers with symmetry labels. |
| Herzberg, *Molecular Spectra and Molecular Structure* II (IR/Raman, 1945) & III (electronic, 1966) | — | Classic compiled geometries, vibrational assignments, electronic-state symmetry labels. |
| Kuchitsu (ed.), *Structure of Free Polyatomic Molecules -- Basic Data* (Springer, 1998), DOI 10.1007/978-3-642-45748-7 | — | Critically-evaluated gas-phase structures (used for acetylene here). |

## How to validate EHT specifically (read before comparing any single eV)

Extended Huckel eigenvalues are **not** ionization energies and must never be
compared 1:1 against a PES peak. EHT systematically overbinds all orbitals by a
roughly constant offset (~3.5-4 eV here) because it omits electron correlation
and orbital relaxation on ionization. Validate on three qualitative axes instead:

1. **Ordering** -- does computed HOMO, HOMO-1, HOMO-2 match the experimentally
   assigned PES band sequence (e.g. water 1b1 > 3a1 > 1b2 > 2a1; benzene 1e1g(pi)
   as the first, lowest-binding band)?
2. **Degeneracy** -- are symmetry-required degeneracies (benzene e1g, acetylene
   and CO2 pi pairs, methane t2) actually returned as degenerate, with correct
   nodal/rotational symmetry in the isosurface?
3. **Relative spacings** -- are adjacent-level gaps in roughly the right
   proportion, even with the absolute zero shifted?

Cheap all-in-one sanity check: reproduce benzene's doubly-degenerate pi HOMO
(e1g) with a2u pi nearby but distinct, and water's non-degenerate 1b1 (out-of-plane
O lone pair) as strictly the highest occupied orbital. Get those two right and the
EHT implementation is doing its job. Matching PES eV quantitatively is NOT the bar
(that bar applies to ab initio/DFT benchmarking, a categorically different exercise).

---

## Water (H2O)

**Geometry (equilibrium, re):** r(O-H) 0.9578 A; angle(H-O-H) 104.48deg.
Method: fit of rotation-vibration levels. Hoy & Bunker, *J. Mol. Spectrosc.* **1979**, 74, 1. DOI 10.1016/0022-2852(79)90019-5

**Vibrations (fundamentals):** nu1 a1 3657 (sym stretch); nu2 a1 1595 (bend);
nu3 b2 3756 (antisym stretch). Shimanouchi NSRDS-NBS 39 (1972); Huber & Herzberg (1979).
These are anharmonic fundamentals; ab-initio harmonic values run a few % higher
(~3832/1649/3943, method-dependent, not independently verified).

**Ionization / ordering:**
- 1st (HOMO) 1b1 (out-of-plane O lone pair): 12.621 +/- 0.002 eV (evaluated). Brundle & Turner 1968/69; NIST WebBook (Lias).
- 2nd 3a1 (in-plane lone pair / OH mix): ~14.7-14.8. Turner et al. 1970; Potts & Price, *Proc. R. Soc. A* 1972, 326, 165. [digit flagged]
- 3rd 1b2 (OH sigma): ~18.5-18.7.
- 4th 2a1 (O 2s deep valence): ~32.2-32.6.
Frontier ordering 1b1 > 3a1 is a classic, undisputed PES result -- the correct
qualitative EHT target. No degeneracies expected (C2v).

**Lowest excitation:** first VUV continuum ~7.4-7.45 eV (~166.5 nm), 1b1 -> 3sa1.

---

## Methane (CH4)

**Geometry:** r(C-H) 1.087 +/- 0.001 A; angle 109.471deg (exact, Td).
Hirota, *J. Mol. Spectrosc.* **1979**, 77, 213.

**Vibrations:** nu1 a1 2917; nu2 e 1534; nu3 t2 3019; nu4 t2 1306. Shimanouchi (1972); Person & Zerbi (1982).

**Ionization / ordering:**
- 1t2 (HOMO, triply degenerate): adiabatic 12.61 +/- 0.01 eV (WebBook, Lias);
  vertical band centroid ~13.6 +/- 0.1 (Bieri et al., *Helv. Chim. Acta* 1977, 60, 2213; Kimura 1981).
- **Jahn-Teller caveat:** CH4+ (2T2) is strongly JT-unstable, so the PES band is a
  broad split composite, not a clean line. EHT returns a single perfectly-degenerate
  t2 HOMO -- correct for a rigid one-electron picture; the vibronic splitting is a
  real-molecule effect EHT is not meant to capture.

---

## Ammonia (NH3)

**Geometry:** r(N-H) 1.012 A; angle(H-N-H) 106.67 +/- 2deg. Herzberg (1966).

**Vibrations:** nu1 a1 3337 (sym stretch); nu2 a1 950 (umbrella/inversion);
nu3 e 3444 (antisym stretch); nu4 e 1627 (bend). Koops, Visser, Smit, *J. Mol. Struct.* **1983**, 96, 203.
nu2 is the famous double-well inversion mode (0.79 cm-1 splitting, maser transition)
-- large-amplitude, not well described by a single-well harmonic treatment.

**Ionization / ordering:**
- 1st (HOMO) 3a1 (N lone pair): 10.16-10.18 +/- 0.01 (vertical). Al-Joboury & Turner, *J. Chem. Soc.* 1964, 4434; Weiss & Lawrence, *J. Chem. Phys.* 1970, 53, 214.
- 2nd 1e (N-H bonding, degenerate): ~14.9-15.0.
- Evaluated *adiabatic* IE 10.070 +/- 0.020 (WebBook) is lower because NH3 flattens
  toward planar on ionization -- a geometry-relaxation effect a vertical
  single-geometry method (incl. EHT at the neutral geometry) cannot capture.
  **Compare the vertical numbers, not the adiabatic one, against an EHT HOMO.**

---

## Ethylene (C2H4)

**Geometry:** r(C=C) 1.339 A; r(C-H) 1.086 A; angle(H-C-H) 117.6 +/- 3deg;
angle(H-C-C) 121.2 +/- 1deg. Herzberg (1966).

**Vibrations (12 modes, D2h):** nu1 Ag 3026; nu2 Ag 1623 (C=C stretch); nu3 Ag 1342;
nu4 Au 1023; nu5 B1u 2989; nu6 B1u 1444; nu7 B2g 940; nu8 B2u 3105; nu9 B2u 826;
nu10 B3g 3086; nu11 B3g 1217; nu12 B3u 949. Van Lerberghe, Wright, Duncan, *J. Mol. Spectrosc.* **1972**, 42, 251; intensities Golike, Mills, Person, Crawford, *J. Chem. Phys.* **1956**, 25, 1266.

**Ionization / ordering:**
- 1st (HOMO) pi (C=C): 10.5138 +/- 0.0006 eV (evaluated), consistent 10.50-10.52
  across 12+ studies. Kimura 1981; Rabalais et al. 1974; NIST WebBook.
- 2nd-5th sigma manifold: ~12.85, 14.66, 15.87, ~19.2 eV **[flagged: secondary/
  textbook tables (Rabalais 1977), not re-derived from a primary source here].**
  Qualitative point solid: pi HOMO well separated (>2 eV) below a cluster of sigma.
- **D2h labeling trap:** axis-convention differs across papers, so the same physical
  orbital is b1u in one and b3u in another. Compare nodal character, not the letter.

**UV:** pi->pi* "V" state vertical ~7.66 eV (some reanalyses argue ~7.80); pi->3s
Rydberg ~7.11 eV. Even this number is vertical-vs-band-max-vs-adiabatic dependent.

---

## Acetylene (C2H2)

**Geometry (rm):** r(C-H) 1.063 A; r(C#C) 1.203 A. Kuchitsu (ed.), Springer 1998, DOI 10.1007/978-3-642-45748-7.

**Vibrations (D-inf-h):** nu1 Sg+ 3374 (sym C-H stretch); nu2 Sg+ 1974 (C#C stretch);
nu3 Su+ 3289 (antisym C-H stretch); nu4 Pg 612 (trans bend); nu5 Pu 730 (cis bend).
Shimanouchi (1972); Koops, Smit, Visser, *J. Mol. Struct.* **1984**, 125.

**Ionization / ordering:**
- 1st (HOMO) 1pi-u (doubly degenerate, C#C pi): 11.400 +/- 0.002 eV (evaluated),
  consistent 1935-1986. Baker & Turner, *Proc. R. Soc. A* **1968**, 308, 19; WebBook.
- 2nd 3sg ~16.3-16.8; 3rd 2su ~18.4-18.8; 4th 2sg ~23.7. Baker & Turner 1968.
- 1pi-u HOMO is a clean degenerate-pair test (nodal plane through the axis).

---

## Benzene (C6H6) -- the key validation molecule

**Geometry:** r(C-C) 1.397 A; r(C-H) 1.084 A; all angles 120deg (D6h). Herzberg (1966).

**Vibrations (selected, of 20, D6h):** nu1 A1g 992 (ring breathing, dominant
Franck-Condon mode in S1<-S0); nu2 A1g 3062; nu6 E2g 606; nu8 E2g 1596 (C=C);
nu13 E1u 1486 (IR-active); nu4 A2u 673 (out-of-plane C-H, IR-active).
Shimanouchi, *Tables of Molecular Vibrational Frequencies*.

**Ionization / ordering:**
- 1st (HOMO) 1e1g (pi, **doubly degenerate**): 9.24378 +/- 0.00007 eV (evaluated),
  ~40 studies 1935-1993 cluster 9.20-9.27. Al-Joboury & Turner 1964; Baker, Brundle
  et al. 1968; Klasinc et al. 1983; NIST WebBook.
- Full increasing-BE ordering: 1e1g(pi) < 2e2g(sigma) < 1a2u(pi) < 2e1u(sigma) <
  1b2u < 1b1u < 2a1g < 1e2g(deep). Assignment: Asbrink, Fridh, Lindholm (~1970-72);
  reconfirmed by many-body Green's-function calc, *J. Chem. Phys.* **1976**, 65, 1378.
- **Non-trivial test:** the a2u pi orbital is NOT simply the 2nd band -- a sigma-type
  e2g is interleaved between e1g and a2u. So the right target is: e1g HOMO degenerate
  and correctly nodal, with a2u appearing as *a* pi level somewhere in the upper
  valence manifold -- do NOT require e1g and a2u to be adjacent.

**UV:** S1 (1B2u) <- S0 origin 4.69 eV (264 nm); the "254 nm" system is the
vibronically-induced (dipole-forbidden, Herzberg-Teller) band built on nu1.

---

## Formaldehyde (H2CO)

**Geometry:** r(C=O) 1.205 A; r(C-H) 1.111 A; angle(H-C-H) 116.1deg; angle(H-C-O) 121.9deg.
Gurvich, Veyts, Alcock, *Thermodynamic Properties of Individual Substances*, 4th ed.
(1989) [compiled; agrees with microwave C=O ~1.203-1.208 A]. **[flagged: compiled
thermochemical source, less direct than a primary rotational-spectroscopy paper]**

**Vibrations (C2v):** nu1 a1 2782 (sym stretch); nu2 a1 1746 (C=O stretch);
nu3 a1 1500 (scissors); nu4 b1 1167 (out-of-plane wag); nu5 b2 2843 (antisym stretch);
nu6 b2 1249 (rock). Nakanaga, Kondo, Saeki, *J. Chem. Phys.* **1982**, 76, 3860.

**Ionization / ordering:**
- 1st (HOMO) 2b2 (in-plane O lone pair, "n"): 10.8887 +/- 0.0030 eV (high-res);
  cation ground state X2B2. Niu, Shirley et al. (1993).
- 2nd 1b1 (out-of-plane C=O pi): ~14-15 eV **[flagged, not pinned here]**.
- HOMO is the "n" lone pair, NOT pi(C=O) -- the textbook reason carbonyls show a
  low-energy n->pi* transition.

**UV:** weak, symmetry-forbidden (vibronically allowed) n->pi* (1A2<-1A1) near-UV,
onset ~350-400 nm (~3.1-3.5 eV) **[order-of-magnitude anchor only, flagged]**.
To pin down: Clouthier & Ramsay, *Annu. Rev. Phys. Chem.* **1983**, 34, 31.

---

## Carbon Dioxide (CO2)

**Geometry:** r(C=O) 1.162 A; linear (D-inf-h). Herzberg (1966).

**Vibrations:** nu1 Sg+ 1333 (sym stretch, unperturbed); nu2 Pu 667 (bend, degenerate);
nu3 Su+ 2349 (antisym stretch, strong IR -- the greenhouse band). Herzberg (1945).
- **Fermi resonance caveat:** nu1 (1333, Raman-only) is near-degenerate with 2*nu2
  (~1334) and mixes strongly; the observed Raman *doublet* sits at ~1285/1388 (the
  classic CO2 Fermi dyad). 1333 is the deperturbed value, not a literal single peak.

**Ionization / ordering:**
- 1st (HOMO) 1pi-g (doubly degenerate, O lone-pair combo, formally non-bonding):
  13.777 +/- 0.001 eV (evaluated), 13.773-13.80 across 10+ studies. Brundle & Turner
  1969; Wang, Reutt et al. 1988 (PFI-ZEKE); WebBook.
- 2nd 1pi-u ~17.2-17.6 (A2Pu); 3rd 3su ~18.1-19.0 (B2Su+). *J. Chem. Phys.* **2000**, 113, 7988.
- 1pi-g HOMO is another clean degenerate-pi test case.

---

## Methanol (CH3OH)

**Geometry (Cs):** r(C-O) 1.427 +/- 0.007 A; r(O-H) 0.956 +/- 0.015 A;
r(C-H) 1.096 +/- 0.010 A; angle(H-C-H) 109.03deg; angle(H-O-C) 108.87deg.
Venkateswarlu & Gordy, *J. Chem. Phys.* **1955**, 23, 1200 (microwave).

**Vibrations (selected):** nu1 A' 3681 (O-H stretch); nu2 A' 3000 (CH3 antisym);
nu3 A' 2844 (CH3 sym); nu8 A' 1033 (C-O stretch); nu12 A'' 200 (methyl torsion,
large-amplitude). Shimanouchi.

**Ionization / ordering:**
- 1st (HOMO) 2a'' (O lone pair, out-of-plane): 10.84 +/- 0.01 eV (evaluated),
  ~18 studies cluster 10.82-10.90. MacNeil & Dixon, *J. Electron Spectrosc.* **1977**, 11, 315 (10.846 +/- 0.002); WebBook.
- 2nd 7a' (in-plane lone pair / C-O sigma mix): **[flagged, not pinned]**.
  von Niessen, Bieri, Asbrink, *J. Electron Spectrosc.* **1980**, 21, 175.

**UV:** first absorption n->sigma* (O lone pair -> C-O sigma*), onset ~183 nm (~6.8 eV).

---

## Caffeine (1,3,7-trimethylxanthine, C8H10N4O2)

Far outside ANI-2x's small-molecule comfort zone for geometry/vibrations (fused
purine rings); useful here mainly for the frontier-orbital/ionization comparison.

**Geometry:** gas electron diffraction heavy-atom skeleton (C=O ~1.20 A, N-C
~1.34-1.45 A), refined against MP2/B3LYP. Egawa, Kamiya, Takeuchi, Konaka,
*J. Mol. Struct.* **2006**, 825, 151. DOI 10.1016/j.molstruc.2006.04.024.
(Solid-state alt: Edwards et al., *J. Chem. Soc. Perkin Trans. 2* **1997**, 1985 --
but the GED gas-phase structure is the right comparison for an isolated-molecule code.)

**Ionization:**
- First vertical IE (gas-phase HeI PES): **7.95 eV**. Dougherty, Younathan, Voll,
  Abdulnur, McGlynn, *J. Electron Spectrosc.* **1978**, 13, 379 [in NIST WebBook].
- Alt (charge-transfer spectra, not PES): 8.50 eV. Slifkin & Allison, *Nature* **1967**, 215, 949.
- This is the Koopmans-comparable anchor for an EHT caffeine HOMO, after the same
  ~3.5-4 eV overbinding correction. The 0.5 eV PES-vs-CT disagreement is itself a
  useful reminder that even "experimental" IEs carry method-dependent uncertainty at
  the several-tenths-eV level.

**UV:** maxima 206 nm and 272 nm (pi->pi* of the xanthine ring) [SIELC applied
reference, secondary]. No symmetry degeneracies expected (C1/Cs, low symmetry).

---

## Honesty flags (verify before treating as gospel)

1. Water 2nd-4th IEs (~14.7, 18.5, 32.2 eV) -- solid PES consensus but exact decimal
   varies by tenths across sources; not re-derived from one pinned primary here.
2. Ethylene 2nd-5th IEs (~12.85, 14.66, 15.87, ~19.2) -- secondary/textbook tables,
   not confirmed against a primary paper; also b1u/b3u/b3g/b1g labeling is
   axis-convention-dependent.
3. Formaldehyde 2nd IE (1b1 ~14-15) and methanol 2nd IE (7a') -- orbital identity
   solid, exact eV not pinned.
4. Formaldehyde n->pi* origin (~3.5 eV) -- order-of-magnitude anchor, not precise.
5. Formaldehyde geometry from a thermochemical compendium (Gurvich 1989), slightly
   less direct than the other molecules' structural citations.
6. CO2 nu1 = 1333 is the deperturbed value; real Raman shows the Fermi doublet
   ~1285/1388 -- account for this rather than comparing to a single 1333 peak.
7. Benzene exact pi/sigma interleaving (a2u position relative to e1g) reflects real
   historical assignment subtlety; ordering given is modern consensus.
8. Caffeine has thin gas-phase coverage vs. the nine textbook molecules; the GED
   geometry and PES first-IE are genuine primary sources but lack the depth of
   cross-validation available for benzene or acetylene.
