# Geometry provenance

Regenerate deterministically with `python _generate.py` (fixed seed).

Two classes:

- **experimental** - exact structures built from published spectroscopic parameters (bond lengths/angles from NIST CCCBDB experimental geometry section / Herzberg). Trustworthy equilibrium/effective structures.
- **computed-mmff** - RDKit ETKDGv3 embedding + MMFF94 (or UFF fallback) optimization from SMILES. A reasonable low-energy structure, **NOT experimental** and **NOT the exact ab initio benchmark conformer**. Use as a starting geometry; re-optimize with the target method for production benchmarking. For the conformer-energy systems, the specific rotamer is only approximate at the force-field level.

| file | class | detail |
|---|---|---|
| acetaldehyde.xyz | computed-mmff | SMILES CC=O; ETKDGv3 + MMFF94 |
| acetic_acid.xyz | computed-mmff | SMILES CC(=O)O; ETKDGv3 + MMFF94 |
| acetylene.xyz | experimental | D_inf_h linear; r(CH)=1.0605, r(CC)=1.2033 A (exp) |
| ammonia.xyz | experimental | C3v; r(NH)=1.0124 A, a(HNH)=106.67 deg (CCCBDB exp) |
| benzene.xyz | computed-mmff | SMILES c1ccccc1; ETKDGv3 + MMFF94 |
| carbon_dioxide.xyz | experimental | D_inf_h; r(CO)=1.1600 A (exp) |
| carbon_monoxide.xyz | experimental | C_inf_v; r(CO)=1.1283 A (exp) |
| chloromethane.xyz | computed-mmff | SMILES CCl; ETKDGv3 + MMFF94 |
| cysteine_confI.xyz | computed-mmff | SMILES SCC(N)C(=O)O; ETKDGv3 + MMFF94 |
| ethane.xyz | computed-mmff | SMILES CC; ETKDGv3 + MMFF94 |
| ethanol.xyz | computed-mmff | SMILES CCO; ETKDGv3 + MMFF94 |
| ethylene.xyz | experimental | D2h planar; r(CC)=1.339, r(CH)=1.086 A, a(HCC)=121.3 deg (exp) |
| fluoromethane.xyz | computed-mmff | SMILES CF; ETKDGv3 + MMFF94 |
| formaldehyde.xyz | experimental | C2v planar; r(CO)=1.208, r(CH)=1.116 A, a(HCO)=121.9 deg (exp) |
| formic_acid.xyz | computed-mmff | SMILES OC=O; ETKDGv3 + MMFF94 |
| furan.xyz | computed-mmff | SMILES c1ccoc1; ETKDGv3 + MMFF94 |
| glycine_confI.xyz | computed-mmff | SMILES NCC(=O)O; ETKDGv3 + MMFF94 |
| hydrogen_cyanide.xyz | experimental | C_inf_v linear; r(CH)=1.0655, r(CN)=1.1532 A (exp) |
| hydrogen_sulfide.xyz | experimental | C2v; r(SH)=1.3356 A, a(HSH)=92.11 deg (CCCBDB exp) |
| methane.xyz | experimental | Td; r(CH)=1.0870 A (CCCBDB exp) |
| methanethiol.xyz | computed-mmff | SMILES CS; ETKDGv3 + MMFF94 |
| methanol.xyz | computed-mmff | SMILES CO; ETKDGv3 + MMFF94 |
| methylamine.xyz | computed-mmff | SMILES CN; ETKDGv3 + MMFF94 |
| n-butane_anti.xyz | computed-mmff | SMILES CCCC; ETKDGv3 + MMFF94; dihedral (0, 1, 2, 3) set to 180.0 deg |
| propane.xyz | computed-mmff | SMILES CCC; ETKDGv3 + MMFF94 |
| propene.xyz | computed-mmff | SMILES CC=C; ETKDGv3 + MMFF94 |
| pyridine.xyz | computed-mmff | SMILES c1ccncc1; ETKDGv3 + MMFF94 |
| thiophene.xyz | computed-mmff | SMILES c1ccsc1; ETKDGv3 + MMFF94 |
| water.xyz | experimental | C2v; r(OH)=0.9578 A, a(HOH)=104.48 deg (CCCBDB exp) |
