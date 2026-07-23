"""
Geometry generator for the compchem accuracy bench.

Two provenance classes (recorded per-file in the XYZ comment line and in PROVENANCE.md):
  [experimental] : exact equilibrium/effective structures built from published
                   spectroscopic parameters (CCCBDB / Herzberg). Small rigid molecules.
  [computed-mmff]: RDKit ETKDG(v3) embedding + MMFF94 optimization from SMILES.
                   A reasonable low-energy geometry, NOT an experimental structure and
                   NOT the exact ab initio benchmark conformer. Intended as a starting
                   structure; a production bench should re-optimize with the target method.

Run:  python _generate.py
Deterministic (fixed random seed) so re-running reproduces identical files.
"""
import math, os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 1) EXACT EXPERIMENTAL GEOMETRIES (built from spectroscopic parameters)
#    Sources: NIST CCCBDB experimental geometry section; Herzberg; standard values.
#    Bond lengths in Angstrom, angles in degrees.
# ----------------------------------------------------------------------------

def water():
    r, ang = 0.9578, 104.48
    h = math.radians(ang/2)
    return "C2v; r(OH)=0.9578 A, a(HOH)=104.48 deg (CCCBDB exp)", [
        ("O", (0.0, 0.0, 0.0)),
        ("H", ( r*math.sin(h),  r*math.cos(h), 0.0)),
        ("H", (-r*math.sin(h),  r*math.cos(h), 0.0)),
    ]

def hydrogen_sulfide():
    r, ang = 1.3356, 92.11
    h = math.radians(ang/2)
    return "C2v; r(SH)=1.3356 A, a(HSH)=92.11 deg (CCCBDB exp)", [
        ("S", (0.0, 0.0, 0.0)),
        ("H", ( r*math.sin(h),  r*math.cos(h), 0.0)),
        ("H", (-r*math.sin(h),  r*math.cos(h), 0.0)),
    ]

def ammonia():
    r, ang = 1.0124, 106.67
    cs = math.cos(math.radians(ang))
    cb2 = (2*cs + 1)/3.0          # cos^2(beta), beta = bond/axis angle
    cb = math.sqrt(cb2); sb = math.sqrt(1-cb2)
    z = -r*cb; rad = r*sb
    atoms = [("N", (0.0, 0.0, 0.0))]
    for k in range(3):
        phi = math.radians(120*k)
        atoms.append(("H", (rad*math.cos(phi), rad*math.sin(phi), z)))
    return "C3v; r(NH)=1.0124 A, a(HNH)=106.67 deg (CCCBDB exp)", atoms

def methane():
    r = 1.0870
    dirs = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
    s = r/math.sqrt(3)
    atoms = [("C",(0.0,0.0,0.0))]
    for d in dirs:
        atoms.append(("H",(s*d[0], s*d[1], s*d[2])))
    return "Td; r(CH)=1.0870 A (CCCBDB exp)", atoms

def carbon_monoxide():
    return "C_inf_v; r(CO)=1.1283 A (exp)", [("C",(0,0,0)),("O",(0,0,1.1283))]

def carbon_dioxide():
    r=1.1600
    return "D_inf_h; r(CO)=1.1600 A (exp)", [("C",(0,0,0)),("O",(0,0,r)),("O",(0,0,-r))]

def hydrogen_cyanide():
    rch, rcn = 1.0655, 1.1532
    return "C_inf_v linear; r(CH)=1.0655, r(CN)=1.1532 A (exp)", [
        ("H",(0,0,0)),("C",(0,0,rch)),("N",(0,0,rch+rcn))]

def acetylene():
    rch, rcc = 1.0605, 1.2033
    return "D_inf_h linear; r(CH)=1.0605, r(CC)=1.2033 A (exp)", [
        ("H",(0,0,0)),("C",(0,0,rch)),("C",(0,0,rch+rcc)),("H",(0,0,rch+rcc+rch))]

def ethylene():
    rcc, rch, hcc = 1.339, 1.086, 121.3
    cx = rcc/2.0
    a = math.radians(180 - hcc)   # bond angle from +x for C1 H's
    dx, dy = rch*math.cos(a), rch*math.sin(a)
    return "D2h planar; r(CC)=1.339, r(CH)=1.086 A, a(HCC)=121.3 deg (exp)", [
        ("C",( cx,0,0)), ("C",(-cx,0,0)),
        ("H",( cx+dx, dy,0)), ("H",( cx+dx,-dy,0)),
        ("H",(-cx-dx, dy,0)), ("H",(-cx-dx,-dy,0)),
    ]

def formaldehyde():
    rco, rch, hcc = 1.208, 1.116, 121.9   # a(HCO)=121.9, a(HCH)=116.2
    # C at origin, O along +y; H's below
    atoms = [("C",(0.0,0.0,0.0)), ("O",(0.0,rco,0.0))]
    a = math.radians(hcc)
    for sgn in (1,-1):
        # H direction: angle hcc from C->O (which is +y)
        hx = rch*math.sin(a)*sgn
        hy = rch*math.cos(a)
        atoms.append(("H",(hx, hy, 0.0)))
    return "C2v planar; r(CO)=1.208, r(CH)=1.116 A, a(HCO)=121.9 deg (exp)", atoms

EXPERIMENTAL = {
    "water": water, "ammonia": ammonia, "methane": methane,
    "hydrogen_sulfide": hydrogen_sulfide, "carbon_monoxide": carbon_monoxide,
    "carbon_dioxide": carbon_dioxide, "hydrogen_cyanide": hydrogen_cyanide,
    "acetylene": acetylene, "ethylene": ethylene, "formaldehyde": formaldehyde,
}

# ----------------------------------------------------------------------------
# 2) COMPUTED MMFF GEOMETRIES (RDKit) for everything else.
# ----------------------------------------------------------------------------
MMFF = {
    "ethane": ("CC", None),
    "propane": ("CCC", None),
    "propene": ("CC=C", None),
    "benzene": ("c1ccccc1", None),
    "methanol": ("CO", None),
    "ethanol": ("CCO", None),   # global-min anti conformer
    "acetaldehyde": ("CC=O", None),
    "acetic_acid": ("CC(=O)O", None),
    "formic_acid": ("OC=O", None),
    "methylamine": ("CN", None),
    "pyridine": ("c1ccncc1", None),
    "furan": ("c1ccoc1", None),
    "thiophene": ("c1ccsc1", None),
    "chloromethane": ("CCl", None),
    "fluoromethane": ("CF", None),
    "methanethiol": ("CS", None),
    # conformer systems (specific rotamers requested by conformers.json)
    "n-butane_anti": ("CCCC", ("dihedral", (0,1,2,3), 180.0)),
    "glycine_confI": ("NCC(=O)O", None),
    "cysteine_confI": ("SCC(N)C(=O)O", None),
}

def write_xyz(name, comment, atoms):
    path = os.path.join(OUT, f"{name}.xyz")
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for sym,(x,y,z) in atoms:
            f.write(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
    return path

prov = []  # (name, class, detail)

for name, fn in EXPERIMENTAL.items():
    detail, atoms = fn()
    write_xyz(name, f"[experimental] {name} | {detail}", atoms)
    prov.append((name, "experimental", detail))

# RDKit part
from rdkit import Chem
from rdkit.Chem import AllChem

def build_mmff(name, smiles, mod):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = 0xC0FFEE
    AllChem.EmbedMolecule(mol, params)
    if mod and mod[0] == "dihedral":
        AllChem.MMFFOptimizeMolecule(mol)
        idx = mod[1]
        from rdkit.Chem import rdMolTransforms as rmt
        conf = mol.GetConformer()
        rmt.SetDihedralDeg(conf, *idx, mod[2])
    ff_ok = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    method = "ETKDGv3 + MMFF94"
    if ff_ok != 0:  # not converged or unavailable -> try UFF
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000); method = "ETKDGv3 + UFF"
    conf = mol.GetConformer()
    atoms = []
    for at in mol.GetAtoms():
        p = conf.GetAtomPosition(at.GetIdx())
        atoms.append((at.GetSymbol(), (p.x, p.y, p.z)))
    return method, atoms

for name,(smiles,mod) in MMFF.items():
    method, atoms = build_mmff(name, smiles, mod)
    detail = f"SMILES {smiles}; {method}" + (f"; dihedral {mod[1]} set to {mod[2]} deg" if mod else "")
    write_xyz(name, f"[computed-mmff] {name} | {detail}", atoms)
    prov.append((name, "computed-mmff", detail))

# PROVENANCE.md
with open(os.path.join(OUT,"PROVENANCE.md"),"w") as f:
    f.write("# Geometry provenance\n\n")
    f.write("Regenerate deterministically with `python _generate.py` (fixed seed).\n\n")
    f.write("Two classes:\n\n")
    f.write("- **experimental** - exact structures built from published spectroscopic ")
    f.write("parameters (bond lengths/angles from NIST CCCBDB experimental geometry section / Herzberg). ")
    f.write("Trustworthy equilibrium/effective structures.\n")
    f.write("- **computed-mmff** - RDKit ETKDGv3 embedding + MMFF94 (or UFF fallback) optimization ")
    f.write("from SMILES. A reasonable low-energy structure, **NOT experimental** and **NOT the exact ")
    f.write("ab initio benchmark conformer**. Use as a starting geometry; re-optimize with the target ")
    f.write("method for production benchmarking. For the conformer-energy systems, the specific rotamer ")
    f.write("is only approximate at the force-field level.\n\n")
    f.write("| file | class | detail |\n|---|---|---|\n")
    for name,cls,detail in sorted(prov):
        f.write(f"| {name}.xyz | {cls} | {detail} |\n")

print(f"Wrote {len(prov)} geometries")
