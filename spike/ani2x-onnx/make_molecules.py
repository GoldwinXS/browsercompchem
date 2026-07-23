"""Generate test-molecule geometries (elements limited to H C N O S F Cl for ANI-2x).
Writes molecules.json: {name: {znums: [...], coords: [[x,y,z],...] (Angstrom)}}.
"""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17}


def from_smiles(smiles, seed=1):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    znums, coords = [], []
    for atom in mol.GetAtoms():
        znums.append(atom.GetAtomicNum())
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    return znums, coords


def main():
    mols = {}
    # water (hardcoded)
    mols["water"] = {
        "znums": [8, 1, 1],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
    }
    # methane
    z, c = from_smiles("C")
    mols["methane"] = {"znums": z, "coords": c}
    # aspirin
    z, c = from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    mols["aspirin"] = {"znums": z, "coords": c}
    # caffeine
    z, c = from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    mols["caffeine"] = {"znums": z, "coords": c}
    # cholesterol
    z, c = from_smiles("CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C")
    mols["cholesterol"] = {"znums": z, "coords": c}
    # a peptide (~150 atoms) via amino-acid sequence (elements stay in HCNOS)
    seq = "AGSVLAGSVL"  # 10 residues
    pm = Chem.MolFromSequence(seq)
    pm = Chem.AddHs(pm)
    p = AllChem.ETKDGv3(); p.randomSeed = 1
    AllChem.EmbedMolecule(pm, p)
    AllChem.MMFFOptimizeMolecule(pm)
    conf = pm.GetConformer()
    z = [a.GetAtomicNum() for a in pm.GetAtoms()]
    c = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(pm.GetNumAtoms())]
    mols["peptide"] = {"znums": z, "coords": c}

    for k, v in mols.items():
        v["coords"] = np.asarray(v["coords"], dtype=float).round(6).tolist()
        print(f"{k}: {len(v['znums'])} atoms, elements {sorted(set(v['znums']))}")

    with open("molecules.json", "w") as fh:
        json.dump(mols, fh, indent=1)
    print("wrote molecules.json")


if __name__ == "__main__":
    main()
