"""Generate fresh TorchANI ANI-2x float64 references (energy + autograd forces)
for the six benchmark molecules, to bake into the engine's vitest fixtures.

Method (same as validate_energy.py): ANI2x(periodic_table_index=True).double(),
energy = model.energies (Ha); forces = -dE/dcoords (Ha/Angstrom) via autograd.

Writes references.json: { name: {symbols, znums, coords, energy, forces[N][3]} }.
"""
import json
import os
import torch
import torchani

ZSYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl"}

m = torchani.models.ANI2x(periodic_table_index=True).double().eval()
mols = json.load(open(os.path.join(os.path.dirname(__file__), "molecules.json")))

out = {}
for name, mol in mols.items():
    z = mol["znums"]
    c = mol["coords"]
    znum = torch.tensor([z])
    co = torch.tensor([c], dtype=torch.double, requires_grad=True)
    e = m((znum, co)).energies
    g = torch.autograd.grad(e.sum(), co)[0]
    forces = (-g[0]).detach().tolist()
    out[name] = {
        "symbols": [ZSYM[int(zz)] for zz in z],
        "znums": z,
        "coords": c,
        "energy": float(e.item()),
        "forces": forces,
    }
    print(f"{name:12s} atoms={len(z):3d} E={e.item():.6f} Ha  maxF={max(abs(f) for row in forces for f in row):.4e}")

dest = os.path.join(os.path.dirname(__file__), "..", "..", "packages", "engine",
                    "test", "fixtures", "ani2x-references.json")
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", os.path.abspath(dest))
