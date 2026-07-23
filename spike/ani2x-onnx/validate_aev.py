"""Validate the dense numpy AEV (aev_ref) against torchani's AEVComputer."""
import json
import numpy as np
import torch
import torchani
import aev_ref

m = torchani.models.ANI2x(periodic_table_index=True).double().eval()

with open("molecules.json") as fh:
    mols = json.load(fh)

print(f"{'molecule':12s} {'atoms':>5s} {'max|dAEV|':>12s} {'meanAEV':>12s}")
worst = 0.0
for name, mol in mols.items():
    znum = torch.tensor([mol["znums"]])
    coords = torch.tensor([mol["coords"]], dtype=torch.double)
    conv = m.species_converter((znum, coords))
    ref = m.aev_computer(conv).aevs[0].detach().numpy()   # (N,1008)
    mine = aev_ref.compute_aev(mol["znums"], mol["coords"])
    d = np.abs(ref - mine).max()
    worst = max(worst, d)
    print(f"{name:12s} {len(mol['znums']):5d} {d:12.3e} {np.abs(ref).mean():12.3e}")
print(f"\nWORST max|dAEV| across all molecules: {worst:.3e}")
print("PASS" if worst < 1e-6 else "FAIL (>1e-6)")
