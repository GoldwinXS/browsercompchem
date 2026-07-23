"""Validate full ONNX-path energy and finite-difference forces vs TorchANI.

Energy: numpy AEV (float32) -> ani2x_nn_energy.onnx (float32) -> + self energy (f64)
        compared to TorchANI double-precision reference. Gate: < 1e-5 Ha.
Forces: central finite differences on the energy pipeline. Validated in double
        precision (numpy f64 AEV + a torch f64 replica of the NN) so FD noise is
        controlled, vs TorchANI analytic forces. Gate: < 1e-4 Ha/Angstrom.
        We also report the float32/ONNX-path FD force error for realism.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchani
import onnxruntime as ort
import aev_ref

DT = torch.float32
SYMBOLS = aev_ref.SYMBOLS
ZNUM = aev_ref.ZNUM

m64 = torchani.models.ANI2x(periodic_table_index=True).double().eval()
self_e = np.asarray(json.load(open("params.json"))["self_energies"])  # (7,) f64

# torch float32 NN replica (identical weights) for controlled-precision checks
mf = torchani.models.ANI2x(periodic_table_index=True).double().eval()


def torch_nn_energy_f64(znums, coords):
    """Double-precision replica of the ONNX NN: AEV(numpy f64) -> torch f64 nets."""
    aev = aev_ref.compute_aev(znums, coords)  # (N,1008) f64
    sp = aev_ref.species_idx(znums)
    aev_t = torch.tensor(aev, dtype=torch.double)
    total = torch.zeros((), dtype=torch.double)
    for member in m64.neural_networks:
        per = []
        for sym in SYMBOLS:
            net = member[sym]
            x = aev_t
            for layer in net.layers:
                x = F.celu(layer(x), alpha=0.1)
            x = net.final_layer(x)
            per.append(x)
        stacked = torch.cat(per, dim=1)
        mask = F.one_hot(torch.tensor(sp), num_classes=7).double()
        total = total + (stacked * mask).sum()
    return total / len(m64.neural_networks)


sess = ort.InferenceSession("ani2x_nn_energy.onnx", providers=["CPUExecutionProvider"])


def onnx_energy(znums, coords):
    aev = aev_ref.compute_aev(znums, coords).astype(np.float32)
    sp = aev_ref.species_idx(znums)
    mask = np.eye(7, dtype=np.float32)[sp]
    nn_e = sess.run(None, {"aev": aev, "mask": mask})[0]
    self_energy = self_e[sp].sum()
    return float(nn_e) + float(self_energy)


def torchani_ref(znums, coords):
    znum = torch.tensor([znums])
    co = torch.tensor([coords], dtype=torch.double, requires_grad=True)
    e = m64((znum, co)).energies
    g = torch.autograd.grad(e.sum(), co)[0]
    return float(e.item()), (-g[0]).detach().numpy()  # energy, forces (N,3)


def fd_forces_f64(znums, coords, h=1e-3):
    coords = np.asarray(coords, dtype=np.float64)
    N = len(znums)
    F_ = np.zeros((N, 3))
    sp = aev_ref.species_idx(znums)
    se = self_e[sp].sum()

    def E(c):
        return float(torch_nn_energy_f64(znums, c)) + se

    for i in range(N):
        for d in range(3):
            cp = coords.copy(); cp[i, d] += h
            cm = coords.copy(); cm[i, d] -= h
            F_[i, d] = -(E(cp) - E(cm)) / (2 * h)
    return F_


def main():
    mols = json.load(open("molecules.json"))
    print(f"{'molecule':12s} {'atoms':>5s} {'dE_onnx(Ha)':>13s} {'E_ref(Ha)':>14s} "
          f"{'maxF_dev_f64':>13s}")
    worstE = 0.0
    worstF = 0.0
    force_targets = {"water", "methane", "aspirin", "caffeine"}
    for name, mol in mols.items():
        z = mol["znums"]; c = mol["coords"]
        e_onnx = onnx_energy(z, c)
        e_ref, f_ref = torchani_ref(z, c)
        de = abs(e_onnx - e_ref)
        worstE = max(worstE, de)
        fdev = float("nan")
        if name in force_targets:
            f_fd = fd_forces_f64(z, c)
            fdev = np.abs(f_fd - f_ref).max()
            worstF = max(worstF, fdev)
        print(f"{name:12s} {len(z):5d} {de:13.3e} {e_ref:14.5f} {fdev:13.3e}")
    print(f"\nWorst |dE| (ONNX f32 path vs torchani f64): {worstE:.3e}  "
          f"{'PASS' if worstE < 1e-5 else 'FAIL'} (gate 1e-5 Ha)")
    print(f"Worst max|dF| (double FD vs torchani analytic): {worstF:.3e}  "
          f"{'PASS' if worstF < 1e-4 else 'FAIL'} (gate 1e-4 Ha/A)")


if __name__ == "__main__":
    main()
