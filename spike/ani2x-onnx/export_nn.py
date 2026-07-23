"""
Strategy (c) export: the AEV featurizer is computed in JS; here we export only
the per-element neural networks (pure MLPs) to ONNX, plus a params.json with all
AEV constants and self-energies needed by the JS AEV implementation.

Outputs:
  ani2x_nn_energy.onnx   inputs (aev[N,1008], mask[N,7]) -> nn_energy scalar (ensemble mean, pre-self-energy)
  ani2x_nn_grad.onnx     inputs (aev[N,1008], mask[N,7]) -> dE/dAEV [N,1008]
  params.json            AEV parameters, species order, self energies
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchani

torch.manual_seed(0)
DT = torch.float32

SYMBOLS = ("H", "C", "N", "O", "S", "F", "Cl")  # SYMBOLS_2X internal order


def tight_celu(x):
    return F.celu(x, alpha=0.1)


class NNEnergy(torch.nn.Module):
    """Runs all 7 element nets on all atoms, masks by species, averages ensemble."""

    def __init__(self, model):
        super().__init__()
        self.ensemble = model.neural_networks  # 8 members
        self.n_members = len(self.ensemble)
        self.symbols = SYMBOLS

    def forward(self, aev, mask):
        # aev [N,1008], mask [N,7] one-hot
        total = aev.new_zeros(())
        for member in self.ensemble:
            atomic = member["atomics"] if False else member  # ANINetworks
            per_elem = []
            for sym in self.symbols:
                net = member[sym]
                x = aev
                for layer in net.layers:
                    x = tight_celu(layer(x))
                x = net.final_layer(x)  # [N,1]
                per_elem.append(x)
            stacked = torch.cat(per_elem, dim=1)  # [N,7]
            masked = (stacked * mask).sum(dim=1)  # [N]
            total = total + masked.sum()
        return total / self.n_members


def get_torchani_refs(m, znum, coords):
    conv = m.species_converter((znum, coords))
    species_idx = conv[0]  # [1,N] internal idx
    sc = m.aev_computer(conv)
    aev = sc.aevs  # [1,N,1008]
    return species_idx, aev


def main():
    m = torchani.models.ANI2x(periodic_table_index=True).to(DT).eval()

    # ---- params.json ----
    aevc = m.aev_computer
    rad = aevc.radial
    ang = aevc.angular
    params = {
        "symbols": list(SYMBOLS),
        "self_energies": m.energy_shifter.self_energies.double().tolist(),
        "aev_dim": int(aevc.out_dim),
        "radial_len": int(aevc.radial_len),
        "angular_len": int(aevc.angular_len),
        "num_species": int(aevc.num_species),
        "num_species_pairs": int(aevc.num_species_pairs),
        "radial": {
            "eta": float(rad.eta),
            "shifts": [float(x) for x in rad.shifts.flatten().tolist()],
            "cutoff": float(rad.cutoff),
        },
        "angular": {
            "eta": float(ang.eta),
            "zeta": float(ang.zeta),
            "shifts": [float(x) for x in ang.shifts.flatten().tolist()],
            "sections": [float(x) for x in ang.sections.flatten().tolist()],
            "cutoff": float(ang.cutoff),
        },
    }
    with open("params.json", "w") as fh:
        json.dump(params, fh, indent=2)
    print("wrote params.json; aev_dim", params["aev_dim"],
          "radial_len", params["radial_len"], "angular_len", params["angular_len"])

    nn = NNEnergy(m).eval()

    # ---- validate NN split vs torchani on caffeine-ish random small mol ----
    import numpy as np
    # water
    znum = torch.tensor([[8, 1, 1]])
    coords = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]], dtype=DT)
    sp_idx, aev = get_torchani_refs(m, znum, coords)
    N = aev.shape[1]
    aev2 = aev[0]  # [N,1008]
    mask = F.one_hot(sp_idx[0], num_classes=7).to(DT)  # [N,7]
    my_nn_e = nn(aev2, mask)
    full_e = m((znum, coords)).energies
    # self energy
    self_e = m.energy_shifter.self_energies[sp_idx[0]].sum()
    ref_nn_e = full_e - self_e
    print("water: my_nn_e", my_nn_e.item(), "ref_nn_e", ref_nn_e.item(),
          "diff", abs(my_nn_e.item() - ref_nn_e.item()))
    print("water: full_e", full_e.item(), "reconstructed", (my_nn_e + self_e).item())

    # ---- export energy onnx ----
    torch.onnx.export(
        nn, (aev2, mask), "ani2x_nn_energy.onnx",
        input_names=["aev", "mask"], output_names=["nn_energy"],
        dynamic_axes={"aev": {0: "n"}, "mask": {0: "n"}},
        opset_version=17, dynamo=False,
    )
    print("exported ani2x_nn_energy.onnx")

    # ---- grad module ----
    class NNGrad(torch.nn.Module):
        def __init__(self, nn):
            super().__init__()
            self.nn = nn

        def forward(self, aev, mask):
            aev = aev.detach().clone().requires_grad_(True)
            e = self.nn(aev, mask)
            g = torch.autograd.grad([e], [aev], create_graph=False)[0]
            return g

    # NN-gradient ONNX export is blocked by aten::elu_backward (CELU derivative
    # not in the ONNX opset-17 symbolic registry). Forces are done by finite
    # differences on the energy graph instead. Keep NNGrad for reference/optional.
    _ = NNGrad


if __name__ == "__main__":
    main()
