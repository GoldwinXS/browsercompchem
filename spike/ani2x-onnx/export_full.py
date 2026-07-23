"""
Export ANI-2x (energy + forces) to ONNX.

Strategy attempt (a): full-model export via the legacy TorchScript-trace ONNX
exporter, with a wrapper that computes forces = -dE/dcoords using autograd
inside forward. The legacy tracer records the backward ops into the graph, so
the exported graph outputs BOTH energy and forces with no autograd needed at
inference time (works in onnxruntime-web).

We trace at a fixed atom count and declare the atom axis dynamic; we then test
empirically whether a different atom count runs (validate_numerics.py).
"""
import sys
import torch
import torchani

torch.manual_seed(0)


class ANIEnergyForce(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, species, coords):
        coords = coords.detach().clone().requires_grad_(True)
        energy = self.model((species, coords)).energies  # (B,)
        grad = torch.autograd.grad(
            [energy.sum()], [coords], create_graph=False, retain_graph=False
        )[0]
        forces = -grad
        return energy, forces


def build(dtype=torch.float32):
    model = torchani.models.ANI2x(periodic_table_index=True).to(dtype)
    model.eval()
    return ANIEnergyForce(model)


def main():
    n_atoms = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ani2x_ef.onnx"
    dtype = torch.float32

    wrapper = build(dtype)

    # dummy: an n_atoms molecule of carbons in a line (valid species for ANI2x)
    species = torch.full((1, n_atoms), 6, dtype=torch.long)
    coords = torch.zeros((1, n_atoms, 3), dtype=dtype)
    coords[0, :, 0] = torch.arange(n_atoms, dtype=dtype) * 1.3

    # sanity run
    e, f = wrapper(species, coords)
    print("dummy energy", e.item(), "force shape", f.shape)

    torch.onnx.export(
        wrapper,
        (species, coords),
        out_path,
        input_names=["species", "coords"],
        output_names=["energy", "forces"],
        dynamic_axes={
            "species": {1: "n_atoms"},
            "coords": {1: "n_atoms"},
            "forces": {1: "n_atoms"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("exported to", out_path)


if __name__ == "__main__":
    main()
