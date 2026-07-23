"""Wave-two export: dump the ANI-2x per-element MLP weights to a compact binary
format for the pure-TypeScript forward/backward implementation (no ONNX runtime).

Rationale (dE/dAEV strategy): the spike proved the NN-gradient ONNX export is
blocked by aten::elu_backward. Rather than fight the ONNX symbolic registry, we
run the MLPs in plain TypeScript (Linear + CELU, tiny nets) with a hand-written
backward pass. CELU'(z, alpha) = 1 if z>0 else exp(z/alpha) is elementary, so
dE/dAEV is exact and trivial. This also makes per-atom species gather, single-
member, and fp16 variants fall out for free (task 4).

Outputs into ../../models/ani2x/:
  manifest.json                 topology + AEV params + self energies + variant list
  weights-full-f32.bin          8 ensemble members, float32   (~55 MB)
  weights-single-f32.bin        member 0 only, float32        (~7 MB)
  weights-full-f16.bin          8 members, float16            (~27 MB)
  weights-single-f16.bin        member 0 only, float16        (~3.5 MB)

Binary layout (per variant), matching manifest["elements"] layer order:
  for member in range(members):
    for sym in symbols:            # H,C,N,O,S,F,Cl
      for layer in element_layers: # 3 hidden (CELU) + 1 final (linear)
        weight[out, in] row-major  # out*in scalars
        bias[out]                  # out scalars
dtype is f32 (little-endian float32) or f16 (little-endian IEEE half as uint16).
"""
import json
import os
import struct
import numpy as np
import torch
import torchani

SYMBOLS = ("H", "C", "N", "O", "S", "F", "Cl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ani2x")


def element_layer_dims(net):
    dims = [[l.out_features, l.in_features] for l in net.layers]
    dims.append([net.final_layer.out_features, net.final_layer.in_features])
    return dims


def iter_layer_tensors(net):
    for l in net.layers:
        yield l.weight.detach().cpu().numpy(), l.bias.detach().cpu().numpy()
    yield net.final_layer.weight.detach().cpu().numpy(), net.final_layer.bias.detach().cpu().numpy()


def collect_flat(ensemble, members):
    """Concatenate all weights/biases in manifest order into one float64 array."""
    chunks = []
    for mi in range(members):
        member = ensemble[mi]
        for sym in SYMBOLS:
            net = member[sym]
            for w, b in iter_layer_tensors(net):
                chunks.append(w.reshape(-1).astype(np.float64))
                chunks.append(b.reshape(-1).astype(np.float64))
    return np.concatenate(chunks)


def write_bin(path, flat, dtype):
    if dtype == "f32":
        flat.astype("<f4").tofile(path)
    elif dtype == "f16":
        # store IEEE half bit patterns as little-endian uint16
        flat.astype(np.float16).view(np.uint16).astype("<u2").tofile(path)
    else:
        raise ValueError(dtype)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m = torchani.models.ANI2x(periodic_table_index=True).double().eval()
    ensemble = m.neural_networks
    n_members = len(ensemble)

    aevc = m.aev_computer
    rad, ang = aevc.radial, aevc.angular

    elements = {}
    for sym in SYMBOLS:
        elements[sym] = element_layer_dims(ensemble[0][sym])

    manifest = {
        "family": "ani2x",
        "symbols": list(SYMBOLS),
        "self_energies": m.energy_shifter.self_energies.double().tolist(),
        "aev_dim": int(aevc.out_dim),
        "radial_len": int(aevc.radial_len),
        "angular_len": int(aevc.angular_len),
        "num_species": int(aevc.num_species),
        "num_species_pairs": int(aevc.num_species_pairs),
        "celu_alpha": 0.1,
        "elements": elements,
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
        "variants": [],
    }

    full = collect_flat(ensemble, n_members)
    single = collect_flat(ensemble, 1)

    specs = [
        ("full-f32", full, n_members, "f32"),
        ("single-f32", single, 1, "f32"),
        ("full-f16", full, n_members, "f16"),
        ("single-f16", single, 1, "f16"),
    ]
    for name, flat, members, dtype in specs:
        fname = f"weights-{name}.bin"
        path = os.path.join(OUT_DIR, fname)
        write_bin(path, flat, dtype)
        size = os.path.getsize(path)
        manifest["variants"].append(
            {"name": name, "file": fname, "dtype": dtype, "members": members,
             "count": int(flat.size), "bytes": int(size)}
        )
        print(f"{name:14s} members={members} dtype={dtype} count={flat.size} bytes={size}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote manifest.json to", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
