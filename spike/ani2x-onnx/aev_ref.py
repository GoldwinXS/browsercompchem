"""Dense all-pairs ANI-2x AEV in numpy (reference for the JS port).
Layout matches torchani AEVComputer exactly (validated in validate_aev.py).
"""
import json
import numpy as np

with open("params.json") as fh:
    P = json.load(fh)

SYMBOLS = P["symbols"]                 # ('H','C','N','O','S','F','Cl')
ZNUM = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6}
NS = P["num_species"]                  # 7
NSP = P["num_species_pairs"]           # 28
R = P["radial"]
A = P["angular"]
RcR = R["cutoff"]; etaR = R["eta"]; shfR = np.asarray(R["shifts"])          # (16,)
RcA = A["cutoff"]; etaA = A["eta"]; zeta = A["zeta"]
shfA = np.asarray(A["shifts"]); sections = np.asarray(A["sections"])         # (8,), (4,)
NR = len(shfR); NA = len(shfA); NZ = len(sections); NAF = NA * NZ            # 16, 8, 4, 32
RADIAL_LEN = P["radial_len"]; ANGULAR_LEN = P["angular_len"]                 # 112, 896


def triu_index_matrix(n=NS):
    m = np.zeros((n, n), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(i, n):
            m[i, j] = c
            m[j, i] = c
            c += 1
    return m


TRIU = triu_index_matrix()


def species_idx(znums):
    return np.asarray([ZNUM[int(z)] for z in znums], dtype=np.int64)


def fc(d, rc):
    return 0.5 * np.cos(np.pi * d / rc) + 0.5


def compute_aev(znums, coords):
    """coords: (N,3) Angstrom. Returns aev (N, 1008)."""
    sp = species_idx(znums)
    N = len(sp)
    coords = np.asarray(coords, dtype=np.float64)
    aev = np.zeros((N, RADIAL_LEN + ANGULAR_LEN), dtype=np.float64)

    # pairwise
    diff = coords[:, None, :] - coords[None, :, :]      # (N,N,3) r_i - r_j
    dist = np.linalg.norm(diff, axis=-1)                # (N,N)

    # ---- radial ----
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d > RcR:
                continue
            terms = 0.25 * np.exp(-etaR * (d - shfR) ** 2) * fc(d, RcR)  # (16,)
            base = sp[j] * NR
            aev[i, base:base + NR] += terms

    # ---- angular ----
    for j in range(N):
        # neighbors of j within RcA
        neigh = [k for k in range(N) if k != j and dist[j, k] <= RcA]
        for ai in range(len(neigh)):
            for bi in range(ai + 1, len(neigh)):
                a = neigh[ai]; b = neigh[bi]
                d_ja = dist[j, a]; d_jb = dist[j, b]
                v_ja = coords[a] - coords[j]
                v_jb = coords[b] - coords[j]
                cosang = np.dot(v_ja, v_jb) / max(d_ja * d_jb, 1e-10)
                angle = np.arccos(0.95 * cosang)
                rad8 = np.exp(-etaA * (((d_ja + d_jb) / 2) - shfA) ** 2)     # (8,)
                ang4 = 2.0 * ((1.0 + np.cos(angle - sections)) / 2.0) ** zeta  # (4,)
                fac = fc(d_ja, RcA) * fc(d_jb, RcA)
                block = (rad8[:, None] * ang4[None, :]).reshape(-1) * fac    # (32,) idx=r*4+z
                pairidx = TRIU[sp[a], sp[b]]
                base = RADIAL_LEN + pairidx * NAF
                aev[j, base:base + NAF] += block
    return aev


if __name__ == "__main__":
    import sys
    with open("molecules.json") as fh:
        mols = json.load(fh)
    name = sys.argv[1] if len(sys.argv) > 1 else "water"
    m = mols[name]
    aev = compute_aev(m["znums"], m["coords"])
    print(name, "aev shape", aev.shape, "sum", aev.sum())
