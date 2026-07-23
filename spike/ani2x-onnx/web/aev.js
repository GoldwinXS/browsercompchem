// Dense all-pairs ANI-2x AEV in plain JS. Mirrors aev_ref.py (validated to 1e-15
// against torchani). computeAEV returns Float32Array of length N*1008 (row-major).
export function makeAEV(params) {
  const R = params.radial, A = params.angular;
  const RcR = R.cutoff, etaR = R.eta, shfR = Float64Array.from(R.shifts);
  const RcA = A.cutoff, etaA = A.eta, zeta = A.zeta;
  const shfA = Float64Array.from(A.shifts), sections = Float64Array.from(A.sections);
  const NR = shfR.length, NA = shfA.length, NZ = sections.length, NAF = NA * NZ;
  const RADIAL_LEN = params.radial_len, ANGULAR_LEN = params.angular_len;
  const AEV_DIM = RADIAL_LEN + ANGULAR_LEN;
  const NS = params.num_species, NSP = params.num_species_pairs;
  const ZMAP = { 1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6 };

  // triu index matrix (symmetric), same construction as torchani
  const TRIU = Array.from({ length: NS }, () => new Int32Array(NS));
  let c = 0;
  for (let i = 0; i < NS; i++) for (let j = i; j < NS; j++) { TRIU[i][j] = c; TRIU[j][i] = c; c++; }

  const fc = (d, rc) => 0.5 * Math.cos(Math.PI * d / rc) + 0.5;

  function speciesIdx(znums) {
    const sp = new Int32Array(znums.length);
    for (let i = 0; i < znums.length; i++) sp[i] = ZMAP[znums[i]];
    return sp;
  }

  // Returns Float32Array (N*AEV_DIM). If out provided, reuse it.
  function computeAEV(znums, coords, sp, out) {
    const N = znums.length;
    if (!sp) sp = speciesIdx(znums);
    const aev = out || new Float32Array(N * AEV_DIM);
    aev.fill(0);
    // pairwise distances
    const dist = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = coords[i * 3] - coords[j * 3];
        const dy = coords[i * 3 + 1] - coords[j * 3 + 1];
        const dz = coords[i * 3 + 2] - coords[j * 3 + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dist[i * N + j] = d; dist[j * N + i] = d;
      }
    }
    // radial
    for (let i = 0; i < N; i++) {
      const row = i * AEV_DIM;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = dist[i * N + j];
        if (d > RcR) continue;
        const cut = 0.25 * fc(d, RcR);
        const base = row + sp[j] * NR;
        for (let r = 0; r < NR; r++) {
          const dd = d - shfR[r];
          aev[base + r] += cut * Math.exp(-etaR * dd * dd);
        }
      }
    }
    // angular
    const neigh = new Int32Array(N);
    for (let j = 0; j < N; j++) {
      let nn = 0;
      for (let k = 0; k < N; k++) {
        if (k !== j && dist[j * N + k] <= RcA) neigh[nn++] = k;
      }
      const row = j * AEV_DIM;
      for (let ai = 0; ai < nn; ai++) {
        const a = neigh[ai];
        const d_ja = dist[j * N + a];
        const fca = fc(d_ja, RcA);
        const vax = coords[a * 3] - coords[j * 3];
        const vay = coords[a * 3 + 1] - coords[j * 3 + 1];
        const vaz = coords[a * 3 + 2] - coords[j * 3 + 2];
        for (let bi = ai + 1; bi < nn; bi++) {
          const b = neigh[bi];
          const d_jb = dist[j * N + b];
          const vbx = coords[b * 3] - coords[j * 3];
          const vby = coords[b * 3 + 1] - coords[j * 3 + 1];
          const vbz = coords[b * 3 + 2] - coords[j * 3 + 2];
          let cosang = (vax * vbx + vay * vby + vaz * vbz) / Math.max(d_ja * d_jb, 1e-10);
          const angle = Math.acos(0.95 * cosang);
          const fac = fca * fc(d_jb, RcA);
          const mean = 0.5 * (d_ja + d_jb);
          const pairidx = TRIU[sp[a]][sp[b]];
          const base = row + RADIAL_LEN + pairidx * NAF;
          for (let r = 0; r < NA; r++) {
            const dm = mean - shfA[r];
            const rad = Math.exp(-etaA * dm * dm);
            const off = base + r * NZ;
            for (let z = 0; z < NZ; z++) {
              const cz = 0.5 * (1.0 + Math.cos(angle - sections[z]));
              aev[off + z] += rad * 2.0 * Math.pow(cz, zeta) * fac;
            }
          }
        }
      }
    }
    return aev;
  }

  return { computeAEV, speciesIdx, AEV_DIM, NS };
}
