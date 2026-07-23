/**
 * ANI-2x atomic environment vector (AEV): dense all-pairs float64 forward, plus
 * the analytic reverse-mode adjoint used for forces.
 *
 * The forward is an exact port of the spike's validated `aev.js`
 * (numerically identical to torchani to ~1e-15). The adjoint contracts an
 * incoming cotangent g = dE/dAEV (shape [N,1008]) with dAEV/dr and scatters the
 * result into a coordinate-gradient buffer, i.e. it computes
 *   gradCoords_k = sum_{i,c} g[i,c] * d(AEV[i,c])/d(r_k)
 * without ever materialising the full [N,1008,N,3] Jacobian. Forces are then
 * -gradCoords. All geometry is in Angstrom; gradients are in (energy)/Angstrom.
 *
 * Angular note: torchani damps the cosine by 0.95 before arccos
 * (angle = arccos(0.95*cos)) to keep the theta derivative non-singular at the
 * collinear limit; that factor is carried through the adjoint exactly (it also
 * appears in dtheta/dcos), so 1 - (0.95*cos)^2 >= 1 - 0.9025 > 0 always.
 */

export interface RadialParams {
  eta: number;
  shifts: number[];
  cutoff: number;
}

export interface AngularParams {
  eta: number;
  zeta: number;
  shifts: number[];
  sections: number[];
  cutoff: number;
}

export interface AevParams {
  radialLen: number;
  angularLen: number;
  numSpecies: number;
  numSpeciesPairs: number;
  radial: RadialParams;
  angular: AngularParams;
}

/** Precomputed, allocation-free AEV kernel bound to a fixed parameter set. */
export class AevComputer {
  readonly aevDim: number;
  readonly radialLen: number;
  private readonly RcR: number;
  private readonly etaR: number;
  private readonly shfR: Float64Array;
  private readonly NR: number;
  private readonly RcA: number;
  private readonly etaA: number;
  private readonly zeta: number;
  private readonly shfA: Float64Array;
  private readonly sections: Float64Array;
  private readonly NA: number;
  private readonly NZ: number;
  private readonly NAF: number;
  private readonly triu: Int32Array; // numSpecies*numSpecies -> pair index

  constructor(p: AevParams) {
    this.radialLen = p.radialLen;
    this.aevDim = p.radialLen + p.angularLen;
    this.RcR = p.radial.cutoff;
    this.etaR = p.radial.eta;
    this.shfR = Float64Array.from(p.radial.shifts);
    this.NR = this.shfR.length;
    this.RcA = p.angular.cutoff;
    this.etaA = p.angular.eta;
    this.zeta = p.angular.zeta;
    this.shfA = Float64Array.from(p.angular.shifts);
    this.sections = Float64Array.from(p.angular.sections);
    this.NA = this.shfA.length;
    this.NZ = this.sections.length;
    this.NAF = this.NA * this.NZ;

    const ns = p.numSpecies;
    this.triu = new Int32Array(ns * ns);
    let c = 0;
    for (let i = 0; i < ns; i++) {
      for (let j = i; j < ns; j++) {
        this.triu[i * ns + j] = c;
        this.triu[j * ns + i] = c;
        c++;
      }
    }
    this.nSpecies = ns;
  }
  private readonly nSpecies: number;

  private fc(d: number, rc: number): number {
    return 0.5 * Math.cos((Math.PI * d) / rc) + 0.5;
  }
  /** d/dd of the cutoff cosine. */
  private dfc(d: number, rc: number): number {
    return -0.5 * (Math.PI / rc) * Math.sin((Math.PI * d) / rc);
  }

  /** Forward AEV. `species` are 0-based element indices. Returns Float64Array[N*aevDim]. */
  forward(species: Int32Array, coords: Float64Array, out?: Float64Array): Float64Array {
    const N = species.length;
    const AEV = this.aevDim;
    const RADIAL_LEN = this.radialLen;
    const NR = this.NR;
    const NA = this.NA;
    const NZ = this.NZ;
    const NAF = this.NAF;
    const ns = this.nSpecies;
    const aev = out ?? new Float64Array(N * AEV);
    aev.fill(0);

    const dist = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = coords[i * 3]! - coords[j * 3]!;
        const dy = coords[i * 3 + 1]! - coords[j * 3 + 1]!;
        const dz = coords[i * 3 + 2]! - coords[j * 3 + 2]!;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dist[i * N + j] = d;
        dist[j * N + i] = d;
      }
    }

    // radial
    for (let i = 0; i < N; i++) {
      const row = i * AEV;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = dist[i * N + j]!;
        if (d > this.RcR) continue;
        const cut = 0.25 * this.fc(d, this.RcR);
        const base = row + species[j]! * NR;
        for (let r = 0; r < NR; r++) {
          const dd = d - this.shfR[r]!;
          aev[base + r] = aev[base + r]! + cut * Math.exp(-this.etaR * dd * dd);
        }
      }
    }

    // angular
    const neigh = new Int32Array(N);
    for (let j = 0; j < N; j++) {
      let nn = 0;
      for (let k = 0; k < N; k++) {
        if (k !== j && dist[j * N + k]! <= this.RcA) neigh[nn++] = k;
      }
      const row = j * AEV;
      for (let ai = 0; ai < nn; ai++) {
        const a = neigh[ai]!;
        const d_ja = dist[j * N + a]!;
        const fca = this.fc(d_ja, this.RcA);
        const vax = coords[a * 3]! - coords[j * 3]!;
        const vay = coords[a * 3 + 1]! - coords[j * 3 + 1]!;
        const vaz = coords[a * 3 + 2]! - coords[j * 3 + 2]!;
        for (let bi = ai + 1; bi < nn; bi++) {
          const b = neigh[bi]!;
          const d_jb = dist[j * N + b]!;
          const vbx = coords[b * 3]! - coords[j * 3]!;
          const vby = coords[b * 3 + 1]! - coords[j * 3 + 1]!;
          const vbz = coords[b * 3 + 2]! - coords[j * 3 + 2]!;
          const cosang =
            (vax * vbx + vay * vby + vaz * vbz) / Math.max(d_ja * d_jb, 1e-10);
          const angle = Math.acos(0.95 * cosang);
          const fac = fca * this.fc(d_jb, this.RcA);
          const mean = 0.5 * (d_ja + d_jb);
          const pairidx = this.triu[species[a]! * ns + species[b]!]!;
          const base = row + RADIAL_LEN + pairidx * NAF;
          for (let r = 0; r < NA; r++) {
            const dm = mean - this.shfA[r]!;
            const rad = Math.exp(-this.etaA * dm * dm);
            const off = base + r * NZ;
            for (let z = 0; z < NZ; z++) {
              const cz = 0.5 * (1.0 + Math.cos(angle - this.sections[z]!));
              aev[off + z] = aev[off + z]! + rad * 2.0 * Math.pow(cz, this.zeta) * fac;
            }
          }
        }
      }
    }
    return aev;
  }

  /**
   * Reverse-mode adjoint: accumulate gradCoords += sum_{i,c} g[i,c] * dAEV[i,c]/dr.
   * `g` is the cotangent [N*aevDim]; `gradCoords` [N*3] is accumulated in place
   * (caller zeroes it). Recomputes geometry to mirror the forward exactly.
   */
  adjoint(
    species: Int32Array,
    coords: Float64Array,
    g: Float64Array,
    gradCoords: Float64Array,
  ): void {
    const N = species.length;
    const AEV = this.aevDim;
    const RADIAL_LEN = this.radialLen;
    const NR = this.NR;
    const NA = this.NA;
    const NZ = this.NZ;
    const NAF = this.NAF;
    const ns = this.nSpecies;

    const dist = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = coords[i * 3]! - coords[j * 3]!;
        const dy = coords[i * 3 + 1]! - coords[j * 3 + 1]!;
        const dz = coords[i * 3 + 2]! - coords[j * 3 + 2]!;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dist[i * N + j] = d;
        dist[j * N + i] = d;
      }
    }

    // ---- radial adjoint ----
    for (let i = 0; i < N; i++) {
      const row = i * AEV;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = dist[i * N + j]!;
        if (d > this.RcR) continue;
        const fcv = this.fc(d, this.RcR);
        const dfcv = this.dfc(d, this.RcR);
        const base = row + species[j]! * NR;
        // S = sum_r g[i,base+r] * d(term_r)/dd
        // term_r = 0.25 * exp(-etaR*(d-s)^2) * fc(d)
        let S = 0;
        for (let r = 0; r < NR; r++) {
          const dd = d - this.shfR[r]!;
          const e = Math.exp(-this.etaR * dd * dd);
          const dterm = 0.25 * e * (-2 * this.etaR * dd * fcv + dfcv);
          S += g[base + r]! * dterm;
        }
        // d depends on r_i (+u) and r_j (-u), u = (r_i - r_j)/d
        const ux = (coords[i * 3]! - coords[j * 3]!) / d;
        const uy = (coords[i * 3 + 1]! - coords[j * 3 + 1]!) / d;
        const uz = (coords[i * 3 + 2]! - coords[j * 3 + 2]!) / d;
        gradCoords[i * 3] = gradCoords[i * 3]! + S * ux;
        gradCoords[i * 3 + 1] = gradCoords[i * 3 + 1]! + S * uy;
        gradCoords[i * 3 + 2] = gradCoords[i * 3 + 2]! + S * uz;
        gradCoords[j * 3] = gradCoords[j * 3]! - S * ux;
        gradCoords[j * 3 + 1] = gradCoords[j * 3 + 1]! - S * uy;
        gradCoords[j * 3 + 2] = gradCoords[j * 3 + 2]! - S * uz;
      }
    }

    // ---- angular adjoint ----
    const neigh = new Int32Array(N);
    const Gp = new Float64Array(NA); // sum_z g*P
    const Gdp = new Float64Array(NA); // sum_z g*dP/dtheta
    const radArr = new Float64Array(NA);
    const dradArr = new Float64Array(NA); // d rad / d mean
    for (let j = 0; j < N; j++) {
      let nn = 0;
      for (let k = 0; k < N; k++) {
        if (k !== j && dist[j * N + k]! <= this.RcA) neigh[nn++] = k;
      }
      const row = j * AEV;
      const jx = coords[j * 3]!;
      const jy = coords[j * 3 + 1]!;
      const jz = coords[j * 3 + 2]!;
      for (let ai = 0; ai < nn; ai++) {
        const a = neigh[ai]!;
        const d_ja = dist[j * N + a]!;
        const fca = this.fc(d_ja, this.RcA);
        const dfca = this.dfc(d_ja, this.RcA);
        const vax = coords[a * 3]! - jx;
        const vay = coords[a * 3 + 1]! - jy;
        const vaz = coords[a * 3 + 2]! - jz;
        for (let bi = ai + 1; bi < nn; bi++) {
          const b = neigh[bi]!;
          const d_jb = dist[j * N + b]!;
          const fcb = this.fc(d_jb, this.RcA);
          const dfcb = this.dfc(d_jb, this.RcA);
          const vbx = coords[b * 3]! - jx;
          const vby = coords[b * 3 + 1]! - jy;
          const vbz = coords[b * 3 + 2]! - jz;

          const dot = vax * vbx + vay * vby + vaz * vbz;
          const denom = Math.max(d_ja * d_jb, 1e-10);
          const cosang = dot / denom;
          const cd = 0.95 * cosang;
          const angle = Math.acos(cd);
          const fac = fca * fcb;
          const mean = 0.5 * (d_ja + d_jb);
          const pairidx = this.triu[species[a]! * ns + species[b]!]!;
          const base = row + RADIAL_LEN + pairidx * NAF;

          // precompute per-r radial factors
          for (let r = 0; r < NA; r++) {
            const dm = mean - this.shfA[r]!;
            const rad = Math.exp(-this.etaA * dm * dm);
            radArr[r] = rad;
            dradArr[r] = rad * (-2 * this.etaA * dm); // d rad / d mean
          }
          // accumulate per-r Gp, Gdp over z; also collect scalar sums
          let Sdja_rad = 0; // sum_r Gp[r]*rad[r]   (for dfac term)
          let Sdja_dmean = 0; // sum_r Gp[r]*drad[r] (for mean term)
          let Stheta = 0; // sum_r rad[r]*Gdp[r]  (times 2*fac later)
          for (let r = 0; r < NA; r++) {
            const off = base + r * NZ;
            let gp = 0;
            let gdp = 0;
            for (let z = 0; z < NZ; z++) {
              const arg = angle - this.sections[z]!;
              const cz = 0.5 * (1.0 + Math.cos(arg));
              const P = Math.pow(cz, this.zeta);
              // dP/dtheta = zeta*cz^(zeta-1) * dcz/dtheta ; dcz/dtheta = -0.5*sin(arg)
              const dP =
                cz > 0
                  ? this.zeta * Math.pow(cz, this.zeta - 1) * (-0.5 * Math.sin(arg))
                  : 0;
              const gv = g[off + z]!;
              gp += gv * P;
              gdp += gv * dP;
            }
            Gp[r] = gp;
            Gdp[r] = gdp;
            Sdja_rad += gp * radArr[r]!;
            Sdja_dmean += gp * dradArr[r]!;
            Stheta += radArr[r]! * gdp;
          }
          // term = 2 * fac * rad * P
          // dterm/dd_ja = 2*P*(dfac_dja*rad + fac*0.5*drad)  -> summed:
          const dfac_dja = dfca * fcb;
          const dfac_djb = fca * dfcb;
          const Sd_ja =
            2 * (dfac_dja * Sdja_rad + fac * 0.5 * Sdja_dmean);
          const Sd_jb =
            2 * (dfac_djb * Sdja_rad + fac * 0.5 * Sdja_dmean);
          const SthetaFull = 2 * fac * Stheta; // sum g * dterm/dtheta

          // spatial gradients of d_ja, d_jb
          const ua_x = vax / d_ja;
          const ua_y = vay / d_ja;
          const ua_z = vaz / d_ja;
          const ub_x = vbx / d_jb;
          const ub_y = vby / d_jb;
          const ub_z = vbz / d_jb;

          // dtheta/dcos = -0.95 / sqrt(1 - (0.95 cos)^2)
          const dtheta_dcos = -0.95 / Math.sqrt(Math.max(1 - cd * cd, 1e-30));
          const dth = SthetaFull * dtheta_dcos;

          // dcos/dr_a = vb/(d_ja d_jb) - cos * va/d_ja^2
          const inv = 1 / denom;
          const dcosa_x = vbx * inv - cosang * vax / (d_ja * d_ja);
          const dcosa_y = vby * inv - cosang * vay / (d_ja * d_ja);
          const dcosa_z = vbz * inv - cosang * vaz / (d_ja * d_ja);
          const dcosb_x = vax * inv - cosang * vbx / (d_jb * d_jb);
          const dcosb_y = vay * inv - cosang * vby / (d_jb * d_jb);
          const dcosb_z = vaz * inv - cosang * vbz / (d_jb * d_jb);

          // accumulate onto a
          const ga_x = Sd_ja * ua_x + dth * dcosa_x;
          const ga_y = Sd_ja * ua_y + dth * dcosa_y;
          const ga_z = Sd_ja * ua_z + dth * dcosa_z;
          const gb_x = Sd_jb * ub_x + dth * dcosb_x;
          const gb_y = Sd_jb * ub_y + dth * dcosb_y;
          const gb_z = Sd_jb * ub_z + dth * dcosb_z;

          gradCoords[a * 3] = gradCoords[a * 3]! + ga_x;
          gradCoords[a * 3 + 1] = gradCoords[a * 3 + 1]! + ga_y;
          gradCoords[a * 3 + 2] = gradCoords[a * 3 + 2]! + ga_z;
          gradCoords[b * 3] = gradCoords[b * 3]! + gb_x;
          gradCoords[b * 3 + 1] = gradCoords[b * 3 + 1]! + gb_y;
          gradCoords[b * 3 + 2] = gradCoords[b * 3 + 2]! + gb_z;
          // center j gets the negative sum (translation invariance)
          gradCoords[j * 3] = gradCoords[j * 3]! - (ga_x + gb_x);
          gradCoords[j * 3 + 1] = gradCoords[j * 3 + 1]! - (ga_y + gb_y);
          gradCoords[j * 3 + 2] = gradCoords[j * 3 + 2]! - (ga_z + gb_z);
        }
      }
    }
  }
}
