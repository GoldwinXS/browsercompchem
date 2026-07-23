import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider, EnergyForces } from "../potentials/types.js";
import { covalentRadius } from "./covalentRadii.js";

/**
 * Classical topology-aware force field: harmonic bonds + harmonic angles +
 * soft non-bonded repulsion, built directly from the known bond graph (no
 * neural network, no RDKit calls). Its sole job is to turn a handful of
 * scattered random 3D starting points into an UNTANGLED, roughly-correct
 * conformer -- topology-consistent bond lengths, sane angles, no atom sitting
 * on top of another -- cheaply enough to try several starts and keep the best.
 * That geometry then seeds the real ANI-2x + FIRE polish. This is deliberately
 * a distance-geometry-flavored relaxation (conceptually a mini ETKDG: embed
 * randomly, then satisfy local geometric constraints by gradient descent)
 * rather than an attempt at a real molecular-mechanics force field -- v1
 * skips torsions/impropers entirely (see the class docstring below) because
 * the ANI-2x polish handles that fine detail; this FF only has to get the
 * gross topology (which atoms are near which) right.
 */

/** A bond from the known connectivity: 0-based atom indices + V2000-style order (1 single, 2 double, 3 triple, 4 aromatic). */
export interface FFBond {
  i: number;
  j: number;
  order: number;
}

/**
 * Bond-order shortening factors applied to the covalent-radii-SUM target
 * bond length (single-bond r0 = radius_i + radius_j from Cordero et al. 2008,
 * see covalentRadii.ts). These are generic ratios tuned against well-known
 * C-C reference bond lengths (single 1.54, aromatic 1.40, double 1.33, triple
 * 1.20 Angstrom vs. the covalent-radii-sum single-bond target of 1.52
 * Angstrom) rather than element-pair-specific double/triple radii (cf.
 * Pyykkoe, "Additive covalent radii for single-, double-, and triple-bonded
 * atoms", J. Phys. Chem. A 2009, 113, 12, 5806-5812, which tabulates those
 * per-element) -- a uniform ratio is simpler and good enough for a classical
 * SEED geometry that the ANI-2x polish subsequently refines to chemical
 * accuracy.
 */
const BOND_ORDER_SHRINK: Record<number, number> = {
  1: 1.0, // single
  2: 0.875, // double
  3: 0.79, // triple
  4: 0.92, // aromatic (V2000 bond type 4)
};

function bondOrderShrink(order: number): number {
  return BOND_ORDER_SHRINK[order] ?? 1.0;
}

const DEG2RAD = Math.PI / 180;

/**
 * Ideal bond angle (degrees) at a center atom from a simple coordination-
 * number + bond-order heuristic ("hybridization by neighbor count"):
 *   - 2 neighbors, at least one triple bond OR both bonds double (a cumulated-
 *     double center like CO2's carbon or an allene's central carbon) -> sp,
 *     linear, 180.
 *   - 2 neighbors otherwise (a divalent bent center: ether/water oxygen,
 *     imine nitrogen, ...) -> no clean hybridization label; 109.47 is used as
 *     a reasonable classical-seed default (the ANI-2x polish moves water to
 *     its true ~104.5 degrees; the classical stage only needs a plausible
 *     starting bend, not the exact value).
 *   - 3 neighbors -> sp2, trigonal planar, 120 (covers aromatic ring atoms,
 *     alkene/carbonyl carbons, amide nitrogens correctly; a known
 *     over-simplification for pyramidal sp3 centers with exactly 3 explicit
 *     single-bonded neighbors, e.g. a simple amine nitrogen -- treated as
 *     planar here, which the ANI-2x polish subsequently pyramidalizes back to
 *     the correct ~107 degrees. Documented as a v1 limitation.)
 *   - 4+ neighbors -> sp3, tetrahedral, 109.47.
 */
function angleTheta0Deg(nNeighbors: number, ordersAtCenter: number[]): number {
  if (nNeighbors === 2) {
    const hasTriple = ordersAtCenter.some((o) => o === 3);
    const doubleCount = ordersAtCenter.filter((o) => o === 2).length;
    if (hasTriple || doubleCount >= 2) return 180;
    return 109.47;
  }
  if (nNeighbors === 3) return 120;
  return 109.47;
}

export interface ForceFieldOptions {
  /** Harmonic bond-stretch force constant (energy units / Angstrom^2). */
  kBond?: number;
  /** Harmonic angle-bend force constant (energy units / radian^2). */
  kAngle?: number;
  /** Non-bonded soft-repulsion strength (energy units). */
  repEpsilon?: number;
  /** Exponent n in the repulsive u^n falloff (u = sigma^2/(r^2+softening^2)); higher = steeper wall. */
  repExponent?: number;
  /** Softening length (Angstrom) that keeps the repulsion finite as r -> 0. */
  repSoftening?: number;
  /**
   * Flat pad (Angstrom) ADDED to the covalent-radii-sum sigma used ONLY by
   * the non-bonded repulsion (bond r0 targets are unaffected). An additive
   * pad -- rather than a multiplicative scale -- is used deliberately: the
   * shared validity gate's flat clash distance (1.15 A, see validity.ts,
   * originally tuned for the ANI-2x worker's POST-relaxation check) is larger
   * than the bare covalent-radii sum for almost any H-involving pair (e.g.
   * H...H = 0.62 A, C...H = 1.07 A), so the classical repulsion needs to push
   * those particular pairs out a bit further than pure covalent radii alone
   * would. A flat additive pad does that proportionally MORE for light pairs
   * (where it matters) and proportionally LESS for heavy pairs -- a
   * multiplicative scale large enough to fix H...H would also inflate
   * heavy-atom 1-4+ non-bonded distances (e.g. a ring's para C...C) well past
   * their normal, perfectly valid separation. Default 0.5 A was picked
   * empirically: it clears the gate for a 30-carbon alkane's H...H contacts
   * (see embed3d.test.ts) while leaving benzene's ring bond length within ~1%
   * of its 1.40 A target.
   */
  repSigmaPad?: number;
}

const DEFAULTS: Required<ForceFieldOptions> = {
  kBond: 400,
  kAngle: 80,
  repEpsilon: 60,
  repExponent: 6,
  repSoftening: 0.2,
  repSigmaPad: 0.5,
};

interface BondTerm {
  i: number;
  j: number;
  r0: number;
}
interface AngleTerm {
  center: number;
  a: number;
  b: number;
  theta0: number; // radians
}
interface RepPair {
  i: number;
  j: number;
  sigma2: number;
}

/**
 * Classical energy+gradient provider built from a molecule's known topology.
 * Implements the engine's EnergyForceProvider seam so it can be relaxed with
 * the SAME FireOptimizer used everywhere else (see embed3d.ts).
 *
 * NOTE ON UNITS: like LennardJonesProvider, this operates in its own
 * arbitrary "classical FF" energy/length units (Angstrom for length, an
 * uncalibrated energy unit for kBond/kAngle/repEpsilon) -- it is never handed
 * to anything that interprets energy chemically (it only ever seeds a
 * geometry for the real ANI-2x provider), so no Hartree calibration is
 * attempted.
 *
 * DEFERRED (v1): torsion/dihedral and out-of-plane (improper/planarity)
 * terms are intentionally omitted. The bond+angle+repulsion terms already
 * fix local bond lengths, local angles, and gross non-bonded overlap --
 * exactly what a 2D+jitter seed gets wrong for large molecules -- while
 * fine conformational detail (ring puckering preferences, sp2 planarity,
 * rotamer preference) is left to the ANI-2x polish, which models it far
 * more accurately than a hand-tuned torsion term would. Keeping the FF to
 * three terms also keeps it fast and numerically robust from wildly
 * scattered random starts.
 */
export class ClassicalForceField implements EnergyForceProvider {
  readonly name = "classical-topology-ff";
  private readonly bondTerms: BondTerm[];
  private readonly angleTerms: AngleTerm[];
  private readonly repPairs: RepPair[];
  private readonly opts: Required<ForceFieldOptions>;

  constructor(atoms: string[], bonds: FFBond[], opts: ForceFieldOptions = {}) {
    this.opts = { ...DEFAULTS, ...opts };
    const n = atoms.length;

    interface Nbr {
      atom: number;
      order: number;
    }
    const neighbors: Nbr[][] = Array.from({ length: n }, () => []);
    for (const b of bonds) {
      neighbors[b.i]!.push({ atom: b.j, order: b.order });
      neighbors[b.j]!.push({ atom: b.i, order: b.order });
    }

    // --- bonds: r0 from covalent-radii sum, shortened by bond order ---
    this.bondTerms = bonds.map((b) => {
      const rsum = covalentRadius(atoms[b.i]!) + covalentRadius(atoms[b.j]!);
      return { i: b.i, j: b.j, r0: rsum * bondOrderShrink(b.order) };
    });

    // --- angles: one per neighbor pair at every atom with >=2 neighbors ---
    // Also builds the 1-3 ("shares a common neighbor") exclusion set the
    // non-bonded repulsion loop below must skip, alongside the direct 1-2
    // bonded pairs.
    const excluded = new Set<number>();
    for (const b of bonds) {
      excluded.add(b.i * n + b.j);
      excluded.add(b.j * n + b.i);
    }
    this.angleTerms = [];
    for (let center = 0; center < n; center++) {
      const nbrs = neighbors[center]!;
      if (nbrs.length < 2) continue;
      const orders = nbrs.map((x) => x.order);
      const theta0 = angleTheta0Deg(nbrs.length, orders) * DEG2RAD;
      for (let p = 0; p < nbrs.length; p++) {
        for (let q = p + 1; q < nbrs.length; q++) {
          const a = nbrs[p]!.atom;
          const b2 = nbrs[q]!.atom;
          this.angleTerms.push({ center, a, b: b2, theta0 });
          excluded.add(a * n + b2);
          excluded.add(b2 * n + a);
        }
      }
    }

    // --- non-bonded repulsion: every remaining (non 1-2, non 1-3) pair ---
    this.repPairs = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (excluded.has(i * n + j)) continue;
        const sigma = covalentRadius(atoms[i]!) + covalentRadius(atoms[j]!) + this.opts.repSigmaPad;
        this.repPairs.push({ i, j, sigma2: sigma * sigma });
      }
    }
  }

  async energyForces(mol: Molecule): Promise<EnergyForces> {
    return Promise.resolve(this.energyForcesSync(mol.positions));
  }

  /** Synchronous core (used directly by embed3d's inner loop and by tests). */
  energyForcesSync(positions: Float64Array): EnergyForces {
    const forces = new Float64Array(positions.length);
    let energy = 0;
    const { kBond, kAngle, repEpsilon, repExponent, repSoftening } = this.opts;
    const softening2 = repSoftening * repSoftening;

    // --- harmonic bonds: E = 0.5 kBond (r - r0)^2 ---
    for (const { i, j, r0 } of this.bondTerms) {
      const dx = positions[3 * i]! - positions[3 * j]!;
      const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
      const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
      const r = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6);
      const dr = r - r0;
      energy += 0.5 * kBond * dr * dr;
      const c = (kBond * dr) / r; // dE/dr * (1/r), so c*d(x,y,z) = gradient
      const gx = c * dx;
      const gy = c * dy;
      const gz = c * dz;
      forces[3 * i] = forces[3 * i]! - gx;
      forces[3 * i + 1] = forces[3 * i + 1]! - gy;
      forces[3 * i + 2] = forces[3 * i + 2]! - gz;
      forces[3 * j] = forces[3 * j]! + gx;
      forces[3 * j + 1] = forces[3 * j + 1]! + gy;
      forces[3 * j + 2] = forces[3 * j + 2]! + gz;
    }

    // --- harmonic angles: E = 0.5 kAngle (theta - theta0)^2 ---
    // Standard bond-angle-bending gradient (as in e.g. Allen & Tildesley /
    // common MD codes): theta = acos(cosTheta) between rA = pos[a]-pos[center]
    // and rB = pos[b]-pos[center]; chain rule through cosTheta.
    for (const { center, a, b, theta0 } of this.angleTerms) {
      const ax = positions[3 * a]! - positions[3 * center]!;
      const ay = positions[3 * a + 1]! - positions[3 * center + 1]!;
      const az = positions[3 * a + 2]! - positions[3 * center + 2]!;
      const bx = positions[3 * b]! - positions[3 * center]!;
      const by = positions[3 * b + 1]! - positions[3 * center + 1]!;
      const bz = positions[3 * b + 2]! - positions[3 * center + 2]!;
      const lenA = Math.max(Math.sqrt(ax * ax + ay * ay + az * az), 1e-6);
      const lenB = Math.max(Math.sqrt(bx * bx + by * by + bz * bz), 1e-6);
      let cosTheta = (ax * bx + ay * by + az * bz) / (lenA * lenB);
      cosTheta = Math.min(1, Math.max(-1, cosTheta));
      const theta = Math.acos(cosTheta);
      const dTheta = theta - theta0;
      energy += 0.5 * kAngle * dTheta * dTheta;

      const sinTheta = Math.sqrt(Math.max(1 - cosTheta * cosTheta, 0));
      if (sinTheta < 1e-6) continue; // near-linear/near-zero: gradient direction ill-defined, energy already counted

      const dEdTheta = kAngle * dTheta;
      const invSin = -1 / sinTheta;
      // d(cosTheta)/d(rA), d(cosTheta)/d(rB)
      const dCosA_x = bx / (lenA * lenB) - (cosTheta * ax) / (lenA * lenA);
      const dCosA_y = by / (lenA * lenB) - (cosTheta * ay) / (lenA * lenA);
      const dCosA_z = bz / (lenA * lenB) - (cosTheta * az) / (lenA * lenA);
      const dCosB_x = ax / (lenA * lenB) - (cosTheta * bx) / (lenB * lenB);
      const dCosB_y = ay / (lenA * lenB) - (cosTheta * by) / (lenB * lenB);
      const dCosB_z = az / (lenA * lenB) - (cosTheta * bz) / (lenB * lenB);

      const gAx = dEdTheta * invSin * dCosA_x;
      const gAy = dEdTheta * invSin * dCosA_y;
      const gAz = dEdTheta * invSin * dCosA_z;
      const gBx = dEdTheta * invSin * dCosB_x;
      const gBy = dEdTheta * invSin * dCosB_y;
      const gBz = dEdTheta * invSin * dCosB_z;

      // gradient wrt center = -(gradA + gradB); forces = -gradient
      forces[3 * a] = forces[3 * a]! - gAx;
      forces[3 * a + 1] = forces[3 * a + 1]! - gAy;
      forces[3 * a + 2] = forces[3 * a + 2]! - gAz;
      forces[3 * b] = forces[3 * b]! - gBx;
      forces[3 * b + 1] = forces[3 * b + 1]! - gBy;
      forces[3 * b + 2] = forces[3 * b + 2]! - gBz;
      forces[3 * center] = forces[3 * center]! + (gAx + gBx);
      forces[3 * center + 1] = forces[3 * center + 1]! + (gAy + gBy);
      forces[3 * center + 2] = forces[3 * center + 2]! + (gAz + gBz);
    }

    // --- soft non-bonded repulsion: E = repEpsilon * u^n, u = sigma^2/(r^2+softening^2) ---
    // Purely repulsive (monotonically decreasing in r), finite at r=0 (u is
    // capped by sigma^2/softening^2), and smooth everywhere -- same intent as
    // the repulsive r^-12 wall of a Lennard-Jones potential, but with no
    // attractive well: its only job is to keep non-bonded atoms from
    // occupying the same space, not to model real van-der-Waals attraction.
    for (const { i, j, sigma2 } of this.repPairs) {
      const dx = positions[3 * i]! - positions[3 * j]!;
      const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
      const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
      const r2 = dx * dx + dy * dy + dz * dz;
      const denom = r2 + softening2;
      const u = sigma2 / denom;
      const un = Math.pow(u, repExponent);
      energy += repEpsilon * un;
      // dE/dr2 = -n * repEpsilon * u^n / denom ; gradient_i = dE/dr2 * 2*(dx,dy,dz)
      const coeff = (2 * repExponent * repEpsilon * un) / denom; // = -2*dE/dr2, i.e. force magnitude coefficient along +d
      const fx = coeff * dx;
      const fy = coeff * dy;
      const fz = coeff * dz;
      forces[3 * i] = forces[3 * i]! + fx;
      forces[3 * i + 1] = forces[3 * i + 1]! + fy;
      forces[3 * i + 2] = forces[3 * i + 2]! + fz;
      forces[3 * j] = forces[3 * j]! - fx;
      forces[3 * j + 1] = forces[3 * j + 1]! - fy;
      forces[3 * j + 2] = forces[3 * j + 2]! - fz;
    }

    return { energy, forces };
  }
}
