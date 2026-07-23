/**
 * Extended Hueckel Theory (EHT) molecular orbitals.
 *
 * The cheapest of the compute tiers (RDKit force fields / ANI-2x ML potential /
 * this): a one-shot, non-iterative semi-empirical MO method that gives fast,
 * qualitative frontier-orbital (HOMO/LUMO) pictures -- shapes, degeneracies,
 * bonding/antibonding ordering -- not quantitative energetics.
 *
 * Pipeline (Hoffmann, J. Chem. Phys. 39, 1397 (1963)):
 *   1. Build a minimal valence Slater basis (H 1s; C,N,O,F 2s+2p; S,Cl 3s+3p),
 *      as real Cartesian orbitals {s, px, py, pz}.
 *   2. Overlap matrix S from analytic STO overlap integrals (overlap.ts).
 *   3. Diagonal H_ii = VSIP from the parameter table; off-diagonal via the
 *      Wolfsberg-Helmholz approximation  H_ij = 1/2 K S_ij (H_ii + H_jj),
 *      K = 1.75.
 *   4. Solve the generalized eigenproblem H C = S C eps by canonical (Loewdin)
 *      orthogonalization: build S^(-1/2) from the shared Jacobi eigensolver,
 *      transform to the orthonormal problem, diagonalize, back-transform.
 *      Near-singular S directions (eigenvalue < 1e-6) are dropped with a flag.
 *   5. Aufbau-fill the neutral valence electrons (2 per MO) and report
 *      HOMO/LUMO.
 *
 * Units: input positions are Angstrom (as everywhere in this codebase); they are
 * converted to bohr internally because the Slater exponents are bohr^-1.
 */
import type { Molecule } from "../geometry/molecule.js";
import { jacobiEigen } from "../vibrations/jacobi.js";
import {
  EHT_PARAMS,
  EHT_SUPPORTED_ELEMENTS,
  BOHR_PER_ANGSTROM,
  type OrbitalType,
} from "./parameters.js";
import { sigmaFundamental, piFundamental } from "./overlap.js";

/** One basis function: a real Cartesian atomic orbital with its EHT data. */
export interface AtomicOrbital {
  /** Index of the atom this orbital sits on. */
  atomIndex: number;
  /** Real-Cartesian flavour: "s" | "px" | "py" | "pz". */
  type: OrbitalType;
  /** Principal quantum number of the STO. */
  n: number;
  /** Slater exponent, bohr^-1. */
  zeta: number;
  /** Diagonal Hamiltonian element H_ii (eV). */
  hii: number;
}

export interface ExtendedHuckelOptions {
  /** Wolfsberg-Helmholz proportionality constant (default 1.75). */
  wolfsbergHelmholzK?: number;
  /** Overlap eigenvalues below this are treated as linear dependencies and
   * dropped from the orthonormal basis (default 1e-6). */
  overlapThreshold?: number;
}

export interface ExtendedHuckelResult {
  /** Number of basis functions (rows of the AO basis / of S). */
  nBasisFunctions: number;
  /** Number of molecular orbitals (== nBasisFunctions unless S was singular). */
  nMO: number;
  /** Total valence electrons occupied (neutral atoms minus molecular charge). */
  nElectrons: number;
  /** MO energies, ascending, eV. Length nMO. */
  orbitalEnergies: Float64Array;
  /** Occupation of each MO (2, 1, or 0). Length nMO. */
  occupations: Float64Array;
  /** Index of the highest occupied MO (-1 if no electrons). */
  homoIndex: number;
  /** Index of the lowest unoccupied MO (nMO if fully filled). */
  lumoIndex: number;
  /** MO coefficients in the AO basis, layout coefficients[ao * nMO + mo]. */
  coefficients: Float64Array;
  /** AO metadata, length nBasisFunctions (index-aligned with S rows). */
  aos: AtomicOrbital[];
  /** Overlap matrix S, row-major nBasis x nBasis (diagnostics / C^T S C tests). */
  overlap: Float64Array;
  /** Atom positions in bohr, flat 3N (for grid evaluation). */
  positionsBohr: Float64Array;
  /** True if any overlap eigenvalue was below the threshold and dropped. */
  singular: boolean;
  /** How many near-singular directions were dropped. */
  droppedCount: number;
}

const AXIS: Record<string, number> = { px: 0, py: 1, pz: 2 };

/** Build the flat AO basis for a molecule (throws on unsupported elements). */
export function buildBasis(symbols: string[]): AtomicOrbital[] {
  const aos: AtomicOrbital[] = [];
  for (let a = 0; a < symbols.length; a++) {
    const el = symbols[a]!;
    const param = EHT_PARAMS[el];
    if (!param) {
      throw new Error(
        `extendedHuckel: element "${el}" is outside the EHT set ` +
          `(${EHT_SUPPORTED_ELEMENTS.join(", ")}).`,
      );
    }
    aos.push({ atomIndex: a, type: "s", n: param.s.n, zeta: param.s.zeta, hii: param.s.hii });
    if (param.p) {
      for (const type of ["px", "py", "pz"] as const) {
        aos.push({ atomIndex: a, type, n: param.p.n, zeta: param.p.zeta, hii: param.p.hii });
      }
    }
  }
  return aos;
}

/**
 * Overlap matrix over the AO basis (row-major nBasis x nBasis). Same-atom
 * orbitals are orthonormal (identity block). Different-atom p-orbital overlaps
 * are assembled from the diatomic-frame sigma/pi fundamentals via the
 * Slater-Koster direction-cosine relations.
 */
export function buildOverlapMatrix(
  aos: AtomicOrbital[],
  positionsBohr: Float64Array,
): Float64Array {
  const nb = aos.length;
  const S = new Float64Array(nb * nb);
  for (let i = 0; i < nb; i++) S[i * nb + i] = 1;

  for (let i = 0; i < nb; i++) {
    const ai = aos[i]!;
    for (let j = i + 1; j < nb; j++) {
      const aj = aos[j]!;
      if (ai.atomIndex === aj.atomIndex) continue; // same atom -> orthonormal

      // Bond vector A(i) -> B(j), in bohr, and its unit direction cosines.
      const A = ai.atomIndex;
      const B = aj.atomIndex;
      const dx = positionsBohr[3 * B]! - positionsBohr[3 * A]!;
      const dy = positionsBohr[3 * B + 1]! - positionsBohr[3 * A + 1]!;
      const dz = positionsBohr[3 * B + 2]! - positionsBohr[3 * A + 2]!;
      const R = Math.hypot(dx, dy, dz);
      if (R < 1e-9) continue;
      const d = [dx / R, dy / R, dz / R];

      const lI = ai.type === "s" ? 0 : 1;
      const lJ = aj.type === "s" ? 0 : 1;
      let val = 0;
      if (lI === 0 && lJ === 0) {
        val = sigmaFundamental(ai.n, ai.zeta, 0, aj.n, aj.zeta, 0, R);
      } else if (lI === 0 && lJ === 1) {
        val = d[AXIS[aj.type]!]! * sigmaFundamental(ai.n, ai.zeta, 0, aj.n, aj.zeta, 1, R);
      } else if (lI === 1 && lJ === 0) {
        val = d[AXIS[ai.type]!]! * sigmaFundamental(ai.n, ai.zeta, 1, aj.n, aj.zeta, 0, R);
      } else {
        const vSig = sigmaFundamental(ai.n, ai.zeta, 1, aj.n, aj.zeta, 1, R);
        const vPi = piFundamental(ai.n, ai.zeta, aj.n, aj.zeta, R);
        const di = d[AXIS[ai.type]!]!;
        const dj = d[AXIS[aj.type]!]!;
        const kron = ai.type === aj.type ? 1 : 0;
        val = di * dj * vSig + (kron - di * dj) * vPi;
      }
      S[i * nb + j] = val;
      S[j * nb + i] = val;
    }
  }
  return S;
}

/**
 * Symmetric matrix multiply helper: given flat row-major A (n x m) and B (m x k)
 * return A*B (n x k). Small dense matrices only.
 */
function matMul(A: Float64Array, n: number, m: number, B: Float64Array, k: number): Float64Array {
  const C = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < m; l++) {
      const a = A[i * m + l]!;
      if (a === 0) continue;
      for (let j = 0; j < k; j++) C[i * k + j] = C[i * k + j]! + a * B[l * k + j]!;
    }
  }
  return C;
}

/**
 * Full EHT solve. Async only to match the other engine tiers' provider surface
 * (the work here is synchronous and fast; the expensive part is grid evaluation
 * in grid.ts, streamed from the worker).
 */
export async function extendedHuckel(
  mol: Molecule,
  options: ExtendedHuckelOptions = {},
): Promise<ExtendedHuckelResult> {
  const K = options.wolfsbergHelmholzK ?? 1.75;
  const threshold = options.overlapThreshold ?? 1e-6;

  const aos = buildBasis(mol.symbols); // throws on unsupported elements
  const nb = aos.length;

  const positionsBohr = new Float64Array(mol.positions.length);
  for (let i = 0; i < positionsBohr.length; i++) {
    positionsBohr[i] = mol.positions[i]! * BOHR_PER_ANGSTROM;
  }

  const S = buildOverlapMatrix(aos, positionsBohr);

  // Hamiltonian: diagonal VSIP; off-diagonal Wolfsberg-Helmholz.
  const H = new Float64Array(nb * nb);
  for (let i = 0; i < nb; i++) {
    H[i * nb + i] = aos[i]!.hii;
    for (let j = i + 1; j < nb; j++) {
      const hij = 0.5 * K * S[i * nb + j]! * (aos[i]!.hii + aos[j]!.hii);
      H[i * nb + j] = hij;
      H[j * nb + i] = hij;
    }
  }

  // Canonical orthogonalization: S = V s V^T (Jacobi). Keep directions with
  // s > threshold; X_col = V_col / sqrt(s). Then X^T S X = I on the kept space.
  const { values: sEig, vectors: sVec } = jacobiEigen(S, nb);
  const keep: number[] = [];
  for (let c = 0; c < nb; c++) if (sEig[c]! > threshold) keep.push(c);
  const m = keep.length;
  const droppedCount = nb - m;

  // X is nBasis x m, row-major. sVec[c] is the c-th eigenvector (length nb).
  const X = new Float64Array(nb * m);
  for (let col = 0; col < m; col++) {
    const c = keep[col]!;
    const inv = 1 / Math.sqrt(sEig[c]!);
    const vec = sVec[c]!;
    for (let row = 0; row < nb; row++) X[row * m + col] = vec[row]! * inv;
  }

  // H' = X^T H X  (m x m).
  const HX = matMul(H, nb, nb, X, m); // (nb x m)
  const Hp = new Float64Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) {
      let s = 0;
      for (let row = 0; row < nb; row++) s += X[row * m + a]! * HX[row * m + b]!;
      Hp[a * m + b] = s;
    }
  }

  // Diagonalize the orthonormal problem; back-transform C = X C'.
  const { values: eps, vectors: cpVec } = jacobiEigen(Hp, m);
  // Cp as row-major m x m, column mo = cpVec[mo].
  const Cp = new Float64Array(m * m);
  for (let mo = 0; mo < m; mo++) {
    const v = cpVec[mo]!;
    for (let a = 0; a < m; a++) Cp[a * m + mo] = v[a]!;
  }
  const C = matMul(X, nb, m, Cp, m); // (nb x m), layout C[ao*m + mo]

  const orbitalEnergies = Float64Array.from(eps);

  // Aufbau occupation of the neutral valence electrons (minus molecular charge).
  // Every symbol is already known valid (buildBasis threw otherwise).
  let nElectrons = 0;
  for (const sym of mol.symbols) nElectrons += EHT_PARAMS[sym]!.valenceElectrons;
  nElectrons -= mol.charge;
  const occupations = new Float64Array(m);
  let remaining = nElectrons;
  for (let mo = 0; mo < m && remaining > 0; mo++) {
    const occ = Math.min(2, remaining);
    occupations[mo] = occ;
    remaining -= occ;
  }
  let homoIndex = -1;
  for (let mo = 0; mo < m; mo++) if (occupations[mo]! > 0) homoIndex = mo;
  const lumoIndex = homoIndex + 1;

  return {
    nBasisFunctions: nb,
    nMO: m,
    nElectrons,
    orbitalEnergies,
    occupations,
    homoIndex,
    lumoIndex,
    coefficients: C,
    aos,
    overlap: S,
    positionsBohr,
    singular: droppedCount > 0,
    droppedCount,
  };
}
