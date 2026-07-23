/**
 * Harmonic vibrational (normal-mode) analysis from an EnergyForceProvider.
 *
 *   1. finite-difference the analytic forces -> Cartesian Hessian
 *   2. mass-weight (Hm_ij = H_ij / sqrt(m_i m_j)) and diagonalize (Jacobi)
 *   3. eigenvalues -> signed wavenumbers (cm^-1); imaginary (negative-curvature)
 *      modes come out negative
 *   4. remove the 6 (or 5, linear) near-zero translation/rotation modes
 *   5. convert the surviving eigenvectors back to Cartesian atom displacements
 *      (multiply by 1/sqrt(m)) and normalize
 *
 * The provider must be evaluated at (or very near) a stationary point for the
 * frequencies to be physically meaningful; near-zero-force geometries give real,
 * positive frequencies, while a non-stationary point leaks curvature into the
 * trans/rot block and can produce spurious imaginary modes.
 *
 * Units follow the provider (ANI-2x: Hartree, Angstrom). Frequencies are cm^-1.
 * The returned `modes` are mass-UN-weighted Cartesian displacement vectors
 * (length 3N, [dx0,dy0,dz0,...]) normalized to unit L2 norm, ready to drive an
 * animation of the atoms oscillating along each mode.
 */
import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider } from "../potentials/types.js";
import {
  finiteDifferenceHessian,
  type HessianOptions,
} from "../hessian/finiteDifference.js";
import { jacobiEigen } from "./jacobi.js";
import { ATOMIC_MASS, VIB_CM_PER_SQRT } from "./constants.js";

export interface NormalModesOptions {
  /** Finite-difference step for the Hessian (Angstrom). Default 1e-3. */
  stepSize?: number;
  /** Progress callback forwarded from the Hessian build (done/total columns). */
  onProgress?: (done: number, total: number) => void;
}

export interface NormalModesResult {
  /**
   * The 3N-6 (or 3N-5, linear) vibrational wavenumbers in cm^-1, ascending.
   * Imaginary modes (negative curvature) are reported as NEGATIVE numbers.
   */
  frequencies: number[];
  /**
   * One entry per vibrational frequency (index-aligned with `frequencies`):
   * the mass-un-weighted Cartesian displacement of every atom, length 3N,
   * normalized to unit norm. position(t) = equilibrium + A * mode * sin(...).
   */
  modes: Float64Array[];
  /** True if the geometry was detected linear (5 trans/rot modes removed). */
  isLinear: boolean;
  /** All 3N signed wavenumbers, ascending (includes the near-zero modes). */
  allWavenumbers: number[];
  /** Largest magnitude among the removed (should-be-zero) trans/rot modes, cm^-1. */
  maxResidualTransRot: number;
}

/**
 * Detect linearity from the moment-of-inertia tensor: a linear molecule has one
 * vanishing principal moment. Diatomics are always linear.
 */
export function isLinearGeometry(symbols: string[], pos: Float64Array): boolean {
  const n = symbols.length;
  if (n === 2) return true;
  let mTot = 0;
  const com = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const m = ATOMIC_MASS[symbols[i]!]!;
    mTot += m;
    com[0] = com[0]! + m * pos[3 * i]!;
    com[1] = com[1]! + m * pos[3 * i + 1]!;
    com[2] = com[2]! + m * pos[3 * i + 2]!;
  }
  com[0] = com[0]! / mTot;
  com[1] = com[1]! / mTot;
  com[2] = com[2]! / mTot;
  const I = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const m = ATOMIC_MASS[symbols[i]!]!;
    const x = pos[3 * i]! - com[0]!;
    const y = pos[3 * i + 1]! - com[1]!;
    const z = pos[3 * i + 2]! - com[2]!;
    I[0] = I[0]! + m * (y * y + z * z);
    I[4] = I[4]! + m * (x * x + z * z);
    I[8] = I[8]! + m * (x * x + y * y);
    I[1] = I[1]! - m * x * y;
    I[3] = I[3]! - m * x * y;
    I[2] = I[2]! - m * x * z;
    I[6] = I[6]! - m * x * z;
    I[5] = I[5]! - m * y * z;
    I[7] = I[7]! - m * y * z;
  }
  const { values } = jacobiEigen(I, 3);
  const smallest = values[0]!;
  const largest = values[2]!;
  return largest > 0 && smallest / largest < 1e-3;
}

/**
 * Full harmonic analysis at a (presumed stationary) geometry. Returns
 * frequencies (cm^-1) paired with normalized Cartesian displacement vectors.
 */
export async function computeNormalModes(
  mol: Molecule,
  provider: EnergyForceProvider,
  options: NormalModesOptions = {},
): Promise<NormalModesResult> {
  const n = mol.symbols.length;
  const n3 = 3 * n;

  const hessOpts: HessianOptions = {};
  if (options.stepSize !== undefined) hessOpts.stepSize = options.stepSize;
  if (options.onProgress !== undefined) hessOpts.onProgress = options.onProgress;
  const { hessian: H } = await finiteDifferenceHessian(mol, provider, hessOpts);

  // Mass-weight: Hm_ij = H_ij / sqrt(m_i m_j); keep invSqrtM to un-weight later.
  const invSqrtM = new Float64Array(n3);
  for (let i = 0; i < n; i++) {
    const m = ATOMIC_MASS[mol.symbols[i]!]!;
    const v = 1 / Math.sqrt(m);
    invSqrtM[3 * i] = v;
    invSqrtM[3 * i + 1] = v;
    invSqrtM[3 * i + 2] = v;
  }
  const Hm = new Float64Array(n3 * n3);
  for (let i = 0; i < n3; i++)
    for (let j = 0; j < n3; j++)
      Hm[i * n3 + j] = H[i * n3 + j]! * invSqrtM[i]! * invSqrtM[j]!;

  const { values, vectors } = jacobiEigen(Hm, n3);

  // Keep eigenvalue, signed wavenumber, and (mass-weighted) eigenvector together
  // so we never lose the frequency<->displacement pairing.
  interface ModeEntry {
    wavenumber: number; // signed cm^-1
    vector: number[]; // mass-weighted eigenvector, length n3
  }
  const entries: ModeEntry[] = values.map((lam, k) => ({
    wavenumber: (lam >= 0 ? 1 : -1) * VIB_CM_PER_SQRT * Math.sqrt(Math.abs(lam)),
    vector: vectors[k]!,
  }));

  const allWavenumbers = entries.map((e) => e.wavenumber).sort((a, b) => a - b);

  const isLinear = isLinearGeometry(mol.symbols, mol.positions);
  const nRemove = isLinear ? 5 : 6;

  // The nRemove modes closest to zero are translations + rotations.
  const byAbs = [...entries].sort(
    (a, b) => Math.abs(a.wavenumber) - Math.abs(b.wavenumber),
  );
  const removed = byAbs.slice(0, nRemove);
  const maxResidualTransRot = removed.reduce(
    (m, e) => Math.max(m, Math.abs(e.wavenumber)),
    0,
  );

  // Surviving vibrational modes, ascending by (signed) wavenumber.
  const vib = byAbs.slice(nRemove).sort((a, b) => a.wavenumber - b.wavenumber);

  const frequencies: number[] = [];
  const modes: Float64Array[] = [];
  for (const e of vib) {
    frequencies.push(e.wavenumber);
    // Mass-un-weight: Cartesian displacement dx_i = q_i / sqrt(m_i).
    const cart = new Float64Array(n3);
    let normSq = 0;
    for (let i = 0; i < n3; i++) {
      const d = e.vector[i]! * invSqrtM[i]!;
      cart[i] = d;
      normSq += d * d;
    }
    const inv = normSq > 0 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < n3; i++) cart[i] = cart[i]! * inv;
    modes.push(cart);
  }

  return { frequencies, modes, isLinear, allWavenumbers, maxResidualTransRot };
}
