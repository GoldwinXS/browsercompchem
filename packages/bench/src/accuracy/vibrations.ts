/**
 * Harmonic vibrational analysis from an EnergyForceProvider:
 *   1. finite-difference the analytic forces -> Cartesian Hessian
 *   2. mass-weight and diagonalize (Jacobi)
 *   3. eigenvalues -> wavenumbers, drop the 6 (or 5, linear) trans/rot modes
 *   4. derive zero-point energy and 0->298 K thermal enthalpy (RRHO)
 *
 * All energies from the provider are Hartree; positions/forces are Angstrom /
 * Hartree-per-Angstrom (ANI-2x convention). Frequencies come out in cm^-1.
 */
import type { EnergyForceProvider, Molecule } from "@browser-comp-chem/engine";
import { cloneMolecule } from "@browser-comp-chem/engine";
import {
  ATOMIC_MASS,
  CM1_TO_KJ_MOL,
  HC_OVER_K,
  R_KJ,
  T_STANDARD,
  VIB_CM_PER_SQRT,
} from "./chem.js";
import { jacobiEigen } from "./linalg.js";

export interface VibResult {
  /** All 3N wavenumbers (cm^-1), ascending; includes the near-zero modes. */
  allWavenumbers: number[];
  /** The 3N-6 (or 3N-5) vibrational wavenumbers, ascending. Imaginary -> negative. */
  frequencies: number[];
  /** True if the molecule was detected linear (5 trans/rot modes removed). */
  linear: boolean;
  /** Largest magnitude of the six/five removed (should-be-zero) modes, cm^-1. */
  maxResidualTransRot: number;
  /** Zero-point vibrational energy from the harmonic frequencies (kJ/mol). */
  zpeKJ: number;
  /** Vibrational thermal energy at 298.15 K, sum h*nu/(exp-1) (kJ/mol, excludes ZPE). */
  vibThermalKJ: number;
}

/** Central-difference Cartesian Hessian (Hartree/Angstrom^2), symmetrized. */
export async function finiteDiffHessian(
  mol: Molecule,
  provider: EnergyForceProvider,
  step = 1e-3,
): Promise<Float64Array> {
  const n3 = mol.positions.length;
  const H = new Float64Array(n3 * n3);
  const work = cloneMolecule(mol);
  for (let c = 0; c < n3; c++) {
    const x0 = mol.positions[c]!;
    work.positions[c] = x0 + step;
    const fPlus = (await provider.energyForces(work)).forces;
    work.positions[c] = x0 - step;
    const fMinus = (await provider.energyForces(work)).forces;
    work.positions[c] = x0;
    // H_ic = d^2E/dx_i dx_c = -dF_i/dx_c
    for (let i = 0; i < n3; i++) {
      H[i * n3 + c] = -(fPlus[i]! - fMinus[i]!) / (2 * step);
    }
  }
  // symmetrize
  for (let i = 0; i < n3; i++) {
    for (let j = i + 1; j < n3; j++) {
      const avg = 0.5 * (H[i * n3 + j]! + H[j * n3 + i]!);
      H[i * n3 + j] = avg;
      H[j * n3 + i] = avg;
    }
  }
  return H;
}

/** Detect linearity from the moment-of-inertia tensor (smallest principal moment ~ 0). */
export function isLinear(symbols: string[], pos: Float64Array): boolean {
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

/** Full harmonic analysis at a (presumed stationary) geometry. */
export async function vibrationalAnalysis(
  mol: Molecule,
  provider: EnergyForceProvider,
  step = 1e-3,
): Promise<VibResult> {
  const n = mol.symbols.length;
  const n3 = 3 * n;
  const H = await finiteDiffHessian(mol, provider, step);

  // mass-weight: Hm_ij = H_ij / sqrt(m_i m_j)
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

  const { values } = jacobiEigen(Hm, n3);
  // eigenvalue -> signed wavenumber
  const allWavenumbers = values
    .map((lam) => (lam >= 0 ? 1 : -1) * VIB_CM_PER_SQRT * Math.sqrt(Math.abs(lam)))
    .sort((a, b) => a - b);

  const linear = isLinear(mol.symbols, mol.positions);
  const nRemove = linear ? 5 : 6;

  // Remove the nRemove modes closest to zero (translations + rotations).
  const byAbs = [...allWavenumbers].sort((a, b) => Math.abs(a) - Math.abs(b));
  const removed = byAbs.slice(0, nRemove);
  const maxResidualTransRot = removed.reduce((m, w) => Math.max(m, Math.abs(w)), 0);
  const frequencies = byAbs.slice(nRemove).sort((a, b) => a - b);

  // ZPE and vibrational thermal energy from positive (real) frequencies.
  let zpeKJ = 0;
  let vibThermalKJ = 0;
  for (const nu of frequencies) {
    if (nu <= 0) continue; // ignore imaginary modes in thermochemistry
    zpeKJ += 0.5 * nu * CM1_TO_KJ_MOL;
    const x = (HC_OVER_K * nu) / T_STANDARD;
    vibThermalKJ += R_KJ * T_STANDARD * (x / (Math.exp(x) - 1));
  }

  return {
    allWavenumbers,
    frequencies,
    linear,
    maxResidualTransRot,
    zpeKJ,
    vibThermalKJ,
  };
}
