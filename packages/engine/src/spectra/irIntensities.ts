/**
 * Harmonic IR intensities from the existing vibrational normal modes.
 *
 * REUSES the vibrations module's normal modes wholesale -- this file contains
 * no Hessian code. It takes the `modes`/`frequencies` that
 * vibrations/normalModes.ts already computed (mass-weighted-Hessian
 * eigenvectors, mass-UN-weighted back to Cartesian and normalized to unit L2
 * norm -- see that file's docstring) and, for each mode, finite-differences the
 * EHT-Mulliken dipole (dipole.ts) along the mode's Cartesian displacement to get
 * d(mu)/dQ, the standard IR-intensity observable.
 *
 * --- Reduced mass, and why the Cartesian finite difference needs one ---
 * computeNormalModes() returns each mode as a Cartesian displacement vector
 * d_i (i over the 3N flattened x/y/z components) normalized so that
 * sum_i d_i^2 = 1 -- i.e. normalized in PLAIN Cartesian space, not in the
 * mass-weighted space the eigensolver actually diagonalized. The true harmonic
 * normal coordinate Q_k (units sqrt(amu)*Angstrom) relates to a Cartesian
 * step ds along d_i by
 *
 *   ds/dQ_k = 1 / sqrt(mu_k),   mu_k = sum_i m_i * d_i^2
 *
 * where mu_k is the mode's "reduced mass" (amu) and m_i is the atomic mass
 * belonging to Cartesian component i (repeated x3 per atom). This is the
 * standard reduced-mass formula for a Cartesian-normalized (not mass-weighted-
 * normalized) displacement vector, and it is what lets a Cartesian finite
 * difference stand in for the mass-weighted one: displacing atoms by
 * +-delta*d_i (Angstrom, plain Cartesian) and central-differencing the dipole
 * gives d(mu)/ds; dividing by sqrt(mu_k) converts it to d(mu)/dQ_k.
 *
 * --- Absolute intensity: the km/mol prefactor ---
 * The standard double-harmonic IR integrated absorption coefficient (SI,
 * Wilson-Decius-Cross / Person-Zerbi convention) is
 *
 *   A_k = N_A / (12 * eps0 * c^2) * (d(mu)/dQ_k)^2
 *
 * With d(mu)/dQ_k in Debye/(Angstrom*sqrt(amu)) and the fundamental constants
 * (N_A = 6.02214076e23 /mol, eps0 = 8.8541878128e-12 F/m, c = 2.99792458e8 m/s,
 * 1 D = 3.33564e-30 C*m, 1 amu = 1.66053906660e-27 kg) converted through, this
 * evaluates to
 *
 *   A_k [km/mol] = 42.2561 * (d(mu)/dQ_k [D/(Angstrom*sqrt(amu))])^2
 *
 * -- independently reverified here from the fundamental constants above
 * (42.256, matching to 4 significant figures) rather than taken purely on
 * faith; it also matches the constant widely quoted by mainstream quantum-
 * chemistry packages for this exact unit combination. Treat the ABSOLUTE
 * km/mol number as best-effort (EHT-Mulliken dipole derivatives are a coarse
 * approximation to a real dipole-derivative tensor); the correctness tests in
 * spectra.test.ts rely only on RELATIVE intensity ordering and the
 * symmetry-forced zeros, both of which are prefactor-independent.
 */
import type { Molecule } from "../geometry/molecule.js";
import { ATOMIC_MASS } from "../vibrations/constants.js";
import { computeDipole, type DipoleResult } from "./dipole.js";

/** A_k [km/mol] per unit (d(mu)/dQ_k)^2 in (D / (Angstrom*sqrt(amu)))^2. See module docstring. */
export const IR_KM_PER_MOL_PREFACTOR = 42.2561;

export interface ModeIntensity {
  /** Signed harmonic wavenumber (cm^-1), copied from the input frequency. */
  frequency: number;
  /** Mode reduced mass (amu), sum_i m_i * d_i^2 over the Cartesian-normalized mode. */
  reducedMass: number;
  /** |d(mu)/dQ_k|, Debye / (Angstrom * sqrt(amu)). */
  dMuDQ: number;
  /** Best-effort absolute IR intensity, km/mol (see module docstring for the prefactor). */
  absoluteKmPerMol: number;
  /** Relative intensity, 0-100, scaled to the strongest mode. */
  relative: number;
  /** True if `relative` clears `activeThreshold` (a real, non-symmetry-forbidden band). */
  isActive: boolean;
}

export interface IrIntensityOptions {
  /** Cartesian finite-difference step along the (unit-normalized) mode, Angstrom. Default 0.01. */
  delta?: number;
  /** Minimum fraction of the strongest mode's intensity to count as "active" (default 0.01 = 1%). */
  activeThreshold?: number;
}

/**
 * IR intensity of every supplied normal mode, via finite-differenced
 * EHT-Mulliken dipoles. `modes`/`frequencies` must be index-aligned, exactly
 * the arrays computeNormalModes() returns (Cartesian, unit-normalized).
 */
export async function computeIRIntensities(
  mol: Molecule,
  modes: Float64Array[],
  frequencies: number[],
  options: IrIntensityOptions = {},
): Promise<ModeIntensity[]> {
  const delta = options.delta ?? 0.01;
  const activeThreshold = options.activeThreshold ?? 0.01;

  const n = mol.symbols.length;
  const masses = new Float64Array(n);
  for (let a = 0; a < n; a++) masses[a] = ATOMIC_MASS[mol.symbols[a]!] ?? 0;

  const dMuDQ: number[] = [];
  const reducedMasses: number[] = [];

  for (let k = 0; k < modes.length; k++) {
    const d = modes[k]!;

    // Reduced mass: sum over atoms of m_a * |displacement vector of atom a|^2.
    let mu = 0;
    for (let a = 0; a < n; a++) {
      const dx = d[3 * a]!;
      const dy = d[3 * a + 1]!;
      const dz = d[3 * a + 2]!;
      mu += masses[a]! * (dx * dx + dy * dy + dz * dz);
    }
    reducedMasses.push(mu);

    // Central-difference the dipole vector along the Cartesian mode direction.
    const plus = displace(mol, d, delta);
    const minus = displace(mol, d, -delta);
    const [muPlus, muMinus] = await Promise.all([computeDipole(plus), computeDipole(minus)]);
    const dMuDs: [number, number, number] = [
      (muPlus.vector[0] - muMinus.vector[0]) / (2 * delta),
      (muPlus.vector[1] - muMinus.vector[1]) / (2 * delta),
      (muPlus.vector[2] - muMinus.vector[2]) / (2 * delta),
    ];
    // ds/dQ = 1/sqrt(mu_k) -> d(mu)/dQ = d(mu)/ds / sqrt(mu_k) = d(mu)/ds * sqrt(mu_k)...
    // no: dQ/ds = sqrt(mu_k), so d(mu)/dQ = d(mu)/ds * (ds/dQ) = d(mu)/ds / sqrt(mu_k).
    const invSqrtMu = mu > 0 ? 1 / Math.sqrt(mu) : 0;
    const dMuDQvec: [number, number, number] = [
      dMuDs[0] * invSqrtMu,
      dMuDs[1] * invSqrtMu,
      dMuDs[2] * invSqrtMu,
    ];
    dMuDQ.push(Math.hypot(dMuDQvec[0], dMuDQvec[1], dMuDQvec[2]));
  }

  const absolute = dMuDQ.map((v) => IR_KM_PER_MOL_PREFACTOR * v * v);
  const maxAbs = absolute.reduce((m, v) => Math.max(m, v), 0);

  const out: ModeIntensity[] = [];
  for (let k = 0; k < modes.length; k++) {
    const relative = maxAbs > 0 ? (100 * absolute[k]!) / maxAbs : 0;
    out.push({
      frequency: frequencies[k]!,
      reducedMass: reducedMasses[k]!,
      dMuDQ: dMuDQ[k]!,
      absoluteKmPerMol: absolute[k]!,
      relative,
      isActive: relative / 100 > activeThreshold,
    });
  }
  return out;
}

/** Molecule with atom positions displaced by amount*d (a flat 3N Cartesian vector). */
function displace(mol: Molecule, d: Float64Array, amount: number): Molecule {
  const positions = new Float64Array(mol.positions.length);
  for (let i = 0; i < positions.length; i++) positions[i] = mol.positions[i]! + amount * d[i]!;
  return { symbols: mol.symbols, positions, charge: mol.charge, multiplicity: mol.multiplicity };
}

export type { DipoleResult };
