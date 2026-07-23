/**
 * Physical constants and atomic data for harmonic vibrational analysis.
 *
 * These are the same numbers the accuracy bench validated against experiment
 * (packages/bench/src/accuracy/chem.ts); they live here so the engine's
 * normal-mode analysis is self-contained and the demo can call it without
 * pulling in the bench package.
 */

/** 1 Hartree in cm^-1 (Rydberg/Hartree spectroscopic factor). */
export const HARTREE_TO_CM1 = 219474.6313632;

// SI constants for the mass-weighted-Hessian -> wavenumber conversion.
const HARTREE_J = 4.3597447222071e-18; // J per Hartree
const AMU_KG = 1.66053906660e-27; // kg per amu
const ANGSTROM2_M2 = 1e-20; // m^2 per Angstrom^2
const C_CM_S = 2.99792458e10; // speed of light, cm/s

/**
 * Multiplier turning a mass-weighted Hessian eigenvalue lambda (in
 * Hartree / (Angstrom^2 * amu)) into a wavenumber:
 *   nu_cm = VIB_CM_PER_SQRT * sqrt(lambda).
 * Derivation: lambda_SI = lambda * HARTREE_J / (ANGSTROM2_M2 * AMU_KG)  [s^-2];
 * nu_cm = sqrt(lambda_SI) / (2*pi*c).
 */
export const VIB_CM_PER_SQRT =
  Math.sqrt(HARTREE_J / (ANGSTROM2_M2 * AMU_KG)) / (2 * Math.PI * C_CM_S);

/** Standard atomic weights (amu), sufficient for mass-weighting the Hessian. */
export const ATOMIC_MASS: Record<string, number> = {
  H: 1.008,
  C: 12.011,
  N: 14.007,
  O: 15.999,
  F: 18.998,
  S: 32.06,
  Cl: 35.45,
};
