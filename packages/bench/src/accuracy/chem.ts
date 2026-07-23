/**
 * Physical constants, atomic data, and unit conversions used by the accuracy
 * suite. Kept in one place so every module (conformers, frequencies, heats of
 * formation) shares exactly the same numbers, and so each hard-coded reference
 * datum can carry a citation in the report.
 *
 * Energy/length conversions that already exist in units.ts are re-used there;
 * this module adds the vibrational / thermochemical constants those modules do
 * not cover.
 */

/** 1 Hartree in kJ/mol (matches units.ts). */
export const HARTREE_TO_KJ_MOL = 2625.499639479;
/** 1 Hartree in kcal/mol (matches units.ts). */
export const HARTREE_TO_KCAL_MOL = 627.5094740631;
/** 1 Hartree in cm^-1 (Rydberg/Hartree spectroscopic factor). */
export const HARTREE_TO_CM1 = 219474.6313632;
/** 1 cm^-1 in kJ/mol. */
export const CM1_TO_KJ_MOL = HARTREE_TO_KJ_MOL / HARTREE_TO_CM1;

// Thermochemistry (298.15 K, ideal gas)
export const T_STANDARD = 298.15; // K
export const R_KJ = 8.314462618e-3; // kJ/mol/K
export const RT_KJ = R_KJ * T_STANDARD; // kJ/mol, ~2.4790
/** hc/k in cm*K, for the vibrational temperature x = hc*nu/(kT) = HC_OVER_K*nu/T. */
export const HC_OVER_K = 1.4387768775; // cm*K

// SI constants for the mass-weighted-Hessian -> wavenumber conversion.
const HARTREE_J = 4.3597447222071e-18; // J per Hartree
const AMU_KG = 1.66053906660e-27; // kg per amu
const ANGSTROM2_M2 = 1e-20; // m^2 per Angstrom^2
const C_CM_S = 2.99792458e10; // speed of light, cm/s

/**
 * Multiplier turning a mass-weighted Hessian eigenvalue lambda (in
 * Hartree / (Angstrom^2 * amu)) into a wavenumber:  nu_cm = VIB_CM_PER_SQRT * sqrt(lambda).
 * Derivation: lambda_SI = lambda * HARTREE_J / (ANGSTROM2_M2 * AMU_KG)  [s^-2];
 * nu_cm = sqrt(lambda_SI) / (2*pi*c).
 */
export const VIB_CM_PER_SQRT =
  Math.sqrt(HARTREE_J / (ANGSTROM2_M2 * AMU_KG)) / (2 * Math.PI * C_CM_S);

/** Standard atomic weights (amu), sufficient for mass-weighting. */
export const ATOMIC_MASS: Record<string, number> = {
  H: 1.008,
  C: 12.011,
  N: 14.007,
  O: 15.999,
  F: 18.998,
  S: 32.06,
  Cl: 35.45,
};

/** Covalent radii (Angstrom) for simple distance-based bond perception. */
export const COVALENT_RADIUS: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  F: 0.57,
  S: 1.05,
  Cl: 1.02,
};

/** Elements ANI-2x is trained on. Anything else must be skipped. */
export const ANI_ELEMENTS = new Set(["H", "C", "N", "O", "F", "S", "Cl"]);

/**
 * Experimental gas-phase atomic standard enthalpies of formation, delta_f H(298.15 K),
 * in kJ/mol. Used by the atomization route to heats of formation.
 *
 * Values: CODATA Key Values for Thermodynamics (Cox, Wagman, Medvedev 1989) and
 * ATcT, as reproduced in bench-data/README.md CAVEAT 1.
 */
export const ATOMIC_DHF298_KJ: Record<string, number> = {
  H: 217.998,
  C: 716.87,
  N: 472.44,
  O: 249.229,
  F: 79.38,
  S: 277.17,
  Cl: 121.301,
};
export const ATOMIC_DHF298_CITATION = {
  source: "CODATA Key Values for Thermodynamics (Cox, Wagman, Medvedev 1989); ATcT",
  reference: "bench-data/README.md CAVEAT 1 (C 716.87, H 217.998, N 472.44, O 249.229, F 79.38, S 277.17, Cl 121.301 kJ/mol)",
  level: "experimental gas-phase atomic delta_f H(298.15 K)",
};

/**
 * Harmonic-to-fundamental frequency scaling factor. ANI-2x is trained to
 * wB97X/6-31G*(d) reference data; CCCBDB lists a fundamentals scale factor of
 * ~0.95 for that method. The task asks for a standard ~0.96 factor; we report
 * BOTH the raw harmonic frequencies and the scaled ones and state the factor
 * explicitly here.
 */
export const FREQ_SCALE_FACTOR = 0.96;
export const FREQ_SCALE_CITATION = {
  source: "CCCBDB vibrational scaling factors",
  reference: "https://cccbdb.nist.gov/vibscalejust.asp",
  level: "standard harmonic-to-fundamental scale factor (0.96; method-appropriate wB97X/6-31G* value is ~0.95)",
};
