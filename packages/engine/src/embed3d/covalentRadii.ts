/**
 * Single-bond covalent radii (Angstrom), from:
 *   B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria,
 *   E. Cremades, F. Barragan, S. Alvarez, "Covalent radii revisited",
 *   Dalton Trans., 2008, 2832-2838. https://doi.org/10.1039/B801115J
 *
 * Cordero et al. tabulate separate sp3/sp2/sp radii for carbon (0.76/0.73/0.69);
 * we use the sp3 value as the single per-element radius and instead apply a
 * bond-ORDER shortening factor uniformly (see forceField.ts) rather than
 * threading hybridization-specific radii through every element -- simpler, and
 * accurate enough for a classical pre-relaxation seed that the ANI-2x FIRE
 * polish subsequently refines. Covers every element ANI-2x is trained on
 * (H, C, N, O, F, S, Cl) plus B and P as required by the embed3d spec.
 */
export const COVALENT_RADII: Record<string, number> = {
  H: 0.31,
  B: 0.84,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  F: 0.57,
  P: 1.07,
  S: 1.05,
  Cl: 1.02,
};

/** Fallback radius for an element missing from the table (rare; keeps the FF from throwing). */
export const FALLBACK_RADIUS = 0.77;

/** Covalent radius of an element symbol, Angstrom, falling back to a generic single-bond default. */
export function covalentRadius(symbol: string): number {
  return COVALENT_RADII[symbol] ?? FALLBACK_RADIUS;
}
