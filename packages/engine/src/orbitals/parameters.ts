/**
 * Extended Hueckel Theory (EHT) valence-shell parameter table.
 *
 * Each supported element carries its neutral valence-electron count and, per
 * valence shell (s and, for non-hydrogen, p), three numbers:
 *   - n    principal quantum number of the Slater-type orbital (STO)
 *   - hii  valence-shell ionization potential (VSIP), in eV, NEGATIVE. This is
 *          the diagonal Hamiltonian element H_ii of the EHT secular problem.
 *   - zeta Slater orbital exponent, in bohr^-1 (atomic units). The STO radial
 *          part is r^(n-1) e^(-zeta r) with r in bohr.
 *
 * Sources / provenance of the numbers:
 *   - Method and the s/p VSIP + exponent values for H, C, N, O, F:
 *       R. Hoffmann, "An Extended Hueckel Theory. I. Hydrocarbons",
 *       J. Chem. Phys. 39, 1397 (1963). Hoffmann's Table lists H 1s (-13.6 eV,
 *       zeta 1.0 in the original hydrocarbon paper) and the 2s/2p VSIPs of the
 *       first-row elements. The H 1s exponent used here is 1.30, the value
 *       adopted in essentially every later "standard EHT" tabulation (it gives
 *       better H-X overlaps than the bare hydrogenic 1.0) -- see below.
 *   - The consolidated "standard EHT parameters" (the widely reproduced set,
 *       e.g. in the YAeHMOP / bind program's default parameter file, and in
 *       J. Howell, A. Rossi, D. Wallace, K. Haraki, R. Hoffmann, "FORTICON8"
 *       QCPE 344, and the tables reprinted in Landrum & Glassey's notes):
 *         H  1s  -13.6 eV  zeta 1.300
 *         C  2s  -21.4     2p -11.4   zeta 1.625
 *         N  2s  -26.0     2p -13.4   zeta 1.950
 *         O  2s  -32.3     2p -14.8   zeta 2.275
 *         F  2s  -40.0     2p -18.1   zeta 2.425
 *         S  3s  -20.0     3p -11.0   zeta 3s 2.122 / 3p 1.827
 *         Cl 3s  -30.0     3p -15.0   zeta 3s 2.356 / 3p 2.039
 *     The first-row zeta values are the single-zeta STO exponents from Clementi
 *     & Raimondi, "Atomic Screening Constants from SCF Functions", J. Chem.
 *     Phys. 38, 2686 (1963); the S/Cl s,p exponents are the standard EHT
 *     third-row values (Chen & Hoffmann and the FORTICON/YAeHMOP defaults).
 *
 * These give qualitatively correct frontier-orbital pictures (HOMO/LUMO shapes,
 * degeneracies, bonding/antibonding ordering) -- the purpose of the cheap EHT
 * tier -- not quantitative total energies.
 */

/** Real-Cartesian atomic-orbital flavours the basis is built from. */
export type OrbitalType = "s" | "px" | "py" | "pz";

/** One valence shell's EHT parameters. */
export interface ShellParam {
  /** Principal quantum number of the STO (1 for H 1s, 2 for C..F, 3 for S/Cl). */
  n: number;
  /** Diagonal Hamiltonian element H_ii = -VSIP, in eV (negative). */
  hii: number;
  /** Slater exponent, bohr^-1. */
  zeta: number;
}

/** Per-element EHT data: valence electron count plus s (and p) shell params. */
export interface ElementParam {
  /** Neutral-atom valence electrons (used for the aufbau electron count). */
  valenceElectrons: number;
  /** The valence s shell (always present). */
  s: ShellParam;
  /** The valence p shell (absent for hydrogen). */
  p?: ShellParam;
}

/**
 * The full EHT parameter table. Keyed by element symbol. Only the elements the
 * ANI-2x tier already supports are included; anything else is rejected upstream
 * with a clear error rather than silently mis-parametrized.
 */
export const EHT_PARAMS: Record<string, ElementParam> = {
  // Hydrogen: 1s only. Hoffmann 1963; exponent 1.30 (standard EHT tabulation).
  H: {
    valenceElectrons: 1,
    s: { n: 1, hii: -13.6, zeta: 1.3 },
  },
  // Carbon: 2s/2p. VSIPs Hoffmann 1963; zeta 1.625 (Clementi-Raimondi 1963).
  C: {
    valenceElectrons: 4,
    s: { n: 2, hii: -21.4, zeta: 1.625 },
    p: { n: 2, hii: -11.4, zeta: 1.625 },
  },
  // Nitrogen: 2s/2p.
  N: {
    valenceElectrons: 5,
    s: { n: 2, hii: -26.0, zeta: 1.95 },
    p: { n: 2, hii: -13.4, zeta: 1.95 },
  },
  // Oxygen: 2s/2p.
  O: {
    valenceElectrons: 6,
    s: { n: 2, hii: -32.3, zeta: 2.275 },
    p: { n: 2, hii: -14.8, zeta: 2.275 },
  },
  // Fluorine: 2s/2p.
  F: {
    valenceElectrons: 7,
    s: { n: 2, hii: -40.0, zeta: 2.425 },
    p: { n: 2, hii: -18.1, zeta: 2.425 },
  },
  // Sulfur: 3s/3p. Standard EHT third-row values (FORTICON/YAeHMOP defaults).
  S: {
    valenceElectrons: 6,
    s: { n: 3, hii: -20.0, zeta: 2.122 },
    p: { n: 3, hii: -11.0, zeta: 1.827 },
  },
  // Chlorine: 3s/3p.
  Cl: {
    valenceElectrons: 7,
    s: { n: 3, hii: -30.0, zeta: 2.356 },
    p: { n: 3, hii: -15.0, zeta: 2.039 },
  },
};

/** Element symbols the EHT tier can parametrize (H,C,N,O,F,S,Cl). */
export const EHT_SUPPORTED_ELEMENTS: readonly string[] = Object.keys(EHT_PARAMS);

/** Bohr radius in Angstrom -- the codebase stores positions in Angstrom; the
 * STO exponents are bohr^-1, so distances are converted at the EHT boundary. */
export const BOHR_PER_ANGSTROM = 1 / 0.529177210903;
