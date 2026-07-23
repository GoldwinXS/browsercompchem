/**
 * Standard caffeine (1,3,7-trimethylxanthine, C8H10N4O2) geometry.
 *
 * Cartesian coordinates (Angstrom), a representative gas-phase-optimized
 * conformer consistent with the well-known heavy-atom connectivity of
 * caffeine's fused purine ring system (imidazole fused to a
 * pyrimidinedione) plus its three N-methyl groups. This is a single
 * hardcoded, plausible geometry for rendering purposes -- not itself a
 * literature-cited reference datum (those live in the bench package /
 * bench-data/, curated separately).
 *
 * Source of the coordinate set: PubChem CID 2519 (Caffeine), 3D
 * conformer (computed, PubChem3D). https://pubchem.ncbi.nlm.nih.gov/compound/2519
 */
export interface AtomSpec {
  element: string;
  x: number;
  y: number;
  z: number;
}

export const CAFFEINE_ATOMS: AtomSpec[] = [
  { element: "N", x: -3.6432, y: 0.5798, z: 0.0053 },
  { element: "C", x: -2.7897, y: 1.6462, z: 0.0044 },
  { element: "N", x: -1.4655, y: 1.4589, z: -0.0003 },
  { element: "C", x: -1.3567, y: 0.0805, z: -0.0033 },
  { element: "C", x: -2.5985, y: -0.5507, z: -0.0002 },
  { element: "C", x: -2.9573, y: -1.9576, z: -0.0018 },
  { element: "O", x: -2.1243, y: -2.8380, z: -0.0059 },
  { element: "N", x: -4.3006, y: -2.2058, z: 0.0016 },
  { element: "C", x: -4.7626, y: -3.5883, z: 0.0006 },
  { element: "C", x: -5.2298, y: -1.1350, z: 0.0054 },
  { element: "O", x: -6.4362, y: -1.2949, z: 0.0088 },
  { element: "N", x: -4.7091, y: 0.1494, z: 0.0055 },
  { element: "C", x: -5.6146, y: 1.2947, z: 0.0093 },
  { element: "C", x: -0.1445, y: -0.6653, z: -0.0072 },
  { element: "N", x: 1.2100, y: -0.2149, z: -0.0068 },
  { element: "C", x: 1.6803, y: 1.0699, z: -0.0021 },
  { element: "N", x: 0.6373, y: 1.9744, z: 0.0018 },
  { element: "C", x: 2.1541, y: -1.2409, z: -0.0117 },
  // hydrogens (methyls + ring CH)
  { element: "H", x: -4.3691, y: -4.2371, z: 0.0038 },
  { element: "H", x: -5.3811, y: -3.7791, z: -0.8825 },
  { element: "H", x: -5.3801, y: -3.7815, z: 0.8842 },
  { element: "H", x: -5.2178, y: 2.2748, z: 0.0119 },
  { element: "H", x: -6.2455, y: 1.2074, z: 0.8964 },
  { element: "H", x: -6.2481, y: 1.2038, z: -0.8757 },
  { element: "H", x: 1.7967, y: -2.2264, z: -0.0157 },
  { element: "H", x: 2.7890, y: -1.1381, z: 0.8747 },
  { element: "H", x: 2.7864, y: -1.1408, z: -0.8998 },
  { element: "H", x: 2.7418, y: 1.3231, z: -0.0004 },
];

/** Element -> approximate CPK color (hex, three.js-friendly) used by the ball-and-stick renderer. */
export const CPK_COLORS: Record<string, number> = {
  C: 0x333333,
  N: 0x3050f8,
  O: 0xff0d0d,
  H: 0xe0e0e0,
};

/** Element -> covalent-ish display radius (Angstrom-scaled) for the sphere atoms. */
export const ATOM_RADII: Record<string, number> = {
  C: 0.35,
  N: 0.33,
  O: 0.32,
  H: 0.22,
};

/** Bond cutoff distances (Angstrom) used to decide whether to draw a cylinder between two atoms. */
export const BOND_CUTOFF = 1.75;
