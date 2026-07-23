/**
 * Schema for the reference-data JSON files consumed by the bench
 * runner (runner.ts). This module is intentionally the single source
 * of truth for the on-disk format -- see README.md in this package for
 * the human-readable version of the same contract.
 *
 * IMPORTANT: this schema describes the format; it does not itself
 * contain or read any data. Real curated reference datasets are being
 * assembled by a different agent in `bench-data/` at the repo root --
 * this package does not read from that directory yet. When that
 * migration happens, files there should conform to this schema (or
 * this schema should be revised to fit what was actually curated,
 * whichever is easier -- it is a living contract, not committed law).
 */

/** Where a reference number came from -- required on every datum so results are always traceable. */
export interface Citation {
  /** Short human label, e.g. "NIST CCCBDB", "Gaussian-4 (G4) reference set", "Karton et al. 2011". */
  source: string;
  /** DOI, URL, or dataset identifier. */
  reference: string;
  /** Level of theory / method used to produce the reference value, e.g. "CCSD(T)/aug-cc-pVTZ", "experimental (gas-phase electron diffraction)". */
  level: string;
  /** Optional free-text caveats (e.g. "zero-point energy not included", "298K, not 0K"). */
  notes?: string;
}

/** Physical units a reference value may be expressed in -- kept explicit so the runner never guesses. */
export type EnergyUnit = "hartree" | "eV" | "kcal/mol" | "kJ/mol";
export type LengthUnit = "angstrom" | "bohr";
export type ForceUnit = "hartree/bohr" | "eV/angstrom";

/** A single reference geometry + whatever known properties it carries. */
export interface BenchMolecule {
  /** Stable identifier within the dataset, e.g. "water-monomer", "benzene-CCSD-T". */
  id: string;
  /** Human-readable name/description. */
  name: string;

  symbols: string[];
  /** Flat Cartesian coordinates, length 3*N, in `positionsUnit`. */
  positions: number[];
  positionsUnit: LengthUnit;

  charge: number;
  multiplicity: number;

  /** Reference properties this molecule carries. All optional -- a
   * dataset can mix molecules that only have an energy with ones that
   * also have forces, geometry, etc. Every populated field must be
   * paired with a citation (see `citations`, keyed by property name). */
  reference: {
    energy?: number;
    energyUnit?: EnergyUnit;
    /** Flat, length 3*N, matching `positions` ordering. */
    forces?: number[];
    forcesUnit?: ForceUnit;
    /** Any other scalar property (dipole moment, HOMO-LUMO gap, bond length, ...). */
    scalars?: Record<string, number>;
  };

  /** Citation per reference property key (e.g. "energy", "forces", or a scalar name). Every key present in `reference` (including each key of `reference.scalars`) must have a matching citation here. */
  citations: Record<string, Citation>;
}

export interface BenchDataset {
  /** Dataset name, e.g. "small-molecule-energies-v1". */
  name: string;
  /** Schema version this file conforms to -- bump if the shape changes. */
  schemaVersion: 1;
  description?: string;
  molecules: BenchMolecule[];
}

/** One property's error statistics across every molecule in a dataset that had a reference value for it. */
export interface PropertyErrorStats {
  property: string;
  unit: string;
  n: number;
  mae: number;
  rmse: number;
  max: number;
  /** Signed error per molecule (predicted - reference), same order as the input dataset's molecules (skipping ones missing this property). Useful for plotting an error distribution / histogram. */
  errors: number[];
  /** Matching molecule ids for each entry in `errors`. */
  moleculeIds: string[];
}

export interface BenchReport {
  dataset: string;
  providerName: string;
  generatedAt: string;
  properties: PropertyErrorStats[];
}
