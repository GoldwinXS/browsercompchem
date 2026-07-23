import type { EnergyUnit, ForceUnit, LengthUnit } from "./schema.js";

// CODATA-ish constants, sufficient precision for benchmark-grade comparisons.
const HARTREE_TO_EV = 27.211386245988;
const HARTREE_TO_KCAL_MOL = 627.5094740631;
const HARTREE_TO_KJ_MOL = 2625.499639479;
const BOHR_TO_ANGSTROM = 0.529177210903;

/** Convert an energy value in `unit` to Hartree (the runner's internal canonical unit). */
export function energyToHartree(value: number, unit: EnergyUnit): number {
  switch (unit) {
    case "hartree":
      return value;
    case "eV":
      return value / HARTREE_TO_EV;
    case "kcal/mol":
      return value / HARTREE_TO_KCAL_MOL;
    case "kJ/mol":
      return value / HARTREE_TO_KJ_MOL;
  }
}

/** Convert a force value in `unit` to Hartree/Bohr (the runner's internal canonical unit). */
export function forceToHartreePerBohr(value: number, unit: ForceUnit): number {
  switch (unit) {
    case "hartree/bohr":
      return value;
    case "eV/angstrom":
      return (value / HARTREE_TO_EV) * BOHR_TO_ANGSTROM;
  }
}

/** Convert a length value in `unit` to Angstrom (the Molecule.positions convention). */
export function lengthToAngstrom(value: number, unit: LengthUnit): number {
  switch (unit) {
    case "angstrom":
      return value;
    case "bohr":
      return value * BOHR_TO_ANGSTROM;
  }
}
