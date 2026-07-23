import type { Molecule } from "../../geometry/molecule.js";
import type { EnergyForceProvider, EnergyForces } from "../types.js";
import { Ani2xModel, type Ani2xLoadOptions } from "./model.js";

/**
 * ANI-2x machine-learned interatomic potential as an EnergyForceProvider.
 *
 * The AEV featurizer and per-element MLPs run in pure float64 TypeScript (no
 * ONNX runtime), so this provider works identically under node (vitest) and in
 * the browser; the only environment-specific part is asset loading (fs vs
 * fetch), handled in model.ts.
 *
 * NOTE ON UNITS: ANI-2x is defined in Angstrom, and Molecule.positions are
 * Angstrom, so this provider returns energy in Hartree and forces in
 * Hartree/Angstrom (-dE/dr with r in Angstrom) -- i.e. forces are per-Angstrom,
 * NOT per-Bohr as the generic EnergyForceProvider docstring suggests. This keeps
 * positions and forces in a single consistent length unit (Angstrom) for the
 * optimizer, matching TorchANI's own convention and the wave-one spike. The
 * deviation is deliberate and documented here, exactly as LennardJonesProvider
 * documents its own reduced-unit deviation.
 */
export class Ani2xProvider implements EnergyForceProvider {
  readonly name: string;
  private readonly model: Ani2xModel;
  private cachedSymbols: string[] | undefined;
  private cachedSpecies: Int32Array | undefined;

  private constructor(model: Ani2xModel) {
    this.model = model;
    this.name = `ani2x-${model.variant}`;
  }

  /** Async factory: loads manifest + weights, then returns a ready provider. */
  static async create(opts: Ani2xLoadOptions): Promise<Ani2xProvider> {
    const model = await Ani2xModel.load(opts);
    return new Ani2xProvider(model);
  }

  /** Number of ensemble members actually being averaged. */
  get members(): number {
    return this.model.members;
  }

  private species(mol: Molecule): Int32Array {
    if (this.cachedSymbols === mol.symbols && this.cachedSpecies) {
      return this.cachedSpecies;
    }
    const sp = this.model.speciesOf(mol.symbols);
    this.cachedSymbols = mol.symbols;
    this.cachedSpecies = sp;
    return sp;
  }

  async energyForces(mol: Molecule): Promise<EnergyForces> {
    const species = this.species(mol);
    // model works directly on the flat Angstrom position buffer
    const { energy, forces } = this.model.energyForces(species, mol.positions);
    return { energy, forces };
  }

  /** Energy only (Hartree); skips the force adjoint. */
  energyOnly(mol: Molecule): number {
    return this.model.energy(this.species(mol), mol.positions);
  }
}
