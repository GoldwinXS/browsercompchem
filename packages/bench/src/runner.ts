import type { EnergyForceProvider, Molecule } from "@browser-comp-chem/engine";
import type { BenchDataset, BenchReport, PropertyErrorStats } from "./schema.js";
import { computeErrorStats } from "./stats.js";
import { energyToHartree, lengthToAngstrom } from "./units.js";

/**
 * Runs `provider` against every molecule in `dataset` that carries a
 * reference energy, and returns MAE/RMSE/max-error statistics plus the
 * raw per-molecule error distribution.
 *
 * All energies are canonicalized to Hartree internally (see units.ts)
 * regardless of what unit each datum's `reference.energyUnit` was
 * authored in, so stats are always comparable across a mixed-unit
 * dataset. Forces are accepted in the schema but not yet scored here --
 * that's a natural next property to add once real reference force data
 * exists (see README.md).
 */
export async function runBenchmark(
  provider: EnergyForceProvider,
  dataset: BenchDataset,
): Promise<BenchReport> {
  const errors: number[] = [];
  const moleculeIds: string[] = [];

  for (const m of dataset.molecules) {
    if (m.reference.energy === undefined || m.reference.energyUnit === undefined) {
      continue;
    }

    const mol: Molecule = {
      symbols: m.symbols,
      positions: Float64Array.from(
        m.positions.map((x) => lengthToAngstrom(x, m.positionsUnit)),
      ),
      charge: m.charge,
      multiplicity: m.multiplicity,
    };

    const predicted = await provider.energyForces(mol);
    const referenceHartree = energyToHartree(m.reference.energy, m.reference.energyUnit);

    errors.push(predicted.energy - referenceHartree);
    moleculeIds.push(m.id);
  }

  const summary = computeErrorStats(errors);

  const energyStats: PropertyErrorStats = {
    property: "energy",
    unit: "hartree",
    n: summary.n,
    mae: summary.mae,
    rmse: summary.rmse,
    max: summary.max,
    errors,
    moleculeIds,
  };

  return {
    dataset: dataset.name,
    providerName: provider.name,
    generatedAt: new Date().toISOString(),
    properties: [energyStats],
  };
}
