import { describe, expect, it } from "vitest";
import { LennardJonesProvider } from "@browser-comp-chem/engine";
import { runBenchmark } from "../src/runner.js";
import type { BenchDataset } from "../src/schema.js";

describe("runBenchmark", () => {
  it("scores a provider against a toy one-molecule dataset", async () => {
    const provider = new LennardJonesProvider({ sigma: 1, epsilon: 1 });
    const r = provider.equilibriumDistance;

    const dataset: BenchDataset = {
      name: "toy-lj-dimer",
      schemaVersion: 1,
      description: "Single LJ dimer at its own analytic minimum, used only to exercise the runner.",
      molecules: [
        {
          id: "lj-dimer",
          name: "LJ dimer at equilibrium",
          symbols: ["Ar", "Ar"],
          positions: [0, 0, 0, r, 0, 0],
          positionsUnit: "angstrom",
          charge: 0,
          multiplicity: 1,
          reference: {
            energy: -1, // exact analytic minimum in these reduced units
            energyUnit: "hartree",
          },
          citations: {
            energy: {
              source: "analytic LJ minimum",
              reference: "V(2^(1/6) sigma) = -epsilon",
              level: "exact (not a real quantum-chemistry reference; test fixture only)",
            },
          },
        },
      ],
    };

    const report = await runBenchmark(provider, dataset);

    expect(report.properties).toHaveLength(1);
    const energy = report.properties[0]!;
    expect(energy.n).toBe(1);
    expect(energy.mae).toBeCloseTo(0, 6);
    expect(energy.rmse).toBeCloseTo(0, 6);
  });
});
