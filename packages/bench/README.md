# @browser-comp-chem/bench

Benchmark harness: given an `EnergyForceProvider` (from `@browser-comp-chem/engine`)
and a reference dataset, produce per-property error statistics (MAE, RMSE, max
error, and the raw signed error distribution for plotting/histograms).

This package is the mechanism behind the project's "permanent public test
bench" -- the standing comparison of every compute tier (RDKit-JS force
fields, ONNX ML potentials, extended Hueckel/HF orbitals) against literature
reference values, at every stage of development.

## Status

Skeleton + one working path: `runBenchmark()` currently scores the `energy`
property only (see `src/runner.ts`). Forces and other scalar properties are
defined in the schema already (`src/schema.ts`) but not yet wired into the
runner -- a natural next step once real reference data with force fields
exists.

## Dataset schema

Source of truth: `src/schema.ts` (`BenchDataset` / `BenchMolecule` /
`Citation`). Summary:

```jsonc
{
  "name": "small-molecule-energies-v1",
  "schemaVersion": 1,
  "description": "optional free text",
  "molecules": [
    {
      "id": "water-monomer",
      "name": "Water monomer, CCSD(T)/aug-cc-pVTZ equilibrium geometry",
      "symbols": ["O", "H", "H"],
      "positions": [0, 0, 0,  0, 0, 0.96,  0.93, 0, -0.24],
      "positionsUnit": "angstrom",
      "charge": 0,
      "multiplicity": 1,
      "reference": {
        "energy": -76.34,
        "energyUnit": "hartree"
      },
      "citations": {
        "energy": {
          "source": "NIST CCCBDB",
          "reference": "https://cccbdb.nist.gov/... (or a DOI)",
          "level": "CCSD(T)/aug-cc-pVTZ",
          "notes": "optional caveats, e.g. ZPE not included"
        }
      }
    }
  ]
}
```

Rules the schema enforces (see doc comments in `schema.ts` for the full
contract):

- Every populated field under `reference` (including each key of
  `reference.scalars`) **must** have a matching entry in `citations`, keyed
  by the same property name. No reference number without a traceable source.
- Units are always explicit (`positionsUnit`, `energyUnit`, `forcesUnit`) --
  the runner canonicalizes everything internally (to Hartree / Hartree-Bohr /
  Angstrom, see `src/units.ts`) rather than assuming a convention.
- `schemaVersion` is bumped whenever the shape changes, so old dataset files
  and old runner code can detect a mismatch instead of silently
  misinterpreting each other.

## bench-data/ (not yet wired in)

A separate agent is curating real literature reference datasets into a
`bench-data/` directory at the repo root. This package does not read from
that directory yet, and this scaffold does not touch it. When that data
lands, either the files there will conform to the schema above, or the
schema will be revised to match what was actually curated -- whichever is
less lossy. Migrating it in (pointing a real runner invocation, e.g. a CLI
or a demo-app benchmark page, at `bench-data/*.json`) is a follow-up task,
not part of this scaffold.

## Usage (once wired to real data)

```ts
import { runBenchmark } from "@browser-comp-chem/bench";
import { LennardJonesProvider } from "@browser-comp-chem/engine";
import dataset from "../bench-data/some-dataset.json";

const report = await runBenchmark(new LennardJonesProvider(), dataset);
console.log(report.properties[0]); // { property: "energy", mae, rmse, max, ... }
```
