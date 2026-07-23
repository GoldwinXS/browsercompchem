/**
 * MODULE 1 -- Conformer relative energies (ANI-2x's strongest suit).
 *
 * bench-data ships one geometry per conformer system, so for each pair we build
 * BOTH rotamers by setting a single well-defined torsion on that base geometry,
 * relax each with FIRE/ANI-2x, and take the electronic energy difference. This
 * is exactly the procedure bench-data/conformers.json's caveat prescribes
 * ("relax BOTH conformers with the potential and take the electronic-energy
 * difference to compare against an electronic reference").
 *
 * Two pairs are evaluated from a single, unambiguous torsion (n-butane
 * anti/gauche; ethanol trans/gauche). The glycine and cysteine pairs differ by
 * several coupled torsions and H-bonding patterns with no published
 * second-conformer geometry in bench-data; constructing them by hand would be a
 * guess, so they are reported as "not yet evaluated" rather than fabricated.
 */
import { readFileSync } from "node:fs";
import type { EnergyForceProvider } from "@browser-comp-chem/engine";
import { HARTREE_TO_KCAL_MOL } from "./chem.js";
import { BENCH_DATA_DIR } from "./paths.js";
import { relax, STATIONARY_THRESHOLD } from "./provider.js";
import { bondGraph, dihedral, loadXyz, setDihedral, toMolecule } from "./xyz.js";
import { computeErrorStats } from "../stats.js";

interface TorsionSpec {
  /** Resolve the (i,j,k,l) dihedral atom indices from the base geometry. */
  atoms: (symbols: string[], pos: Float64Array) => [number, number, number, number];
  refDeg: number;
  confDeg: number;
}

interface ConformerCase {
  id: string; // matches bench-data/conformers.json entry id
  geometryFile: string;
  torsion: TorsionSpec;
}

/** Find the hydroxyl hydrogen: the H bonded to the (first) oxygen. */
function hydroxylH(symbols: string[], pos: Float64Array): number {
  const adj = bondGraph(symbols, pos);
  const o = symbols.indexOf("O");
  for (const nb of adj[o] ?? []) if (symbols[nb] === "H") return nb;
  throw new Error("no hydroxyl H found");
}

const CASES: ConformerCase[] = [
  {
    id: "butane_gauche_anti",
    geometryFile: "n-butane_anti.xyz",
    // C0-C1-C2-C3 backbone torsion; anti = 180, gauche ~ 65.
    torsion: { atoms: () => [0, 1, 2, 3], refDeg: 180, confDeg: 65 },
  },
  {
    id: "ethanol_gauche_trans",
    geometryFile: "ethanol.xyz",
    // C-C-O-H torsion; trans/anti = 180, gauche ~ 60.
    torsion: {
      atoms: (s, p) => {
        const c0 = 0; // methyl C
        const c1 = 1; // methylene C
        const o = s.indexOf("O");
        return [c0, c1, o, hydroxylH(s, p)];
      },
      refDeg: 180,
      confDeg: 60,
    },
  },
];

export interface ConformerResult {
  id: string;
  molecule: string;
  status: "evaluated" | "not_evaluated";
  predictedKcal?: number;
  referenceKcal?: number;
  signedErrorKcal?: number;
  absErrorKcal?: number;
  refConformerMaxForce?: number;
  confConformerMaxForce?: number;
  refConformerFinalDeg?: number;
  confConformerFinalDeg?: number;
  stationary?: boolean;
  energyType?: string;
  levelOfTheory?: string;
  uncertaintyKcal?: number | undefined;
  citation?: { source: string; reference: string; level: string };
  note?: string | undefined;
}

export interface ConformerReport {
  results: ConformerResult[];
  stats: { n: number; mae: number; rmse: number; max: number };
  unit: "kcal/mol";
  chemicalAccuracyKcal: 1.0;
}

interface ConfEntry {
  id: string;
  molecule: string;
  value: number;
  uncertainty?: number;
  energy_type: string;
  level_of_theory: string;
  source: { paper: string; doi: string; url: string };
  reference_conformer: string;
  conformer: string;
}

export async function evaluateConformers(
  provider: EnergyForceProvider,
  opts: { ids?: string[] } = {},
): Promise<ConformerReport> {
  const data = JSON.parse(
    readFileSync(`${BENCH_DATA_DIR}/conformers.json`, "utf8"),
  ) as { entries: ConfEntry[]; deferred?: { id: string; reason: string }[] };
  const byId = new Map(data.entries.map((e) => [e.id, e]));

  const results: ConformerResult[] = [];
  const errors: number[] = [];

  const activeCases = opts.ids ? CASES.filter((c) => opts.ids!.includes(c.id)) : CASES;

  for (const c of activeCases) {
    const entry = byId.get(c.id)!;
    const base = loadXyz(`${BENCH_DATA_DIR}/geometries/${c.geometryFile}`);
    const [i, j, k, l] = c.torsion.atoms(base.symbols, base.positions);

    const refStart = toMolecule({
      ...base,
      positions: setDihedral(base.symbols, base.positions, i, j, k, l, c.torsion.refDeg),
    });
    const confStart = toMolecule({
      ...base,
      positions: setDihedral(base.symbols, base.positions, i, j, k, l, c.torsion.confDeg),
    });

    const refRelax = await relax(refStart, provider);
    const confRelax = await relax(confStart, provider);

    const refFinalDeg =
      (dihedral(refRelax.molecule.positions, i, j, k, l) * 180) / Math.PI;
    const confFinalDeg =
      (dihedral(confRelax.molecule.positions, i, j, k, l) * 180) / Math.PI;

    const predictedKcal = (confRelax.energy - refRelax.energy) * HARTREE_TO_KCAL_MOL;
    const referenceKcal = entry.value;
    const signed = predictedKcal - referenceKcal;
    const stationary =
      refRelax.maxForce < STATIONARY_THRESHOLD && confRelax.maxForce < STATIONARY_THRESHOLD;

    errors.push(signed);
    results.push({
      id: c.id,
      molecule: entry.molecule,
      status: "evaluated",
      predictedKcal,
      referenceKcal,
      signedErrorKcal: signed,
      absErrorKcal: Math.abs(signed),
      refConformerMaxForce: refRelax.maxForce,
      confConformerMaxForce: confRelax.maxForce,
      refConformerFinalDeg: refFinalDeg,
      confConformerFinalDeg: confFinalDeg,
      stationary,
      energyType: entry.energy_type,
      levelOfTheory: entry.level_of_theory,
      uncertaintyKcal: entry.uncertainty,
      citation: {
        source: entry.source.paper,
        reference: entry.source.doi,
        level: entry.level_of_theory,
      },
      note:
        Math.sign(predictedKcal) !== Math.sign(referenceKcal) && Math.abs(referenceKcal) > 0.05
          ? "SIGN MISMATCH vs reference ordering"
          : undefined,
    });
  }

  // Deferred systems: report honestly as not-yet-evaluated.
  if (!opts.ids) {
    for (const entry of data.entries) {
      if (CASES.some((c) => c.id === entry.id)) continue;
      results.push({
        id: entry.id,
        molecule: entry.molecule,
        status: "not_evaluated",
        referenceKcal: entry.value,
        energyType: entry.energy_type,
        levelOfTheory: entry.level_of_theory,
        uncertaintyKcal: entry.uncertainty,
        citation: {
          source: entry.source.paper,
          reference: entry.source.doi,
          level: entry.level_of_theory,
        },
        note:
          "Not yet evaluated: differs from the reference conformer by several coupled torsions / H-bonding patterns with no second-conformer geometry in bench-data. Constructing it by hand would be a guess (see bench-data caveat on level-sensitive sub-kcal orderings).",
      });
    }
  }

  const s = computeErrorStats(errors);
  return {
    results,
    stats: { n: s.n, mae: s.mae, rmse: s.rmse, max: s.max },
    unit: "kcal/mol",
    chemicalAccuracyKcal: 1.0,
  };
}
