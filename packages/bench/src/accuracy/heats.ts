/**
 * MODULE 3 -- Standard enthalpies of formation via the atomization route.
 *
 * A model returns a total electronic energy, NOT a delta_f H. The intended
 * conversion (bench-data CAVEAT 1) is an atomization + RRHO thermochemistry
 * cycle:
 *
 *   De   = sum(E_free_atom)  -  E_molecule                 [electronic atomization]
 *   D0   = De - ZPE
 *   dHf(298,M) = sum_atoms dHf298(atom,gas) - D0
 *                - n_atoms*(5/2 RT) + [H298-H0](M, RRHO)
 *
 * HONEST BLOCKER (a real finding this suite exists to surface): the cycle needs
 * E_free_atom at ANI-2x's reference level (wB97X/6-31G*). Two from-model routes
 * were tried and BOTH fail for this ANI-2x port:
 *
 *   (a) self-energies as atom references. ANI-2x ships linear-fit self-atomic-
 *       energies (SAE) that already absorb ~0.34 Ha/atom of binding, so
 *       De_sae = sum(SAE) - E_mol collapses to ~0 (water: ~10 kJ/mol vs the
 *       physical ~971 kJ/mol). SAE are NOT free-atom energies.
 *   (b) evaluating ANI-2x on an isolated atom. ANI-2x is undefined out here
 *       (zero AEV); the lone-atom energies are unphysical, worst for heavy atoms
 *       (lone O is ~0.14 Ha too low), giving De_loneatom(water)=451 kJ/mol.
 *
 * A valid conversion therefore requires verified wB97X/6-31G* free-atom total
 * energies, which were not available from a citable source this session. Rather
 * than fabricate seven atomic constants (each 0.01 Ha error = 26 kJ/mol), every
 * molecule is reported as "not yet evaluated", WITH the diagnostic De values and
 * the (correct, model-independent) ZPE + thermal terms so the blocker is fully
 * evidenced and the cycle is ready the moment the constants are supplied.
 *
 * The full Hess cycle is implemented in dHfFromAtomization() below and is
 * exercised end-to-end whenever `freeAtomEnergies` is provided.
 */
import { readFileSync } from "node:fs";
import type { EnergyForceProvider, Molecule } from "@browser-comp-chem/engine";
import {
  ATOMIC_DHF298_CITATION,
  ATOMIC_DHF298_KJ,
  ANI_ELEMENTS,
  HARTREE_TO_KJ_MOL,
  RT_KJ,
} from "./chem.js";
import { BENCH_DATA_DIR, MODEL_DIR } from "./paths.js";
import { relax, STATIONARY_THRESHOLD } from "./provider.js";
import { loadXyz, toMolecule } from "./xyz.js";
import { isLinear, vibrationalAnalysis } from "./vibrations.js";
import { computeErrorStats } from "../stats.js";

/** Subset for which we compute the diagnostic + (if constants supplied) the full cycle. */
const SUBSET_IDS = [
  "water",
  "ammonia",
  "methane",
  "carbon_dioxide",
  "carbon_monoxide",
  "hydrogen_cyanide",
  "formaldehyde",
  "ethylene",
  "methanol",
  "benzene",
];

interface ThermoEntry {
  id: string;
  name: string;
  formula: string;
  value: number;
  uncertainty: number | null;
  verification: string;
  source: { database: string; url: string };
  note?: string;
}

function selfEnergyMap(): Record<string, number> {
  const manifest = JSON.parse(readFileSync(`${MODEL_DIR}/manifest.json`, "utf8")) as {
    symbols: string[];
    self_energies: number[];
  };
  const map: Record<string, number> = {};
  manifest.symbols.forEach((s, i) => (map[s] = manifest.self_energies[i]!));
  return map;
}

export interface AtomizationInput {
  symbols: string[];
  eMolHartree: number;
  zpeKJ: number;
  thermalMolKJ: number;
  freeAtomEnergies: Record<string, number>; // Hartree, ANI-2x reference level
}

/** The full, correct Hess cycle. Ready for use once verified free-atom energies exist. */
export function dHfFromAtomization(inp: AtomizationInput): number {
  let sumFree = 0;
  let sumAtomDhf = 0;
  for (const s of inp.symbols) {
    sumFree += inp.freeAtomEnergies[s]!;
    sumAtomDhf += ATOMIC_DHF298_KJ[s]!;
  }
  const deKJ = (sumFree - inp.eMolHartree) * HARTREE_TO_KJ_MOL;
  const d0KJ = deKJ - inp.zpeKJ;
  const atomThermal = inp.symbols.length * 2.5 * RT_KJ;
  return sumAtomDhf - d0KJ - atomThermal + inp.thermalMolKJ;
}

export interface HeatResult {
  id: string;
  name: string;
  formula: string;
  status: "evaluated" | "not_evaluated" | "skipped";
  reason?: string;
  referenceKJ?: number;
  uncertaintyKJ?: number | null;
  predictedKJ?: number;
  signedErrorKJ?: number;
  absErrorKJ?: number;
  // diagnostics (always populated for the subset)
  deSaeKJ?: number; // atomization via shipped self-energies (invalid; ~0)
  deLoneAtomKJ?: number; // atomization via lone-atom ANI energies (invalid)
  zpeKJ?: number; // harmonic ZPE (valid, model-consistent)
  thermalMolKJ?: number; // 0->298 enthalpy of the molecule (valid)
  stationary?: boolean;
  maxForce?: number;
  citation?: { source: string; reference: string; level: string };
  note?: string | undefined;
}

export interface HeatReport {
  results: HeatResult[];
  stats: { n: number; mae: number; rmse: number; max: number };
  unit: "kJ/mol";
  evaluated: boolean;
  atomicDhfCitation: typeof ATOMIC_DHF298_CITATION;
  methodology: string;
  blocker: string;
}

/**
 * Evaluate heats of formation. If `freeAtomEnergies` (Hartree, ANI-2x reference
 * level) are supplied, the full cycle runs and produces numbers; otherwise the
 * property is reported as not-yet-evaluated with full diagnostics.
 */
export async function evaluateHeats(
  provider: EnergyForceProvider,
  opts: { ids?: string[]; freeAtomEnergies?: Record<string, number> } = {},
): Promise<HeatReport> {
  const data = JSON.parse(
    readFileSync(`${BENCH_DATA_DIR}/thermochemistry.json`, "utf8"),
  ) as { entries: ThermoEntry[] };
  const self = selfEnergyMap();

  const subset = opts.ids ?? SUBSET_IDS;
  const results: HeatResult[] = [];
  const errors: number[] = [];

  // Cache lone-atom ANI energies (for the diagnostic route).
  const loneAtom: Record<string, number> = {};
  async function loneAtomEnergy(sym: string): Promise<number> {
    if (loneAtom[sym] === undefined) {
      const m: Molecule = {
        symbols: [sym],
        positions: Float64Array.from([0, 0, 0]),
        charge: 0,
        multiplicity: 1,
      };
      loneAtom[sym] = (await provider.energyForces(m)).energy;
    }
    return loneAtom[sym]!;
  }

  for (const entry of data.entries) {
    if (!subset.includes(entry.id)) {
      results.push({
        id: entry.id,
        name: entry.name,
        formula: entry.formula,
        status: "not_evaluated",
        referenceKJ: entry.value,
        uncertaintyKJ: entry.uncertainty,
        note: "Not yet evaluated: outside the diagnostic subset (no Hessian computed this run).",
      });
      continue;
    }

    const base = loadXyz(`${BENCH_DATA_DIR}/geometries/${entry.id}.xyz`);
    const unsupported = base.symbols.filter((s) => !ANI_ELEMENTS.has(s));
    if (unsupported.length) {
      results.push({
        id: entry.id,
        name: entry.name,
        formula: entry.formula,
        status: "skipped",
        reason: `contains non-ANI-2x element(s): ${[...new Set(unsupported)].join(", ")}`,
        referenceKJ: entry.value,
      });
      continue;
    }

    const relaxed = await relax(toMolecule(base), provider);
    const stationary = relaxed.maxForce < STATIONARY_THRESHOLD;

    // ZPE + molecule thermal (valid, model-consistent).
    let zpeKJ = 0;
    let thermalMolKJ: number;
    if (base.symbols.length === 1) {
      thermalMolKJ = 1.5 * RT_KJ + RT_KJ;
    } else {
      const vib = await vibrationalAnalysis(relaxed.molecule, provider);
      zpeKJ = vib.zpeKJ;
      const linear = isLinear(base.symbols, relaxed.molecule.positions);
      const eRot = linear ? RT_KJ : 1.5 * RT_KJ;
      thermalMolKJ = 1.5 * RT_KJ + eRot + vib.vibThermalKJ + RT_KJ;
    }

    // Diagnostic atomization energies (both invalid; recorded as evidence).
    let sumSelf = 0;
    for (const s of base.symbols) sumSelf += self[s]!;
    const deSaeKJ = (sumSelf - relaxed.energy) * HARTREE_TO_KJ_MOL;
    let sumLone = 0;
    for (const s of base.symbols) sumLone += await loneAtomEnergy(s);
    const deLoneAtomKJ = (sumLone - relaxed.energy) * HARTREE_TO_KJ_MOL;

    const common: HeatResult = {
      id: entry.id,
      name: entry.name,
      formula: entry.formula,
      status: "not_evaluated",
      referenceKJ: entry.value,
      uncertaintyKJ: entry.uncertainty,
      deSaeKJ,
      deLoneAtomKJ,
      zpeKJ,
      thermalMolKJ,
      stationary,
      maxForce: relaxed.maxForce,
      citation: {
        source: entry.source.database,
        reference: entry.source.url,
        level: `experimental delta_f H(298.15 K), ${entry.verification}`,
      },
    };

    if (opts.freeAtomEnergies) {
      const predictedKJ = dHfFromAtomization({
        symbols: base.symbols,
        eMolHartree: relaxed.energy,
        zpeKJ,
        thermalMolKJ,
        freeAtomEnergies: opts.freeAtomEnergies,
      });
      const signed = predictedKJ - entry.value;
      errors.push(signed);
      results.push({
        ...common,
        status: "evaluated",
        predictedKJ,
        signedErrorKJ: signed,
        absErrorKJ: Math.abs(signed),
        note: !stationary ? "NOT a tight stationary point" : undefined,
      });
    } else {
      results.push({
        ...common,
        note:
          "Not yet evaluated: no verified wB97X/6-31G* free-atom reference energies available; " +
          `both from-model routes are invalid (De_sae=${deSaeKJ.toFixed(0)}, De_loneAtom=${deLoneAtomKJ.toFixed(0)} kJ/mol vs a physical atomization of hundreds of kJ/mol). ` +
          "ZPE and molecule thermal shown are correct and model-consistent.",
      });
    }
  }

  const s = computeErrorStats(errors);
  return {
    results,
    stats: { n: s.n, mae: s.mae, rmse: s.rmse, max: s.max },
    unit: "kJ/mol",
    evaluated: Boolean(opts.freeAtomEnergies),
    atomicDhfCitation: ATOMIC_DHF298_CITATION,
    methodology:
      "Atomization route: dHf(298) = sum(atomic dHf298) - (De - ZPE) - n*(5/2)RT + [H298-H0](molecule, RRHO), " +
      "with De = sum(E_free_atom) - E_molecule. Harmonic (unscaled) ZPE; classical trans+rot; harmonic vibrational thermal. " +
      "NOT applied: ZPE scaling, anharmonic/hindered-rotor corrections, atomic spin-orbit corrections.",
    blocker:
      "De cannot be formed from the model alone for this ANI-2x port: the shipped self-energies are linear-fit " +
      "baselines that already contain the binding (De_sae ~ 0), and ANI-2x is undefined on isolated atoms " +
      "(lone-atom route unphysical, worst for heavy atoms). Verified wB97X/6-31G* free-atom total energies are " +
      "required and were not fabricated. Supply them to dHfFromAtomization()/evaluateHeats({freeAtomEnergies}) to produce numbers.",
  };
}
