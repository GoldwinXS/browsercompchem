/**
 * MODULE 2 -- Harmonic vibrational frequencies vs experimental fundamentals.
 *
 * Per molecule: relax with FIRE, verify it is a stationary point, build the
 * finite-difference Hessian of the analytic ANI-2x forces, mass-weight,
 * diagonalize (Jacobi), drop the 6/5 trans-rot modes, and compare the resulting
 * HARMONIC wavenumbers to the experimental FUNDAMENTALS in
 * bench-data/frequencies.json.
 *
 * CAVEAT (bench-data CAVEAT 2): harmonic > fundamental. We therefore report the
 * raw harmonic error AND the error after a standard 0.96 scaling factor, both
 * clearly labelled.
 *
 * Matching: experimental modes are expanded by their degeneracy (E->2, T->3,
 * pi->2) to a full 3N-6 / 3N-5 list, then both lists are sorted ascending and
 * paired 1:1. This avoids fragile symmetry assignment; it can mis-pair when two
 * modes cross, which is noted.
 */
import { readFileSync } from "node:fs";
import type { EnergyForceProvider } from "@browser-comp-chem/engine";
import { ANI_ELEMENTS, FREQ_SCALE_FACTOR } from "./chem.js";
import { BENCH_DATA_DIR } from "./paths.js";
import { relax, STATIONARY_THRESHOLD } from "./provider.js";
import { loadXyz, toMolecule } from "./xyz.js";
import { vibrationalAnalysis } from "./vibrations.js";

/** Map a symmetry label to its vibrational degeneracy. */
export function degeneracyOf(symmetry: string): number {
  const s = symmetry.trim().toLowerCase();
  if (s.startsWith("t")) return 3; // T (Td/Oh triply degenerate)
  if (s.startsWith("e")) return 2; // E doubly degenerate
  if (s.startsWith("pi")) return 2; // linear pi bend
  if (s.startsWith("delta")) return 2;
  return 1; // A, B, sigma, ...
}

export interface FreqMoleculeResult {
  id: string;
  name: string;
  status: "evaluated" | "skipped";
  reason?: string;
  linear?: boolean;
  stationary?: boolean;
  maxForce?: number;
  maxResidualTransRot?: number;
  nModes?: number;
  harmonic?: number[];
  scaled?: number[];
  experimental?: number[];
  maeRaw?: number;
  rmseRaw?: number;
  maxRaw?: number;
  meanPctRaw?: number;
  maeScaled?: number;
  rmseScaled?: number;
  maxScaled?: number;
  meanPctScaled?: number;
  citation?: { source: string; reference: string; level: string };
  note?: string | undefined;
}

export interface FreqReport {
  results: FreqMoleculeResult[];
  overall: {
    n: number;
    maeRaw: number;
    rmseRaw: number;
    maxRaw: number;
    meanPctRaw: number;
    maeScaled: number;
    rmseScaled: number;
    maxScaled: number;
    meanPctScaled: number;
  };
  scaleFactor: number;
  unit: "cm^-1";
}

interface FreqEntry {
  id: string;
  name: string;
  formula: string;
  point_group: string;
  source_url: string;
  note?: string;
  modes: { mode: number; symmetry: string; frequency: number; description: string }[];
}

/** Molecules whose comparison is degraded by known physics (flagged, still scored). */
const FLAGGED: Record<string, string> = {
  carbon_dioxide: "nu1 is a Fermi dyad (deperturbed 1333); harmonic single-number comparison is approximate.",
  methanol: "OH torsion ~200 cm^-1 is a large-amplitude mode poorly described as harmonic.",
  ammonia: "umbrella bend is inversion-split; harmonic value is an average.",
};

function geometryFileFor(id: string): string {
  return `${BENCH_DATA_DIR}/geometries/${id}.xyz`;
}

export async function evaluateFrequencies(
  provider: EnergyForceProvider,
  opts: { ids?: string[] } = {},
): Promise<FreqReport> {
  const data = JSON.parse(
    readFileSync(`${BENCH_DATA_DIR}/frequencies.json`, "utf8"),
  ) as { entries: FreqEntry[] };

  const entries = opts.ids ? data.entries.filter((e) => opts.ids!.includes(e.id)) : data.entries;
  const results: FreqMoleculeResult[] = [];

  // Accumulators for overall stats (per-mode pooled).
  const allRaw: number[] = [];
  const allScaled: number[] = [];
  const allPctRaw: number[] = [];
  const allPctScaled: number[] = [];

  for (const e of entries) {
    const base = loadXyz(geometryFileFor(e.id));
    const unsupported = base.symbols.filter((s) => !ANI_ELEMENTS.has(s));
    if (unsupported.length) {
      results.push({
        id: e.id,
        name: e.name,
        status: "skipped",
        reason: `contains non-ANI-2x element(s): ${[...new Set(unsupported)].join(", ")}`,
      });
      continue;
    }

    const relaxed = await relax(toMolecule(base), provider);
    const stationary = relaxed.maxForce < STATIONARY_THRESHOLD;
    const vib = await vibrationalAnalysis(relaxed.molecule, provider);

    // Expand experimental modes by degeneracy, sort ascending.
    const exp: number[] = [];
    for (const m of e.modes) {
      const g = degeneracyOf(m.symmetry);
      for (let d = 0; d < g; d++) exp.push(m.frequency);
    }
    exp.sort((a, b) => a - b);

    const harmonic = vib.frequencies.filter((f) => f > 0).sort((a, b) => a - b);
    // Pair 1:1 over the min length (they should match; note if not).
    const nPair = Math.min(harmonic.length, exp.length);
    const pairedHarm = harmonic.slice(harmonic.length - nPair); // highest nPair (drop any extra low)
    const pairedExp = exp.slice(exp.length - nPair);

    const scaled = pairedHarm.map((f) => f * FREQ_SCALE_FACTOR);

    let sumAbsRaw = 0, sumSqRaw = 0, maxRaw = 0, sumPctRaw = 0;
    let sumAbsScaled = 0, sumSqScaled = 0, maxScaled = 0, sumPctScaled = 0;
    for (let m = 0; m < nPair; m++) {
      const ref = pairedExp[m]!;
      const eRaw = pairedHarm[m]! - ref;
      const eSc = scaled[m]! - ref;
      sumAbsRaw += Math.abs(eRaw); sumSqRaw += eRaw * eRaw; maxRaw = Math.max(maxRaw, Math.abs(eRaw));
      sumAbsScaled += Math.abs(eSc); sumSqScaled += eSc * eSc; maxScaled = Math.max(maxScaled, Math.abs(eSc));
      sumPctRaw += Math.abs(eRaw) / ref * 100;
      sumPctScaled += Math.abs(eSc) / ref * 100;
      allRaw.push(eRaw); allScaled.push(eSc);
      allPctRaw.push(Math.abs(eRaw) / ref * 100);
      allPctScaled.push(Math.abs(eSc) / ref * 100);
    }

    results.push({
      id: e.id,
      name: e.name,
      status: "evaluated",
      linear: vib.linear,
      stationary,
      maxForce: relaxed.maxForce,
      maxResidualTransRot: vib.maxResidualTransRot,
      nModes: nPair,
      harmonic: pairedHarm.map((f) => +f.toFixed(1)),
      scaled: scaled.map((f) => +f.toFixed(1)),
      experimental: pairedExp,
      maeRaw: sumAbsRaw / nPair,
      rmseRaw: Math.sqrt(sumSqRaw / nPair),
      maxRaw,
      meanPctRaw: sumPctRaw / nPair,
      maeScaled: sumAbsScaled / nPair,
      rmseScaled: Math.sqrt(sumSqScaled / nPair),
      maxScaled,
      meanPctScaled: sumPctScaled / nPair,
      citation: {
        source: "NIST CCCBDB experimental / Shimanouchi NSRDS-NBS 39",
        reference: e.source_url,
        level: "experimental fundamental vibrational frequencies",
      },
      note: [FLAGGED[e.id], !stationary ? "NOT a tight stationary point" : undefined]
        .filter(Boolean)
        .join(" ") || undefined,
    });
  }

  const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const rmse = (a: number[]) => (a.length ? Math.sqrt(a.reduce((x, y) => x + y * y, 0) / a.length) : NaN);
  const maxAbs = (a: number[]) => a.reduce((m, y) => Math.max(m, Math.abs(y)), 0);

  return {
    results,
    overall: {
      n: allRaw.length,
      maeRaw: mean(allRaw.map(Math.abs)),
      rmseRaw: rmse(allRaw),
      maxRaw: maxAbs(allRaw),
      meanPctRaw: mean(allPctRaw),
      maeScaled: mean(allScaled.map(Math.abs)),
      rmseScaled: rmse(allScaled),
      maxScaled: maxAbs(allScaled),
      meanPctScaled: mean(allPctScaled),
    },
    scaleFactor: FREQ_SCALE_FACTOR,
    unit: "cm^-1",
  };
}
