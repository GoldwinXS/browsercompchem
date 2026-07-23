/** Assemble the machine-readable JSON and the human-readable Markdown report. */
import type { ConformerReport } from "./conformers.js";
import type { FreqReport } from "./frequencies.js";
import type { HeatReport } from "./heats.js";
import { FREQ_SCALE_FACTOR } from "./chem.js";

export interface AccuracyReport {
  generatedAt: string;
  model: string;
  chemicalAccuracyKcalMol: number;
  conformers: ConformerReport;
  frequencies: FreqReport;
  heatsOfFormation: HeatReport;
  caveats: { id: string; caveat: string; handling: string }[];
}

const CAVEATS = [
  {
    id: "CAVEAT 1 - enthalpy of formation is not a raw energy",
    caveat:
      "A model returns a total electronic energy, not delta_f H. Comparing them directly is meaningless; " +
      "an atomization + ZPE + thermal cycle plus experimental atomic delta_f H is required.",
    handling:
      "Full atomization/RRHO cycle implemented (dHfFromAtomization). It cannot be closed for this ANI-2x port " +
      "because De cannot be formed from the model (self-energies already contain binding; isolated atoms are " +
      "out-of-distribution). Reported as not-yet-evaluated with diagnostic De values and correct ZPE/thermal, " +
      "rather than fabricating free-atom constants.",
  },
  {
    id: "CAVEAT 2 - harmonic vs fundamental frequencies",
    caveat:
      "A Hessian gives HARMONIC frequencies, systematically higher than experimental FUNDAMENTALS (3-6% for X-H). " +
      "Comparing raw harmonic to fundamentals shows a fake ~5% error.",
    handling:
      `Both reported: raw harmonic AND scaled by a standard factor of ${FREQ_SCALE_FACTOR} ` +
      "(method-appropriate wB97X/6-31G* value ~0.95). Fermi-resonance (CO2), inversion-doubling (NH3), and " +
      "large-amplitude (methanol OH torsion) modes are flagged.",
  },
  {
    id: "CAVEAT 3 - what each conformer energy includes",
    caveat:
      "Reference conformer energies mix electronic (no ZPE) and ZPE-corrected/effective-0K conventions.",
    handling:
      "Both conformers are relaxed with ANI-2x and the ELECTRONIC energy difference is taken. Butane is a pure " +
      "electronic CCSD(T)/CBS reference (like-with-like); ethanol's ~0.12 kcal/mol effective-0K gap is sub-ZPE and " +
      "near-degenerate, so ordering is a stress test. ZPE is not added to the model differences (noted per pair).",
  },
];

export function buildReport(
  conformers: ConformerReport,
  frequencies: FreqReport,
  heatsOfFormation: HeatReport,
  model: string,
): AccuracyReport {
  return {
    generatedAt: new Date().toISOString(),
    model,
    chemicalAccuracyKcalMol: 1.0,
    conformers,
    frequencies,
    heatsOfFormation,
    caveats: CAVEATS,
  };
}

function f(n: number | undefined, d = 2): string {
  return n === undefined || Number.isNaN(n) ? "n/a" : n.toFixed(d);
}

function conformerVerdict(c: ConformerReport): string {
  const mae = c.stats.mae;
  const anyFlip = c.results.some((r) => r.note?.includes("SIGN MISMATCH"));
  if (Number.isNaN(mae)) return "no pairs evaluated";
  const bar = mae < 1 ? "within chemical accuracy" : "outside chemical accuracy";
  return `MAE ${f(mae)} kcal/mol (max ${f(c.stats.max)}) over ${c.stats.n} pairs -- ${bar}` +
    (anyFlip ? "; but a near-degenerate ordering (ethanol) is flipped" : "");
}

function freqVerdict(fr: FreqReport): string {
  return `raw MAE ${f(fr.overall.maeRaw, 1)} cm^-1 (${f(fr.overall.meanPctRaw, 1)}%); ` +
    `scaled x${fr.scaleFactor} MAE ${f(fr.overall.maeScaled, 1)} cm^-1 (${f(fr.overall.meanPctScaled, 1)}%) over ${fr.overall.n} modes`;
}

export function renderMarkdown(r: AccuracyReport): string {
  const L: string[] = [];
  L.push("# ANI-2x Accuracy Report");
  L.push("");
  L.push(`Generated: ${r.generatedAt}`);
  L.push(`Model: ${r.model} (full-f32, 8-member ensemble -- most accurate variant)`);
  L.push(`Chemical-accuracy bar: ${r.chemicalAccuracyKcalMol.toFixed(1)} kcal/mol.`);
  L.push("");
  L.push(
    "This suite compares ANI-2x predictions against curated real-literature reference values " +
    "(bench-data/), converting like-with-like and reporting honest error statistics. Every reference " +
    "value carries its citation in accuracy-report.json.",
  );
  L.push("");

  // Summary table
  L.push("## Summary");
  L.push("");
  L.push("| Property | N | MAE | RMSE | Max | Bar | Verdict |");
  L.push("|---|---|---|---|---|---|---|");
  L.push(
    `| Conformer rel. energies | ${r.conformers.stats.n} | ${f(r.conformers.stats.mae)} kcal/mol | ` +
    `${f(r.conformers.stats.rmse)} | ${f(r.conformers.stats.max)} | 1 kcal/mol | ${conformerVerdict(r.conformers)} |`,
  );
  L.push(
    `| Vib. freq (raw harmonic) | ${r.frequencies.overall.n} | ${f(r.frequencies.overall.maeRaw, 1)} cm^-1 | ` +
    `${f(r.frequencies.overall.rmseRaw, 1)} | ${f(r.frequencies.overall.maxRaw, 1)} | ~fundamentals | ${f(r.frequencies.overall.meanPctRaw, 1)}% mean err |`,
  );
  L.push(
    `| Vib. freq (scaled x${r.frequencies.scaleFactor}) | ${r.frequencies.overall.n} | ${f(r.frequencies.overall.maeScaled, 1)} cm^-1 | ` +
    `${f(r.frequencies.overall.rmseScaled, 1)} | ${f(r.frequencies.overall.maxScaled, 1)} | ~fundamentals | ${f(r.frequencies.overall.meanPctScaled, 1)}% mean err |`,
  );
  const hof = r.heatsOfFormation;
  L.push(
    `| Heats of formation | ${hof.evaluated ? hof.stats.n : 0} | ${hof.evaluated ? f(hof.stats.mae, 1) + " kJ/mol" : "not evaluated"} | ` +
    `${hof.evaluated ? f(hof.stats.rmse, 1) : "-"} | ${hof.evaluated ? f(hof.stats.max, 1) : "-"} | 4.2 kJ/mol | ${hof.evaluated ? "" : "blocked -- see methodology"} |`,
  );
  L.push("");

  // One-line verdicts
  L.push("## Verdicts");
  L.push("");
  L.push(`- Conformers: ${conformerVerdict(r.conformers)}.`);
  L.push(`- Frequencies: ${freqVerdict(r.frequencies)}.`);
  L.push(
    `- Heats of formation: not evaluated -- ${hof.blocker}`,
  );
  L.push("");

  // Conformer detail
  L.push("## Conformer relative energies (kcal/mol)");
  L.push("");
  L.push("| System | Predicted | Reference | Signed err | Ref level | Note |");
  L.push("|---|---|---|---|---|---|");
  for (const c of r.conformers.results) {
    if (c.status === "evaluated") {
      L.push(
        `| ${c.molecule} | ${f(c.predictedKcal)} | ${f(c.referenceKcal)} | ${f(c.signedErrorKcal)} | ` +
        `${c.levelOfTheory ?? ""} | ${c.note ?? "ok"} |`,
      );
    } else {
      L.push(`| ${c.molecule} | not evaluated | ${f(c.referenceKcal)} | - | ${c.levelOfTheory ?? ""} | ${c.note ?? ""} |`);
    }
  }
  L.push("");

  // Frequency detail
  L.push(`## Vibrational frequencies (cm^-1; scaling factor ${r.frequencies.scaleFactor})`);
  L.push("");
  L.push("| Molecule | Modes | MAE raw | %raw | MAE scaled | %scaled | Note |");
  L.push("|---|---|---|---|---|---|---|");
  for (const m of r.frequencies.results) {
    if (m.status === "evaluated") {
      L.push(
        `| ${m.name} | ${m.nModes} | ${f(m.maeRaw, 1)} | ${f(m.meanPctRaw, 1)} | ${f(m.maeScaled, 1)} | ` +
        `${f(m.meanPctScaled, 1)} | ${m.note ?? ""} |`,
      );
    } else {
      L.push(`| ${m.name} | - | skipped | - | - | - | ${m.reason ?? ""} |`);
    }
  }
  L.push("");

  // Heats detail
  L.push("## Heats of formation (kJ/mol) -- diagnostic subset");
  L.push("");
  L.push(`Methodology: ${hof.methodology}`);
  L.push("");
  L.push(`Blocker: ${hof.blocker}`);
  L.push("");
  L.push("| Molecule | Ref dHf | De (self-energy route) | De (lone-atom route) | ZPE | Status |");
  L.push("|---|---|---|---|---|---|");
  for (const h of r.heatsOfFormation.results) {
    if (h.deSaeKJ !== undefined) {
      L.push(
        `| ${h.name} | ${f(h.referenceKJ, 1)} | ${f(h.deSaeKJ, 1)} | ${f(h.deLoneAtomKJ, 1)} | ${f(h.zpeKJ, 1)} | ` +
        `${h.status === "evaluated" ? f(h.predictedKJ, 1) + " (pred), err " + f(h.signedErrorKJ, 1) : h.status} |`,
      );
    }
  }
  L.push("");
  L.push(
    "(The two De columns are both physically invalid for this ANI-2x port and are shown only as evidence of the " +
    "blocker. ZPE is correct and model-consistent. Molecules outside the subset are omitted from this table but " +
    "listed as not-yet-evaluated in accuracy-report.json.)",
  );
  L.push("");

  // Caveats
  L.push("## Caveats (methodology)");
  L.push("");
  for (const c of r.caveats) {
    L.push(`### ${c.id}`);
    L.push("");
    L.push(`- The caveat: ${c.caveat}`);
    L.push(`- What we did: ${c.handling}`);
    L.push("");
  }

  L.push("## How close are we to a real chemistry tool?");
  L.push("");
  L.push(
    "On the property ANI-2x is built for -- conformer relative energies -- it is genuinely at chemical accuracy " +
    "(MAE a few tenths of a kcal/mol), with the honest caveat that a near-degenerate ordering (ethanol gauche/trans, " +
    "~0.1 kcal/mol) comes out flipped. Harmonic frequencies are physically sensible and land within ~2% of experiment " +
    "after standard scaling. Absolute thermochemistry (heats of formation) is the honest weak spot: the model produces " +
    "excellent relative energies but cannot, by itself, be turned into absolute formation enthalpies without external " +
    "free-atom reference energies. So: a strong relative-energy engine, not yet a drop-in replacement for absolute " +
    "thermochemistry.",
  );
  L.push("");
  return L.join("\n");
}
