/**
 * Accuracy suite entry point. Runs all three comparison modules against ANI-2x
 * and writes:
 *   - accuracy-report.json (repo root)   machine-readable, per-molecule, cited
 *   - ACCURACY_REPORT.md   (repo root)   human-readable summary + caveats
 * and prints the headline numbers to stdout.
 *
 * Run with:  npm run accuracy   (from packages/bench)
 */
import { writeFileSync } from "node:fs";
import { getProvider } from "./provider.js";
import { evaluateConformers } from "./conformers.js";
import { evaluateFrequencies } from "./frequencies.js";
import { evaluateHeats } from "./heats.js";
import { buildReport, renderMarkdown } from "./report.js";
import { REPO_ROOT } from "./paths.js";

async function main(): Promise<void> {
  const t0 = Date.now();
  const provider = await getProvider();
  const model = provider.name;

  process.stdout.write("[accuracy] 1/3 conformer relative energies...\n");
  const conformers = await evaluateConformers(provider);

  process.stdout.write("[accuracy] 2/3 vibrational frequencies (Hessians)...\n");
  const frequencies = await evaluateFrequencies(provider);

  process.stdout.write("[accuracy] 3/3 heats of formation (atomization diagnostic)...\n");
  const heats = await evaluateHeats(provider);

  const report = buildReport(conformers, frequencies, heats, model);

  const jsonPath = `${REPO_ROOT}accuracy-report.json`;
  const mdPath = `${REPO_ROOT}ACCURACY_REPORT.md`;
  writeFileSync(jsonPath, JSON.stringify(report, null, 2) + "\n", "utf8");
  writeFileSync(mdPath, renderMarkdown(report), "utf8");

  // Headline numbers to stdout.
  const c = conformers.stats;
  const fo = frequencies.overall;
  const nl = "\n";
  process.stdout.write(nl + "=== ANI-2x accuracy headline ===" + nl);
  process.stdout.write(
    `conformers:  MAE ${c.mae.toFixed(3)} kcal/mol, max ${c.max.toFixed(3)} (${c.n} pairs; bar 1.0)` + nl,
  );
  for (const r of conformers.results) {
    if (r.status === "evaluated") {
      process.stdout.write(
        `   - ${r.molecule}: pred ${r.predictedKcal!.toFixed(3)}, ref ${r.referenceKcal!.toFixed(3)}` +
          (r.note ? `  [${r.note}]` : "") + nl,
      );
    }
  }
  process.stdout.write(
    `frequencies: raw MAE ${fo.maeRaw.toFixed(1)} cm^-1 (${fo.meanPctRaw.toFixed(1)}%), ` +
      `scaled x${frequencies.scaleFactor} MAE ${fo.maeScaled.toFixed(1)} cm^-1 (${fo.meanPctScaled.toFixed(1)}%), ${fo.n} modes` + nl,
  );
  process.stdout.write(
    `heats of formation: ${heats.evaluated ? `MAE ${heats.stats.mae.toFixed(1)} kJ/mol` : "NOT EVALUATED (see methodology/blocker)"}` + nl,
  );
  process.stdout.write(nl + `wrote ${jsonPath}` + nl + `wrote ${mdPath}` + nl);
  process.stdout.write(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s` + nl);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
