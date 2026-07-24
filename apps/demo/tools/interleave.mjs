/**
 * Permanent head-of-line-blocking bench for the optimizer worker.
 *
 * REGRESSION GUARDED: the ANI-2x provider's energyForces() is declared async but
 * contains no real await -- it is synchronous math wrapped in a promise. So a
 * FIRE optimization (or a 6N-evaluation Hessian) resolves entirely through
 * MICROtasks, and a worker only dispatches `message` events as MACROtasks. Before
 * the cooperative-yield fix, self.onmessage in optimizer.worker.ts could not run
 * again until the current op had fully completed: a molecule the user selected
 * mid-computation stayed frozen at its raw unrelaxed seed (for a SMILES molecule,
 * RDKit's near-planar 2D layout) until the ABANDONED op finished grinding.
 *
 * Measured before the fix: a perturbed-water optimize took 1219 ms from idle but
 * 12805 ms (10.5x) when requested while an abandoned naphthalene (18-atom) embed
 * was still running; this bench reproduced it at 1286 ms vs 13515 ms (10.51x),
 * with the worker-side FIRE time unchanged at ~1.15 s -- i.e. all of the extra
 * 12 s was queue wait, not slower math. After the fix the abandoned op bails out
 * at its next yield point and the ratio collapses to ~1x (measured 0.97x).
 *
 * The bench fails if the interleaved optimize takes more than 3x the from-idle
 * time. It also prints the from-idle baseline itself, which is the guard against
 * over-yielding: if a future change to YIELD_EVERY pushes `baselineMs` well above
 * ~1219 ms, the yields have started costing more than they buy.
 *
 * RASTER ONLY: ray tracing is forced off (it hangs under automation on this
 * machine); the bench never touches window.__rt or demo.rtEnabled.
 *
 * Uses playwright-core against the installed Chrome (channel: "chrome",
 * headless: false) so no browser download is needed. Needs the dev server up:
 *
 *   npm run dev                       # separate shell, serves http://localhost:8142/
 *   npm run interleave -w apps/demo
 */
import { chromium } from "playwright-core";

const URL = "http://localhost:8142/";

/** Naphthalene: 18 atoms, the largest molecule the fuzz pool uses. */
const BLOCKER_SMILES = "c1ccc2ccccc2c1";

/** Fail threshold: interleaved / from-idle. Pre-fix this ratio was 10.5x. */
const MAX_RATIO = 3;

/** How long to let the blocker run before abandoning it (must be well inside it). */
const BLOCKER_LEAD_MS = 300;

const READY_TIMEOUT = 180_000;
const OPT_TIMEOUT = 180_000;

/** Thrown when the bench itself cannot set up the scenario it means to measure. */
class BenchError extends Error {}

function log(line) {
  process.stdout.write(`${line}\n`);
}

async function main() {
  log(`\ninterleave-bench: url=${URL} threshold=${MAX_RATIO}x\n`);

  const browser = await chromium.launch({ channel: "chrome", headless: false });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();

  // Force raster + no auto-rotate before any app code runs (fresh profile, but
  // be explicit so a persisted RT pref can never turn the tracer on).
  await page.addInitScript(() => {
    try {
      localStorage.setItem("bcc.pref.rt", "0");
      localStorage.setItem("bcc.pref.autoRotate", "0");
    } catch {
      /* private mode — defaults are already raster/off */
    }
  });

  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e && e.stack ? e.stack : e)));
  page.on("console", (m) => {
    if (m.type() === "error") log(`  · console.error: ${m.text()}`);
  });

  const flags = () =>
    page.evaluate(() => {
      const d = window.__demo;
      const dis = (id) => {
        const e = document.getElementById(id);
        return e ? !!e.disabled : true;
      };
      return {
        ready: d.ready,
        optimizing: d.optimizing,
        loading: d.loading,
        done: d.done,
        molecule: d.molecule,
        atomCount: d.atomCount,
        elapsedMs: d.elapsedMs,
        converged: d.converged,
        // Streamed {step,...} messages, i.e. energy/force evaluations - 1. The
        // per-evaluation cost is what the yield overhead is measured against.
        evals: d.step + 1,
        disOptimize: dis("optimize"),
        disPerturb: dis("perturb"),
        error: d.error,
      };
    });

  const waitSettle = (timeout) =>
    page.waitForFunction(
      () => {
        const d = window.__demo;
        return d && !d.optimizing && !d.loading && !d.vibComputing && !d.orbComputing && !d.irComputing;
      },
      undefined,
      { timeout },
    );

  const waitDone = (timeout) =>
    page.waitForFunction(() => window.__demo && window.__demo.done === true, undefined, { timeout });

  const clickId = (id) => page.evaluate((i) => document.getElementById(i).click(), id);

  const selectPreset = (value) =>
    page.evaluate((val) => {
      const sel = document.getElementById("mol");
      sel.value = val;
      // dispatch rather than click: this fires the change handler even while the
      // control is disabled, which is exactly the state under test (the app locks
      // the dropdown during a compute, but we need the geometry switch to happen
      // WHILE the abandoned worker op is running).
      sel.dispatchEvent(new Event("change"));
    }, value);

  /** Perturb the current preset geometry, optimize it, and time to demo.done. */
  async function timedPerturbOptimize(label) {
    const before = await flags();
    if (before.disPerturb || before.disOptimize) {
      throw new BenchError(
        `${label}: perturb/optimize disabled (perturb=${before.disPerturb} optimize=${before.disOptimize}) — cannot time the op`,
      );
    }
    await clickId("perturb");
    const t0 = Date.now();
    await clickId("optimize");
    await waitDone(OPT_TIMEOUT);
    const wallMs = Date.now() - t0;
    const after = await flags();
    log(
      `  ${label}: wall ${wallMs} ms (worker-side FIRE ${Math.round(after.elapsedMs ?? -1)} ms, ` +
        `${after.evals} evaluations, converged=${after.converged})`,
    );
    return wallMs;
  }

  try {
    await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    log("waiting for model ready…");
    await page.waitForFunction(() => window.__demo && window.__demo.ready === true, undefined, {
      timeout: READY_TIMEOUT,
    });
    await waitSettle(READY_TIMEOUT);
    const boot = await flags();
    log(`model ready. boot molecule=${boot.molecule} atoms=${boot.atomCount}\n`);

    // --- 1. from-idle reference: perturbed water, nothing else running --------
    await selectPreset("water");
    await waitSettle(OPT_TIMEOUT);
    const baselineMs = await timedPerturbOptimize("from idle   ");

    // --- 2. same op, requested while an abandoned embed is running ------------
    log(`  starting blocker: embed "${BLOCKER_SMILES}" (18 atoms)…`);
    await page.evaluate((smi) => {
      const inp = document.getElementById("smiles");
      inp.value = smi;
      document.getElementById("load-smiles").click();
    }, BLOCKER_SMILES);
    await page.waitForFunction(() => window.__demo && window.__demo.optimizing === true, undefined, {
      timeout: OPT_TIMEOUT,
    });
    await page.waitForTimeout(BLOCKER_LEAD_MS); // let the embed get properly under way

    // Abandon it by switching molecules mid-flight -- the exact user action that
    // used to leave the new molecule frozen at its raw seed. Note this posts NO
    // compute request: the worker only learns the geometry changed if the app
    // tells it (the "abandon" message).
    await selectPreset("water");
    const mid = await flags();
    log(`  abandoned mid-embed: molecule=${mid.molecule} atoms=${mid.atomCount} optimizing=${mid.optimizing}`);

    const blockedMs = await timedPerturbOptimize("interleaved ");

    if (pageErrors.length) {
      throw new BenchError(`page error(s):\n${pageErrors.join("\n---\n")}`);
    }
    // Abandoning an op is intentional, not a failure: the worker must post no
    // "error" reply for it (a live-epoch error would surface in the UI).
    const end = await flags();
    if (end.error) throw new BenchError(`demo.error set after an abandoned op: ${end.error}`);

    const ratio = blockedMs / Math.max(1, baselineMs);
    log(`\nbaselineMs=${baselineMs}  blockedMs=${blockedMs}  ratio=${ratio.toFixed(2)}x  (limit ${MAX_RATIO}x)`);

    await browser.close();
    if (ratio > MAX_RATIO) {
      log(
        `\nFAIL: an optimize requested while an abandoned op was running took ${ratio.toFixed(2)}x the ` +
          `from-idle time. The worker is not yielding/cancelling — see optimizer.worker.ts (yieldToEventLoop, ` +
          `CooperativeProvider, the "abandon" message) and main.ts resetForNewGeometry.\n`,
      );
      process.exit(1);
    }
    log(`\nPASS: interleaved optimize within ${MAX_RATIO}x of from-idle; abandoned work is cancelled promptly.\n`);
    process.exit(0);
  } catch (err) {
    log(`\nFAIL: ${err instanceof Error ? err.message : String(err)}`);
    if (pageErrors.length) log(`\npage errors:\n${pageErrors.join("\n---\n")}`);
    try {
      await browser.close();
    } catch {
      /* ignore */
    }
    process.exit(1);
  }
}

main().catch((err) => {
  process.stderr.write(`interleave harness crashed: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(2);
});
