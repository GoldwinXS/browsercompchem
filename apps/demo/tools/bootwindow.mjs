/**
 * Permanent boot-window bench: a SMILES loaded before the model is ready must
 * still end up as a real 3D geometry.
 *
 * REGRESSION GUARDED: nothing locks the molecule dropdown or the Load button
 * while the ~26 MB ANI-2x weights stream in (no compute is running, so
 * lockControlsForCompute has never run). A first-time visitor who pastes a
 * SMILES in those first seconds took the not-ready branch of loadCustomMolecule:
 * it showed RDKit's planarity-broken 2D seed, said "press Optimize once the
 * model is ready", and queued NOTHING. The "ready" reply then overwrote that
 * status with "model ready — <variant>, <n> members", so the instruction
 * vanished too and the molecule sat at a flat cartoon indefinitely
 * (demo.done === false, demo.step === 0, demo.optimizing === false). This is the
 * FIRST thing a cold visitor does, which is what makes it worth a bench.
 *
 * The fix is a single-shot `pendingEmbed` in main.ts that the "ready" handler
 * consumes; the two cases below are its two halves.
 *
 *   1. AUTO-EMBED. Load cyclohexane during the window; once "ready" lands the
 *      embed must start by itself and finish as a genuine 3D structure.
 *      Cyclohexane is the right probe precisely because the two states are
 *      geometrically unmistakable: RDKit's 2D seed is a flat hexagon (a ~0.6 A
 *      z-jitter on an otherwise planar ring) while the real molecule is a chair.
 *      The measure is orientation-invariant -- embed3d starts from RANDOM 3D
 *      coordinates, so the result lands in an arbitrary orientation and
 *      axis-aligned x/y/z extents would be meaningless. See planarity() below.
 *
 *   2. STALE. Do the same load but switch to a built-in preset before "ready".
 *      The abandoned SMILES molecule must NOT be embedded afterwards: no atom
 *      count mismatch, no surprise recompute on a geometry the user left.
 *
 * DETERMINISM: the boot window is not raced against a warm cache -- every
 * /models/** request is held for MODEL_DELAY_MS by a Playwright route, and the
 * bench asserts demo.ready === false at the instant it clicks Load. If that
 * assert ever fires, the glob stopped matching (the request URLs are logged with
 * --verbose) or the weights stopped being fetched over the network.
 *
 * RASTER ONLY: ray tracing is forced off (it hangs under automation on this
 * machine); the bench never touches window.__rt or demo.rtEnabled.
 *
 * Uses playwright-core against the installed Chrome (channel: "chrome",
 * headless: false) so no browser download is needed. Needs the dev server up:
 *
 *   npm run dev                        # separate shell, serves http://localhost:8142/
 *   npm run bootwindow -w apps/demo
 */
import { chromium } from "playwright-core";

const URL = "http://localhost:8142/";
const VERBOSE = process.argv.includes("--verbose");

/** Cyclohexane: 18 atoms with H. Flat hexagon as a 2D seed, chair when embedded. */
const PROBE_SMILES = "C1CCCCC1";
const PROBE_ATOMS = 18;

/** Preset switched to in the stale case (3 atoms — trivially distinguishable). */
const STALE_PRESET = "water";
const STALE_PRESET_ATOMS = 3;

/**
 * How long each /models/** response is held back. Two requests are delayed
 * (manifest.json, then weights-full-f16.bin) and they are sequential, so this
 * widens the boot window by ~2x this value — plenty of room to type a SMILES,
 * watch it parse, and still be pre-ready.
 */
const MODEL_DELAY_MS = 4000;

/**
 * Minimum thinnest/fattest width ratio for an EMBEDDED cyclohexane.
 *
 * Measured, not guessed — both numbers are printed by this bench on every run:
 *   RDKit 2D seed, i.e. exactly what the pre-fix app left on screen : 0.184
 *   embedded chair                                                  : 0.591
 * The threshold sits between them with real headroom on both sides: 2.0x above
 * the flat seed, 1.6x below the measured chair. embed3d and FIRE are both
 * deterministic (seeded PRNG), so run-to-run drift is nil today; the headroom is
 * there so a future model/force-field change that lands in a slightly different
 * chair — or a twist-boat — still passes, while anything still lying in a plane
 * fails outright.
 */
const MIN_PLANARITY = 0.36;

/** How long to watch for an unwanted recompute in the stale case. */
const STALE_WATCH_MS = 3000;

const READY_TIMEOUT = 180_000;
const SETTLE_TIMEOUT = 180_000;
const PARSE_TIMEOUT = 60_000;

/** Thrown when an assertion about the app's behaviour fails. */
class BenchError extends Error {}

function log(line) {
  process.stdout.write(`${line}\n`);
}

/**
 * Orientation-invariant flatness measure, evaluated page-side on demo.positions.
 *
 * width(d) = max_i (r_i · d) - min_i (r_i · d) is the caliper width of the atom
 * cloud along unit direction d. Sampling d over a Fibonacci hemisphere (widths
 * are symmetric under d -> -d) and returning min/max over the sample gives
 * "thinnest extent / fattest extent" without needing a principal-axis solve and
 * without any dependence on how the molecule happens to be oriented — which
 * matters because embed3d relaxes from random 3D starts. A planar molecule has
 * a near-zero thinnest width and so a small ratio; a chair has a genuinely
 * three-dimensional cloud and a large one.
 */
const planarityFn = (nDirs) => {
  const p = window.__demo.positions;
  const n = p.length / 3;
  if (n < 3) return 1;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += p[3 * i];
    cy += p[3 * i + 1];
    cz += p[3 * i + 2];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  let minW = Infinity;
  let maxW = 0;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < nDirs; k++) {
    // Hemisphere: z in (0, 1] so d and -d are never both sampled.
    const dz = (k + 0.5) / nDirs;
    const r = Math.sqrt(Math.max(0, 1 - dz * dz));
    const th = golden * k;
    const dx = Math.cos(th) * r;
    const dy = Math.sin(th) * r;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const t = (p[3 * i] - cx) * dx + (p[3 * i + 1] - cy) * dy + (p[3 * i + 2] - cz) * dz;
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
    const w = hi - lo;
    if (w < minW) minW = w;
    if (w > maxW) maxW = w;
  }
  return maxW > 0 ? minW / maxW : 1;
};

async function main() {
  log(`\nbootwindow-bench: url=${URL} modelDelay=${MODEL_DELAY_MS} ms minPlanarity=${MIN_PLANARITY}\n`);

  const browser = await chromium.launch({ channel: "chrome", headless: false });
  const pageErrors = [];

  /** Fresh context per case: a cold cache, a cold worker, a cold boot window. */
  async function newPage() {
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

    // THE determinism knob. These fetches come from the optimizer Worker, not the
    // page; page.route does intercept them (verified: 2 hits per boot, manifest
    // then weights). Enabling routing also bypasses the HTTP cache, so a second
    // context in the same run still pays the delay.
    let hits = 0;
    await page.route("**/models/**", async (route) => {
      hits++;
      if (VERBOSE) log(`  · delaying model request: ${route.request().url()}`);
      await new Promise((r) => setTimeout(r, MODEL_DELAY_MS));
      await route.continue();
    });

    page.on("pageerror", (e) => pageErrors.push(String(e && e.stack ? e.stack : e)));
    page.on("console", (m) => {
      if (m.type() === "error") log(`  · console.error: ${m.text()}`);
    });
    // Names any missing asset instead of leaving a bare "404" in the console log.
    // Note the one 404 the other benches show -- Chrome's automatic
    // /favicon.ico fetch, which the app ships no icon for -- is browser-initiated
    // and never surfaces here; only page/worker requests do.
    page.on("response", (r) => {
      if (r.status() >= 400) log(`  · HTTP ${r.status()}: ${r.url()}`);
    });

    return { page, modelHits: () => hits };
  }

  const flags = (page) =>
    page.evaluate(() => {
      const d = window.__demo;
      let atomMeshes = 0;
      const scene = window.__scene;
      if (scene && scene.traverse) {
        scene.traverse((o) => {
          if (o.name === "atom" && o.visible) atomMeshes++;
        });
      }
      return {
        ready: d.ready,
        optimizing: d.optimizing,
        loading: d.loading,
        done: d.done,
        step: d.step,
        molecule: d.molecule,
        atomCount: d.atomCount,
        symbolCount: d.symbols.length,
        atomMeshes,
        error: d.error,
        status: (document.getElementById("status") || {}).textContent || "",
      };
    });

  const waitSettle = (page, timeout) =>
    page.waitForFunction(
      () => {
        const d = window.__demo;
        return d && !d.optimizing && !d.loading && !d.vibComputing && !d.orbComputing && !d.irComputing;
      },
      undefined,
      { timeout },
    );

  /** Type the SMILES and click Load, exactly as a visitor would. */
  const loadSmiles = (page, smi) =>
    page.evaluate((s) => {
      const inp = document.getElementById("smiles");
      inp.value = s;
      document.getElementById("load-smiles").click();
    }, smi);

  /** Switch the preset dropdown (dispatch, so it works while disabled too). */
  const selectPreset = (page, value) =>
    page.evaluate((val) => {
      const sel = document.getElementById("mol");
      sel.value = val;
      sel.dispatchEvent(new Event("change"));
    }, value);

  /** Open the app and stop the moment the app object exists — NOT at ready. */
  async function openInBootWindow(page) {
    await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForFunction(() => !!window.__demo, undefined, { timeout: 30_000 });
    const boot = await flags(page);
    if (boot.ready) {
      throw new BenchError(
        "the model was already ready when the page object appeared — the boot window never opened. " +
          "Check that the page.route(\"**/models/**\") glob still matches the weight requests (run with --verbose).",
      );
    }
    return boot;
  }

  try {
    // =====================================================================
    // Case 1 — a SMILES loaded during the boot window must auto-embed.
    // =====================================================================
    log("case 1: load a SMILES before the model is ready");
    const c1 = await newPage();
    await openInBootWindow(c1.page);
    log("  page up, demo.ready=false");

    await loadSmiles(c1.page, PROBE_SMILES);
    // The click itself must land inside the window: assert before the RDKit
    // parse can finish, so this is the state loadCustomMolecule actually saw.
    const atClick = await flags(c1.page);
    if (atClick.ready) throw new BenchError("demo.ready was already true at the moment Load was clicked");
    log(`  clicked Load "${PROBE_SMILES}" with demo.ready=${atClick.ready}`);

    // Parse done: geometry installed, still no model.
    await c1.page.waitForFunction(
      (n) => window.__demo && !window.__demo.loading && window.__demo.atomCount === n,
      PROBE_ATOMS,
      { timeout: PARSE_TIMEOUT },
    );
    const seeded = await flags(c1.page);
    const seedPlanarity = await c1.page.evaluate(planarityFn, 2000);
    log(
      `  seeded: molecule=${seeded.molecule} atoms=${seeded.atomCount} ready=${seeded.ready} ` +
        `planarity=${seedPlanarity.toFixed(3)} status="${seeded.status}"`,
    );
    if (seeded.ready) throw new BenchError("the model became ready during the SMILES parse — window too narrow");

    log("  waiting for the model…");
    await c1.page.waitForFunction(() => window.__demo && window.__demo.ready === true, undefined, {
      timeout: READY_TIMEOUT,
    });
    await waitSettle(c1.page, SETTLE_TIMEOUT);
    const after = await flags(c1.page);
    const planarity = await c1.page.evaluate(planarityFn, 2000);
    log(
      `  after ready+settle: done=${after.done} step=${after.step} molecule=${after.molecule} ` +
        `atoms=${after.atomCount} planarity=${planarity.toFixed(3)} status="${after.status}"`,
    );
    log(`  (model requests delayed: ${c1.modelHits()})`);

    if (after.error) throw new BenchError(`demo.error set: ${after.error}`);
    if (after.molecule !== `smiles:${PROBE_SMILES}`) {
      throw new BenchError(`molecule changed unexpectedly: ${after.molecule}`);
    }
    if (!after.done) {
      throw new BenchError(
        `no embed ever ran for a SMILES loaded during the boot window (demo.done=false, step=${after.step}). ` +
          "The molecule is stuck at RDKit's 2D seed — main.ts must carry the pending embed into the \"ready\" handler.",
      );
    }
    if (!(planarity >= MIN_PLANARITY)) {
      throw new BenchError(
        `cyclohexane is still flat: planarity ${planarity.toFixed(3)} < ${MIN_PLANARITY} ` +
          `(a chair measures ~0.56, RDKit's 2D seed ~0.21). demo.done was true, so something ran — ` +
          "but it did not produce a three-dimensional geometry.",
      );
    }
    log(`  PASS case 1: auto-embedded, planarity ${planarity.toFixed(3)} >= ${MIN_PLANARITY}\n`);
    await c1.page.context().close();

    // =====================================================================
    // Case 2 — abandon the SMILES before ready; nothing must embed it later.
    // =====================================================================
    log("case 2: abandon the SMILES (switch preset) before the model is ready");
    const c2 = await newPage();
    await openInBootWindow(c2.page);
    await loadSmiles(c2.page, PROBE_SMILES);
    const atClick2 = await flags(c2.page);
    if (atClick2.ready) throw new BenchError("demo.ready was already true at the moment Load was clicked (case 2)");
    await c2.page.waitForFunction(
      (n) => window.__demo && !window.__demo.loading && window.__demo.atomCount === n,
      PROBE_ATOMS,
      { timeout: PARSE_TIMEOUT },
    );
    log(`  seeded ${PROBE_ATOMS}-atom SMILES with demo.ready=false`);

    await selectPreset(c2.page, STALE_PRESET);
    const switched = await flags(c2.page);
    log(`  switched to preset "${STALE_PRESET}": molecule=${switched.molecule} atoms=${switched.atomCount}`);
    if (switched.ready) throw new BenchError("the model became ready before the preset switch — window too narrow");

    await c2.page.waitForFunction(() => window.__demo && window.__demo.ready === true, undefined, {
      timeout: READY_TIMEOUT,
    });
    // Watch for a stale embed waking up: sample continuously rather than once, so
    // a recompute that starts AND finishes inside the window is still caught.
    const watch = await c2.page.evaluate(async (ms) => {
      const d = window.__demo;
      let sawCompute = false;
      let sawWrongAtoms = false;
      const t0 = Date.now();
      while (Date.now() - t0 < ms) {
        if (d.optimizing || d.loading) sawCompute = true;
        if (d.atomCount !== d.symbols.length) sawWrongAtoms = true;
        await new Promise((r) => setTimeout(r, 25));
      }
      return { sawCompute, sawWrongAtoms, done: d.done, molecule: d.molecule, atomCount: d.atomCount };
    }, STALE_WATCH_MS);
    const end2 = await flags(c2.page);
    log(
      `  after ready + ${STALE_WATCH_MS} ms watch: recompute=${watch.sawCompute} molecule=${watch.molecule} ` +
        `atoms=${watch.atomCount} meshes=${end2.atomMeshes} done=${watch.done} status="${end2.status}"`,
    );

    if (watch.sawCompute) {
      throw new BenchError(
        "a worker op started after \"ready\" for a molecule the user had already left — the pending embed " +
          "was not discarded when the geometry changed (main.ts resetForNewGeometry / the epoch check).",
      );
    }
    if (watch.molecule !== STALE_PRESET || watch.atomCount !== STALE_PRESET_ATOMS) {
      throw new BenchError(`the abandoned SMILES came back: molecule=${watch.molecule} atoms=${watch.atomCount}`);
    }
    if (watch.sawWrongAtoms || end2.atomMeshes !== STALE_PRESET_ATOMS || end2.symbolCount !== STALE_PRESET_ATOMS) {
      throw new BenchError(
        `atom-count mismatch: symbols=${end2.symbolCount} meshes=${end2.atomMeshes} demo.atomCount=${watch.atomCount}`,
      );
    }
    if (end2.error) throw new BenchError(`demo.error set after abandoning a pre-ready SMILES: ${end2.error}`);
    log("  PASS case 2: abandoned SMILES was discarded, not embedded\n");
    await c2.page.context().close();

    if (pageErrors.length) throw new BenchError(`page error(s):\n${pageErrors.join("\n---\n")}`);

    await browser.close();
    log("PASS: a SMILES loaded during the model download embeds itself when the model lands; an abandoned one does not.\n");
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
  process.stderr.write(`bootwindow harness crashed: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(2);
});
