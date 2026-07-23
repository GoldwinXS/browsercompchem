/**
 * Permanent state-fuzz bench for the demo app.
 *
 * Drives the live dev server (http://localhost:8142/) with a SEEDED
 * pseudo-random sequence of real UI actions -- switching presets, loading
 * SMILES, perturbing, optimizing, computing vibrations/orbitals/IR spectra,
 * selecting modes/orbitals, changing the isovalue, hiding orbitals, clearing
 * the selection, and clicking atoms to measure -- WITHOUT ever refreshing. After
 * every action it asserts the state-management invariants that the "state not
 * reset" bugs violated:
 *
 *   - demo.symbols.length === the visible atom-mesh count in window.__scene
 *     (a stale worker reply repainting the old molecule breaks this);
 *   - demo.positions contains no NaN/Inf (a stale geometry clobber injects NaN);
 *   - demo.optimizing / vibComputing / orbComputing / irComputing all return to
 *     false, and the relevant buttons re-enable, once every started compute
 *     settles (a stuck flag or a leaked lock breaks this);
 *   - zero uncaught page errors.
 *
 * To actually exercise the races, a fraction of the compute-starting actions do
 * NOT wait for completion -- the next action fires while the worker is still
 * busy. The action picker only ever chooses controls that are currently ENABLED,
 * so the fuzz does exactly what a real user could do through the UI.
 *
 * RASTER ONLY: ray tracing is forced off (it hangs under automation on this
 * machine); the bench never touches window.__rt or demo.rtEnabled.
 *
 * Uses playwright-core against the installed Chrome (channel: "chrome",
 * headless: false) so no browser download is needed.
 *
 *   npm run fuzz -w apps/demo            # default seed 12345, 40 actions
 *   npm run fuzz -w apps/demo -- 777     # seed 777
 *   npm run fuzz -w apps/demo -- 777 60  # seed 777, 60 actions
 */
import { chromium } from "playwright-core";

const URL = "http://localhost:8142/";
const SEED = (Number(process.argv[2] ?? 12345) >>> 0) || 12345;
const N_ACTIONS = Number(process.argv[3] ?? 40);

// water, methane, ethylene, and one ~20-atom molecule (naphthalene, 18 atoms).
const SMILES_POOL = ["O", "C", "C=C", "c1ccc2ccccc2c1"];

// Compute-starting actions that may be launched without waiting for completion,
// so the following action interleaves with the still-running worker op.
const INTERLEAVE_KINDS = new Set(["optimize", "orbitals", "smiles", "selOrb"]);
const INTERLEAVE_PROB = 0.4;

// Ceiling only: waitForSettle returns the instant the app settles. It must
// cover the SLOWEST op that could still be running -- including an interleaved
// embed of the ~20-atom molecule launched by the previous action.
const SETTLE_TIMEOUT = 180_000;
const READY_TIMEOUT = 180_000;

/** mulberry32 — same PRNG the app uses, for reproducible runs. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(SEED);
const randInt = (n) => Math.floor(rng() * n);
const pick = (arr) => arr[randInt(arr.length)];

const actionLog = [];
function record(line) {
  actionLog.push(line);
  process.stdout.write(`  [${String(actionLog.length).padStart(2, "0")}] ${line}\n`);
}

/** Thrown on an invariant violation; carries no extra machinery, just a message. */
class InvariantError extends Error {}

async function main() {
  process.stdout.write(`\nstate-fuzz: seed=${SEED} actions=${N_ACTIONS} url=${URL}\n\n`);

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
    if (m.type() === "error") process.stdout.write(`  · console.error: ${m.text()}\n`);
  });

  // --- page-side helpers -------------------------------------------------
  const snap = () =>
    page.evaluate(() => {
      const d = window.__demo;
      const dis = (id) => {
        const e = document.getElementById(id);
        return e ? !!e.disabled : true;
      };
      let atomMeshes = 0;
      const scene = window.__scene;
      if (scene && scene.traverse) {
        scene.traverse((o) => {
          if (o.name === "atom" && o.visible) atomMeshes++;
        });
      }
      const molEl = document.getElementById("mol");
      return {
        webgl: d.webgl,
        ready: d.ready,
        optimizing: d.optimizing,
        loading: d.loading,
        vibComputing: d.vibComputing,
        orbComputing: d.orbComputing,
        irComputing: d.irComputing,
        vibReady: d.vibReady,
        orbReady: d.orbReady,
        irReady: d.irReady,
        orbSelected: d.orbSelected,
        atomCount: d.atomCount,
        symbolsLen: d.symbols.length,
        positions: d.positions,
        selected: d.selected.slice(),
        coverageOK: d.coverageOK,
        rtEnabled: d.rtEnabled,
        molecule: d.molecule,
        vibFreqLen: d.vibFrequencies.length,
        orbEnergiesLen: d.orbEnergies.length,
        irModesLen: d.irModes.length,
        atomMeshes,
        dis: {
          mol: dis("mol"),
          load: dis("load-smiles"),
          draw: dis("open-sketch"),
          perturb: dis("perturb"),
          optimize: dis("optimize"),
          vibrations: dis("vibrations"),
          orbitals: dis("orbitals"),
          ir: dis("ir-compute"),
        },
        molOptions: molEl ? Array.from(molEl.options).map((o) => o.value) : [],
        molValue: molEl ? molEl.value : "",
      };
    });

  const settled = (s) => !s.optimizing && !s.loading && !s.vibComputing && !s.orbComputing && !s.irComputing;

  const waitForSettle = (timeout) =>
    // NB: waitForFunction(fn, arg, options) is positional -- options MUST be the
    // third argument, so pass `undefined` as arg or the timeout is ignored.
    page.waitForFunction(
      () => {
        const d = window.__demo;
        return d && !d.optimizing && !d.loading && !d.vibComputing && !d.orbComputing && !d.irComputing;
      },
      undefined,
      { timeout },
    );

  // --- assertions --------------------------------------------------------
  function cheapCheck(s, when) {
    if (pageErrors.length) {
      throw new InvariantError(`page error(s) after ${when}:\n${pageErrors.join("\n---\n")}`);
    }
    if (s.rtEnabled) throw new InvariantError(`ray tracing became enabled after ${when} (must stay raster)`);
    if (s.symbolsLen !== s.atomMeshes) {
      throw new InvariantError(
        `atom-mesh mismatch after ${when}: demo.symbols=${s.symbolsLen} but ${s.atomMeshes} visible "atom" meshes in __scene`,
      );
    }
    if (s.atomCount !== s.symbolsLen) {
      throw new InvariantError(`demo.atomCount=${s.atomCount} != demo.symbols.length=${s.symbolsLen} after ${when}`);
    }
    if (s.positions.length !== 3 * s.symbolsLen) {
      throw new InvariantError(
        `demo.positions length ${s.positions.length} != 3*${s.symbolsLen} after ${when}`,
      );
    }
    for (let i = 0; i < s.positions.length; i++) {
      if (!Number.isFinite(s.positions[i])) {
        throw new InvariantError(`non-finite position[${i}]=${s.positions[i]} after ${when}`);
      }
    }
    for (const idx of s.selected) {
      if (!(idx >= 0 && idx < s.atomCount)) {
        throw new InvariantError(`selected atom index ${idx} out of range [0,${s.atomCount}) after ${when}`);
      }
    }
  }

  function settleCheck(s, when) {
    if (!settled(s)) throw new InvariantError(`flags not settled after ${when}: ${JSON.stringify(pickFlags(s))}`);
    const canRelax = s.ready && s.coverageOK;
    const expectEnabled = [];
    if (canRelax && s.dis.optimize) expectEnabled.push("optimize");
    if (canRelax && s.dis.perturb) expectEnabled.push("perturb");
    if (s.coverageOK && s.atomCount >= 1 && s.dis.orbitals) expectEnabled.push("orbitals");
    if (canRelax && s.atomCount >= 2 && s.dis.vibrations) expectEnabled.push("vibrations");
    if (canRelax && s.atomCount >= 2 && s.dis.ir) expectEnabled.push("ir");
    if (s.dis.mol) expectEnabled.push("mol");
    if (s.dis.load) expectEnabled.push("load-smiles");
    if (s.dis.draw) expectEnabled.push("draw");
    if (expectEnabled.length) {
      throw new InvariantError(
        `controls stuck disabled at idle after ${when}: ${expectEnabled.join(", ")} (ready=${s.ready} coverageOK=${s.coverageOK} atomCount=${s.atomCount})`,
      );
    }
  }

  const pickFlags = (s) => ({
    optimizing: s.optimizing,
    loading: s.loading,
    vibComputing: s.vibComputing,
    orbComputing: s.orbComputing,
    irComputing: s.irComputing,
  });

  // --- action performers -------------------------------------------------
  const clickId = (id) => page.evaluate((i) => document.getElementById(i).click(), id);

  async function perform(act) {
    switch (act.kind) {
      case "preset": {
        const opts = act.options.filter((v) => v !== act.current);
        const target = opts.length ? pick(opts) : act.current;
        record(`preset -> ${target}`);
        await page.evaluate((val) => {
          const sel = document.getElementById("mol");
          sel.value = val;
          sel.dispatchEvent(new Event("change"));
        }, target);
        return { compute: false };
      }
      case "smiles": {
        const smi = pick(SMILES_POOL);
        record(`load SMILES "${smi}"`);
        await page.evaluate((s) => {
          const inp = document.getElementById("smiles");
          inp.value = s;
          document.getElementById("load-smiles").click();
        }, smi);
        return { compute: true };
      }
      case "perturb":
        record("perturb");
        await clickId("perturb");
        return { compute: false };
      case "optimize":
        record("optimize");
        await clickId("optimize");
        return { compute: true };
      case "vibrations":
        record(`compute vibrations (${act.atomCount} atoms)`);
        await clickId("vibrations");
        return { compute: true };
      case "ir":
        record(`compute IR spectrum (${act.atomCount} atoms)`);
        await clickId("ir-compute");
        return { compute: true };
      case "orbitals":
        record("compute orbitals");
        await clickId("orbitals");
        return { compute: true };
      case "selVib": {
        const i = randInt(act.nModes);
        record(`select vib mode ${i}/${act.nModes}`);
        await page.evaluate((k) => window.__vib.selectMode(k), i);
        return { compute: false };
      }
      case "selOrb": {
        const i = randInt(act.nMO);
        record(`select orbital ${i}/${act.nMO}`);
        await page.evaluate((k) => window.__orb.select(k), i);
        return { compute: true };
      }
      case "iso": {
        const v = (0.02 + rng() * 0.16).toFixed(3);
        record(`change isovalue -> ${v}`);
        await page.evaluate((val) => {
          const el = document.getElementById("orb-iso");
          el.value = String(val);
          el.dispatchEvent(new Event("input"));
          el.dispatchEvent(new Event("change"));
        }, v);
        return { compute: true };
      }
      case "hideOrb":
        record("hide orbitals");
        await page.evaluate(() => window.__orb.hide());
        return { compute: false };
      case "clickAtom": {
        const i = randInt(act.atomCount);
        record(`click atom ${i}`);
        await page.evaluate((k) => window.__measure.toggle(k), i);
        return { compute: false };
      }
      case "clear":
        record("clear selection");
        await page.evaluate(() => window.__measure.clear());
        return { compute: false };
      default:
        throw new Error(`unknown action ${act.kind}`);
    }
  }

  function validActions(s) {
    const acts = [];
    if (!s.dis.mol) acts.push({ kind: "preset", options: s.molOptions, current: s.molValue });
    if (!s.dis.load) acts.push({ kind: "smiles" });
    if (!s.dis.perturb) acts.push({ kind: "perturb" });
    if (!s.dis.optimize) acts.push({ kind: "optimize" });
    // Cap Hessian size so the bench stays fast (a real user could go bigger).
    if (!s.dis.vibrations && s.atomCount <= 15) acts.push({ kind: "vibrations", atomCount: s.atomCount });
    if (!s.dis.ir && s.atomCount <= 15) acts.push({ kind: "ir", atomCount: s.atomCount });
    if (!s.dis.orbitals) acts.push({ kind: "orbitals" });
    if (s.vibReady && s.vibFreqLen > 0) acts.push({ kind: "selVib", nModes: s.vibFreqLen });
    if (s.orbReady && !s.orbComputing && s.orbEnergiesLen > 0) acts.push({ kind: "selOrb", nMO: s.orbEnergiesLen });
    if (s.orbSelected !== null && !s.orbComputing) acts.push({ kind: "iso" });
    if (s.orbReady && settled(s)) acts.push({ kind: "hideOrb" });
    if (s.atomCount > 0) acts.push({ kind: "clickAtom", atomCount: s.atomCount });
    acts.push({ kind: "clear" });
    return acts;
  }

  // --- run ---------------------------------------------------------------
  try {
    await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    process.stdout.write("waiting for model ready…\n");
    await page.waitForFunction(() => window.__demo && window.__demo.ready === true, undefined, {
      timeout: READY_TIMEOUT,
    });
    const boot = await snap();
    if (!boot.webgl) throw new InvariantError("WebGL2 unavailable — the atom-mesh invariant needs a real GL context");
    process.stdout.write(`model ready. boot molecule=${boot.molecule} atoms=${boot.atomCount}\n\n`);
    cheapCheck(boot, "boot");
    settleCheck(boot, "boot");

    let interleavedPrev = false;
    for (let n = 0; n < N_ACTIONS; n++) {
      const s = await snap();
      const acts = validActions(s);
      const act = pick(acts);
      // Interleave only from a settled state (so we launch a fresh op and let the
      // NEXT action fire while it runs); never stack two skipped settles.
      const interleave =
        !interleavedPrev && settled(s) && INTERLEAVE_KINDS.has(act.kind) && rng() < INTERLEAVE_PROB;

      const res = await perform(act);

      if (res.compute && interleave) {
        interleavedPrev = true;
        await page.waitForTimeout(60); // let the worker stream a little
        cheapCheck(await snap(), `${act.kind} (interleaved, mid-compute)`);
      } else {
        interleavedPrev = false;
        // Waits out this action AND any still-running op an earlier interleave left.
        await waitForSettle(SETTLE_TIMEOUT);
        const s2 = await snap();
        cheapCheck(s2, act.kind);
        settleCheck(s2, act.kind);
      }
    }

    // Final drain: whatever was interleaved must settle and leave a usable UI.
    await waitForSettle(READY_TIMEOUT);
    const end = await snap();
    cheapCheck(end, "final settle");
    settleCheck(end, "final settle");

    process.stdout.write(`\nPASS seed=${SEED}: ${N_ACTIONS} actions, ${actionLog.length} performed, 0 invariant violations, 0 page errors.\n`);
    await browser.close();
    process.exit(0);
  } catch (err) {
    process.stdout.write(`\nFAIL seed=${SEED} after ${actionLog.length} actions\n`);
    process.stdout.write(`reason: ${err instanceof Error ? err.message : String(err)}\n`);
    process.stdout.write(`\nreproduce: npm run fuzz -w apps/demo -- ${SEED} ${N_ACTIONS}\n`);
    process.stdout.write(`\naction log:\n${actionLog.map((l, i) => `  [${String(i + 1).padStart(2, "0")}] ${l}`).join("\n")}\n`);
    if (pageErrors.length) process.stdout.write(`\npage errors:\n${pageErrors.join("\n---\n")}\n`);
    try {
      await browser.close();
    } catch {
      /* ignore */
    }
    process.exit(1);
  }
}

main().catch((err) => {
  process.stderr.write(`fuzz harness crashed: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(2);
});
