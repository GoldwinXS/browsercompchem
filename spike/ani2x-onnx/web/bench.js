import { makeAEV } from "./aev.js";

const ort = window.ort;
const log = (s) => {
  const el = document.getElementById("log");
  el.textContent += s + "\n";
  console.log(s);
};
const setStatus = (s) => (document.getElementById("status").textContent = s);

let PARAMS, MOLS, AEV, SELF_E;
const RESULTS = { env: {}, latency: [], optimization: null, model: {} };
window.RESULTS = RESULTS;

async function loadAll() {
  PARAMS = await (await fetch("../params.json")).json();
  MOLS = await (await fetch("../molecules.json")).json();
  AEV = makeAEV(PARAMS);
  SELF_E = Float64Array.from(PARAMS.self_energies);
  const head = await fetch("../ani2x_nn_energy.onnx", { method: "HEAD" });
  RESULTS.model.bytes = Number(head.headers.get("content-length")) || null;
}

function prep(mol) {
  const znums = mol.znums;
  const N = znums.length;
  const coords = new Float64Array(N * 3);
  for (let i = 0; i < N; i++) {
    coords[i * 3] = mol.coords[i][0];
    coords[i * 3 + 1] = mol.coords[i][1];
    coords[i * 3 + 2] = mol.coords[i][2];
  }
  const sp = AEV.speciesIdx(znums);
  const mask = new Float32Array(N * 7);
  let selfE = 0;
  for (let i = 0; i < N; i++) { mask[i * 7 + sp[i]] = 1; selfE += SELF_E[sp[i]]; }
  return { znums, N, coords, sp, mask, selfE, aevBuf: new Float32Array(N * AEV.AEV_DIM) };
}

async function energy(session, st, coords) {
  AEV.computeAEV(st.znums, coords, st.sp, st.aevBuf);
  const aevT = new ort.Tensor("float32", st.aevBuf, [st.N, AEV.AEV_DIM]);
  const maskT = new ort.Tensor("float32", st.mask, [st.N, 7]);
  const out = await session.run({ aev: aevT, mask: maskT });
  const nnE = out.nn_energy.data[0];
  return nnE + st.selfE;
}

async function fdForces(session, st, coords, h = 1e-3) {
  const N = st.N;
  const F = new Float64Array(N * 3);
  const cp = Float64Array.from(coords);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < 3; d++) {
      const k = i * 3 + d;
      const orig = cp[k];
      cp[k] = orig + h; const ep = await energy(session, st, cp);
      cp[k] = orig - h; const em = await energy(session, st, cp);
      cp[k] = orig;
      F[k] = -(ep - em) / (2 * h);
    }
  }
  return F;
}

function median(a) {
  const b = Float64Array.from(a).sort();
  const n = b.length;
  return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]);
}

async function makeSession(ep) {
  const opts = { executionProviders: [ep], graphOptimizationLevel: "all" };
  return await ort.InferenceSession.create("../ani2x_nn_energy.onnx", opts);
}

async function benchBackend(ep, molNames) {
  let session;
  try {
    session = await makeSession(ep);
  } catch (e) {
    log(`[${ep}] session create FAILED: ${e}`);
    return;
  }
  for (const name of molNames) {
    const mol = MOLS[name];
    const st = prep(mol);
    // sanity energy
    const e0 = await energy(session, st, st.coords);
    // warmup 10
    for (let i = 0; i < 10; i++) await energy(session, st, st.coords);
    // time >=50 energy calls
    const NCALL = 60;
    const t = new Float64Array(NCALL);
    for (let i = 0; i < NCALL; i++) {
      const t0 = performance.now();
      await energy(session, st, st.coords);
      t[i] = performance.now() - t0;
    }
    const medE = median(t);
    // aev-only timing (JS featurizer cost) over 20
    const ta = new Float64Array(20);
    for (let i = 0; i < 20; i++) {
      const t0 = performance.now();
      AEV.computeAEV(st.znums, st.coords, st.sp, st.aevBuf);
      ta[i] = performance.now() - t0;
    }
    const medAEV = median(ta);
    const perForce = medE * (6 * st.N); // FD: 6N energy calls per force eval
    // directly measured FD force-eval time (median of 3) for small/mid molecules
    let measForce = null;
    if (st.N <= 24) {
      const tf = [];
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        await fdForces(session, st, st.coords);
        tf.push(performance.now() - t0);
      }
      measForce = +median(tf).toFixed(2);
    }
    RESULTS.latency.push({
      backend: ep, molecule: name, atoms: st.N,
      energy_ms: +medE.toFixed(4), aev_ms: +medAEV.toFixed(4),
      force_eval_derived_ms: +perForce.toFixed(2),
      force_eval_measured_ms: measForce, energy_Ha: +e0.toFixed(5),
    });
    log(`[${ep}] ${name} N=${st.N}: E=${e0.toFixed(5)} Ha | energy ${medE.toFixed(3)} ms `
      + `(aev ${medAEV.toFixed(3)} ms) | FD force-eval ~${perForce.toFixed(1)} ms`);
  }
  await session.release?.();
}

async function optimize(ep) {
  const session = await makeSession(ep);
  const mol = MOLS["aspirin"];
  const st = prep(mol);
  // perturb geometry
  const coords = Float64Array.from(st.coords);
  let seed = 42;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  for (let i = 0; i < coords.length; i++) coords[i] += 0.15 * rnd();

  const E = async (c) => energy(session, st, c);
  const e_start = await E(coords);
  const lr = 0.02;
  let maxF = Infinity, steps = 0, e_last = e_start;
  const t0 = performance.now();
  const MAXSTEP = 150;
  for (steps = 0; steps < MAXSTEP; steps++) {
    const F = await fdForces(session, st, coords);
    maxF = 0;
    for (let k = 0; k < F.length; k++) maxF = Math.max(maxF, Math.abs(F[k]));
    if (maxF < 0.02) break; // Ha/Angstrom
    for (let k = 0; k < coords.length; k++) coords[k] += lr * F[k]; // step downhill
    e_last = await E(coords);
  }
  const wall = performance.now() - t0;
  const e_final = await E(coords);
  RESULTS.optimization = {
    backend: ep, molecule: "aspirin", atoms: st.N,
    steps, wall_ms: +wall.toFixed(1),
    e_start: +e_start.toFixed(5), e_final: +e_final.toFixed(5),
    dE_Ha: +(e_final - e_start).toFixed(5), final_maxF: +maxF.toFixed(4),
    force_evals: steps, energy_calls_approx: steps * (6 * st.N + 1),
  };
  log(`[opt ${ep}] aspirin: ${steps} steps, ${wall.toFixed(0)} ms, `
    + `E ${e_start.toFixed(4)} -> ${e_final.toFixed(4)} Ha, maxF ${maxF.toFixed(4)}`);
  await session.release?.();
}

async function probeWebGPU() {
  if (!navigator.gpu) return false;
  try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function main() {
  setStatus("loading model + data...");
  await loadAll();
  const gpuOk = await probeWebGPU();
  RESULTS.env = {
    webgpu_navigator: !!navigator.gpu,
    webgpu_adapter: gpuOk,
    threads: ort.env.wasm.numThreads,
    ua: navigator.userAgent,
    ort: ort.env.versions ? ort.env.versions.common : "?",
  };
  log("webgpu adapter available: " + gpuOk + " | wasm threads: " + ort.env.wasm.numThreads);
  log("model bytes: " + RESULTS.model.bytes);

  const molNames = ["water", "methane", "aspirin", "caffeine", "cholesterol", "peptide"];
  const backends = [];
  if (gpuOk) backends.push("webgpu");
  backends.push("wasm");

  for (const ep of backends) {
    setStatus(`benchmarking ${ep}...`);
    log(`\n=== backend: ${ep} ===`);
    await benchBackend(ep, molNames);
  }

  setStatus("running geometry optimization (aspirin)...");
  log(`\n=== geometry optimization ===`);
  const optEp = gpuOk ? "webgpu" : "wasm";
  await optimize(optEp);

  setStatus("DONE");
  log("\n=== DONE ===");
  window.BENCH_DONE = true;
  document.getElementById("done").textContent = "DONE";
}

main().catch((e) => { log("FATAL: " + e + "\n" + (e.stack || "")); setStatus("ERROR"); });
