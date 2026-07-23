import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { MoleculeView } from "./scene.js";
import { MOLECULES, MOLECULE_ORDER, DEFAULT_MOLECULE } from "./molecules.js";

/**
 * Interactive geometry-optimization demo.
 *
 * Main thread: renders the ball-and-stick molecule (three-realtime-rt) and
 * drives the UI. A Web Worker (optimizer.worker.ts) runs the ANI-2x provider +
 * FIRE optimizer and streams per-step {energy, maxForce, positions} messages;
 * we just paint them. The heavy numerics never touch this thread, so camera
 * orbit / UI stay responsive throughout a relaxation.
 *
 * Change the model variant here. "full-f16" = 8-member f16 ensemble (~26 MB,
 * chemically excellent, aspirin relaxes in ~2 s). Alternatives in
 * models/ani2x/manifest.json: full-f32, single-f16, single-f32.
 */
const MODEL_VARIANT = "full-f16";
const MODEL_DIR = `${location.origin}/models/ani2x`;
const FIRE_OPTIONS = { forceTolerance: 0.02, maxSteps: 400, dtMax: 0.3 };
const PERTURB_MAX = 0.2; // Angstrom, max per-component displacement seed

// ---------------------------------------------------------------------------
// Headless-verifiable state (read by the browser-preview harness).
// ---------------------------------------------------------------------------
interface DemoState {
  webgl: boolean;
  ready: boolean;
  optimizing: boolean;
  molecule: string;
  frameCount: number; // rAF ticks — proves the main thread keeps running
  step: number;
  energies: number[]; // energy curve received from worker
  lastEnergy: number | null;
  lastMaxForce: number | null;
  done: boolean;
  converged: boolean | null;
  finalEnergy: number | null;
  finalSteps: number | null;
  elapsedMs: number | null;
  error: string | null;
}
const demo: DemoState = {
  webgl: false,
  ready: false,
  optimizing: false,
  molecule: DEFAULT_MOLECULE,
  frameCount: 0,
  step: 0,
  energies: [],
  lastEnergy: null,
  lastMaxForce: null,
  done: false,
  converged: null,
  finalEnergy: null,
  finalSteps: null,
  elapsedMs: null,
  error: null,
};
(window as unknown as { __demo: DemoState }).__demo = demo;

// ---------------------------------------------------------------------------
// Seeded RNG (mulberry32) for reproducible perturbations.
// ---------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// DOM handles.
// ---------------------------------------------------------------------------
const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};
const molSelect = $<HTMLSelectElement>("mol");
const btnPerturb = $<HTMLButtonElement>("perturb");
const btnOptimize = $<HTMLButtonElement>("optimize");
const rEnergy = $("r-energy");
const rForce = $("r-force");
const rStep = $("r-step");
const rElapsed = $("r-elapsed");
const statusEl = $("status");
const plotEl = $("plot"); // SVG
const noglEl = $("nogl");

for (const key of MOLECULE_ORDER) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = key;
  molSelect.appendChild(opt);
}
molSelect.value = DEFAULT_MOLECULE;

function setStatus(text: string, cls: "" | "good" | "warn" = ""): void {
  statusEl.textContent = text;
  statusEl.className = cls;
}

// ---------------------------------------------------------------------------
// Energy plot (self-drawn SVG polyline, no libs).
// ---------------------------------------------------------------------------
const PLOT_W = 300;
const PLOT_H = 96;
function drawPlot(energies: number[]): void {
  if (energies.length < 2) {
    plotEl.innerHTML = "";
    return;
  }
  let min = Infinity;
  let max = -Infinity;
  for (const e of energies) {
    if (e < min) min = e;
    if (e > max) max = e;
  }
  const span = max - min || 1;
  const n = energies.length;
  const pad = 4;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = pad + (i / (n - 1)) * (PLOT_W - 2 * pad);
    // higher energy -> nearer the top; descend as it relaxes
    const y = pad + ((energies[i]! - min) / span) * (PLOT_H - 2 * pad);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  plotEl.innerHTML =
    `<polyline fill="none" stroke="#5b8cff" stroke-width="1.5" ` +
    `points="${pts.join(" ")}" />` +
    `<circle cx="${pts[pts.length - 1]!.split(",")[0]}" cy="${
      pts[pts.length - 1]!.split(",")[1]
    }" r="2.5" fill="#46d18a" />`;
}

// ---------------------------------------------------------------------------
// Three.js scene (guarded: degrade gracefully with no WebGL2).
// ---------------------------------------------------------------------------
let renderer: THREE.WebGLRenderer | undefined;
let rt: { render: (s: THREE.Scene, c: THREE.PerspectiveCamera) => void; compileScene: (s: THREE.Scene) => void; setSize: (w: number, h: number) => void } | undefined;
let controls: OrbitControls | undefined;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f16);
const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.1,
  500,
);

function initWebGL(): boolean {
  const probe = document.createElement("canvas");
  const gl = probe.getContext("webgl2");
  if (!gl) return false;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    $("app").appendChild(renderer.domElement);

    const key = new THREE.PointLight(0xffffff, 60);
    key.position.set(6, 8, 6);
    scene.add(key);
    const fill = new THREE.PointLight(0x88aaff, 20);
    fill.position.set(-8, 4, -4);
    scene.add(fill);
    scene.add(new THREE.AmbientLight(0x223044, 1.2));

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true; // keep it alive
    controls.autoRotateSpeed = 1.2;

    // Ray tracer is optional; wire it lazily so a failure here can't kill UI.
    return true;
  } catch (err) {
    demo.error = `webgl init: ${err instanceof Error ? err.message : String(err)}`;
    return false;
  }
}

async function initRaytracer(): Promise<void> {
  if (!renderer) return;
  try {
    const { RealtimeRaytracer } = await import("three-realtime-rt");
    const r = new RealtimeRaytracer(renderer);
    r.compileScene(scene);
    rt = r as unknown as typeof rt;
  } catch (err) {
    // Fall back to the plain WebGL rasterizer; not a fatal error.
    demo.error = `raytracer: ${err instanceof Error ? err.message : String(err)}`;
  }
}

// ---------------------------------------------------------------------------
// Molecule state.
// ---------------------------------------------------------------------------
let view: MoleculeView | undefined;
let symbols: string[] = [];
let basePositions = new Float64Array(0); // equilibrium geometry
let currentPositions: Float32Array<ArrayBufferLike> = new Float32Array(0); // live geometry

function loadMolecule(key: string): void {
  const data = MOLECULES[key];
  if (!data) throw new Error(`unknown molecule ${key}`);
  demo.molecule = key;
  symbols = data.symbols;
  basePositions = Float64Array.from(data.positions);
  currentPositions = Float32Array.from(data.positions);

  if (view && renderer) view.dispose(scene);
  if (renderer) {
    view = new MoleculeView(symbols, currentPositions);
    scene.add(view.group);
    const c = view.centroid(currentPositions);
    if (controls) controls.target.copy(c);
    // frame: back off along +z/+y proportional to spread
    let maxR = 0;
    for (let i = 0; i < symbols.length; i++) {
      const dx = currentPositions[3 * i]! - c.x;
      const dy = currentPositions[3 * i + 1]! - c.y;
      const dz = currentPositions[3 * i + 2]! - c.z;
      maxR = Math.max(maxR, Math.hypot(dx, dy, dz));
    }
    const dist = Math.max(6, maxR * 2.6);
    camera.position.set(c.x + dist * 0.5, c.y + dist * 0.4, c.z + dist);
    camera.lookAt(c);
    if (rt) rt.compileScene(scene);
  }
  resetReadout();
}

function resetReadout(): void {
  demo.step = 0;
  demo.energies = [];
  demo.lastEnergy = null;
  demo.lastMaxForce = null;
  demo.done = false;
  demo.converged = null;
  rEnergy.textContent = "—";
  rForce.textContent = "—";
  rStep.textContent = "—";
  rElapsed.textContent = "—";
  drawPlot([]);
}

function perturb(): void {
  const rng = mulberry32(0xc0ffee ^ symbols.length);
  const p = Float32Array.from(basePositions);
  for (let k = 0; k < p.length; k++) {
    p[k] = p[k]! + (rng() * 2 - 1) * PERTURB_MAX;
  }
  currentPositions = p;
  if (view) view.update(currentPositions);
  resetReadout();
  setStatus("perturbed — press Optimize to relax", "warn");
}

// ---------------------------------------------------------------------------
// Worker wiring.
// ---------------------------------------------------------------------------
const worker = new Worker(new URL("./optimizer.worker.ts", import.meta.url), {
  type: "module",
});

type StepMsg = { type: "step"; step: number; energy: number; maxForce: number; positions: Float32Array };
type DoneMsg = { type: "done"; converged: boolean; energy: number; maxForce: number; steps: number; elapsedMs: number; positions: Float32Array };
type ReadyMsg = { type: "ready"; variant: string; members: number };
type ErrMsg = { type: "error"; message: string };

worker.onmessage = (ev: MessageEvent<StepMsg | DoneMsg | ReadyMsg | ErrMsg>) => {
  const msg = ev.data;
  if (msg.type === "ready") {
    demo.ready = true;
    btnOptimize.disabled = false;
    btnPerturb.disabled = false;
    setStatus(`model ready — ${msg.variant}, ${msg.members} members`, "good");
    return;
  }
  if (msg.type === "step") {
    demo.step = msg.step;
    demo.lastEnergy = msg.energy;
    demo.lastMaxForce = msg.maxForce;
    demo.energies.push(msg.energy);
    currentPositions = msg.positions;
    if (view) view.update(currentPositions);
    rEnergy.textContent = msg.energy.toFixed(4);
    rForce.textContent = msg.maxForce.toFixed(4);
    rStep.textContent = String(msg.step);
    drawPlot(demo.energies);
    return;
  }
  if (msg.type === "done") {
    demo.optimizing = false;
    demo.done = true;
    demo.converged = msg.converged;
    demo.finalEnergy = msg.energy;
    demo.finalSteps = msg.steps;
    demo.elapsedMs = msg.elapsedMs;
    currentPositions = msg.positions;
    if (view) view.update(currentPositions);
    rEnergy.textContent = msg.energy.toFixed(4);
    rForce.textContent = msg.maxForce.toFixed(4);
    rStep.textContent = String(msg.steps);
    rElapsed.textContent = `${msg.elapsedMs.toFixed(0)} ms`;
    btnOptimize.disabled = false;
    btnPerturb.disabled = false;
    molSelect.disabled = false;
    if (msg.converged) {
      setStatus(
        `relaxed (maxF < 0.02) — E = ${msg.energy.toFixed(4)} Ha in ${msg.elapsedMs.toFixed(0)} ms`,
        "good",
      );
    } else {
      setStatus(
        `stopped at ${msg.steps} steps (maxF = ${msg.maxForce.toFixed(4)}) — E = ${msg.energy.toFixed(4)} Ha`,
        "warn",
      );
    }
    return;
  }
  if (msg.type === "error") {
    demo.error = msg.message;
    demo.optimizing = false;
    btnOptimize.disabled = false;
    btnPerturb.disabled = false;
    molSelect.disabled = false;
    setStatus(`error: ${msg.message}`, "warn");
  }
};

function optimize(): void {
  if (!demo.ready || demo.optimizing) return;
  demo.optimizing = true;
  demo.done = false;
  demo.energies = [];
  drawPlot([]);
  btnOptimize.disabled = true;
  btnPerturb.disabled = true;
  molSelect.disabled = true;
  const t0 = performance.now();
  rElapsed.textContent = "…";
  setStatus("optimizing…");
  worker.postMessage({
    type: "optimize",
    symbols,
    positions: Array.from(currentPositions),
    options: FIRE_OPTIONS,
  });
  // live elapsed ticker (main-thread, proves non-block)
  const tick = () => {
    if (!demo.optimizing) return;
    rElapsed.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------
molSelect.addEventListener("change", () => loadMolecule(molSelect.value));
btnPerturb.addEventListener("click", perturb);
btnOptimize.addEventListener("click", optimize);
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  if (renderer) {
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (rt) {
      const db = renderer.getDrawingBufferSize(new THREE.Vector2());
      rt.setSize(db.x, db.y);
    }
  }
});

// ---------------------------------------------------------------------------
// Render / animation loop (always runs — even without WebGL — so main-thread
// liveness is observable via demo.frameCount).
// ---------------------------------------------------------------------------
function loop(): void {
  requestAnimationFrame(loop);
  demo.frameCount++;
  if (controls) controls.update();
  if (renderer) {
    if (rt) rt.render(scene, camera);
    else renderer.render(scene, camera);
  }
}

// ---------------------------------------------------------------------------
// Boot.
// ---------------------------------------------------------------------------
async function boot(): Promise<void> {
  demo.webgl = initWebGL();
  if (!demo.webgl) {
    noglEl.style.display = "flex";
  } else {
    await initRaytracer();
  }
  loadMolecule(DEFAULT_MOLECULE);
  loop();

  btnOptimize.disabled = true;
  btnPerturb.disabled = true;
  setStatus(`loading model (${MODEL_VARIANT})…`);
  worker.postMessage({
    type: "init",
    modelDir: MODEL_DIR,
    variant: MODEL_VARIANT,
  });
}

boot().catch((err) => {
  demo.error = err instanceof Error ? err.message : String(err);
  setStatus(`boot error: ${demo.error}`, "warn");
});
