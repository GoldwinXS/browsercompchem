import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { MoleculeView, ATOM_RADII } from "./scene.js";
import { MOLECULES, MOLECULE_ORDER, DEFAULT_MOLECULE } from "./molecules.js";
import { smilesToSeed, SMILES_EXAMPLES, rdkitVersion } from "./rdkit.js";
import { createSketcher, sanitizeSketchSmiles, type JsmeApplet } from "./jsme.js";
import { distance as measureDistance, angle as measureAngle } from "./measure.js";

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
  // --- extensions (RDKit SMILES + viewer controls + render mode) ---
  isCustom: boolean; // molecule came from a typed SMILES
  coverageOK: boolean; // all elements within ANI-2x set -> Optimize allowed
  unsupported: string[]; // elements outside ANI-2x, if any
  atomCount: number;
  rtAvailable: boolean; // ray tracer initialised successfully
  rtEnabled: boolean; // ray tracing currently active (vs raster)
  renderMode: "raytraced" | "raster" | "none";
  selected: number[]; // atom indices selected for measurement
  measurement: string | null; // current distance/angle readout text
  rdkitVersion: string | null;
  positions: number[]; // live flat geometry snapshot (headless verification)
  symbols: string[]; // element symbols of the current molecule
  // --- vibrational normal-mode analysis ---
  vibComputing: boolean; // Hessian/normal-mode analysis running in the worker
  vibReady: boolean; // modes available for the current geometry
  vibFrequencies: number[]; // harmonic wavenumbers (cm^-1); imaginary -> negative
  vibIsLinear: boolean; // molecule treated as linear (3N-5 modes)
  vibImaginaryCount: number; // # of negative (imaginary) modes -> non-stationary flag
  vibSelectedMode: number | null; // index into vibFrequencies currently animating
  vibAnimating: boolean; // atoms currently oscillating along a mode
  vibAmplitude: number; // visual displacement amplitude (Angstrom)
  vibMaxForce: number | null; // max force at the geometry the Hessian was built on
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
  isCustom: false,
  coverageOK: true,
  unsupported: [],
  atomCount: 0,
  rtAvailable: false,
  rtEnabled: false, // default to plain raster; ray tracing is opt-in via the toggle
  renderMode: "none",
  selected: [],
  measurement: null,
  rdkitVersion: null,
  positions: [],
  symbols: [],
  vibComputing: false,
  vibReady: false,
  vibFrequencies: [],
  vibIsLinear: false,
  vibImaginaryCount: 0,
  vibSelectedMode: null,
  vibAnimating: false,
  vibAmplitude: 0.3,
  vibMaxForce: null,
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
const smilesInput = $<HTMLInputElement>("smiles");
const btnLoadSmiles = $<HTMLButtonElement>("load-smiles");
const smilesPresets = $<HTMLDataListElement>("smiles-presets");
const warnEl = $("warn");
const chkAutoRotate = $<HTMLInputElement>("autorotate");
const btnToggleRt = $<HTMLButtonElement>("toggle-rt");
const rRender = $("r-render");
const btnClearSel = $<HTMLButtonElement>("clear-sel");
const rMeasure = $("r-measure");
const btnOpenSketch = $<HTMLButtonElement>("open-sketch");
const sketchOverlay = $("sketch-overlay");
const jsmeContainer = $("jsme-container");
const btnSketchUse = $<HTMLButtonElement>("sketch-use");
const btnSketchSeed = $<HTMLButtonElement>("sketch-seed");
const btnSketchClear = $<HTMLButtonElement>("sketch-clear");
const btnSketchCancel = $<HTMLButtonElement>("sketch-cancel");
const sketchStatus = $("sketch-status");
const btnVibrations = $<HTMLButtonElement>("vibrations");
const vibStatusEl = $("vib-status");
const spectrumEl = $("spectrum"); // SVG
const vibNoteEl = $("vib-note");
const vibModeEl = $("vib-mode");
const vibFreqEl = $("vib-freq");
const btnVibStop = $<HTMLButtonElement>("vib-stop");
const vibAmpEl = $<HTMLInputElement>("vib-amp");

for (const key of MOLECULE_ORDER) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = key;
  molSelect.appendChild(opt);
}
molSelect.value = DEFAULT_MOLECULE;

for (const ex of SMILES_EXAMPLES) {
  const opt = document.createElement("option");
  opt.value = ex.smiles;
  opt.textContent = ex.name;
  smilesPresets.appendChild(opt);
}

function setWarn(text: string): void {
  if (text) {
    warnEl.textContent = text;
    warnEl.classList.add("show");
  } else {
    warnEl.textContent = "";
    warnEl.classList.remove("show");
  }
}

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

    const key = new THREE.PointLight(0xffffff, 220);
    key.position.set(6, 8, 6);
    scene.add(key);
    const fill = new THREE.PointLight(0x88aaff, 90);
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
    // ReSTIR re-enabled: the v0.6.0 duplicate-`luminance` shader bug that forced
    // a black image is fixed in three-realtime-rt >= 0.6.1. Default options.
    const r = new RealtimeRaytracer(renderer);
    // Ambient environment light so shadowed faces read (the traced path has no
    // free ambient like the rasterizer). Tuned by eye; safe to adjust.
    r.envColor = new THREE.Color(0x556688);
    r.envIntensity = 2.2;
    r.compileScene(scene);
    rt = r as unknown as typeof rt;
    (window as unknown as { __rt: unknown }).__rt = r; // debug handle
    demo.rtAvailable = true;
  } catch (err) {
    // Fall back to the plain WebGL rasterizer; not a fatal error.
    demo.error = `raytracer: ${err instanceof Error ? err.message : String(err)}`;
    demo.rtAvailable = false;
  }
  updateRenderReadout();
}

/**
 * Reflect the active render path into the `render:` readout and the toggle
 * button. Three cases: ray traced (tracer active), raster because the user
 * toggled it off, or raster because the tracer is unavailable (with the reason
 * the init caught, e.g. "no WebGL2" or a shader-compile failure).
 */
function updateRenderReadout(): void {
  rRender.classList.remove("rt");
  if (!renderer) {
    demo.renderMode = "none";
    rRender.textContent = "render: raster (no WebGL2)";
    btnToggleRt.textContent = "ray tracing: n/a";
    btnToggleRt.disabled = true;
    return;
  }
  if (rt && demo.rtEnabled) {
    demo.renderMode = "raytraced";
    rRender.textContent = "render: ray traced";
    rRender.classList.add("rt");
    btnToggleRt.textContent = "ray tracing: on";
    btnToggleRt.disabled = false;
    return;
  }
  demo.renderMode = "raster";
  if (rt && !demo.rtEnabled) {
    rRender.textContent = "render: raster (ray tracing off)";
    btnToggleRt.textContent = "ray tracing: off";
    btnToggleRt.disabled = false;
  } else {
    // tracer never came up — show why
    const reason = demo.error ? demo.error.replace(/^raytracer:\s*/, "") : "unavailable";
    rRender.textContent = `render: raster (${reason})`;
    btnToggleRt.textContent = "ray tracing: n/a";
    btnToggleRt.disabled = true;
  }
}

// ---------------------------------------------------------------------------
// Molecule state.
// ---------------------------------------------------------------------------
let view: MoleculeView | undefined;
let symbols: string[] = [];
/** Canonical SMILES of the most recent custom (typed/drawn) molecule, for re-seeding the sketcher. */
let lastCustomSmiles: string | null = null;
let basePositions = new Float64Array(0); // reference geometry (Perturb origin)
let currentPositions: Float32Array<ArrayBufferLike> = new Float32Array(0); // live geometry

// --- vibrational normal-mode state (separate view mode; never touches the
// optimize flow). Modes are stored flat (nModes * n3) with their frequencies. ---
let vibEquilibrium: Float64Array<ArrayBufferLike> = new Float64Array(0); // geometry the Hessian was built on
let vibModes: Float64Array<ArrayBufferLike> = new Float64Array(0); // flat (nModes * n3) normalized displacements
let vibNModes = 0;
let vibN3 = 0;
let vibClock = 0; // seconds, advanced by the render loop while animating
let vibLastT = 0; // performance.now() of the previous animation frame
const VIB_DISPLAY_HZ = 1.1; // visual oscillation rate (decoupled from the real cm^-1)
const vibScratch = { buf: new Float32Array(0) }; // reused per-frame position buffer

/**
 * Install a geometry (symbols + flat Angstrom positions) into the scene and
 * frame the camera. Shared by the built-in dropdown and the SMILES loader.
 */
function setGeometry(newSymbols: string[], positions: ArrayLike<number>): void {
  symbols = newSymbols;
  basePositions = Float64Array.from(positions);
  currentPositions = Float32Array.from(positions);
  demo.atomCount = newSymbols.length;
  demo.symbols = newSymbols;
  demo.positions = Array.from(currentPositions);
  clearSelection();

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
  resetVibrations();
  refreshVibButton();
}

/** Load one of the baked built-in molecules (all within the ANI-2x element set). */
function loadMolecule(key: string): void {
  const data = MOLECULES[key];
  if (!data) throw new Error(`unknown molecule ${key}`);
  demo.molecule = key;
  demo.isCustom = false;
  demo.coverageOK = true;
  demo.unsupported = [];
  setWarn("");
  setGeometry(data.symbols, data.positions);
  btnPerturb.disabled = !demo.ready;
  btnOptimize.disabled = !demo.ready;
}

/**
 * Load a molecule typed as SMILES. RDKit supplies connectivity + explicit H;
 * ANI-2x + FIRE (in the worker) produces the genuine 3D geometry from the
 * planarity-broken seed. See rdkit.ts for why RDKit alone can't embed 3D here.
 */
async function loadSmiles(): Promise<void> {
  const smiles = smilesInput.value.trim();
  await loadCustomMolecule(smiles);
}

/**
 * Shared "custom molecule from a SMILES string" path. Both the typed-SMILES
 * input and the JSME sketcher funnel through here: RDKit supplies connectivity +
 * explicit H, ANI-2x + FIRE (in the worker) produces the genuine 3D geometry
 * from the planarity-broken seed, and the coverage guard disables Optimize for
 * elements outside the ANI-2x set. See rdkit.ts for why RDKit alone can't embed
 * 3D here.
 */
async function loadCustomMolecule(smiles: string): Promise<void> {
  if (!smiles) return;
  if (demo.optimizing) return;
  setWarn("");
  btnLoadSmiles.disabled = true;
  setStatus("parsing SMILES…");
  try {
    if (demo.rdkitVersion === null) demo.rdkitVersion = await rdkitVersion();
    const seed = await smilesToSeed(smiles);
    lastCustomSmiles = seed.canonicalSmiles;
    demo.molecule = `smiles:${seed.canonicalSmiles}`;
    demo.isCustom = true;
    demo.unsupported = seed.unsupported;
    demo.coverageOK = seed.unsupported.length === 0;
    setGeometry(seed.symbols, seed.positions);

    if (!demo.coverageOK) {
      btnOptimize.disabled = true;
      btnPerturb.disabled = true;
      setWarn(
        `contains ${seed.unsupported.join(", ")} — outside ANI-2x (H,C,N,O,F,S,Cl). ` +
          `Structure shown for display only; Optimize disabled.`,
      );
      setStatus(`loaded ${seed.symbols.length} atoms — display only`, "warn");
      return;
    }

    setStatus(`embedding 3D geometry for ${seed.symbols.length} atoms…`);
    // The seed is a planarity-broken 2D layout; relax it into a real 3D
    // minimum with ANI-2x before the user does anything else.
    if (demo.ready) {
      optimize({ embedding: true });
    } else {
      btnPerturb.disabled = true;
      btnOptimize.disabled = true;
      setStatus("model still loading — will embed once ready…");
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setWarn(`SMILES error: ${msg}`);
    setStatus("SMILES parse failed", "warn");
  } finally {
    btnLoadSmiles.disabled = false;
  }
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
  resetVibrations();
  refreshVibButton();
  setStatus("perturbed — press Optimize to relax", "warn");
}

// ---------------------------------------------------------------------------
// Click-to-measure: raycast atom spheres, select up to 3, show distance/angle.
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const markerGeo = new THREE.SphereGeometry(1, 20, 12);
const markerMat = new THREE.MeshBasicMaterial({
  color: 0x46d18a,
  wireframe: true,
  transparent: true,
  opacity: 0.9,
});
const markers: THREE.Mesh[] = [];
function ensureMarkers(): void {
  if (markers.length || !renderer) return;
  for (let k = 0; k < 3; k++) {
    const m = new THREE.Mesh(markerGeo, markerMat);
    m.visible = false;
    scene.add(m);
    markers.push(m);
  }
}

function clearSelection(): void {
  demo.selected = [];
  updateMeasureReadout();
  syncMarkers();
}

/** Position/show the selection markers over the currently selected atoms. */
function syncMarkers(): void {
  ensureMarkers();
  for (let k = 0; k < markers.length; k++) {
    const idx = demo.selected[k];
    const mk = markers[k]!;
    if (idx === undefined) {
      mk.visible = false;
      continue;
    }
    const el = symbols[idx] ?? "";
    const r = (ATOM_RADII[el] ?? 0.3) * 1.45;
    mk.scale.setScalar(r);
    mk.position.set(
      currentPositions[3 * idx]!,
      currentPositions[3 * idx + 1]!,
      currentPositions[3 * idx + 2]!,
    );
    mk.visible = true;
  }
}

function updateMeasureReadout(): void {
  const s = demo.selected;
  const tag = (i: number): string => `${symbols[i] ?? "?"}${i}`;
  if (s.length === 0) {
    demo.measurement = null;
    rMeasure.textContent = "select 2 atoms for distance, 3 for angle";
  } else if (s.length === 1) {
    demo.measurement = null;
    rMeasure.textContent = `${tag(s[0]!)} selected`;
  } else if (s.length === 2) {
    const d = measureDistance(currentPositions, s[0]!, s[1]!);
    demo.measurement = `d(${tag(s[0]!)},${tag(s[1]!)}) = ${d.toFixed(3)} A`;
    rMeasure.textContent = `${tag(s[0]!)}–${tag(s[1]!)}: ${d.toFixed(3)} Å`;
  } else {
    const a = measureAngle(currentPositions, s[0]!, s[1]!, s[2]!);
    demo.measurement = `angle(${tag(s[0]!)},${tag(s[1]!)},${tag(s[2]!)}) = ${a.toFixed(1)} deg`;
    rMeasure.textContent = `∠ ${tag(s[0]!)}–${tag(s[1]!)}–${tag(s[2]!)}: ${a.toFixed(1)}° (vertex ${tag(s[1]!)})`;
  }
}

function toggleAtom(idx: number): void {
  const at = demo.selected.indexOf(idx);
  if (at >= 0) {
    demo.selected.splice(at, 1); // clicking a selected atom deselects it
  } else if (demo.selected.length >= 3) {
    demo.selected = [idx]; // 4th pick -> start a fresh measurement
  } else {
    demo.selected.push(idx);
  }
  updateMeasureReadout();
  syncMarkers();
}

/** Raycast the atom spheres at NDC (x,y); return atom index or -1. */
function pickAtom(clientX: number, clientY: number): number {
  if (!renderer || !view) return -1;
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(view.atomMeshes, false);
  if (!hits.length) return -1;
  return view.atomMeshes.indexOf(hits[0]!.object as THREE.Mesh);
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
type VibProgressMsg = { type: "vib-progress"; done: number; total: number };
type VibDoneMsg = {
  type: "vib-done";
  frequencies: number[];
  modes: Float64Array; // flat (nModes * n3)
  nModes: number;
  n3: number;
  isLinear: boolean;
  maxResidualTransRot: number;
  maxForce: number;
  relaxed: boolean;
  equilibrium: Float32Array;
  elapsedMs: number;
};

type WorkerMsg = StepMsg | DoneMsg | ReadyMsg | ErrMsg | VibProgressMsg | VibDoneMsg;

worker.onmessage = (ev: MessageEvent<WorkerMsg>) => {
  const msg = ev.data;
  if (msg.type === "ready") {
    demo.ready = true;
    btnOptimize.disabled = !demo.coverageOK;
    btnPerturb.disabled = !demo.coverageOK;
    btnLoadSmiles.disabled = false;
    refreshVibButton();
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
    if (demo.selected.length) {
      syncMarkers();
      updateMeasureReadout();
    }
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
    demo.positions = Array.from(currentPositions);
    // A SMILES seed is not an equilibrium; adopt the relaxed geometry as the
    // new Perturb origin so subsequent Perturb/Optimize behave sensibly.
    if (demo.isCustom) basePositions = Float64Array.from(currentPositions);
    if (demo.selected.length) {
      syncMarkers();
      updateMeasureReadout();
    }
    rEnergy.textContent = msg.energy.toFixed(4);
    rForce.textContent = msg.maxForce.toFixed(4);
    rStep.textContent = String(msg.steps);
    rElapsed.textContent = `${msg.elapsedMs.toFixed(0)} ms`;
    btnOptimize.disabled = !demo.coverageOK;
    btnPerturb.disabled = !demo.coverageOK;
    btnLoadSmiles.disabled = false;
    molSelect.disabled = false;
    refreshVibButton();
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
  if (msg.type === "vib-progress") {
    vibStatusEl.textContent = `computing Hessian… ${msg.done}/${msg.total} force columns`;
    return;
  }
  if (msg.type === "vib-done") {
    demo.vibComputing = false;
    vibN3 = msg.n3;
    vibNModes = msg.nModes;
    vibModes = msg.modes;
    vibEquilibrium = Float64Array.from(msg.equilibrium);
    vibScratch.buf = new Float32Array(msg.n3);
    demo.vibFrequencies = msg.frequencies.slice();
    demo.vibIsLinear = msg.isLinear;
    demo.vibImaginaryCount = msg.frequencies.filter((f) => f < 0).length;
    demo.vibMaxForce = msg.maxForce;
    demo.vibReady = true;
    demo.vibSelectedMode = null;
    demo.vibAnimating = false;

    // Adopt the (possibly further-relaxed) equilibrium so the animation
    // oscillates around a true stationary point and Perturb stays sensible.
    currentPositions = Float32Array.from(msg.equilibrium);
    basePositions = Float64Array.from(msg.equilibrium);
    demo.positions = Array.from(currentPositions);
    if (view) view.update(currentPositions);

    drawSpectrum();
    spectrumEl.style.display = "block";
    vibNoteEl.style.display = "block";
    vibModeEl.style.display = "none";
    btnVibrations.disabled = !demo.coverageOK;
    btnOptimize.disabled = !demo.coverageOK;
    btnPerturb.disabled = !demo.coverageOK;
    molSelect.disabled = false;
    btnLoadSmiles.disabled = false;

    const imag = demo.vibImaginaryCount;
    const kind = msg.isLinear ? "linear" : "non-linear";
    const relaxNote = msg.relaxed ? " (relaxed to stationary point first)" : "";
    if (imag > 0) {
      setVibStatus(
        `${msg.nModes} modes, ${kind} — ${imag} imaginary (negative): not a true minimum. ` +
          `Click a stick to animate.${relaxNote}`,
        "warn",
      );
    } else {
      setVibStatus(
        `${msg.nModes} real modes, ${kind}, maxF ${msg.maxForce.toExponential(1)} ` +
          `in ${msg.elapsedMs.toFixed(0)} ms. Click a stick to animate.${relaxNote}`,
        "good",
      );
    }
    return;
  }
  if (msg.type === "error") {
    demo.error = msg.message;
    demo.optimizing = false;
    demo.vibComputing = false;
    btnOptimize.disabled = !demo.coverageOK;
    btnPerturb.disabled = !demo.coverageOK;
    btnLoadSmiles.disabled = false;
    btnVibrations.disabled = !(demo.ready && demo.coverageOK && demo.atomCount >= 2);
    molSelect.disabled = false;
    setStatus(`error: ${msg.message}`, "warn");
    setVibStatus(`error: ${msg.message}`, "warn");
  }
};

function optimize(opts: { embedding?: boolean } = {}): void {
  if (!demo.ready || demo.optimizing) return;
  if (!demo.coverageOK) return; // elements outside ANI-2x -> no energies
  resetVibrations(); // geometry is about to move; any modes become stale
  demo.optimizing = true;
  demo.done = false;
  demo.energies = [];
  drawPlot([]);
  btnOptimize.disabled = true;
  btnPerturb.disabled = true;
  btnLoadSmiles.disabled = true;
  btnVibrations.disabled = true;
  molSelect.disabled = true;
  const t0 = performance.now();
  rElapsed.textContent = "…";
  setStatus(opts.embedding ? "embedding 3D geometry (ANI-2x)…" : "optimizing…");
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
// Vibrational normal-mode analysis (separate view state — does not disturb the
// optimize flow). The worker builds the Hessian off-thread; here we draw the
// harmonic spectrum and animate the atoms along a chosen mode.
// ---------------------------------------------------------------------------
function setVibStatus(text: string, cls: "" | "good" | "warn" = ""): void {
  vibStatusEl.textContent = text;
  vibStatusEl.className = cls;
}

/** Whether normal-mode analysis is currently allowed for the loaded molecule. */
function vibAllowed(): boolean {
  return demo.ready && demo.coverageOK && demo.atomCount >= 2 && !demo.optimizing && !demo.vibComputing;
}

function refreshVibButton(): void {
  btnVibrations.disabled = !vibAllowed();
}

/** Clear any computed modes/animation (called when the geometry changes). */
function resetVibrations(): void {
  stopVibAnimation();
  demo.vibReady = false;
  demo.vibFrequencies = [];
  demo.vibSelectedMode = null;
  demo.vibImaginaryCount = 0;
  demo.vibMaxForce = null;
  vibNModes = 0;
  vibModes = new Float64Array(0);
  spectrumEl.style.display = "none";
  vibNoteEl.style.display = "none";
  vibModeEl.style.display = "none";
  spectrumEl.innerHTML = "";
}

/** Kick off a Hessian + normal-mode computation in the worker. */
function computeVibrations(): void {
  if (!vibAllowed()) return;
  resetVibrations();
  demo.vibComputing = true;
  btnVibrations.disabled = true;
  btnOptimize.disabled = true;
  btnPerturb.disabled = true;
  molSelect.disabled = true;
  btnLoadSmiles.disabled = true;
  setVibStatus("computing Hessian (6N force evals, off-thread)…");
  worker.postMessage({
    type: "vibrations",
    symbols,
    positions: Array.from(currentPositions),
  });
}

// --- spectrum geometry (viewBox 0 0 300 72) ---
const SPEC_W = 300;
const SPEC_H = 72;
const SPEC_L = 6; // left margin
const SPEC_R = 8; // right margin
const SPEC_TOP = 8; // stick top
const SPEC_BASE = 54; // baseline (axis) y
const STICK_TOP = 12; // sticks rise to here (uniform height — see honesty note)

function spectrumAxisMax(): number {
  let m = 0;
  for (const f of demo.vibFrequencies) m = Math.max(m, Math.abs(f));
  // round up to the next 500, floor 4000 so X-H stretches sit comfortably
  return Math.max(4000, Math.ceil((m * 1.05) / 500) * 500);
}

function freqToX(absFreq: number, axisMax: number): number {
  return SPEC_L + (absFreq / axisMax) * (SPEC_W - SPEC_L - SPEC_R);
}

/** Draw the clickable stick spectrum of harmonic wavenumbers (equal heights). */
function drawSpectrum(): void {
  const freqs = demo.vibFrequencies;
  if (!freqs.length) {
    spectrumEl.innerHTML = "";
    return;
  }
  const axisMax = spectrumAxisMax();
  const parts: string[] = [];
  // baseline axis
  parts.push(
    `<line class="axis" x1="${SPEC_L}" y1="${SPEC_BASE}" x2="${SPEC_W - SPEC_R}" y2="${SPEC_BASE}" />`,
  );
  // x-axis ticks + labels every 1000 cm^-1
  for (let t = 0; t <= axisMax; t += 1000) {
    const x = freqToX(t, axisMax);
    parts.push(`<line class="axis" x1="${x.toFixed(1)}" y1="${SPEC_BASE}" x2="${x.toFixed(1)}" y2="${SPEC_BASE + 3}" />`);
    parts.push(`<text x="${x.toFixed(1)}" y="${SPEC_H - 4}" text-anchor="middle">${t}</text>`);
  }
  parts.push(
    `<text x="${SPEC_W - SPEC_R}" y="${STICK_TOP - 2}" text-anchor="end">cm⁻¹</text>`,
  );
  // sticks (uniform height) + wide transparent hit targets
  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i]!;
    const x = freqToX(Math.abs(f), axisMax);
    const cls = `stick${i === demo.vibSelectedMode ? " sel" : ""}${f < 0 ? " imag" : ""}`;
    parts.push(
      `<line class="${cls}" x1="${x.toFixed(1)}" y1="${SPEC_BASE}" x2="${x.toFixed(1)}" y2="${STICK_TOP}" />`,
    );
    parts.push(
      `<rect class="hit" data-mode="${i}" x="${(x - 4).toFixed(1)}" y="${STICK_TOP - 2}" width="8" height="${SPEC_BASE - STICK_TOP + 4}" />`,
    );
  }
  spectrumEl.innerHTML = parts.join("");
}

function modeCharacter(freq: number): string {
  const f = Math.abs(freq);
  if (freq < 0) return "imaginary (negative curvature)";
  if (f > 2800) return "X–H stretch region";
  if (f > 1500) return "double-bond / bend region";
  return "";
}

/** Select a mode by index: label it, and start the oscillation animation. */
function selectMode(i: number): void {
  if (!demo.vibReady || i < 0 || i >= vibNModes) return;
  demo.vibSelectedMode = i;
  const f = demo.vibFrequencies[i]!;
  const sign = f < 0 ? "i" : "";
  const character = modeCharacter(f);
  vibFreqEl.textContent =
    `mode ${i + 1}/${vibNModes}: ${Math.abs(f).toFixed(1)}${sign} cm⁻¹` +
    (character ? ` · ${character}` : "");
  vibModeEl.style.display = "block";
  drawSpectrum(); // repaint to highlight the selected stick
  // start animation
  demo.vibAnimating = true;
  vibClock = 0;
  vibLastT = performance.now();
}

/** Compute the oscillating geometry for the selected mode at animation time t. */
function vibPositionsAt(t: number): Float32Array {
  const out = vibScratch.buf;
  const i = demo.vibSelectedMode;
  if (i === null || vibN3 === 0 || out.length !== vibN3) {
    return currentPositions;
  }
  const base = i * vibN3;
  const s = demo.vibAmplitude * Math.sin(2 * Math.PI * VIB_DISPLAY_HZ * t);
  for (let k = 0; k < vibN3; k++) {
    out[k] = vibEquilibrium[k]! + s * vibModes[base + k]!;
  }
  return out;
}

/** Advance and paint the mode animation; called from the render loop. */
function updateVibAnimation(): void {
  if (!demo.vibAnimating || demo.vibSelectedMode === null) return;
  const now = performance.now();
  vibClock += (now - vibLastT) / 1000;
  vibLastT = now;
  const p = vibPositionsAt(vibClock);
  if (view) view.update(p);
  if (demo.selected.length) syncMarkers();
}

/** Stop animating and snap the atoms back to the equilibrium geometry. */
function stopVibAnimation(): void {
  demo.vibAnimating = false;
  demo.vibSelectedMode = null;
  vibModeEl.style.display = "none";
  if (vibEquilibrium.length && view) {
    view.update(vibEquilibrium);
  } else if (view && currentPositions.length) {
    view.update(currentPositions);
  }
  if (demo.vibReady) drawSpectrum();
}

// ---------------------------------------------------------------------------
// Structure sketcher (JSME). The 2D editor lives in a modal; on "Use structure"
// we read its SMILES and push it through the SAME loadCustomMolecule path the
// typed-SMILES box uses (RDKit seed -> ANI-2x + FIRE). Exposed for headless
// verification via window.__sketch.
// ---------------------------------------------------------------------------
let sketchApplet: JsmeApplet | undefined;
let sketchLoading = false;

function setSketchStatus(text: string, warn = false): void {
  sketchStatus.textContent = text;
  sketchStatus.classList.toggle("warn", warn);
}

/** Ensure the JSME applet exists inside the modal (lazy, first-open only). */
async function ensureSketcher(): Promise<JsmeApplet | undefined> {
  if (sketchApplet) return sketchApplet;
  if (sketchLoading) return undefined;
  sketchLoading = true;
  setSketchStatus("loading sketcher…");
  try {
    // Pass the DOM id (not the element) — GWT resolves the container by id.
    sketchApplet = await createSketcher(jsmeContainer.id, { width: "360px", height: "320px" });
    setSketchStatus("");
    return sketchApplet;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setSketchStatus(`sketcher failed to load: ${msg}`, true);
    return undefined;
  } finally {
    sketchLoading = false;
  }
}

async function openSketch(): Promise<void> {
  sketchOverlay.classList.add("show");
  const applet = await ensureSketcher();
  // Convenience: if the user already typed a SMILES (or loaded one), seed it so
  // they can edit rather than start blank. Silent — a bad seed just no-ops.
  if (applet) {
    const seed = smilesInput.value.trim() || lastCustomSmiles;
    if (seed) {
      try {
        applet.readGenericMolecularInput(seed);
      } catch {
        /* ignore — leave the canvas blank */
      }
    }
  }
}

function closeSketch(): void {
  sketchOverlay.classList.remove("show");
}

/** Read the sketch, sanitize, and feed it through the shared custom-molecule path. */
async function useSketch(): Promise<void> {
  if (!sketchApplet) {
    setSketchStatus("sketcher not ready yet", true);
    return;
  }
  try {
    const smiles = sanitizeSketchSmiles(sketchApplet.smiles());
    smilesInput.value = smiles; // reflect into the typed box for transparency
    closeSketch();
    await loadCustomMolecule(smiles); // parse problems surface via setWarn there
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setSketchStatus(msg, true);
  }
}

/** Seed the canvas from the current custom molecule (or the typed SMILES). */
function seedSketch(): void {
  if (!sketchApplet) return;
  const seed = lastCustomSmiles ?? smilesInput.value.trim();
  if (!seed) {
    setSketchStatus("no current molecule SMILES to seed from", true);
    return;
  }
  try {
    sketchApplet.readGenericMolecularInput(seed);
    setSketchStatus("");
  } catch {
    setSketchStatus("could not seed the canvas from that SMILES", true);
  }
}

function clearSketch(): void {
  if (sketchApplet) sketchApplet.reset();
  setSketchStatus("");
}

// Headless-verification handle: lets the browser harness drive the round-trip
// (readGenericMolecularInput -> smiles() -> loadCustomMolecule) without a mouse.
(window as unknown as {
  __sketch: {
    open: () => Promise<void>;
    ensure: () => Promise<JsmeApplet | undefined>;
    use: () => Promise<void>;
    applet: () => JsmeApplet | undefined;
    loadSmiles: (s: string) => Promise<void>;
  };
}).__sketch = {
  open: openSketch,
  ensure: ensureSketcher,
  use: useSketch,
  applet: () => sketchApplet,
  loadSmiles: loadCustomMolecule,
};

// Headless-verification handle for the vibrations flow: drive compute/select/
// stop without a mouse, and sample the animated geometry at an arbitrary time
// so a harness can confirm the atoms actually move along the mode.
(window as unknown as {
  __vib: {
    compute: () => void;
    selectMode: (i: number) => void;
    stop: () => void;
    sampleAt: (t: number) => number[];
    frequencies: () => number[];
    ready: () => boolean;
  };
}).__vib = {
  compute: computeVibrations,
  selectMode,
  stop: stopVibAnimation,
  sampleAt: (t: number) => Array.from(vibPositionsAt(t)),
  frequencies: () => demo.vibFrequencies.slice(),
  ready: () => demo.vibReady,
};

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------
molSelect.addEventListener("change", () => loadMolecule(molSelect.value));
btnPerturb.addEventListener("click", () => perturb());
btnOptimize.addEventListener("click", () => optimize());
btnLoadSmiles.addEventListener("click", () => void loadSmiles());
smilesInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    void loadSmiles();
  }
});

// auto-rotate toggle (default on; degrades gracefully with no controls)
chkAutoRotate.addEventListener("change", () => {
  if (controls) controls.autoRotate = chkAutoRotate.checked;
});

// ray-tracing on/off toggle (falls back to renderer.render when off)
btnToggleRt.addEventListener("click", () => {
  if (!rt) return;
  demo.rtEnabled = !demo.rtEnabled;
  updateRenderReadout();
});

// clear measurement selection
btnClearSel.addEventListener("click", clearSelection);

// normal-mode analysis
btnVibrations.addEventListener("click", () => computeVibrations());
btnVibStop.addEventListener("click", () => stopVibAnimation());
vibAmpEl.addEventListener("input", () => {
  demo.vibAmplitude = Number(vibAmpEl.value);
});
// click a spectrum stick's hit target to select + animate that mode
spectrumEl.addEventListener("click", (e) => {
  const target = e.target as Element | null;
  const attr = target?.getAttribute?.("data-mode");
  if (attr !== null && attr !== undefined) selectMode(Number(attr));
});

// structure sketcher (JSME modal)
btnOpenSketch.addEventListener("click", () => void openSketch());
btnSketchUse.addEventListener("click", () => void useSketch());
btnSketchSeed.addEventListener("click", () => seedSketch());
btnSketchClear.addEventListener("click", () => clearSketch());
btnSketchCancel.addEventListener("click", () => closeSketch());
// click the dim backdrop (outside the modal) to dismiss
sketchOverlay.addEventListener("click", (e) => {
  if (e.target === sketchOverlay) closeSketch();
});

// Click-to-measure: distinguish a click from an orbit drag by pointer travel.
if (renderer) {
  let downX = 0;
  let downY = 0;
  let dragged = false;
  const canvas = renderer.domElement;
  canvas.addEventListener("pointerdown", (e) => {
    downX = e.clientX;
    downY = e.clientY;
    dragged = false;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) dragged = true;
  });
  canvas.addEventListener("pointerup", (e) => {
    if (dragged) return; // it was an orbit drag, not a pick
    const idx = pickAtom(e.clientX, e.clientY);
    if (idx >= 0) toggleAtom(idx);
    else clearSelection(); // clicked empty space
  });
}

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
  updateVibAnimation(); // oscillate atoms along the selected normal mode
  if (demo.selected.length) syncMarkers(); // keep markers glued during orbit
  if (renderer) {
    if (rt && demo.rtEnabled) rt.render(scene, camera);
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
  }
  // Load the molecule (adds meshes) BEFORE compiling the ray tracer:
  // three-realtime-rt rejects an empty scene ("no meshes found in scene"),
  // which would otherwise leave us permanently on the raster fallback.
  loadMolecule(DEFAULT_MOLECULE);
  if (controls) controls.autoRotate = chkAutoRotate.checked;
  if (demo.webgl) {
    await initRaytracer();
  }
  updateRenderReadout(); // covers the no-WebGL / no-tracer cases too
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
