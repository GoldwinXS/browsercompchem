import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RealtimeRaytracer } from "three-realtime-rt";
import { buildMoleculeGroup } from "./scene.js";

/**
 * Minimal demo: render a hardcoded caffeine molecule (ball-and-stick)
 * with three-realtime-rt's hybrid ray-traced renderer. This is meant
 * only to prove out the rendering seam the rest of the workbench UI
 * will build on -- the interesting chemistry (RDKit-JS sketch->3D, ML
 * potentials, orbitals) is not wired up yet, see packages/engine.
 */

const canvas = document.querySelector<HTMLDivElement>("#app");
if (!canvas) throw new Error("#app root element not found");

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
canvas.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11131a);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  200,
);
camera.position.set(6, 5, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-2, -0.5, 0); // roughly the molecule's centroid
controls.enableDamping = true;

// Ordinary three.js content: MeshStandardMaterial spheres/cylinders + real lights.
const molecule = buildMoleculeGroup();
scene.add(molecule);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({ color: 0x1c1f2b, roughness: 1.0 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -4;
scene.add(floor);

const key = new THREE.PointLight(0xffffff, 60);
key.position.set(6, 8, 6);
scene.add(key);

const fill = new THREE.PointLight(0x88aaff, 20);
fill.position.set(-8, 4, -4);
scene.add(fill);

// Turn on ray tracing (falls back to renderer.render transparently on
// hardware that can't trace -- see three-realtime-rt's README).
const rt = new RealtimeRaytracer(renderer);
rt.compileScene(scene);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  const db = renderer.getDrawingBufferSize(new THREE.Vector2());
  rt.setSize(db.x, db.y);
});

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  rt.render(scene, camera);
}
loop();
