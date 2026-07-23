import * as THREE from "three";
import { ATOM_RADII, BOND_CUTOFF, CAFFEINE_ATOMS, CPK_COLORS, type AtomSpec } from "./caffeine.js";

const DEFAULT_COLOR = 0xcccccc;
const DEFAULT_RADIUS = 0.3;
const BOND_RADIUS = 0.09;

function atomColor(element: string): number {
  return CPK_COLORS[element] ?? DEFAULT_COLOR;
}

function atomRadius(element: string): number {
  return ATOM_RADII[element] ?? DEFAULT_RADIUS;
}

function distance(a: AtomSpec, b: AtomSpec): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * Builds a ball-and-stick THREE.Group for a fixed list of atoms:
 * one sphere per atom (CPK-ish coloring), one cylinder per bonded
 * pair (naive distance-cutoff bond perception -- fine for a single
 * hardcoded, already-sensible geometry like this demo's caffeine
 * molecule; a real bond-perception algorithm belongs in the engine
 * package once RDKit-JS is wired in).
 */
export function buildMoleculeGroup(atoms: AtomSpec[] = CAFFEINE_ATOMS): THREE.Group {
  const group = new THREE.Group();

  const sphereGeometry = new THREE.SphereGeometry(1, 24, 16);
  const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 12);

  const materialCache = new Map<number, THREE.MeshStandardMaterial>();
  function materialFor(color: number): THREE.MeshStandardMaterial {
    let mat = materialCache.get(color);
    if (!mat) {
      mat = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05 });
      materialCache.set(color, mat);
    }
    return mat;
  }

  for (const atom of atoms) {
    const mesh = new THREE.Mesh(sphereGeometry, materialFor(atomColor(atom.element)));
    const r = atomRadius(atom.element);
    mesh.scale.setScalar(r);
    mesh.position.set(atom.x, atom.y, atom.z);
    group.add(mesh);
  }

  const bondMaterial = materialFor(0x999999);
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const a = atoms[i]!;
      const b = atoms[j]!;
      const d = distance(a, b);
      if (d > BOND_CUTOFF) continue;

      const mid = new THREE.Vector3(
        (a.x + b.x) / 2,
        (a.y + b.y) / 2,
        (a.z + b.z) / 2,
      );
      const bond = new THREE.Mesh(cylinderGeometry, bondMaterial);
      bond.scale.set(BOND_RADIUS, d, BOND_RADIUS);
      bond.position.copy(mid);

      // orient the unit cylinder (default axis = +Y) along a->b
      const dir = new THREE.Vector3(b.x - a.x, b.y - a.y, b.z - a.z).normalize();
      const quaternion = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir,
      );
      bond.quaternion.copy(quaternion);

      group.add(bond);
    }
  }

  return group;
}
