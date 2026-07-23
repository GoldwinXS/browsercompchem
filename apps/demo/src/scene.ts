import * as THREE from "three";

/**
 * Ball-and-stick molecule view with a FIXED bond topology (perceived once from
 * the initial geometry by distance cutoff) whose sphere positions and bond
 * cylinders are cheaply re-placed as atoms move during optimization. Fixed
 * topology is correct for a relaxation (bonds don't break) and avoids an
 * O(N^2) re-perception every frame.
 */

/** Element -> CPK-ish color (hex). */
export const CPK_COLORS: Record<string, number> = {
  H: 0xffffff,
  C: 0x555555,
  N: 0x3050f8,
  O: 0xff0d0d,
  S: 0xf0d020,
  Cl: 0x1fd01f,
  F: 0x50e050,
};

/** Element -> display sphere radius (Angstrom-scaled). */
export const ATOM_RADII: Record<string, number> = {
  H: 0.22,
  C: 0.35,
  N: 0.33,
  O: 0.32,
  S: 0.42,
  Cl: 0.4,
  F: 0.3,
};

const DEFAULT_COLOR = 0xcccccc;
const DEFAULT_RADIUS = 0.3;
const BOND_RADIUS = 0.09;
/** Per-element covalent radius used for topology perception. */
const COVALENT: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  S: 1.05,
  Cl: 1.02,
  F: 0.57,
};
const BOND_SLACK = 0.45; // Angstrom tolerance added to covalent-radius sum

interface Bond {
  i: number;
  j: number;
}

const Y_AXIS = new THREE.Vector3(0, 1, 0);

/**
 * A live molecule view. `update(positions)` re-places every atom sphere and
 * bond cylinder from a flat [x0,y0,z0,...] buffer (Float32Array or number[]).
 */
export class MoleculeView {
  readonly group = new THREE.Group();
  private readonly atomMeshes: THREE.Mesh[] = [];
  private readonly bondMeshes: THREE.Mesh[] = [];
  private readonly bonds: Bond[] = [];
  private readonly symbols: string[];
  private readonly tmpMid = new THREE.Vector3();
  private readonly tmpDir = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
  private readonly a = new THREE.Vector3();
  private readonly b = new THREE.Vector3();

  constructor(symbols: string[], positions: ArrayLike<number>) {
    this.symbols = symbols;

    const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
    const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 12);
    const matCache = new Map<number, THREE.MeshStandardMaterial>();
    const materialFor = (color: number): THREE.MeshStandardMaterial => {
      let m = matCache.get(color);
      if (!m) {
        m = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.05 });
        matCache.set(color, m);
      }
      return m;
    };

    for (let i = 0; i < symbols.length; i++) {
      const el = symbols[i]!;
      const mesh = new THREE.Mesh(sphereGeo, materialFor(CPK_COLORS[el] ?? DEFAULT_COLOR));
      mesh.scale.setScalar(ATOM_RADII[el] ?? DEFAULT_RADIUS);
      this.atomMeshes.push(mesh);
      this.group.add(mesh);
    }

    // Perceive bonds once from the initial geometry.
    const bondMat = materialFor(0x9aa0aa);
    const n = symbols.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = positions[3 * i]! - positions[3 * j]!;
        const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
        const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const cutoff =
          (COVALENT[symbols[i]!] ?? 0.7) + (COVALENT[symbols[j]!] ?? 0.7) + BOND_SLACK;
        if (d <= cutoff) {
          this.bonds.push({ i, j });
          this.bondMeshes.push(new THREE.Mesh(cylGeo, bondMat));
        }
      }
    }
    for (const bm of this.bondMeshes) this.group.add(bm);

    this.update(positions);
  }

  /** Re-place all atoms and bonds from a flat position buffer (Angstrom). */
  update(positions: ArrayLike<number>): void {
    for (let i = 0; i < this.atomMeshes.length; i++) {
      this.atomMeshes[i]!.position.set(
        positions[3 * i]!,
        positions[3 * i + 1]!,
        positions[3 * i + 2]!,
      );
    }
    for (let k = 0; k < this.bonds.length; k++) {
      const { i, j } = this.bonds[k]!;
      this.a.set(positions[3 * i]!, positions[3 * i + 1]!, positions[3 * i + 2]!);
      this.b.set(positions[3 * j]!, positions[3 * j + 1]!, positions[3 * j + 2]!);
      const bond = this.bondMeshes[k]!;
      this.tmpMid.addVectors(this.a, this.b).multiplyScalar(0.5);
      const len = this.a.distanceTo(this.b);
      bond.position.copy(this.tmpMid);
      bond.scale.set(BOND_RADIUS, len, BOND_RADIUS);
      this.tmpDir.subVectors(this.b, this.a).normalize();
      this.tmpQuat.setFromUnitVectors(Y_AXIS, this.tmpDir);
      bond.quaternion.copy(this.tmpQuat);
    }
  }

  /** Centroid of the current atom positions (for framing the camera). */
  centroid(positions: ArrayLike<number>): THREE.Vector3 {
    const c = new THREE.Vector3();
    const n = this.symbols.length;
    for (let i = 0; i < n; i++) {
      c.x += positions[3 * i]!;
      c.y += positions[3 * i + 1]!;
      c.z += positions[3 * i + 2]!;
    }
    return c.multiplyScalar(1 / n);
  }

  dispose(scene: THREE.Scene): void {
    scene.remove(this.group);
  }
}
