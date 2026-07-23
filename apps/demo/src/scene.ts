import * as THREE from "three";

/**
 * Ball-and-stick molecule view with a FIXED bond topology whose sphere positions
 * and bond cylinders are cheaply re-placed as atoms move during optimization.
 * Fixed topology is correct for a relaxation (bonds don't break) and avoids an
 * O(N^2) re-perception every frame.
 *
 * Connectivity comes from ONE of two sources:
 *  - an explicit bond list (from RDKit's real perceived connectivity, passed for
 *    SMILES/drawn molecules) — rendered exactly, with double/triple bonds as
 *    parallel offset cylinders; OR
 *  - if no list is given (the baked PRESET molecules), a distance-cutoff
 *    perception from the initial geometry (single bonds only).
 * Explicit bonds are the fix for distance-perception artefacts (dangling H's,
 * H's with two bonds, spurious rings, a C=O drawn as C-OH) on imperfect seeds.
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
/** Perpendicular separation of the parallel cylinders in a double bond. */
const DOUBLE_OFFSET = 0.13;
/** Perpendicular separation of the two outer cylinders in a triple bond. */
const TRIPLE_OFFSET = 0.19;

/** Explicit chemical bond (0-indexed atoms) with a bond order for rendering. */
export interface ExplicitBond {
  i: number;
  j: number;
  /** 1 single, 2 double, 3 triple, 4 aromatic (rendered as a single stick). */
  order: number;
}

/**
 * One rendered cylinder. A single bond is one cylinder (offset 0); a double is
 * two (offsets ±DOUBLE_OFFSET); a triple is three (±TRIPLE_OFFSET and 0). The
 * offset shifts the cylinder perpendicular to the bond axis so the parallel
 * sticks read as a multiple bond.
 */
interface BondCyl {
  i: number;
  j: number;
  offset: number;
}

const Y_AXIS = new THREE.Vector3(0, 1, 0);

/**
 * A live molecule view. `update(positions)` re-places every atom sphere and
 * bond cylinder from a flat [x0,y0,z0,...] buffer (Float32Array or number[]).
 */
export class MoleculeView {
  readonly group = new THREE.Group();
  /** Per-atom sphere meshes, index-aligned with `symbols` (for raycasting). */
  readonly atomMeshes: THREE.Mesh[] = [];
  /** One mesh per rendered cylinder, index-aligned with `bondCyls`. Public so
   * the ray tracer can register them as dynamic (re-baked when atoms move). */
  readonly bondMeshes: THREE.Mesh[] = [];
  private readonly bondCyls: BondCyl[] = [];
  private readonly symbols: string[];
  private readonly tmpMid = new THREE.Vector3();
  private readonly tmpDir = new THREE.Vector3();
  private readonly tmpPerp = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
  private readonly a = new THREE.Vector3();
  private readonly b = new THREE.Vector3();

  /**
   * @param explicitBonds real connectivity to render exactly (SMILES/drawn
   *   molecules). Omit for the baked presets to fall back to distance perception.
   */
  constructor(
    symbols: string[],
    positions: ArrayLike<number>,
    explicitBonds?: readonly ExplicitBond[],
  ) {
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
      // Tagged so headless harnesses can count atom spheres in the scene graph
      // (and tell them apart from bond cylinders, markers, and orbital lobes).
      mesh.name = "atom";
      this.atomMeshes.push(mesh);
      this.group.add(mesh);
    }

    const bondMat = materialFor(0x9aa0aa);
    // One or more cylinders per bond depending on order (double -> 2, triple ->
    // 3, single/aromatic -> 1). Offsets separate the parallel sticks.
    const addBond = (i: number, j: number, order: number): void => {
      const offsets =
        order === 2
          ? [-DOUBLE_OFFSET, DOUBLE_OFFSET]
          : order === 3
            ? [-TRIPLE_OFFSET, 0, TRIPLE_OFFSET]
            : [0]; // single or aromatic -> one clean stick
      for (const offset of offsets) {
        this.bondCyls.push({ i, j, offset });
        const bm = new THREE.Mesh(cylGeo, bondMat);
        bm.name = "bond";
        this.bondMeshes.push(bm);
      }
    };

    if (explicitBonds && explicitBonds.length > 0) {
      // Render RDKit's real connectivity exactly — no distance perception.
      for (const b of explicitBonds) addBond(b.i, b.j, b.order);
    } else {
      // No explicit list (presets): perceive single bonds once by distance.
      const n = symbols.length;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = positions[3 * i]! - positions[3 * j]!;
          const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
          const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const cutoff =
            (COVALENT[symbols[i]!] ?? 0.7) + (COVALENT[symbols[j]!] ?? 0.7) + BOND_SLACK;
          if (d <= cutoff) addBond(i, j, 1);
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
    for (let k = 0; k < this.bondCyls.length; k++) {
      const { i, j, offset } = this.bondCyls[k]!;
      this.a.set(positions[3 * i]!, positions[3 * i + 1]!, positions[3 * i + 2]!);
      this.b.set(positions[3 * j]!, positions[3 * j + 1]!, positions[3 * j + 2]!);
      const bond = this.bondMeshes[k]!;
      this.tmpMid.addVectors(this.a, this.b).multiplyScalar(0.5);
      const len = this.a.distanceTo(this.b);
      bond.scale.set(BOND_RADIUS, len, BOND_RADIUS);
      this.tmpDir.subVectors(this.b, this.a).normalize();
      this.tmpQuat.setFromUnitVectors(Y_AXIS, this.tmpDir);
      bond.quaternion.copy(this.tmpQuat);
      if (offset !== 0) {
        // Perpendicular to the bond axis (world-up × dir), for the parallel
        // offset of a double/triple bond. Guard against a bond nearly along Y.
        this.tmpPerp.copy(Y_AXIS);
        if (Math.abs(this.tmpDir.y) > 0.9) this.tmpPerp.set(1, 0, 0);
        this.tmpPerp.cross(this.tmpDir).normalize();
        bond.position.copy(this.tmpMid).addScaledVector(this.tmpPerp, offset);
      } else {
        bond.position.copy(this.tmpMid);
      }
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
