/**
 * Pure marching cubes (Lorensen & Cline, "Marching Cubes: A High Resolution 3D
 * Surface Construction Algorithm", SIGGRAPH 1987) over a regular scalar grid.
 *
 * Dependency-free: returns plain typed arrays (positions + triangle indices),
 * which the demo wraps into a THREE.BufferGeometry. The scalar field uses the
 * same index convention as the orbital grid evaluator:
 *     field[ix + nx*(iy + ny*iz)]   -- x fastest, z slowest.
 *
 * Vertices are placed by linear interpolation along each cube edge to the
 * isolevel crossing, and welded by edge-key so the resulting mesh is a proper
 * closed manifold (each interior edge shared by exactly two triangles) when the
 * surface does not run off the grid.
 */
import { EDGE_TABLE, TRI_TABLE } from "./tables.js";

/** Grid geometry for marching cubes. Origin/spacing in whatever unit the caller
 * uses for the emitted vertex positions (the demo uses Angstrom). */
export interface MarchingGrid {
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: [number, number, number];
}

export interface IsoMesh {
  /** Vertex positions, flat [x0,y0,z0, x1,...], length 3*nVertices. */
  positions: Float32Array;
  /** Triangle vertex indices, length 3*nTriangles. */
  indices: Uint32Array;
  /** Number of triangles. */
  triangleCount: number;
}

// Cube corner offsets, matching the table numbering in tables.ts.
const CORNER: readonly [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
// The two corners each edge connects.
const EDGE_ENDS: readonly [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/**
 * Extract the isosurface of `field` at value `isoLevel`. Returns welded
 * positions + triangle indices. An empty mesh (0 triangles) is returned if the
 * surface does not intersect the grid.
 */
export function marchingCubes(
  field: Float32Array | Float64Array,
  grid: MarchingGrid,
  isoLevel: number,
): IsoMesh {
  const [nx, ny, nz] = grid.dims;
  const [ox, oy, oz] = grid.origin;
  const [sx, sy, sz] = grid.spacing;
  const at = (ix: number, iy: number, iz: number): number =>
    field[ix + nx * (iy + ny * iz)]!;

  // Welding: map a per-edge global key -> emitted vertex index.
  const vertexMap = new Map<number, number>();
  const positions: number[] = [];
  const indices: number[] = [];

  // A stable global key for the interpolated vertex on a given cube edge. Each
  // edge is identified by its lower endpoint grid coords + an axis (0/1/2).
  const edgeKey = (ix: number, iy: number, iz: number, axis: number): number =>
    (((ix * (ny + 1) + iy) * (nz + 1) + iz) * 3 + axis);

  const cornerVal = new Float64Array(8);
  const cornerPos: [number, number, number][] = [
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
  ];

  const getVertex = (edge: number, cx: number, cy: number, cz: number): number => {
    const [c0, c1] = EDGE_ENDS[edge]!;
    // Identify the edge globally by its minimum-corner grid coords + axis.
    const p0 = CORNER[c0]!;
    const p1 = CORNER[c1]!;
    const ax0 = cx + p0[0], ay0 = cy + p0[1], az0 = cz + p0[2];
    const ax1 = cx + p1[0], ay1 = cy + p1[1], az1 = cz + p1[2];
    // axis of this edge (the coordinate that differs)
    let axis: number;
    let bx: number, by: number, bz: number; // lower endpoint
    if (ax0 !== ax1) { axis = 0; bx = Math.min(ax0, ax1); by = ay0; bz = az0; }
    else if (ay0 !== ay1) { axis = 1; bx = ax0; by = Math.min(ay0, ay1); bz = az0; }
    else { axis = 2; bx = ax0; by = ay0; bz = Math.min(az0, az1); }
    const kkey = edgeKey(bx, by, bz, axis);
    const existing = vertexMap.get(kkey);
    if (existing !== undefined) return existing;

    const v0 = cornerVal[c0]!;
    const v1 = cornerVal[c1]!;
    let t = (isoLevel - v0) / (v1 - v0);
    if (!Number.isFinite(t)) t = 0.5;
    const g0 = cornerPos[c0]!;
    const g1 = cornerPos[c1]!;
    const idx = positions.length / 3;
    positions.push(
      g0[0] + t * (g1[0] - g0[0]),
      g0[1] + t * (g1[1] - g0[1]),
      g0[2] + t * (g1[2] - g0[2]),
    );
    vertexMap.set(kkey, idx);
    return idx;
  };

  for (let iz = 0; iz < nz - 1; iz++) {
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        // Gather the 8 corner values / world positions.
        let cubeindex = 0;
        for (let c = 0; c < 8; c++) {
          const off = CORNER[c]!;
          const gx = ix + off[0], gy = iy + off[1], gz = iz + off[2];
          const val = at(gx, gy, gz);
          cornerVal[c] = val;
          cornerPos[c] = [ox + gx * sx, oy + gy * sy, oz + gz * sz];
          if (val < isoLevel) cubeindex |= 1 << c;
        }
        const edges = EDGE_TABLE[cubeindex]!;
        if (edges === 0) continue;

        const tris = TRI_TABLE[cubeindex]!;
        for (let t = 0; tris[t] !== -1; t += 3) {
          const a = getVertex(tris[t]!, ix, iy, iz);
          const b = getVertex(tris[t + 1]!, ix, iy, iz);
          const c = getVertex(tris[t + 2]!, ix, iy, iz);
          indices.push(a, b, c);
        }
      }
    }
  }

  return {
    positions: Float32Array.from(positions),
    indices: Uint32Array.from(indices),
    triangleCount: indices.length / 3,
  };
}

/**
 * Per-vertex normals from the scalar-field gradient (central differences),
 * evaluated at each vertex's fractional grid coordinate. `sign` flips the
 * result: for a positive lobe {field > iso} the outward normal points toward
 * decreasing field (sign = -1); for a negative lobe {field < -iso} it points
 * toward increasing field (sign = +1).
 */
export function gradientNormals(
  field: Float32Array | Float64Array,
  grid: MarchingGrid,
  positions: Float32Array,
  sign: number,
): Float32Array {
  const [nx, ny, nz] = grid.dims;
  const [ox, oy, oz] = grid.origin;
  const [sx, sy, sz] = grid.spacing;
  const at = (ix: number, iy: number, iz: number): number =>
    field[ix + nx * (iy + ny * iz)]!;
  const clampI = (v: number, hi: number): number => (v < 0 ? 0 : v > hi ? hi : v);

  const normals = new Float32Array(positions.length);
  const nVerts = positions.length / 3;
  for (let v = 0; v < nVerts; v++) {
    // Fractional grid coordinate of the vertex.
    const fx = (positions[3 * v]! - ox) / sx;
    const fy = (positions[3 * v + 1]! - oy) / sy;
    const fz = (positions[3 * v + 2]! - oz) / sz;
    const ix = clampI(Math.round(fx), nx - 1);
    const iy = clampI(Math.round(fy), ny - 1);
    const iz = clampI(Math.round(fz), nz - 1);
    const gx = at(clampI(ix + 1, nx - 1), iy, iz) - at(clampI(ix - 1, nx - 1), iy, iz);
    const gy = at(ix, clampI(iy + 1, ny - 1), iz) - at(ix, clampI(iy - 1, ny - 1), iz);
    const gz = at(ix, iy, clampI(iz + 1, nz - 1)) - at(ix, iy, clampI(iz - 1, nz - 1));
    let nxv = sign * gx;
    let nyv = sign * gy;
    let nzv = sign * gz;
    const len = Math.hypot(nxv, nyv, nzv) || 1;
    normals[3 * v] = nxv / len;
    normals[3 * v + 1] = nyv / len;
    normals[3 * v + 2] = nzv / len;
  }
  return normals;
}
