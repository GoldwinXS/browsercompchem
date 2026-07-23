/**
 * Marching-cubes tests.
 *
 * Two properties are checked:
 *   - a sphere-like scalar field yields a CLOSED manifold mesh (every edge is
 *     shared by exactly two triangles) with a sane triangle count; and
 *   - a p-orbital-like field (z * gaussian) produces disjoint +/- lobes when
 *     iso-surfaced at +iso and -iso.
 */
import { describe, it, expect } from "vitest";
import { marchingCubes, type MarchingGrid } from "../src/isosurface/marchingCubes.js";

/** Sample a scalar function f(x,y,z) onto a regular grid. */
function sampleField(
  f: (x: number, y: number, z: number) => number,
  grid: MarchingGrid,
): Float32Array {
  const [nx, ny, nz] = grid.dims;
  const [ox, oy, oz] = grid.origin;
  const [sx, sy, sz] = grid.spacing;
  const out = new Float32Array(nx * ny * nz);
  for (let iz = 0; iz < nz; iz++)
    for (let iy = 0; iy < ny; iy++)
      for (let ix = 0; ix < nx; ix++)
        out[ix + nx * (iy + ny * iz)] = f(ox + ix * sx, oy + iy * sy, oz + iz * sz);
  return out;
}

/** Count how many triangles reference each undirected edge. */
function edgeManifoldCounts(indices: Uint32Array): Map<string, number> {
  const counts = new Map<string, number>();
  const bump = (a: number, b: number): void => {
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  };
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
    bump(a, b);
    bump(b, c);
    bump(c, a);
  }
  return counts;
}

describe("marching cubes", () => {
  it("iso-surfaces a sphere into a closed manifold with a sane triangle count", () => {
    const N = 32;
    const span = 4; // -2 .. 2
    const grid: MarchingGrid = {
      dims: [N, N, N],
      origin: [-2, -2, -2],
      spacing: [span / (N - 1), span / (N - 1), span / (N - 1)],
    };
    // f = radius^2 - r^2 : positive inside a sphere of radius 1.2, well inside
    // the grid (so the surface never runs off the boundary -> closed).
    const radius = 1.2;
    const field = sampleField((x, y, z) => radius * radius - (x * x + y * y + z * z), grid);
    const mesh = marchingCubes(field, grid, 0);

    expect(mesh.triangleCount).toBeGreaterThan(200);
    expect(mesh.triangleCount).toBeLessThan(6000);

    // Closed manifold: every edge shared by exactly two triangles.
    const counts = edgeManifoldCounts(mesh.indices);
    let nonManifold = 0;
    for (const c of counts.values()) if (c !== 2) nonManifold++;
    expect(nonManifold).toBe(0);

    // All vertices sit ~on the sphere of radius 1.2.
    for (let v = 0; v < mesh.positions.length; v += 3) {
      const r = Math.hypot(mesh.positions[v]!, mesh.positions[v + 1]!, mesh.positions[v + 2]!);
      expect(Math.abs(r - radius)).toBeLessThan(0.12);
    }
  });

  it("splits a p-orbital-like field into disjoint +/- lobes", () => {
    const N = 40;
    const grid: MarchingGrid = {
      dims: [N, N, N],
      origin: [-3, -3, -3],
      spacing: [6 / (N - 1), 6 / (N - 1), 6 / (N - 1)],
    };
    // pz-like: z * exp(-(r^2)) -> positive lobe for z>0, negative for z<0.
    const field = sampleField((x, y, z) => z * Math.exp(-(x * x + y * y + z * z) / 2), grid);
    const iso = 0.05;
    const pos = marchingCubes(field, grid, iso);
    const neg = marchingCubes(field, grid, -iso);

    expect(pos.triangleCount).toBeGreaterThan(0);
    expect(neg.triangleCount).toBeGreaterThan(0);

    // The positive lobe lives entirely at z > 0, the negative at z < 0.
    const zRange = (m: { positions: Float32Array }): [number, number] => {
      let lo = Infinity, hi = -Infinity;
      for (let v = 2; v < m.positions.length; v += 3) {
        lo = Math.min(lo, m.positions[v]!);
        hi = Math.max(hi, m.positions[v]!);
      }
      return [lo, hi];
    };
    const [posLo] = zRange(pos);
    const [, negHi] = zRange(neg);
    expect(posLo).toBeGreaterThan(0); // positive lobe strictly above the z=0 plane
    expect(negHi).toBeLessThan(0); // negative lobe strictly below
  });
});
