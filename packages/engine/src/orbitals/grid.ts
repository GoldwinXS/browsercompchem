/**
 * Evaluate a molecular orbital's wavefunction psi(r) on a regular 3D grid.
 *
 * psi(r) = sum_ao C[ao, mo] * chi_ao(r), where chi_ao is a NORMALIZED real
 * Cartesian Slater-type orbital with the same normalization used to build the
 * overlap matrix, so the coefficients are directly reusable:
 *
 *   chi_s   = N_r e^(-zeta r) * (1/sqrt(4pi))
 *   chi_p_a = N_r r^(n-1) e^(-zeta r) * sqrt(3/4pi) * (a / r)     (a in {x,y,z})
 *
 * with N_r = (2 zeta)^(n+1/2) / sqrt((2n)!) and r, x, y, z measured from the
 * orbital's atom in BOHR. Input grid coordinates are Angstrom (as everywhere in
 * this codebase); they are converted to bohr internally. The returned field is
 * therefore in atomic units (bohr^(-3/2)), matching an isovalue ~0.05 au.
 *
 * Grid index convention (shared with the marching-cubes module): the flat index
 * of cell (ix, iy, iz) is  ix + nx * (iy + ny * iz)  -- x fastest, z slowest.
 */
import { radialNorm } from "./overlap.js";
import { BOHR_PER_ANGSTROM } from "./parameters.js";
import type { ExtendedHuckelResult } from "./extendedHuckel.js";

const INV_SQRT_4PI = 1 / Math.sqrt(4 * Math.PI);
const SQRT_3_4PI = Math.sqrt(3 / (4 * Math.PI));

/** Regular-grid specification. Origin and spacing are in Angstrom. */
export interface GridSpec {
  /** Corner (minimum) coordinate of the grid, Angstrom [x, y, z]. */
  origin: [number, number, number];
  /** Grid spacing, Angstrom; a scalar (isotropic) or per-axis [sx, sy, sz]. */
  spacing: number | [number, number, number];
  /** Number of sample points per axis [nx, ny, nz]. */
  dims: [number, number, number];
}

/** Precomputed per-AO evaluation record (atom position in bohr + STO data). */
interface AoEval {
  ax: number;
  ay: number;
  az: number;
  n: number;
  zeta: number;
  norm: number; // radial normalization, folded with the angular constant
  type: 0 | 1 | 2 | 3; // 0=s, 1=px, 2=py, 3=pz
  coeff: number; // C[ao, mo] for the selected MO
}

/**
 * Sample orbital `orbitalIndex` of `result` on the grid `spec`, returning a
 * Float32Array of length nx*ny*nz. `onProgress(done, total)` (optional) fires
 * once per completed z-slab so the worker can stream a progress bar.
 */
export function evaluateOrbitalOnGrid(
  result: ExtendedHuckelResult,
  orbitalIndex: number,
  spec: GridSpec,
  onProgress?: (done: number, total: number) => void,
): Float32Array {
  const [nx, ny, nz] = spec.dims;
  const [sx, sy, sz] = typeof spec.spacing === "number"
    ? [spec.spacing, spec.spacing, spec.spacing]
    : spec.spacing;
  // Grid origin/spacing in bohr (STO exponents are bohr^-1).
  const ox = spec.origin[0] * BOHR_PER_ANGSTROM;
  const oy = spec.origin[1] * BOHR_PER_ANGSTROM;
  const oz = spec.origin[2] * BOHR_PER_ANGSTROM;
  const bx = sx * BOHR_PER_ANGSTROM;
  const by = sy * BOHR_PER_ANGSTROM;
  const bz = sz * BOHR_PER_ANGSTROM;

  const nMO = result.nMO;
  const aos = result.aos;
  const pos = result.positionsBohr;

  // Precompute the active AO evaluators (skip AOs with ~zero coefficient).
  const active: AoEval[] = [];
  for (let a = 0; a < aos.length; a++) {
    const coeff = result.coefficients[a * nMO + orbitalIndex]!;
    if (Math.abs(coeff) < 1e-10) continue;
    const ao = aos[a]!;
    const isS = ao.type === "s";
    active.push({
      ax: pos[3 * ao.atomIndex]!,
      ay: pos[3 * ao.atomIndex + 1]!,
      az: pos[3 * ao.atomIndex + 2]!,
      n: ao.n,
      zeta: ao.zeta,
      norm: radialNorm(ao.n, ao.zeta) * (isS ? INV_SQRT_4PI : SQRT_3_4PI),
      type: ao.type === "s" ? 0 : ao.type === "px" ? 1 : ao.type === "py" ? 2 : 3,
      coeff,
    });
  }

  const field = new Float32Array(nx * ny * nz);
  for (let iz = 0; iz < nz; iz++) {
    const z = oz + iz * bz;
    for (let iy = 0; iy < ny; iy++) {
      const y = oy + iy * by;
      const rowBase = nx * (iy + ny * iz);
      for (let ix = 0; ix < nx; ix++) {
        const x = ox + ix * bx;
        let psi = 0;
        for (let k = 0; k < active.length; k++) {
          const e = active[k]!;
          const dx = x - e.ax;
          const dy = y - e.ay;
          const dz = z - e.az;
          const r2 = dx * dx + dy * dy + dz * dz;
          const r = Math.sqrt(r2);
          const radial = e.norm * Math.exp(-e.zeta * r) * (e.n > 1 ? Math.pow(r, e.n - 1) : 1);
          let chi: number;
          if (e.type === 0) {
            chi = radial; // s
          } else if (r < 1e-12) {
            chi = 0; // p vanishes at its own nucleus
          } else {
            const comp = e.type === 1 ? dx : e.type === 2 ? dy : dz;
            chi = (radial * comp) / r; // r^(n-1)*(comp/r)
          }
          psi += e.coeff * chi;
        }
        field[rowBase + ix] = psi;
      }
    }
    if (onProgress) onProgress(iz + 1, nz);
  }
  return field;
}

/**
 * Convenience: an axis-aligned grid that comfortably encloses all atoms, with a
 * uniform spacing and a padding margin (both Angstrom). Dimensions are capped at
 * `maxDim` per axis (spacing is coarsened to fit if necessary) so the evaluation
 * stays interactive.
 */
export function autoGridSpec(
  positionsAngstrom: Float64Array,
  opts: { padding?: number; spacing?: number; maxDim?: number } = {},
): GridSpec {
  const padding = opts.padding ?? 3.0;
  let spacing = opts.spacing ?? 0.3;
  const maxDim = opts.maxDim ?? 80;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const n = positionsAngstrom.length / 3;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, positionsAngstrom[3 * i]!);
    minY = Math.min(minY, positionsAngstrom[3 * i + 1]!);
    minZ = Math.min(minZ, positionsAngstrom[3 * i + 2]!);
    maxX = Math.max(maxX, positionsAngstrom[3 * i]!);
    maxY = Math.max(maxY, positionsAngstrom[3 * i + 1]!);
    maxZ = Math.max(maxZ, positionsAngstrom[3 * i + 2]!);
  }
  const lx = maxX - minX + 2 * padding;
  const ly = maxY - minY + 2 * padding;
  const lz = maxZ - minZ + 2 * padding;
  const longest = Math.max(lx, ly, lz);
  // Coarsen spacing if the longest axis would exceed maxDim samples.
  if (longest / spacing + 1 > maxDim) spacing = longest / (maxDim - 1);
  const dims: [number, number, number] = [
    Math.max(2, Math.ceil(lx / spacing) + 1),
    Math.max(2, Math.ceil(ly / spacing) + 1),
    Math.max(2, Math.ceil(lz / spacing) + 1),
  ];
  return {
    origin: [minX - padding, minY - padding, minZ - padding],
    spacing,
    dims,
  };
}
