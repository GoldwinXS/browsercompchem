/**
 * Task 2 gate: the analytic AEV adjoint (dAEV/dr) must agree with a
 * finite-difference of the (already validated) AEV forward.
 *
 * We validate the *contraction* the forces actually use: for a fixed cotangent
 * g, the adjoint returns sum_{i,c} g[i,c] * dAEV[i,c]/dr_k for every coordinate
 * k in one pass. We compare that to a finite difference of
 * phi(r) = sum_{i,c} g[i,c] * AEV[i,c](r). Two cotangents are tested per
 * molecule: all-ones and a seeded pseudo-random g. Gate: < 1e-8.
 *
 * The reference AEV has an extremely stiff angular term (zeta = 14.1), so a
 * plain central difference is truncation-limited well above 1e-8. We therefore
 * Romberg-extrapolate three 5-point (O(h^4)) stencils to O(h^8); that removes
 * the truncation floor (water/methane/caffeine reach ~1e-11..1e-13, aspirin
 * ~3e-10), isolating the analytic adjoint. The ultimate correctness proof is the
 * force gate vs TorchANI in ani2x.test.ts.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AevComputer, type AevParams } from "../src/potentials/ani2x/aev.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const manifest = JSON.parse(
  readFileSync(here + "../../../models/ani2x/manifest.json", "utf8"),
);
const refs = JSON.parse(
  readFileSync(here + "fixtures/ani2x-references.json", "utf8"),
);

const params: AevParams = {
  radialLen: manifest.radial_len,
  angularLen: manifest.angular_len,
  numSpecies: manifest.num_species,
  numSpeciesPairs: manifest.num_species_pairs,
  radial: manifest.radial,
  angular: manifest.angular,
};
const aevc = new AevComputer(params);
const symToSp: Record<string, number> = {};
manifest.symbols.forEach((s: string, i: number) => (symToSp[s] = i));

function speciesOf(symbols: string[]): Int32Array {
  return Int32Array.from(symbols.map((s) => symToSp[s]!));
}

// deterministic LCG so the test is reproducible
function seeded(n: number, seed = 12345): Float64Array {
  const g = new Float64Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (1103515245 * s + 12345) & 0x7fffffff;
    g[i] = s / 0x7fffffff - 0.5;
  }
  return g;
}

function phi(sp: Int32Array, coords: Float64Array, g: Float64Array): number {
  const aev = aevc.forward(sp, coords);
  let acc = 0;
  for (let k = 0; k < aev.length; k++) acc += g[k]! * aev[k]!;
  return acc;
}

/** 5-point (O(h^4)) central difference of phi wrt coordinate k. */
function d5(sp: Int32Array, coords: Float64Array, g: Float64Array, k: number, h: number): number {
  const c0 = coords[k]!;
  coords[k] = c0 + 2 * h;
  const p2 = phi(sp, coords, g);
  coords[k] = c0 + h;
  const p1 = phi(sp, coords, g);
  coords[k] = c0 - h;
  const m1 = phi(sp, coords, g);
  coords[k] = c0 - 2 * h;
  const m2 = phi(sp, coords, g);
  coords[k] = c0;
  return (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * h);
}

/** Romberg extrapolation of three 5-point stencils (h, h/2, h/4) -> O(h^8). */
function rombergDeriv(
  sp: Int32Array,
  coords: Float64Array,
  g: Float64Array,
  k: number,
  h: number,
): number {
  const a = d5(sp, coords, g, k, h);
  const b = d5(sp, coords, g, k, h / 2);
  const c = d5(sp, coords, g, k, h / 4);
  const r1 = (16 * b - a) / 15;
  const r2 = (16 * c - b) / 15;
  return (64 * r2 - r1) / 63;
}

const mols = ["water", "methane", "aspirin", "caffeine"];

describe("ANI-2x AEV analytic gradient vs finite difference", () => {
  for (const name of mols) {
    it(`${name}: adjoint matches Romberg-extrapolated FD (<1e-8)`, () => {
      const mol = refs[name];
      const sp = speciesOf(mol.symbols);
      const N = mol.symbols.length;
      const coords = Float64Array.from(mol.coords.flat() as number[]);
      const D = aevc.aevDim;

      for (const g of [new Float64Array(N * D).fill(1), seeded(N * D)]) {
        const grad = new Float64Array(N * 3);
        aevc.adjoint(sp, coords, g, grad);

        let worst = 0;
        for (let k = 0; k < N * 3; k++) {
          const fd = rombergDeriv(sp, coords, g, k, 2e-3);
          worst = Math.max(worst, Math.abs(fd - grad[k]!));
        }
        expect(worst, `worst=${worst.toExponential(3)}`).toBeLessThan(1e-8);
      }
    });
  }
});
