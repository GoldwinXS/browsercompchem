import { describe, expect, it } from "vitest";
import { LennardJonesProvider } from "../src/potentials/lennardJones.js";
import { FireOptimizer } from "../src/optimize/fire.js";
import { distance, type Molecule } from "../src/geometry/molecule.js";

/**
 * These tests exist to make the optimizer *real* from day one: FIRE is
 * the one fully-implemented optimizer in this package, and the
 * Lennard-Jones provider is the one fully-implemented potential, so
 * together they give a small energy surface with textbook-known
 * minima to converge to and check against.
 */

function ljMolecule(positions: number[]): Molecule {
  const n = positions.length / 3;
  return {
    symbols: Array(n).fill("Ar"),
    positions: Float64Array.from(positions),
    charge: 0,
    multiplicity: 1,
  };
}

describe("FIRE + Lennard-Jones: dimer", () => {
  it("relaxes a 2-atom LJ cluster to r = 2^(1/6) * sigma", async () => {
    const sigma = 1.0;
    const epsilon = 1.0;
    const provider = new LennardJonesProvider({ sigma, epsilon });
    const optimizer = new FireOptimizer();

    // start well off the minimum (both too close and too far are worth checking)
    const r0 = 1.5 * sigma;
    const mol = ljMolecule([0, 0, 0, r0, 0, 0]);

    const result = await optimizer.optimize(mol, provider, { forceTolerance: 1e-10 });

    expect(result.converged).toBe(true);

    const rFinal = distance(result.molecule.positions, 0, 1);
    const rExpected = provider.equilibriumDistance; // sigma * 2^(1/6)

    expect(rFinal).toBeCloseTo(rExpected, 6);
    // energy at the minimum of an isolated LJ pair is exactly -epsilon
    expect(result.energy).toBeCloseTo(-epsilon, 6);
  });

  it("converges to the same minimum starting from a compressed geometry", async () => {
    const provider = new LennardJonesProvider({ sigma: 1, epsilon: 1 });
    const optimizer = new FireOptimizer();

    const r0 = 0.9; // inside the repulsive wall
    const mol = ljMolecule([0, 0, 0, r0, 0, 0]);

    const result = await optimizer.optimize(mol, provider, { forceTolerance: 1e-10 });

    expect(result.converged).toBe(true);
    const rFinal = distance(result.molecule.positions, 0, 1);
    expect(rFinal).toBeCloseTo(provider.equilibriumDistance, 6);
  });
});

describe("FIRE + Lennard-Jones: LJ7 cluster", () => {
  it("relaxes a pentagonal-bipyramid 7-atom cluster near the literature global minimum", async () => {
    const sigma = 1.0;
    const epsilon = 1.0;
    const provider = new LennardJonesProvider({ sigma, epsilon });
    const optimizer = new FireOptimizer();

    const b = provider.equilibriumDistance; // sigma * 2^(1/6)

    // Build an "all edges = b" pentagonal bipyramid as the initial guess:
    // 5 atoms on a regular pentagon (z=0) + 2 apex atoms on +/-z.
    // For a regular pentagon inscribed in circle radius R, the side
    // length is s = 2*R*sin(pi/5); setting s = b gives R below. The
    // apex height h is then chosen so the apex-to-vertex distance is
    // also b: h = sqrt(b^2 - R^2). This is a well-known good starting
    // point for LJ7 (a fragment of an icosahedron) that FIRE relaxes
    // into the true (slightly distorted) global minimum.
    const R = b / (2 * Math.sin(Math.PI / 5));
    const h = Math.sqrt(b * b - R * R);

    const positions: number[] = [];
    for (let k = 0; k < 5; k++) {
      const theta = (2 * Math.PI * k) / 5;
      positions.push(R * Math.cos(theta), R * Math.sin(theta), 0);
    }
    positions.push(0, 0, h);
    positions.push(0, 0, -h);

    const mol = ljMolecule(positions);

    const result = await optimizer.optimize(mol, provider, {
      forceTolerance: 1e-8,
      maxSteps: 5000,
    });

    expect(result.converged).toBe(true);

    // Reference: Cambridge Cluster Database (Wales & Doye), LJ7 global
    // minimum energy in reduced units (epsilon=sigma=1) is
    // E = -16.505384 epsilon. See http://doye.chem.ox.ac.uk/jon/structures/LJ.html
    const referenceEnergy = -16.505384;
    expect(result.energy).toBeCloseTo(referenceEnergy, 2);

    // sanity: relaxation should only ever lower the energy from the
    // (already near-optimal) starting guess, never raise it.
    const firstEnergy = result.history[0]!.energy;
    expect(result.energy).toBeLessThanOrEqual(firstEnergy + 1e-9);
  });
});
