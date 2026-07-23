/**
 * Regression guards for the ANI-2x accuracy suite. These assert the suite RUNS
 * end-to-end in Node (no Python; committed weights) and that each property's
 * error sits in a SANE range -- they are guardrails against a regression in the
 * engine or the harness, NOT hard chemistry pass/fail gates.
 *
 * A fast subset is used (a couple of molecules per property) so the whole file
 * stays well under the vitest timeout while still exercising every code path:
 * FIRE relaxation, dihedral construction, finite-difference Hessian, the Jacobi
 * eigensolver, and the atomization diagnostic.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { getProvider } from "../src/accuracy/provider.js";
import { evaluateConformers } from "../src/accuracy/conformers.js";
import { evaluateFrequencies } from "../src/accuracy/frequencies.js";
import { evaluateHeats } from "../src/accuracy/heats.js";
import { jacobiEigen } from "@browser-comp-chem/engine";
import type { Ani2xProvider } from "@browser-comp-chem/engine";

let provider: Ani2xProvider;
beforeAll(async () => {
  provider = await getProvider();
}, 60000);

describe("Jacobi eigensolver", () => {
  it("diagonalizes a known symmetric 2x2", () => {
    // [[2,1],[1,2]] -> eigenvalues 1 and 3
    const { values } = jacobiEigen(Float64Array.from([2, 1, 1, 2]), 2);
    expect(values[0]!).toBeCloseTo(1, 8);
    expect(values[1]!).toBeCloseTo(3, 8);
  });
});

describe("conformer relative energies", () => {
  it("butane anti/gauche: correct sign and within a sane range", async () => {
    const r = await evaluateConformers(provider, { ids: ["butane_gauche_anti"] });
    const butane = r.results.find((x) => x.id === "butane_gauche_anti")!;
    expect(butane.status).toBe("evaluated");
    // gauche must be ABOVE anti (positive), close to the 0.611 kcal/mol reference.
    expect(butane.predictedKcal!).toBeGreaterThan(0.2);
    expect(butane.predictedKcal!).toBeLessThan(1.0);
    expect(butane.absErrorKcal!).toBeLessThan(1.5);
    expect(butane.stationary).toBe(true);
  }, 120000);
});

describe("vibrational frequencies", () => {
  it("water harmonic freqs: scaled mean %err < 10%", async () => {
    const r = await evaluateFrequencies(provider, { ids: ["water"] });
    const water = r.results.find((x) => x.id === "water")!;
    expect(water.status).toBe("evaluated");
    expect(water.nModes).toBe(3);
    // harmonic > fundamental, so raw error should be a few %, scaled smaller.
    expect(water.meanPctScaled!).toBeLessThan(10);
    expect(water.meanPctRaw!).toBeGreaterThan(water.meanPctScaled!);
    // near-zero trans/rot modes were actually near zero.
    expect(water.maxResidualTransRot!).toBeLessThan(60);
  }, 120000);
});

describe("heats of formation", () => {
  it("runs the atomization diagnostic and reports honestly (not fabricated)", async () => {
    const r = await evaluateHeats(provider, { ids: ["water"] });
    const water = r.results.find((x) => x.id === "water")!;
    // Not evaluated (no free-atom energies supplied), but diagnostics present.
    expect(r.evaluated).toBe(false);
    expect(water.status).toBe("not_evaluated");
    expect(water.zpeKJ!).toBeGreaterThan(30); // water ZPE ~55 kJ/mol
    expect(water.zpeKJ!).toBeLessThan(80);
    expect(typeof water.deSaeKJ).toBe("number");
  }, 120000);

  it("full cycle produces a sane number when free-atom energies are supplied", async () => {
    // Illustrative wB97X-scale free-atom energies (Hartree). NOT a curated
    // reference; only here to prove dHfFromAtomization is wired correctly end to
    // end. The assertion range is deliberately loose.
    const freeAtomEnergies = { H: -0.4993, O: -75.0456, C: -37.7817, N: -54.5011 };
    const r = await evaluateHeats(provider, { ids: ["water"], freeAtomEnergies });
    const water = r.results.find((x) => x.id === "water")!;
    expect(r.evaluated).toBe(true);
    expect(water.status).toBe("evaluated");
    expect(Number.isFinite(water.predictedKJ!)).toBe(true);
    // Loose sanity: an atomization-route dHf for water should land within a few
    // hundred kJ/mol of the experimental -241.8 with plausible free-atom values.
    expect(Math.abs(water.predictedKJ!)).toBeLessThan(600);
  }, 120000);
});
