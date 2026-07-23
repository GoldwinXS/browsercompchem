import { describe, it, expect } from "vitest";
import { distance, angle } from "./measure.js";

describe("measure.distance", () => {
  it("computes a unit distance along an axis", () => {
    const p = [0, 0, 0, 1, 0, 0];
    expect(distance(p, 0, 1)).toBeCloseTo(1, 10);
  });
  it("computes a 3-4-5 diagonal", () => {
    const p = [0, 0, 0, 3, 4, 0];
    expect(distance(p, 0, 1)).toBeCloseTo(5, 10);
  });
  it("computes a 3D space diagonal", () => {
    const p = [0, 0, 0, 1, 2, 2];
    expect(distance(p, 0, 1)).toBeCloseTo(3, 10);
  });
});

describe("measure.angle", () => {
  it("returns 90 degrees for a right angle at the vertex", () => {
    // i=(1,0,0), j=(0,0,0) vertex, k=(0,1,0)
    const p = [1, 0, 0, 0, 0, 0, 0, 1, 0];
    expect(angle(p, 0, 1, 2)).toBeCloseTo(90, 10);
  });
  it("returns 180 degrees for a straight (collinear) arrangement", () => {
    const p = [-1, 0, 0, 0, 0, 0, 1, 0, 0];
    expect(angle(p, 0, 1, 2)).toBeCloseTo(180, 10);
  });
  it("returns ~109.47 degrees for ideal tetrahedral vectors", () => {
    // two tetrahedral bond directions from origin
    const p = [1, 1, 1, 0, 0, 0, 1, -1, -1];
    expect(angle(p, 0, 1, 2)).toBeCloseTo(109.4712206, 5);
  });
  it("is independent of vertex ordering symmetry (i<->k)", () => {
    const p = [1, 0, 0, 0, 0, 0, 0.3, 0.9, 0];
    expect(angle(p, 0, 1, 2)).toBeCloseTo(angle(p, 2, 1, 0), 10);
  });
});
