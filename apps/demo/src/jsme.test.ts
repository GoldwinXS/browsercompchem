import { describe, it, expect } from "vitest";
import { sanitizeSketchSmiles } from "./jsme.js";

describe("sanitizeSketchSmiles", () => {
  it("passes a normal single-molecule SMILES through unchanged", () => {
    expect(sanitizeSketchSmiles("CC(=O)Oc1ccccc1C(=O)O")).toBe(
      "CC(=O)Oc1ccccc1C(=O)O",
    );
  });

  it("trims surrounding whitespace", () => {
    expect(sanitizeSketchSmiles("  CCO\n")).toBe("CCO");
  });

  it("rejects an empty canvas", () => {
    expect(() => sanitizeSketchSmiles("")).toThrow(/empty/i);
    expect(() => sanitizeSketchSmiles("   ")).toThrow(/empty/i);
  });

  it("rejects reaction SMILES (contains '>')", () => {
    expect(() => sanitizeSketchSmiles("CCO>>CC=O")).toThrow(/reaction/i);
  });

  it("rejects multi-fragment / salt SMILES (contains '.')", () => {
    expect(() => sanitizeSketchSmiles("[Na+].[Cl-]")).toThrow(/fragment/i);
  });
});
