import { describe, it, expect } from "vitest";
import { parseMolblock } from "./rdkit.js";

/**
 * The bond-block parser is the load-bearing part of the topology fix: it turns
 * RDKit's V2000 molblock into the real connectivity we render (instead of
 * distance-perceiving bonds). These tests feed known molblocks directly — no
 * RDKit wasm required — and assert atoms, bonds, and bond orders.
 */

// 3-methyl-2-butanone (CC(=O)C(C)C) with explicit H's, straight from RDKit's
// add_hs(). The carbonyl is bond "2 3 2" (atom2 C = atom3 O, order 2).
const KETONE_MOLBLOCK = `
     RDKit          2D

 16 15  0  0  0  0  0  0  0  0999 V2000
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1828   -1.4888    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5981    1.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0490   -0.5490    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981    1.5000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0490   -0.5490    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5490    2.0490    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5490    2.0490    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.8971    2.2500    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8481    2.7990    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1311    0.7926    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7990   -1.8481    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8644   -0.2030    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8511   -1.4518    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  1  0
  4  6  1  0
  1  7  1  0
  1  8  1  0
  1  9  1  0
  4 10  1  0
  5 11  1  0
  5 12  1  0
  5 13  1  0
  6 14  1  0
  6 15  1  0
  6 16  1  0
M  END
`;

describe("parseMolblock", () => {
  it("parses atom count, symbols and coords", () => {
    const { symbols, positions } = parseMolblock(KETONE_MOLBLOCK);
    expect(symbols.length).toBe(16);
    expect(positions.length).toBe(48);
    // C5 O1 H10
    expect(symbols.filter((s) => s === "C").length).toBe(5);
    expect(symbols.filter((s) => s === "O").length).toBe(1);
    expect(symbols.filter((s) => s === "H").length).toBe(10);
    // atom 3 is the carbonyl O
    expect(symbols[2]).toBe("O");
    expect(positions[6]).toBeCloseTo(0.1828, 3); // x of atom 3
    expect(positions[7]).toBeCloseTo(-1.4888, 3); // y of atom 3
  });

  it("parses the bond block with correct 0-indexed atoms and orders", () => {
    const { bonds } = parseMolblock(KETONE_MOLBLOCK);
    expect(bonds.length).toBe(15);
    // First bond "1 2 1" -> atoms 0,1 single.
    expect(bonds[0]).toEqual({ i: 0, j: 1, order: 1 });
    // Second bond "2 3 2" -> the carbonyl C=O (atoms 1,2 order 2 DOUBLE).
    expect(bonds[1]).toEqual({ i: 1, j: 2, order: 2 });
    // Exactly one double bond in this molecule.
    expect(bonds.filter((b) => b.order === 2).length).toBe(1);
    // The carbonyl O (atom index 2) appears in exactly one bond and it is the
    // C=O double bond — never an O-H (that was the distance-perception bug).
    const oBonds = bonds.filter((b) => b.i === 2 || b.j === 2);
    expect(oBonds.length).toBe(1);
    expect(oBonds[0]!.order).toBe(2);
  });

  it("gives every hydrogen exactly one bond", () => {
    const { symbols, bonds } = parseMolblock(KETONE_MOLBLOCK);
    const degree = new Array<number>(symbols.length).fill(0);
    for (const b of bonds) {
      degree[b.i] = degree[b.i]! + 1;
      degree[b.j] = degree[b.j]! + 1;
    }
    symbols.forEach((s, idx) => {
      if (s === "H") expect(degree[idx]).toBe(1);
    });
  });

  it("implies zero rings for an acyclic molecule (bonds = atoms - 1)", () => {
    const { symbols, bonds } = parseMolblock(KETONE_MOLBLOCK);
    // For a single connected component, rings = bonds - atoms + 1.
    expect(bonds.length - symbols.length + 1).toBe(0);
  });

  it("handles a triple bond order (hydrogen cyanide molblock)", () => {
    const hcn = `
     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
  1  3  1  0
M  END
`;
    const { symbols, bonds } = parseMolblock(hcn);
    expect(symbols).toEqual(["C", "N", "H"]);
    expect(bonds).toEqual([
      { i: 0, j: 1, order: 3 },
      { i: 0, j: 2, order: 1 },
    ]);
  });
});
