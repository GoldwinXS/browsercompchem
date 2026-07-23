/**
 * Core geometry representation shared by every compute tier (RDKit-JS
 * instant tier, ONNX ML-potential tier, Hueckel/HF orbital tier).
 *
 * Positions are stored as a flat Float64Array of length 3*N (x0,y0,z0,
 * x1,y1,z1, ...) rather than an array of Vec3 objects so that the same
 * buffer can be handed directly to onnxruntime-web tensors or WASM
 * linear memory without a marshalling copy.
 *
 * Units: positions in Angstrom unless a function's docstring says
 * otherwise (e.g. providers that work in Bohr should convert at their
 * boundary and document it).
 */
export interface Molecule {
  /** Element symbols, one per atom, e.g. ["O", "H", "H"]. Length N. */
  symbols: string[];

  /** Flat Cartesian coordinates, length 3*N, Angstrom. */
  positions: Float64Array;

  /** Net molecular charge, in units of e. */
  charge: number;

  /** Spin multiplicity (2S+1). 1 = closed-shell singlet. */
  multiplicity: number;
}

/** Number of atoms implied by a Molecule's symbols/positions arrays. */
export function atomCount(mol: Molecule): number {
  return mol.symbols.length;
}

/**
 * Euclidean distance between atoms i and j (Angstrom), reading directly
 * out of the flat positions buffer.
 */
export function distance(positions: Float64Array, i: number, j: number): number {
  const dx = positions[3 * i]! - positions[3 * j]!;
  const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
  const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Deep-copy a Molecule's mutable positions buffer (symbols are shared, they're immutable in practice). */
export function cloneMolecule(mol: Molecule): Molecule {
  return {
    symbols: mol.symbols,
    positions: Float64Array.from(mol.positions),
    charge: mol.charge,
    multiplicity: mol.multiplicity,
  };
}
