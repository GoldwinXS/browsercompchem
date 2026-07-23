/**
 * XYZ parsing, distance-based bond perception, and a controlled dihedral-set
 * operation. The dihedral operation is what lets us build the SECOND conformer
 * of each pair (e.g. n-butane gauche from anti) since bench-data ships only one
 * geometry per conformer system; we rotate a single well-defined torsion and
 * then relax with ANI, exactly as the bench-data caveat recommends.
 */
import { readFileSync } from "node:fs";
import type { Molecule } from "@browser-comp-chem/engine";
import { COVALENT_RADIUS } from "./chem.js";

export interface ParsedXyz {
  symbols: string[];
  positions: Float64Array; // flat 3N, Angstrom
  comment: string;
}

export function parseXyz(text: string): ParsedXyz {
  const lines = text.split(/\r?\n/);
  const n = parseInt(lines[0]!.trim(), 10);
  const comment = (lines[1] ?? "").trim();
  const symbols: string[] = [];
  const positions = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const parts = lines[2 + i]!.trim().split(/\s+/);
    symbols.push(parts[0]!);
    positions[3 * i] = parseFloat(parts[1]!);
    positions[3 * i + 1] = parseFloat(parts[2]!);
    positions[3 * i + 2] = parseFloat(parts[3]!);
  }
  return { symbols, positions, comment };
}

export function loadXyz(path: string): ParsedXyz {
  return parseXyz(readFileSync(path, "utf8"));
}

export function toMolecule(p: ParsedXyz, charge = 0, multiplicity = 1): Molecule {
  return {
    symbols: p.symbols,
    positions: Float64Array.from(p.positions),
    charge,
    multiplicity,
  };
}

/** Adjacency list from distance-based bond perception (bond if d < 1.3*(r_i+r_j)). */
export function bondGraph(symbols: string[], pos: Float64Array): number[][] {
  const n = symbols.length;
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = pos[3 * i]! - pos[3 * j]!;
      const dy = pos[3 * i + 1]! - pos[3 * j + 1]!;
      const dz = pos[3 * i + 2]! - pos[3 * j + 2]!;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ri = COVALENT_RADIUS[symbols[i]!] ?? 0.7;
      const rj = COVALENT_RADIUS[symbols[j]!] ?? 0.7;
      if (d < 1.3 * (ri + rj)) {
        adj[i]!.push(j);
        adj[j]!.push(i);
      }
    }
  }
  return adj;
}

function vec(pos: Float64Array, i: number): [number, number, number] {
  return [pos[3 * i]!, pos[3 * i + 1]!, pos[3 * i + 2]!];
}
function sub(a: number[], b: number[]): number[] {
  return [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
}
function cross(a: number[], b: number[]): number[] {
  return [
    a[1]! * b[2]! - a[2]! * b[1]!,
    a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!,
  ];
}
function dot(a: number[], b: number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}
function normalize(a: number[]): number[] {
  const n = Math.sqrt(dot(a, a));
  return [a[0]! / n, a[1]! / n, a[2]! / n];
}

/** Dihedral angle i-j-k-l in radians (IUPAC/atan2 convention). */
export function dihedral(pos: Float64Array, i: number, j: number, k: number, l: number): number {
  const b1 = sub(vec(pos, j), vec(pos, i));
  const b2 = sub(vec(pos, k), vec(pos, j));
  const b3 = sub(vec(pos, l), vec(pos, k));
  const n1 = cross(b1, b2);
  const n2 = cross(b2, b3);
  const m1 = cross(n1, normalize(b2));
  const x = dot(n1, n2);
  const y = dot(m1, n2);
  return Math.atan2(y, x);
}

/** Atoms reachable from `start` without traversing through `blocked` (edge-blocked node). */
function fragmentFrom(adj: number[][], start: number, blocked: number): Set<number> {
  const seen = new Set<number>([start]);
  const stack = [start];
  while (stack.length) {
    const u = stack.pop()!;
    for (const v of adj[u]!) {
      if (v === blocked || seen.has(v)) continue;
      seen.add(v);
      stack.push(v);
    }
  }
  return seen;
}

function rotateAboutAxis(
  p: number[],
  axisPoint: number[],
  axisUnit: number[],
  angle: number,
): number[] {
  const rel = sub(p, axisPoint);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // Rodrigues' rotation formula
  const kdotr = dot(axisUnit, rel);
  const kxr = cross(axisUnit, rel);
  const out = [
    rel[0]! * c + kxr[0]! * s + axisUnit[0]! * kdotr * (1 - c),
    rel[1]! * c + kxr[1]! * s + axisUnit[1]! * kdotr * (1 - c),
    rel[2]! * c + kxr[2]! * s + axisUnit[2]! * kdotr * (1 - c),
  ];
  return [out[0]! + axisPoint[0]!, out[1]! + axisPoint[1]!, out[2]! + axisPoint[2]!];
}

/**
 * Return a copy of `pos` with the i-j-k-l dihedral set to `targetDeg`, achieved
 * by rigidly rotating the fragment on the k/l side of the j-k bond about that
 * bond axis. The j-i side (and atoms j, i) are held fixed.
 */
export function setDihedral(
  symbols: string[],
  pos: Float64Array,
  i: number,
  j: number,
  k: number,
  l: number,
  targetDeg: number,
): Float64Array {
  const adj = bondGraph(symbols, pos);
  const current = dihedral(pos, i, j, k, l);
  // A right-hand rotation about the j->k axis DECREASES the i-j-k-l dihedral
  // (opposite sense to the atan2 convention used in dihedral()), so rotate by
  // -(target - current) to actually reach `targetDeg`. Verified empirically.
  const delta = -((targetDeg * Math.PI) / 180 - current);
  const moving = fragmentFrom(adj, k, j); // k/l side, blocked from crossing back to j
  const axisPoint = vec(pos, j);
  const axisUnit = normalize(sub(vec(pos, k), vec(pos, j)));
  const out = Float64Array.from(pos);
  for (const a of moving) {
    if (a === j) continue; // safety; j is the pivot side, never in `moving`
    const rotated = rotateAboutAxis(vec(pos, a), axisPoint, axisUnit, delta);
    out[3 * a] = rotated[0]!;
    out[3 * a + 1] = rotated[1]!;
    out[3 * a + 2] = rotated[2]!;
  }
  return out;
}
