/**
 * ANI-2x model container: loads the manifest (AEV params + self energies +
 * topology + variant list) and a weight blob, wiring together the float64 AEV
 * kernel and the pure-TS ensemble. Environment-agnostic asset loading: node
 * (vitest) reads from the filesystem, the browser fetches from a URL.
 */
import { AevComputer, type AevParams } from "./aev.js";
import { Ensemble, type EnsembleTopology } from "./mlp.js";

interface VariantSpec {
  name: string;
  file: string;
  dtype: "f32" | "f16";
  members: number;
  count: number;
  bytes: number;
}

interface Manifest {
  family: string;
  symbols: string[];
  self_energies: number[];
  aev_dim: number;
  radial_len: number;
  angular_len: number;
  num_species: number;
  num_species_pairs: number;
  celu_alpha: number;
  elements: Record<string, number[][]>; // sym -> per-layer [out, in]
  radial: { eta: number; shifts: number[]; cutoff: number };
  angular: {
    eta: number;
    zeta: number;
    shifts: number[];
    sections: number[];
    cutoff: number;
  };
  variants: VariantSpec[];
}

export interface Ani2xLoadOptions {
  /** Directory (filesystem path in node, or URL base in browser) with manifest.json + weight bins. */
  modelDir: string;
  /** Variant name from the manifest (default "full-f32"). */
  variant?: string;
  /** Override number of ensemble members to actually run (<= variant members). */
  members?: number;
}

const isNode =
  typeof process !== "undefined" &&
  !!(process as unknown as { versions?: { node?: string } }).versions?.node;

function joinPath(dir: string, file: string): string {
  return dir.endsWith("/") ? dir + file : dir + "/" + file;
}

async function readBytes(dir: string, file: string): Promise<ArrayBuffer> {
  const target = joinPath(dir, file);
  if (isNode && !/^https?:\/\//.test(dir)) {
    const { readFile } = await import("node:fs/promises");
    const buf = await readFile(target);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  }
  const res = await fetch(target);
  if (!res.ok) throw new Error(`ANI-2x: failed to fetch ${target}: ${res.status}`);
  return res.arrayBuffer();
}

async function readJson<T>(dir: string, file: string): Promise<T> {
  if (isNode && !/^https?:\/\//.test(dir)) {
    const { readFile } = await import("node:fs/promises");
    return JSON.parse(await readFile(joinPath(dir, file), "utf8")) as T;
  }
  const res = await fetch(joinPath(dir, file));
  if (!res.ok) throw new Error(`ANI-2x: failed to fetch manifest: ${res.status}`);
  return (await res.json()) as T;
}

export class Ani2xModel {
  readonly symbols: string[];
  readonly variant: string;
  readonly members: number;
  private readonly aevc: AevComputer;
  private readonly ensemble: Ensemble;
  private readonly selfEnergies: Float64Array;
  private readonly symbolToSpecies: Map<string, number>;

  private constructor(
    manifest: Manifest,
    aevc: AevComputer,
    ensemble: Ensemble,
    variant: string,
  ) {
    this.symbols = manifest.symbols;
    this.variant = variant;
    this.members = ensemble.members;
    this.aevc = aevc;
    this.ensemble = ensemble;
    this.selfEnergies = Float64Array.from(manifest.self_energies);
    this.symbolToSpecies = new Map(manifest.symbols.map((s, i) => [s, i]));
  }

  static async load(opts: Ani2xLoadOptions): Promise<Ani2xModel> {
    const manifest = await readJson<Manifest>(opts.modelDir, "manifest.json");
    const variantName = opts.variant ?? "full-f32";
    const variant = manifest.variants.find((v) => v.name === variantName);
    if (!variant) {
      throw new Error(
        `ANI-2x: variant "${variantName}" not in manifest (have: ${manifest.variants
          .map((v) => v.name)
          .join(", ")})`,
      );
    }

    const aevParams: AevParams = {
      radialLen: manifest.radial_len,
      angularLen: manifest.angular_len,
      numSpecies: manifest.num_species,
      numSpeciesPairs: manifest.num_species_pairs,
      radial: manifest.radial,
      angular: manifest.angular,
    };
    const aevc = new AevComputer(aevParams);

    const topo: EnsembleTopology = {
      symbols: manifest.symbols,
      celuAlpha: manifest.celu_alpha,
      aevDim: manifest.aev_dim,
      elements: manifest.elements,
    };

    const buf = await readBytes(opts.modelDir, variant.file);
    const weights =
      variant.dtype === "f16" ? new Uint16Array(buf) : new Float32Array(buf);
    const requested = opts.members ?? variant.members;
    const ensemble = new Ensemble(
      topo,
      weights,
      variant.dtype,
      requested,
      variant.members,
    );

    return new Ani2xModel(manifest, aevc, ensemble, variantName);
  }

  speciesOf(symbols: string[]): Int32Array {
    const sp = new Int32Array(symbols.length);
    for (let i = 0; i < symbols.length; i++) {
      const idx = this.symbolToSpecies.get(symbols[i]!);
      if (idx === undefined) throw new Error(`ANI-2x: unsupported element ${symbols[i]}`);
      sp[i] = idx;
    }
    return sp;
  }

  get aevDim(): number {
    return this.aevc.aevDim;
  }

  /**
   * Total energy (Hartree) and forces (Hartree/Angstrom, -dE/dr, flat [N*3]) at
   * the given geometry (coords flat [N*3], Angstrom). `species` from speciesOf().
   */
  energyForces(
    species: Int32Array,
    coords: Float64Array,
  ): { energy: number; forces: Float64Array } {
    const N = species.length;
    const D = this.aevc.aevDim;
    const aev = this.aevc.forward(species, coords);
    const g = new Float64Array(N * D);
    let energy = this.ensemble.energyAndGrad(species, aev, g);
    // add self energies
    for (let i = 0; i < N; i++) energy += this.selfEnergies[species[i]!]!;

    const gradCoords = new Float64Array(N * 3);
    this.aevc.adjoint(species, coords, g, gradCoords);
    const forces = new Float64Array(N * 3);
    for (let k = 0; k < forces.length; k++) forces[k] = -gradCoords[k]!;
    return { energy, forces };
  }

  /** Energy only (Hartree). Cheaper: skips the AEV adjoint. */
  energy(species: Int32Array, coords: Float64Array): number {
    const N = species.length;
    const D = this.aevc.aevDim;
    const aev = this.aevc.forward(species, coords);
    const g = new Float64Array(N * D);
    let e = this.ensemble.energyAndGrad(species, aev, g);
    for (let i = 0; i < N; i++) e += this.selfEnergies[species[i]!]!;
    return e;
  }
}
