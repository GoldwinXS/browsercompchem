/**
 * Per-element ANI-2x MLPs (Linear + CELU) evaluated in pure float64 TypeScript,
 * with a hand-written backward pass for dE/dAEV.
 *
 * Why pure TS rather than an ONNX gradient graph: the spike proved
 * aten::elu_backward cannot be lowered to opset-17, blocking an autograd export.
 * The nets are tiny (<=4 Linear layers) and CELU'(z,alpha)=z>0?1:exp(z/alpha) is
 * elementary, so implementing forward+backward directly is exact, avoids the ORT
 * node/web split, and makes per-atom species gather + single-member/fp16 variants
 * trivial. Each atom runs only its own element's net (gather), summed/averaged
 * over the ensemble members.
 */

/** One dense layer: y = W x + b, with W stored row-major [out*in]. */
interface Layer {
  W: Float64Array;
  b: Float64Array;
  inDim: number;
  outDim: number;
}

/** An element network = hidden Linear+CELU layers followed by a linear scalar head. */
type Net = Layer[]; // last layer is the linear head (outDim 1), no activation

export interface EnsembleTopology {
  symbols: string[];
  celuAlpha: number;
  /** element symbol -> per-layer [out, in] dims; last entry is the scalar head. */
  elements: Record<string, number[][]>;
  aevDim: number;
}

function decodeF16(u: Uint16Array, i: number): number {
  const h = u[i]!;
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  let val: number;
  if (e === 0) {
    val = f * Math.pow(2, -24);
  } else if (e === 0x1f) {
    val = f ? NaN : Infinity;
  } else {
    val = (1 + f / 1024) * Math.pow(2, e - 15);
  }
  return s ? -val : val;
}

/**
 * The ensemble of per-element nets. Loads a flat weight blob laid out as
 *   for member: for symbol(H..Cl): for layer: W[out*in] then b[out]
 * (see spike/ani2x-onnx/export_weights.py). Runs atoms through only their own
 * element net and averages over the requested number of members.
 */
export class Ensemble {
  readonly members: number;
  readonly celuAlpha: number;
  private readonly aevDim: number;
  private readonly symbolIndex: Map<string, number>;
  // nets[member][elementIdx] = Net
  private readonly nets: Net[][];
  private readonly maxWidth: number;

  constructor(
    topo: EnsembleTopology,
    weights: Float32Array | Uint16Array,
    dtype: "f32" | "f16",
    requestedMembers: number,
    availableMembers: number,
  ) {
    this.celuAlpha = topo.celuAlpha;
    this.aevDim = topo.aevDim;
    this.members = Math.min(requestedMembers, availableMembers);
    this.symbolIndex = new Map(topo.symbols.map((s, i) => [s, i]));

    const read = (i: number): number =>
      dtype === "f16"
        ? decodeF16(weights as Uint16Array, i)
        : (weights as Float32Array)[i]!;

    let ptr = 0;
    let maxWidth = 0;
    this.nets = [];
    for (let mi = 0; mi < availableMembers; mi++) {
      const memberNets: Net[] = [];
      for (const sym of topo.symbols) {
        const dims = topo.elements[sym]!;
        const net: Net = [];
        for (const [outDim, inDim] of dims) {
          const W = new Float64Array(outDim! * inDim!);
          for (let k = 0; k < W.length; k++) W[k] = read(ptr++);
          const b = new Float64Array(outDim!);
          for (let k = 0; k < b.length; k++) b[k] = read(ptr++);
          net.push({ W, b, inDim: inDim!, outDim: outDim! });
          if (outDim! > maxWidth) maxWidth = outDim!;
        }
        memberNets.push(net);
      }
      this.nets.push(memberNets);
    }
    this.maxWidth = maxWidth;
  }

  elementIndex(symbol: string): number {
    const idx = this.symbolIndex.get(symbol);
    if (idx === undefined) throw new Error(`ANI-2x: unsupported element ${symbol}`);
    return idx;
  }

  /**
   * Energy (pre-self-energy, ensemble-averaged) and dE/dAEV for all atoms.
   * `aev` is [N*aevDim]; `species` are element indices. `gOut` [N*aevDim] is
   * filled with dE/dAEV. Returns the summed NN energy.
   */
  energyAndGrad(
    species: Int32Array,
    aev: Float64Array,
    gOut: Float64Array,
  ): number {
    const N = species.length;
    const D = this.aevDim;
    gOut.fill(0);
    const alpha = this.celuAlpha;
    const invAlpha = 1 / alpha;
    const M = this.members;

    // scratch buffers sized to the widest layer
    const act: Float64Array[] = [];
    const deriv: Float64Array[] = [];
    for (let l = 0; l < 8; l++) {
      act.push(new Float64Array(this.maxWidth));
      deriv.push(new Float64Array(this.maxWidth));
    }
    const dTmpA = new Float64Array(this.maxWidth);
    const dTmpB = new Float64Array(this.maxWidth);

    let energy = 0;
    for (let i = 0; i < N; i++) {
      const el = species[i]!;
      const aevOff = i * D;
      for (let mi = 0; mi < M; mi++) {
        const net = this.nets[mi]![el]!;
        const nLayers = net.length;
        // ---- forward, caching activations + celu derivatives per hidden layer ----
        let input: Float64Array = aev.subarray(aevOff, aevOff + D);
        let inDim = D;
        for (let l = 0; l < nLayers - 1; l++) {
          const layer = net[l]!;
          const out = act[l]!;
          const der = deriv[l]!;
          const W = layer.W;
          const b = layer.b;
          const od = layer.outDim;
          for (let o = 0; o < od; o++) {
            let s = b[o]!;
            const wRow = o * inDim;
            for (let k = 0; k < inDim; k++) s += W[wRow + k]! * input[k]!;
            // CELU
            if (s > 0) {
              out[o] = s;
              der[o] = 1;
            } else {
              const ex = Math.exp(s * invAlpha);
              out[o] = alpha * (ex - 1);
              der[o] = ex;
            }
          }
          input = out;
          inDim = od;
        }
        // ---- linear head (outDim 1) ----
        const head = net[nLayers - 1]!;
        let e = head.b[0]!;
        for (let k = 0; k < inDim; k++) e += head.W[k]! * input[k]!;
        energy += e;

        // ---- backward: dE/dAEV ----
        // da for last hidden = head.W (since head out=1)
        let da: Float64Array = dTmpA;
        for (let k = 0; k < inDim; k++) da[k] = head.W[k]!;
        let other: Float64Array = dTmpB;
        for (let l = nLayers - 2; l >= 0; l--) {
          const layer = net[l]!;
          const der = deriv[l]!;
          const W = layer.W;
          const od = layer.outDim;
          const id = layer.inDim;
          // dz = da * celu'(z)
          // dx[k] = sum_o W[o*id+k] * dz[o]
          if (l === 0) {
            // propagate straight into gOut for this atom
            const gRow = aevOff;
            for (let k = 0; k < id; k++) {
              let acc = 0;
              for (let o = 0; o < od; o++) {
                acc += W[o * id + k]! * (da[o]! * der[o]!);
              }
              gOut[gRow + k] = gOut[gRow + k]! + acc;
            }
          } else {
            const dx = other;
            for (let k = 0; k < id; k++) {
              let acc = 0;
              for (let o = 0; o < od; o++) {
                acc += W[o * id + k]! * (da[o]! * der[o]!);
              }
              dx[k] = acc;
            }
            // swap buffers
            const tmp = da;
            da = dx;
            other = tmp;
          }
        }
      }
    }

    // ensemble average
    if (M > 1) {
      const inv = 1 / M;
      energy *= inv;
      for (let k = 0; k < gOut.length; k++) gOut[k] = gOut[k]! * inv;
    }
    return energy;
  }
}
