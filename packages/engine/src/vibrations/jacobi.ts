/**
 * Pure-TypeScript classical (cyclic) Jacobi eigensolver for real symmetric
 * matrices. Robust and dependency-free -- fine for the small 3N x 3N Hessians
 * (N <= ~20 here). Returns eigenvalues ascending with matching eigenvectors.
 *
 * This is the single, shared implementation used by the engine's vibrational
 * analysis AND (via re-import) by the accuracy bench, which used to carry its
 * own copy in packages/bench/src/accuracy/linalg.ts.
 */
export interface Eigen {
  values: number[]; // ascending
  vectors: number[][]; // vectors[k] is the eigenvector for values[k]
}

/** `a` is a flat row-major n x n symmetric matrix (mutated internally on a copy). */
export function jacobiEigen(a: Float64Array, n: number, maxSweeps = 100): Eigen {
  const A = Float64Array.from(a);
  // eigenvector matrix V (column j = j-th eigenvector), start as identity
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  const offDiagNorm = (): number => {
    let s = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) s += A[p * n + q]! * A[p * n + q]!;
    return Math.sqrt(s);
  };

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (offDiagNorm() < 1e-14) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p * n + p]!;
        const aqq = A[q * n + q]!;
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        // Apply rotation J^T A J
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p]!;
          const akq = A[k * n + q]!;
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k]!;
          const aqk = A[q * n + k]!;
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        // Accumulate into V
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p]!;
          const vkq = V[k * n + q]!;
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const idx = Array.from({ length: n }, (_, i) => i);
  const evals = idx.map((i) => A[i * n + i]!);
  idx.sort((x, y) => evals[x]! - evals[y]!);

  const values = idx.map((i) => evals[i]!);
  const vectors = idx.map((i) => {
    const v: number[] = [];
    for (let k = 0; k < n; k++) v.push(V[k * n + i]!);
    return v;
  });
  return { values, vectors };
}
