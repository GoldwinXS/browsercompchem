/**
 * Analytic two-center overlap integrals over Slater-type orbitals (STOs), for
 * the s/p valence basis the EHT tier uses.
 *
 * Method: the classic auxiliary-function (Mulliken-Rieke-Orloff-Orloff)
 * formulation. Put the two atoms on the z-axis (the "diatomic frame"), atom A at
 * the origin and atom B a distance R away along +z. In prolate spheroidal
 * (elliptical) coordinates
 *      xi  = (r_A + r_B) / R  in [1, inf),
 *      eta = (r_A - r_B) / R  in [-1, 1],
 *      phi in [0, 2pi),
 * every STO product times the volume element reduces to a polynomial in (xi,
 * eta) multiplied by exp(-p*xi - q*eta), with
 *      p = (zeta_A + zeta_B) * R / 2,   q = (zeta_A - zeta_B) * R / 2.
 * The integral of xi^j eta^k exp(-p*xi - q*eta) over the domain is exactly
 * A_j(p) * B_k(q), where
 *      A_k(p) = integral_1^inf   xi^k  e^(-p xi)  dxi,
 *      B_k(q) = integral_{-1}^1  eta^k e^(-q eta) deta.
 * So each diatomic-frame overlap is a finite sum  sum_jk c_jk A_j(p) B_k(q).
 *
 * Only same-|m| pairs survive the phi integral, so in the diatomic frame there
 * are just four non-vanishing fundamentals: (s,s), (s,p_sigma), (p_sigma,s),
 * (p_sigma,p_sigma) [all m=0], and (p_pi,p_pi) [m=+-1]. Overlaps in the real
 * molecular frame between global Cartesian p orbitals are then assembled from
 * these via the Slater-Koster direction-cosine formulas (Slater & Koster,
 * Phys. Rev. 94, 1498 (1954)).
 *
 * Validated against the textbook value S(1s,1s; zeta=1.0, R=0.74 A) ~ 0.75 and
 * the R->0 (->1) / R->inf (->0) limits in overlap.test.ts.
 */
import type { OrbitalType } from "./parameters.js";

/**
 * A_k(p) = integral_1^inf xi^k e^(-p xi) dxi, for k = 0..kMax.
 * Upward recurrence from A_0 = e^-p / p, A_k = (e^-p + k A_{k-1}) / p. Stable for
 * p > 0 (always true here: p = (zeta_A+zeta_B) R / 2 > 0 for R > 0).
 */
export function aArray(p: number, kMax: number): Float64Array {
  const a = new Float64Array(kMax + 1);
  const e = Math.exp(-p);
  a[0] = e / p;
  for (let k = 1; k <= kMax; k++) a[k] = (e + k * a[k - 1]!) / p;
  return a;
}

/**
 * B_k(q) = integral_{-1}^1 eta^k e^(-q eta) deta, for k = 0..kMax.
 * For q == 0 this is 2/(k+1) for even k and 0 for odd k. For q != 0 the upward
 * recurrence B_0 = 2 sinh(q)/q, B_k = ((-1)^k e^q - e^-q + k B_{k-1}) / q holds,
 * but it loses precision as q -> 0 (subtractive cancellation), so below a small
 * threshold we fall back to a direct series of the integral. q == 0 exactly for
 * homonuclear same-exponent pairs (e.g. C-C in benzene, H-H in H2), which is
 * common, so the small-q branch matters.
 */
export function bArray(q: number, kMax: number): Float64Array {
  const b = new Float64Array(kMax + 1);
  if (Math.abs(q) < 1e-7) {
    // Series: integral_{-1}^1 eta^k e^(-q eta) deta = sum_m (-q)^m / m! * I_{k+m},
    // where I_n = integral_{-1}^1 eta^n deta = 2/(n+1) for even n, 0 for odd n.
    for (let k = 0; k <= kMax; k++) {
      let sum = 0;
      let term = 1; // (-q)^m / m!
      for (let m = 0; m < 24; m++) {
        const n = k + m;
        if ((n & 1) === 0) sum += term * (2 / (n + 1));
        term *= -q / (m + 1);
        if (Math.abs(term) < 1e-18) break;
      }
      b[k] = sum;
    }
    return b;
  }
  const eq = Math.exp(q);
  const emq = Math.exp(-q);
  b[0] = (eq - emq) / q;
  let sign = 1; // (-1)^k
  for (let k = 1; k <= kMax; k++) {
    sign = -sign;
    b[k] = (sign * eq - emq + k * b[k - 1]!) / q;
  }
  return b;
}

// ---------------------------------------------------------------------------
// Tiny bivariate-polynomial helper (coefficients keyed by xi-power, eta-power).
// The fundamentals are products of small binomials, so a sparse Map is plenty.
// ---------------------------------------------------------------------------
type Poly = Map<number, number>; // key = xiPow * 32 + etaPow  (powers < 32)

const key = (j: number, k: number): number => j * 32 + k;

/** (xi + s*eta)^m expanded via the binomial theorem (s = +1 or -1). */
function binomial(m: number, s: number): Poly {
  const p: Poly = new Map();
  let c = 1; // C(m,i)
  for (let i = 0; i <= m; i++) {
    // term: c * xi^(m-i) * (s eta)^i
    p.set(key(m - i, i), c * Math.pow(s, i));
    c = (c * (m - i)) / (i + 1);
  }
  return p;
}

/** Multiply two sparse bivariate polynomials. */
function polyMul(a: Poly, b: Poly): Poly {
  const out: Poly = new Map();
  for (const [ka, va] of a) {
    const ja = Math.floor(ka / 32);
    const ea = ka % 32;
    for (const [kb, vb] of b) {
      const jb = Math.floor(kb / 32);
      const eb = kb % 32;
      const kk = key(ja + jb, ea + eb);
      out.set(kk, (out.get(kk) ?? 0) + va * vb);
    }
  }
  return out;
}

/** Small explicit polynomial from a list of [xiPow, etaPow, coeff] triples. */
function polyFromTerms(terms: [number, number, number][]): Poly {
  const p: Poly = new Map();
  for (const [j, k, c] of terms) p.set(key(j, k), (p.get(key(j, k)) ?? 0) + c);
  return p;
}

/** Integrate sum c_jk xi^j eta^k e^(-p xi - q eta) over the elliptical domain. */
function integratePoly(poly: Poly, p: number, q: number): number {
  let jMax = 0;
  let kMax = 0;
  for (const kk of poly.keys()) {
    jMax = Math.max(jMax, Math.floor(kk / 32));
    kMax = Math.max(kMax, kk % 32);
  }
  const A = aArray(p, jMax);
  const B = bArray(q, kMax);
  let sum = 0;
  for (const [kk, c] of poly) {
    const j = Math.floor(kk / 32);
    const k = kk % 32;
    sum += c * A[j]! * B[k]!;
  }
  return sum;
}

/** Normalized-STO radial prefactor N = (2 zeta)^(n + 1/2) / sqrt((2n)!). */
export function radialNorm(n: number, zeta: number): number {
  return Math.pow(2 * zeta, n + 0.5) / Math.sqrt(factorial(2 * n));
}

function factorial(n: number): number {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

/**
 * Diatomic-frame SIGMA fundamental for an ordered pair: orbital A (angular
 * momentum lA in {0=s,1=p_sigma}) at the origin, orbital B (lB) a distance R
 * (bohr) away along +z. Both p_sigma point along the global +z axis (a fixed,
 * self-consistent convention shared with the grid evaluator; the eigenproblem
 * is invariant to per-orbital sign conventions).
 */
export function sigmaFundamental(
  nA: number,
  zA: number,
  lA: 0 | 1,
  nB: number,
  zB: number,
  lB: 0 | 1,
  R: number,
): number {
  const p = ((zA + zB) * R) / 2;
  const q = ((zA - zB) * R) / 2;
  const half = R / 2;

  // Angular/normalization prefactor (folds in the real-harmonic normalization
  // constants 1/sqrt(4pi) [s] or sqrt(3/4pi) [p] and the phi integral 2pi).
  let pref: number;
  let poly: Poly;
  if (lA === 0 && lB === 0) {
    pref = 0.5; // (1/4pi)*2pi
    poly = polyMul(binomial(nA, +1), binomial(nB, -1));
  } else if (lA === 0 && lB === 1) {
    pref = Math.sqrt(3) / 2;
    // (xi+eta)^nA (xi-eta)^(nB-1) (xi*eta - 1)
    poly = polyMul(
      polyMul(binomial(nA, +1), binomial(nB - 1, -1)),
      polyFromTerms([
        [1, 1, 1],
        [0, 0, -1],
      ]),
    );
  } else if (lA === 1 && lB === 0) {
    pref = Math.sqrt(3) / 2;
    // (xi+eta)^(nA-1) (xi-eta)^nB (1 + xi*eta)
    poly = polyMul(
      polyMul(binomial(nA - 1, +1), binomial(nB, -1)),
      polyFromTerms([
        [0, 0, 1],
        [1, 1, 1],
      ]),
    );
  } else {
    pref = 1.5; // (3/4pi)*2pi
    // (xi+eta)^(nA-1) (xi-eta)^(nB-1) (xi^2 eta^2 - 1)
    poly = polyMul(
      polyMul(binomial(nA - 1, +1), binomial(nB - 1, -1)),
      polyFromTerms([
        [2, 2, 1],
        [0, 0, -1],
      ]),
    );
  }

  const scale =
    radialNorm(nA, zA) * radialNorm(nB, zB) * pref * Math.pow(half, nA + nB + 1);
  return scale * integratePoly(poly, p, q);
}

/**
 * Diatomic-frame PI fundamental for a p_pi(A)-p_pi(B) pair (A at origin, B at
 * +z distance R bohr). Uses the cos^2(phi) phi-integral (= pi) rather than 2pi.
 */
export function piFundamental(
  nA: number,
  zA: number,
  nB: number,
  zB: number,
  R: number,
): number {
  const p = ((zA + zB) * R) / 2;
  const q = ((zA - zB) * R) / 2;
  const half = R / 2;
  // pref = (3/4pi) * pi  = 3/4
  const pref = 0.75;
  // (xi+eta)^(nA-1) (xi-eta)^(nB-1) (xi^2 - 1)(1 - eta^2)
  const poly = polyMul(
    polyMul(binomial(nA - 1, +1), binomial(nB - 1, -1)),
    polyFromTerms([
      [2, 0, 1],
      [0, 0, -1],
      [2, 2, -1],
      [0, 2, 1],
    ]),
  );
  const scale =
    radialNorm(nA, zA) * radialNorm(nB, zB) * pref * Math.pow(half, nA + nB + 1);
  return scale * integratePoly(poly, p, q);
}

/**
 * Convenience: the 1s-1s overlap (used by the overlap unit test and as a
 * sanity check). zeta in bohr^-1, R in bohr.
 */
export function overlap1s1s(zA: number, zB: number, R: number): number {
  return sigmaFundamental(1, zA, 0, 1, zB, 0, R);
}
