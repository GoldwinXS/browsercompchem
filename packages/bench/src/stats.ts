/** Basic error-distribution statistics shared by every property the runner scores. */
export interface ErrorStatsSummary {
  n: number;
  mae: number;
  rmse: number;
  max: number;
}

/** `errors` are signed (predicted - reference) in the property's native unit. */
export function computeErrorStats(errors: number[]): ErrorStatsSummary {
  const n = errors.length;
  if (n === 0) {
    return { n: 0, mae: NaN, rmse: NaN, max: NaN };
  }

  let sumAbs = 0;
  let sumSq = 0;
  let max = 0;

  for (const e of errors) {
    const abs = Math.abs(e);
    sumAbs += abs;
    sumSq += e * e;
    if (abs > max) max = abs;
  }

  return {
    n,
    mae: sumAbs / n,
    rmse: Math.sqrt(sumSq / n),
    max,
  };
}
