/**
 * Simulated IR spectrum: orchestrates the dipole + per-mode intensity pieces
 * (dipole.ts, irIntensities.ts) into the { modes, curve, dipole } bundle the
 * UI plots. Pure function over plain data -- no DOM, no worker-specific code,
 * so it is unit-testable directly and reusable from any host (worker, node
 * test, future CLI).
 *
 * `modes`/`frequencies` are NOT recomputed here -- pass in exactly what
 * vibrations/normalModes.ts's computeNormalModes() returned (this module has
 * no Hessian code of its own; see irIntensities.ts's docstring for why the
 * reduced-mass bookkeeping only works with THOSE Cartesian-normalized mode
 * vectors specifically).
 *
 * --- Plotting convention ---
 * IR spectra are conventionally plotted with wavenumber DECREASING left to
 * right (a historical convention from prism/grating instruments), and often as
 * %transmittance (T = 100 - normalized-absorbance-ish) rather than absorbance,
 * with strong bands pointing DOWN. This module returns plain absorbance
 * (arbitrary units, 0 at baseline, peaks at the summed relative-intensity
 * Lorentzians) on an ASCENDING wavenumber grid; the UI is responsible for
 * reversing the x-axis (and, if desired, inverting to a transmittance-style
 * downward-pointing display) at render time.
 */
import type { Molecule } from "../geometry/molecule.js";
import { computeDipole, type DipoleResult } from "./dipole.js";
import { computeIRIntensities, type ModeIntensity, type IrIntensityOptions } from "./irIntensities.js";

export interface IRSpectrumOptions extends IrIntensityOptions {
  /** Broadened-curve grid lower bound, cm^-1. Default 400. */
  gridMin?: number;
  /** Broadened-curve grid upper bound, cm^-1. Default 4000. */
  gridMax?: number;
  /** Broadened-curve grid spacing, cm^-1. Default 2. */
  gridStep?: number;
  /** Lorentzian half-width at half-maximum, cm^-1. Default 8. */
  hwhm?: number;
}

export interface IRSpectrumCurve {
  /** Ascending wavenumber grid, cm^-1. */
  wavenumbers: Float64Array;
  /** Summed-Lorentzian absorbance (arbitrary units, index-aligned with wavenumbers). */
  absorbance: Float64Array;
}

export interface IRSpectrumResult {
  modes: ModeIntensity[];
  curve: IRSpectrumCurve;
  dipole: DipoleResult;
}

/**
 * Build the full IR spectrum bundle: per-mode intensities (irIntensities.ts),
 * the equilibrium dipole (dipole.ts), and a Lorentzian-broadened plot curve.
 */
export async function computeIRSpectrum(
  mol: Molecule,
  modes: Float64Array[],
  frequencies: number[],
  options: IRSpectrumOptions = {},
): Promise<IRSpectrumResult> {
  const gridMin = options.gridMin ?? 400;
  const gridMax = options.gridMax ?? 4000;
  const gridStep = options.gridStep ?? 2;
  const hwhm = options.hwhm ?? 8;

  const [modeIntensities, dipole] = await Promise.all([
    computeIRIntensities(mol, modes, frequencies, options),
    computeDipole(mol),
  ]);

  const nPoints = Math.max(2, Math.floor((gridMax - gridMin) / gridStep) + 1);
  const wavenumbers = new Float64Array(nPoints);
  const absorbance = new Float64Array(nPoints);
  const hwhm2 = hwhm * hwhm;
  for (let i = 0; i < nPoints; i++) {
    const v = gridMin + i * gridStep;
    wavenumbers[i] = v;
    let a = 0;
    for (const m of modeIntensities) {
      const dv = v - Math.abs(m.frequency); // an imaginary (negative) mode has no physical IR line
      a += m.relative * (hwhm2 / (dv * dv + hwhm2));
    }
    absorbance[i] = a;
  }

  return { modes: modeIntensities, curve: { wavenumbers, absorbance }, dipole };
}
