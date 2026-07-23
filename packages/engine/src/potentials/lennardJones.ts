import type { Molecule } from "../geometry/molecule.js";
import type { EnergyForceProvider, EnergyForces } from "./types.js";

/**
 * Pairwise Lennard-Jones 12-6 potential:
 *
 *   V(r) = 4*epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
 *
 * This is the one *real, fully implemented* provider in the engine
 * package. Its purpose is to give the optimizer/hessian/bench harnesses
 * a cheap, analytically-understood energy surface (known dimer minimum
 * at r = 2^(1/6) * sigma, well-documented cluster minima) so those
 * pieces can be tested for real from day one, ahead of RDKit-JS/ONNX
 * integration.
 *
 * NOTE ON UNITS: sigma/epsilon here are arbitrary test parameters, not
 * calibrated to any real element. Energy is returned in units of
 * epsilon and forces in units of epsilon/sigma (i.e. "LJ reduced
 * units"), *not* Hartree/Bohr like the rest of the EnergyForceProvider
 * contract. Molecule.symbols are ignored -- every atom is treated as
 * the same LJ particle type. This is a deliberate simplification for a
 * test fixture, not a chemistry model.
 */
export interface LennardJonesParams {
  /** Distance at which the potential is zero (Angstrom-like units). */
  sigma: number;
  /** Depth of the potential well. */
  epsilon: number;
}

export class LennardJonesProvider implements EnergyForceProvider {
  readonly name = "lennard-jones-test";
  readonly sigma: number;
  readonly epsilon: number;

  constructor(params: LennardJonesParams = { sigma: 1, epsilon: 1 }) {
    this.sigma = params.sigma;
    this.epsilon = params.epsilon;
  }

  /** Distance at which two isolated LJ particles sit at their energy minimum. */
  get equilibriumDistance(): number {
    return this.sigma * Math.pow(2, 1 / 6);
  }

  async energyForces(mol: Molecule): Promise<EnergyForces> {
    return Promise.resolve(this.energyForcesSync(mol.positions));
  }

  /** Synchronous core (handy for the optimizer's inner loop and for tests). */
  energyForcesSync(positions: Float64Array): EnergyForces {
    const n = positions.length / 3;
    const forces = new Float64Array(positions.length);
    let energy = 0;

    const { sigma, epsilon } = this;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = positions[3 * i]! - positions[3 * j]!;
        const dy = positions[3 * i + 1]! - positions[3 * j + 1]!;
        const dz = positions[3 * i + 2]! - positions[3 * j + 2]!;
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(r2);

        const sr6 = Math.pow(sigma / r, 6);
        const sr12 = sr6 * sr6;

        energy += 4 * epsilon * (sr12 - sr6);

        // F_i = (24*epsilon/r^2) * (2*sr12 - sr6) * r_vec ; F_j = -F_i
        const coeff = (24 * epsilon * (2 * sr12 - sr6)) / r2;
        const fx = coeff * dx;
        const fy = coeff * dy;
        const fz = coeff * dz;

        forces[3 * i] = forces[3 * i]! + fx;
        forces[3 * i + 1] = forces[3 * i + 1]! + fy;
        forces[3 * i + 2] = forces[3 * i + 2]! + fz;

        forces[3 * j] = forces[3 * j]! - fx;
        forces[3 * j + 1] = forces[3 * j + 1]! - fy;
        forces[3 * j + 2] = forces[3 * j + 2]! - fz;
      }
    }

    return { energy, forces };
  }
}
