// geometry
export type { Molecule } from "./geometry/molecule.js";
export { atomCount, distance, cloneMolecule } from "./geometry/molecule.js";

// potentials
export type { EnergyForceProvider, EnergyForces } from "./potentials/types.js";
export { LennardJonesProvider, type LennardJonesParams } from "./potentials/lennardJones.js";
export {
  Ani2xProvider,
  Ani2xModel,
  type Ani2xLoadOptions,
  AevComputer,
  type AevParams,
} from "./potentials/ani2x/index.js";

// optimize
export type {
  Optimizer,
  OptimizerOptions,
  OptimizeResult,
  OptimizeStep,
} from "./optimize/types.js";
export { FireOptimizer, type FireOptions } from "./optimize/fire.js";
export { BfgsOptimizer, type BfgsOptions } from "./optimize/bfgs.js";

// hessian
export {
  finiteDifferenceHessian,
  type HessianOptions,
  type HessianResult,
} from "./hessian/finiteDifference.js";

// vibrations (harmonic normal-mode analysis)
export {
  computeNormalModes,
  isLinearGeometry,
  jacobiEigen,
  type Eigen,
  type NormalModesOptions,
  type NormalModesResult,
  ATOMIC_MASS,
  VIB_CM_PER_SQRT,
  HARTREE_TO_CM1,
} from "./vibrations/index.js";

// orbitals (extended Hueckel theory tier)
export {
  extendedHuckel,
  buildBasis,
  buildOverlapMatrix,
  type AtomicOrbital,
  type ExtendedHuckelOptions,
  type ExtendedHuckelResult,
} from "./orbitals/extendedHuckel.js";
export {
  EHT_PARAMS,
  EHT_SUPPORTED_ELEMENTS,
  BOHR_PER_ANGSTROM,
  type OrbitalType,
  type ShellParam,
  type ElementParam,
} from "./orbitals/parameters.js";
export {
  aArray,
  bArray,
  radialNorm,
  sigmaFundamental,
  piFundamental,
  overlap1s1s,
} from "./orbitals/overlap.js";
export {
  evaluateOrbitalOnGrid,
  autoGridSpec,
  type GridSpec,
} from "./orbitals/grid.js";
export {
  mullikenCharges,
  orbitalComposition,
  type MullikenResult,
  type AoContribution,
} from "./orbitals/population.js";

// isosurface (marching cubes for orbital lobes)
export {
  marchingCubes,
  gradientNormals,
  type MarchingGrid,
  type IsoMesh,
} from "./isosurface/marchingCubes.js";
