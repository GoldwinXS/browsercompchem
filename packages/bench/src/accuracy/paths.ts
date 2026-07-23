/** Filesystem locations of the model weights and the read-only reference data. */
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url)); // .../packages/bench/src/accuracy/

/** Directory holding manifest.json + weight blobs for ANI-2x. */
export const MODEL_DIR = resolve(here, "../../../../models/ani2x");
/** Repo-root reference-data directory (READ ONLY). */
export const BENCH_DATA_DIR = resolve(here, "../../../../bench-data");
/** Repo root, where ACCURACY_REPORT.md is written (trailing separator included). */
export const REPO_ROOT = resolve(here, "../../../..") + "/";
