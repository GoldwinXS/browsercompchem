/**
 * JSME (Peter Ertl's JavaScript Molecular Editor) bridge.
 *
 * JSME is a self-contained, framework-free 2D structure sketcher compiled from
 * Java with GWT. It is NOT an ES module: the entry `jsme.nocache.js` is a GWT
 * bootstrap that, once its permutation finishes loading, invokes a *global*
 * callback `window.jsmeOnLoad()` and exposes the `JSApplet.JSME` constructor.
 *
 * The GWT bootstrap resolves every sibling asset (`*.cache.js`, the chrome CSS,
 * `clear.cache.gif`, …) *relative to the URL of the `jsme.nocache.js` script
 * tag*. We therefore serve the whole `jsme-editor` package dir at a single
 * stable path `/jsme/*` via a dev middleware in vite.config.ts (mirroring the
 * existing `/rdkit/*` and `/models/*` middlewares — no binaries copied into the
 * repo). Loading `/jsme/jsme.nocache.js` makes GWT compute its base as `/jsme/`
 * and fetch `/jsme/<hash>.cache.js` etc., all of which the middleware answers.
 *
 * This module wraps the non-module global-callback loading in a Promise and
 * gives the rest of the (strict-TS) app a small typed surface.
 */

/** The applet handle we actually use. JSME exposes far more; we type the bits we call. */
export interface JsmeApplet {
  /** Current structure as a SMILES string ("" for an empty canvas). */
  smiles(): string;
  /**
   * Load a structure from a generic textual input — JSME auto-detects SMILES vs
   * MOL/SDF. This is how we seed the canvas (and how the headless round-trip
   * test injects a known structure, since we cannot mouse-draw).
   */
  readGenericMolecularInput(input: string): void;
  /** Empty the canvas. */
  reset(): void;
}

interface JSAppletNamespace {
  // The container MUST be passed as a DOM id STRING: GWT resolves it by id and
  // attaches its widget tree there. Passing the element object instead leaves
  // the applet un-attached (its molecular area never builds, and reads throw a
  // null-internal-area error) — verified empirically in this codebase.
  JSME: new (
    containerId: string,
    width: string,
    height: string,
    options?: Record<string, string>,
  ) => JsmeApplet;
}

declare global {
  interface Window {
    JSApplet?: JSAppletNamespace;
    jsmeOnLoad?: () => void;
  }
}

/** Served by the serveJsme() dev middleware (see vite.config.ts). */
const JSME_URL = `${import.meta.env.BASE_URL}jsme/jsme.nocache.js`;

let jsmePromise: Promise<JSAppletNamespace> | undefined;

/**
 * Lazily load JSME. Resolves with the `JSApplet` namespace once GWT fires its
 * global `jsmeOnLoad` callback (and the constructor is genuinely present). The
 * heavy script (~1.2 MB permutation) is only fetched the first time the user
 * opens the sketcher.
 */
export function loadJsme(): Promise<JSAppletNamespace> {
  if (jsmePromise) return jsmePromise;
  jsmePromise = new Promise<JSAppletNamespace>((resolve, reject) => {
    // Already present (e.g. HMR re-import)? Use it.
    if (window.JSApplet?.JSME) {
      resolve(window.JSApplet);
      return;
    }
    const timeout = window.setTimeout(() => {
      reject(new Error("JSME load timed out (jsmeOnLoad never fired)"));
    }, 30000);

    window.jsmeOnLoad = () => {
      window.clearTimeout(timeout);
      if (window.JSApplet?.JSME) resolve(window.JSApplet);
      else reject(new Error("jsmeOnLoad fired but JSApplet.JSME is missing"));
    };

    const s = document.createElement("script");
    s.src = JSME_URL;
    s.onerror = () => {
      window.clearTimeout(timeout);
      reject(new Error(`failed to load ${JSME_URL}`));
    };
    document.head.appendChild(s);
  });
  return jsmePromise;
}

/**
 * Construct a JSME applet inside the element with DOM id `containerId`. The
 * element must already exist and be laid out (non-zero size) when this is
 * called. Width/height are CSS pixel strings (GWT wants strings like "360px").
 * Returns the applet handle.
 */
export async function createSketcher(
  containerId: string,
  opts: { width?: string; height?: string; options?: string } = {},
): Promise<JsmeApplet> {
  const JSApplet = await loadJsme();
  const width = opts.width ?? "360px";
  const height = opts.height ?? "320px";
  // JSME option string is comma-separated; a dark-friendly, uncluttered set.
  const optionString = opts.options ?? "newLook,noMarvinJSSketcherWarning";
  return new JSApplet.JSME(containerId, width, height, { options: optionString });
}

/**
 * Clean/validate a SMILES string read out of the sketcher before it enters the
 * ANI-2x pipeline. Pure (no DOM / no JSME) so it is unit-testable.
 *
 * JSME returns "" for an empty canvas, and can emit multi-component SMILES
 * (salts / mixtures, dot-separated) or — if a reaction were drawn — a ">"-laden
 * reaction SMILES. The downstream RDKit path wants a single connected molecule,
 * so we surface those cases as friendly errors rather than passing junk on.
 *
 * @throws Error with a user-facing message when the sketch is empty, a reaction,
 *         or multi-fragment.
 */
export function sanitizeSketchSmiles(raw: string): string {
  const s = raw.trim();
  if (!s) throw new Error("the canvas is empty — draw a structure first");
  if (s.includes(">")) {
    throw new Error("reaction SMILES aren't supported — draw a single molecule");
  }
  if (s.includes(".")) {
    throw new Error(
      "multiple disconnected fragments — draw one connected molecule",
    );
  }
  return s;
}
