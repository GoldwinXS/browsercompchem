import { defineConfig, type Plugin } from "vite";
import { createReadStream, statSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { createRequire } from "node:module";

/**
 * Serve the repo-root `models/` directory at `/models/*` during dev.
 *
 * The ANI-2x weights live at <repoRoot>/models/ani2x/* (git-tracked, ~90 MB
 * total across variants), OUTSIDE this app. Rather than copy/symlink them into
 * apps/demo/public (which would duplicate large binaries cross-platform), this
 * tiny middleware streams them straight off disk. The worker fetches
 * `${location.origin}/models/ani2x/{manifest.json,weights-full-f16.bin}`.
 */
function serveModels(): Plugin {
  const modelsRoot = resolve(__dirname, "../../models");
  return {
    name: "serve-models",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/models/")) return next();
        const rel = decodeURIComponent(req.url.slice("/models/".length).split("?")[0]!);
        // prevent path traversal
        const filePath = resolve(modelsRoot, rel);
        if (!filePath.startsWith(modelsRoot)) {
          res.statusCode = 403;
          return res.end("forbidden");
        }
        try {
          const st = statSync(filePath);
          if (!st.isFile()) return next();
          res.setHeader(
            "Content-Type",
            filePath.endsWith(".json")
              ? "application/json"
              : "application/octet-stream",
          );
          res.setHeader("Content-Length", String(st.size));
          res.setHeader("Cache-Control", "no-cache");
          createReadStream(filePath).pipe(res);
        } catch {
          return next();
        }
      });
    },
  };
}

/**
 * Serve the `@rdkit/rdkit` MinimalLib dist (RDKit_minimal.{js,wasm}) at
 * `/rdkit/*` during dev.
 *
 * The wasm is ~7 MB; rather than copy it into apps/demo/public (duplicating a
 * large binary that npm already installed) this streams the two files straight
 * out of node_modules, exactly like serveModels() does for the ANI-2x weights.
 * rdkit.ts injects `/rdkit/RDKit_minimal.js` and points locateFile at
 * `/rdkit/RDKit_minimal.wasm`.
 */
function serveRdkit(): Plugin {
  // Resolve the installed package dir robustly (it may be hoisted to the repo
  // root node_modules rather than apps/demo/node_modules).
  const require = createRequire(resolve(__dirname, "vite.config.ts"));
  const distRoot = dirname(require.resolve("@rdkit/rdkit/dist/RDKit_minimal.js"));
  return {
    name: "serve-rdkit",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/rdkit/")) return next();
        const rel = decodeURIComponent(req.url.slice("/rdkit/".length).split("?")[0]!);
        const filePath = resolve(distRoot, rel);
        if (!filePath.startsWith(distRoot)) {
          res.statusCode = 403;
          return res.end("forbidden");
        }
        try {
          const st = statSync(filePath);
          if (!st.isFile()) return next();
          res.setHeader(
            "Content-Type",
            filePath.endsWith(".wasm")
              ? "application/wasm"
              : filePath.endsWith(".js")
                ? "text/javascript"
                : "application/octet-stream",
          );
          res.setHeader("Content-Length", String(st.size));
          res.setHeader("Cache-Control", "no-cache");
          createReadStream(filePath).pipe(res);
        } catch {
          return next();
        }
      });
    },
  };
}

/**
 * Serve the `jsme-editor` package dir at `/jsme/*` during dev.
 *
 * JSME is a GWT app: its `jsme.nocache.js` bootstrap resolves every sibling
 * asset (`*.cache.js`, `gwt/chrome/chrome.css`, `clear.cache.gif`, …) relative
 * to its own script URL. Serving the whole package under a single stable prefix
 * makes that self-consistent (base = `/jsme/`) without copying ~2.5 MB of GWT
 * output into the repo — exactly the pattern serveRdkit()/serveModels() use.
 * jsme.ts injects `/jsme/jsme.nocache.js`.
 */
function serveJsme(): Plugin {
  const require = createRequire(resolve(__dirname, "vite.config.ts"));
  const distRoot = dirname(require.resolve("jsme-editor/jsme.nocache.js"));
  const contentType = (p: string): string => {
    if (p.endsWith(".js")) return "text/javascript";
    if (p.endsWith(".css")) return "text/css";
    if (p.endsWith(".png")) return "image/png";
    if (p.endsWith(".gif")) return "image/gif";
    if (p.endsWith(".html")) return "text/html";
    return "application/octet-stream";
  };
  return {
    name: "serve-jsme",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/jsme/")) return next();
        const rel = decodeURIComponent(req.url.slice("/jsme/".length).split("?")[0]!);
        const filePath = resolve(distRoot, rel);
        if (!filePath.startsWith(distRoot)) {
          res.statusCode = 403;
          return res.end("forbidden");
        }
        try {
          const st = statSync(filePath);
          if (!st.isFile()) return next();
          res.setHeader("Content-Type", contentType(filePath));
          res.setHeader("Content-Length", String(st.size));
          res.setHeader("Cache-Control", "no-cache");
          createReadStream(filePath).pipe(res);
        } catch {
          return next();
        }
      });
    },
  };
}

export default defineConfig({
  root: __dirname,
  plugins: [serveModels(), serveRdkit(), serveJsme()],
  server: {
    host: true, // 0.0.0.0 — reachable over the tailnet
    port: 8142,
    strictPort: true,
  },
  worker: {
    format: "es",
  },
});
