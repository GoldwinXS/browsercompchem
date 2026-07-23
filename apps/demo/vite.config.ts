import { defineConfig, type Plugin } from "vite";
import { createReadStream, statSync } from "node:fs";
import { resolve } from "node:path";

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

export default defineConfig({
  root: __dirname,
  plugins: [serveModels()],
  server: {
    host: true, // 0.0.0.0 — reachable over the tailnet
    port: 8142,
    strictPort: true,
  },
  worker: {
    format: "es",
  },
});
