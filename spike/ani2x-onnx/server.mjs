// Static server with COOP/COEP (enables SharedArrayBuffer -> multithreaded wasm)
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8137);
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".map": "application/json", ".css": "text/css",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/web/index.html";
  const fp = path.normalize(path.join(ROOT, urlPath));
  if (!fp.startsWith(ROOT)) { res.writeHead(403); res.end("forbidden"); return; }
  fs.stat(fp, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); res.end("not found: " + urlPath); return; }
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    res.setHeader("Content-Type", MIME[path.extname(fp)] || "application/octet-stream");
    res.setHeader("Content-Length", st.size);
    if (req.method === "HEAD") { res.writeHead(200); res.end(); return; }
    fs.createReadStream(fp).pipe(res);
  });
});
server.listen(PORT, () => console.log(`serving ${ROOT} on http://localhost:${PORT}`));
