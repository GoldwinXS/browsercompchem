import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    environment: "node",
  },
  resolve: {
    alias: {
      // Resolve straight to source so bench tests don't require the
      // engine package to be built first.
      "@browser-comp-chem/engine": path.resolve(__dirname, "../engine/src/index.ts"),
    },
  },
});
