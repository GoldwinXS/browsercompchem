import { defineConfig } from "vite";

export default defineConfig({
  root: __dirname,
  server: {
    port: 8142,
    strictPort: true,
  },
});
