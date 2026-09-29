import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// `npm run dev` serves the UI and passes everything else to `aid web`, whose address AID_WEB_BACKEND names.
// public/ (the manifest and icons) Vite serves itself, and copies into dist for `aid web`.
const backend = process.env.AID_WEB_BACKEND ?? "http://127.0.0.1:8080";
const proxied = ["/api", "/login", "/logout", "/auth", "/theme.css"];

export default defineConfig({
  plugins: [svelte()],
  // No inlining: a small asset becomes a data: URL, and the CSP refuses script from one (the mic worklet).
  build: { outDir: "dist", assetsDir: "assets", assetsInlineLimit: 0 },
  server: { proxy: Object.fromEntries(proxied.map((path) => [path, backend])) },
});
