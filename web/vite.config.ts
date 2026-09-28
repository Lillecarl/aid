import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// `npm run dev` serves the UI and passes everything else to `aid web`, whose address AID_WEB_BACKEND names.
const backend = process.env.AID_WEB_BACKEND ?? "http://127.0.0.1:8080";
const proxied = ["/api", "/login", "/logout", "/auth"];

export default defineConfig({
  plugins: [svelte()],
  build: { outDir: "dist", assetsDir: "assets" },
  server: { proxy: Object.fromEntries(proxied.map((path) => [path, backend])) },
});
