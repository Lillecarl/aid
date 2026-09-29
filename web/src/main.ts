import { mount } from "svelte";
import App from "./App.svelte";
import "./app.css";

const target = document.getElementById("app");
if (target === null) throw new Error("no #app element");

// An installed app's title bar takes theme-color: match the page, whether /theme.css or the browser chose it.
function themeColor(): void {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta) meta.content = getComputedStyle(document.body).backgroundColor;
}
window.addEventListener("load", themeColor);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeColor);

export default mount(App, { target });
