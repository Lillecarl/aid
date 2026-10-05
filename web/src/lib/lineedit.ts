// Readline editing for prompt boxes: the browser leaves Ctrl+U/K/W to the page (or, for Ctrl+U,
// opens view-source), so the box handles them. Ctrl+A stays the browser's select-all; Ctrl+E moves,
// since the browser binds nothing there. Edits go through setRangeText plus an input event, the same
// path typed text takes, so two-way bindings follow without a flush.

export function editLine(event: KeyboardEvent, area: HTMLTextAreaElement): boolean {
  if (event.isComposing || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return false;
  const key = event.key.toLowerCase();
  if (key !== "u" && key !== "k" && key !== "w" && key !== "e") return false;
  event.preventDefault();
  const value = area.value;
  const point = area.selectionStart;
  const lineStart = value.lastIndexOf("\n", point - 1) + 1;
  if (key === "e") {
    let lineEnd = value.indexOf("\n", point);
    if (lineEnd === -1) lineEnd = value.length;
    area.setSelectionRange(lineEnd, lineEnd);
    return true;
  }
  if (key === "u") {
    area.setRangeText("", lineStart, point, "start");
  } else if (key === "k") {
    let lineEnd = value.indexOf("\n", point);
    if (lineEnd === -1) lineEnd = value.length;
    area.setRangeText("", point, lineEnd, "start");
  } else {
    const before = value.slice(0, point);
    const anchor = /\s$/.test(before)
      ? before.replace(/\s+$/, "").length
      : before.replace(/\S+$/, "").length;
    area.setRangeText("", anchor, point, "start");
  }
  area.dispatchEvent(new Event("input", { bubbles: true }));
  return true;
}
