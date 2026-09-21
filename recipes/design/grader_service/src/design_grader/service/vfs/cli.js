#!/usr/bin/env node
/* CLI over the vendored sft_viewer VFS core: rebuild the workspace file state
 * from a message trajectory.
 *
 *   node cli.js < '{"messages": [...]}'   (stdin JSON)
 *   -> {"entry": "/workspace/dist/index.html" | null,
 *       "files": {"<path>": {"content": "...", "fidelity": "exact|approx|stale|binary|unknown"}},
 *       "report": ["rm ...", ...]}
 *
 * vfs_core.js is VERBATIM lines @@VFS_CORE_START@@..pickEntry of
 * superollout/scripts/sft_viewer.html — re-extract from the viewer to update,
 * never edit in place (the viewer is the single source of truth).
 */
"use strict";
const fs = require("fs");
const path = require("path");

const core = fs.readFileSync(path.join(__dirname, "vfs_core.js"), "utf8");
// the core is written as top-level script for the viewer; evaluate it and pull
// out the two functions the CLI needs
const { buildVfs, pickEntry } = new Function(
  core + "; return { buildVfs, pickEntry };"
)();

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const vfs = buildVfs(input.messages || []);
const files = {};
const fidelity = { exact: 0, approx: 0, stale: 0, binary: 0, unknown: 0 };
let editMisses = 0;
for (const [p, f] of vfs.files.entries()) {
  fidelity[f.status] = (fidelity[f.status] || 0) + 1;
  editMisses += f.misses || 0;
  if (typeof f.content !== "string") continue;   // binary/unknown carry no rebuildable content
  files[p] = { content: f.content, fidelity: f.status };
}
process.stdout.write(JSON.stringify({
  entry: pickEntry(vfs),
  files,
  fidelity,
  edit_misses: editMisses,
  report: vfs.report.slice(-40),
}));
