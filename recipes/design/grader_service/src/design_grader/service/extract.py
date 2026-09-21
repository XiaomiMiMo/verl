# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rebuild the delivered site from a rollout (trajectory replay or chat response).

The RL side ships the raw model output; the deliverable is reconstructed here so
no rollout environment ever runs a grading step. Two delivery shapes:

  * agent trajectory (`extract_traj`) — full VFS reconstruction via the vendored
    sft_viewer core (vfs/vfs_core.js, run in a node subprocess): structured
    editors (Write/Edit/MultiEdit/str_replace family), bash heredocs/echo/sed/
    mv/cp/rm, codex apply_patch, python/node inline writers — with per-file
    fidelity (exact/approx/stale/binary/unknown). Only exact+approx files are
    materialized; true build outputs (npm/vite) are NOT reproducible — by design
    the graded contract stays "deliver files, don't rely on a build step", and
    `n_bash` keeps that drift visible.

  * chat response (`extract_response`) — a single-turn generation whose text
    contains the html document (fenced ```html block or a bare <!DOCTYPE>/<html>).

  * pod workspace (`extract_workspace`) — the agent's real working directory,
    shipped by the RL side as a base64 tar.gz at reward time. This is the
    authoritative agent surface: it is what the pod actually holds, so scripted
    edits (python/sed/node), build outputs and rejected tool calls all come out
    right, where trajectory replay could only guess. `extract_traj` stays for
    offline analysis of trajectories that have no workspace capture.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field

_VFS_CLI = os.path.join(os.path.dirname(__file__), "vfs", "cli.js")


@dataclass
class ExtractResult:
    entry: str | None  # vfs path of the chosen entry, None = no delivery
    files: dict[str, str | bytes] = field(default_factory=dict)
    n_writes: int = 0
    n_edits: int = 0
    n_edit_misses: int = 0  # edits whose anchor never matched (agent bug, not ours)
    n_bash: int = 0
    fidelity: dict = field(default_factory=dict)  # per-status file counts (traj mode)

    def entry_text(self) -> str:
        """The entry html as text, "" when there is none.

        `files` is keyed RELATIVE TO THE ENTRY'S DIR (that is what materialize
        writes), while `entry` keeps the path the source used — `index.html` for
        a chat response, `dist/index.html` for a workspace tar, `/w/dist/index.html`
        for a replayed trajectory. So `files.get(entry)` only hits in the chat
        case; the runtime gate read "" for every agent page until this was fixed.
        Look up by the entry's basename, which is the one key materialize
        guarantees.
        """
        if self.entry is None:
            return ""
        v = self.files.get(self.entry)
        if v is None:
            v = self.files.get(os.path.basename(self.entry))
        return v if isinstance(v, str) else ""


_node_path: str | None = None


def _node() -> str:
    """System node, else the node binary the playwright pip package ships."""
    global _node_path
    if _node_path is None:
        _node_path = shutil.which("node")
        if _node_path is None:
            import playwright  # the service depends on playwright anyway

            cand = os.path.join(os.path.dirname(playwright.__file__), "driver", "node")
            if not os.path.isfile(cand):
                raise RuntimeError(
                    "no `node` on PATH and playwright driver node missing "
                    "— the trajectory VFS rebuild needs a node runtime"
                )
            _node_path = cand
    return _node_path


def _count_tools(messages: list[dict]) -> tuple[int, int, int]:
    writes = edits = bash = 0
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = ((tc.get("function") or {}).get("name") or "").lower()
            if name in ("write", "create", "multiedit"):
                writes += 1
            elif "edit" in name or "str_replace" in name:
                edits += 1
            elif name in ("bash", "shell", "exec_command", "run"):
                bash += 1
    return writes, edits, bash


def extract_traj(messages: list[dict], *, timeout_s: float = 60) -> ExtractResult:
    """Full-surface trajectory replay (see module docstring). Tolerant by design:
    the agent's own mistakes must render as the (bad) page they produce; only a
    broken VFS runtime raises (infra, caller drops)."""
    r = subprocess.run(
        [_node(), _VFS_CLI],
        input=json.dumps({"messages": messages or []}),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if r.returncode != 0:
        raise RuntimeError(f"vfs rebuild failed rc={r.returncode}: {(r.stderr or '')[-300:]}")
    out = json.loads(r.stdout)

    res = ExtractResult(entry=None, fidelity=out.get("fidelity", {}))
    res.n_writes, res.n_edits, res.n_bash = _count_tools(messages)
    res.n_edit_misses = int(out.get("edit_misses", 0))

    usable = {p: f["content"] for p, f in out.get("files", {}).items() if f.get("fidelity") in ("exact", "approx")}
    entry = out.get("entry")
    dist_entries = [p for p in usable if re.search(r"(^|/)dist/index\.html$", p)]
    if dist_entries and entry not in dist_entries:
        entry = min(dist_entries, key=len)
    if entry not in usable:
        return res
    root = os.path.dirname(entry)
    res.entry = entry
    res.files = {
        os.path.relpath(p, root) if root else p: c
        for p, c in usable.items()
        if p == entry or not root or p.startswith(root + "/")
    }
    if os.path.basename(entry) != "index.html":
        res.files["index.html"] = res.files.pop(os.path.relpath(entry, root) if root else entry)
    return res



_WORKSPACE_SKIP_DIRS = ("node_modules", ".git", ".cache", "__pycache__")
_ENTRY_RE = re.compile(r"\.(html?|svg)$", re.IGNORECASE)
_INDEX_RE = re.compile(r"(^|/)index\.html?$", re.IGNORECASE)
_DIST_INDEX_RE = re.compile(r"(^|/)dist/index\.html$")


def _untar_workspace(tgz_b64: str, *, max_file_bytes: int, max_total_bytes: int) -> dict[str, bytes]:
    """Decode the RL side's `tar -czf - -C <cwd> . | base64` into {relpath: bytes}.

    Model-controlled archive, so extraction is done by hand rather than
    tarfile.extractall: only regular files, only paths that stay under the
    root, per-file and total caps. Anything else is skipped, never an error —
    the agent's stray junk must not turn its page into a drop."""
    raw = base64.b64decode(tgz_b64, validate=False)
    out: dict[str, bytes] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        for m in tf:
            if not m.isfile() or m.size > max_file_bytes:
                continue
            rel = os.path.normpath(m.name)
            if rel.startswith(("/", "../")) or rel == ".." or rel in (".", ""):
                continue
            parts = rel.split("/")
            if any(p in _WORKSPACE_SKIP_DIRS for p in parts):
                continue
            if total + m.size > max_total_bytes:
                break
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            out[rel] = data
            total += len(data)
    return out


def pick_workspace_entry(paths) -> str | None:
    """Same ranking as the VFS core's pickEntry: index.html > other .html > .svg,
    shallower first, then larger. `dist/index.html` wins when present — a
    build (vite/astro) leaves the source index.html next to the real output."""
    cand = [p for p in paths if _ENTRY_RE.search(p)]
    if not cand:
        return None
    dist = [p for p in cand if _DIST_INDEX_RE.search(p)]
    if dist:
        return min(dist, key=len)
    sizes = dict(paths) if isinstance(paths, dict) else {p: 0 for p in paths}

    def rank(p: str) -> int:
        return 0 if _INDEX_RE.search(p) else 1 if p.lower().endswith((".html", ".htm")) else 2

    return sorted(cand, key=lambda p: (rank(p), p.count("/"), -sizes[p]))[0]


def extract_workspace(
    tgz_b64: str, *, max_file_bytes: int = 2 * 1024 * 1024, max_total_bytes: int = 20 * 1024 * 1024
) -> ExtractResult:
    """Pick the delivered site out of the agent's real working directory.

    The subtree rooted at the entry's directory is served, entry as index.html
    (same rule as `extract_traj`, so relative asset links resolve). Text files
    are decoded as utf-8 for the html post-processing downstream; everything
    else (images, fonts) is carried as bytes so a page that references its own
    generated assets renders the way it did in the pod."""
    files = _untar_workspace(tgz_b64, max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes)
    res = ExtractResult(entry=None)
    res.fidelity = {"workspace_files": len(files)}
    entry = pick_workspace_entry({p: len(b) for p, b in files.items()})
    if entry is None:
        return res
    root = os.path.dirname(entry)
    res.entry = entry
    for p, data in files.items():
        if not (p == entry or not root or p.startswith(root + "/")):
            continue
        rel = os.path.relpath(p, root) if root else p
        if p == entry:
            rel = "index.html"
        try:
            res.files[rel] = data.decode("utf-8") if _is_text_path(rel) else data
        except UnicodeDecodeError:
            res.files[rel] = data
    return res


_TEXT_SUFFIXES = (
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".svg",
    ".txt",
    ".md",
    ".xml",
    ".ts",
    ".jsx",
    ".tsx",
    ".vue",
    ".map",
    ".webmanifest",
)


def _is_text_path(rel: str) -> bool:
    return rel.lower().endswith(_TEXT_SUFFIXES)


_FENCED_HTML = re.compile(r"```+\s*html[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_HTML_START = re.compile(r"<!DOCTYPE\s+html|<html[\s>]", re.IGNORECASE)


def extract_response(text: str) -> ExtractResult:
    """Cut the delivered html document out of a chat response.

    The generation arrives VERBATIM, <think> block included (the RL side ships
    it raw so future rewards can read the reasoning) — delivery is whatever
    comes after the think block; a draft fenced block inside <think> is not a
    delivery. Fenced ```html blocks win over bare markup (the fence is the
    delivery contract stated in the chat prompt); with several fenced blocks
    the longest one is taken — models often show a fragment before the full
    document. A bare document is accepted as fallback so a model that skips
    the fence still gets graded on what it produced.
    """
    text = (text or "").rsplit("</think>", 1)[-1]
    res = ExtractResult(entry=None)
    fenced = _FENCED_HTML.findall(text)
    if fenced:
        html = max(fenced, key=len).strip()
    else:
        m = _HTML_START.search(text)
        if not m:
            return res
        end = text.lower().rfind("</html>")
        html = text[m.start() : end + len("</html>")] if end > m.start() else text[m.start() :]
    res.entry = "index.html"
    res.files = {"index.html": html}
    return res


def materialize(res: ExtractResult, out_dir: str) -> str:
    """Write the extracted subtree under out_dir; returns the entry file path."""
    out_abs = os.path.abspath(out_dir)
    for rel, content in res.files.items():
        dst = os.path.normpath(os.path.join(out_abs, rel))
        if not dst.startswith(out_abs + os.sep):
            continue  # path traversal in a generated path: skip, never escape
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if isinstance(content, bytes):
                with open(dst, "wb") as f:
                    f.write(content)
            else:
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(content)
        except (FileExistsError, NotADirectoryError, IsADirectoryError):
            continue
    return os.path.join(out_abs, "index.html")
