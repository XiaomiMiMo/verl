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
"""The `runtime` gate — does the delivered page's JavaScript even run.

    reward = base_reward x runtime,    runtime in {0.0, 1.0}

A hard gate, not a multiplier: three things a page can do that no working page
ever does, each observable without a judge and without a second browser pass —

  1. syntax    an inline <script> that `node --check` rejects. The browser skips
               the WHOLE block, and 84% of delivered pages have exactly one
               block, so this is "zero interactivity" with certainty.
  2. pageerror an uncaught exception during load + settle. The render pass
               already records every `pageerror` event; we only classify.
  3. hang      the renderer stopped answering (`page unresponsive` /
               `unrenderable`) — a `while(1){}` or an infinite DOM loop.

Why 0 and not a soft factor: a soft factor is tradable against the visual score
(vis=1.0 x 0.5 still beats a clean vis=0.5 page), so RL treats a runtime error
as a cost to amortise rather than a precondition. The screenshot judges are
blind to errors that fire after the page has painted, and in practice
15% of pages with an uncaught error still scored >= 0.5.

Why 0 and not drop for a hang: drop -> adv masked -> zero contribution, which
beats the negative contribution a failing rollout would otherwise get, so the
model learns that wedging the renderer is safe (the `while(1){}` incident,
render.py). The one exception is a saturated box, which render.py already
labels as an env verdict before we get here.

What is NOT an error here (the environment's fault or not JS at all):
  * console.error without a throw — SVG attribute warnings, our own probe noise
  * failed external requests — fonts / CDNs; the dist is localised, and a
    missing library would surface as its own ReferenceError anyway only when
    the network was cut, which is the sandbox's doing (see ENV_NOISE)
  * localStorage / sessionStorage denied — only happens on about:blank pages,
    i.e. a test harness, never on the loopback server the grader uses
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from . import extract

RUNTIME_ZERO = 0.0

_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)
_TYPE = re.compile(r"""\btype\s*=\s*["']?([^"'\s>]+)""", re.I)
_SRC = re.compile(r"\bsrc\s*=", re.I)
_JS_TYPES = {None, "", "text/javascript", "application/javascript", "module"}

NODE_CHECK_TIMEOUT_S = 20

ENV_NOISE = re.compile(
    r"localStorage|sessionStorage"  # about:blank storage denial (test rigs only)
    r"|history state object with URL",  # pushState cross-origin on about:blank
    re.I,
)


def js_blocks(html: str) -> list[tuple[str, bool]]:
    """[(code, is_module)] for every inline JS block worth parsing."""
    out = []
    for attrs, code in _SCRIPT.findall(html or ""):
        if _SRC.search(attrs) or not code.strip():
            continue
        m = _TYPE.search(attrs)
        t = m.group(1).lower() if m else None
        if t not in _JS_TYPES:
            continue
        out.append((code, t == "module"))
    return out


def syntax_errors(html: str) -> list[str]:
    """node --check on each inline block; returns the first error line per bad block."""
    errs = []
    node = extract._node()
    for code, is_module in js_blocks(html):
        suffix = ".mjs" if is_module else ".js"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
            f.write(code)
            path = f.name
        try:
            r = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=NODE_CHECK_TIMEOUT_S)
            if r.returncode != 0:
                lines = (r.stderr or "").strip().splitlines()
                msg = next((l for l in lines if "Error" in l), lines[-1] if lines else "syntax error")
                errs.append(msg.strip()[:200])
        except subprocess.TimeoutExpired:
            errs.append("node --check timed out")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return errs


def page_errors(signals: dict) -> list[str]:
    """Uncaught exceptions the render pass saw, minus environment noise.

    render.py folds `pageerror` events into sig["console_errors"] with a
    "pageerror: " prefix; plain console.error lines have none and are ignored.
    """
    out = []
    for line in signals.get("console_errors") or []:
        if not isinstance(line, str) or not line.startswith("pageerror: "):
            continue
        msg = line[len("pageerror: ") :]
        if ENV_NOISE.search(msg):
            continue
        out.append(msg[:200])
    return out


def hung(signals: dict) -> str | None:
    """The renderer stopped answering — render.py's liveness probe or a missing frame."""
    if signals.get("unrenderable"):
        return "unrenderable: no frame could be captured"
    for line in signals.get("console_errors") or []:
        if isinstance(line, str) and line.startswith("page unresponsive"):
            return line[:200]
    return None


def verdict(html: str, signals: dict) -> dict:
    """The gate. Pure on its inputs; `signals` is render.capture_hero's sig dict.

    Returns {"factor", "syntax", "pageerror", "hang", "why"} — the evidence
    rides on the dim for the dump, the factor is what the group reward reads
    (group_pick.reward_one floors a factor-0 row at RUNTIME_FLOOR).
    """
    syn = syntax_errors(html)
    pe = page_errors(signals)
    hg = hung(signals)
    bugs = []
    if syn:
        bugs.append(f"syntax: {syn[0]}")
    if pe:
        bugs.append(f"uncaught: {pe[0]}")
    if hg:
        bugs.append(f"hang: {hg}")
    return {
        "factor": RUNTIME_ZERO if bugs else 1.0,
        "syntax": syn,
        "pageerror": pe,
        "hang": hg,
        "why": "; ".join(bugs) if bugs else "clean",
    }
