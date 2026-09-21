#!/usr/bin/env python3
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
"""Leak soak: prove a running grader does not accumulate chrome zombies.

WHY THIS EXISTS
---------------
Two render-recursion bugs shipped because neither was caught before training:

  1. subprocess.run kills only its direct child. playwright's chrome is a
     GRANDchild, so it was orphaned on every timeout — 130k orphans, the box
     ran out of pids. (fixed: start_new_session + killpg)
  2. chromium calls setsid() on launch, so it ESCAPES that process group.
     killpg still worked — the browser died when its parent died — but it was
     then reparented to PID 1, and at 15 renders/sec the init's reap could not
     keep up: 52754 zombies in 58 minutes. (fixed: PR_SET_CHILD_SUBREAPER +
     waitpid)

Both were invisible to unit tests. A synthetic grandchild does not setsid(), and
even one that does is reaped by this pod's supervisord PID 1 at a rate far above
what a soak can confuse with a leak. Only real load reproduces #2.

USAGE
-----
  python3 leak_soak.py --url http://<grader>/ --cases 24 --concurrency 6

It renders N real rollouts through the FULL /grade pipeline, then compares the
zombie count against the baseline taken before the run.

PASS  zombie delta <= ZOMBIE_OK (default 2) — the subreaper is keeping up
FAIL  zombie delta grows with N    — the leak is back; do not train

Reads real rollouts from a corpus of graded pages by default; pass --html-dir
and --cases-json to point it elsewhere.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request


def chrome_census():
    """(alive, zombies) chrome-headless counts. Zombies are the leak signal —
    pgrep counts both, so counting only live processes hides the problem."""
    alive = zombies = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                st = f.read().split()
        except OSError:
            continue
        if "chrome" not in st[1]:
            continue
        if st[2] == "Z":
            zombies += 1
        else:
            alive += 1
    return alive, zombies


def grade(url, key, query, html, timeout):
    body = json.dumps({"task_id": key, "query": query, "response": html}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/grade", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return key, out.get("status"), (out.get("timing") or {}).get("total_s"), None
    except Exception as e:  # noqa: BLE001
        return key, "ERROR", round(time.time() - t0, 1), f"{type(e).__name__}: {e}"[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1")
    ap.add_argument("--html-dir", required=True, help="directory of sample HTML pages to soak against")
    ap.add_argument("--cases-json", default="/tmp/cmp_cases.json")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=400)
    ap.add_argument("--zombie-ok", type=int, default=2, help="max acceptable zombie delta after the soak")
    a = ap.parse_args()

    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,.local")
    try:
        ver = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/version", timeout=10).read())
    except Exception as e:  # noqa: BLE001
        print(f"grader unreachable at {a.url}: {type(e).__name__}: {e}", file=sys.stderr)
        print("start it first — a soak against a dead service proves nothing", file=sys.stderr)
        return 2
    print(f"grader: {ver.get('grader_version')}  semantics={ver.get('reward_semantics')}")

    try:
        picked = json.load(open(a.cases_json))[: a.cases]
    except FileNotFoundError:
        print(f"no {a.cases_json}; pass --cases-json", file=sys.stderr)
        return 2
    items = []
    for c in picked:
        p = os.path.join(a.html_dir, c["key"] + ".html")
        if os.path.isfile(p):
            items.append((c["key"], c["query"], open(p).read()))
    print(f"cases: {len(items)} @ concurrency {a.concurrency}")

    a0, z0 = chrome_census()
    print(f"\nbaseline   alive={a0}  zombies={z0}")
    t0 = time.time()
    res = []
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(grade, a.url, k, q, h, a.timeout) for k, q, h in items]
        for f in cf.as_completed(futs):
            res.append(f.result())
    wall = time.time() - t0
    time.sleep(3)  # let any stragglers exit and be reaped
    a1, z1 = chrome_census()

    st = collections.Counter(r[1] for r in res)
    ts = sorted(r[2] for r in res if r[2])
    print(f"done in {wall:.1f}s   status={dict(st)}")
    if ts:
        print(f"total_s  p50={ts[len(ts) // 2]:.1f}  p90={ts[int(len(ts) * 0.9)]:.1f}  max={ts[-1]:.1f}")
    print(f"after      alive={a1}  zombies={z1}")
    print(f"delta      alive={a1 - a0:+d}  zombies={z1 - z0:+d}")

    errs = [r for r in res if r[3]]
    for r in errs[:5]:
        print(f"  ERR {r[0]}: {r[3]}")

    dz = z1 - z0
    if dz > a.zombie_ok:
        print(
            f"\nFAIL: {dz} new zombies over {len(items)} renders — the leak is back. Do not train on this build.",
            file=sys.stderr,
        )
        return 1
    if a1 > 0 and a1 > a.concurrency * 4:
        print(
            f"\nWARN: {a1} chrome still alive after the soak; "
            f"expected <= {a.concurrency * 4} (render concurrency x ~4).",
            file=sys.stderr,
        )
        return 1
    print(f"\nPASS: {len(items)} renders left {dz} zombies — reaping is keeping up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
