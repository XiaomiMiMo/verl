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
"""Burst soak: a step's worth of /grade at once, against a router that congests.

WHY THIS EXISTS
---------------
An early training run lost 13% of its grades and 98% of those were judge
ReadTimeouts. They were not the router being slow on its own: judge latency was
a monotone function of how many calls WE had in flight (p50 17s under 500,
420s above 3500) and throughput collapsed from ~30 calls/s to ~7. Meanwhile
render_s p50 was 139s for a ~6s render — a queue behind 64 chromium slots that
each burned 8.2 core-seconds, 6.5 of them an OpenBLAS import.

Unit tests pin each fix in isolation. This puts them together under the real
shape of the load — one burst of N requests, real chromium, a router stub whose
latency grows with its own in-flight count — and reports what a training step
would have seen: drop rate by reason, render/judge/total latency, the limiter's
decisions, cpu_busy and the unrenderable count.

No API key is used. The stub answers both wire formats with a verdict the
query judge parses.

USAGE
-----
  PYTHONPATH=src python3 scripts/burst_soak.py --n 1024 --budget 720

  The stub's latency curve defaults to the measured router (see Router);
  --knee/--base-latency/--power/--sigma reshape it, --slope switches to linear.
  Needs PLAYWRIGHT_BROWSERS_PATH (+ DESIGN_GRADER_FONTCONF) like the service.

PASS  drops are only `queue:`/`budget` reasons (fast, honest, before the client
      would have timed out), unrenderable stays ~0, and the limiter's limit
      ends near the stub's knee.
FAIL  ReadTimeouts, `starved`, or unrenderable climbing with N.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import http.server
import json
import os
import random
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")

VERDICT = json.dumps(
    {
        "query_score": 1.0,
        "layout_score": 1.0,
        "score": 1.0,
        "tier": 2,
        "reason": "stub",
        "rationale": "stub",
        "query_rationale": "stub",
        "layout_rationale": "stub",
    }
)


class Router(http.server.BaseHTTPRequestHandler):
    """Both wire formats; latency follows the measured router curve.

    median = base * max(1, inflight/knee) ** power, times a lognormal tail
    (sigma) — measured data has p50 17s under 500 in flight, 69s at
    1000-1499, 176s at 2000-2499, and p90 ~3-5x p50 in every bucket. The
    defaults reproduce that; --slope>0 switches to a plain linear ramp.
    """

    inflight = 0
    lock = threading.Lock()
    served = []  # (latency, inflight at start)
    base = knee = slope = power = sigma = 0.0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with Router.lock:
            Router.inflight += 1
            n = Router.inflight
        if Router.slope > 0:
            lat = (Router.base + Router.slope * max(0, n - Router.knee)) * (0.8 + 0.4 * random.random())
        else:
            lat = Router.base * max(1.0, n / Router.knee) ** Router.power * random.lognormvariate(0, Router.sigma)
        time.sleep(lat)
        with Router.lock:
            Router.inflight -= 1
            Router.served.append((lat, n))
        if "generateContent" in self.path:
            body = {
                "candidates": [{"content": {"parts": [{"text": VERDICT}], "role": "model"}, "finishReason": "STOP"}]
            }
        else:
            body = {"choices": [{"message": {"role": "assistant", "content": VERDICT}}]}
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 4096


def _page(i: int) -> str:
    sections = 3 + i % 6
    return (
        "<!DOCTYPE html><html><head><meta charset=utf-8><style>"
        "body{font-family:sans-serif;margin:0}section{padding:60px;min-height:600px}"
        ".g{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}"
        ".c{background:#f3f4f6;border-radius:12px;padding:24px}"
        f".h{{background:linear-gradient(135deg,hsl({i * 37 % 360},60%,55%),#764ba2);color:#fff}}"
        "</style></head><body><section class=h><h1>页面 "
        + str(i)
        + "</h1></section>"
        + "".join(
            f"<section><h2>S{s}</h2><div class=g>"
            + "".join(f"<div class=c><h3>Card {c}</h3><p>{'text ' * 30}</p></div>" for c in range(6))
            + "</div></section>"
            for s in range(sections)
        )
        + "</body></html>"
    )


def _grade(url: str, i: int, budget: float) -> dict:
    body = {
        "task_id": f"soak{i}",
        "query": f"build page {i}",
        "budget_s": budget,
        "response": "</think>```html\n" + _page(i) + "\n```",
    }
    req = urllib.request.Request(
        url + "/grade", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t = time.time()
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=budget + 30) as r:
            out = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 — the client's view of a failure is the point
        out = {"status": "client_error", "drop_reason": f"{type(e).__name__}: {e}"[:120]}
    out["_wall"] = time.time() - t
    return out


def _q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p))] if a else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="requests in the burst")
    ap.add_argument("--budget", type=float, default=400, help="client budget per request (s)")
    ap.add_argument("--knee", type=int, default=500, help="stub router: in-flight before latency climbs")
    ap.add_argument("--base-latency", type=float, default=17.0, help="stub router: median latency under the knee")
    ap.add_argument("--power", type=float, default=1.7, help="stub router: median ~ (inflight/knee)^power")
    ap.add_argument("--sigma", type=float, default=0.8, help="stub router: lognormal tail (0.8 -> p90 ~2.8x p50)")
    ap.add_argument("--slope", type=float, default=0.0, help="stub router: >0 switches to a linear ramp, s/call")
    ap.add_argument("--keep-log", action="store_true")
    args = ap.parse_args()

    Router.base, Router.knee, Router.slope = args.base_latency, args.knee, args.slope
    Router.power, Router.sigma = args.power, args.sigma
    router = _TS(("127.0.0.1", 0), Router)
    threading.Thread(target=router.serve_forever, daemon=True).start()
    rbase = f"http://127.0.0.1:{router.server_address[1]}"

    with socketserver.TCPServer(("127.0.0.1", 0), None) as s:
        port = s.server_address[1]
    dump = tempfile.NamedTemporaryFile("w", suffix="_soak_dump.jsonl", delete=False).name
    log = open(tempfile.NamedTemporaryFile("w", suffix="_soak_server.log", delete=False).name, "w")
    env = {
        **os.environ,
        "LLM_JUDGE_API_KEY": "stub",
        "LLM_JUDGE_BASE_URL": rbase,
        "DESIGN_GRADER_DUMP": dump,
        "PYTHONPATH": SRC,
        "no_proxy": "127.0.0.1,localhost",
    }
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    srv = subprocess.Popen(
        [sys.executable, "-m", "design_grader.service.server", "--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/healthz", timeout=2).read()
            break
        except Exception:
            if srv.poll() is not None:
                print(open(log.name).read()[-3000:])
                sys.exit("server died on startup")
            time.sleep(0.5)
    print(
        f"[soak] grader {url} | stub router {rbase} "
        f"base={args.base_latency}s knee={args.knee} power={args.power} sigma={args.sigma} slope={args.slope} | log {log.name}"
    )

    def healthz():
        try:
            return json.loads(
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/healthz", timeout=10).read()
            )
        except urllib.error.HTTPError as e:  # 503 = ok:false, still a full body
            return json.loads(e.read())
        except Exception as e:  # noqa: BLE001
            return {"err": str(e)[:60]}

    stop = threading.Event()
    trace = []

    def monitor():
        while not stop.is_set():
            h = healthz()
            j = h.get("judge", {})
            trace.append((time.time(), h))
            rp = h.get("render_pool", {})
            print(
                f"[soak] t+{time.time() - t0:5.0f}s ok={h.get('ok')} cpu={h.get('cpu_busy')} "
                f"render={h.get('render_inflight')}/{h.get('render_max')} "
                f"pool(idle={rp.get('idle')} respawn={rp.get('respawns')}) "
                f"judge={j.get('inflight')}/{j.get('limit')} +{j.get('waiting')} "
                f"stub_inflight={Router.inflight} chrome={h.get('chrome')} thr={h.get('threads')}"
                + (f" err={h['err']}" if "err" in h else ""),
                flush=True,
            )
            stop.wait(10)

    t0 = time.time()
    threading.Thread(target=monitor, daemon=True).start()
    with cf.ThreadPoolExecutor(max_workers=args.n) as ex:
        outs = list(ex.map(lambda i: _grade(url, i, args.budget), range(args.n)))
    stop.set()
    wall = time.time() - t0
    time.sleep(1)
    h = healthz()
    srv.terminate()
    try:
        srv.wait(10)
    except subprocess.TimeoutExpired:
        srv.kill()
    log.close()

    st = collections.Counter(o.get("status") for o in outs)
    reasons = collections.Counter((o.get("drop_reason") or "")[:60] for o in outs if o.get("status") != "ok")
    tm = collections.defaultdict(list)
    for o in outs:
        for k, v in (o.get("timing") or {}).items():
            if v is not None:
                tm[k].append(v)
        tm["client_wall"].append(o["_wall"])
    unr = sum(1 for o in outs if (o.get("signals") or {}).get("unrenderable"))
    starved = sum(1 for o in outs if "starved" in (o.get("drop_reason") or ""))
    decisions = [l.strip() for l in open(log.name) if l.startswith("[judge-limit]")]
    print(f"\n[soak] {args.n} requests in {wall:.0f}s wall -> {dict(st)}  unrenderable={unr} starved={starved}")
    for r, c in reasons.most_common():
        print(f"        {c:5d}  {r}")
    print("[soak] timing            p50     p90     p99     max")
    for k in ("localize_s", "render_s", "judges_s", "total_s", "client_wall"):
        a = tm.get(k)
        if a:
            print(f"        {k:12s} {_q(a, 0.5):7.1f} {_q(a, 0.9):7.1f} {_q(a, 0.99):7.1f} {max(a):7.1f}")
    if Router.served:
        lat = [l for l, _ in Router.served]
        print(
            f"[soak] stub router served {len(lat)} calls: latency p50={_q(lat, 0.5):.1f}s "
            f"p90={_q(lat, 0.9):.1f}s max={max(lat):.1f}s; peak in-flight={max(n for _, n in Router.served)}"
        )
    print(
        f"[soak] limiter: {len(decisions)} decisions, final {h.get('judge', {}).get('limit')}; "
        f"chrome left={h.get('chrome')} threads={h.get('threads')}"
    )
    for d in decisions[:12]:
        print("        " + d)
    print(f"[soak] server log {log.name}\n[soak] dump {dump}")
    if not args.keep_log and st.get("ok", 0) == args.n:
        os.unlink(dump)
    bad = [r for r in reasons if "ReadTimeout" in r or "starved" in r or "server error" in r]
    print("[soak] PASS" if not bad and st.get("client_error", 0) == 0 else f"[soak] FAIL: {bad or st}")


if __name__ == "__main__":
    main()
