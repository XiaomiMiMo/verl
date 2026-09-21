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
"""Design grader service — rollout in, webdev GROUP reward out.

THIS SERVICE SERVES ONE REWARD: webdev group_v1, in two calls per GRPO group.
The pointwise rubric (five judges, a per-rollout absolute score) is gone — the
training path never called it, because every `/grade` it sends carries
`judge_scope="query"`, which short-circuited the whole rubric anyway. Evaluation
does not come here at all: recipes/design/webdev/eval_mode.py judges in-pod.

    POST /grade   {"task_id", "query", "judge_scope": "query",
                   "workspace_tgz_b64": "...", "messages": [...]}
                                                     agent: pod workspace (authoritative)
                                                     + trajectory (signals only)
                  ... or {"messages": [...]} alone   agent trajectory replay (offline)
                  ... or {"response": "..."}         single-turn chat text
      -> {"status": "ok", "judge_scope": "query", "reward": null,
          "query_score": float, "runtime": {factor, why, syntax, pageerror, hang},
          "dims": {"query": {...}, "runtime": {...}},
          "reason", "signals": {...}, "shot_jpg_b64": str,
          "grader_version", "timing": {...}}
      -> {"status": "drop", "reward": null, "drop_reason", "grader_version"}

      `judge_scope="query"` IS REQUIRED. A request without it is dropped with a
      drop_reason saying so, rather than silently taking a path that no longer
      exists — see _grade_task.

    POST /grade_group  {"task_id", "query", "shots_jpg_b64": [...],
          "query_scores": [...]|null, "runtime_factors": [...]|null,
          "seed", "budget_s"}  — the relative visual-quality pick over one GRPO
          group (group_pick.py; 8-round Williams, the "pick" judge model).
          Returns per-shot rewards in [0, 1]:
          (pick_norm − query_deduct + 2) / 3, floor 0.0 (see group_pick.py).

    GET  /healthz | /version
    GET  /            built-in dashboard: live requests, screenshots, verdicts, raw output
    GET  /api/recent  recent verdicts (no image/text payloads) + running counters
    GET  /api/shot/<record_id>   the page JPEG the judges saw (in-memory ring buffer;
                                 keyed by the per-record _id from /api/recent —
                                 task_id is NOT unique: n rollouts of one prompt share it)
    GET  /api/raw/<record_id>    the model's raw generation (thinking + output)

Pipeline per /grade: rebuild delivery -> localize remote assets -> render the
FULL PAGE (subprocess playwright) -> query judge + runtime gate.

The reward has exactly three inputs, and
`group_pick.reward_one(pick_norm, query_score, runtime_factor)` reads all of
them:

  pick_norm       POST /grade_group, 8 Williams rounds of the `pick` judge
  query_score     POST /grade  judge_scope=query, the `query` judge   (per rollout)
  runtime_factor  the same call's runtime gate (runtime_gate.py, no LLM)

Verdict semantics — the reward-shaping decisions live HERE, hot-updatable:
  * no delivered html          -> ok, query_score 0.0 (failed delivery, not infra).
                                 The floor goes on the SCORE, not on `reward`:
                                 `reward` is the group rank and does not exist
                                 until the siblings land, and a missing score
                                 would make the driver drop the row instead of
                                 punishing it.
  * page unrenderable/blank    -> ok, SCORED LOW. Hanging used to be a drop,
                                 which made it strictly better than a 0 for a
                                 rollout that could tell it was doing badly.
  * query text missing         -> drop (the brief is what reward is measured against)
  * render infra failure       -> drop (page never seen: mask, don't score)
  * the query judge has no verdict -> drop, FAIL-CLOSED. The original
                                 implementation skipped the deduction for
                                 unreadable dimensions, paying 0.5 for a page no
                                 judge could parse — in RL that is a bounty on
                                 breaking the judge.

Concurrency model — TWO bounds, because they fail differently:
  HTTP_CONCURRENCY   in-flight /grade. Acquired in do_POST, NOT on the accept
                     thread: an acquire there costs HTTP_ACQUIRE_TIMEOUT_S of
                     accept() time per overflow connection, which is how a
                     saturated grader stopped answering the requests it could
                     have served. Acquired WITHOUT blocking — a waiting grader
                     holds a handler thread, and handler threads are the
                     headroom below.
  READ_ONLY_HEADROOM extra handler threads beyond that, so /healthz, /version,
                     /api/* and the dashboard still answer while grading is
                     pinned. A grader you cannot observe is one you cannot
                     tune, and a /healthz that 503s under load is worse than no
                     probe (at HTTP_CONCURRENCY=20 every
                     healthz and all 30 probes were rejected). This only holds
                     because the grade bound above never waits.
  thread count must NOT track traffic — a per-request ThreadPoolExecutor(4)
  multiplied threads by 5x and exhausted RLIMIT_NPROC
  HTTP_BACKLOG       kernel listen queue behind that cap. When IT fills the
                     kernel refuses the connection and the gateway 502s — this
                     is how 15% of a run was lost while the service itself
                     dropped 0.27%. Rollouts burst at step boundaries, so size
                     it for a step's concurrency, not the steady state.
                     (HTTP_ACQUIRE_TIMEOUT_S applies to the thread bound only.)
  JUDGE_CONCURRENCY  one process-wide judge THREAD pool shared by all judges —
                     a thread per possible call, so no grade waits for one.
                     MUST sit below the httpx socket ceiling or the queue forms
                     inside this process — see vision.JUDGE_MAX_CONNECTIONS.
  vision.LIMITER     how many of those calls are on the wire to the router at
                     once. ADAPTIVE: the router's throughput collapses past
                     ~1000 concurrent calls (30/s -> 7/s, measured), so a
                     fixed number is either too low on a good day or the
                     outage on a bad one. It shrinks on timeouts / p90 and
                     grows back while the channel answers fast.
  RENDER_CONCURRENCY chromium subprocesses (CPU-bound; the local ceiling)
  LOCALIZE_CONCURRENCY  asset fetch pool (in assets.py)
  Steady-state threads ~= HTTP_CONCURRENCY + JUDGE_CONCURRENCY.

Every queue above is bounded by the request's DEADLINE (client budget, sent in
the request as `budget_s`): a grade waits for a render slot only while a
verdict could still be judged in time, and for a router slot only until its
budget is gone. Past that it drops fast with a queue reason. This is what
turns a step burst of 2048 into a queue rather than a retry storm — a request
that times out on the client is retried INTO the queues that made it late.

The numbers are hardcoded and deliberately NOT env-tunable. HTTP/JUDGE are
sized for this 100-core box; RENDER is sized for CPU headroom and is the one
knob with a measured failure mode above it — see the block above
RENDER_CONCURRENCY. The relationships checked at import are what actually
failed in the outages, not the individual numbers.

Run:  GRADER_MODEL=... LLM_JUDGE_BASE_URL=... LLM_JUDGE_API_KEY=... \
        python -m design_grader.service.server --port 80
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..client import DEFAULT_TIMEOUT_S as _CLIENT_DEFAULT_TIMEOUT_S
from . import assets, extract, group_pick, judges, render, runtime_gate, vision
from .history import RecentGrades

GRADER_VERSION = (
    os.getenv("GRADER_VERSION")
    or subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    or "unknown"
)


# One process, one reward. No --rubric any more: the three pointwise rubrics
# and their formulas are gone, and what is left has a single shape.
# Reported on /version; the training launcher's handshake reads `endpoints` and
# `runtime_gate` from there (recipes/design/webdev/grader_client.py).
REWARD_SEMANTICS = "group_v1:pick-query-runtime"


def _apply_judge_models(overrides: list) -> None:
    """Apply --judge-model DIM=MODEL pairs, then report the effective table.

    Reported at startup because a model swap is invisible from the outside: the
    reward shape does not change, only which endpoint answered. An operator who
    cannot see the table cannot tell a deliberately retargeted judge from a
    stale deployment.
    """
    global MODELS
    for spec in overrides:
        dim, _, model = spec.partition("=")
        if not model or (dim != "all" and dim not in vision.JUDGE_MODELS):
            raise SystemExit(
                f"--judge-model {spec!r}: want DIM=MODEL with DIM one of {sorted(vision.JUDGE_MODELS)} or 'all'"
            )
        if dim == "all":
            for d in vision.JUDGE_MODELS:
                vision.JUDGE_MODELS[d] = model
        else:
            vision.JUDGE_MODELS[dim] = model
    MODELS = _models_for()
    if overrides:
        print("[design-grader] judge models (overridden ones marked *):", flush=True)
        for dim, model in sorted(vision.JUDGE_MODELS.items()):
            mark = "*" if any(o.partition("=")[0] in (dim, "all") for o in overrides) else " "
            print(f"  {mark} {dim:<15} {model}", flush=True)


def _models_for() -> str:
    """The distinct models this service actually calls, for /version.

    Read off JUDGE_MODELS rather than hardcoded, so a --judge-model swap shows
    up here instead of /version quietly claiming a model the process no longer
    calls. Both entries are load-bearing: `query` scores each rollout, `pick`
    ranks the group.
    """
    return " + ".join(sorted({vision.JUDGE_MODELS[d] for d in ("query", "pick")}))


MODELS = _models_for()


RENDER_CONCURRENCY = render.POOL_SIZE
_RENDER_SEMA = threading.Semaphore(RENDER_CONCURRENCY)

RENDER_MIN_BUDGET_S = 60.0
RESPONSE_MARGIN_S = 10.0

WORK_ROOT = os.environ.get("DESIGN_GRADER_WORK") or tempfile.gettempdir()
WORK_MIN_FREE_BYTES = 4 << 30

JUDGE_CONCURRENCY = 6144
_JUDGE_POOL = ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY, thread_name_prefix="judge")

HTTP_CONCURRENCY = 4096
HTTP_BACKLOG = 4096
HTTP_ACQUIRE_TIMEOUT_S = 3.0
READ_ONLY_HEADROOM = 128

if vision.JUDGE_MAX_CONNECTIONS < JUDGE_CONCURRENCY:
    raise RuntimeError(
        f"JUDGE_MAX_CONNECTIONS={vision.JUDGE_MAX_CONNECTIONS} < JUDGE_CONCURRENCY="
        f"{JUDGE_CONCURRENCY}: the excess judge threads would queue on our own "
        f"httpx pool instead of the router. Raise the socket ceiling with the pool."
    )
if JUDGE_CONCURRENCY <= HTTP_CONCURRENCY:
    raise RuntimeError(
        f"JUDGE_CONCURRENCY={JUDGE_CONCURRENCY} must sit above HTTP_CONCURRENCY="
        f"{HTTP_CONCURRENCY}, or requests park on http slots waiting for judges."
    )
if HTTP_BACKLOG < HTTP_CONCURRENCY:
    raise RuntimeError(
        f"HTTP_BACKLOG={HTTP_BACKLOG} < HTTP_CONCURRENCY={HTTP_CONCURRENCY}: "
        f"the kernel would refuse connections the service could have served."
    )
if vision.AdaptiveLimit.CEILING > vision.JUDGE_MAX_CONNECTIONS:
    raise RuntimeError(
        f"AdaptiveLimit.CEILING={vision.AdaptiveLimit.CEILING} > JUDGE_MAX_CONNECTIONS={vision.JUDGE_MAX_CONNECTIONS}"
    )
if RENDER_MIN_BUDGET_S < vision.MIN_ATTEMPT_S:
    raise RuntimeError(
        f"RENDER_MIN_BUDGET_S={RENDER_MIN_BUDGET_S} < vision.MIN_ATTEMPT_S="
        f"{vision.MIN_ATTEMPT_S}: renders would start that no judge can finish."
    )

HISTORY = RecentGrades()
_DASHBOARD_HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")
_t_start = time.time()

DUMP_PATH = os.environ.get("DESIGN_GRADER_DUMP", "").strip()
_DUMP_LOCK = threading.Lock()
_DUMP_FH = None


def _flatten_dims(out: dict) -> dict:
    """Pull the per-judge numbers and rationales up to the top level.

    Flat rather than nested because the consumer is a trend analysis: one row
    per grade, one column per factor, groupby step. A nested `dims` blob would
    make every downstream question a json_normalize away.

    Rubric-agnostic — it walks whatever dims the judge set produced, so a new
    rubric does not need a new flattener. Fields that only some rubrics have
    simply do not appear on the other rows.
    """
    flat = {}
    for dim, v in (out.get("dims") or {}).items():
        if not isinstance(v, dict):
            continue
        for k, val in v.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                flat[f"{dim}.{k}"] = val
    return flat


def _dump_grade(out: dict, *, task_id: str, global_step, query: str) -> None:
    """Append one verdict line. Never raises — a dump failure must not lose a
    grade that already succeeded, and the caller is on the response path."""
    global _DUMP_FH
    if not DUMP_PATH:
        return
    rec = {
        "ts": round(time.time(), 3),
        "task_id": task_id,
        "global_step": global_step,
        "semantics": REWARD_SEMANTICS,
        "status": out.get("status"),
        "query_score": out.get("query_score"),
        "runtime_factor": (out.get("runtime") or {}).get("factor"),
        "reason": out.get("reason"),
        "drop_reason": out.get("drop_reason"),
        "no_delivery": bool(out.get("no_delivery")),
        "unrenderable": bool((out.get("signals") or {}).get("unrenderable")),
        "grader_version": out.get("grader_version"),
        "query": (query or "")[:400],
        "timing": out.get("timing"),
    }
    rec.update(_flatten_dims(out))
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    try:
        with _DUMP_LOCK:
            if _DUMP_FH is None:
                os.makedirs(os.path.dirname(DUMP_PATH) or ".", exist_ok=True)
                _DUMP_FH = open(DUMP_PATH, "a", buffering=1, encoding="utf-8")
            _DUMP_FH.write(line)
    except Exception as e:  # noqa: BLE001 — see docstring: never lose the grade
        print(f"[grade] dump failed ({type(e).__name__}: {e}) -> {DUMP_PATH}", flush=True)


HEALTH_CHROME_MAX = RENDER_CONCURRENCY * 8
HEALTH_THREADS_MAX = (HTTP_CONCURRENCY + READ_ONLY_HEADROOM + JUDGE_CONCURRENCY) * 2
_START_CHROME_WARN = RENDER_CONCURRENCY * 12
_HEALTH_TTL_S = 5.0
_health_cache: dict = {"at": 0.0, "data": None}


def _count_chrome() -> int:
    """chrome-headless processes on this box.

    Each playwright spawns 2-3 of them, so a busy-but-healthy service sits
    around RENDER_CONCURRENCY*2-3. Anything in the hundreds means a leak;
    anything in the tens of thousands means the machine is already gone
    (and, with a non-reaping PID 1, unrecoverable).
    """
    n = 0
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if b"chrome-headless" in f.read():
                        n += 1
            except OSError:
                continue  # process exited mid-scan
    except OSError:
        return -1
    return n


def _thread_count() -> int:
    """Threads of THIS process (kernel count, not just Python's)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return threading.active_count()


def _fd_count() -> int:
    """Open fds of THIS process.

    Reported alongside threads because the two together are the only outside
    view of root cause 3: a judge pool larger than the httpx socket ceiling
    shows as threads climbing while fds stay flat at the ceiling — every
    excess thread parked inside our own process instead of on the router.
    Threads up + fds up = the pool is really working. Threads up + fds flat =
    we are the queue again.
    """
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


_cpu_prev: dict = {"ticks": render._cpu_ticks(), "at": time.time(), "busy": None}


def _cpu_busy() -> float | None:
    """Box-wide busy fraction since the previous health sample.

    Reported because it is the number that decides whether RENDER_CONCURRENCY
    is right: a queue with cpu_busy well under 1.0 is a queue, the same queue
    at cpu_busy ~1.0 is starvation and the unrenderable rate is about to move.
    """
    now, ticks = time.time(), render._cpu_ticks()
    if now - _cpu_prev["at"] >= 1.0:
        _cpu_prev["busy"] = render.cpu_busy_since(_cpu_prev["ticks"])
        _cpu_prev["ticks"], _cpu_prev["at"] = ticks, now
    return None if _cpu_prev["busy"] is None else round(_cpu_prev["busy"], 2)


def _health_probe() -> dict:
    """The slow half of health: everything that reads /proc. Reading a thousand
    /proc/<pid>/cmdline files takes 10ms on an idle box and blocked for >10s
    while 128 chromiums were forking (under soak) — the kernel holds each
    target's mm lock. So this runs on its own thread (see _health_refresher)
    and a probe only ever reads the last result; a probe that stalls behind the
    kernel is a probe that fails exactly when it is needed."""
    chrome, threads = _count_chrome(), _thread_count()
    work_free = shutil.disk_usage(WORK_ROOT).free
    return {
        "ok": chrome >= 0
        and chrome <= HEALTH_CHROME_MAX
        and threads <= HEALTH_THREADS_MAX
        and work_free >= WORK_MIN_FREE_BYTES,
        "threads": threads,
        "chrome": chrome,
        "fds": _fd_count(),
        "cpu_busy": _cpu_busy(),
        "work_root": WORK_ROOT,
        "work_free_gb": round(work_free / 2**30, 1),
        "tmp_free_gb": round(shutil.disk_usage(tempfile.gettempdir()).free / 2**30, 1),
        "mem_available_gb": _mem_available_gb(),
        "probed_at": round(time.time(), 1),
    }


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


def _sweep_stale_workdirs() -> int:
    """Remove leftovers of grades that died without cleaning up (SIGKILL, a
    crashed pod): our own `design_grade_*` under WORK_ROOT and playwright's
    profile dirs under the system tmp. Only prefixes we own, only at startup —
    while serving, every one of these belongs to a grade in flight."""
    n = 0
    for root, prefixes in (
        (WORK_ROOT, ("design_grade_", "design_probe_")),
        (
            tempfile.gettempdir(),
            ("design_grade_", "design_probe_", "playwright_chromiumdev_profile-", "playwright-artifacts-"),
        ),
    ):
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if name.startswith(prefixes):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                n += 1
    return n


def _health_refresher() -> None:
    while True:
        try:
            _health_cache["data"] = _health_probe()
        except Exception as e:  # noqa: BLE001 — a failed probe must not kill the refresher
            print(f"[health] probe failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(_HEALTH_TTL_S)


_health_thread_started = False


def health() -> dict:
    """Liveness + the pressure counters that would have caught this incident early.
    Cheap by construction: the /proc numbers come from the refresher's last
    pass; only the in-process counters are read live."""
    global _health_thread_started
    if not _health_thread_started:
        _health_thread_started = True
        _health_cache["data"] = _health_probe()  # first answer is synchronous
        threading.Thread(target=_health_refresher, daemon=True, name="health").start()
    now = time.time()
    return {
        **_health_cache["data"],
        "uptime_s": round(now - _t_start),
        "chrome_max": HEALTH_CHROME_MAX,
        "threads_max": HEALTH_THREADS_MAX,
        "sockets_max": vision.JUDGE_MAX_CONNECTIONS,
        "render_inflight": RENDER_CONCURRENCY - _RENDER_SEMA._value,
        "render_max": RENDER_CONCURRENCY,
        "render_pool": render.POOL.snapshot(),
        "judge": vision.LIMITER.snapshot(),
    }


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a hard cap on worker threads.

    The slot is taken in `process_request` — BEFORE the thread is spawned — so
    thread count cannot track traffic.

    THE ACQUIRE MUST NOT BLOCK INDEFINITELY. `serve_forever` calls
    `process_request` on the accept thread, so a blocking acquire freezes
    accept() itself once every slot is held. From that moment no connection is
    accepted at all: the kernel backlog fills and then stops completing new
    ones, so the gateway's connect() times out and answers 502 to *everything*
    — including brand-new grades that never reached us. Measured: a
    batch of 64 rollouts against 48 slots stalled the accept loop for tens of
    seconds at a time, while /healthz kept reporting ok:true and 0 drops,
    because the refused connections are invisible from inside.

    The earlier design assumed overflow would "wait in the kernel queue". It
    cannot: a frozen accept loop never calls accept() to drain that queue.

    So: wait a few seconds for a slot, then answer 503 and move on. The client
    already retries any HTTPError, so a fast, honest rejection costs one retry
    and keeps the service answering. Backlog still absorbs sub-second bursts.
    """

    def __init__(self, addr, handler, max_workers: int, backlog: int = HTTP_BACKLOG):
        self.request_queue_size = backlog
        super().__init__(addr, handler)
        self._thread_slots = threading.Semaphore(max_workers + READ_ONLY_HEADROOM)
        self._grade_slots = threading.Semaphore(max_workers)

    def _reject(self, request, client_address) -> None:
        """Answer 503 on a raw socket and close.

        Everything — the counter, the log line, the socket write — happens on
        the throwaway thread. This runs ON the accept loop, and a flush=True
        print there is still stdout I/O: the incident this whole bound exists
        for was accept() starvation, so the accept thread does nothing but
        spawn the thread.
        """

        def _run():
            n = HISTORY.note_rejected()
            print(
                f"[reject] 503 threads exhausted (http={HTTP_CONCURRENCY} "
                f"headroom={READ_ONLY_HEADROOM} total_rejected={n})",
                flush=True,
            )
            body = b'{"error": "grader at capacity, retry"}'
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
            except OSError:
                pass
            finally:
                try:
                    request.close()
                except OSError:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def process_request(self, request, client_address):
        if not self._thread_slots.acquire(timeout=HTTP_ACQUIRE_TIMEOUT_S):
            self._reject(request, client_address)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._thread_slots.release()  # thread never started — hand it back
            try:
                request.close()  # and don't leak the accepted socket
            except OSError:
                pass
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._thread_slots.release()


def grade_task(
    query: str,
    messages: list[dict] | None = None,
    *,
    response: str | None = None,
    workspace_tgz_b64: str | None = None,
    capture=render.capture_hero,
    checks=None,
    budget_s: float | None = None,
    judge_scope: str | None = None,
) -> dict:
    """The whole grading pipeline for one rollout (injectable for tests).

    Three delivery surfaces, first match wins: the agent's real pod workspace
    (`workspace_tgz_b64`, what the pod actually holds at reward time), a
    single-turn chat generation (`response`, html cut from the text), or an
    agent trajectory replay (`messages` alone; offline analysis — replay can
    only guess at scripted edits and build outputs, see extract.py). With a
    workspace, `messages` still feeds the tool-usage signals.

    `judge_scope="query"` is REQUIRED — it is the only grading path this
    service has. It is the group-reward (group_v1) pod call: extract → localize
    → render, then the query judge and the runtime gate. The response carries
    query_score + shot_jpg_b64 + runtime, and `reward=None`: there is no
    per-rollout reward in group mode, because the aesthetic verdict is the
    RELATIVE pick over the whole sibling group (POST /grade_group), judged once
    every sibling has finished. Any other `judge_scope` drops and says so
    rather than falling back — a caller that wanted an absolute per-page score
    must not silently receive a group-relative one under the same field name.

    FAIL-CLOSED. The query judge has to hand back a verdict inside its attempt
    budget; one that does not drops the whole sample (INVALID_REWARD on the
    client, masked from advantage). The implementation this replaced skipped
    the deduction for any dimension it could not read, which paid a partial
    score for a page no judge could parse — in RL that is a bounty on breaking
    the judge.

    The two exceptions, both deliberate:
      * no HTML delivered at all  -> ok, query_score 0.0. A failure to deliver
        must be learned from, not masked.
      * the page rendered blank / unresponsive -> SCORED (low). See render.py:
        hanging used to be a drop, which made it strictly better than a 0.

    `budget_s` is how long the CLIENT will wait for this answer (it sends its
    own timeout in the request; the client default when absent). Everything
    that can queue — the render slot, the judge slot, each judge attempt — is
    bounded by the deadline it implies, so a grade that cannot answer in time
    drops early and cheaply rather than finishing into a closed socket while
    the client retries a copy of it into the same queues. None = unbounded
    (tests, offline tools).
    """
    deadline = None if budget_s is None else time.time() + budget_s - RESPONSE_MARGIN_S
    out = _grade_task(
        query,
        messages,
        response=response,
        workspace_tgz_b64=workspace_tgz_b64,
        capture=capture,
        checks=checks,
        deadline=deadline,
        judge_scope=judge_scope,
    )
    return out


def _grade_task(
    query: str,
    messages: list[dict] | None,
    *,
    response: str | None,
    workspace_tgz_b64: str | None = None,
    capture,
    checks,
    deadline: float | None,
    judge_scope: str | None = None,
) -> dict:
    """The pipeline itself; see grade_task for the contract."""
    if judge_scope != "query":
        # Absent used to select the pointwise rubric — five judges and an absolute
        # per-page score. That surface is gone, so absent is an error rather than a
        # default: answering it with the group half's query score would hand back a
        # number on a different scale under the same field name.
        return {
            "status": "drop",
            "reward": None,
            "drop_reason": (
                f"judge_scope: want \"query\", got {judge_scope!r}. This service serves the "
                "webdev group reward only (POST /grade judge_scope=query for the per-rollout "
                "query score and shot, POST /grade_group for the relative pick)."
            ),
            "signals": {},
            "timing": {},
        }
    dims = ("query",)
    needs_q = frozenset({"query"})
    checks = checks or {"query": judges.check_query}
    t0 = time.perf_counter()
    timing: dict = {}

    def left() -> float:
        return float("inf") if deadline is None else deadline - time.time()

    if workspace_tgz_b64 is not None:
        try:
            ex = extract.extract_workspace(workspace_tgz_b64)
        except Exception as e:  # noqa: BLE001 — an undecodable capture is infra, never a score
            return {
                "status": "drop",
                "reward": None,
                "drop_reason": f"workspace untar: {type(e).__name__}: {e}"[:300],
                "signals": {"mode": "workspace"},
                "timing": timing,
            }
        n_writes, n_edits, n_bash = extract._count_tools(messages or [])
        stats = {
            "mode": "workspace",
            "n_writes": n_writes,
            "n_edits": n_edits,
            "n_bash": n_bash,
            "workspace_files": ex.fidelity.get("workspace_files", 0),
            "n_files": len(ex.files),
        }
        no_delivery_reason = "no_entry: no html file in the agent's workspace"
    elif response is not None:
        ex = extract.extract_response(response)
        stats = {"mode": "chat", "response_chars": len(response)}
        no_delivery_reason = "no_html: response contains no html document"
    else:
        try:
            ex = extract.extract_traj(messages)
        except Exception as e:  # noqa: BLE001 — broken VFS runtime is infra, never a score
            return {
                "status": "drop",
                "reward": None,
                "drop_reason": f"vfs rebuild: {type(e).__name__}: {e}"[:300],
                "signals": {"mode": "traj"},
                "timing": timing,
            }
        stats = {
            "mode": "traj",
            "n_writes": ex.n_writes,
            "n_edits": ex.n_edits,
            "n_edit_misses": ex.n_edit_misses,
            "n_bash": ex.n_bash,
            "vfs_fidelity": ex.fidelity,
            "n_files": len(ex.files),
        }
        no_delivery_reason = "no_entry: no renderable file rebuilt from the trajectory"
    timing["extract_s"] = round(time.perf_counter() - t0, 3)
    if ex.entry is None:
        # ok, not drop: nothing was delivered, and that is the rollout's own doing
        # rather than an infrastructure failure — masking it would pay the same as a
        # page nobody could grade. `query_score` is 0.0 rather than None so the
        # driver's query_deduct reads a real floor; reward stays None because a rank
        # still cannot exist until the siblings are in.
        return {
            "status": "ok",
            "reward": None,
            "judge_scope": "query",
            "query_score": 0.0,
            "runtime": None,
            "dims": None,
            "no_delivery": True,
            "reason": no_delivery_reason,
            "signals": stats,
            "shot_jpg_b64": None,
            "timing": timing,
        }

    t = time.perf_counter()
    if not _RENDER_SEMA.acquire(timeout=max(0.0, left() - RENDER_MIN_BUDGET_S)):
        timing["render_s"] = round(time.perf_counter() - t, 3)
        return {
            "status": "drop",
            "reward": None,
            "drop_reason": f"queue: no render slot with {RENDER_MIN_BUDGET_S:g}s "
            f"of budget left (waited {timing['render_s']:.0f}s)",
            "signals": stats,
            "timing": timing,
        }
    free = shutil.disk_usage(WORK_ROOT).free
    if free < WORK_MIN_FREE_BYTES:
        _RENDER_SEMA.release()
        return {
            "status": "drop",
            "reward": None,
            "drop_reason": f"disk: {free >> 30}GB free under {WORK_ROOT}, refusing to materialize",
            "signals": stats,
            "timing": timing,
        }
    workdir = tempfile.mkdtemp(prefix="design_grade_", dir=WORK_ROOT)
    try:
        dist = os.path.join(workdir, "dist")
        try:
            os.makedirs(dist)
            extract.materialize(ex, dist)
            tl = time.perf_counter()
            try:
                stats.update(assets.localize(dist))
            except Exception as e:  # noqa: BLE001 — un-localized page still renders (slower/holier)
                stats["localize_error"] = f"{type(e).__name__}: {e}"[:120]
            timing["localize_s"] = round(time.perf_counter() - tl, 3)
            page = os.path.join(workdir, "page.jpg")
            respawns0 = render.POOL.snapshot()["respawns"]
            ok, reason, signals = capture(dist, page)
        finally:
            _RENDER_SEMA.release()
        timing["render_s"] = round(time.perf_counter() - t, 3)
        stats.update(signals)
        if not ok:
            return {
                "status": "drop",
                "reward": None,
                "drop_reason": f"render: {reason}",
                "signals": stats,
                "timing": timing,
            }
        if signals.get("unrenderable") and render.POOL.snapshot()["respawns"] > respawns0:
            return {
                "status": "drop",
                "reward": None,
                "drop_reason": "render: unrenderable while the pool was respawning a browser (infra, not the page)",
                "signals": stats,
                "timing": timing,
            }
        page_b64 = render.to_jpeg_b64(page)
        runtime = None
        if judge_scope == "query":
            tr = time.perf_counter()
            runtime = runtime_gate.verdict(ex.entry_text(), signals)
            timing["runtime_s"] = round(time.perf_counter() - tr, 3)
        return _judge(
            query,
            page_b64,
            dims,
            needs_q,
            checks,
            deadline,
            stats,
            timing,
            t0,
            runtime,
            judge_scope=judge_scope,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _judge(
    query,
    page_b64,
    dims,
    needs_q,
    checks,
    deadline,
    stats,
    timing,
    t0,
    runtime=None,
    judge_scope: str | None = None,
):
    """Judges (and the runtime gate) -> verdict. Split from
    _grade_task so the files' lifetime (above) and the scoring (here) read as two
    straight lines."""

    if not (query or "").strip():
        return {
            "status": "drop",
            "reward": None,
            "drop_reason": "query: empty — cannot judge requirement coverage against a missing brief",
            "signals": stats,
            "shot_jpg_b64": page_b64,
            "timing": timing,
        }

    t = time.perf_counter()

    def _timed(fn, *args):
        vision.set_deadline(deadline)
        s = time.perf_counter()
        try:
            out = fn(*args)
        finally:
            vision.set_deadline(None)  # pool threads are reused
        return out, round(time.perf_counter() - s, 3)

    fut, got, err = {}, {}, {}
    for dim in dims:
        args = (page_b64, query) if dim in needs_q else (page_b64,)
        fut[dim] = _JUDGE_POOL.submit(_timed, checks[dim], *args)
    for dim in dims:
        (got[dim], err[dim]), timing[f"j_{dim}_s"] = fut[dim].result()
    timing["judges_s"] = round(time.perf_counter() - t, 3)
    timing["total_s"] = round(time.perf_counter() - t0, 3)

    # Unconditional: _grade_task drops anything but judge_scope="query" before a
    # single judge is dispatched, so there is one scoring shape to produce here.
    q = got["query"]
    if q is None:
        # FAIL-CLOSED. A rollout whose query judge never answered is masked from
        # advantage, not scored on the surviving factors — see grade_task.
        return {
            "status": "drop",
            "reward": None,
            "drop_reason": f"judge: query={err['query'] or 'no verdict'}"[:300],
            "signals": stats,
            "shot_jpg_b64": page_b64,
            "timing": timing,
        }
    rt_txt = f" runtime x{runtime['factor']} ({runtime['why'][:120]})" if runtime is not None else ""
    return {
        "status": "ok",
        "judge_scope": "query",
        # No reward, by contract: this half judges one rollout in isolation and the
        # group reward is a RANK, which cannot exist until every sibling has been
        # graded. The driver computes it in group_pick.reward_one from query_score,
        # the runtime factor, and the group's pick. Any float here would be a number
        # on a different scale wearing the field name the trainer reads.
        "reward": None,
        # Three placeholders the pointwise rubrics used to fill. Kept as explicit
        # Nones rather than dropped: rl.py / chat.py / history.py all read them off
        # the verdict, and a missing key reads as "old grader" rather than "no such
        # number any more".
        "visual": None,
        "aes_tier": None,
        "breakdown": None,
        "query_score": q["query_score"],
        "runtime": runtime,
        "dims": {"query": q, "runtime": runtime},
        "reason": f"query x{q['query_score']} (group mode: aesthetics judged per group){rt_txt}",
        "signals": stats,
        "shot_jpg_b64": page_b64,
        "timing": timing,
    }


class Handler(BaseHTTPRequestHandler):
    """Handler for BoundedThreadingHTTPServer — do_POST takes a slot from
    ``self.server._grade_slots``, so this is not usable with a bare
    ThreadingHTTPServer."""

    def log_message(self, *a):  # keep stdout for structured lines only
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            with open(_DASHBOARD_HTML, "rb") as f:
                self._bytes(f.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/recent"):
            limit = 100
            if "limit=" in self.path:
                try:
                    limit = min(int(self.path.split("limit=")[1].split("&")[0]), 300)
                except ValueError:
                    pass
            self._json(HISTORY.recent(limit) | {"version": GRADER_VERSION, "model": MODELS})
        elif self.path.startswith("/api/shot/"):
            shot = HISTORY.shot(self.path.rsplit("/", 1)[-1].split("?")[0])
            if shot is None:
                return self._json({"error": "no shot"}, 404)
            self._bytes(shot, "image/jpeg")
        elif self.path.startswith("/api/raw/"):
            raw = HISTORY.raw(self.path.rsplit("/", 1)[-1].split("?")[0])
            if raw is None:
                return self._json({"error": "no raw output"}, 404)
            self._bytes(raw.encode(), "text/plain; charset=utf-8")
        elif self.path == "/healthz":
            h = health()
            self._json(h, code=200 if h["ok"] else 503)
        elif self.path == "/version":
            self._json(
                {
                    "grader_version": GRADER_VERSION,
                    "model": MODELS,
                    "reward_semantics": REWARD_SEMANTICS,
                    "endpoints": ["/grade", "/grade_group"],
                    "judge_scopes": ["query"],
                    "runtime_gate": True,
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def _reject_grade(self) -> None:
        """No grading slot: 503 and count it. Runs in the handler thread, so it
        can speak HTTP properly — unlike _reject, which has no handler yet."""
        n = HISTORY.note_rejected()
        print(
            f"[reject] 503 grade at capacity (http={HTTP_CONCURRENCY} "
            f"headroom={READ_ONLY_HEADROOM} total_rejected={n})",
            flush=True,
        )
        self._json({"error": "grader at capacity, retry"}, 503)

    def do_POST(self):
        if self.path not in ("/grade", "/grade_group"):
            return self._json({"error": "not found"}, 404)
        if not self.server._grade_slots.acquire(blocking=False):
            return self._reject_grade()
        try:
            if self.path == "/grade_group":
                self._grade_group()
            else:
                self._grade()
        finally:
            self.server._grade_slots.release()

    def _grade_group(self) -> None:
        """POST /grade_group — webdev group_v1: the relative aesthetic pick over
        one GRPO group's screenshots (see group_pick.py for the reward).

            {"task_id", "query", "shots_jpg_b64": [b64, ...],
             "query_scores": [float|null, ...] | null,     # pod-judged; null entries re-judged
             "runtime_factors": [0|1|null, ...] | null,     # pod-side runtime gate; null/absent = 1.0
             "seed": int | null, "budget_s": float, "global_step": int|null}
          -> {"status": "ok", "items": [...], "group_ok", "rounds_ok", ...}   (group_pick.grade_group)
          -> {"status": "drop", "reward": null, "drop_reason": ...}

        No render here — the shots ARE this service's own renders, returned as
        shot_jpg_b64 by the per-rollout judge_scope="query" grades and landed
        beside each trajectory dump. One request fans out PICK_ROUNDS pick calls
        plus any query re-judges on the shared judge pool, so it costs one grade
        slot but ~9 judge calls; the pool is sized for that (a full-rubric grade
        is already 5).
        """
        t0 = time.perf_counter()
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            task_id = str(payload.get("task_id", ""))
            query = str(payload.get("query", ""))
            step = payload.get("global_step")
            budget = float(payload.get("budget_s") or _CLIENT_DEFAULT_TIMEOUT_S)
            shots = payload.get("shots_jpg_b64")
            if not isinstance(shots, list) or len(shots) < 2:
                raise ValueError(f"shots_jpg_b64 must be a list of >=2 b64 JPEGs, got {type(shots).__name__}")
            deadline = time.time() + budget - RESPONSE_MARGIN_S
            out = group_pick.grade_group(
                shots,
                query,
                payload.get("query_scores"),
                runtime_factors=payload.get("runtime_factors"),
                pool=_JUDGE_POOL,
                deadline=deadline,
                seed=payload.get("seed"),
            )
            out["status"] = "ok"
        except Exception as e:  # noqa: BLE001 — a bad request must answer, not hang the socket
            out = {"status": "drop", "reward": None, "drop_reason": f"server error: {type(e).__name__}: {e}"[:300]}
            task_id, query, step = "?", "", None
        out["task_id"] = task_id
        out["grader_version"] = GRADER_VERSION
        out["latency_s"] = round(time.perf_counter() - t0, 2)
        items = out.get("items") or []
        print(
            f"[grade_group] step={step} task={task_id} status={out['status']} "
            f"n={len(items)} group_ok={out.get('group_ok')} rounds_ok={out.get('rounds_ok')} "
            f"ncd_zeroed={out.get('ncd_zeroed')} runtime_gated={out.get('n_runtime_gated')} "
            f"rewards={[it.get('reward') for it in items]} "
            f"drop={str(out.get('drop_reason') or '')[:180]} latency={out['latency_s']}s",
            flush=True,
        )
        try:
            self._json(out)
        except (BrokenPipeError, ConnectionResetError):
            n = HISTORY.note_client_gone()
            print(
                f"[grade_group] task={task_id} response lost: peer closed the socket "
                f"after {out['latency_s']}s (#{n} such)",
                flush=True,
            )

    def _grade(self) -> None:
        payload = None
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            task_id = str(payload.get("task_id", ""))
            query = str(payload.get("query", ""))
            step = payload.get("global_step")
            budget = float(payload.get("budget_s") or _CLIENT_DEFAULT_TIMEOUT_S)
            out = grade_task(
                query,
                payload.get("messages") or [],
                response=payload.get("response"),
                workspace_tgz_b64=payload.get("workspace_tgz_b64"),
                budget_s=budget,
                judge_scope=payload.get("judge_scope"),
            )
        except Exception as e:  # noqa: BLE001 — a bad request must answer, not hang the socket
            out = {"status": "drop", "reward": None, "drop_reason": f"server error: {type(e).__name__}: {e}"[:300]}
            task_id, query, step = "?", "", None
        out["task_id"] = task_id
        out["grader_version"] = GRADER_VERSION
        rt = (out.get("runtime") or {}).get("factor")
        print(
            f"[grade] step={step} task={task_id} status={out['status']} "
            f"query={out.get('query_score')} runtime={rt} "
            f"drop={str(out.get('drop_reason') or '')[:180]} "
            f"timing={out.get('timing')}",
            flush=True,
        )
        _dump_grade(out, task_id=task_id, global_step=step, query=query)
        HISTORY.add(
            {k: v for k, v in out.items() if k != "shot_jpg_b64"} | {"query": query[:800]},
            out.get("shot_jpg_b64"),
            raw=payload.get("response") if isinstance(payload, dict) else None,
        )
        try:
            self._json(out)
        except (BrokenPipeError, ConnectionResetError):
            t = (out.get("timing") or {}).get("total_s")
            n = HISTORY.note_client_gone()
            print(
                f"[grade] task={task_id} response lost: peer closed the socket "
                f"after {t}s of judging (#{n} such; if that is at/over the "
                f"client budget the client is retrying into a saturated pool)",
                flush=True,
            )


_ENV_DEFAULTS = (
    ("GRADER_MODEL", "<unset>", "model every judge rides unless --judge-model overrides it"),
    ("LLM_JUDGE_BASE_URL", "<unset>", "openai-style endpoint"),
    ("JUDGE_PAUSE_MAX_S", "30", "process-wide cooldown ceiling"),
    ("RENDER_TIMEOUT_S", "120", "whole capture subprocess"),
    ("LOCALIZE_BUDGET_S", "20", "asset fetch budget"),
    ("DASHBOARD_MAX_RECORDS", "300", "ring buffer"),
    ("DESIGN_GRADER_DUMP", "<empty=disabled>", "append-only per-grade jsonl: the durable sub-score record"),
    ("DESIGN_GRADER_WORK", "<system tmp>", "where grades materialize (NOT the 10GB root overlay — see WORK_ROOT)"),
    ("PLAYWRIGHT_BROWSERS_PATH", "<shared disk>", "chromium install"),
    ("DESIGN_GRADER_FONTCONF", "<fonts.conf>", "CJK fonts; missing = tofu on ~11% of pages"),
    ("DESIGN_GRADER_ASSETS", "<asset cache>", "localized asset cache"),
    ("RENDER_PROXY", "<empty=direct>", "comma-separated failover chain for external assets"),
    ("http_proxy", "<box proxy>", "asset localization only — judges bypass it"),
    ("no_proxy", "localhost,...", "must cover the judge endpoint or calls go through the proxy"),
    ("GRADER_VERSION", "<git sha>", "reported on /version"),
)


def _env_audit() -> None:
    """Print every env-derived setting and whether it is inherited or defaulted.

    A stale export is not an error — an override is a legitimate escape hatch —
    but it must not be a surprise.
    """
    inherited = []
    print("[design-grader] env-derived settings (inherited overrides marked *):", flush=True)
    for name, default, why in _ENV_DEFAULTS:
        v = os.getenv(name)
        mark = "*" if v is not None else " "
        if v is not None:
            inherited.append(name)
        shown = v if v is not None else default
        if "KEY" in name:
            shown = "(set)" if v else default
        print(f"  {mark} {name:<26} {shown:<46} {why}", flush=True)
    if inherited:
        print(f"[design-grader] {len(inherited)} inherited from this shell: {', '.join(inherited)}", flush=True)


def judge_worst_case_s() -> float:
    """Worst-case wall clock for ONE judge: every attempt, plus everything that
    can sleep before or after it.

    ATTEMPTS x JUDGE_TIMEOUT_S alone is not the budget, and believing it was is
    how 180x3=540s shipped against a 600s client budget and then measured 12.4%
    of 80k grades finishing past it. The judge loop can sleep BEFORE each
    attempt (the process-wide cooldown, capped at JUDGE_PAUSE_MAX_S, plus up to
    2s of jitter) and AFTER a failure (exponential backoff, 2 * 2**attempt).

    So: attempts x (timeout + cooldown + jitter) + the backoff sleeps between
    them. Render and localize sit on top of this in total_s, but they are single
    digits and the client budget has to clear them too.
    """
    per = vision.JUDGE_TIMEOUT_S + vision.JUDGE_PAUSE_MAX_S + 2.0  # +2s jitter
    backoff = sum(min(60, 2 * 2**a) for a in range(max(0, vision.ATTEMPTS - 1)))
    return vision.ATTEMPTS * per + backoff


def _check_judge_budget() -> None:
    """Refuse to start when the judge budget cannot fit under the client's.

    Not hypothetical: at 480s x 1 against a 300s client budget the service was
    finishing into a socket nobody read, while holding a judge slot and an HTTP
    slot for the full 480s — and the first version of THIS check counted only
    `timeout x attempts`, which is how the 636s worst case got through against
    600s. Kept out of main() so the import path and the arithmetic are both
    reachable from a test — the first version of this lived inline in main()
    and shipped with a wrong relative import, which nobody noticed until the
    first real launch.
    """
    from ..client import DEFAULT_TIMEOUT_S

    worst = judge_worst_case_s()
    if worst >= DEFAULT_TIMEOUT_S:
        raise SystemExit(
            f"judge budget overruns the client: {vision.ATTEMPTS} attempts of "
            f"{vision.JUDGE_TIMEOUT_S:g}s + cooldown/backoff = {worst:g}s "
            f">= DEFAULT_TIMEOUT_S={DEFAULT_TIMEOUT_S:g}s.\n"
            f"  The last attempt's verdict would land in a socket nobody reads while it\n"
            f"  still holds a judge slot and an HTTP slot. Lower JUDGE_TIMEOUT_S or ATTEMPTS,\n"
            f"  or raise the client budget past judge_worst_case_s()."
        )


def _disable_core_dumps() -> None:
    """RLIMIT_CORE=0 for this process and everything it spawns.

    A chrome-headless crash writes its core to the cwd, and a full-page render
    of a 12000px canvas is a 5-10 GB process. Nine of them (65 GB) were found
    in the repo directory — each one a 10 GB write to a network
    disk at the exact moment the box was already struggling, and none of them
    ever read. Children inherit rlimits, so one call here covers chromium.
    """
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError) as e:
        print(
            f"[design-grader] WARNING: could not disable core dumps ({e}); "
            f"a chrome crash will write a multi-GB core to the cwd",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument(
        "--max-threads",
        type=int,
        default=HTTP_CONCURRENCY,
        help="in-flight /grade cap (also the grade-slot bound). Overflow is shed with a 503, not queued.",
    )
    ap.add_argument(
        "--judge-model",
        action="append",
        default=[],
        metavar="DIM=MODEL",
        help="retarget one judge, or all of them with DIM=all. Repeatable. "
        "Every judge rides the OpenAI-format endpoint at LLM_JUDGE_BASE_URL, "
        "so this only picks which model answers.",
    )
    ap.add_argument(
        "--dump",
        default=None,
        metavar="FILE",
        help="append-only per-grade jsonl (every sub-score, rationale, "
        "drop_reason, timing, training step). Overrides "
        'DESIGN_GRADER_DUMP; "" disables. Unset and no env var = no dump.',
    )
    args = ap.parse_args()
    if args.dump is not None:
        global DUMP_PATH
        DUMP_PATH = args.dump.strip()
        os.environ["DESIGN_GRADER_DUMP"] = DUMP_PATH
    _apply_judge_models(args.judge_model)
    _env_audit()
    os.makedirs(WORK_ROOT, exist_ok=True)
    swept = _sweep_stale_workdirs()
    print(
        f"[design-grader] work root {WORK_ROOT} "
        f"({shutil.disk_usage(WORK_ROOT).free >> 30}GB free; swept {swept} stale dirs)",
        flush=True,
    )
    shm = shutil.disk_usage("/dev/shm")
    print(
        f"[design-grader] /dev/shm {shm.total >> 30}GB total, {shm.free >> 30}GB free -> chromium "
        f"{'uses /dev/shm' if render.dev_shm_flag() == '0' else 'keeps shared memory on local tmp (--disable-dev-shm-usage)'}; "
        f"local tmp {shutil.disk_usage(tempfile.gettempdir()).free >> 30}GB free",
        flush=True,
    )
    _check_judge_budget()
    _disable_core_dumps()

    # A judge with no model answers nothing, and a judge with no key 401s on every
    # grade while /healthz still reports healthy -- so both are startup failures.
    unset = sorted(d for d, m in vision.JUDGE_MODELS.items() if not m)
    if unset:
        raise SystemExit(
            f"no model configured for judge(s): {', '.join(unset)}\n"
            "  set GRADER_MODEL for all of them, or --judge-model DIM=MODEL per judge"
        )
    if not os.getenv("LLM_JUDGE_API_KEY"):
        raise SystemExit(
            "missing judge key: LLM_JUDGE_API_KEY\n"
            "  every judge rides the OpenAI-format endpoint at LLM_JUDGE_BASE_URL"
        )
    # Both judge prompts, checked eagerly: they are read lazily at first use, so a
    # missing file would otherwise surface as a drop storm mid-run instead of a
    # refusal to start. Two, because the reward has exactly two LLM inputs —
    # query_fit for the per-rollout score, group_pick for the group's ranking.
    for name in ("query_fit.md", "group_pick.md"):
        if not (judges.PROMPTS / name).is_file():
            raise SystemExit(f"missing judge prompt: {name}")
    extract._node()  # fail fast if the trajectory VFS rebuild has no node runtime
    pre = _count_chrome()
    if pre > _START_CHROME_WARN:
        print(
            f"[design-grader] WARNING: {pre} chrome-headless already running on this box "
            f"(render concurrency is {RENDER_CONCURRENCY}, so a busy service shows "
            f"~{RENDER_CONCURRENCY * 2}-{RENDER_CONCURRENCY * 3}). That is a leak from an "
            f"earlier run — on a container whose PID 1 does not reap, those pids are "
            f"unrecoverable. Clean the box or use a fresh pod.",
            flush=True,
        )
    print(
        f"[design-grader] v={GRADER_VERSION} model={MODELS} semantics={REWARD_SEMANTICS} assets={assets.ASSETS}",
        flush=True,
    )
    lim = vision.AdaptiveLimit
    print(
        f"[design-grader] concurrency: render={RENDER_CONCURRENCY} (pool of long-lived chromium "
        f"workers, launches paced {1 / render.LAUNCH_SPACING_S:g}/s, recycled every "
        f"{render.MAX_JOBS_PER_BROWSER} pages) http={args.max_threads} "
        f"headroom={READ_ONLY_HEADROOM} judge_threads={JUDGE_CONCURRENCY} "
        f"judge_limit={lim.INITIAL} (adaptive {lim.FLOOR}..{lim.CEILING}) "
        f"sockets={vision.JUDGE_MAX_CONNECTIONS} backlog={HTTP_BACKLOG} "
        f"acquire_timeout={HTTP_ACQUIRE_TIMEOUT_S}s "
        f"localize={assets.LOCALIZE_CONCURRENCY}",
        flush=True,
    )
    print(
        f"[design-grader] budget: client default {_CLIENT_DEFAULT_TIMEOUT_S:g}s, "
        f"render admitted with >={RENDER_MIN_BUDGET_S:g}s left, "
        f"judge attempt with >={vision.MIN_ATTEMPT_S:g}s left, "
        f"judge timeout {vision.JUDGE_TIMEOUT_S:g}s x {vision.ATTEMPTS}",
        flush=True,
    )
    print(f"[design-grader] health ceilings: chrome<={HEALTH_CHROME_MAX} threads<={HEALTH_THREADS_MAX}", flush=True)
    print(f"[design-grader] http://{args.host}:{args.port}", flush=True)
    render.POOL.prewarm()
    BoundedThreadingHTTPServer(
        (args.host, args.port), Handler, max_workers=args.max_threads, backlog=HTTP_BACKLOG
    ).serve_forever()


if __name__ == "__main__":
    main()
