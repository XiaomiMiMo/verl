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
"""Render the rebuilt site and capture a FULL-PAGE screenshot.

Consolidates the hardening of two independent capture stacks that predate it.

PROCESS MODEL: a pool of long-lived render WORKERS, each one
python process holding one chromium, fed jobs over a pipe. It used to be one
subprocess per capture — python start, playwright import, node driver, chromium
launch, render, teardown — and that churn, not CPU, was the render ceiling:

    measured on the 100-core box, 128 captures at once, cpu 40% busy
      subprocess per capture   per-capture wall 6s alone -> 9s @64 -> 19s @128
                               throughput ~5/s at BOTH 64 and 128 slots
      one browser per worker   per-page wall 3.6s @64 AND @128 — no inflation

    and 64+ chromiums launched in the same instant fail nondeterministically:
      simultaneous launches     64 -> 1-25 failed, 128 -> 93-118 failed
      paced at <=50/s           0 failed
    with `Page.captureScreenshot: Unable to capture screenshot` — the capture
    then wrote a blank frame and the sample was SCORED as an empty page.

So: workers are launched once, paced (LAUNCH_SPACING_S), and each renders many
pages with a fresh browser context per job. Isolation moves from "one process
per capture" to "one process per worker": a wedged page hits its job timeout,
the pool kills that worker's whole process group and respawns it, and nothing
else notices. Workers are recycled after MAX_JOBS_PER_BROWSER pages so a slow
chromium leak cannot grow for days.

Full-page rather than first-viewport: both judges this service runs (the
per-rollout `query` coverage judge and the in-group relative `pick`) were
calibrated on whole-page captures, and a dashboard's charts or a landing page's
second fold are exactly where their evidence lives. Height is capped at
MAX_PAGE_H — a pathological page must not turn into a 200MB bitmap.

Hard-won constraints carried over, each with an incident behind it:
  * scroll-behavior:auto first, scroll-through for reveal-on-scroll, freeze
    animations, re-top before the shot;
  * EVERY in-page wait is bounded (document.fonts.ready can hang forever);
  * CJK fonts: without them ~11% of pages render tofu blocks and read as broken
    — FONTCONFIG_FILE is honored (deploy note in README);
  * background-attachment:fixed paints only the first viewport under
    captureBeyondViewport, so the bottom half comes out black — FREEZE_CSS
    forces it to scroll;
  * report a render verdict so "we never saw the page" is a DROP, not a score.

Verdict semantics for RL (differs from the offline pipeline in TWO cases):
  * `blank` with a clean load is NOT an infra drop — assets are localized and
    the site is served on loopback, so a blank page that loaded fine is an
    empty delivery and must be SCORED (low), not masked;
  * a page that never finishes loading is also NOT a drop. It used to be
    `load_failed` → drop → adv masked, which made hanging strictly better than
    scoring 0 for a rollout that could predict it was doing badly
    (`<script>while(1){}</script>` turned a negative advantage into 0). We now
    capture whatever rendered and let the judges score it.
  * an `unrenderable` verdict taken while the BOX was saturated is not
    believed (STARVED_BUSY_FRAC) — that is infra and drops.
"""

from __future__ import annotations

import base64
import collections
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

VIEW_W, VIEW_H = 1440, 900
MAX_PAGE_H = 12000

POOL_SIZE = int(os.getenv("DESIGN_GRADER_RENDER_POOL", "128"))
LAUNCH_SPACING_S = 0.1
MAX_JOBS_PER_BROWSER = 200
WORKER_READY_TIMEOUT_S = 90.0

DEV_SHM_MIN_BYTES = 32 << 30


def dev_shm_flag() -> str:
    """'1' = pass --disable-dev-shm-usage (small /dev/shm), '0' = let chromium use it."""
    try:
        free = shutil.disk_usage("/dev/shm").free
    except OSError:
        return "1"
    return "0" if free >= DEV_SHM_MIN_BYTES else "1"


_CAPTURE_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "DESIGN_GRADER_DISABLE_DEV_SHM": dev_shm_flag(),
}

STARVED_BUSY_FRAC = 0.90

_WORKER_SCRIPT = r'''
import functools, http.server, json, os, socketserver, sys, threading

_FC = os.environ.get("DESIGN_GRADER_FONTCONF", "")
if _FC and os.path.isfile(_FC):
    os.environ.setdefault("FONTCONFIG_FILE", _FC)

VIEW_W, VIEW_H = 1440, 900
MAX_PAGE_H = 12000
LOAD_TIMEOUT_MS = 30000
NETIDLE_MS = 8000
POST_LOAD_MS = 1500
SETTLE_BUDGET_MS = 9000
JPEG_Q = 75

PROXY_CHAIN = [s.strip() for s in os.environ.get("RENDER_PROXY", "").split(",")]
FATAL_NET = ("ERR_PROXY_CONNECTION_FAILED", "ERR_TUNNEL_CONNECTION_FAILED",
             "ERR_SOCKS_CONNECTION_FAILED", "ERR_NAME_NOT_RESOLVED",
             "ERR_INTERNET_DISCONNECTED", "ERR_CONNECTION_TIMED_OUT",
             "ERR_CONNECTION_REFUSED", "ERR_CONNECTION_RESET", "ERR_TIMED_OUT",
             "ERR_EMPTY_RESPONSE")

SETTLE_JS = """async (budgetMs) => {
  const t0 = Date.now();
  const left = () => budgetMs - (Date.now() - t0);
  const nap = ms => new Promise(r => setTimeout(r, Math.max(0, Math.min(ms, left()))));
  const done = {fonts: false, scrolled_to: 0, completed: false};
  try {
    document.documentElement.style.setProperty('scroll-behavior', 'auto', 'important');
    if (document.body) document.body.style.setProperty('scroll-behavior', 'auto', 'important');
  } catch (e) {}
  if (document.fonts && document.fonts.ready) {          // can hang forever: race it
    try {
      await Promise.race([document.fonts.ready, nap(2500)]);
      done.fonts = document.fonts.status === 'loaded';
    } catch (e) {}
  }
  const H = () => Math.max(document.body ? document.body.scrollHeight : 0,
                           document.documentElement.scrollHeight);
  const step = Math.max(400, Math.floor(window.innerHeight * 0.8));
  for (let y = 0; y <= Math.min(H(), 30000); y += step) {
    if (left() < 1200) break;                            // reserve budget for the re-top
    window.scrollTo(0, y);
    done.scrolled_to = y;
    await nap(110);
  }
  if (left() > 600) { window.scrollTo(0, H()); await nap(300); }
  window.scrollTo(0, 0);
  await nap(250);
  done.completed = left() > 0;
  return done;
}"""

FREEZE_CSS = ("*,*::before,*::after{animation-delay:0s !important;animation-duration:.01s !important;"
              "transition-delay:0s !important;transition-duration:.01s !important;"
              "background-attachment:scroll !important;scroll-behavior:auto !important;}")

PROBE_JS = """() => {
  const de = document.documentElement, imgs = [...document.images], vh = window.innerHeight;
  const hero = imgs.filter(i => i.getBoundingClientRect().top < vh);
  const txt = ((document.body && document.body.innerText) || '').trim();
  return {page_h: de.scrollHeight, horizontal_overflow: de.scrollWidth > de.clientWidth + 4,
          total_images: imgs.length,
          broken_images: imgs.filter(i => i.complete && i.naturalWidth === 0).length,
          hero_images: hero.length,
          broken_images_hero: hero.filter(i => i.complete && i.naturalWidth === 0).length,
          body_text_len: txt.length, n_elements: document.querySelectorAll('body *').length,
          styled: !!document.querySelector('style, link[rel=stylesheet]')};
}"""

LAUNCH_ARGS = ["--no-sandbox", "--use-gl=angle",
               "--enable-webgl", "--ignore-gpu-blocklist",
               "--enable-unsafe-swiftshader", "--hide-scrollbars"]
if os.environ.get("DESIGN_GRADER_DISABLE_DEV_SHM", "1") == "1":
    LAUNCH_ARGS.insert(1, "--disable-dev-shm-usage")


def flat_ratio(png_path):
    from PIL import Image
    im = Image.open(png_path).convert("RGB").resize((160, 100))
    counts = im.getcolors(160 * 100) or []
    return (max(c for c, _ in counts) / (160 * 100)) if counts else 1.0


def _blank(out_path):
    """A neutral frame for a page we could not render.

    Deliberately featureless: the judges see "nothing rendered", which is the
    truth, and score it the way they score an empty delivery. Anything richer
    (a text banner saying "unrenderable") would be us putting words in the
    judge's mouth.
    """
    from PIL import Image
    try:
        Image.new("RGB", (VIEW_W, VIEW_H), (245, 245, 245)).save(out_path, "JPEG", quality=JPEG_Q)
        return out_path if os.path.isfile(out_path) else None
    except Exception:
        return None


def _cap_height(raw, out_path):
    """Write the JPEG, cropping to MAX_PAGE_H if the page came out taller."""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(raw))
    if im.height > MAX_PAGE_H:
        im = im.crop((0, 0, im.width, MAX_PAGE_H))
    im.convert("RGB").save(out_path, "JPEG", quality=JPEG_Q)
    return out_path if os.path.isfile(out_path) else None


def _proxy_cfg(purl):
    from urllib.parse import urlparse
    pu = urlparse(purl)
    cfg = {"server": f"{pu.scheme}://{pu.hostname}:{pu.port}",
           "bypass": "127.0.0.1,localhost"}
    if pu.username:
        cfg["username"], cfg["password"] = pu.username, pu.password or ""
    return cfg


def _launch(p, purl):
    kw = {"headless": True, "args": list(LAUNCH_ARGS)}
    if purl:
        kw["proxy"] = _proxy_cfg(purl)
        kw["args"].append("--proxy-bypass-list=127.0.0.1;localhost;<local>")
    return p.chromium.launch(**kw)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def render_job(p, shared, dist_dir, hero_path):
    """One capture. `shared` is the worker's long-lived browser, launched for the
    FIRST proxy hop (direct when RENDER_PROXY is unset); later hops — the rare
    proxy-flake failover — get a throwaway browser for this job only, exactly
    as the subprocess-per-capture version did. Returns (sig, render_env)."""
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=dist_dir))
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.05), daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"

    errs, sig, settle = [], {}, {}
    fatal = []   # fatal external failures of the CURRENT load attempt
    render_env = {"ok": True, "reason": None, "settle": {},
                  "proxy_failed": False, "attempts": [], "fatal": []}
    ctx = pg = None
    temp_browsers = []

    def _on_request_failed(req):
        try:
            u = req.url or ""
            if not u.startswith("http") or u.startswith(url.rsplit("/", 1)[0]):
                return
            f = req.failure or ""
            if any(k in f for k in FATAL_NET):
                fatal.append(f + " " + u[:120])
        except Exception:
            pass

    def _open(purl):
        nonlocal ctx, pg
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
            ctx = pg = None
        if purl == CHAIN[0]:
            b = shared
        else:
            b = _launch(p, purl)
            temp_browsers.append(b)
        ctx = b.new_context(viewport={"width": VIEW_W, "height": VIEW_H})
        pg = ctx.new_page()
        pg.set_default_timeout(20000)
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
        pg.on("requestfailed", _on_request_failed)

    def _load():
        del fatal[:]
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
        except Exception as e:
            errs.append("goto: " + str(e))
        try:
            pg.wait_for_load_state("networkidle", timeout=NETIDLE_MS)
        except Exception:
            pass
        pg.wait_for_timeout(POST_LOAD_MS)

    try:
        for purl in CHAIN:
            label = (purl.split("@")[-1] or "direct") if purl else "direct"
            try:
                _open(purl)
                _load()
            except Exception as e:
                errs.append(f"launch via {label}: {e}")
                del fatal[:]
                fatal.append("LAUNCH_FAILED " + label)
            if fatal:
                render_env["attempts"].append({"via": label, "fatal": fatal[:3]})
                if pg is not None and not fatal[0].startswith("LAUNCH_FAILED"):
                    _load()           # one reload retry: the proxy flakes transiently
            if not fatal and pg is not None:
                if render_env["attempts"]:
                    render_env["recovered"] = "via " + label
                break
        render_env["proxy_failed"] = bool(fatal) or pg is None
        render_env["fatal"] = fatal[:5]

        if pg is None:
            render_env.update(ok=False, reason="browser_launch_failed")
            sig["console_errors"] = errs[:12]
            return sig, render_env

        try:
            pg.wait_for_function("() => 1", timeout=2500)
            alive = True
        except Exception as e:
            alive = False
            errs.append("page unresponsive: " + str(e)[:120])

        hero = None
        if alive:
            for label, fn in (("settle", lambda: pg.evaluate(SETTLE_JS, SETTLE_BUDGET_MS)),
                              ("freeze", lambda: pg.add_style_tag(content=FREEZE_CSS)),
                              ("probe", lambda: pg.evaluate(PROBE_JS)),
                              ("retop", lambda: pg.evaluate("window.scrollTo(0, 0)"))):
                try:
                    res = fn()
                    if label == "settle":
                        settle = res or {}
                    elif label == "probe":
                        sig = res or {}
                except Exception as e:
                    errs.append(f"{label}: {e}")
                if label in ("freeze", "retop"):
                    pg.wait_for_timeout(300)
            try:
                raw = pg.screenshot(full_page=True, type="jpeg", quality=JPEG_Q)
                hero = _cap_height(raw, hero_path)
            except Exception as e:
                errs.append("hero: " + str(e))

        render_env["settle"] = settle
        if hero is None:
            hero = _blank(hero_path)
            render_env.update(reason="unrenderable")
            sig["unrenderable"] = True
        elif render_env["proxy_failed"]:
            render_env.update(ok=False, reason="proxy_failed")
        else:
            sig["page_flat_ratio"] = round(flat_ratio(hero), 3)
        sig["console_errors"] = errs[:12]
        return sig, render_env
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
        for b in temp_browsers:
            try:
                b.close()
            except Exception:
                pass
        httpd.shutdown()
        httpd.server_close()


def main():
    global CHAIN
    from playwright.sync_api import sync_playwright
    CHAIN = PROXY_CHAIN or [""]
    if CHAIN != [""] and "" not in CHAIN:
        CHAIN.append("")          # an implicit direct attempt at the end
    with sync_playwright() as p:
        shared = _launch(p, CHAIN[0])
        print("READY", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            try:
                sig, env = render_job(p, shared, job["dist"], job["hero"])
            except Exception as e:  # noqa: BLE001 — report, let the pool decide
                sig, env = {"console_errors": [f"worker: {type(e).__name__}: {e}"[:200]]}, \
                           {"ok": False, "reason": f"render_failed: {type(e).__name__}"}
            print("SIGNALS:" + json.dumps(sig), flush=True)
            print("RENDER_ENV:" + json.dumps(env), flush=True)
            print("DONE", flush=True)
            if not shared.is_connected():
                sys.exit(3)
        shared.close()


if __name__ == "__main__":
    main()
'''


def _enable_subreaper() -> bool:
    """Adopt orphaned grandchildren so they can be reaped.

    chromium calls setsid() on launch, so it escapes the worker's process
    group and os.killpg() cannot reach it directly. What DOES reach it is the
    group kill's side effect: the worker dies first, the browser is
    reparented, and it exits shortly after. On this container PID 1 is `sleep
    infinity`, which never calls wait() — so every reparented browser became a
    permanent zombie. Measured: 52754 zombies created in one 58-minute run
    despite killpg working.

    As PR_SET_CHILD_SUBREAPER we inherit those orphans instead of PID 1, which
    lets _reap_zombies() actually collect them. With the worker pool this only
    matters on the kill path (timeouts, recycles, crashes) — a healthy worker
    never exits — but that path is exactly where the leak lived.
    """
    try:
        import ctypes

        PR_SET_CHILD_SUBREAPER = 36
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except Exception:  # noqa: BLE001 — non-Linux or a hardened libc; the service still runs
        return False


def _reap_zombies() -> int:
    """Collect children reparented to us by the subreaper. Returns how many."""
    n = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            break  # nothing left to wait for
        if pid == 0:
            break
        n += 1
    return n


_subreaper_on = False


def _ensure_subreaper() -> None:
    """Enable the subreaper once per process (idempotent, cheap)."""
    global _subreaper_on
    if not _subreaper_on:
        _enable_subreaper()
        _subreaper_on = True


def _kill_group(pid: int) -> None:
    """SIGKILL a whole process group; no-op if it is already empty.

    Workers run with start_new_session=True, so the pgid equals the worker's
    pid and the group still covers playwright's node driver after the worker
    itself is gone. chrome-headless setsid()s out of it — see _enable_subreaper
    for how that one is collected.
    """
    import signal

    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass  # group empty or not ours — nothing to clean


def _cpu_ticks() -> tuple[int, int] | None:
    """(busy, total) jiffies for the whole box, from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = v[3] + v[4]  # idle + iowait
    return sum(v) - idle, sum(v)


def cpu_busy_since(before: tuple[int, int] | None) -> float | None:
    """Fraction of all cores busy since `before` (a _cpu_ticks() reading)."""
    after = _cpu_ticks()
    if before is None or after is None or after[1] <= before[1]:
        return None
    return (after[0] - before[0]) / (after[1] - before[1])



_launch_lock = threading.Lock()
_last_launch = [0.0]


def _paced() -> None:
    """Space chromium launches LAUNCH_SPACING_S apart, process-wide."""
    with _launch_lock:
        wait = _last_launch[0] + LAUNCH_SPACING_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.time()


class _Worker:
    """One render worker process: python + node driver + one chromium."""

    def __init__(self, idx: int, script: str):
        self.idx = idx
        self.script = script
        self.proc: subprocess.Popen | None = None
        self.jobs = 0
        self.stderr_tail: collections.deque = collections.deque(maxlen=40)

    def start(self) -> None:
        _paced()
        self.jobs = 0
        self.stderr_tail.clear()
        self.proc = subprocess.Popen(
            [sys.executable, self.script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=_CAPTURE_ENV,
        )
        threading.Thread(target=self._drain, args=(self.proc,), daemon=True).start()
        line = self._readline(WORKER_READY_TIMEOUT_S)
        if line.strip() != "READY":
            tail = self._tail()
            self.kill()
            raise RuntimeError(f"render worker {self.idx} did not come up: {line.strip()[:80]!r} {tail}")

    def _drain(self, proc) -> None:
        try:
            for line in proc.stderr:
                self.stderr_tail.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _tail(self) -> str:
        return " | ".join(list(self.stderr_tail)[-6:])[-300:]

    def _readline(self, timeout_s: float) -> str:
        """Blocking readline with a kill-switch: on timeout the whole worker
        group is killed, which ends the read with EOF."""

        def _expire():
            self.timed_out = True
            self.kill()

        timer = threading.Timer(timeout_s, _expire)
        timer.start()
        try:
            return self.proc.stdout.readline()
        except (OSError, ValueError):
            return ""
        finally:
            timer.cancel()

    def render(self, dist_dir: str, hero_path: str, timeout_s: float) -> tuple[str, str, str]:
        """Run one job. Returns (status, stdout lines, stderr tail) with status
        "ok" | "timeout" | "died". Anything but "ok" means the worker is
        already killed and the caller must respawn it — EOF before DONE is
        never a healthy worker, whatever the exit code says."""
        self.timed_out = False
        try:
            self.proc.stdin.write(json.dumps({"dist": dist_dir, "hero": hero_path}) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self.kill()
            return "died", "", "worker stdin closed: " + self._tail()
        deadline = time.time() + timeout_s
        out = []
        while True:
            line = self._readline(max(0.1, deadline - time.time()))
            if not line:  # EOF: killed or crashed
                self.kill()
                if self.timed_out:
                    return "timeout", "".join(out), self._tail()
                return "died", "".join(out), f"rc={self.proc.poll()} {self._tail()}"
            if line.startswith("DONE"):
                break
            out.append(line)
        self.jobs += 1
        return "ok", "".join(out), ""

    def kill(self) -> None:
        """Idempotent. Pipes are left to the Popen object's own cleanup: closing
        them here would race a reader blocked in _readline on another thread."""
        if self.proc is None:
            return
        _kill_group(self.proc.pid)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(4):
            if _reap_zombies():
                break
            time.sleep(0.05)
        else:
            _reap_zombies()


class RenderPool:
    """Idle workers in a queue; the server's render semaphore keeps callers at
    or under `size`, so a caller normally never waits here — but the wait is
    bounded by its timeout regardless."""

    def __init__(self, size: int = POOL_SIZE):
        self.size = size
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._spawned = 0
        self._lock = threading.Lock()
        self._script: str | None = None
        self.respawns = 0
        self.recycles = 0

    def _script_path(self) -> str:
        if self._script is None:
            import tempfile

            fd, path = tempfile.mkstemp(suffix="_render_worker.py")
            with os.fdopen(fd, "w") as f:
                f.write(_WORKER_SCRIPT)
            self._script = path
        return self._script

    def _spawn(self) -> _Worker:
        _ensure_subreaper()  # must be on BEFORE we spawn, so orphans come to us
        self._ensure_reaper()
        w = _Worker(self._spawned, self._script_path())
        w.start()
        return w

    _reaper_on = False

    def _ensure_reaper(self) -> None:
        """Collect orphans on a clock, not only right after a kill.

        A killed worker's chromium notices its parent is gone on its own
        schedule; the 200ms reap window in _Worker.kill catches most of it, and
        the subprocess-per-capture design swept the rest on the next render.
        Kills are rare now, so the stragglers need their own sweep — three were
        left behind by two kills in a functional test. Safe with
        the workers' own waits because nothing here reads an exit code: EOF
        before DONE is the failure signal, whatever waitpid says.
        """
        if RenderPool._reaper_on:
            return
        RenderPool._reaper_on = True

        def _sweep():
            while True:
                time.sleep(5.0)
                _reap_zombies()

        threading.Thread(target=_sweep, daemon=True, name="render-reaper").start()

    def _take(self, timeout_s: float) -> _Worker | None:
        with self._lock:
            grow = self._spawned < self.size and self._idle.empty()
            if grow:
                self._spawned += 1
        if grow:
            try:
                return self._spawn()
            except Exception:
                with self._lock:
                    self._spawned -= 1
                raise
        try:
            return self._idle.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def prewarm(self) -> None:
        """Bring every worker up now, paced, so the first step burst does not
        pay 128 launches on the request path. Runs in the background."""

        def _run():
            while True:
                with self._lock:
                    if self._spawned >= self.size:
                        return
                    self._spawned += 1
                try:
                    self._idle.put(self._spawn())
                except Exception as e:  # noqa: BLE001 — one bad launch must not stop the warm-up
                    with self._lock:
                        self._spawned -= 1
                    print(f"[render] prewarm: worker failed to start ({e}); the pool will retry on demand", flush=True)
                    time.sleep(1.0)

        threading.Thread(target=_run, daemon=True, name="render-prewarm").start()

    def capture(self, dist_dir: str, hero_path: str, timeout_s: float) -> tuple[str, str, str]:
        """(status, stdout lines, stderr tail) — see _Worker.render."""
        try:
            w = self._take(timeout_s)
        except Exception as e:  # noqa: BLE001 — a worker that cannot start is an infra verdict
            return "died", "", f"worker failed to start: {e}"[:300]
        if w is None:
            return "timeout", "", "no render worker became free in time"
        status, out, err = w.render(dist_dir, hero_path, timeout_s)
        if status != "ok" or w.jobs >= MAX_JOBS_PER_BROWSER:
            if status == "ok":
                self.recycles += 1
                w.kill()
            else:
                self.respawns += 1
            try:
                w.start()
            except Exception as e:  # noqa: BLE001 — slot stays free; the next taker respawns
                with self._lock:
                    self._spawned -= 1
                print(f"[render] worker {w.idx} respawn failed: {e}", flush=True)
                return status, out, err
        self._idle.put(w)
        return status, out, err

    def snapshot(self) -> dict:
        return {
            "size": self.size,
            "spawned": self._spawned,
            "idle": self._idle.qsize(),
            "respawns": self.respawns,
            "recycles": self.recycles,
        }


POOL = RenderPool()


def capture_hero(
    dist_dir: str, hero_path: str, *, timeout_s: float = float(os.getenv("RENDER_TIMEOUT_S", "120"))
) -> tuple[bool, str, dict]:
    """Render `dist_dir`/index.html to `hero_path` on a pool worker.
    Returns (ok, reason, signals). ok=False means the page was never seen
    (infra) — the caller must drop."""
    cpu0 = _cpu_ticks()
    status, out, err = POOL.capture(dist_dir, hero_path, timeout_s)
    if status == "timeout":
        return False, "render_timeout" + (f": {err}" if "no render worker" in err else ""), {}
    return _verdict(0 if status == "ok" else 1, out, err, cpu_busy_since(cpu0))


def _verdict(returncode: int, out: str | None, err: str | None, busy: float | None) -> tuple[bool, str, dict]:
    """Turn a worker's reply into (ok, reason, signals).

    `returncode` is 0 for a worker that answered and non-zero for one that
    died mid-job (`err` then carries its exit code and stderr tail).

    `busy` is the box-wide CPU fraction across the capture. It only matters for
    one verdict: an `unrenderable` page taken on a saturated box is not
    believed (STARVED_BUSY_FRAC) — that is a render infra failure and drops,
    where the same verdict on an idle box is the page's fault and scores.
    """
    sig, env = {}, {}
    for line in (out or "").splitlines():
        if line.startswith("SIGNALS:"):
            try:
                sig = json.loads(line[len("SIGNALS:") :])
            except json.JSONDecodeError:
                pass
        elif line.startswith("RENDER_ENV:"):
            try:
                env = json.loads(line[len("RENDER_ENV:") :])
            except json.JSONDecodeError:
                pass
    if returncode != 0:
        tail = (err or "")[-300:]
        return False, f"render_crashed rc={returncode}: {tail}", sig
    if not env.get("ok"):
        return False, str(env.get("reason") or "render_failed"), sig
    if busy is not None:
        sig["cpu_busy"] = round(busy, 2)
    if sig.get("unrenderable") and busy is not None and busy > STARVED_BUSY_FRAC:
        return (
            False,
            (f"starved: box {busy:.0%} busy across the capture, cannot tell a hung page from a starved renderer"),
            sig,
        )
    return True, "ok", sig


def to_jpeg_b64(page_path: str, *, width: int | None = None, quality: int = 85) -> str:
    """Capture -> JPEG b64 in the shape the five judges were calibrated on:
    full width (they read layout off the real column widths), height already
    capped at MAX_PAGE_H by the capture, q85.

    `width` is a caller-side override for artifacts that want a thumbnail; the
    default must stay None — downscaling to a fixed width changes what the
    judges see relative to the demos they were labelled against.
    """
    import io

    from PIL import Image

    im = Image.open(page_path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def hero_b64(page_b64: str) -> str:
    """Full-page JPEG b64 -> first-screen JPEG b64.

    The first VIEW_H rows at capture width — the exact frame the viewport
    rendered. No judge on the group-reward path asks for it any more (the
    first-screen judges went with the pointwise rubric); kept because it is
    capture geometry this file owns and the next above-the-fold consumer
    should not re-derive it.

    Lives here rather than in a judge module because it is CAPTURE GEOMETRY:
    VIEW_W / VIEW_H / the JPEG quality all belong to this file, and a second
    copy of "what is a hero" is how the two drift. Re-encodes at the same
    quality `to_jpeg_b64` emits — re-encoding above the source cannot recover
    anything and only grows the payload.

    A page shorter than one screen passes through untouched rather than being
    upscaled: the crop is a ceiling, not a target size.
    """
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(page_b64))).convert("RGB")
    if im.height > VIEW_H:
        im = im.crop((0, 0, min(im.width, VIEW_W), VIEW_H))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)  # to_jpeg_b64's quality
    return base64.b64encode(buf.getvalue()).decode()
