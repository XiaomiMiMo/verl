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
"""Shared vision-LLM call for the graders (aes judge + layout gate).

Two process-wide controls sit in front of the router, because it fails in two
ways:
  * LIMITER — how many calls may be on the wire at once, adapted from the
    latency the router hands back. Past ~1000 concurrent calls the channel's
    throughput COLLAPSES rather than plateaus (table on AdaptiveLimit), so
    an unbounded pool turns a busy step into a wall of ReadTimeouts.
  * _PAUSE — one shared cooldown on 429/503: the channel is overloaded as a
    whole, so every thread stands back behind one timestamp (per-thread
    backoff from dozens of threads is how a channel gets pinned down —
    measured at 5000 concurrency).
Every wait here is bounded by the request's own deadline (set_deadline), so a
grade that can no longer answer in time fails fast instead of finishing into
a socket the client has already closed.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

MODEL = os.getenv("GRADER_MODEL", "")
BASE = os.getenv("LLM_JUDGE_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("LLM_JUDGE_API_KEY", "")

# Socket ceiling for judge calls. Not a pool size — one request, one socket —
# but the number JUDGE_CONCURRENCY has to stay under, or the queue forms inside
# this process instead of at the limiter where it is observable.
JUDGE_MAX_CONNECTIONS = 6144
JUDGE_TIMEOUT_S = 300.0
ATTEMPTS = 2

MIN_ATTEMPT_S = 20.0
QUEUE_GAVE_UP = "judge queue: budget ran out waiting for a router slot"


def _budget_gone(last: str, attempt: int) -> str:
    return (
        f"budget exhausted after attempt {attempt} ({last})" if attempt else "budget exhausted before the first attempt"
    )


def _is_timeout(e: BaseException) -> bool:
    """Did this attempt spend its whole wait?

    urllib raises socket.timeout / TimeoutError (or URLError wrapping one).
    Timeouts feed the limiter's shrink signal and skip the backoff sleep — a
    timeout has already waited JUDGE_TIMEOUT_S, another nap on top is pure loss.
    """
    if isinstance(e, TimeoutError):
        return True
    name = type(e).__name__
    if "Timeout" in name or "timeout" in name:
        return True
    reason = getattr(e, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(e).lower()


JUDGE_PAUSE_MAX_S = float(os.getenv("JUDGE_PAUSE_MAX_S", "30"))

_PAUSE = {"until": 0.0}
_pause_lock = threading.Lock()


_TL = threading.local()


def set_deadline(t: float | None) -> None:
    """Absolute unix time by which this thread's judge call must have answered."""
    _TL.deadline = t


def deadline() -> float | None:
    return getattr(_TL, "deadline", None)


def _remaining(floor: float = 0.0) -> float:
    """Seconds left on this thread's deadline; +inf when there is none."""
    d = deadline()
    return float("inf") if d is None else max(floor, d - time.time())


class AdaptiveLimit:
    """How many judge calls may be on the wire at once — tuned from the router's
    own behaviour, not fixed.

    Measured over 262k judge calls, bucketed by
    how many of OUR calls were already in flight when each one started:

        in-flight    p50      p90     timed out (>=300s)
          <500      16.8s    60.1s      0.6%
        500-999     27.3s   144.8s      2.5%
       1000-1499    68.6s   301.6s     10.1%
       2000-2499   176.0s   556.8s     34.8%
       3500-3999   421.8s   611.6s     54.3%

    Throughput (in-flight / latency) falls from ~30 calls/s under 1000 to ~7
    calls/s above 3500. That is congestion collapse, not a slow channel: a call
    we abandon at JUDGE_TIMEOUT_S is work the router still finishes, and the
    retry behind it is a second copy. The old fixed 6144-slot pool was sized so
    that "the router, not us, is the ceiling"; what it produced was the table
    above — 98% of that run's 11.4k drops were ReadTimeouts from this regime.

    Control loop (AIMD, one decision per WINDOW_S):
      * a timeout rate over SHRINK_TIMEOUT_FRAC, or p90 over SHRINK_P90_S,
        multiplies the limit by SHRINK;
      * p90 under GROW_P90_S while the limit was actually being used adds
        GROW_STEP;
      * anything in between holds.

    Two things about the samples, both learned the hard way:
      * A completed call reports the latency of a call that STARTED up to that
        latency ago. After any change, calls started under the old limit are
        ignored (`_epoch`) — else one congested minute yields fifteen "slow"
        windows in a row and the limit falls to the floor while the router has
        long recovered.
      * Completed calls alone are a biased sample: right after a change the
        only calls that have finished are the FAST ones, so a window read p90
        14-21s while the router's true p90 was 83s and the limit grew straight
        past the knee (measured in burst soak). So a call still in flight
        counts as "at least this long" (censored), and GROWTH is judged only on
        calls started at least GROW_P90_S ago — old enough that "under 90s" is
        known either way. Shrinking uses everything: a censored call past
        SHRINK_P90_S is already slow, whatever it ends up at.

    Waiters queue on a condition variable and give up at their own deadline
    (see acquire), so a request that cannot possibly answer in time is dropped
    before it costs the router anything.
    """

    INITIAL, FLOOR, CEILING = 512, 256, 768
    WINDOW_S, MIN_SAMPLES = 20.0, 8
    SHRINK, SHRINK_TIMEOUT_FRAC, SHRINK_P90_S = 0.7, 0.02, 180.0
    GROW_STEP, GROW_P90_S = 128, 90.0

    def __init__(self, initial: int | None = None, *, clock=time.time):
        self._clock = clock
        self.limit = float(initial or self.INITIAL)
        self.inflight = 0
        self.waiting = 0
        self._cv = threading.Condition()
        self._epoch = clock()  # calls started before this are measured under an old limit
        self._window_at = clock()
        self._running: list[float] = []  # start times of calls on the wire
        self._done: list[tuple[float, float, bool]] = []  # (start, latency, timed_out)
        self._peak_inflight = 0
        self.decisions = 0
        self.gave_up = 0  # waiters that hit their deadline in the queue
        self.last: dict = {}

    def acquire(self, deadline: float | None = None) -> float | None:
        """Block until a slot is free or `deadline` passes. Returns the start
        time to hand back to release(), or None if the deadline won."""
        with self._cv:
            self.waiting += 1
            try:
                while self.inflight >= self.limit:
                    now = self._clock()
                    if deadline is not None and now >= deadline:
                        self.gave_up += 1
                        return None
                    self._cv.wait(1.0 if deadline is None else min(1.0, deadline - now))
                self.inflight += 1
                self._peak_inflight = max(self._peak_inflight, self.inflight)
                started = self._clock()
                self._running.append(started)
                return started
            finally:
                self.waiting -= 1

    def release(self, started: float, *, timed_out: bool = False) -> None:
        now = self._clock()
        with self._cv:
            self.inflight -= 1
            try:
                self._running.remove(started)
            except ValueError:
                pass
            if started >= self._epoch:
                self._done.append((started, now - started, timed_out))
            if now - self._window_at >= self.WINDOW_S:
                self._decide(now)
            self._cv.notify()

    @staticmethod
    def _p(sorted_vals: list[float], q: float) -> float:
        return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]

    def _decide(self, now: float) -> None:
        self._window_at = now
        done = [d for d in self._done if d[0] >= self._epoch]
        if len(done) < self.MIN_SAMPLES:
            return
        running = [now - s for s in self._running if s >= self._epoch]
        everything = sorted([lat for _, lat, _ in done] + running)
        p90_all = self._p(everything, 0.9)
        timeouts = sum(1 for _, _, t in done if t) / len(done)
        cutoff = now - self.GROW_P90_S
        mature = sorted(
            [lat for s, lat, _ in done if s <= cutoff] + [now - s for s in self._running if self._epoch <= s <= cutoff]
        )
        before = self.limit
        if timeouts > self.SHRINK_TIMEOUT_FRAC or p90_all > self.SHRINK_P90_S:
            self.limit = max(self.FLOOR, self.limit * self.SHRINK)
        elif (
            len(mature) >= self.MIN_SAMPLES
            and self._p(mature, 0.9) < self.GROW_P90_S
            and self._peak_inflight >= 0.5 * self.limit
        ):
            self.limit = min(self.CEILING, self.limit + self.GROW_STEP)
        self.last = {
            "p50": self._p(everything, 0.5),
            "p90": p90_all,
            "timeouts": timeouts,
            "n": len(done),
            "running": len(running),
            "mature": len(mature),
            "peak_inflight": self._peak_inflight,
        }
        if self.limit != before:
            self._epoch = now  # the next window must be measured under the new limit
            self.decisions += 1
            print(
                f"[judge-limit] {before:.0f} -> {self.limit:.0f} (done={len(done)} "
                f"running={len(running)} p50={self.last['p50']:.0f}s p90={p90_all:.0f}s "
                f"timeouts={timeouts:.1%} peak_inflight={self._peak_inflight})",
                flush=True,
            )
        self._peak_inflight = self.inflight
        horizon = now - 2 * JUDGE_TIMEOUT_S
        self._done = [d for d in self._done if d[0] >= max(self._epoch, horizon)]

    def snapshot(self) -> dict:
        """For /healthz and the dashboard: the one number that used to be
        invisible (how many calls the router is actually being asked to hold)
        next to how many are queued behind it."""
        with self._cv:
            return {
                "limit": int(self.limit),
                "inflight": self.inflight,
                "waiting": self.waiting,
                "gave_up": self.gave_up,
                "decisions": self.decisions,
                "window": dict(self.last),
            }


LIMITER = AdaptiveLimit()


def _openai_parts(parts: list) -> list:
    """Normalise mixed text/image parts into strict OpenAI content blocks.

    Bare strings in a multimodal content array are NOT valid Chat Completions:
    the router answers 502 "Upstream request failed" for
    ``["text", {"type":"image_url", ...}]`` while accepting the same message as
    ``[{"type":"text",...}, {"type":"image_url",...}]``. This path was passing
    parts through raw, which is why every demo-bearing judge failed while the
    demo-free ones worked.

    It also strips the `resolution` key, which has no OpenAI equivalent.
    """
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
            continue
        if isinstance(p, dict) and "text" in p and "type" not in p:
            out.append({"type": "text", "text": p["text"]})
            continue
        if isinstance(p, dict) and "resolution" in p:
            p = {k: v for k, v in p.items() if k != "resolution"}
        out.append(p)
    return out


def _unparseable(text: str) -> str:
    """Drop-reason for a verdict the judge's parse rejected, quoting the reply.

    "unparseable verdict" alone says a judge failed but not WHY, and parse
    returns None for several distinct reasons: no JSON object at all, a score
    outside [0,1], a pick verdict with no usable index list, and — the one that
    was actually biting — a reply cut off mid-object by the model's output-token
    cap. They need different fixes, so the reply has to ride along.

    HEAD *and* TAIL, because the interesting failure lives at the end: a
    truncated JSON is perfectly valid for the first N issues and only shows its
    hand when the braces stop matching. The first version of this quoted only
    the head and every snippet looked like a healthy verdict. Total length goes
    in the middle — it separates "the judge wrote too much" (~16k chars of
    Chinese at a 4096-token cap) from "the judge wrote something off-contract"
    (a few hundred chars) at a glance.

    Whitespace is collapsed and both ends capped: this lands in `drop_reason`,
    which the log prints on one line and HISTORY keeps.
    """
    flat = " ".join((text or "").split())
    if not flat:
        return "unparseable verdict: <empty reply>"
    if len(flat) <= 320:
        return "unparseable verdict: " + flat
    return f"unparseable verdict: [{len(flat)} chars] {flat[:160]} …… {flat[-140:]}"


def chat_vision(
    parts: list, parse, *, retries: int | None = None, temperature: float | None = None, model: str | None = None
) -> tuple[dict | None, str]:
    """One user message of content `parts` -> `parse(text)` (dict, or None to
    retry as unparseable). Returns (verdict, "") or (None, last_error).

    `model` overrides the judge's configured model for this call. Every judge
    rides the same OpenAI-format transport, so the override only ever names
    another model on that wire format.
    """
    if not API_KEY:
        return None, "LLM_JUDGE_API_KEY unset"
    retries = ATTEMPTS if retries is None else retries
    body = {"model": model or MODEL, "messages": [{"role": "user", "content": _openai_parts(parts)}]}
    if temperature is not None:
        body["temperature"] = temperature
    if MODEL.startswith("claude"):
        body["max_tokens"] = 800  # claude path: max_completion_tokens -> empty response
    else:
        body["max_completion_tokens"] = 4000
    payload = json.dumps(body).encode()
    last = "?"
    for attempt in range(retries):
        if _remaining() < MIN_ATTEMPT_S:
            return None, _budget_gone(last, attempt)
        hold = _PAUSE["until"] - time.time()
        if hold > 0:
            time.sleep(min(hold, JUDGE_PAUSE_MAX_S, _remaining()) + random.random() * 2)
        started = LIMITER.acquire(deadline())
        if started is None:
            return None, QUEUE_GAVE_UP
        timed_out, nap = False, 0.0
        try:
            req = urllib.request.Request(
                BASE + "/v1/chat/completions",
                data=payload,
                headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=min(JUDGE_TIMEOUT_S, _remaining(1.0))) as r:
                d = json.loads(r.read())
            verdict = parse(d["choices"][0]["message"]["content"] or "")
            if verdict is None:
                last = _unparseable(d["choices"][0]["message"]["content"] or "")
                continue
            return verdict, ""
        except urllib.error.HTTPError as e:
            last = f"HTTPError: {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                pause = 5 if attempt == 0 else 10 + 20 * random.random() * attempt
                with _pause_lock:
                    _PAUSE["until"] = max(_PAUSE["until"], time.time() + min(pause, JUDGE_PAUSE_MAX_S))
            else:
                nap = min(60, 2 * 2**attempt)
        except Exception as e:  # noqa: BLE001 — transport errors: retry with backoff
            last = f"{type(e).__name__}: {e}"[:150]
            timed_out = _is_timeout(e)
            if not timed_out:  # a timeout already spent its wait
                nap = min(60, 2 * 2**attempt)
        finally:
            LIMITER.release(started, timed_out=timed_out)
        if nap:
            time.sleep(min(nap, _remaining()))
    return None, last


def b64_jpeg(path: str, width: int) -> str:
    """File -> resized JPEG q86 b64 — the exact shape the judges were calibrated on."""
    import base64
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode()


def json_object(text: str) -> dict | None:
    """First balanced JSON object in `text`, or None.

    Brace-MATCHING, not a regex. Every judge parse used to do this with
    ``\\{[^{}]*"tier"[^{}]*\\}`` — no nesting allowed — so a judge writing
    ``{"mult": 0.6, "reason": "CSS {color-scheme:dark}"}`` produced no match and
    the whole verdict was thrown away. Lives here rather than in one judge
    module because every judge that speaks JSON needs it and vision.py is
    already the module that owns talking to the models.
    """
    s = text or ""
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return out if isinstance(out, dict) else None
    return None


# Which model each judge rides. No defaults: a judge silently answering on a
# different model than the numbers it is compared against is worse than one that
# refuses to start, so every entry has to be named -- by GRADER_MODEL for all of
# them at once, or per judge with --judge-model DIM=MODEL.
#
# Two judges, because the group reward has exactly two LLM inputs: the
# per-rollout `query` coverage score and the in-group relative `pick`. The
# third input, the runtime gate, calls no model at all. Retargeting `pick`
# shifts every group reward -- see group_pick.py.
JUDGE_MODELS = {
    "query": MODEL,
    "pick": MODEL,
}


def call_for(dim: str, parts: list, parse, *, temperature: float | None = None):
    """One judge call on that judge's configured model.

    `dim` is the JUDGE_MODELS key: "query" or "pick".
    """
    model = JUDGE_MODELS[dim]
    if not model:
        return None, f"no model configured for the {dim} judge (set GRADER_MODEL or --judge-model {dim}=MODEL)"
    return chat_vision(parts, parse, temperature=temperature, model=model)


def image_part(jpg_b64: str, resolution: str | None = None) -> dict:
    """One image for a judge message.

    `resolution` has no OpenAI equivalent and is carried as a sibling key for
    backends that honour it; _openai_parts strips it at the wire boundary. The
    layout judge wants its target at the highest resolution available: its first
    checklist item is element truncation, and the truncation boundary is
    invisible at low resolution on the dark backgrounds generated sites favour.
    Dropping this field cost exactly that.
    """
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpg_b64}}
    if resolution:
        part["resolution"] = resolution
    return part
