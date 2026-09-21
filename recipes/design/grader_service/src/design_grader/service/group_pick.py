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
"""Group-relative visual-quality pick — the service half of webdev group_v1.

Port of the group reward selection logic onto this
service's vision channel, so an RL run that
grades through DESIGN_GRADER_URL can use the same group reward as the local
in-driver grader:

    raw    = pick_norm − query_deduct                      ∈ [−1.8, 1]
    reward = (raw + 2) / 3                                 ∈ [0.067, 1]
      pick_norm    Σ votes over the ok rounds / n_ok ∈ [−1, 1]; per round a shot
                   picked good is +1, bad −1, else 0. One judge model (the
                   ``pick`` slot of vision.JUDGE_MODELS — retargeting it shifts
                   every group reward), PICK_ROUNDS=8 rounds over a Williams row-complete
                   latin square (every shot in every position exactly once,
                   every ordered adjacency exactly once — measured to flatten
                   the first-position bias from ±0.28 to ±0.03 votes).
      query_deduct query_score ≥0.9 → 0 / ≥0.6 → 0.2 / ≥0.4 → 0.4 / ≥0.2 → 0.8;
                   below 0.2 the reward is QUERY_FLOOR (0.0) outright.
      runtime      the runtime gate (service/runtime_gate.py) from the pod's own
                   judge_scope=query render: inline <script> that fails
                   `node --check`, an uncaught pageerror during load, or a
                   renderer that stopped answering → factor 0 → reward is
                   RUNTIME_FLOOR (0.0) outright, same rung as the query floor.
                   Absent (older pod) = 1.0. Why a floor and not a deduct:
                   pick_norm is a bounded rank, so without it a page whose JS
                   does not run shares the bottom (−1) with a merely plain page
                   — in practice, gated agent pages averaged −0.11 vs
                   +0.02 clean on the raw scale, i.e. the judges could not see
                   the breakage at all. See prompts/group_pick.md for what the
                   pick judge does look at.

    Why the affine remap: the per-rollout absolute grade (the pointwise rubric
    this service no longer serves) and every
    other blackbox task hand the trainer a reward in [0, 1]; group_v1 used to be
    the lone [−2, 1] producer, which forced a clamp branch in the trainer's
    passrate (pr pinned to 0.5, histograms degenerate) and made 0.0 ambiguous
    ("mediocre" vs "no delivery / grade failed"). Mapping onto [0, 1] keeps the
    in-group ordering and spacing (GRPO only sees reward − group mean, so the
    shift cancels; the ÷3 scales every advantage equally) and puts the floor at
    0.0 where the trainer's `None → 0.0` fallback already lives. Absolute-mode
    (`grade`) is untouched — it was already [0, 1].

    Denoising (all ported unchanged):
      * ok rounds < PICK_MIN_OK      → whole group's pick zeroed (group_ok=False),
                                       raw = −query_deduct; rows are kept.
      * ≥ PICK_NCD_ZERO_MIN rounds judged "no clear difference" → group pick zeroed.
      * |net votes| ≤ PICK_NET_DEADZONE for one shot → that shot's pick zeroed.

The caller (POST /grade_group in server.py) sends the n sibling screenshots as
JPEG b64 — the very shot_jpg_b64 this service returned when it graded each
rollout with judge_scope="query" — plus the pod-judged query scores; only the
None entries are re-judged here (judges.check_query, same prompt the per-rollout
query judge uses, so the two sources cannot drift).

The result dict is shape-compatible with group_v1.grade_group: the training
side's WebdevGroupRewarder and the dashboard's group_reward.json parser consume
it unchanged. `items[*].shot` is the shot's INDEX here (the service never sees
paths); the client re-attaches its paths by position.
"""

from __future__ import annotations

import base64
import io
import os
import random
import zlib
from pathlib import Path

from . import judges, vision
from .vision import call_for, image_part, json_object

PROMPTS = Path(__file__).resolve().parent / "prompts"

PICK_ROUNDS = 8  # = group size 8 in williams mode: each shot sees each position once
PICK_MIN_OK = 5  # ok rounds < this → whole group's pick 0 (group_ok False)
PICK_NCD_ZERO_MIN = int(
    os.getenv("WEBDEV_PICK_NCD_ZERO_MIN", "2")
)  # ≥ this many "no clear difference" rounds → group pick 0; 0 disables
PICK_NET_DEADZONE = int(os.getenv("WEBDEV_PICK_NET_DEADZONE", "1"))  # |net votes| ≤ this → that shot's pick 0
PICK_ORDER = os.getenv("WEBDEV_PICK_ORDER", "williams")  # williams | random
PICK_IMG_W, PICK_IMG_MAXH = 720, 12000  # per-shot: full page scaled to 720w (both judges read the footer at 720)

QUERY_FLOOR = 0.0  # reward when query_score < 0.2 (already on the [0, 1] scale)
QUERY_DEDUCT = [(0.9, 0.0), (0.6, 0.2), (0.4, 0.4), (0.2, 0.8)]  # (lower bound, deduct), matched high→low
RAW_LO, RAW_HI = -2.0, 1.0
RUNTIME_FLOOR = 0.0


def williams_orders(n: int, seed: int = 0) -> list[list[int]]:
    """Williams design for even n: n row orders, every shot in every position
    exactly once, every ordered adjacency (a→b) exactly once. `seed` relabels
    which shot is number 0 per group, so the template never binds to rollout order."""
    base = [0]
    lo, hi = 1, n - 1
    while len(base) < n:
        base.append(hi)
        hi -= 1
        if len(base) < n:
            base.append(lo)
            lo += 1
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    return [[perm[(x + r) % n] for x in base] for r in range(n)]


def pick_orders(n: int, rounds: int, seed: int = 0) -> list[list[int]]:
    """Round orders per PICK_ORDER: Williams when rounds == n and n is even, else random."""
    if PICK_ORDER == "williams" and rounds == n and n % 2 == 0:
        return williams_orders(n, seed)
    rnd = random.Random(seed)
    return [rnd.sample(range(n), n) for _ in range(rounds)]


def query_deduct(s: float | None) -> float | None:
    """Deduction for one query score; None = missing; inf = QUERY_FLOOR triggered."""
    if s is None:
        return None
    for lo, d in QUERY_DEDUCT:
        if s >= lo:
            return d
    return float("inf")


def combine(votes: list[list[str]], n: int) -> dict:
    """votes: one length-n list per OK round, elements 'G'/'B'/'-'/'N' (N = that
    round judged no-clear-difference); failed rounds are not in the list.

    ok rounds < PICK_MIN_OK → ok=False, whole group's pick_norm zeroed (rows kept,
    query deduct still applies). Otherwise every ok round counts, normalised by
    the ok-round count. Denoising: ≥ PICK_NCD_ZERO_MIN N-rounds → group pick 0;
    |net votes| ≤ PICK_NET_DEADZONE for a shot → that shot 0 (by raw votes, so
    the rule reads the same when rounds differ)."""
    R = len(votes)
    ok = R >= PICK_MIN_OK
    if not R:
        return {
            "ok": ok,
            "pick_sum": [0] * n,
            "pick_norm": [0.0] * n,
            "rounds": 0,
            "ncd_zeroed": False,
            "n_ncd": 0,
            "noise_zeroed": [],
        }
    n_ncd = sum(1 for v in votes if v[0] == "N")
    ncd = PICK_NCD_ZERO_MIN > 0 and n_ncd >= PICK_NCD_ZERO_MIN
    pick_sum = [sum(1 if v[i] == "G" else -1 if v[i] == "B" else 0 for v in votes) for i in range(n)]
    noise = [i for i in range(n) if pick_sum[i] != 0 and abs(pick_sum[i]) <= PICK_NET_DEADZONE]
    eff = [0 if (ncd or i in noise) else pick_sum[i] for i in range(n)]
    pick_norm = [s / R for s in eff] if ok else [0.0] * n
    return {
        "ok": ok,
        "pick_sum": pick_sum,
        "pick_norm": pick_norm,
        "rounds": R,
        "ncd_zeroed": ncd,
        "n_ncd": n_ncd,
        "noise_zeroed": noise,
    }


def reward_one(
    pick_norm: float | None, query_score: float | None, runtime_factor: float | None = None
) -> tuple[float | None, list[str]]:
    """One row's reward on [0, 1]. `runtime_factor` is the gate's {0.0, 1.0}; None means
    "no verdict travelled" (older pod / driver) and is treated as 1.0 — the gate is an
    add-on, never a reason to invalidate a row."""
    missing = []
    if pick_norm is None:
        missing.append("pick")
    dq = query_deduct(query_score)
    if dq is None:
        missing.append("query")
    if missing:
        return None, missing
    if runtime_factor is not None and runtime_factor == 0.0:
        return RUNTIME_FLOOR, []
    if dq == float("inf"):
        return QUERY_FLOOR, []
    return round((pick_norm - dq - RAW_LO) / (RAW_HI - RAW_LO), 4), []


def parse_pick(text: str) -> dict | None:
    """A pick verdict: good/bad lists of display indices, or an explicit
    no_clear_difference. Out-of-range / non-int entries are FILTERED rather than
    retried (the round is still information), matching group_v1; only a verdict
    with no parseable JSON object at all is a failed round."""
    d = json_object(text)
    if d is None:
        return None

    def _ints(lst):
        out = []
        for k in lst if isinstance(lst, list) else []:
            try:
                out.append(int(k))
            except (TypeError, ValueError):
                continue
        return out

    return {
        "good": _ints(d.get("good")),
        "bad": _ints(d.get("bad")),
        "no_clear_difference": bool(d.get("no_clear_difference")),
        "reason": str(d.get("reason", ""))[:400],
    }


def pick_once(shots_720: list[str], query: str, order: list[int]) -> dict:
    """One pick round: the whole group in one call, shown in `order` (display
    position → original index). Returns good/bad as ORIGINAL indices, plus
    no_clear_difference / reason / order; {"error": True} on a failed call."""
    n = len(shots_720)
    assert sorted(order) == list(range(n))
    rubric = (
        (PROMPTS / "group_pick.md")
        .read_text()
        .replace("{n}", str(n))
        .replace("{n_max}", str(n - 1))
        .replace("{query}", (query or "").strip())
    )
    parts: list = []
    for k, idx in enumerate(order):
        parts.append(f"【编号 {k}】")
        parts.append(image_part(shots_720[idx], "high"))
    parts.append(rubric)
    got, err = call_for("pick", parts, parse_pick, temperature=1.0)
    if got is None:
        return {
            "good": [],
            "bad": [],
            "no_clear_difference": None,
            "reason": f"pick failed: {err}"[:200],
            "order": order,
            "error": True,
        }
    good = sorted({order[k] for k in got["good"] if 0 <= k < n})
    bad = [i for i in sorted({order[k] for k in got["bad"] if 0 <= k < n}) if i not in good]
    ncd = got["no_clear_difference"] and not (good or bad)
    return {"good": good, "bad": bad, "no_clear_difference": ncd, "reason": got["reason"], "order": order}


def _shrink(jpg_b64: str) -> str:
    """Full page → 720w JPEG for the pick call (the judge reads the footer fine
    at 720 and eight full-width pages would blow the request budget)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(io.BytesIO(base64.b64decode(jpg_b64))).convert("RGB")
    w, h = im.size
    if h > PICK_IMG_MAXH:
        im = im.crop((0, 0, w, PICK_IMG_MAXH))
    if im.width > PICK_IMG_W:
        im = im.resize((PICK_IMG_W, int(im.height * PICK_IMG_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def grade_group(
    shots_b64: list[str],
    query: str,
    query_scores: list | None = None,
    *,
    runtime_factors: list | None = None,
    pool,
    deadline: float | None = None,
    seed: int | None = None,
) -> dict:
    """The whole group: PICK_ROUNDS pick calls + query re-judges for the None
    entries, all fanned out on `pool` (the server's judge pool — a thread per
    call, so the wall clock is max(call), not the sum). `deadline` bounds every
    LLM attempt via vision's thread-local budget, exactly like the per-rollout path.

    `query_scores` aligns with `shots_b64`: the pod-side judgments; None entries
    are re-judged here on the full-resolution shot. `runtime_factors` aligns the
    same way: the pod-side runtime gate ({0.0, 1.0}); None / absent = no verdict,
    treated as 1.0 (the gate cannot be re-derived here — no render happens)."""
    n = len(shots_b64)
    pre = list(query_scores) if query_scores is not None else [None] * n
    assert len(pre) == n, f"query_scores must align with shots: {len(pre)} vs {n}"
    rt = list(runtime_factors) if runtime_factors is not None else [None] * n
    assert len(rt) == n, f"runtime_factors must align with shots: {len(rt)} vs {n}"
    need_q = [i for i in range(n) if pre[i] is None]
    if seed is None:
        seed = zlib.crc32("|".join(s[:64] for s in shots_b64).encode()) % 100000
    orders = pick_orders(n, PICK_ROUNDS, seed)
    small = [_shrink(s) for s in shots_b64]
    backend = vision.JUDGE_MODELS["pick"]

    def _with_deadline(fn, *args):
        vision.set_deadline(deadline)
        try:
            return fn(*args)
        finally:
            vision.set_deadline(None)  # pool threads are reused

    def _query(i):
        try:
            got, err = judges.check_query(shots_b64[i], query)
        except Exception as e:  # noqa: BLE001 — one judge crash must not kill the group
            got, err = None, f"{type(e).__name__}: {e}"[:200]
        if got is None:
            return {"query_score": None, "reason": str(err)[:200], "error": True}
        return got

    def _pick(r):
        try:
            return pick_once(small, query, orders[r])
        except Exception as e:  # noqa: BLE001
            return {
                "good": [],
                "bad": [],
                "no_clear_difference": None,
                "reason": f"{type(e).__name__}: {e}"[:200],
                "order": orders[r],
                "error": True,
            }

    pick_futs = [pool.submit(_with_deadline, _pick, r) for r in range(PICK_ROUNDS)]
    query_futs = {i: pool.submit(_with_deadline, _query, i) for i in need_q}
    picks = [f.result() for f in pick_futs]
    queries = {i: f.result() for i, f in query_futs.items()}
    for i in range(n):
        if pre[i] is not None:
            queries[i] = {"query_score": float(pre[i]), "reason": "pod-side", "source": "pod"}

    votes = []
    for r in picks:
        if r.get("error"):
            continue
        v = ["-"] * n
        for i in r["good"]:
            v[i] = "G"
        for i in r["bad"]:
            v[i] = "B"
        if r["no_clear_difference"]:
            v = ["N"] * n
        votes.append(v)
    agg = combine(votes, n)

    out = []
    for i in range(n):
        q = queries.get(i, {})
        qs = q.get("query_score")
        rw, missing = reward_one(agg["pick_norm"][i], qs, rt[i])
        out.append(
            {
                "shot": i,
                "reward": rw,
                "pick_sum": agg["pick_sum"][i],
                "pick_norm": agg["pick_norm"][i],
                "pick_votes": {backend: [v[i] for v in votes]},
                "query_score": qs,
                "query_deduct": (
                    None if qs is None else ("floor" if query_deduct(qs) == float("inf") else query_deduct(qs))
                ),
                "runtime_factor": rt[i],
                "runtime_gated": bool(rt[i] is not None and rt[i] == 0.0),
                "query_reason": q.get("reason", ""),
                "query_source": q.get("source", "driver"),
                "missing": missing,
            }
        )
    return {
        "query": query,
        "group_ok": agg["ok"],
        "ncd_zeroed": agg["ncd_zeroed"],
        "n_ncd": agg["n_ncd"],
        "noise_zeroed": agg["noise_zeroed"],
        "rounds_ok": agg["rounds"],
        "rounds_by_backend": {backend: agg["rounds"]},
        "rounds_total": len(picks),
        "n_query_pod": n - len(need_q),
        "n_query_driver": len(need_q),
        "n_runtime_gated": sum(1 for x in rt if x is not None and x == 0.0),
        "picks": [{"backend": backend, "round": r, **p} for r, p in enumerate(picks)],
        "items": out,
    }
