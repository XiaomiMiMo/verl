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
"""HTTP client for the external grading service, as the TRAINING path uses it.

Everything that decides a score lives in the service: it rebuilds the delivered site out of
the workspace tarball we ship, localizes remote assets, renders it with its own browser, and
runs the vision judges. This module only moves bytes and maps transport failures to "drop".

**Not replaceable by the client inside the vendored service package.** That one
(``design_grader.client.grade_remote``) delivers either a chat ``response`` string or a
message list; it has no workspace-tarball surface, and none of ``/grade_group``, the startup
handshake or the runtime gate. The two are different endpoints of the same service, not two
spellings of one.

Two endpoints, used in sequence per GRPO group:

  ``POST /grade`` with ``judge_scope=query`` -> query_score plus the rendered shot, reward null
  ``POST /grade_group`` once per group, n shots -> per-row reward from a relative pick minus a
       query deduct

``/grade`` is called here for the SHOT and the query score, never for a reward of its own:
the service's pointwise reward is group-independent and this arm does not use it. Evaluation
does not go through this service at all -- see ``eval_mode.py``.

**``drop`` versus ``0.0`` is the one semantic the training side must preserve.** "No
deliverable, or an ugly page" is a real 0.0 the policy learns from; "we never saw the page"
(render infra, judge unreachable, service down) must be masked, not scored. Collapsing the
two teaches the policy that infrastructure failures are its fault.

stdlib only on purpose: this runs inside the environment Ray actors and inside the driver's
group-reward threads.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
import zlib

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

DEFAULT_TIMEOUT_S = 720.0
GROUP_TIMEOUT_S = 1200.0


def resolve_service_url(cfg: dict | None = None, *, env_var: str = "DESIGN_GRADER_URL") -> str:
    """``cfg["service_url"]`` (harness profile) or the environment variable.

    Raises when neither is set. A missing address is a launcher bug, not a page problem, and
    scoring nothing would be indistinguishable from a run of terrible pages.
    """
    url = (cfg or {}).get("service_url") or os.getenv(env_var)
    if not url:
        raise RuntimeError(
            f"grading service address missing: set {env_var} in the launcher, or "
            "traj_grader.service_url in the harness profile"
        )
    return url


def _post_json(url: str, body: dict, *, timeout_s: float, retries: int = 2, log=print) -> dict:
    """POST with backoff; a service that stays unreachable becomes a synthetic drop.

    Returning a drop rather than raising is what keeps one unreachable service from taking
    the rollout down: the caller masks the row instead of scoring it 0.
    """
    payload = json.dumps(body, ensure_ascii=False).encode()
    last = "?"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=timeout_s) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - any transport error: retry, then synthetic drop
            last = f"{type(e).__name__}: {e}"[:200]
            log(f"[grader] attempt {attempt + 1}/{retries} failed: {last}")
            time.sleep(min(30, 2 * 2**attempt))
    return {
        "status": "drop",
        "reward": None,
        "drop_reason": f"service unreachable: {last}",
        "grader_version": "unknown",
    }


def grade_remote(
    service_url: str,
    *,
    task_id: str,
    query: str,
    workspace_tgz_b64: str,
    messages: list[dict] | None = None,
    judge_scope: str | None = None,
    global_step=None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = 2,
    log=print,
) -> dict:
    """POST one rollout to the service; return its verdict dict.

    ``workspace_tgz_b64`` is the graded surface -- the agent's pod working directory.
    ``messages`` only feed the service's tool-usage signals, and must already have their
    media parts stripped (see :func:`slim_messages`). ``global_step`` labels the service's own
    dump and never affects the score.

    ``judge_scope="query"`` is the group-reward per-rollout call: the service runs the same
    extract-and-render pipeline but only the query judge, and answers with ``query_score``
    plus ``shot_jpg_b64`` and ``reward: null``. The aesthetic half is judged per GROUP later
    by :func:`grade_group_remote`.

    Response contract::

      {"status":"ok", "reward":0..1, "breakdown":{...}, "dims":{...},
       "shot_jpg_b64":str|null, "reasoning":str, "grader_version":str, ...}
      {"status":"ok", "reward":null, "query_score":0..1, "shot_jpg_b64":str, ...}
      {"status":"drop", "reward":null, "drop_reason":str, ...}
    """
    body: dict = {
        "task_id": task_id,
        "query": query,
        "budget_s": timeout_s,
        "workspace_tgz_b64": workspace_tgz_b64,
        "messages": messages or [],
    }
    if judge_scope is not None:
        body["judge_scope"] = judge_scope
    if global_step is not None:
        body["global_step"] = global_step
    return _post_json(service_url.rstrip("/") + "/grade", body, timeout_s=timeout_s, retries=retries, log=log)


def grade_group_remote(
    service_url: str,
    shots: list[str],
    query: str,
    query_scores: list | None = None,
    *,
    runtime_factors: list | None = None,
    task_id: str = "",
    global_step=None,
    timeout_s: float = GROUP_TIMEOUT_S,
    retries: int = 2,
    log=print,
) -> dict:
    """POST one complete GRPO group to ``/grade_group``; return the group verdict.

    ``shots`` are the judged screenshots the per-rollout calls landed on the shared dump
    directory -- the very ``shot_jpg_b64`` the service returned. ``query_scores`` aligns with
    them, and a None entry is re-judged by the service. ``runtime_factors`` aligns the same
    way: it is the runtime gate from that same per-rollout call ({0.0, 1.0}, None meaning no
    gate), which the service floors on and cannot re-derive here because ``/grade_group``
    renders nothing.

    The seed is derived from the shot PATHS so a re-run of the same group sees the same
    relabeling. ``items[*].shot`` comes back as an index and is rewritten to the path here.

    An unreadable shot or a transport failure becomes a synthetic drop with no ``items``,
    which the caller turns into INVALID for the whole group -- not a zero for the group.
    """
    shots_b64 = []
    for path in shots:
        try:
            with open(path, "rb") as f:
                shots_b64.append(base64.b64encode(f.read()).decode())
        except OSError as e:
            return {"status": "drop", "reward": None, "drop_reason": f"shot unreadable: {path}: {e}"[:300]}
    body = {
        "task_id": task_id,
        "query": query,
        "shots_jpg_b64": shots_b64,
        "query_scores": query_scores,
        "runtime_factors": runtime_factors,
        "budget_s": timeout_s,
        "seed": zlib.crc32("|".join(shots).encode()) % 100000,
    }
    if global_step is not None:
        body["global_step"] = global_step
    out = _post_json(service_url.rstrip("/") + "/grade_group", body, timeout_s=timeout_s, retries=retries, log=log)
    for item in out.get("items") or []:
        if isinstance(item.get("shot"), int) and 0 <= item["shot"] < len(shots):
            item["shot"] = shots[item["shot"]]
    return out


def fetch_version(service_url: str, *, timeout_s: float = 30.0) -> dict:
    """GET ``/version``. Raises on transport failure -- the launcher wants to fail fast."""
    req = urllib.request.Request(service_url.rstrip("/") + "/version")
    with _OPENER.open(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def check_capabilities(service_url: str, *, need_group: bool = True, timeout_s: float = 30.0) -> dict:
    """Startup handshake. Raises when the service cannot do this job.

    Pins CAPABILITIES, not the version string. Pinning ``grader_version`` to a single value
    and then upgrading the service made every training retry fatal in the handshake and
    burned the retry budget for nothing. What actually matters is:

    * ``/grade_group`` in ``endpoints``. Without it the group call 404s, the whole group goes
      INVALID, and the failure surfaces late and unrecognisably.
    * ``runtime_gate: true``. Without it gated rows silently keep their pick reward, and this
      training line loses its point.

    ``need_group=False`` only skips those two assertions -- it does NOT select a different
    service. Kept so a caller that merely wants reachability confirmed can say so.
    """
    try:
        info = fetch_version(service_url, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001 - unreachable at launch is fatal, by design
        raise RuntimeError(f"grading service handshake failed for {service_url}: {type(e).__name__}: {e}") from e
    if need_group:
        endpoints = info.get("endpoints") or []
        if "/grade_group" not in endpoints:
            raise RuntimeError(
                f"grading service at {service_url} has no /grade_group (endpoints={endpoints}); "
                "this is not a group-capable instance"
            )
        if not info.get("runtime_gate"):
            raise RuntimeError(
                f"grading service at {service_url} reports runtime_gate={info.get('runtime_gate')!r}; "
                "the group half requires the hard runtime gate"
            )
    return info


def slim_messages(messages: list[dict]) -> list[dict]:
    """The trajectory as the service wants it: text only.

    The service never reads image parts, and on the path this arm's bridge uses they are only
    ``webdev://image/K`` markers anyway -- but a caller that assembles messages from anywhere
    else can carry base64 image parts of megabytes each, which turned a 100 KB request into
    tens of MB. Media parts become a marker so the turn count and the fact that an image was
    viewed both survive. Builds new dicts; the caller's messages are not touched.
    """
    out = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if k != "responses_items"}
        content = m.get("content")
        if isinstance(content, list):
            m2["content"] = [
                part
                if not (isinstance(part, dict) and part.get("type") not in ("text", "input_text", "output_text"))
                else {"type": "text", "text": f"[{part.get('type', 'media')} omitted]"}
                for part in content
            ]
        out.append(m2)
    return out


def runtime_gate(verdict: dict) -> tuple[float | None, str]:
    """``(factor, why)`` from a verdict's ``dims.runtime`` block.

    The gate is {0.0, 1.0}: a page whose inline script fails a syntax check, throws during
    load, or hangs the renderer scores 0 outright. A hard floor rather than a soft multiplier
    because a soft one can be bought back by the aesthetic score -- RL then treats a runtime
    error as an amortisable cost instead of a precondition.

    ``None`` means the service predates the gate; the caller treats that as 1.0.
    """
    dims = verdict.get("dims")
    rt = dims.get("runtime") if isinstance(dims, dict) else verdict.get("runtime")
    if not isinstance(rt, dict) or rt.get("factor") is None:
        return None, ""
    return float(rt["factor"]), str(rt.get("why") or "")[:200]


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "GROUP_TIMEOUT_S",
    "check_capabilities",
    "fetch_version",
    "grade_group_remote",
    "grade_remote",
    "resolve_service_url",
    "runtime_gate",
    "slim_messages",
]
