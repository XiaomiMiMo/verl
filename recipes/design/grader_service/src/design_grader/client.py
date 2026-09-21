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
"""Zero-dependency HTTP client for the design grader service.

stdlib-only on purpose: this module is imported inside the reference RL framework env actors, and a
grader must never pull framework dependencies into the RL side. The service owns
every scoring decision; the client only moves bytes and maps transport failures
to a drop verdict.
"""

from __future__ import annotations

import json
import time
import urllib.request

DEFAULT_TIMEOUT_S = 720.0

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def grade_remote(
    service_url: str,
    *,
    task_id: str,
    query: str,
    messages: list[dict] | None = None,
    response: str | None = None,
    global_step=None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = 2,
    log=print,
) -> dict:
    """POST one rollout to the service; return the service verdict dict.

    Delivery surface is either `messages` (agent trajectory, tool-call replay)
    or `response` (single-turn chat text with an html document in it).

    `global_step` is the training step, for the service's per-grade dump. It
    does not affect the score — it only labels the durable record so a
    sub-score trend can be grouped by step without joining on timestamps.
    None on a hand-rolled call; the dump writes null rather than guessing.

    Response contract (see design_grader.service.server):
      {"status": "ok",   "reward": 0..1, "score": 1-4 | null, "reason": str,
       "signals": {...}, "shot_jpg_b64": str | null, "grader_version": str}
      {"status": "drop", "reward": null, "drop_reason": str, "grader_version": str}

    Transport failures retry with backoff; a service that stays unreachable
    yields a synthetic drop (the trajectory is masked, never scored blind).
    """
    body = {"task_id": task_id, "query": query, "budget_s": timeout_s}
    if global_step is not None:
        body["global_step"] = global_step
    if response is not None:
        body["response"] = response
    else:
        body["messages"] = messages or []
    payload = json.dumps(body, ensure_ascii=False).encode()
    url = service_url.rstrip("/") + "/grade"
    last = "?"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=timeout_s) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — any transport error: retry, then drop
            last = f"{type(e).__name__}: {e}"[:200]
            log(f"[design_grader client] attempt {attempt + 1}/{retries} failed: {last}")
            time.sleep(min(30, 2 * 2**attempt))
    return {
        "status": "drop",
        "reward": None,
        "drop_reason": f"service unreachable: {last}",
        "grader_version": "unknown",
    }
